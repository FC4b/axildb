//! What the daemon knows about itself: its phase, the record it keeps in
//! `daemon.json`, its request statistics, and the fingerprint of the files
//! whose change makes its open handle stale.

use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use axil_core::{GraphState, PrewarmReport};
use serde::{Deserialize, Serialize};

/// Where the daemon is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    /// Bound and answering, the database not open yet.
    Starting = 0,
    /// Serving.
    Ready = 1,
    /// Reopening a stale handle; requests are not served meanwhile.
    Reloading = 2,
    /// Shutting down; nothing new is served.
    Stopping = 3,
}

impl Phase {
    /// The name `hello` and `daemon status` use.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Starting => "starting",
            Phase::Ready => "ready",
            Phase::Reloading => "reloading",
            Phase::Stopping => "stopping",
        }
    }
}

/// A [`Phase`] shared between threads.
pub struct PhaseCell(AtomicU8);

impl PhaseCell {
    pub fn new(phase: Phase) -> Self {
        Self(AtomicU8::new(phase as u8))
    }

    pub fn get(&self) -> Phase {
        match self.0.load(Ordering::Acquire) {
            0 => Phase::Starting,
            1 => Phase::Ready,
            2 => Phase::Reloading,
            _ => Phase::Stopping,
        }
    }

    pub fn set(&self, phase: Phase) {
        self.0.store(phase as u8, Ordering::Release);
    }

    /// Move from `from` to `to`, unless the phase has moved on meanwhile
    /// (a shutdown that began is never undone).
    pub fn advance(&self, from: Phase, to: Phase) -> bool {
        self.0
            .compare_exchange(from as u8, to as u8, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// `daemon.json`: what the daemon last said about itself. Written at each
/// change of phase, and last when it stops, so `daemon status` can tell a
/// daemon that exited from one that died.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonInfo {
    /// The daemon's protocol version.
    pub proto: u32,
    /// Its process id.
    pub pid: u32,
    /// `starting`, `ready`, `reloading`, `stopping` or `stopped`.
    pub state: String,
    /// Its `axil --version`.
    pub version: String,
    /// Its build ([`super::proto::ExeId::build_id`]).
    pub build_id: String,
    /// The executable it runs.
    pub exe: String,
    /// Its compiled features.
    pub features: Vec<String>,
    /// The database it serves.
    pub db: String,
    /// The socket it listens on.
    pub socket: String,
    /// When it started (RFC 3339).
    pub started_at: String,
    /// When it first became ready.
    pub ready_at: Option<String>,
    /// When it stopped.
    pub stopped_at: Option<String>,
    /// Why it stopped.
    pub exit_reason: Option<String>,
    /// How long it stays up with nothing to do, in seconds.
    pub idle_timeout_secs: u64,
}

/// The latencies kept per command: the last this many.
const RING: usize = 256;

/// One command's request statistics.
#[derive(Debug, Default)]
struct OpStats {
    served: u64,
    not_served: u64,
    /// The last [`RING`] served latencies, in microseconds.
    latencies: Vec<u32>,
    /// Where the next latency goes once the ring is full.
    next: usize,
}

impl OpStats {
    fn record(&mut self, latency: Duration) {
        let micros = u32::try_from(latency.as_micros()).unwrap_or(u32::MAX);
        if self.latencies.len() < RING {
            self.latencies.push(micros);
        } else {
            self.latencies[self.next] = micros;
        }
        self.next = (self.next + 1) % RING;
    }
}

/// The nearest-rank percentile `p` (0 < p ≤ 1) of `sorted`, in milliseconds.
fn percentile_ms(sorted: &[u32], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    Some(f64::from(sorted[rank - 1]) / 1000.0)
}

/// One command's figures, as `daemon status` reports them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpReport {
    /// Requests served.
    pub served: u64,
    /// Requests handed back to the client.
    pub not_served: u64,
    /// Over the last 256 served, in milliseconds, measured from the request
    /// being read to its output being ready.
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
}

/// The commands latencies are kept for; anything else counts as `other`.
const OPS: &[&str] = &[
    "recall",
    "boot",
    "recall-for-file",
    "code-search",
    "fts",
    "since",
    "store",
    "checkpoint",
];

/// The command a command line runs, as [`Stats`] files it: the first word
/// that is not an option or the value of one of the global options.
pub fn op_label(argv: &[String]) -> &'static str {
    let mut words = argv.iter().skip(1);
    while let Some(word) = words.next() {
        if word == "--db" || word == "--format" {
            words.next();
            continue;
        }
        if word.starts_with('-') {
            continue;
        }
        return OPS.iter().find(|op| *op == word).copied().unwrap_or("other");
    }
    "other"
}

/// The daemon's request statistics.
#[derive(Default)]
pub struct Stats {
    ops: Mutex<BTreeMap<&'static str, OpStats>>,
    /// Requests refused before running: incompatible, overloaded, expired,
    /// starting or stopping.
    pub refused: AtomicU64,
    /// Requests that panicked.
    pub panics: AtomicU64,
    /// Times the database was reopened.
    pub reloads: AtomicU64,
    last_reload: Mutex<Option<String>>,
    prewarm: Mutex<Option<PrewarmReport>>,
}

impl Stats {
    /// A served request for `op` that took `latency`.
    pub fn served(&self, op: &'static str, latency: Duration) {
        self.with(op, |s| {
            s.served += 1;
            s.record(latency);
        });
    }

    /// A request for `op` handed back to the client.
    pub fn not_served(&self, op: &'static str) {
        self.with(op, |s| s.not_served += 1);
    }

    fn with(&self, op: &'static str, f: impl FnOnce(&mut OpStats)) {
        let mut ops = self.ops.lock().unwrap_or_else(PoisonError::into_inner);
        f(ops.entry(op).or_default());
    }

    /// Each command's figures.
    pub fn ops(&self) -> BTreeMap<String, OpReport> {
        let ops = self.ops.lock().unwrap_or_else(PoisonError::into_inner);
        ops.iter()
            .map(|(op, s)| {
                let mut sorted = s.latencies.clone();
                sorted.sort_unstable();
                let report = OpReport {
                    served: s.served,
                    not_served: s.not_served,
                    p50_ms: percentile_ms(&sorted, 0.50),
                    p95_ms: percentile_ms(&sorted, 0.95),
                };
                (op.to_string(), report)
            })
            .collect()
    }

    /// A reload, and why.
    pub fn reloaded(&self, why: &str) {
        self.reloads.fetch_add(1, Ordering::AcqRel);
        *self.last_reload.lock().unwrap_or_else(PoisonError::into_inner) = Some(why.to_string());
    }

    /// Why the last reload ran.
    pub fn last_reload(&self) -> Option<String> {
        self.last_reload
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// What the last warm-up loaded.
    pub fn set_prewarm(&self, report: PrewarmReport) {
        *self.prewarm.lock().unwrap_or_else(PoisonError::into_inner) = Some(report);
    }

    pub fn prewarm(&self) -> Option<PrewarmReport> {
        self.prewarm
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// The files whose change makes the open handle stale: the database's
/// config file (where it is, its size and modification time), the database
/// and each companion store (device and inode), and the plugins directory
/// (its modification time). Each can change without a write through the
/// handle: a config edit, a restored backup renamed over a store, a store
/// dropped by `compact --drop-engine` or created by a one-shot command, an
/// installed plugin.
///
/// The full-text index is covered by its directory's inode, not
/// `meta.json`'s: tantivy writes a new `meta.json` and renames it over the
/// old one at every commit, so its inode changes with each write the handle
/// makes. A rebuilt or restored index replaces the directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFp {
    config: Option<(PathBuf, u64, i128)>,
    stores: Vec<(&'static str, Option<(u64, u64)>)>,
    plugins: Option<i128>,
}

/// The companion stores, by suffix.
const STORES: &[&str] = &["", ".vec", ".graph", ".ts", ".fts"];

fn mtime_ns(meta: &std::fs::Metadata) -> i128 {
    i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec())
}

impl FileFp {
    /// The fingerprint of the database at `key` now.
    pub fn read(key: &Path) -> FileFp {
        let dir = key.parent().unwrap_or(Path::new("."));
        let config = axil_core::find_config_file(dir).map(|path| {
            let (len, mtime) = std::fs::metadata(&path)
                .map(|m| (m.len(), mtime_ns(&m)))
                .unwrap_or((0, 0));
            (path, len, mtime)
        });
        let stores = STORES
            .iter()
            .map(|suffix| {
                let path = axil_core::companion_path(key, suffix);
                let id = std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()));
                (*suffix, id)
            })
            .collect();
        let plugins = std::fs::metadata(dir.join("plugins"))
            .ok()
            .map(|m| mtime_ns(&m));
        FileFp {
            config,
            stores,
            plugins,
        }
    }

    /// What changed from `self` to `now`, in words; `None` when nothing.
    pub fn changed(&self, now: &FileFp) -> Option<String> {
        if self == now {
            return None;
        }
        if self.config != now.config {
            return Some(match (&self.config, &now.config) {
                (_, Some((path, ..))) => format!("{} changed", path.display()),
                (Some((path, ..)), None) => format!("{} was removed", path.display()),
                (None, None) => unreachable!(),
            });
        }
        for ((suffix, was), (_, is)) in self.stores.iter().zip(&now.stores) {
            if was != is {
                let name = if suffix.is_empty() { "the database" } else { suffix };
                return Some(match (was, is) {
                    (None, Some(_)) => format!("{name} appeared"),
                    (Some(_), None) => format!("{name} was removed"),
                    _ => format!("{name} was replaced"),
                });
            }
        }
        Some("the plugins directory changed".to_string())
    }
}

/// The graph's state as `daemon status` reports it.
pub fn graph_name(state: &Option<GraphState>) -> &'static str {
    match state {
        None => "none",
        Some(GraphState::Unloaded) => "unloaded",
        Some(GraphState::Disk) => "disk",
        Some(GraphState::Memory) => "memory",
        Some(GraphState::Failed(_)) => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_label_is_the_subcommand_after_the_global_options() {
        let argv = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(op_label(&argv(&["axil", "recall", "q"])), "recall");
        assert_eq!(
            op_label(&argv(&["axil", "--db", "x", "--format", "table", "fts", "q"])),
            "fts"
        );
        assert_eq!(op_label(&argv(&["axil", "--quiet", "boot"])), "boot");
        assert_eq!(op_label(&argv(&["axil", "get", "id"])), "other");
        assert_eq!(op_label(&argv(&["axil", "--help"])), "other");
    }

    #[test]
    fn the_percentiles_cover_the_last_256() {
        let stats = Stats::default();
        for ms in 1..=300u64 {
            stats.served("recall", Duration::from_millis(ms));
        }
        stats.not_served("recall");
        let report = &stats.ops()["recall"];
        assert_eq!(report.served, 300);
        assert_eq!(report.not_served, 1);
        // The ring holds 45..=300.
        assert_eq!(report.p50_ms, Some(172.0));
        assert_eq!(report.p95_ms, Some(288.0));
    }

    #[test]
    fn the_fingerprint_sees_the_changes_a_handle_does_not_make() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(".axil");
        std::fs::create_dir(&dir).unwrap();
        let key = dir.join("memory.axil");
        std::fs::write(&key, b"db").unwrap();
        std::fs::create_dir(dir.join("memory.axil.fts")).unwrap();
        let base = FileFp::read(&key);
        assert_eq!(base.changed(&FileFp::read(&key)), None);

        // Writing through the open file keeps its inode, as a handle's own
        // writes do, and so does a commit inside the index directory.
        std::fs::write(&key, b"db, written").unwrap();
        std::fs::write(dir.join("memory.axil.fts/meta.json"), b"{}").unwrap();
        assert_eq!(base.changed(&FileFp::read(&key)), None);

        // A backup renamed over the database.
        let backup = dir.join("backup");
        std::fs::write(&backup, b"old db").unwrap();
        std::fs::rename(&backup, &key).unwrap();
        let restored = FileFp::read(&key);
        assert_eq!(
            base.changed(&restored).as_deref(),
            Some("the database was replaced")
        );

        // A store appearing; the config file.
        std::fs::write(dir.join("memory.axil.ts"), b"").unwrap();
        let with_ts = FileFp::read(&key);
        assert_eq!(restored.changed(&with_ts).as_deref(), Some(".ts appeared"));
        std::fs::write(tmp.path().join("axil.toml"), b"[engines]\n").unwrap();
        let configured = FileFp::read(&key);
        assert!(with_ts
            .changed(&configured)
            .is_some_and(|why| why.ends_with("axil.toml changed")));
    }
}
