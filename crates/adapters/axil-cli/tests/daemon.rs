//! The warm daemon as a user runs it: `axil daemon start`, `send`, `stop`
//! and `status`, against a temp database, with the built binary.
//!
//! Databases get a vector store only when the bge-small model is installed,
//! so no test downloads one; the vector checks are skipped loudly without
//! it. Every daemon is started with a short idle timeout and stopped when
//! its test ends, pass or fail.

#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_axil");

/// What the daemon reads from its environment, removed so each test sees
/// the defaults.
const ENV: &[&str] = &[
    "AXIL_DB",
    "AXIL_MCP_TOOLS",
    "AXIL_VERBOSE",
    "AXIL_EMBED_SESSIONS",
    "AXIL_SLOW_QUERY_LOG",
    "AXIL_DAEMON",
];

struct Fixture {
    _tmp: tempfile::TempDir,
    proj: PathBuf,
    db: PathBuf,
    vectors: bool,
}

fn model_available() -> bool {
    #[cfg(feature = "embed")]
    {
        use axil_vector::models::EmbeddingModel;
        if axil_vector::download::is_model_available(&EmbeddingModel::BgeSmall) {
            return true;
        }
    }
    eprintln!("no vector store: the bge-small model is not installed");
    false
}

/// A project with a database holding a few records.
fn fixture() -> Fixture {
    // Short enough for the socket to sit in the run directory.
    let tmp = tempfile::Builder::new().prefix("d").tempdir().unwrap();
    let proj = tmp.path().canonicalize().unwrap().join("p");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::create_dir_all(proj.join(".axil")).unwrap();
    std::fs::write(
        proj.join("src/auth.rs"),
        "/// Validate a login token.\npub fn validate_token(t: &str) -> bool { !t.is_empty() }\n",
    )
    .unwrap();
    std::fs::write(proj.join("Cargo.toml"), "[package]\nname = \"p\"\n").unwrap();
    let db = proj.join(".axil/memory.axil");
    let fx = |args: &[&str]| {
        let out = axil_in(BIN, &proj, args, None);
        assert!(out.status.success(), "{args:?}: {}", text(&out.stderr));
    };
    fx(&["init", db.to_str().unwrap()]);
    let vectors = model_available();
    if !vectors {
        let _ = std::fs::remove_file(proj.join(".axil/memory.axil.vec"));
    }
    for (table, data) in [
        (
            "decisions",
            r#"{"summary":"Cap login tokens at 512 bytes"}"#,
        ),
        (
            "errors",
            r#"{"error":"auth timeout during uploads","fix":"heartbeat"}"#,
        ),
        (
            "context",
            r#"{"type":"architecture","summary":"auth.rs validates tokens"}"#,
        ),
    ] {
        fx(&["store", table, data]);
    }
    Fixture {
        _tmp: tmp,
        proj,
        db,
        vectors,
    }
}

/// Whether the binary was built with what a test needs (`has`, from
/// `cfg!`), saying when it was not: CI also tests a core-only build.
fn built_with(has: bool, what: &str) -> bool {
    if !has {
        eprintln!("skipped: this build lacks {what}");
    }
    has
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Run `exe` with `args` in `dir`, stdin from `stdin` or nothing.
fn axil_in(exe: &str, dir: &Path, args: &[&str], stdin: Option<&[u8]>) -> Output {
    axil_env(exe, dir, args, stdin, &[])
}

/// [`axil_in`] with the variables `env` set.
fn axil_env(
    exe: &str,
    dir: &Path,
    args: &[&str],
    stdin: Option<&[u8]>,
    env: &[(&str, &str)],
) -> Output {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in ENV {
        cmd.env_remove(name);
    }
    cmd.envs(env.iter().copied());
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = cmd.spawn().unwrap();
    if let Some(bytes) = stdin {
        child.stdin.take().unwrap().write_all(bytes).unwrap();
    }
    child.wait_with_output().unwrap()
}

impl Fixture {
    fn axil(&self, args: &[&str]) -> Output {
        axil_in(BIN, &self.proj, args, None)
    }

    fn start_with(&self, exe: &str, idle: &str) -> Value {
        let out = axil_in(
            exe,
            &self.proj,
            &["daemon", "start", "--idle-timeout", idle],
            None,
        );
        assert!(
            out.status.success(),
            "start: {}{}",
            text(&out.stdout),
            text(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }

    fn start(&self) -> Value {
        self.start_with(BIN, "60s")
    }

    fn status(&self) -> Value {
        let out = self.axil(&["daemon", "status", "--json"]);
        assert!(out.status.success(), "{}", text(&out.stderr));
        serde_json::from_slice(&out.stdout).unwrap()
    }

    fn send(&self, args: &[&str], stdin: Option<&[u8]>) -> Output {
        let mut argv = vec!["daemon", "send", "--"];
        argv.extend_from_slice(args);
        axil_in(BIN, &self.proj, &argv, stdin)
    }

    /// [`Fixture::send`], again while the daemon answers that it is
    /// reloading: a tick may have found the change first.
    fn send_settled(&self, args: &[&str]) -> Output {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let out = self.send(args, None);
            let reloading = text(&out.stderr).contains("\"reason\":\"unavailable\"");
            if !reloading || Instant::now() >= deadline {
                return out;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn run_dir(&self) -> PathBuf {
        self.proj.join(".axil/run/memory.axil")
    }

    /// Whether a daemon holds the lock.
    fn locked(&self) -> bool {
        let Ok(file) = std::fs::File::open(self.run_dir().join("daemon.lock")) else {
            return false;
        };
        matches!(file.try_lock(), Err(std::fs::TryLockError::WouldBlock))
    }

    fn info(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.run_dir().join("daemon.json")).unwrap()).unwrap()
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.run_dir().join("daemon.log")).unwrap_or_default()
    }

    /// What the running daemon says it is doing.
    fn live(&self) -> Value {
        self.status()["live"].clone()
    }

    /// Until the daemon has the database open again after the leases on
    /// it ended, and serves a command.
    fn warm_again(&self) {
        assert!(
            wait_for(Duration::from_secs(30), || self.live()["state"] == "ready"),
            "{}",
            self.live()
        );
        let boot = self.send(&["boot"], None);
        assert!(boot.status.success(), "{}", text(&boot.stderr));
    }

    /// Run `args` here as a one-shot command, which leases the database:
    /// it must succeed without finding the database busy.
    fn leasing(&self, args: &[&str]) -> Output {
        let out = self.axil(args);
        let printed = format!("{}{}", text(&out.stdout), text(&out.stderr));
        assert!(out.status.success(), "{args:?}: {printed}");
        assert!(!printed.contains("busy"), "{args:?}: {printed}");
        out
    }

    /// `axil mcp` on the database, its stdin and stdout piped.
    fn mcp(&self) -> Mcp {
        let mut cmd = Command::new(BIN);
        cmd.arg("mcp")
            .current_dir(&self.proj)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for name in ENV {
            cmd.env_remove(name);
        }
        let mut child = cmd.spawn().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Mcp { child, lines: rx }
    }

    /// How many commands the daemon has run, by command.
    fn served(&self, op: &str) -> u64 {
        self.live()["ops"][op]["served"].as_u64().unwrap_or(0)
    }

    /// The lease holders the daemon reports, by pid.
    fn holders(&self) -> Vec<u64> {
        self.live()["lease_holders"]
            .as_array()
            .map(|h| h.iter().filter_map(|h| h["pid"].as_u64()).collect())
            .unwrap_or_default()
    }

    /// The stores no handle has open any more: each opens read-only, which
    /// redb refuses for a file a writer left without closing.
    fn closed_cleanly(&self) -> Result<(), String> {
        for suffix in ["", ".vec", ".graph", ".ts"] {
            let path = PathBuf::from(format!("{}{suffix}", self.db.display()));
            if path.exists() {
                redb::ReadOnlyDatabase::open(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.locked() {
            let _ = self.axil(&["daemon", "stop", "--wait", "10"]);
        }
    }
}

/// An `axil mcp` server, spoken to over its stdin and stdout.
struct Mcp {
    child: std::process::Child,
    lines: std::sync::mpsc::Receiver<String>,
}

impl Mcp {
    /// Send one JSON-RPC message, and read the answer when it has an id.
    fn call(&mut self, message: Value) -> Option<Value> {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{message}").unwrap();
        stdin.flush().unwrap();
        message.get("id")?;
        let line = self
            .lines
            .recv_timeout(Duration::from_secs(60))
            .expect("mcp did not answer");
        Some(serde_json::from_str(&line).unwrap())
    }

    fn initialize(&mut self) {
        let init = self.call(serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "daemon-test", "version": "0"},
            },
        }));
        assert!(init.unwrap()["result"].is_object());
        self.call(serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    /// Close its stdin, and wait for it to exit.
    fn close(mut self) -> std::process::ExitStatus {
        drop(self.child.stdin.take());
        self.child.wait().unwrap()
    }
}

/// Copy the directory `from` to `to`, sockets left out.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let kind = entry.file_type().unwrap();
        let dest = to.join(entry.file_name());
        if kind.is_dir() {
            copy_dir(&entry.path(), &dest);
        } else if kind.is_file() {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// What differs between two runs of the same commands: ULIDs, renumbered
/// in order of appearance, ISO-8601 times, and the line the embedder prints
/// once per process, which a daemon prints once in its life.
fn normalize(bytes: &[u8]) -> String {
    const CROCKFORD: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    let text = text(bytes);
    let text: String = text
        .split_inclusive('\n')
        .filter(|line| !line.starts_with("[axil-vector]"))
        .collect();
    let b = text.as_bytes();
    let mut ids: Vec<&[u8]> = Vec::new();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        let ulid = b.get(i..i + 26).filter(|w| {
            w[0] == b'0'
                && w.iter().all(|c| CROCKFORD.contains(c))
                && b.get(i + 26).is_none_or(|c| !c.is_ascii_alphanumeric())
        });
        if let (true, Some(id)) = (boundary, ulid) {
            let n = ids.iter().position(|seen| *seen == id).unwrap_or_else(|| {
                ids.push(id);
                ids.len() - 1
            });
            out.push_str(&format!("<ID{n}>"));
            i += 26;
            continue;
        }
        let digits = |at: usize, n: usize| {
            b.get(at..at + n)
                .is_some_and(|w| w.iter().all(u8::is_ascii_digit))
        };
        if digits(i, 4)
            && b.get(i + 4) == Some(&b'-')
            && digits(i + 5, 2)
            && b.get(i + 10) == Some(&b'T')
            && digits(i + 11, 2)
            && b.get(i + 13) == Some(&b':')
        {
            let mut j = i + 19;
            while j < b.len() && (b[j].is_ascii_digit() || b"+-:.Z".contains(&b[j])) {
                j += 1;
            }
            out.push_str("<TS>");
            i = j;
            continue;
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn wait_for(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    done()
}

/// The `daemon run` processes serving `db`.
fn daemons_for(db: &Path) -> Vec<String> {
    let out = Command::new("ps")
        .args(["-axo", "pid=,command="])
        .output()
        .unwrap();
    text(&out.stdout)
        .lines()
        .filter(|l| l.contains(&*db.to_string_lossy()) && l.contains("daemon run"))
        .map(str::to_string)
        .collect()
}

fn signal(pid: u64, sig: &str) {
    let status = Command::new("kill")
        .args([sig, &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

/// Send one raw request frame and read the answer frame.
fn raw_call(socket: &Path, request: &str) -> Value {
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream
        .write_all(&(request.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
    stream.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[test]
fn eight_starts_at_once_leave_one_daemon() {
    let fx = fixture();
    let starts: Vec<_> = (0..8)
        .map(|_| {
            let mut cmd = Command::new(BIN);
            cmd.args(["daemon", "start", "--idle-timeout", "60s"])
                .current_dir(&fx.proj)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for name in ENV {
                cmd.env_remove(name);
            }
            cmd.spawn().unwrap()
        })
        .collect();
    let reports: Vec<Value> = starts
        .into_iter()
        .map(|child| {
            let out = child.wait_with_output().unwrap();
            assert!(out.status.success(), "{}", text(&out.stderr));
            serde_json::from_slice(&out.stdout).unwrap()
        })
        .collect();
    let pid = &reports[0]["pid"];
    assert!(
        reports
            .iter()
            .all(|r| &r["pid"] == pid && r["state"] == "ready"),
        "{reports:?}"
    );
    let started = reports.iter().filter(|r| r["started"] == true).count();
    assert_eq!(started, 1, "{reports:?}");
    // The losers leave as soon as the winner answers.
    assert!(
        wait_for(Duration::from_secs(10), || daemons_for(&fx.db).len() == 1),
        "{:?}",
        daemons_for(&fx.db)
    );
    assert_eq!(fx.status()["live"]["pid"], *pid);
}

#[test]
fn a_killed_daemon_refuses_connections_and_the_next_start_takes_over() {
    let fx = fixture();
    let first = fx.start();
    let pid = first["pid"].as_u64().unwrap();
    let socket = PathBuf::from(first["socket"].as_str().unwrap());
    signal(pid, "-KILL");
    assert!(wait_for(Duration::from_secs(10), || !fx.locked()));

    // The socket file is still there, and nothing listens on it.
    let err = UnixStream::connect(&socket).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::ConnectionRefused);
    // Not run, as a command the daemon declines is not.
    let sent = fx.send(&["boot"], None);
    assert_eq!(sent.status.code(), Some(125), "{}", text(&sent.stderr));
    // The stores are left as a killed writer leaves them.
    assert!(fx.closed_cleanly().is_err());
    let status = fx.status();
    assert_eq!(status["running"], false);
    assert_eq!(status["last"]["state"], "ready");

    let second = fx.start();
    assert_ne!(second["pid"].as_u64().unwrap(), pid);
    let boot = fx.send(&["boot"], None);
    assert!(boot.status.success(), "{}", text(&boot.stderr));
    if fx.vectors {
        assert!(
            fx.log().contains("was not closed cleanly; repaired it"),
            "{}",
            fx.log()
        );
    }
}

#[test]
fn idle_sigterm_and_stop_each_close_every_store() {
    let fx = fixture();

    fx.start_with(BIN, "2s");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["exit_reason"], "idle");

    let pid = fx.start()["pid"].as_u64().unwrap();
    assert!(fx.send(&["boot"], None).status.success());
    signal(pid, "-TERM");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["exit_reason"], "signal 15");

    fx.start();
    assert!(fx.send(&["boot"], None).status.success());
    let out = fx.axil(&["daemon", "stop"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stopped["stopped"], true, "{stopped}");
    assert_eq!(stopped["clean"], true, "{stopped}");
    assert!(!fx.locked());
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["state"], "stopped");
    let status = fx.status();
    assert_eq!(status["running"], false);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(
        human.starts_with("daemon: not running; last one stopped at"),
        "{human}"
    );
}

#[test]
fn replacing_the_executable_retires_the_daemon() {
    let fx = fixture();
    let bin_dir = fx.proj.join("bin");
    std::fs::create_dir(&bin_dir).unwrap();
    let copy = bin_dir.join("axil");
    std::fs::copy(BIN, &copy).unwrap();
    let pid = fx.start_with(copy.to_str().unwrap(), "60s")["pid"]
        .as_u64()
        .unwrap();

    // A build installed over it, as `cargo install` does: a new file
    // renamed into place.
    let new = bin_dir.join("axil.new");
    std::fs::copy(BIN, &new).unwrap();
    std::fs::rename(&new, &copy).unwrap();
    assert!(
        wait_for(Duration::from_secs(10), || !fx.locked()),
        "pid {pid} still running"
    );
    assert_eq!(fx.info()["exit_reason"], "its executable changed");
    fx.closed_cleanly().unwrap();

    // A symlink moved to another build, as a package manager upgrades: the
    // binary the daemon runs is unchanged, the path it was started by now
    // leads to another.
    let (one, two) = (bin_dir.join("axil-one"), bin_dir.join("axil-two"));
    std::fs::copy(BIN, &one).unwrap();
    std::fs::copy(BIN, &two).unwrap();
    let link = bin_dir.join("axil-link");
    std::os::unix::fs::symlink(&one, &link).unwrap();
    let pid = fx.start_with(link.to_str().unwrap(), "60s")["pid"]
        .as_u64()
        .unwrap();
    let moved = bin_dir.join("axil-link.new");
    std::os::unix::fs::symlink(&two, &moved).unwrap();
    std::fs::rename(&moved, &link).unwrap();
    assert!(
        wait_for(Duration::from_secs(10), || !fx.locked()),
        "pid {pid} still running"
    );
    assert_eq!(fx.info()["exit_reason"], "its executable changed");
}

#[test]
fn a_dropped_engine_and_a_failed_graph_reload_the_daemon() {
    let engines = cfg!(all(
        feature = "timeseries",
        feature = "fts",
        feature = "graph"
    ));
    if !built_with(engines, "the time-series, full-text or graph store") {
        return;
    }
    let fx = fixture();
    let socket = PathBuf::from(fx.start()["socket"].as_str().unwrap());
    let reloads = || fx.status()["live"]["reloads"].as_u64().unwrap();
    assert!(fx.send(&["since", "1h"], None).status.success());
    assert_eq!(reloads(), 0);

    // A store removed by something that never asks the daemon first.
    let ts = fx.proj.join(".axil/memory.axil.ts");
    if ts.is_dir() {
        std::fs::remove_dir_all(&ts).unwrap();
    } else {
        std::fs::remove_file(&ts).unwrap();
    }
    std::thread::sleep(Duration::from_millis(300));
    let since = fx.send_settled(&["since", "1h"]);
    assert_eq!(since.status.code(), Some(125), "{}", text(&since.stderr));
    assert!(text(&since.stderr).contains("\"reason\":\"lacks\""));
    assert_eq!(reloads(), 1);
    assert!(fx.status()["live"]["last_reload"]
        .as_str()
        .unwrap()
        .contains(".ts was removed"));

    // `compact --drop-engine` removes one with the database leased, and
    // the daemon opens it again without it rather than reload.
    fx.leasing(&["compact", "--drop-engine", "fts"]);
    fx.warm_again();
    let fts = fx.send(&["fts", "tokens"], None);
    assert_eq!(fts.status.code(), Some(125), "{}", text(&fts.stderr));
    assert!(text(&fts.stderr).contains("\"reason\":\"lacks\""));
    assert_eq!(fx.live()["leases_granted"], 1);
    assert_eq!(reloads(), 1);

    let ack = raw_call(&socket, r#"{"op":"debug_fail_graph"}"#);
    assert_eq!(ack["ok"], true);
    assert!(fx.send_settled(&["boot"]).status.success());
    assert_eq!(reloads(), 2);
    assert_eq!(
        fx.status()["live"]["last_reload"],
        "the graph failed to load"
    );
    assert_eq!(fx.status()["live"]["state"], "ready");
}

/// What the hot commands print goes to the client, never into the
/// daemon's log: every line there is one of the daemon's own.
#[test]
fn the_hot_commands_print_nothing_into_the_log() {
    let fx = fixture();
    fx.start();
    let cp = r#"{"goal":"g","state":"s","next_steps":["n"]}"#;
    // Each with whether this build serves it.
    let recall = cfg!(feature = "embed");
    let cases: Vec<(Vec<&str>, Option<&[u8]>, bool)> = vec![
        (vec!["boot"], None, true),
        (vec!["boot", "--boot-format", "narrative"], None, true),
        (vec!["boot", "--files", "src/auth.rs"], None, true),
        (vec!["recall-for-file", "src/auth.rs"], None, true),
        (vec!["recall", "auth timeout"], None, recall),
        (
            vec!["recall", "auth timeout", "--recall-format", "context-block"],
            None,
            recall,
        ),
        (
            vec!["--format", "table", "recall", "auth timeout"],
            None,
            recall,
        ),
        (
            vec!["code-search", "validate token"],
            None,
            cfg!(feature = "indexer"),
        ),
        (vec!["fts", "tokens"], None, cfg!(feature = "fts")),
        (
            vec!["fts", "tokens", "--jsonl"],
            None,
            cfg!(feature = "fts"),
        ),
        (vec!["since", "1h"], None, cfg!(feature = "timeseries")),
        (
            vec!["store", "decisions", r#"{"summary":"sent"}"#],
            None,
            true,
        ),
        (
            vec!["store", "context", "-"],
            Some(br#"{"summary":"piped"}"#),
            true,
        ),
        (vec!["store", "decisions", "{nope"], None, true),
        (vec!["checkpoint", cp], None, cfg!(feature = "checkpoint")),
        (
            vec!["checkpoint", "-"],
            Some(cp.as_bytes()),
            cfg!(feature = "checkpoint"),
        ),
        (vec!["--help"], None, true),
        (vec!["store"], None, true),
        (vec!["--version"], None, true),
    ];
    let mut served = 0;
    for (args, stdin, _) in &cases {
        let out = fx.send(args, *stdin);
        if out.status.code() != Some(125) {
            served += 1;
            assert!(
                !out.stdout.is_empty() || !out.stderr.is_empty(),
                "{args:?} printed nothing"
            );
            // The daemon's open guard fails a debug build's command that
            // opens the database itself rather than use the open handle.
            assert!(
                !text(&out.stderr).contains("panicked"),
                "{args:?}: {}",
                text(&out.stderr)
            );
        }
    }
    let servable = cases.iter().filter(|(_, _, served)| *served).count();
    assert!(served >= servable - 2, "only {served} served");
    let live = fx.live();
    assert_eq!(
        (&live["state"], &live["panics"]),
        (&Value::from("ready"), &Value::from(0))
    );
    let log = fx.log();
    let foreign: Vec<&str> = log
        .lines()
        .filter(|line| {
            let mut parts = line.splitn(3, ' ');
            let (stamp, who) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            !(stamp.len() == 20 && stamp.ends_with('Z') && who.starts_with("axil"))
        })
        .collect();
    assert!(
        foreign.is_empty(),
        "not the daemon's own lines:\n{}",
        foreign.join("\n")
    );
    assert!(log.contains("ready"), "{log}");
    // ...and the check those commands ran under is armed.
    let socket = PathBuf::from(fx.live()["socket"].as_str().unwrap());
    let guard = raw_call(&socket, r#"{"op":"debug_open_guard"}"#);
    assert_eq!(guard["tripped"], true, "{guard}");

    // A checkpoint given its JSON does not read stdin, so an open pipe
    // nobody writes to does not hold it up.
    if !built_with(cfg!(feature = "checkpoint"), "checkpoint") {
        return;
    }
    let mut cmd = Command::new(BIN);
    cmd.args(["daemon", "send", "--", "checkpoint", cp])
        .current_dir(&fx.proj)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in ENV {
        cmd.env_remove(name);
    }
    let mut child = cmd.spawn().unwrap();
    let open_stdin = child.stdin.take();
    let finished = wait_for(Duration::from_secs(60), || {
        child.try_wait().unwrap().is_some()
    });
    drop(open_stdin);
    let out = child.wait_with_output().unwrap();
    assert!(finished, "send waited on stdin");
    assert!(out.status.success(), "{}", text(&out.stderr));
}

/// `AXIL_DAEMON=off` keeps a daemon from starting.
#[test]
fn the_kill_switch_keeps_the_daemon_from_starting() {
    let fx = fixture();
    let out = Command::new(BIN)
        .args(["daemon", "start"])
        .current_dir(&fx.proj)
        .env_remove("AXIL_DB")
        .env("AXIL_DAEMON", "off")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("AXIL_DAEMON=off"),
        "{}",
        text(&out.stderr)
    );
    assert!(!fx.locked());
}

/// A database whose open prints a warning is not served: `start` says
/// why and fails, and so does `status`, and the next `start` tries again.
#[test]
fn a_database_whose_open_warns_is_not_served() {
    let fx = fixture();
    std::fs::write(fx.proj.join("axil.toml"), "[healing\nevent_log = true\n").unwrap();
    // Four at once: whichever daemon wins, every start reports its reason.
    let starts: Vec<_> = (0..4)
        .map(|_| {
            let mut cmd = Command::new(BIN);
            cmd.args(["daemon", "start", "--idle-timeout", "60s"])
                .current_dir(&fx.proj)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for name in ENV {
                cmd.env_remove(name);
            }
            cmd.spawn().unwrap()
        })
        .collect();
    for child in starts {
        let out = child.wait_with_output().unwrap();
        assert_eq!(out.status.code(), Some(1), "{}", text(&out.stderr));
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["started"], false);
        assert!(
            report["blocked"]
                .as_str()
                .unwrap()
                .contains("invalid config"),
            "{report}"
        );
    }
    assert!(!fx.locked());
    let status = fx.status();
    assert_eq!(status["running"], false);
    assert!(status["blocked"]["reason"]
        .as_str()
        .unwrap()
        .contains("invalid config"));
    assert!(text(&fx.axil(&["daemon", "status"]).stdout).contains("not served: "));

    std::fs::remove_file(fx.proj.join("axil.toml")).unwrap();
    assert_eq!(fx.start()["state"], "ready");
    assert_eq!(fx.status()["blocked"], Value::Null);
}

/// Two databases in one directory each have a daemon of their own:
/// starting, stopping and asking about one leaves the other be.
#[test]
fn two_databases_in_one_directory_have_a_daemon_each() {
    let fx = fixture();
    let other = fx.proj.join(".axil/other.axil");
    let other = other.to_str().unwrap();
    let init = fx.axil(&["init", other]);
    assert!(init.status.success(), "{}", text(&init.stderr));
    let on_other = |args: &[&str]| {
        let mut argv = vec!["--db", other];
        argv.extend_from_slice(args);
        fx.axil(&argv)
    };

    let first = fx.start();
    let status: Value =
        serde_json::from_slice(&on_other(&["daemon", "status", "--json"]).stdout).unwrap();
    assert_eq!(status["running"], false, "{status}");

    let out = on_other(&["daemon", "start", "--idle-timeout", "60s"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let second: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(second["started"], true, "{second}");
    assert_ne!(second["pid"], first["pid"]);
    assert_ne!(second["socket"], first["socket"]);

    let out = on_other(&["daemon", "stop"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stopped["stopped"], true, "{stopped}");
    assert_eq!(stopped["pid"], second["pid"]);
    assert_eq!(fx.status()["live"]["pid"], first["pid"]);
}

/// A daemon slow to shut down (here holding the database 7 s, as a long
/// warm-up would): `status` says it is stopping, `stop` waits for it and
/// reports it closed the database, and a `start` meanwhile waits for it to
/// go and takes over.
#[test]
fn a_slow_shutdown_is_waited_for() {
    let fx = fixture();
    let slow_shutdown = || {
        let first = fx.start();
        let socket = PathBuf::from(first["socket"].as_str().unwrap());
        let hold = raw_call(&socket, r#"{"op":"debug_hold","ms":7000}"#);
        assert_eq!(hold["ok"], true);
        let shutdown = raw_call(&socket, r#"{"op":"shutdown","reason":"test"}"#);
        assert_eq!(shutdown["ok"], true);
        assert!(wait_for(Duration::from_secs(10), || fx.info()["state"] == "stopping"));
        first["pid"].as_u64().unwrap()
    };

    let pid = slow_shutdown();
    assert_eq!(fx.status()["stopping"], true);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(human.starts_with("daemon: stopping"), "{human}");
    let out = fx.axil(&["daemon", "stop", "--wait", "20"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        (
            &stopped["stopped"],
            &stopped["clean"],
            stopped["pid"].as_u64()
        ),
        (&Value::Bool(true), &Value::Bool(true), Some(pid)),
        "{stopped}"
    );
    fx.closed_cleanly().unwrap();

    let pid = slow_shutdown();
    let next = fx.start();
    assert_ne!(next["pid"].as_u64().unwrap(), pid);
    assert_eq!(next["started"], true, "{next}");
}

/// A backup copied over the database in place, as `cp` does, keeping its
/// inode: the daemon stops without closing the database, which would write
/// its stale view over the copy, so the copy is what the next command
/// reads.
#[test]
fn a_backup_copied_over_the_database_is_kept() {
    let fx = fixture();
    let backup = std::fs::read(&fx.db).unwrap();
    fx.start();
    let stored = fx.send(
        &["store", "decisions", r#"{"summary":"after the backup"}"#],
        None,
    );
    assert!(stored.status.success(), "{}", text(&stored.stderr));
    let id = serde_json::from_slice::<Value>(&stored.stdout).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&fx.db)
            .unwrap();
        file.write_all(&backup).unwrap();
    }
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    let info = fx.info();
    assert!(
        info["exit_reason"]
            .as_str()
            .unwrap()
            .starts_with("its files were overwritten"),
        "{info}"
    );
    assert_eq!(info["closed_cleanly"], false);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(human.contains("for the next open to repair"), "{human}");
    assert!(
        std::fs::read(&fx.db).unwrap() == backup,
        "the copy was written over"
    );
    let get = fx.axil(&["get", &id]);
    assert!(
        text(&get.stderr).contains("not found"),
        "{}",
        text(&get.stderr)
    );
}

/// A daemon that holds the lock but answers nothing (stopped with SIGSTOP
/// here) is sent SIGTERM by `stop`, which first makes sure the process is
/// this database's daemon.
#[test]
fn stop_signals_a_daemon_that_does_not_answer() {
    let fx = fixture();
    let pid = fx.start()["pid"].as_u64().unwrap();
    signal(pid, "-STOP");
    let out = fx.axil(&["daemon", "stop", "--wait", "2"]);
    signal(pid, "-CONT");
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    // Still stopped when the wait ran out; the signal was waiting for it.
    assert_eq!(report["running"], true, "{report}");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    assert_eq!(fx.info()["exit_reason"], "signal 15");
    fx.closed_cleanly().unwrap();
}

/// Commands the daemon does not run lease the database from it: each one
/// opens it as with no daemon, its vector store and embedder included, and
/// the daemon serves again once it is done.
#[test]
fn the_commands_it_does_not_serve_lease_the_database() {
    let fx = fixture();
    fx.start();

    let listed = fx.leasing(&["list", "decisions"]);
    fx.warm_again();
    let rows: Value = serde_json::from_slice(&listed.stdout).unwrap();
    let id = rows[0]["id"].as_str().unwrap().to_string();
    let got = fx.leasing(&["get", &id]);
    assert!(
        text(&got.stdout).contains("512 bytes"),
        "{}",
        text(&got.stdout)
    );
    fx.warm_again();

    // Exits 1 for the warning that no code graph is indexed.
    let doctor = fx.axil(&["doctor"]);
    let printed = format!("{}{}", text(&doctor.stdout), text(&doctor.stderr));
    assert!(!printed.contains("busy"), "{printed}");
    let report: Value = serde_json::from_slice(&doctor.stdout).unwrap();
    let failed: Vec<&Value> = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] == "error")
        .collect();
    assert!(failed.is_empty(), "{report}");
    if fx.vectors {
        let vector = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "vector_index")
            .cloned()
            .unwrap_or_default();
        assert_eq!(vector["status"], "ok", "{report}");
    }
    fx.warm_again();

    if built_with(cfg!(feature = "indexer"), "index") {
        let index = fx.leasing(&["index"]);
        if fx.vectors {
            // The embedder announces itself once a process loads it.
            assert!(
                text(&index.stderr).contains("[axil-vector]"),
                "indexed without the embedder: {}",
                text(&index.stderr)
            );
        }
        fx.warm_again();
    }
    if fx.vectors {
        let reembed = fx.leasing(&["reembed", "--field", "summary"]);
        let report: Value = serde_json::from_slice(&reembed.stdout).unwrap();
        assert!(report["reembedded"].as_u64().unwrap() > 0, "{report}");
        fx.warm_again();
    }
    if built_with(cfg!(feature = "deps"), "deps") {
        fx.leasing(&["deps", "refresh"]);
        fx.warm_again();
    }

    if !built_with(
        cfg!(all(feature = "mcp", feature = "embed")),
        "mcp or recall",
    ) {
        assert_eq!(fx.live()["reloads"], 0, "{}", fx.live());
        return;
    }
    let mut mcp = fx.mcp();
    mcp.initialize();
    let pid = u64::from(mcp.child.id());
    assert!(
        wait_for(Duration::from_secs(10), || fx.holders() == [pid]),
        "{}",
        fx.live()
    );
    let mut store = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "store", "arguments": {
            "table": "decisions", "data": {"summary": "stored over mcp"},
        }},
    });
    if fx.vectors {
        store["params"]["arguments"]["embed_fields"] = serde_json::json!(["summary"]);
    }
    let stored = mcp.call(store).unwrap();
    let said = stored["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(stored["result"]["isError"] != true, "{stored}");
    assert!(!said.contains("failed"), "{stored}");
    // Served while the server runs: its lease holds the database.
    let boot = fx.send(&["boot"], None);
    assert_eq!(boot.status.code(), Some(125));
    assert!(
        text(&boot.stderr).contains("leased"),
        "{}",
        text(&boot.stderr)
    );
    assert!(mcp.close().success());
    fx.warm_again();
    let found = fx.send(&["recall", "stored over mcp"], None);
    assert!(
        text(&found.stdout).contains("stored over mcp"),
        "{}",
        text(&found.stdout)
    );
    assert_eq!(fx.live()["reloads"], 0, "{}", fx.live());
}

/// `AXIL_DAEMON=off` turns the lease off with the rest: the command opens
/// as with no daemon, and finds the database the daemon has open busy.
#[test]
fn with_the_daemon_off_a_command_asks_for_no_lease() {
    let fx = fixture();
    fx.start();
    let mut cmd = Command::new(BIN);
    cmd.args([
        "store",
        "decisions",
        r#"{"summary":"stored with the daemon off"}"#,
    ])
    .current_dir(&fx.proj)
    .stdin(Stdio::null());
    for name in ENV {
        cmd.env_remove(name);
    }
    let out = cmd.env("AXIL_DAEMON", "off").output().unwrap();
    assert!(!out.status.success(), "{}", text(&out.stdout));
    assert!(text(&out.stderr).contains("busy"), "{}", text(&out.stderr));
    let live = fx.live();
    assert_eq!(live["leases_granted"], 0, "{live}");
    assert_eq!(live["state"], "ready", "{live}");
}

/// The hot commands typed as usual go to the daemon, which prints what each
/// prints with no daemon: the same bytes as with `AXIL_DAEMON=off`, ids and
/// times aside. None of them leases the database.
#[test]
fn the_hot_commands_go_to_the_daemon_and_print_as_without_one() {
    let fx = fixture();
    let cp = r#"{"goal":"g","state":"s","next_steps":["n"]}"#;
    let mut cases: Vec<(Vec<&str>, Option<&[u8]>)> = vec![
        (vec!["boot"], None),
        (vec!["boot", "--boot-format", "narrative"], None),
        (vec!["boot", "--files", "src/auth.rs"], None),
        (vec!["recall-for-file", "src/auth.rs"], None),
        (vec!["store", "decisions", r#"{"summary":"sent"}"#], None),
        (
            vec!["store", "context", "-"],
            Some(br#"{"summary":"piped"}"#),
        ),
        (vec!["store", "decisions", "{nope"], None),
    ];
    if built_with(cfg!(feature = "embed"), "recall") {
        cases.push((vec!["recall", "auth timeout"], None));
        cases.push((
            vec!["recall", "login tokens", "--recall-format", "context-block"],
            None,
        ));
    }
    if built_with(cfg!(feature = "indexer"), "code-search") {
        cases.push((vec!["code-search", "validate token"], None));
    }
    if built_with(cfg!(feature = "fts"), "fts") {
        cases.push((vec!["fts", "tokens"], None));
    }
    if built_with(cfg!(feature = "timeseries"), "since") {
        cases.push((vec!["since", "1h"], None));
    }
    if built_with(cfg!(feature = "checkpoint"), "checkpoint") {
        cases.push((vec!["checkpoint", cp], None));
        cases.push((vec!["checkpoint", "-"], Some(cp.as_bytes())));
    }
    let saved = fx._tmp.path().join("saved");
    copy_dir(&fx.proj.join(".axil"), &saved);
    let run_all = |env: &[(&str, &str)]| -> Vec<(String, String, Option<i32>)> {
        cases
            .iter()
            .map(|(args, stdin)| {
                let out = axil_env(BIN, &fx.proj, args, *stdin, env);
                (
                    normalize(&out.stdout),
                    normalize(&out.stderr),
                    out.status.code(),
                )
            })
            .collect()
    };

    fx.start();
    let sent = run_all(&[]);
    let live = fx.live();
    let served: u64 = live["ops"]
        .as_object()
        .unwrap()
        .values()
        .map(|op| op["served"].as_u64().unwrap())
        .sum();
    assert_eq!(served, cases.len() as u64, "{live}");
    assert_eq!(live["leases_granted"], 0, "{live}");
    let stop = fx.axil(&["daemon", "stop"]);
    assert!(stop.status.success(), "{}", text(&stop.stderr));

    std::fs::remove_dir_all(fx.proj.join(".axil")).unwrap();
    copy_dir(&saved, &fx.proj.join(".axil"));
    let alone = run_all(&[("AXIL_DAEMON", "off")]);
    for ((args, _), (sent, alone)) in cases.iter().zip(sent.iter().zip(&alone)) {
        assert_eq!(sent, alone, "{args:?}");
    }
}

/// A connection that breaks once the command went out: a write is not run
/// again, since the daemon may have run it, and says so; a read runs here.
#[test]
fn a_lost_connection_fails_a_write_and_runs_a_read_here() {
    let fx = fixture();
    // A daemon that reads each request and hangs up.
    let dir = fx.run_dir();
    std::fs::create_dir_all(&dir).unwrap();
    for d in [dir.parent().unwrap(), &dir] {
        std::fs::set_permissions(d, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    }
    let listener = std::os::unix::net::UnixListener::bind(dir.join("axil.sock")).unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut len = [0u8; 4];
            if stream.read_exact(&mut len).is_ok() {
                let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
                let _ = stream.read_exact(&mut body);
            }
        }
    });

    let store = fx.axil(&["store", "decisions", r#"{"summary":"lost on the way"}"#]);
    assert_eq!(store.status.code(), Some(1));
    assert_eq!(
        text(&store.stderr),
        "{\"error\":\"daemon connection lost; write status unknown\"}\n"
    );
    let boot = fx.axil(&["boot"]);
    assert!(boot.status.success(), "{}", text(&boot.stderr));
    assert!(
        text(&boot.stdout).contains("512 bytes"),
        "{}",
        text(&boot.stdout)
    );
    let off = axil_env(
        BIN,
        &fx.proj,
        &["list", "decisions"],
        None,
        &[("AXIL_DAEMON", "off")],
    );
    assert!(!text(&off.stdout).contains("lost on the way"));
}

/// A client of another build, or with another environment, runs the
/// command itself and leases the database; one for another database never
/// reaches this one's daemon, which refuses a command for one.
#[test]
fn another_build_environment_or_database_is_not_sent() {
    let fx = fixture();
    let copy = fx.proj.join("axil-copy");
    std::fs::copy(BIN, &copy).unwrap();
    fx.start_with(copy.to_str().unwrap(), "60s");
    let boot = fx.leasing(&["boot"]);
    assert!(text(&boot.stdout).contains("512 bytes"));
    assert_eq!(fx.served("boot"), 0, "{}", fx.live());
    assert_eq!(fx.live()["leases_granted"], 1, "{}", fx.live());
    fx.warm_again();
    let stop = fx.axil(&["daemon", "stop"]);
    assert!(stop.status.success(), "{}", text(&stop.stderr));

    fx.start();
    let keyed = axil_env(BIN, &fx.proj, &["boot"], None, &[("AXIL_ENC_KEY", "x")]);
    assert!(keyed.status.success(), "{}", text(&keyed.stderr));
    assert_eq!(fx.served("boot"), 0, "{}", fx.live());
    assert_eq!(fx.live()["leases_granted"], 1, "{}", fx.live());
    fx.warm_again();
    assert_eq!(fx.served("boot"), 1, "{}", fx.live());

    let other = fx.proj.join("other.axil");
    let init = fx.axil(&["init", other.to_str().unwrap()]);
    assert!(init.status.success(), "{}", text(&init.stderr));
    let elsewhere = fx.axil(&["--db", other.to_str().unwrap(), "boot"]);
    assert!(elsewhere.status.success(), "{}", text(&elsewhere.stderr));
    assert_eq!(fx.served("boot"), 1, "{}", fx.live());
    let refused = fx.send(&["--db", other.to_str().unwrap(), "boot"], None);
    assert_eq!(refused.status.code(), Some(125));
    assert!(
        text(&refused.stderr).contains("\"reason\":\"wrong_db\""),
        "{}",
        text(&refused.stderr)
    );
}

/// Hook writes queued while the daemon runs: the drainer's commands that
/// the daemon runs go to it, the others lease the database, and every job
/// runs once, in the order queued, none of them finding the database busy.
#[test]
fn queued_hook_writes_run_once_in_order_beside_the_daemon() {
    let fx = fixture();
    fx.start();
    let queue = fx.proj.join(".axil/hook-queue");
    std::fs::create_dir_all(&queue).unwrap();
    let mut stores = 0;
    for n in 0..50 {
        let job = match n % 5 {
            // Not a command the daemon runs: it leases the database.
            4 => serde_json::json!({"op": "axil", "args": ["list", "decisions"]}),
            3 => {
                stores += 1;
                serde_json::json!({
                    "op": "axil",
                    "args": ["store", "context", "-"],
                    "stdin": serde_json::json!({"summary": format!("queued {n}"), "seq": n}).to_string(),
                })
            }
            _ => {
                stores += 1;
                let data = serde_json::json!({"summary": format!("queued {n}"), "seq": n});
                serde_json::json!({"op": "axil", "args": ["store", "decisions", data.to_string()]})
            }
        };
        let name = format!("20260101T000000.{n:09}Z-1-{n}.job");
        std::fs::write(queue.join(name), job.to_string()).unwrap();
    }
    let drain = fx.axil(&["hook", "drain"]);
    assert!(drain.status.success(), "{}", text(&drain.stderr));
    let log = std::fs::read_to_string(queue.join("drain.log")).unwrap_or_default();
    assert!(!log.contains("busy") && !log.contains("failed"), "{log}");
    let left: Vec<_> = std::fs::read_dir(&queue)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x == "job" || x == "running")
        })
        .collect();
    assert!(left.is_empty(), "{left:?}");
    assert_eq!(fx.served("store"), stores, "{}", fx.live());

    // Every one once, and in order: ids rise with the time they were made.
    let mut rows: Vec<(String, i64)> = Vec::new();
    for table in ["decisions", "context"] {
        let out = fx.axil(&["list", table]);
        let list: Value = serde_json::from_slice(&out.stdout).unwrap();
        for row in list.as_array().unwrap() {
            if let Some(seq) = row["data"]["seq"].as_i64() {
                rows.push((row["id"].as_str().unwrap().to_string(), seq));
            }
        }
    }
    rows.sort();
    let seqs: Vec<i64> = rows.iter().map(|(_, seq)| *seq).collect();
    let want: Vec<i64> = (0..50).filter(|n| n % 5 != 4).collect();
    assert_eq!(seqs, want);
}

/// `reembed` has the database leased before it touches the vector store,
/// even to put back the copy an interrupted run left: here one it cannot
/// move into place.
#[test]
fn reembed_leases_before_it_moves_the_vector_store() {
    let fx = fixture();
    if !fx.vectors {
        return;
    }
    fx.start();
    std::fs::create_dir_all(fx.proj.join(".axil/memory.axil.reembed-bak/x")).unwrap();
    let out = fx.axil(&["reembed", "--field", "summary"]);
    assert!(!out.status.success());
    assert!(
        text(&out.stderr).contains("failed to restore"),
        "{}",
        text(&out.stderr)
    );
    assert_eq!(fx.live()["leases_granted"], 1, "{}", fx.live());
    fx.warm_again();
}

/// A command for a store the daemon's handle lacks is not served; the
/// one-shot command leases the database and creates the store, and once
/// the daemon has opened the database again it serves the command.
#[test]
fn a_store_the_daemon_lacks_is_created_under_a_lease() {
    let engines = cfg!(all(feature = "fts", feature = "timeseries"));
    if !built_with(engines, "the full-text or time-series store") {
        return;
    }
    let fx = fixture();
    let mut engines = vec!["fts", "ts"];
    if fx.vectors {
        engines.push("vec");
    } else {
        eprintln!("store --embed is not tried: it needs the model");
    }
    for engine in &engines {
        let dropped = fx.axil(&["compact", "--drop-engine", engine]);
        assert!(dropped.status.success(), "{}", text(&dropped.stderr));
    }
    fx.start();
    let mut cases = vec![vec!["fts", "tokens"], vec!["since", "1h"]];
    if fx.vectors {
        cases.push(vec![
            "store",
            "decisions",
            r#"{"summary":"embedded on creation"}"#,
            "--embed",
            "summary",
        ]);
    }
    for args in &cases {
        let refused = fx.send(args, None);
        assert_eq!(refused.status.code(), Some(125), "{args:?}");
        assert!(
            text(&refused.stderr).contains("\"reason\":\"lacks\""),
            "{args:?}: {}",
            text(&refused.stderr)
        );
        fx.leasing(args);
        fx.warm_again();
        let served = fx.send(args, None);
        assert_eq!(
            served.status.code(),
            Some(0),
            "{args:?}: {}",
            text(&served.stderr)
        );
    }
    let engines = fx.live()["engines"].clone();
    assert_eq!(
        (&engines["fts"], &engines["timeseries"]),
        (&Value::Bool(true), &Value::Bool(true))
    );
    if fx.vectors {
        assert_eq!(engines["embedder"], true, "{engines}");
    }
}

/// The daemon runs a hook's recall. While a server holds the lease, the
/// hook's recall finds the database held and returns no context, at once.
#[test]
fn a_hook_recall_under_a_lease_returns_nothing_at_once() {
    let needs = cfg!(all(feature = "mcp", feature = "embed", feature = "fts"));
    if !built_with(needs, "mcp or recall") {
        return;
    }
    let fx = fixture();
    fx.start();
    let hook = |session: &str| {
        let event = serde_json::json!({
            "hook_event_name": "UserPromptSubmit",
            "session_id": session,
            "cwd": fx.proj,
            "prompt": "auth timeout during uploads",
        });
        let started = Instant::now();
        let out = axil_in(
            BIN,
            &fx.proj,
            &["hook", "run", "--dialect", "claude"],
            Some(event.to_string().as_bytes()),
        );
        assert!(out.status.success(), "{}", text(&out.stderr));
        (text(&out.stdout), started.elapsed())
    };
    // With the daemon serving, it runs the hook's recall, which finds the
    // memory and leases nothing.
    let (context, _) = hook("s-served");
    assert!(context.contains("auth timeout"), "{context}");
    assert_eq!(fx.served("recall"), 1, "{}", fx.live());
    assert_eq!(fx.live()["leases_granted"], 0, "{}", fx.live());

    let mut mcp = fx.mcp();
    mcp.initialize();
    let pid = u64::from(mcp.child.id());
    assert!(wait_for(Duration::from_secs(10), || fx.holders() == [pid]));
    let (context, took) = hook("s-leased");
    assert_eq!(context, "");
    assert!(took < Duration::from_secs(5), "took {took:?}");
    // The server still holds it. The recall's own lease, which ended with
    // its process, may not be counted out yet.
    assert!(fx.holders().contains(&pid), "{}", fx.live());
    assert!(mcp.close().success());
    fx.warm_again();
}

/// A lease held by a process that is killed ends with it: the daemon opens
/// the database again and serves.
#[test]
fn a_killed_lease_holder_lets_the_daemon_open_again() {
    if !built_with(cfg!(feature = "mcp"), "mcp, the lease holder here") {
        return;
    }
    let fx = fixture();
    fx.start();
    let mut mcp = fx.mcp();
    mcp.initialize();
    let pid = u64::from(mcp.child.id());
    assert!(wait_for(Duration::from_secs(10), || fx.holders() == [pid]));
    assert_eq!(fx.live()["state"], "leased");
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(human.contains(&format!("leased to: pid {pid}")), "{human}");
    assert!(human.contains("engines: the database is closed"), "{human}");

    signal(pid, "-KILL");
    let _ = mcp.child.wait();
    fx.warm_again();
    assert!(fx.holders().is_empty());
    let log = fx.log();
    assert!(
        log.contains(&format!("pid {pid} let go of its lease")),
        "{log}"
    );
    assert!(log.contains("opened the database again"), "{log}");
}
