//! Code-ref resolution and content fingerprinting.
//!
//! A cached answer can reference code. To invalidate the answer when that
//! code changes, each reference carries a *fingerprint* captured at
//! put time; on read the fingerprint is recomputed and compared. A
//! mismatch means the referenced code moved on and the cached answer can
//! no longer be trusted.
//!
//! ## Two fingerprint signals
//!
//! Each reference captures whichever of these are available:
//!
//! - **File hash** — a hash of the referenced file's on-disk content. This
//!   is the primary signal: it is recomputed straight from disk on every
//!   read, so a raw edit invalidates the entry *without* waiting for the
//!   indexer to re-run. It is captured whenever the reference resolves to a
//!   real file on disk.
//! - **Proxy hash** — a hash of the matching `_idx_code_proxies` row's
//!   structural text. This is the code-aware signal: it changes when the
//!   indexer records a structural change (a new signature, a rewritten
//!   summary) and, critically, goes absent when the symbol is removed and
//!   re-indexed. It is captured only when a proxy row matches the reference.
//!
//! An entry is stale when *either* stored signal no longer matches the
//! freshly computed one (a changed hash, or a hash that was present at put
//! time but is gone now).
//!
//! This module reads the `_idx_code_proxies` table directly by name rather
//! than depending on `axil-indexer`, keeping `axil-cache` a leaf crate —
//! the same approach `axil-checkpoint` takes when it reads `_sessions`
//! without depending on `axil-memory`.

use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use axil_core::{Axil, Op};

/// The structural-proxy table owned by `axil-indexer`. Duplicated here as a
/// string constant so `axil-cache` needs no dependency on the indexer crate
/// (proxies are plain JSON rows readable through the core query API).
pub const TABLE_CODE_PROXIES: &str = "_idx_code_proxies";

/// The content fingerprint of a single code reference, captured at put time
/// and recomputed on read.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodeFingerprint {
    /// Hash of the matching proxy row's structural text, when a proxy
    /// matched. `None` when the reference did not resolve to a proxy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_hash: Option<String>,
    /// Hash of the referenced file's on-disk content, when the file was
    /// readable. `None` when the reference carried no resolvable path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    /// Absolute path the `file_hash` was computed from. Stored so the
    /// re-hash on read targets the same file regardless of the working
    /// directory the read runs from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

impl CodeFingerprint {
    /// `true` when neither signal was captured — the reference is a bare
    /// pointer with nothing to invalidate against.
    pub fn is_empty(&self) -> bool {
        self.proxy_hash.is_none() && self.file_hash.is_none()
    }
}

/// Why a reference is considered stale, for a human-readable miss reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    /// The proxy row's structural text changed, or the symbol was removed.
    ProxyChanged,
    /// The referenced file's content changed, or the file is gone.
    FileChanged,
}

impl StaleReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProxyChanged => "proxy content changed",
            Self::FileChanged => "file content changed",
        }
    }
}

/// Resolve a `--code-ref`-style spec into a normalized code-ref object with
/// its fingerprint attached, ready to store on a cache entry.
///
/// Accepted spec forms mirror `axil store --code-ref`:
/// - `proxy_id` / `canonical_id` — exact match against `_idx_code_proxies`.
/// - `path` / `path:line` — matched against proxy rows by path (closest
///   symbol wins when a line is given); always also fingerprinted against
///   the file on disk so a bare path with no index still invalidates.
///
/// `base_dir` is the directory that a relative `path` is resolved against
/// (the working directory at put time). The resolved absolute path is
/// stored in the fingerprint so the read-time re-hash is location-stable.
pub fn resolve_ref(db: &Axil, spec: &str, base_dir: &Path) -> Value {
    let proxies = db.list(TABLE_CODE_PROXIES).unwrap_or_default();

    // 1) Exact proxy_id / canonical_id match.
    if let Some(proxy) = proxies.iter().find(|r| {
        r.data.get("proxy_id").and_then(|v| v.as_str()) == Some(spec)
            || r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(spec)
    }) {
        return build_ref_from_proxy(&proxy.data, base_dir);
    }

    // 2) path[:line] form.
    let (path_part, line_part) = split_path_line(spec);
    let mut path_matches: Vec<&Value> = proxies
        .iter()
        .filter(|r| r.data.get("path").and_then(|v| v.as_str()) == Some(path_part))
        .map(|r| &r.data)
        .collect();

    let matched_proxy: Option<&Value> = if path_matches.is_empty() {
        None
    } else if let Some(line) = line_part {
        // Prefer the symbol proxy whose line_start is closest to the line.
        path_matches.sort_by_key(|d| {
            let start = d.get("line_start").and_then(|v| v.as_u64()).unwrap_or(0);
            (line as i64 - start as i64).abs()
        });
        path_matches.first().copied()
    } else {
        // No line — prefer the file proxy over any symbol proxy.
        path_matches
            .iter()
            .find(|d| d.get("kind").and_then(|v| v.as_str()) == Some("file"))
            .or_else(|| path_matches.first())
            .copied()
    };

    match matched_proxy {
        Some(proxy) => build_ref_from_proxy(proxy, base_dir),
        // No proxy matched — keep the path pointer and fingerprint the file
        // directly, so code refs work even before (or without) an index.
        None => {
            let fp = fingerprint_path_only(path_part, base_dir);
            let mut obj = serde_json::Map::new();
            obj.insert("spec".into(), json!(spec));
            obj.insert("path".into(), json!(path_part));
            if let Some(line) = line_part {
                obj.insert("line".into(), json!(line));
            }
            obj.insert("fingerprint".into(), serde_json::to_value(&fp).unwrap_or(Value::Null));
            Value::Object(obj)
        }
    }
}

/// Build a normalized code-ref object (pointer fields + fingerprint) from a
/// matched proxy row.
fn build_ref_from_proxy(proxy: &Value, base_dir: &Path) -> Value {
    let mut obj = serde_json::Map::new();
    for key in [
        "proxy_id",
        "canonical_id",
        "path",
        "symbol",
        "line_start",
        "line_end",
    ] {
        if let Some(v) = proxy.get(key) {
            obj.insert(key.into(), v.clone());
        }
    }
    let proxy_hash = proxy
        .get("proxy_text")
        .and_then(|v| v.as_str())
        .map(hash_str);
    let (file_hash, file_path) = proxy
        .get("path")
        .and_then(|v| v.as_str())
        .map(|p| hash_file(p, base_dir))
        .unwrap_or((None, None));
    let fp = CodeFingerprint {
        proxy_hash,
        file_hash,
        file_path,
    };
    obj.insert("fingerprint".into(), serde_json::to_value(&fp).unwrap_or(Value::Null));
    Value::Object(obj)
}

/// Fingerprint a path that resolved to no proxy — file signal only.
fn fingerprint_path_only(path: &str, base_dir: &Path) -> CodeFingerprint {
    let (file_hash, file_path) = hash_file(path, base_dir);
    CodeFingerprint {
        proxy_hash: None,
        file_hash,
        file_path,
    }
}

/// Recompute the *current* fingerprint of a stored code-ref, for comparison
/// against the fingerprint captured at put time.
///
/// The proxy signal is recomputed by re-reading the matching
/// `_idx_code_proxies` row (a removed symbol yields `None`). The file signal
/// is recomputed from the absolute path stored at put time when present, so
/// the read is independent of the working directory.
pub fn current_fingerprint(db: &Axil, code_ref: &Value, base_dir: &Path) -> CodeFingerprint {
    let proxy_id = code_ref.get("proxy_id").and_then(|v| v.as_str());
    let canonical_id = code_ref.get("canonical_id").and_then(|v| v.as_str());

    // Recompute the proxy hash only when the stored ref pointed at a proxy —
    // a path-only ref never had one and must not gain one on read.
    //
    // Resolve the proxy row through a scoped field query instead of pulling the
    // whole `_idx_code_proxies` table into the extension and scanning it by
    // hand. `proxy_id` is the precise pointer, so try it first and fall back to
    // `canonical_id`, preserving the prior "match either id" behaviour. A
    // removed symbol matches no row → `None` → the ref reads as stale, exactly
    // as before.
    //
    // Axil's storage keeps only a table→ids index, so the query engine still
    // resolves these `where_field` equalities with a filtered table read; this
    // expresses the point lookup (and benefits automatically if a per-field
    // index is ever added) rather than being asymptotically cheaper today.
    let proxy_hash = if proxy_id.is_some() || canonical_id.is_some() {
        lookup_proxy_text(db, "proxy_id", proxy_id)
            .or_else(|| lookup_proxy_text(db, "canonical_id", canonical_id))
            .as_deref()
            .map(hash_str)
    } else {
        None
    };

    // Prefer the absolute path recorded at put time; fall back to the
    // relative pointer resolved against base_dir.
    let stored_fp = code_ref
        .get("fingerprint")
        .and_then(|v| serde_json::from_value::<CodeFingerprint>(v.clone()).ok())
        .unwrap_or_default();
    let (file_hash, file_path) = match stored_fp.file_path.as_deref() {
        Some(abs) => (read_and_hash(Path::new(abs)), Some(abs.to_string())),
        None => code_ref
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| hash_file(p, base_dir))
            .unwrap_or((None, None)),
    };

    CodeFingerprint {
        proxy_hash,
        file_hash,
        file_path,
    }
}

/// Fetch the `proxy_text` of the first `_idx_code_proxies` row whose `field`
/// equals `id`, via a scoped field query. `None` when `id` is `None` or no row
/// matches (e.g. the symbol was removed and re-indexed).
fn lookup_proxy_text(db: &Axil, field: &str, id: Option<&str>) -> Option<String> {
    let id = id?;
    db.query()
        .table(TABLE_CODE_PROXIES)
        .where_field(field, Op::Eq, json!(id))
        .limit(1)
        .exec()
        .ok()?
        .into_iter()
        .next()
        .and_then(|r| {
            r.data
                .get("proxy_text")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
}

/// Compare a stored fingerprint against a freshly computed one. Returns the
/// reason the reference is stale, or `None` when it still matches.
///
/// A signal that was captured at put time (`Some`) but no longer matches —
/// whether because the content changed or the source vanished (`None`) —
/// makes the reference stale. A signal that was never captured is ignored,
/// so a path-only ref is judged on its file hash alone.
pub fn staleness(stored: &CodeFingerprint, current: &CodeFingerprint) -> Option<StaleReason> {
    if stored.proxy_hash.is_some() && stored.proxy_hash != current.proxy_hash {
        return Some(StaleReason::ProxyChanged);
    }
    if stored.file_hash.is_some() && stored.file_hash != current.file_hash {
        return Some(StaleReason::FileChanged);
    }
    None
}

/// Split a `path` or `path:line` spec into its path and optional line.
fn split_path_line(spec: &str) -> (&str, Option<u64>) {
    match spec.rsplit_once(':') {
        Some((p, l)) => match l.parse::<u64>() {
            Ok(n) => (p, Some(n)),
            Err(_) => (spec, None),
        },
        None => (spec, None),
    }
}

/// Resolve `path` against `base_dir` (unless already absolute), hash the
/// file's content, and return `(hash, absolute_path)`. Both are `None` when
/// the file cannot be read.
fn hash_file(path: &str, base_dir: &Path) -> (Option<String>, Option<String>) {
    let candidate = {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir.join(p)
        }
    };
    let abs = std::fs::canonicalize(&candidate).unwrap_or(candidate);
    match read_and_hash(&abs) {
        Some(h) => (Some(h), Some(path_to_string(&abs))),
        None => (None, None),
    }
}

fn read_and_hash(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|bytes| {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    })
}

fn hash_str(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn path_to_string(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("t.axil")).build().unwrap();
        (db, dir)
    }

    #[test]
    fn path_only_ref_captures_file_hash() {
        let (db, dir) = temp_db();
        let file = dir.path().join("foo.txt");
        std::fs::write(&file, "hello").unwrap();
        let r = resolve_ref(&db, "foo.txt", dir.path());
        let fp: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();
        assert!(fp.file_hash.is_some());
        assert!(fp.proxy_hash.is_none());
        assert!(fp.file_path.is_some());
    }

    #[test]
    fn file_edit_makes_ref_stale() {
        let (db, dir) = temp_db();
        let file = dir.path().join("foo.txt");
        std::fs::write(&file, "before").unwrap();
        let r = resolve_ref(&db, "foo.txt", dir.path());
        let stored: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();

        // Unchanged file: not stale.
        let now = current_fingerprint(&db, &r, dir.path());
        assert_eq!(staleness(&stored, &now), None);

        // Edit the file: stale via the file signal.
        std::fs::write(&file, "after").unwrap();
        let now = current_fingerprint(&db, &r, dir.path());
        assert_eq!(staleness(&stored, &now), Some(StaleReason::FileChanged));
    }

    #[test]
    fn vanished_file_is_stale() {
        let (db, dir) = temp_db();
        let file = dir.path().join("gone.txt");
        std::fs::write(&file, "x").unwrap();
        let r = resolve_ref(&db, "gone.txt", dir.path());
        let stored: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();
        std::fs::remove_file(&file).unwrap();
        let now = current_fingerprint(&db, &r, dir.path());
        assert_eq!(staleness(&stored, &now), Some(StaleReason::FileChanged));
    }

    #[test]
    fn proxy_ref_captures_and_detects_proxy_change() {
        let (db, dir) = temp_db();
        // Seed a proxy row directly (no indexer dependency in the test).
        db.insert(
            TABLE_CODE_PROXIES,
            json!({
                "proxy_id": "px1",
                "kind": "symbol",
                "path": "src/lib.rs",
                "symbol": "login",
                "proxy_text": "fn login(user)"
            }),
        )
        .unwrap();
        let r = resolve_ref(&db, "px1", dir.path());
        let stored: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();
        assert!(stored.proxy_hash.is_some());

        // Unchanged proxy: not stale.
        assert_eq!(
            staleness(&stored, &current_fingerprint(&db, &r, dir.path())),
            None
        );

        // Rewrite the proxy's structural text: stale via the proxy signal.
        let px = db.list(TABLE_CODE_PROXIES).unwrap().pop().unwrap();
        db.update(
            &px.id,
            json!({
                "proxy_id": "px1",
                "kind": "symbol",
                "path": "src/lib.rs",
                "symbol": "login",
                "proxy_text": "fn login(user, mfa_token)"
            }),
        )
        .unwrap();
        assert_eq!(
            staleness(&stored, &current_fingerprint(&db, &r, dir.path())),
            Some(StaleReason::ProxyChanged)
        );
    }

    #[test]
    fn vanished_proxy_is_stale() {
        let (db, dir) = temp_db();
        db.insert(
            TABLE_CODE_PROXIES,
            json!({"proxy_id": "px2", "kind": "symbol", "path": "a.rs", "proxy_text": "body"}),
        )
        .unwrap();
        let r = resolve_ref(&db, "px2", dir.path());
        let stored: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();
        // Remove the proxy (symbol deleted + re-indexed).
        let px = db.list(TABLE_CODE_PROXIES).unwrap().pop().unwrap();
        db.delete(&px.id).unwrap();
        assert_eq!(
            staleness(&stored, &current_fingerprint(&db, &r, dir.path())),
            Some(StaleReason::ProxyChanged)
        );
    }

    #[test]
    fn current_fingerprint_resolves_proxy_via_field_lookup() {
        let (db, dir) = temp_db();
        // Seed a proxy addressed by canonical_id only (no proxy_id) to exercise
        // the canonical_id fallback branch of the scoped field lookup.
        db.insert(
            TABLE_CODE_PROXIES,
            json!({
                "canonical_id": "cargo::auth::login",
                "kind": "symbol",
                "path": "src/auth.rs",
                "proxy_text": "fn login(user)"
            }),
        )
        .unwrap();
        let r = resolve_ref(&db, "cargo::auth::login", dir.path());
        let stored: CodeFingerprint =
            serde_json::from_value(r.get("fingerprint").unwrap().clone()).unwrap();
        assert!(stored.proxy_hash.is_some());

        // current_fingerprint must recompute the same hash through the new
        // where_field path (proxy_id absent → canonical_id fallback).
        let now = current_fingerprint(&db, &r, dir.path());
        assert_eq!(now.proxy_hash, Some(hash_str("fn login(user)")));
        assert_eq!(staleness(&stored, &now), None);
    }

    #[test]
    fn split_path_line_parses_trailing_line() {
        assert_eq!(split_path_line("src/a.rs:42"), ("src/a.rs", Some(42)));
        assert_eq!(split_path_line("src/a.rs"), ("src/a.rs", None));
        // A trailing non-numeric segment is not a line number.
        assert_eq!(split_path_line("C:\\x"), ("C:\\x", None));
    }
}
