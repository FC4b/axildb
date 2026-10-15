//! Version-drift detection — know when a manifest changed so its docs
//! can be re-ingested.
//!
//! `deps sync` records each manifest's content hash (plus its lockfile's)
//! in `_dep_manifests`. A later `deps refresh` re-hashes and compares: an
//! unchanged manifest is fresh and skipped; a changed one is stale and
//! re-synced. This is the "bump a version, docs update" mechanism.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use axil_core::Axil;
use serde_json::json;

use crate::ingest::TABLE_DEP_MANIFESTS;
use crate::manifest::DetectedManifest;
use crate::DocsError;

/// Drift status of a manifest relative to the last recorded sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drift {
    /// No `_dep_manifests` row — this manifest has never been synced.
    New,
    /// Manifest and lockfile are byte-identical to the last sync.
    Fresh,
    /// Manifest or lockfile changed — the ingested docs are stale.
    Stale,
}

impl Drift {
    /// Lowercase identifier for status output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Drift::New => "new",
            Drift::Fresh => "fresh",
            Drift::Stale => "stale",
        }
    }

    /// Whether this manifest needs a (re-)sync.
    pub fn needs_sync(&self) -> bool {
        !matches!(self, Drift::Fresh)
    }
}

/// Content hash of a file; `"absent"` when the file does not exist.
///
/// A non-cryptographic hash is sufficient — drift detection only needs
/// "changed or not", never collision resistance.
pub fn hash_file(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        }
        Err(_) => "absent".to_string(),
    }
}

/// Content hash of a string — the in-memory analogue of [`hash_file`],
/// used to detect whether a dependency's doc text actually changed.
pub(crate) fn hash_text(s: &str) -> String {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// The `(manifest_hash, lockfile_hash)` pair for a detected manifest.
fn manifest_hashes(manifest: &DetectedManifest) -> (String, String) {
    let manifest_hash = hash_file(&manifest.path);
    let lockfile_hash = manifest
        .lockfile
        .as_deref()
        .map(hash_file)
        .unwrap_or_else(|| "none".to_string());
    (manifest_hash, lockfile_hash)
}

/// The key a manifest's `_dep_manifests` row is stored under: its path made
/// absolute and resolved (symlinks too, when the file is there). The same
/// file so has one key however the project was named: `deps sync` run
/// with `--path .` or with no path, the hooks' `deps refresh --path <abs>`,
/// or boot's own search for the project root.
pub fn manifest_key(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| axil_core::absolute_path(path))
        .display()
        .to_string()
}

/// Whether a row's stored `path` is the manifest whose key is `key`: the
/// key itself, or a path stored before keys were normalized, as given to
/// `deps`. A relative one was relative to the directory `deps` ran in, which
/// is tried as `bases` (this process's working directory and the
/// database's project directory).
fn stored_as(stored: &str, key: &str, bases: &[PathBuf]) -> bool {
    if stored == key {
        return true;
    }
    let stored = Path::new(stored);
    if stored.is_absolute() {
        return manifest_key(stored) == key;
    }
    bases
        .iter()
        .any(|base| manifest_key(&base.join(stored)) == key)
}

/// Where a relative `path` stored in `db`'s `_dep_manifests` may be
/// relative to: this process's working directory, and the project the
/// database sits in (the directory holding its `.axil/`).
fn relative_bases(db: &Axil) -> Vec<PathBuf> {
    let mut bases: Vec<PathBuf> = std::env::current_dir().into_iter().collect();
    let dir = db.path().parent();
    if let Some(project) = dir
        .filter(|dir| dir.file_name().is_some_and(|name| name == ".axil"))
        .and_then(Path::parent)
    {
        bases.push(project.to_path_buf());
    }
    bases
}

/// Compare a manifest's current hashes against its `_dep_manifests` row.
/// A row stored under the manifest's normalized key wins over one stored
/// under an older form of its path.
pub fn manifest_drift(db: &Axil, manifest: &DetectedManifest) -> Result<Drift, DocsError> {
    let (manifest_hash, lockfile_hash) = manifest_hashes(manifest);
    let key = manifest_key(&manifest.path);
    let path_of =
        |d: &serde_json::Value| d.get("path").and_then(|v| v.as_str()).map(str::to_string);
    let bases = relative_bases(db);
    let row = match crate::find_row(db, TABLE_DEP_MANIFESTS, |d| {
        path_of(d).as_deref() == Some(&key)
    })? {
        Some(row) => Some(row),
        None => crate::find_row(db, TABLE_DEP_MANIFESTS, |d| {
            path_of(d).is_some_and(|stored| stored_as(&stored, &key, &bases))
        })?,
    };
    let Some(row) = row else {
        return Ok(Drift::New);
    };
    let stored_manifest = row
        .data
        .get("manifest_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let stored_lock = row
        .data
        .get("lockfile_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    Ok(
        if stored_manifest == manifest_hash && stored_lock == lockfile_hash {
            Drift::Fresh
        } else {
            Drift::Stale
        },
    )
}

/// Upsert a manifest's current hash state into `_dep_manifests`.
///
/// Call this after a successful sync so the next `manifest_drift` has a
/// baseline to compare against.
/// The row is stored under the manifest's normalized key
/// ([`manifest_key`]); any row stored for the same file under an older
/// form of its path goes, so a project synced before keys were normalized
/// keeps one row per manifest.
pub fn record_manifest_state(db: &Axil, manifest: &DetectedManifest) -> Result<(), DocsError> {
    let (manifest_hash, lockfile_hash) = manifest_hashes(manifest);
    let key = manifest_key(&manifest.path);
    let bases = relative_bases(db);
    crate::delete_rows_where(db, TABLE_DEP_MANIFESTS, |d| {
        d.get("path")
            .and_then(|v| v.as_str())
            .is_some_and(|stored| stored_as(stored, &key, &bases))
    })?;
    let data = json!({
        "path": key,
        "ecosystem": manifest.ecosystem.as_str(),
        "manifest_hash": manifest_hash,
        "lockfile_hash": lockfile_hash,
        "lockfile_path": manifest.lockfile.as_ref().map(|p| p.display().to_string()),
    });
    db.insert(TABLE_DEP_MANIFESTS, data)
        .map_err(|e| DocsError::Db(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Ecosystem;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn hash_changes_with_content_and_handles_absence() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("Cargo.toml");
        fs::write(&f, "a = 1").unwrap();
        let h1 = hash_file(&f);
        fs::write(&f, "a = 2").unwrap();
        let h2 = hash_file(&f);
        assert_ne!(h1, h2, "content change must change the hash");
        assert_eq!(hash_file(&dir.path().join("missing")), "absent");
    }

    #[test]
    fn drift_lifecycle_new_then_fresh_then_stale() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("m.axil")).build().unwrap();
        let mpath = dir.path().join("Cargo.toml");
        fs::write(&mpath, "[package]\nname = \"x\"\n").unwrap();
        let manifest = DetectedManifest {
            path: mpath.clone(),
            ecosystem: Ecosystem::Cargo,
            lockfile: None,
        };

        // Never synced.
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::New);

        // Record state → fresh.
        record_manifest_state(&db, &manifest).unwrap();
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::Fresh);

        // Edit the manifest → stale.
        fs::write(&mpath, "[package]\nname = \"x\"\nversion = \"2\"\n").unwrap();
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::Stale);

        // Re-record → fresh again, still one row.
        record_manifest_state(&db, &manifest).unwrap();
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::Fresh);
        assert_eq!(db.list(TABLE_DEP_MANIFESTS).unwrap().len(), 1);
    }

    /// The key is the manifest's resolved absolute path, however it was
    /// named. A row stored under an older form, relative to where `deps`
    /// ran or absolute but unresolved, still reads as the manifest's, and
    /// the next record replaces it: one row per manifest, the drift kept.
    #[test]
    fn rows_stored_under_older_paths_still_count_and_are_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        std::fs::create_dir(proj.join(".axil")).unwrap();
        let db = Axil::open(proj.join(".axil/memory.axil")).build().unwrap();
        let mpath = proj.join("Cargo.toml");
        fs::write(&mpath, "[package]\nname = \"x\"\n").unwrap();
        let named = |path: PathBuf| DetectedManifest {
            path,
            ecosystem: Ecosystem::Cargo,
            lockfile: None,
        };
        let absolute = named(mpath.clone());
        let dotted = named(proj.join(".").join("Cargo.toml"));
        assert_eq!(manifest_key(&dotted.path), manifest_key(&mpath));
        assert_eq!(manifest_key(&mpath), mpath.display().to_string());

        // As a sync run in the project with `--path .` stored it, and as
        // one given the project's path unresolved (on macOS a temporary
        // directory under `/var` resolves to `/private/var`).
        record_manifest_state(&db, &absolute).unwrap();
        let rows = db.list(TABLE_DEP_MANIFESTS).unwrap();
        db.delete(&rows[0].id).unwrap();
        let unresolved = dir.path().join("Cargo.toml").display().to_string();
        for stored in ["./Cargo.toml", unresolved.as_str()] {
            let mut legacy = rows[0].data.clone();
            legacy["path"] = json!(stored);
            db.insert(TABLE_DEP_MANIFESTS, legacy).unwrap();
        }

        for manifest in [&absolute, &dotted] {
            assert_eq!(manifest_drift(&db, manifest).unwrap(), Drift::Fresh);
        }
        fs::write(&mpath, "[package]\nname = \"x\"\nversion = \"2\"\n").unwrap();
        assert_eq!(manifest_drift(&db, &absolute).unwrap(), Drift::Stale);

        record_manifest_state(&db, &absolute).unwrap();
        let rows = db.list(TABLE_DEP_MANIFESTS).unwrap();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].data["path"], json!(manifest_key(&mpath)));
        assert_eq!(manifest_drift(&db, &dotted).unwrap(), Drift::Fresh);
    }

    #[test]
    fn drift_helpers() {
        assert!(Drift::New.needs_sync());
        assert!(Drift::Stale.needs_sync());
        assert!(!Drift::Fresh.needs_sync());
        assert_eq!(Drift::Stale.as_str(), "stale");
    }

    #[test]
    fn lockfile_change_alone_marks_stale() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("m.axil")).build().unwrap();
        let mpath = dir.path().join("Cargo.toml");
        let lpath = dir.path().join("Cargo.lock");
        fs::write(&mpath, "[package]\nname = \"x\"\n").unwrap();
        fs::write(&lpath, "v1").unwrap();
        let manifest = DetectedManifest {
            path: mpath,
            ecosystem: Ecosystem::Cargo,
            lockfile: Some(lpath.clone()),
        };
        record_manifest_state(&db, &manifest).unwrap();
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::Fresh);
        // Bump only the lockfile.
        fs::write(&lpath, "v2").unwrap();
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::Stale);
    }

    #[test]
    fn unknown_manifest_is_new() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("m.axil")).build().unwrap();
        let manifest = DetectedManifest {
            path: PathBuf::from("never/seen/Cargo.toml"),
            ecosystem: Ecosystem::Cargo,
            lockfile: None,
        };
        assert_eq!(manifest_drift(&db, &manifest).unwrap(), Drift::New);
    }
}
