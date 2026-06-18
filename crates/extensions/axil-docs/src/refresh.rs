//! Version-drift detection — know when a manifest changed so its docs
//! can be re-ingested.
//!
//! `deps sync` records each manifest's content hash (plus its lockfile's)
//! in `_dep_manifests`. A later `deps refresh` re-hashes and compares: an
//! unchanged manifest is fresh and skipped; a changed one is stale and
//! re-synced. This is the "bump a version, docs update" mechanism.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

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

/// Compare a manifest's current hashes against its `_dep_manifests` row.
pub fn manifest_drift(db: &Axil, manifest: &DetectedManifest) -> Result<Drift, DocsError> {
    let (manifest_hash, lockfile_hash) = manifest_hashes(manifest);
    let key = manifest.path.display().to_string();
    let Some(row) = crate::find_row(db, TABLE_DEP_MANIFESTS, |d| {
        d.get("path").and_then(|v| v.as_str()) == Some(key.as_str())
    })?
    else {
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
pub fn record_manifest_state(db: &Axil, manifest: &DetectedManifest) -> Result<(), DocsError> {
    let (manifest_hash, lockfile_hash) = manifest_hashes(manifest);
    let key = manifest.path.display().to_string();
    crate::delete_rows_where(db, TABLE_DEP_MANIFESTS, |d| {
        d.get("path").and_then(|v| v.as_str()) == Some(key.as_str())
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
