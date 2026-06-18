//! Data versioning and snapshots (8b.16).
//!
//! File-copy based snapshots of the database and all companion files.
//! Supports create, list, and restore operations.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Metadata for a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// Snapshot identifier (timestamp-based).
    pub id: String,
    /// When the snapshot was created.
    pub created_at: DateTime<Utc>,
    /// Optional human-readable label.
    pub label: String,
    /// Total size of all files in bytes.
    pub total_bytes: u64,
    /// Number of files in the snapshot.
    pub file_count: usize,
    /// Per-file SHA-256 checksums.
    pub checksums: BTreeMap<String, String>,
}

/// Create a snapshot of the database and all companion files.
///
/// Copies all `.axil*` files from the database directory into a timestamped
/// subdirectory under `<db_path>.snapshots/`.
pub fn create_snapshot(db_path: &Path, label: &str) -> io::Result<SnapshotMeta> {
    let now = Utc::now();
    let id = now.format("%Y%m%d_%H%M%S").to_string();

    let snapshot_dir = snapshots_dir(db_path).join(&id);
    fs::create_dir_all(&snapshot_dir)?;

    // Find all companion files
    let files = find_companion_files(db_path)?;
    let mut checksums = BTreeMap::new();
    let mut total_bytes = 0u64;

    for src in &files {
        let name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let dest = snapshot_dir.join(&name);

        // Copy file
        fs::copy(src, &dest)?;

        // Compute checksum
        let hash = sha256_file(&dest)?;
        let size = fs::metadata(&dest)?.len();
        total_bytes += size;
        checksums.insert(name, hash);
    }

    // Also copy any FTS directory
    let fts_dir = crate::companion_path(db_path, ".fts");
    if fts_dir.is_dir() {
        let dest_fts = snapshot_dir.join("fts");
        copy_dir_recursive(&fts_dir, &dest_fts)?;
        let size = dir_size(&dest_fts);
        total_bytes += size;
        checksums.insert("fts/".to_string(), format!("dir:{size}bytes"));
    }

    let meta = SnapshotMeta {
        id,
        created_at: now,
        label: label.to_string(),
        total_bytes,
        file_count: checksums.len(),
        checksums,
    };

    // Write metadata
    let meta_path = snapshot_dir.join("snapshot.json");
    let meta_json =
        serde_json::to_string_pretty(&meta).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(&meta_path, meta_json)?;

    Ok(meta)
}

/// List all snapshots for a database.
pub fn list_snapshots(db_path: &Path) -> io::Result<Vec<SnapshotMeta>> {
    let snap_dir = snapshots_dir(db_path);
    if !snap_dir.exists() {
        return Ok(Vec::new());
    }

    let mut snapshots = Vec::new();
    for entry in fs::read_dir(&snap_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let meta_path = entry.path().join("snapshot.json");
            if meta_path.exists() {
                if let Ok(content) = fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<SnapshotMeta>(&content) {
                        snapshots.push(meta);
                    }
                }
            }
        }
    }

    snapshots.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    Ok(snapshots)
}

/// Restore a snapshot by copying files back to the database location.
///
/// Validates checksums before overwriting to prevent restoring corrupt snapshots.
pub fn restore_snapshot(db_path: &Path, snapshot_id: &str) -> io::Result<SnapshotMeta> {
    let snap_dir = snapshots_dir(db_path).join(snapshot_id);
    if !snap_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("snapshot not found: {snapshot_id}"),
        ));
    }

    let meta_path = snap_dir.join("snapshot.json");
    let meta_content = fs::read_to_string(&meta_path)?;
    let meta: SnapshotMeta = serde_json::from_str(&meta_content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Validate checksums before restoring
    for (name, expected_hash) in &meta.checksums {
        if name.ends_with('/') {
            continue; // directory entry
        }
        let src = snap_dir.join(name);
        if src.exists() {
            let actual_hash = sha256_file(&src)?;
            if actual_hash != *expected_hash {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
                    ),
                ));
            }
        }
    }

    // Remove current companion files that may not exist in the snapshot.
    // This prevents stale indexes from persisting after restore.
    for suffix in &[".vec", ".graph", ".ts"] {
        let companion = crate::companion_path(db_path, suffix);
        if companion.exists() {
            let _ = fs::remove_file(&companion);
        }
    }
    let fts_path = crate::companion_path(db_path, ".fts");
    if fts_path.exists() {
        let _ = fs::remove_dir_all(&fts_path);
    }

    // Restore files from snapshot
    let db_dir = db_path.parent().unwrap_or(Path::new("."));
    for (name, _hash) in &meta.checksums {
        if name.ends_with('/') {
            continue;
        }
        let src = snap_dir.join(name);
        let dest = db_dir.join(name);
        if src.exists() {
            fs::copy(&src, &dest)?;
        }
    }

    // Restore FTS directory if present in snapshot
    let snap_fts = snap_dir.join("fts");
    if snap_fts.is_dir() {
        let dest_fts = crate::companion_path(db_path, ".fts");
        copy_dir_recursive(&snap_fts, &dest_fts)?;
    }

    Ok(meta)
}

// ── Helpers ─────────────────────────────────────────────────────────

fn snapshots_dir(db_path: &Path) -> PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".snapshots");
    PathBuf::from(p)
}

fn find_companion_files(db_path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    // Main database file
    if db_path.exists() {
        files.push(db_path.to_path_buf());
    }

    // Companion files: .vec, .graph, .ts
    for suffix in &[".vec", ".graph", ".ts"] {
        let companion = crate::companion_path(db_path, suffix);
        if companion.exists() {
            files.push(companion);
        }
    }

    Ok(files)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    use io::Read;
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn dir_size(path: &Path) -> u64 {
    let mut size = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    size += meta.len();
                } else if meta.is_dir() {
                    size += dir_size(&entry.path());
                }
            }
        }
    }
    size
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let dest_path = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_list_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        fs::write(&db_path, b"test database content").unwrap();

        let meta = create_snapshot(&db_path, "test snapshot").unwrap();
        assert_eq!(meta.label, "test snapshot");
        assert!(meta.total_bytes > 0);
        assert!(meta.file_count > 0);

        let snapshots = list_snapshots(&db_path).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].id, meta.id);
    }

    #[test]
    fn restore_snapshot_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let original_content = b"original database content";
        fs::write(&db_path, original_content).unwrap();

        let meta = create_snapshot(&db_path, "before change").unwrap();

        // Modify the database
        fs::write(&db_path, b"modified content").unwrap();
        assert_ne!(fs::read(&db_path).unwrap(), original_content);

        // Restore
        restore_snapshot(&db_path, &meta.id).unwrap();
        assert_eq!(fs::read(&db_path).unwrap(), original_content);
    }

    #[test]
    fn empty_snapshots_list() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("nonexistent.axil");
        let snapshots = list_snapshots(&db_path).unwrap();
        assert!(snapshots.is_empty());
    }
}
