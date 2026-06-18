//! Index freshness detection — tells agents whether the index is up-to-date.
//!
//! Compares stored content hashes against current files on disk to detect
//! staleness without doing a full re-index. Returns a compact freshness
//! report that agents can use to decide whether to re-index.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::Axil;

use crate::indexer::{TABLE_FILES, TABLE_PROJECT};
use crate::proxy::TABLE_CODE_PROXIES;
use crate::scanner;

/// Freshness status of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FreshnessStatus {
    /// Index matches disk — no re-index needed.
    Fresh,
    /// A few files changed — incremental re-index recommended.
    Stale,
    /// Many files changed or index is very old — full re-index recommended.
    Outdated,
    /// No index exists yet.
    Missing,
}

impl FreshnessStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Stale => "stale",
            Self::Outdated => "outdated",
            Self::Missing => "missing",
        }
    }
}

/// Compact freshness report for agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessReport {
    pub status: FreshnessStatus,
    /// How many files on disk have changed since the index was built.
    pub changed_files: usize,
    /// How many new files exist that aren't in the index.
    pub new_files: usize,
    /// How many indexed files no longer exist on disk.
    pub deleted_files: usize,
    /// Total indexed files.
    pub indexed_files: usize,
    /// Total files currently on disk.
    pub disk_files: usize,
    /// When the index was last built (ISO 8601).
    pub indexed_at: String,
    /// Agent-readable guidance.
    pub recommendation: String,
    /// Total proxy records currently stored.
    pub indexed_proxies: usize,
    /// Proxies whose `path` no longer exists on disk or has changed since
    /// the last index. A non-zero value means the index is drifting and
    /// proxy pointers may be stale.
    pub stale_proxies: usize,
}

/// Check how fresh the index is compared to files on disk.
///
/// This is a fast operation — it reads file content and computes hashes
/// but does not parse or store anything. Designed to be called at the
/// start of every agent session.
pub fn check_freshness(
    db: &Axil,
    project_root: &Path,
    config: &axil_core::IndexConfig,
) -> FreshnessReport {
    // Check if index exists
    let projects = db.list(TABLE_PROJECT).unwrap_or_default();
    if projects.is_empty() {
        return FreshnessReport {
            status: FreshnessStatus::Missing,
            changed_files: 0,
            new_files: 0,
            deleted_files: 0,
            indexed_files: 0,
            disk_files: 0,
            indexed_at: String::new(),
            recommendation: "No index found. Run `axil index .` to build one.".to_string(),
            indexed_proxies: 0,
            stale_proxies: 0,
        };
    }

    let indexed_at = projects
        .first()
        .and_then(|p| p.data.get("indexed_at").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();

    let drift = scan_file_drift(db, project_root, config);
    let total_drift = drift.changed.len() + drift.new_files.len() + drift.deleted.len();
    let indexed_count = drift.indexed_count;
    let disk_count = drift.disk_count;
    let changed = drift.changed.len();
    let new = drift.new_files.len();
    let deleted = drift.deleted.len();

    let proxies = db.list(TABLE_CODE_PROXIES).unwrap_or_default();
    let indexed_proxies = proxies.len();
    let stale_proxies = if total_drift == 0 {
        0
    } else {
        proxies
            .iter()
            .filter(|p| {
                let path = match p.data.get("path").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => return false,
                };
                drift.changed.contains(path) || drift.deleted.contains(path)
            })
            .count()
    };

    // Determine status and recommendation
    let (status, recommendation) = if total_drift == 0 {
        (
            FreshnessStatus::Fresh,
            "Index is up-to-date. No action needed.".to_string(),
        )
    } else if total_drift <= 5
        || (indexed_count > 0 && total_drift * 100 / indexed_count.max(1) < 10)
    {
        (
            FreshnessStatus::Stale,
            format!(
                "{total_drift} file(s) changed. Run `axil index .` for incremental update (~instant)."
            ),
        )
    } else {
        (
            FreshnessStatus::Outdated,
            format!(
                "{total_drift} file(s) changed ({changed} modified, {new} new, {deleted} deleted). Run `axil index . --full` to rebuild."
            ),
        )
    };

    FreshnessReport {
        status,
        changed_files: changed,
        new_files: new,
        deleted_files: deleted,
        indexed_files: indexed_count,
        disk_files: disk_count,
        indexed_at,
        recommendation,
        indexed_proxies,
        stale_proxies,
    }
}

/// Return the set of relative file paths that are stale (changed or deleted since indexing).
///
/// Useful for filtering recall/search results to exclude records from stale files.
pub fn stale_file_paths(
    db: &Axil,
    project_root: &Path,
    config: &axil_core::IndexConfig,
) -> std::collections::HashSet<String> {
    let drift = scan_file_drift(db, project_root, config);
    let mut stale = drift.changed;
    stale.extend(drift.new_files);
    stale.extend(drift.deleted);
    stale
}

// ── Shared scan logic ──────────────────────────────────────────────

struct FileDrift {
    changed: std::collections::HashSet<String>,
    new_files: std::collections::HashSet<String>,
    deleted: std::collections::HashSet<String>,
    indexed_count: usize,
    disk_count: usize,
}

/// Scan files on disk and compare against stored hashes to find drift.
fn scan_file_drift(db: &Axil, project_root: &Path, config: &axil_core::IndexConfig) -> FileDrift {
    let stored_files = db.list(TABLE_FILES).unwrap_or_default();
    let mut stored_hashes: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for record in &stored_files {
        if let (Some(path), Some(hash)) = (
            record.data.get("path").and_then(|v| v.as_str()),
            record.data.get("content_hash").and_then(|v| v.as_str()),
        ) {
            stored_hashes.insert(path.to_string(), hash.to_string());
        }
    }

    let current_files = scanner::scan_files(project_root, config);
    let mut changed = std::collections::HashSet::new();
    let mut new_files = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();

    for file in &current_files {
        seen.insert(file.rel_path.clone());
        let source = match std::fs::read_to_string(&file.path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let current_hash = crate::indexer::hash_content(&source);
        match stored_hashes.get(&file.rel_path) {
            Some(stored_hash) if *stored_hash == current_hash => {}
            Some(_) => {
                changed.insert(file.rel_path.clone());
            }
            None => {
                new_files.insert(file.rel_path.clone());
            }
        }
    }

    let deleted: std::collections::HashSet<String> = stored_hashes
        .keys()
        .filter(|p| !seen.contains(p.as_str()))
        .cloned()
        .collect();

    FileDrift {
        changed,
        new_files,
        deleted,
        indexed_count: stored_files.len(),
        disk_count: current_files.len(),
    }
}

/// Convert a freshness report to a compact JSON value.
pub fn freshness_to_json(report: &FreshnessReport) -> Value {
    json!({
        "status": report.status.as_str(),
        "changed_files": report.changed_files,
        "new_files": report.new_files,
        "deleted_files": report.deleted_files,
        "indexed_files": report.indexed_files,
        "disk_files": report.disk_files,
        "indexed_at": report.indexed_at,
        "recommendation": report.recommendation,
        "indexed_proxies": report.indexed_proxies,
        "stale_proxies": report.stale_proxies,
    })
}
