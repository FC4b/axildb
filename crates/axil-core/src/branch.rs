//! Memory branching — create, list, delete, and diff database branches.
//!
//! Branches are file-level copies of the database and all companion files.
//! No merge support (yet) — branches are for experimentation and rollback.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AxilError, Result};

/// Difference between a branch and its parent database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDiff {
    /// Branch name.
    pub branch_name: String,
    /// Per-table record counts in the main database.
    pub main_tables: BTreeMap<String, usize>,
    /// Per-table record counts in the branch.
    pub branch_tables: BTreeMap<String, usize>,
    /// Tables that exist only in the branch.
    pub new_tables: Vec<String>,
    /// Tables that exist only in main.
    pub deleted_tables: Vec<String>,
    /// Per-table count difference (branch - main). Positive = branch has more.
    pub count_diff: BTreeMap<String, i64>,
}

/// Known companion file suffixes (same as db.rs COMPANION_SUFFIXES).
const COMPANION_SUFFIXES: &[&str] = &[".vec", ".graph", ".ts"];

/// Build the branch base path: `<db_path>.branch.<name>`.
fn branch_path(db_path: &Path, name: &str) -> PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(format!(".branch.{name}"));
    PathBuf::from(p)
}

/// Create a branch by copying the database and all companion files.
///
/// The branch is stored at `<db_path>.branch.<name>` with companion files
/// at `<db_path>.branch.<name>.vec`, etc.
pub fn branch_create(db_path: &Path, name: &str) -> Result<PathBuf> {
    validate_branch_name(name)?;

    let dest = branch_path(db_path, name);
    if dest.exists() {
        return Err(AxilError::plugin(format!(
            "branch '{name}' already exists at {}",
            dest.display()
        )));
    }

    // Copy main database file.
    if db_path.exists() {
        fs::copy(db_path, &dest)
            .map_err(|e| AxilError::plugin(format!("failed to copy database: {e}")))?;
    } else {
        return Err(AxilError::plugin(format!(
            "database not found: {}",
            db_path.display()
        )));
    }

    // Copy companion files.
    for suffix in COMPANION_SUFFIXES {
        let src = crate::companion_path(db_path, suffix);
        if src.exists() {
            let companion_dest = crate::companion_path(&dest, suffix);
            fs::copy(&src, &companion_dest).map_err(|e| {
                AxilError::plugin(format!(
                    "failed to copy companion file {}: {e}",
                    src.display()
                ))
            })?;
        }
    }

    // Copy FTS directory if it exists.
    let fts_src = crate::companion_path(db_path, ".fts");
    if fts_src.is_dir() {
        let fts_dest = crate::companion_path(&dest, ".fts");
        copy_dir_recursive(&fts_src, &fts_dest)
            .map_err(|e| AxilError::plugin(format!("failed to copy FTS directory: {e}")))?;
    }

    Ok(dest)
}

/// List all branches for a database.
pub fn branch_list(db_path: &Path) -> Result<Vec<String>> {
    let parent = db_path.parent().unwrap_or(Path::new("."));
    let db_name = db_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    let prefix = format!("{db_name}.branch.");
    let mut branches = Vec::new();

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(branch_name) = name_str.strip_prefix(&prefix) {
                // Exclude companion files of branches (e.g. .branch.foo.vec).
                if !branch_name.contains('.') {
                    branches.push(branch_name.to_string());
                }
            }
        }
    }

    branches.sort();
    Ok(branches)
}

/// Delete a branch and all its companion files.
pub fn branch_delete(db_path: &Path, name: &str) -> Result<()> {
    validate_branch_name(name)?;

    let dest = branch_path(db_path, name);
    if !dest.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    // Delete companion files first.
    for suffix in COMPANION_SUFFIXES {
        let companion = crate::companion_path(&dest, suffix);
        if companion.exists() {
            let _ = fs::remove_file(&companion);
        }
    }

    // Delete FTS directory.
    let fts = crate::companion_path(&dest, ".fts");
    if fts.is_dir() {
        let _ = fs::remove_dir_all(&fts);
    }

    // Delete main branch file.
    fs::remove_file(&dest)
        .map_err(|e| AxilError::plugin(format!("failed to delete branch: {e}")))?;

    Ok(())
}

/// Get the database path for a named branch.
///
/// Returns the path if the branch exists, or an error if not found.
/// Use this path with `AXIL_DB` to switch to the branch.
pub fn branch_switch(db_path: &Path, name: &str) -> Result<PathBuf> {
    validate_branch_name(name)?;

    let bp = branch_path(db_path, name);
    if !bp.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    Ok(bp)
}

/// Compare a branch to the main database.
///
/// Opens both databases, compares record counts per table, and reports
/// which tables are new, deleted, or have different counts.
pub fn branch_diff(db_path: &Path, name: &str) -> Result<BranchDiff> {
    validate_branch_name(name)?;

    let branch_db_path = branch_path(db_path, name);
    if !branch_db_path.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    // Open both databases.
    let main_db = crate::Axil::open(db_path).build()?;
    let branch_db = crate::Axil::open(&branch_db_path).build()?;

    let main_tc = main_db.tables_with_counts()?;
    let branch_tc = branch_db.tables_with_counts()?;

    let main_tables: BTreeMap<String, usize> = main_tc.into_iter().collect();
    let branch_tables: BTreeMap<String, usize> = branch_tc.into_iter().collect();

    let new_tables: Vec<String> = branch_tables
        .keys()
        .filter(|k| !main_tables.contains_key(*k))
        .cloned()
        .collect();

    let deleted_tables: Vec<String> = main_tables
        .keys()
        .filter(|k| !branch_tables.contains_key(*k))
        .cloned()
        .collect();

    let mut count_diff = BTreeMap::new();
    for table in main_tables.keys().chain(branch_tables.keys()) {
        if count_diff.contains_key(table) {
            continue;
        }
        let main_count = main_tables.get(table).copied().unwrap_or(0) as i64;
        let branch_count = branch_tables.get(table).copied().unwrap_or(0) as i64;
        let diff = branch_count - main_count;
        if diff != 0 {
            count_diff.insert(table.clone(), diff);
        }
    }

    Ok(BranchDiff {
        branch_name: name.to_string(),
        main_tables,
        branch_tables,
        new_tables,
        deleted_tables,
        count_diff,
    })
}

/// Validate branch name: alphanumeric, hyphens, underscores only.
fn validate_branch_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(AxilError::InvalidQuery(
            "branch name cannot be empty".to_string(),
        ));
    }
    if name.len() > 64 {
        return Err(AxilError::InvalidQuery(
            "branch name too long (max 64 chars)".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(AxilError::InvalidQuery(
            "branch name must contain only alphanumeric, hyphen, or underscore characters"
                .to_string(),
        ));
    }
    Ok(())
}

/// How to resolve conflicts when merging a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Branch record wins on conflict (default).
    #[default]
    BranchWins,
    /// Main record wins on conflict (skip conflicting branch records).
    MainWins,
    /// Keep both by inserting the branch record as a new record.
    KeepBoth,
}

impl std::str::FromStr for MergeStrategy {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "main-wins" => Ok(Self::MainWins),
            "keep-both" => Ok(Self::KeepBoth),
            _ => Ok(Self::BranchWins),
        }
    }
}

/// Result of merging a branch into the main database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeReport {
    /// Branch that was merged.
    pub branch_name: String,
    /// Number of new records added from the branch.
    pub records_added: usize,
    /// Number of records updated (conflicts resolved in favor of branch).
    pub records_updated: usize,
    /// Number of records skipped (conflicts resolved in favor of main).
    pub records_skipped: usize,
    /// Tables that were affected.
    pub tables_affected: Vec<String>,
    /// Whether indexes (vector, graph, FTS) need rebuilding after merge.
    pub indexes_need_rebuild: bool,
}

/// Merge a branch back into the main database.
///
/// Opens both databases, iterates all tables in the branch, and for each
/// record either inserts (if new) or applies the merge strategy (if conflict).
///
/// Note: companion indexes (vector, graph, FTS) are NOT merged — after merge,
/// run `axil doctor` and `axil heal` to rebuild indexes if needed.
pub fn branch_merge(db_path: &Path, name: &str, strategy: MergeStrategy) -> Result<MergeReport> {
    validate_branch_name(name)?;

    let branch_db_path = branch_path(db_path, name);
    if !branch_db_path.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    let main_db = crate::Axil::open(db_path).build()?;
    let branch_db = crate::Axil::open(&branch_db_path).build()?;

    let branch_tables = branch_db.tables_with_counts()?;
    let mut records_added = 0usize;
    let mut records_updated = 0usize;
    let mut records_skipped = 0usize;
    let mut tables_affected = Vec::new();

    for (table, _count) in &branch_tables {
        let branch_records = branch_db.list(table)?;
        if branch_records.is_empty() {
            continue;
        }

        // Batch-load main records for this table to avoid N+1 get() calls.
        let main_records: BTreeMap<crate::record::RecordId, crate::record::Record> = main_db
            .list(table)
            .unwrap_or_default()
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect();

        let mut table_changed = false;

        for branch_record in &branch_records {
            match main_records.get(&branch_record.id) {
                None => {
                    // New record — insert into main.
                    // Use storage directly to preserve the original ID and timestamps.
                    main_db.storage().insert(branch_record)?;
                    records_added += 1;
                    table_changed = true;
                }
                Some(ref main_record) => {
                    // Record exists in both — check if it changed.
                    if main_record.data == branch_record.data
                        && main_record.updated_at == branch_record.updated_at
                    {
                        // Identical — skip.
                        continue;
                    }

                    // Conflict: records differ.
                    match strategy {
                        MergeStrategy::BranchWins => {
                            main_db.update(&branch_record.id, branch_record.data.clone())?;
                            records_updated += 1;
                            table_changed = true;
                        }
                        MergeStrategy::MainWins => {
                            records_skipped += 1;
                        }
                        MergeStrategy::KeepBoth => {
                            // Insert the branch version as a new record.
                            main_db.insert(table, branch_record.data.clone())?;
                            records_added += 1;
                            table_changed = true;
                        }
                    }
                }
            }
        }

        if table_changed {
            tables_affected.push(table.clone());
        }
    }

    Ok(MergeReport {
        branch_name: name.to_string(),
        indexes_need_rebuild: records_added > 0 || records_updated > 0,
        records_added,
        records_updated,
        records_skipped,
        tables_affected,
    })
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
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
    use crate::Axil;
    use serde_json::json;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        // Insert a record so the DB file exists.
        db.insert("notes", json!({"text": "hello"})).unwrap();
        drop(db);
        (dir, db_path)
    }

    #[test]
    fn create_and_list_branch() {
        let (_dir, db_path) = temp_db();

        let branch_path = branch_create(&db_path, "experiment").unwrap();
        assert!(branch_path.exists());

        let branches = branch_list(&db_path).unwrap();
        assert_eq!(branches, vec!["experiment"]);
    }

    #[test]
    fn delete_branch() {
        let (_dir, db_path) = temp_db();

        branch_create(&db_path, "to-delete").unwrap();
        assert!(!branch_list(&db_path).unwrap().is_empty());

        branch_delete(&db_path, "to-delete").unwrap();
        assert!(branch_list(&db_path).unwrap().is_empty());
    }

    #[test]
    fn duplicate_branch_errors() {
        let (_dir, db_path) = temp_db();

        branch_create(&db_path, "dup").unwrap();
        let result = branch_create(&db_path, "dup");
        assert!(result.is_err());
    }

    #[test]
    fn delete_nonexistent_branch_errors() {
        let (_dir, db_path) = temp_db();
        let result = branch_delete(&db_path, "nope");
        assert!(result.is_err());
    }

    #[test]
    fn diff_branch() {
        let (_dir, db_path) = temp_db();

        // Create branch.
        let bp = branch_create(&db_path, "diff-test").unwrap();

        // Add a record to the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .insert("notes", json!({"text": "branch record"}))
            .unwrap();
        drop(branch_db);

        let diff = branch_diff(&db_path, "diff-test").unwrap();
        assert_eq!(diff.branch_name, "diff-test");
        // Branch should have 1 more record in "notes".
        assert_eq!(diff.count_diff.get("notes"), Some(&1));
    }

    #[test]
    fn merge_branch_adds_new_records() {
        let (_dir, db_path) = temp_db();

        let bp = branch_create(&db_path, "feature").unwrap();

        // Add a record only in the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .insert("tasks", json!({"text": "branch-only task"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "feature", MergeStrategy::BranchWins).unwrap();
        assert_eq!(report.records_added, 1);
        assert!(report.tables_affected.contains(&"tasks".to_string()));

        // Verify the record exists in main.
        let main_db = Axil::open(&db_path).build().unwrap();
        let tasks = main_db.list("tasks").unwrap();
        assert!(tasks.iter().any(|r| r.data["text"] == "branch-only task"));
    }

    #[test]
    fn merge_branch_wins_updates_conflicts() {
        let (_dir, db_path) = temp_db();

        // Get the existing record ID.
        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "fix").unwrap();

        // Modify the record in the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "updated in branch"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "fix", MergeStrategy::BranchWins).unwrap();
        assert_eq!(report.records_updated, 1);

        let main_db = Axil::open(&db_path).build().unwrap();
        let r = main_db.get(&record_id).unwrap().unwrap();
        assert_eq!(r.data["text"], "updated in branch");
    }

    #[test]
    fn merge_main_wins_skips_conflicts() {
        let (_dir, db_path) = temp_db();

        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "skip").unwrap();

        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "branch change"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "skip", MergeStrategy::MainWins).unwrap();
        assert_eq!(report.records_skipped, 1);
        assert_eq!(report.records_updated, 0);

        // Main record unchanged.
        let main_db = Axil::open(&db_path).build().unwrap();
        let r = main_db.get(&record_id).unwrap().unwrap();
        assert_eq!(r.data["text"], "hello");
    }

    #[test]
    fn merge_keep_both_duplicates() {
        let (_dir, db_path) = temp_db();

        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "both").unwrap();

        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "branch version"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "both", MergeStrategy::KeepBoth).unwrap();
        assert_eq!(report.records_added, 1);

        // Main should now have 2 records in notes.
        let main_db = Axil::open(&db_path).build().unwrap();
        let notes = main_db.list("notes").unwrap();
        assert_eq!(notes.len(), 2);
    }

    #[test]
    fn merge_nonexistent_branch_errors() {
        let (_dir, db_path) = temp_db();
        let result = branch_merge(&db_path, "nope", MergeStrategy::BranchWins);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_branch_names() {
        let (_dir, db_path) = temp_db();

        assert!(branch_create(&db_path, "").is_err());
        assert!(branch_create(&db_path, "has.dot").is_err());
        assert!(branch_create(&db_path, "has space").is_err());
        assert!(branch_create(&db_path, "ok-name_123").is_ok());
    }

    #[test]
    fn list_empty_branches() {
        let (_dir, db_path) = temp_db();
        let branches = branch_list(&db_path).unwrap();
        assert!(branches.is_empty());
    }
}
