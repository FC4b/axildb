//! Integration tests for memory branching (create, list, diff, delete).

use axil_core::branch::{branch_create, branch_delete, branch_diff, branch_list};
use axil_core::Axil;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    let p = path.clone();
    (db, dir, p)
}

// ── Create ───────────────────────────────────────────────────────────────

#[test]
fn create_branch() {
    let (db, _dir, path) = temp_db();
    db.insert("notes", json!({"text": "hello"})).unwrap();
    drop(db);

    let branch_path = branch_create(&path, "experiment").unwrap();
    assert!(branch_path.exists(), "branch file should exist");
}

#[test]
fn create_duplicate_branch_fails() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    branch_create(&path, "dup").unwrap();
    let result = branch_create(&path, "dup");
    assert!(result.is_err(), "duplicate branch should fail");
}

#[test]
fn invalid_branch_name_rejected() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    let result = branch_create(&path, "bad name!");
    assert!(
        result.is_err(),
        "branch name with spaces/special chars should be rejected"
    );
}

// ── List ─────────────────────────────────────────────────────────────────

#[test]
fn list_branches_empty() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    let branches = branch_list(&path).unwrap();
    assert!(branches.is_empty());
}

#[test]
fn list_branches_multiple() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    // Create in non-alphabetical order to verify sorting.
    branch_create(&path, "gamma").unwrap();
    branch_create(&path, "alpha").unwrap();
    branch_create(&path, "beta").unwrap();

    let branches = branch_list(&path).unwrap();
    assert_eq!(branches, vec!["alpha", "beta", "gamma"]);
}

// ── Delete ───────────────────────────────────────────────────────────────

#[test]
fn delete_branch() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    branch_create(&path, "temp").unwrap();
    assert_eq!(branch_list(&path).unwrap().len(), 1);

    branch_delete(&path, "temp").unwrap();
    assert!(branch_list(&path).unwrap().is_empty());
}

#[test]
fn delete_nonexistent_branch_fails() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    let result = branch_delete(&path, "nope");
    assert!(result.is_err());
}

// ── Diff ─────────────────────────────────────────────────────────────────

#[test]
fn diff_empty_branch_no_changes() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    branch_create(&path, "clean").unwrap();

    let diff = branch_diff(&path, "clean").unwrap();
    assert_eq!(diff.branch_name, "clean");
    assert!(
        diff.new_tables.is_empty(),
        "no new tables expected: {:?}",
        diff.new_tables
    );
    assert!(
        diff.deleted_tables.is_empty(),
        "no deleted tables expected: {:?}",
        diff.deleted_tables
    );
    assert!(
        diff.count_diff.values().all(|&v| v == 0),
        "no count changes expected: {:?}",
        diff.count_diff
    );
}

#[test]
fn diff_detects_new_records_in_branch() {
    let (db, _dir, path) = temp_db();
    db.insert("notes", json!({"text": "main record"})).unwrap();
    drop(db);

    let bp = branch_create(&path, "add-test").unwrap();

    // Open the branch and add a record.
    let branch_db = Axil::open(&bp).build().unwrap();
    branch_db
        .insert("notes", json!({"text": "branch record"}))
        .unwrap();
    drop(branch_db);

    let diff = branch_diff(&path, "add-test").unwrap();
    let notes_diff = diff.count_diff.get("notes").copied().unwrap_or(0);
    assert_eq!(notes_diff, 1, "branch should have 1 more record than main");
}

#[test]
fn diff_detects_new_table_in_branch() {
    let (_db, _dir, path) = temp_db();
    drop(_db);

    let bp = branch_create(&path, "new-table").unwrap();

    let branch_db = Axil::open(&bp).build().unwrap();
    branch_db
        .insert("experiments", json!({"data": "test"}))
        .unwrap();
    drop(branch_db);

    let diff = branch_diff(&path, "new-table").unwrap();
    assert!(
        diff.new_tables.contains(&"experiments".to_string()),
        "expected 'experiments' in new_tables: {:?}",
        diff.new_tables
    );
}
