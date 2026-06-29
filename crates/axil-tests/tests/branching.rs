//! Integration tests for memory branching (create, list, diff, delete).

use axil_core::branch::{branch_create, branch_delete, branch_diff, branch_list};
use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use axil_vector::AxilBuilderVectorExt;
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

// ── Point-in-time consistency ──────────────────────────────────────────────

/// A branch taken from the live handle is internally consistent: the core
/// record count must match what both the vector and FTS companion indexes
/// reflect. This exercises the cross-platform copy path — on Windows redb holds
/// a byte-range lock on an open core file, so `Axil::branch_create` must flush
/// the engines and close every handle before copying.
#[test]
fn live_branch_is_internally_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");

    const DIMS: usize = 4;
    const N: usize = 8;

    let db = Axil::open(&path)
        .with_vector(DIMS)
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();

    // Insert N records, each with a unique FTS term and a manual vector. The
    // insert path auto-indexes the text field for FTS; add a vector per record.
    let mut ids = Vec::with_capacity(N);
    for i in 0..N {
        let rec = db
            .insert("notes", json!({"text": format!("alpharecord{i}")}))
            .unwrap();
        db.add_vector(&rec.id, &[i as f32, 1.0, 0.0, 0.0]).unwrap();
        ids.push(rec.id);
    }

    // Branch from the live handle (consumes it: flush engines, close, copy).
    let bp = db.branch_create("snap").unwrap();
    assert!(bp.exists(), "branch core file should exist");

    // Reopen the branch with the same engines and verify all three stores agree.
    let branch = Axil::open(&bp)
        .with_vector(DIMS)
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();

    let core_count = branch.list("notes").unwrap().len();
    assert_eq!(core_count, N, "branch core should hold every record");

    // Vector index: every record must be searchable by its own embedding.
    for (i, id) in ids.iter().enumerate() {
        let hits = branch
            .similar_to_vector(&[i as f32, 1.0, 0.0, 0.0], N)
            .unwrap();
        assert!(
            hits.iter().any(|(r, _)| r.id == *id),
            "vector index in branch missing record {id}"
        );
    }

    // FTS index: every record's unique term must resolve in the branch.
    for (i, id) in ids.iter().enumerate() {
        let hits = branch.search_text(&format!("alpharecord{i}"), N).unwrap();
        assert!(
            hits.iter().any(|(r, _)| r.id == *id),
            "FTS index in branch missing record {id}"
        );
    }
}
