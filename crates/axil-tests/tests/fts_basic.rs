use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use serde_json::json;

fn temp_fts_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn insert_auto_indexes_text_fields() {
    let (db, _dir) = temp_fts_db();
    db.insert(
        "sessions",
        json!({"summary": "Fixed the authentication timeout bug", "project": "my-app"}),
    )
    .unwrap();

    let results = db.search_text("authentication", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].0.data["summary"],
        "Fixed the authentication timeout bug"
    );
    assert!(results[0].1 > 0.0);
}

#[test]
fn search_returns_ranked_results() {
    let (db, _dir) = temp_fts_db();

    db.insert("docs", json!({"title": "Rust programming guide"}))
        .unwrap();
    db.insert("docs", json!({"title": "Python web development"}))
        .unwrap();
    db.insert(
        "docs",
        json!({"title": "Advanced Rust patterns and performance"}),
    )
    .unwrap();

    let results = db.search_text("Rust", 10).unwrap();
    assert_eq!(results.len(), 2);
    // Both should have positive scores.
    assert!(results[0].1 > 0.0);
    assert!(results[1].1 > 0.0);
}

#[test]
fn manual_index_text() {
    let (db, _dir) = temp_fts_db();
    let record = db.insert("notes", json!({"count": 42})).unwrap();

    // Manually index text not in the record.
    db.index_text(
        &record.id,
        "description",
        "manually indexed content about databases",
    )
    .unwrap();

    let results = db.search_text("databases", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, record.id);
}

#[test]
fn delete_removes_from_fts_index() {
    let (db, _dir) = temp_fts_db();
    let record = db
        .insert("sessions", json!({"summary": "Important session data"}))
        .unwrap();

    assert_eq!(db.search_text("important", 10).unwrap().len(), 1);

    db.delete(&record.id).unwrap();
    assert!(db.search_text("important", 10).unwrap().is_empty());
}

#[test]
fn search_empty_query() {
    let (db, _dir) = temp_fts_db();
    db.insert("sessions", json!({"summary": "test data"}))
        .unwrap();

    let results = db.search_text("", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_no_matches() {
    let (db, _dir) = temp_fts_db();
    db.insert("sessions", json!({"summary": "hello world"}))
        .unwrap();

    let results = db.search_text("nonexistent", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn fts_query_builder() {
    let (db, _dir) = temp_fts_db();
    db.insert("sessions", json!({"summary": "Fixed auth bug in login"}))
        .unwrap();
    db.insert(
        "sessions",
        json!({"summary": "Added new dashboard feature"}),
    )
    .unwrap();
    db.insert("notes", json!({"summary": "Auth design document"}))
        .unwrap();

    // FTS through query builder.
    let results = db.query().search_text("auth").exec().unwrap();
    assert_eq!(results.len(), 2);

    // FTS + table filter.
    let results = db
        .query()
        .search_text("auth")
        .table("sessions")
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].table, "sessions");
}

#[test]
fn fts_with_limit() {
    let (db, _dir) = temp_fts_db();
    for i in 0..10 {
        db.insert(
            "docs",
            json!({"title": format!("Document about testing number {i}")}),
        )
        .unwrap();
    }

    let results = db.search_text("testing", 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn persistence_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");

    // First session: insert and index.
    {
        let db = Axil::open(&path)
            .with_fts_engine()
            .unwrap()
            .build()
            .unwrap();
        db.insert("sessions", json!({"summary": "Persistent search data"}))
            .unwrap();
    }

    // Second session: reopen and search.
    {
        let db = Axil::open(&path)
            .with_fts_engine()
            .unwrap()
            .build()
            .unwrap();
        let results = db.search_text("persistent", 10).unwrap();
        assert_eq!(results.len(), 1);
    }
}

#[test]
fn no_fts_index_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();

    let record = db.insert("notes", json!({"text": "hello"})).unwrap();
    assert!(db.index_text(&record.id, "text", "hello").is_err());
    assert!(db.search_text("hello", 10).is_err());
}

#[test]
fn update_refreshes_fts_index() {
    let (db, _dir) = temp_fts_db();
    let record = db
        .insert(
            "sessions",
            json!({"summary": "Original authentication bug"}),
        )
        .unwrap();

    assert_eq!(db.search_text("authentication", 10).unwrap().len(), 1);

    // Update the record's text.
    db.update(&record.id, json!({"summary": "Refactored database layer"}))
        .unwrap();

    // Old content should no longer be searchable.
    assert!(db.search_text("authentication", 10).unwrap().is_empty());
    // New content should be searchable.
    assert_eq!(db.search_text("database", 10).unwrap().len(), 1);
}

#[test]
fn re_index_text_replaces_old_content() {
    let (db, _dir) = temp_fts_db();
    let record = db.insert("notes", json!({"count": 1})).unwrap();

    // Manually index some text.
    db.index_text(&record.id, "description", "original content about rust")
        .unwrap();
    assert_eq!(db.search_text("rust", 10).unwrap().len(), 1);

    // Re-index with different text — old content should be replaced.
    db.index_text(&record.id, "description", "updated content about python")
        .unwrap();
    assert!(db.search_text("rust", 10).unwrap().is_empty());
    assert_eq!(db.search_text("python", 10).unwrap().len(), 1);
}

#[test]
fn index_text_multiple_fields_preserves_all() {
    let (db, _dir) = temp_fts_db();
    let record = db.insert("notes", json!({"count": 1})).unwrap();

    // Index two different fields explicitly.
    db.index_text(&record.id, "title", "Rust programming guide")
        .unwrap();
    db.index_text(&record.id, "body", "Systems programming with memory safety")
        .unwrap();

    // Both fields should be searchable.
    let results = db.search_text("Rust", 10).unwrap();
    assert_eq!(results.len(), 1);
    let results = db.search_text("memory safety", 10).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn index_text_after_auto_index_preserves_auto_fields() {
    let (db, _dir) = temp_fts_db();
    let record = db
        .insert("sessions", json!({"summary": "Fixed authentication bug"}))
        .unwrap();

    // Auto-indexed on insert — verify.
    assert_eq!(db.search_text("authentication", 10).unwrap().len(), 1);

    // Supplement with an explicit field.
    db.index_text(&record.id, "notes", "Extra context about login")
        .unwrap();

    // Both auto-indexed and explicit content should be searchable.
    assert_eq!(db.search_text("authentication", 10).unwrap().len(), 1);
    assert_eq!(db.search_text("login", 10).unwrap().len(), 1);
}

#[test]
fn search_zero_limit_returns_empty() {
    let (db, _dir) = temp_fts_db();
    db.insert("sessions", json!({"summary": "test data"}))
        .unwrap();

    let results = db.search_text("test", 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_rejects_overly_long_query() {
    let (db, _dir) = temp_fts_db();
    let long_query = "a".repeat(600);
    let result = db.search_text(&long_query, 10);
    assert!(result.is_err());
}

#[test]
fn search_rejects_internal_field_targeting() {
    let (db, _dir) = temp_fts_db();
    db.insert("sessions", json!({"summary": "test"})).unwrap();

    assert!(db.search_text("id:some_value", 10).is_err());
    assert!(db.search_text("doc_key:some_value", 10).is_err());
}

#[test]
fn numeric_fields_not_auto_indexed() {
    let (db, _dir) = temp_fts_db();
    db.insert("data", json!({"count": 42, "name": "test"}))
        .unwrap();

    // Numeric fields should not be indexed.
    let results = db.search_text("42", 10).unwrap();
    assert!(results.is_empty());

    // String fields should be indexed.
    let results = db.search_text("test", 10).unwrap();
    assert_eq!(results.len(), 1);
}
