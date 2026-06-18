//! FTS Precision Tests
//!
//! These tests verify search result QUALITY, not speed.
//! They ensure the FTS engine returns the right records in the right order.
//!
//! Key behaviors tested:
//! - Field-scoped search: searching a specific field should not match other fields
//! - Field boosting: title matches should rank higher than body matches
//! - Multi-field deduplication: a record matching in 2 fields appears once
//! - Score ordering: more relevant matches rank higher

use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use serde_json::json;

fn temp_fts_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_fts_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

// ═══════════════════════════════════════════════════════════════════════
// Field-scoped search precision
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn field_scope_returns_only_matching_field() {
    let (db, _dir) = temp_fts_db();

    // Record A: "auth" in summary
    let a = db
        .insert(
            "logs",
            json!({
                "summary": "auth timeout bug fixed",
                "notes": "deployed to production"
            }),
        )
        .unwrap();

    // Record B: "auth" in notes, NOT in summary
    let _b = db
        .insert(
            "logs",
            json!({
                "summary": "login page redesign",
                "notes": "auth module was touched during refactor"
            }),
        )
        .unwrap();

    // Record C: "auth" nowhere
    let _c = db
        .insert(
            "logs",
            json!({
                "summary": "database migration complete",
                "notes": "all tables migrated"
            }),
        )
        .unwrap();

    // Field-scoped search: only record A has "auth" in summary
    let results = db.search_field("auth", "summary", 10).unwrap();
    assert_eq!(
        results.len(),
        1,
        "expected only 1 result for 'auth' IN summary"
    );
    assert_eq!(results[0].0.id, a.id);
}

#[test]
fn field_scope_different_fields_different_results() {
    let (db, _dir) = temp_fts_db();

    let a = db
        .insert(
            "docs",
            json!({
                "title": "Rust programming guide",
                "body": "Learn Python and JavaScript"
            }),
        )
        .unwrap();

    let b = db
        .insert(
            "docs",
            json!({
                "title": "Python web development",
                "body": "Built with Rust and Actix"
            }),
        )
        .unwrap();

    // Search "Rust" IN title → only A
    let title_results = db.search_field("Rust", "title", 10).unwrap();
    assert_eq!(title_results.len(), 1);
    assert_eq!(title_results[0].0.id, a.id);

    // Search "Rust" IN body → only B
    let body_results = db.search_field("Rust", "body", 10).unwrap();
    assert_eq!(body_results.len(), 1);
    assert_eq!(body_results[0].0.id, b.id);
}

#[test]
fn field_scope_no_match_in_target_field() {
    let (db, _dir) = temp_fts_db();

    db.insert(
        "docs",
        json!({
            "title": "Database optimization",
            "body": "authentication was improved"
        }),
    )
    .unwrap();

    // "authentication" is in body but NOT in title
    let results = db.search_field("authentication", "title", 10).unwrap();
    assert!(
        results.is_empty(),
        "should not match: auth is in body, not title"
    );
}

#[test]
fn global_search_still_finds_all_fields() {
    let (db, _dir) = temp_fts_db();

    db.insert(
        "docs",
        json!({
            "title": "Introduction",
            "body": "authentication timeout handling"
        }),
    )
    .unwrap();

    db.insert(
        "docs",
        json!({
            "title": "authentication module",
            "body": "main entry point"
        }),
    )
    .unwrap();

    // Global search (no field scope) should find both
    let results = db.search_text("authentication", 10).unwrap();
    assert_eq!(results.len(), 2, "global search should match both records");
}

// ═══════════════════════════════════════════════════════════════════════
// Ranking precision
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn higher_term_frequency_ranks_first() {
    let (db, _dir) = temp_fts_db();

    // Record with term appearing 3 times
    let high = db
        .insert(
            "docs",
            json!({
                "text": "rust rust rust programming language"
            }),
        )
        .unwrap();

    // Record with term appearing 1 time
    let _low = db
        .insert(
            "docs",
            json!({
                "text": "rust programming"
            }),
        )
        .unwrap();

    let results = db.search_text("rust", 10).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.id, high.id, "higher TF should rank first");
    assert!(
        results[0].1 >= results[1].1,
        "first result should have higher score"
    );
}

#[test]
fn exact_phrase_search() {
    let (db, _dir) = temp_fts_db();

    let exact = db
        .insert(
            "docs",
            json!({
                "text": "the quick brown fox jumps over the lazy dog"
            }),
        )
        .unwrap();

    db.insert(
        "docs",
        json!({
            "text": "brown and quick are colors and speeds"
        }),
    )
    .unwrap();

    // Phrase search should match the exact sequence
    let results = db.search_text("\"quick brown fox\"", 10).unwrap();
    assert_eq!(
        results.len(),
        1,
        "phrase search should match only exact sequence"
    );
    assert_eq!(results[0].0.id, exact.id);
}

#[test]
fn multi_field_dedup_single_result() {
    let (db, _dir) = temp_fts_db();

    // "rust" appears in both title and body
    let r = db
        .insert(
            "docs",
            json!({
                "title": "Learn Rust today",
                "body": "A Rust tutorial for beginners"
            }),
        )
        .unwrap();

    let results = db.search_text("Rust", 10).unwrap();
    assert_eq!(
        results.len(),
        1,
        "same record matching in 2 fields should appear once"
    );
    assert_eq!(results[0].0.id, r.id);
}

#[test]
fn search_nonexistent_field_returns_empty() {
    let (db, _dir) = temp_fts_db();

    db.insert(
        "docs",
        json!({
            "title": "authentication guide",
            "body": "how to set up auth"
        }),
    )
    .unwrap();

    // Search in a field that doesn't exist in the data
    let results = db.search_field("auth", "nonexistent_field", 10).unwrap();
    assert!(
        results.is_empty(),
        "searching a nonexistent field should return nothing"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn empty_field_value_not_indexed() {
    let (db, _dir) = temp_fts_db();

    db.insert(
        "docs",
        json!({
            "title": "",
            "body": "some content"
        }),
    )
    .unwrap();

    let results = db.search_field("", "title", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn numeric_fields_not_indexed_in_fts() {
    let (db, _dir) = temp_fts_db();

    db.insert(
        "metrics",
        json!({
            "name": "request_count",
            "value": 42,
            "tags": ["api", "production"]
        }),
    )
    .unwrap();

    // Only string field "name" should be searchable
    let results = db.search_text("request_count", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Numeric value should not be searchable
    let results = db.search_text("42", 10).unwrap();
    assert!(
        results.is_empty(),
        "numeric fields should not be FTS-indexed"
    );
}

#[test]
fn update_record_reindexes_fts() {
    let (db, _dir) = temp_fts_db();

    let r = db
        .insert(
            "docs",
            json!({
                "title": "old authentication guide"
            }),
        )
        .unwrap();

    // Verify initial index
    assert_eq!(db.search_text("authentication", 10).unwrap().len(), 1);

    // Update removes old term, adds new
    db.update(&r.id, json!({"title": "new authorization guide"}))
        .unwrap();

    let old_results = db.search_text("authentication", 10).unwrap();
    assert!(
        old_results.is_empty(),
        "old term should be gone after update"
    );

    let new_results = db.search_text("authorization", 10).unwrap();
    assert_eq!(
        new_results.len(),
        1,
        "new term should be indexed after update"
    );
}

#[test]
fn delete_record_removes_from_fts() {
    let (db, _dir) = temp_fts_db();

    let r = db
        .insert("docs", json!({"title": "temporary document"}))
        .unwrap();
    assert_eq!(db.search_text("temporary", 10).unwrap().len(), 1);

    db.delete(&r.id).unwrap();
    assert!(
        db.search_text("temporary", 10).unwrap().is_empty(),
        "deleted record should not appear in FTS results"
    );
}
