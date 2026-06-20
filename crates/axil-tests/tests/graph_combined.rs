//! Integration tests for combined graph + query builder operations,
//! and graph + vector combined queries.

use axil_core::{Axil, Op};
use axil_graph::AxilBuilderGraphExt;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

// ── Helpers ────────────────────────────────────────────────────────

fn temp_graph_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

fn temp_vector_graph_db(dims: usize) -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_vector(dims)
        .unwrap()
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

// ── QueryBuilder standalone graph traversal ────────────────────────

#[test]
fn query_traverse_from_table() {
    let (db, _dir) = temp_graph_db();
    let session = db
        .insert("sessions", json!({"summary": "auth fix"}))
        .unwrap();
    let file = db.insert("files", json!({"path": "auth.rs"})).unwrap();
    db.relate(&session.id, "modified", &file.id, None).unwrap();

    // Use QueryBuilder to start from sessions table, then traverse.
    let results = db
        .query()
        .table("sessions")
        .traverse("->modified")
        .exec()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "auth.rs");
}

#[test]
fn query_traverse_with_where_filter() {
    let (db, _dir) = temp_graph_db();
    let s1 = db
        .insert("sessions", json!({"summary": "auth fix", "project": "app"}))
        .unwrap();
    let s2 = db
        .insert(
            "sessions",
            json!({"summary": "db refactor", "project": "lib"}),
        )
        .unwrap();
    let f1 = db.insert("files", json!({"path": "auth.rs"})).unwrap();
    let f2 = db.insert("files", json!({"path": "db.rs"})).unwrap();

    db.relate(&s1.id, "modified", &f1.id, None).unwrap();
    db.relate(&s2.id, "modified", &f2.id, None).unwrap();

    // Only traverse from sessions matching the where filter.
    let results = db
        .query()
        .table("sessions")
        .where_field("project", Op::Eq, json!("app"))
        .traverse("->modified")
        .exec()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "auth.rs");
}

#[test]
fn query_traverse_multi_hop() {
    let (db, _dir) = temp_graph_db();
    let session = db
        .insert("sessions", json!({"summary": "refactor"}))
        .unwrap();
    let commit = db.insert("commits", json!({"sha": "abc"})).unwrap();
    let file = db.insert("files", json!({"path": "main.rs"})).unwrap();

    db.relate(&session.id, "created", &commit.id, None).unwrap();
    db.relate(&commit.id, "modified", &file.id, None).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->created->modified")
        .exec()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "main.rs");
}

#[test]
fn query_traverse_deduplicates_results() {
    let (db, _dir) = temp_graph_db();
    let s1 = db.insert("sessions", json!({"summary": "fix A"})).unwrap();
    let s2 = db.insert("sessions", json!({"summary": "fix B"})).unwrap();
    let file = db.insert("files", json!({"path": "shared.rs"})).unwrap();

    // Both sessions modified the same file.
    db.relate(&s1.id, "modified", &file.id, None).unwrap();
    db.relate(&s2.id, "modified", &file.id, None).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->modified")
        .exec()
        .unwrap();

    // Should return the file only once.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "shared.rs");
}

#[test]
fn query_traverse_empty_result() {
    let (db, _dir) = temp_graph_db();
    db.insert("sessions", json!({"summary": "lonely"})).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->nonexistent")
        .exec()
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn query_traverse_requires_table() {
    let (db, _dir) = temp_graph_db();
    db.insert("sessions", json!({"summary": "test"})).unwrap();

    // traverse() without table() should error.
    let result = db.query().traverse("->modified").exec();
    assert!(result.is_err());
}

#[test]
fn query_traverse_without_graph_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    db.insert("sessions", json!({"summary": "test"})).unwrap();

    let result = db.query().table("sessions").traverse("->modified").exec();
    assert!(result.is_err());
}

// ── QueryBuilder traversal with pagination ────────────────────────

#[test]
fn query_traverse_respects_limit() {
    let (db, _dir) = temp_graph_db();
    // 3 sessions each modified a distinct file.
    let s1 = db.insert("sessions", json!({"summary": "A"})).unwrap();
    let s2 = db.insert("sessions", json!({"summary": "B"})).unwrap();
    let s3 = db.insert("sessions", json!({"summary": "C"})).unwrap();
    let f1 = db.insert("files", json!({"path": "a.rs"})).unwrap();
    let f2 = db.insert("files", json!({"path": "b.rs"})).unwrap();
    let f3 = db.insert("files", json!({"path": "c.rs"})).unwrap();

    db.relate(&s1.id, "modified", &f1.id, None).unwrap();
    db.relate(&s2.id, "modified", &f2.id, None).unwrap();
    db.relate(&s3.id, "modified", &f3.id, None).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->modified")
        .limit(2)
        .exec()
        .unwrap();

    assert_eq!(results.len(), 2);
}

#[test]
fn query_traverse_respects_offset() {
    let (db, _dir) = temp_graph_db();
    let s1 = db.insert("sessions", json!({"summary": "A"})).unwrap();
    let s2 = db.insert("sessions", json!({"summary": "B"})).unwrap();
    let s3 = db.insert("sessions", json!({"summary": "C"})).unwrap();
    let f1 = db.insert("files", json!({"path": "a.rs"})).unwrap();
    let f2 = db.insert("files", json!({"path": "b.rs"})).unwrap();
    let f3 = db.insert("files", json!({"path": "c.rs"})).unwrap();

    db.relate(&s1.id, "modified", &f1.id, None).unwrap();
    db.relate(&s2.id, "modified", &f2.id, None).unwrap();
    db.relate(&s3.id, "modified", &f3.id, None).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->modified")
        .offset(1)
        .limit(1)
        .exec()
        .unwrap();

    // offset(1) skips 1 endpoint, limit(1) takes 1 — must return exactly 1.
    assert_eq!(results.len(), 1);
}

#[test]
fn query_traverse_order_by_sorts_endpoints() {
    let (db, _dir) = temp_graph_db();
    let session = db
        .insert("sessions", json!({"summary": "refactor"}))
        .unwrap();
    let f1 = db
        .insert("files", json!({"path": "a.rs", "priority": 3}))
        .unwrap();
    let f2 = db
        .insert("files", json!({"path": "b.rs", "priority": 1}))
        .unwrap();
    let f3 = db
        .insert("files", json!({"path": "c.rs", "priority": 2}))
        .unwrap();

    db.relate(&session.id, "modified", &f1.id, None).unwrap();
    db.relate(&session.id, "modified", &f2.id, None).unwrap();
    db.relate(&session.id, "modified", &f3.id, None).unwrap();

    let results = db
        .query()
        .table("sessions")
        .traverse("->modified")
        .order_by("priority", axil_core::SortDirection::Asc)
        .exec()
        .unwrap();

    assert_eq!(results.len(), 3);
    let priorities: Vec<i64> = results
        .iter()
        .map(|r| r.data["priority"].as_i64().unwrap())
        .collect();
    assert_eq!(priorities, vec![1, 2, 3]);
}

// ── Vector + graph combined queries ────────────────────────────────

#[test]
fn vector_then_traverse() {
    let dims = 3;
    let (db, _dir) = temp_vector_graph_db(dims);

    // Insert records with vectors.
    let session = db
        .insert("sessions", json!({"summary": "auth bug"}))
        .unwrap();
    let file = db.insert("files", json!({"path": "auth.rs"})).unwrap();
    let _unrelated = db
        .insert("sessions", json!({"summary": "unrelated"}))
        .unwrap();

    // Add vectors.
    db.add_vector(&session.id, &[1.0, 0.0, 0.0]).unwrap();
    db.add_vector(&_unrelated.id, &[0.0, 1.0, 0.0]).unwrap();

    // Create graph edge.
    db.relate(&session.id, "modified", &file.id, None).unwrap();

    // Combined: find similar to [1, 0, 0], then traverse ->modified.
    let results = db
        .query()
        .similar_to_vector(vec![1.0, 0.0, 0.0], 1)
        .traverse("->modified")
        .exec()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "auth.rs");
}

#[test]
fn vector_then_traverse_no_edges() {
    let dims = 3;
    let (db, _dir) = temp_vector_graph_db(dims);

    let session = db
        .insert("sessions", json!({"summary": "no edges"}))
        .unwrap();
    db.add_vector(&session.id, &[1.0, 0.0, 0.0]).unwrap();

    // Vector search finds the record, but it has no outgoing edges.
    let results = db
        .query()
        .similar_to_vector(vec![1.0, 0.0, 0.0], 1)
        .traverse("->modified")
        .exec()
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn vector_then_traverse_multi_start() {
    let dims = 3;
    let (db, _dir) = temp_vector_graph_db(dims);

    let s1 = db.insert("sessions", json!({"summary": "fix A"})).unwrap();
    let s2 = db.insert("sessions", json!({"summary": "fix B"})).unwrap();
    let f1 = db.insert("files", json!({"path": "a.rs"})).unwrap();
    let f2 = db.insert("files", json!({"path": "b.rs"})).unwrap();

    db.add_vector(&s1.id, &[1.0, 0.0, 0.0]).unwrap();
    db.add_vector(&s2.id, &[0.9, 0.1, 0.0]).unwrap();

    db.relate(&s1.id, "modified", &f1.id, None).unwrap();
    db.relate(&s2.id, "modified", &f2.id, None).unwrap();

    // Both sessions are similar to query; traversal fans out from both.
    let results = db
        .query()
        .similar_to_vector(vec![1.0, 0.0, 0.0], 2)
        .traverse("->modified")
        .exec()
        .unwrap();

    assert_eq!(results.len(), 2);
    let paths: Vec<&str> = results
        .iter()
        .map(|r| r.data["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"a.rs"));
    assert!(paths.contains(&"b.rs"));
}

#[test]
fn vector_then_traverse_with_offset() {
    let dims = 3;
    let (db, _dir) = temp_vector_graph_db(dims);

    let s1 = db.insert("sessions", json!({"summary": "fix A"})).unwrap();
    let s2 = db.insert("sessions", json!({"summary": "fix B"})).unwrap();
    let f1 = db.insert("files", json!({"path": "a.rs"})).unwrap();
    let f2 = db.insert("files", json!({"path": "b.rs"})).unwrap();

    db.add_vector(&s1.id, &[1.0, 0.0, 0.0]).unwrap();
    db.add_vector(&s2.id, &[0.9, 0.1, 0.0]).unwrap();

    db.relate(&s1.id, "modified", &f1.id, None).unwrap();
    db.relate(&s2.id, "modified", &f2.id, None).unwrap();

    // Both sessions match. Traverse fans out to 2 files.
    // offset(1) should skip 1, limit(1) should take 1 — must return 1.
    let results = db
        .query()
        .similar_to_vector(vec![1.0, 0.0, 0.0], 2)
        .traverse("->modified")
        .offset(1)
        .limit(1)
        .exec()
        .unwrap();

    assert_eq!(results.len(), 1);
}

// ── Graph info in database stats ───────────────────────────────────

#[test]
fn info_includes_graph_plugin() {
    let (db, _dir) = temp_graph_db();
    let info = db.info().unwrap();
    assert!(info.plugins.contains_key("graph"));
}

#[test]
fn graph_files_detected() {
    let (db, _dir) = temp_graph_db();
    let files = db.files();
    // Should have core (.axil) + graph (.axil.graph).
    assert!(files.len() >= 2);
    assert!(files
        .iter()
        .any(|p| p.extension().map_or(false, |e| e == "graph")));
}
