use axil_core::{Axil, Direction};
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

fn temp_fts_graph_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_fts_plugin()
        .unwrap()
        .with_graph_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn fts_with_graph_traversal() {
    let (db, _dir) = temp_fts_graph_db();

    let session = db
        .insert(
            "sessions",
            json!({"summary": "Fixed authentication timeout"}),
        )
        .unwrap();
    let file = db
        .insert(
            "files",
            json!({"path": "auth.rs", "description": "Auth module"}),
        )
        .unwrap();

    db.relate(&session.id, "modified", &file.id, None).unwrap();

    // FTS finds the session, traversal follows to the file.
    let results = db
        .query()
        .search_text("authentication")
        .traverse("->modified")
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "auth.rs");
}

#[test]
fn fts_query_with_where_filter() {
    let (db, _dir) = temp_fts_graph_db();

    db.insert(
        "sessions",
        json!({"summary": "Auth bug fix", "priority": "high"}),
    )
    .unwrap();
    db.insert(
        "sessions",
        json!({"summary": "Auth refactor", "priority": "low"}),
    )
    .unwrap();

    let results = db
        .query()
        .search_text("auth")
        .where_field("priority", axil_core::Op::Eq, json!("high"))
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["priority"], "high");
}

#[test]
fn fts_with_graph_query_builder() {
    let (db, _dir) = temp_fts_graph_db();

    let session = db
        .insert("sessions", json!({"summary": "Database optimization work"}))
        .unwrap();
    let entity1 = db
        .insert("entities", json!({"name": "database", "type": "concept"}))
        .unwrap();
    let entity2 = db
        .insert("entities", json!({"name": "indexing", "type": "concept"}))
        .unwrap();

    db.relate(&session.id, "mentions", &entity1.id, None)
        .unwrap();
    db.relate(&session.id, "mentions", &entity2.id, None)
        .unwrap();

    // Search via FTS, then traverse to find mentioned entities.
    let results = db
        .query()
        .search_text("optimization")
        .traverse("->mentions")
        .exec()
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn fts_standalone_and_graph_independent() {
    let (db, _dir) = temp_fts_graph_db();

    let a = db
        .insert("nodes", json!({"text": "Alpha node content"}))
        .unwrap();
    let b = db
        .insert("nodes", json!({"text": "Beta node content"}))
        .unwrap();
    db.relate(&a.id, "links", &b.id, None).unwrap();

    // FTS standalone.
    let fts_results = db.search_text("Alpha", 10).unwrap();
    assert_eq!(fts_results.len(), 1);
    assert_eq!(fts_results[0].0.id, a.id);

    // Graph standalone.
    let neighbors = db.neighbors(&a.id, None, Direction::Out).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, b.id);
}
