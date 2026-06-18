use axil_core::{Axil, Direction, RecordId};
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

fn temp_graph_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn relate_and_neighbors() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("people", json!({"name": "Alice"})).unwrap();
    let b = db.insert("people", json!({"name": "Bob"})).unwrap();

    let edge_id = db.relate(&a.id, "knows", &b.id, None).unwrap();
    assert!(!edge_id.as_str().is_empty());

    let neighbors = db.neighbors(&a.id, Some("knows"), Direction::Out).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, b.id);
    assert_eq!(neighbors[0].data["name"], "Bob");
}

#[test]
fn relate_nonexistent_record_fails() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("people", json!({"name": "Alice"})).unwrap();
    let fake_id = RecordId::new();

    let result = db.relate(&a.id, "knows", &fake_id, None);
    assert!(result.is_err());

    let result = db.relate(&fake_id, "knows", &a.id, None);
    assert!(result.is_err());
}

#[test]
fn unrelate() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("people", json!({"name": "Alice"})).unwrap();
    let b = db.insert("people", json!({"name": "Bob"})).unwrap();

    let edge_id = db.relate(&a.id, "knows", &b.id, None).unwrap();
    assert!(db.unrelate(&edge_id).unwrap());
    assert!(!db.unrelate(&edge_id).unwrap()); // already deleted

    let neighbors = db.neighbors(&a.id, Some("knows"), Direction::Out).unwrap();
    assert!(neighbors.is_empty());
}

#[test]
fn traverse_single_hop() {
    let (db, _dir) = temp_graph_db();
    let session = db
        .insert("sessions", json!({"summary": "fixed bug"}))
        .unwrap();
    let file = db.insert("files", json!({"path": "auth.rs"})).unwrap();

    db.relate(&session.id, "modified", &file.id, None).unwrap();

    let results = db.traverse(&session.id, "->modified").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "auth.rs");
}

#[test]
fn traverse_multi_hop() {
    let (db, _dir) = temp_graph_db();
    let session = db
        .insert("sessions", json!({"summary": "refactor"}))
        .unwrap();
    let commit = db.insert("commits", json!({"sha": "abc123"})).unwrap();
    let file = db.insert("files", json!({"path": "main.rs"})).unwrap();

    db.relate(&session.id, "created", &commit.id, None).unwrap();
    db.relate(&commit.id, "modified", &file.id, None).unwrap();

    let results = db.traverse(&session.id, "->created->modified").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["path"], "main.rs");
}

#[test]
fn traverse_incoming() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();

    db.relate(&a.id, "points_to", &b.id, None).unwrap();

    let results = db.traverse(&b.id, "<-points_to").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["name"], "A");
}

#[test]
fn traverse_empty_result() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();

    let results = db.traverse(&a.id, "->nonexistent").unwrap();
    assert!(results.is_empty());
}

#[test]
fn traverse_cycle_returns_start() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();

    db.relate(&a.id, "links", &b.id, None).unwrap();
    db.relate(&b.id, "links", &a.id, None).unwrap();

    // Two hops: a -> b -> a. The start node is a valid traversal result.
    let results = db.traverse(&a.id, "->links->links").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["name"], "A");
}

#[test]
fn traverse_cycle_oscillates() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();

    db.relate(&a.id, "links", &b.id, None).unwrap();
    db.relate(&b.id, "links", &a.id, None).unwrap();

    // Three hops: a->b->a->b. Nodes can reappear across steps.
    let results = db.traverse(&a.id, "->links->links->links").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["name"], "B");
}

#[test]
fn cascade_delete_removes_edges() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();
    let c = db.insert("nodes", json!({"name": "C"})).unwrap();

    db.relate(&a.id, "knows", &b.id, None).unwrap();
    db.relate(&c.id, "knows", &a.id, None).unwrap();

    // Delete A — edges should be cleaned up.
    db.delete(&a.id).unwrap();

    // B should have no incoming edges from A.
    let neighbors = db.neighbors(&b.id, None, Direction::In).unwrap();
    assert!(neighbors.is_empty());

    // C should have no outgoing edges to A.
    let neighbors = db.neighbors(&c.id, None, Direction::Out).unwrap();
    assert!(neighbors.is_empty());
}

#[test]
fn neighbors_direction_filter() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();
    let c = db.insert("nodes", json!({"name": "C"})).unwrap();

    db.relate(&a.id, "knows", &b.id, None).unwrap();
    db.relate(&c.id, "follows", &a.id, None).unwrap();

    let out = db.neighbors(&a.id, None, Direction::Out).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].data["name"], "B");

    let inc = db.neighbors(&a.id, None, Direction::In).unwrap();
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].data["name"], "C");

    let both = db.neighbors(&a.id, None, Direction::Both).unwrap();
    assert_eq!(both.len(), 2);
}

#[test]
fn neighbors_type_filter() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();
    let c = db.insert("nodes", json!({"name": "C"})).unwrap();

    db.relate(&a.id, "knows", &b.id, None).unwrap();
    db.relate(&a.id, "follows", &c.id, None).unwrap();

    let knows = db.neighbors(&a.id, Some("knows"), Direction::Out).unwrap();
    assert_eq!(knows.len(), 1);
    assert_eq!(knows[0].data["name"], "B");

    let follows = db
        .neighbors(&a.id, Some("follows"), Direction::Out)
        .unwrap();
    assert_eq!(follows.len(), 1);
    assert_eq!(follows[0].data["name"], "C");
}

#[test]
fn persistence_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");

    let a_id;
    let b_id;

    // First session: create records and edge.
    {
        let db = Axil::open(&path)
            .with_graph_plugin()
            .unwrap()
            .build()
            .unwrap();

        let a = db.insert("nodes", json!({"name": "A"})).unwrap();
        let b = db.insert("nodes", json!({"name": "B"})).unwrap();
        a_id = a.id.clone();
        b_id = b.id.clone();

        db.relate(&a.id, "knows", &b.id, None).unwrap();
    }

    // Second session: reopen and verify.
    {
        let db = Axil::open(&path)
            .with_graph_plugin()
            .unwrap()
            .build()
            .unwrap();

        let neighbors = db.neighbors(&a_id, Some("knows"), Direction::Out).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, b_id);
        assert_eq!(neighbors[0].data["name"], "B");
    }
}

#[test]
fn relate_with_properties_and_edges_api() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();

    db.relate(
        &a.id,
        "knows",
        &b.id,
        Some(json!({"since": "2026-01-01", "weight": 0.9})),
    )
    .unwrap();

    // Verify via edges() API that properties and metadata survive.
    let edges = db.edges(&a.id, Some("knows"), Direction::Out).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].from, a.id);
    assert_eq!(edges[0].to, b.id);
    assert_eq!(edges[0].edge_type, "knows");
    assert_eq!(edges[0].properties["since"], "2026-01-01");
    assert_eq!(edges[0].properties["weight"], 0.9);
    assert!(!edges[0].created_at.is_empty());
}

#[test]
fn unrelate_nonexistent_edge() {
    let (db, _dir) = temp_graph_db();
    let fake = RecordId::new();
    assert_eq!(db.unrelate(&fake).unwrap(), false);
}

#[test]
fn self_loop_edge() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    db.relate(&a.id, "self_ref", &a.id, None).unwrap();

    let neighbors = db
        .neighbors(&a.id, Some("self_ref"), Direction::Out)
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, a.id);

    // Delete should clean up the self-loop edge.
    db.delete(&a.id).unwrap();
}

#[test]
fn edges_both_directions() {
    let (db, _dir) = temp_graph_db();
    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();
    let c = db.insert("nodes", json!({"name": "C"})).unwrap();

    db.relate(&a.id, "knows", &b.id, None).unwrap();
    db.relate(&c.id, "follows", &a.id, None).unwrap();

    let all = db.edges(&a.id, None, Direction::Both).unwrap();
    assert_eq!(all.len(), 2);

    let out_only = db.edges(&a.id, None, Direction::Out).unwrap();
    assert_eq!(out_only.len(), 1);
    assert_eq!(out_only[0].edge_type, "knows");

    let in_only = db.edges(&a.id, None, Direction::In).unwrap();
    assert_eq!(in_only.len(), 1);
    assert_eq!(in_only[0].edge_type, "follows");
}

#[test]
fn no_graph_index_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();

    let a = db.insert("nodes", json!({"name": "A"})).unwrap();
    let b = db.insert("nodes", json!({"name": "B"})).unwrap();

    assert!(db.relate(&a.id, "knows", &b.id, None).is_err());
    assert!(db.neighbors(&a.id, None, Direction::Out).is_err());
    assert!(db.traverse(&a.id, "->knows").is_err());
}
