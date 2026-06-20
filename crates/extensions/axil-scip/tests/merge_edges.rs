//! `Axil::merge_entities` must move edges, not copy them — after the
//! merge, the source record must have no edges left pointing at or from
//! it, so traversals starting on the source's old neighbors cannot reach
//! the tombstoned row.

use axil_core::{Axil, Direction};
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

#[test]
fn merge_moves_edges_and_removes_originals() {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("t.axil"))
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();

    // Two entity rows with distinct canonical ids, and one mention edge
    // per entity from an outside record so we exercise both Direction::In
    // and Direction::Out during the merge.
    let from = db
        .insert(
            "_entities",
            json!({ "name": "login", "canonical_id": "provisional:abc" }),
        )
        .unwrap();
    let to = db
        .insert(
            "_entities",
            json!({ "name": "login", "canonical_id": "grounded:def" }),
        )
        .unwrap();
    let note = db.insert("decisions", json!({ "summary": "x" })).unwrap();

    let gi = db.graph_index_ref().unwrap();
    gi.relate(note.id.clone(), "mentions", from.id.clone(), json!({}))
        .unwrap();
    gi.relate(from.id.clone(), "defined_in", to.id.clone(), json!({}))
        .unwrap();

    let moved = db
        .merge_entities("provisional:abc", "grounded:def")
        .unwrap();
    assert!(moved >= 1, "merge reported 0 moved edges");

    // No edges should remain on the source.
    let remaining_in = gi.edges(from.id.clone(), None, Direction::In).unwrap();
    let remaining_out = gi.edges(from.id.clone(), None, Direction::Out).unwrap();
    assert!(
        remaining_in.is_empty(),
        "source has {} incoming edges after merge",
        remaining_in.len()
    );
    assert!(
        remaining_out.is_empty(),
        "source has {} outgoing edges after merge",
        remaining_out.len()
    );

    // Target must carry the re-homed incoming edge.
    let to_in = gi
        .edges(to.id.clone(), Some("mentions"), Direction::In)
        .unwrap();
    assert_eq!(to_in.len(), 1, "target missing re-homed mention edge");
    assert_eq!(to_in[0].from, note.id);
}
