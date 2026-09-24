//! `traverse(..).order_by(..).limit(n)` returns the true top-n endpoints.
//!
//! Traversal used to stop collecting endpoints at `offset + limit` *before*
//! the sort ran, so an ordered query returned a sorted handful of whichever
//! endpoints happened to be reached first. Each fixture here reaches the true
//! top-n last.

use axil_core::{Axil, Record, SortDirection};
use axil_graph::AxilBuilderGraphExt;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

fn open(dir: &tempfile::TempDir) -> Axil {
    Axil::open(dir.path().join("test.axil"))
        .with_vector(3)
        .unwrap()
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap()
}

/// A hub session linked to ten files. Files are created and linked in
/// ascending `rank` / `created_at` order, so the highest-ranked and newest
/// files are the last endpoints any traversal reaches.
fn hub_with_files(db: &Axil) -> Record {
    let hub = db.insert("sessions", json!({"summary": "hub"})).unwrap();
    let base = chrono::Utc::now() - chrono::Duration::days(1);
    for rank in 0..10i64 {
        let file = db
            .insert_at(
                "files",
                json!({"path": format!("f{rank}.rs"), "rank": rank}),
                base + chrono::Duration::minutes(rank),
            )
            .unwrap();
        db.relate(&hub.id, "touched", &file.id, None).unwrap();
    }
    hub
}

fn ranks(records: &[Record]) -> Vec<i64> {
    records
        .iter()
        .map(|r| r.data["rank"].as_i64().unwrap())
        .collect()
}

#[test]
fn table_traversal_order_by_field_returns_true_top_n() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    hub_with_files(&db);

    let top = db
        .query()
        .table("sessions")
        .traverse("->touched")
        .order_by("rank", SortDirection::Desc)
        .limit(3)
        .exec()
        .unwrap();
    assert_eq!(ranks(&top), [9, 8, 7]);

    // Offset pages through the true order too.
    let page = db
        .query()
        .table("sessions")
        .traverse("->touched")
        .order_by("rank", SortDirection::Desc)
        .offset(3)
        .limit(2)
        .exec()
        .unwrap();
    assert_eq!(ranks(&page), [6, 5]);
}

#[test]
fn table_traversal_order_by_time_returns_true_top_n() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    hub_with_files(&db);

    let newest = db
        .query()
        .table("sessions")
        .traverse("->touched")
        .order_by_time(SortDirection::Desc)
        .limit(3)
        .exec()
        .unwrap();
    assert_eq!(ranks(&newest), [9, 8, 7]);
}

#[test]
fn vector_seeded_traversal_order_by_returns_true_top_n() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let hub = hub_with_files(&db);
    db.add_vector(&hub.id, &[1.0, 0.0, 0.0]).unwrap();

    let top = db
        .query()
        .similar_to_vector(vec![1.0, 0.0, 0.0], 1)
        .traverse("->touched")
        .order_by("rank", SortDirection::Desc)
        .limit(3)
        .exec()
        .unwrap();
    assert_eq!(ranks(&top), [9, 8, 7]);
}

#[test]
fn unordered_traversal_still_honors_limit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    hub_with_files(&db);

    let some = db
        .query()
        .table("sessions")
        .traverse("->touched")
        .limit(4)
        .exec()
        .unwrap();
    assert_eq!(some.len(), 4);
}
