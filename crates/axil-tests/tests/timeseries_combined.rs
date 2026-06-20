use axil_core::Axil;
use axil_graph::AxilBuilderGraphExt;
use axil_timeseries::AxilBuilderTimeSeriesExt;
use serde_json::json;

fn temp_graph_ts_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .with_timeseries_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn graph_and_timeseries_together() {
    let (db, _dir) = temp_graph_ts_db();

    let r1 = db
        .insert("sessions", json!({"summary": "session 1"}))
        .unwrap();
    let r2 = db
        .insert("sessions", json!({"summary": "session 2"}))
        .unwrap();

    // Create graph edge.
    db.relate(&r1.id, "followed_by", &r2.id, None).unwrap();

    // Both should be in timeseries.
    let recent = db.since(None, 60).unwrap();
    assert_eq!(recent.len(), 2);

    // Graph traversal should still work.
    let neighbors = db
        .neighbors(&r1.id, Some("followed_by"), axil_core::Direction::Out)
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, r2.id);

    // Timeline should show both.
    let timeline = db.timeline(None, 10).unwrap();
    assert_eq!(timeline.len(), 2);
}

#[test]
fn delete_cascades_to_both_plugins() {
    let (db, _dir) = temp_graph_ts_db();

    let r1 = db.insert("sessions", json!({"data": "A"})).unwrap();
    let r2 = db.insert("sessions", json!({"data": "B"})).unwrap();
    db.relate(&r1.id, "knows", &r2.id, None).unwrap();

    // Verify both are in the time index.
    assert_eq!(db.since(None, 60).unwrap().len(), 2);

    // Delete r1 — should cascade to graph edges AND time index.
    db.delete(&r1.id).unwrap();

    // Only r2 should remain in time index.
    let remaining = db.since(None, 60).unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, r2.id);

    // Graph edge should be gone too.
    let edges = db.edges(&r2.id, None, axil_core::Direction::Both).unwrap();
    assert!(edges.is_empty());
}

#[test]
fn query_builder_time_filter_with_table() {
    let (db, _dir) = temp_graph_ts_db();

    db.insert("sessions", json!({"type": "session"})).unwrap();
    db.insert("decisions", json!({"type": "decision"})).unwrap();
    db.insert("sessions", json!({"type": "session2"})).unwrap();

    // Time filter with table.
    let results = db.query().table("sessions").since(60).exec().unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r.table, "sessions");
    }
}

#[test]
fn time_filter_with_traverse() {
    let (db, _dir) = temp_graph_ts_db();

    let r1 = db.insert("sessions", json!({"data": "A"})).unwrap();
    let r2 = db.insert("files", json!({"data": "B"})).unwrap();
    db.relate(&r1.id, "modified", &r2.id, None).unwrap();

    // Traverse with time filter — recent records should find endpoints.
    let results = db
        .query()
        .table("sessions")
        .since(60)
        .traverse("->modified")
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, r2.id);

    // Future after() should exclude the starting record, yielding no traversal seeds.
    let future_us = r1.created_at.timestamp_micros() + 1_000_000_000;
    let results = db
        .query()
        .table("sessions")
        .after(future_us)
        .traverse("->modified")
        .exec()
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn combined_info_shows_all_plugins() {
    let (db, _dir) = temp_graph_ts_db();

    db.insert("sessions", json!({"data": "test"})).unwrap();

    let info = db.info().unwrap();
    assert!(info.plugins.contains_key("graph"));
    assert!(info.plugins.contains_key("timeseries"));
    assert_eq!(info.plugins["timeseries"]["entries"], 1);
}
