use axil_core::Axil;
use axil_timeseries::AxilBuilderTimeSeriesExt;
use serde_json::json;

fn temp_ts_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_timeseries_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn insert_auto_indexes() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"summary": "test"})).unwrap();

    // Record should be findable via since().
    let results = db.since(None, 60).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, r.id);
}

#[test]
fn since_filters_by_table() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"a": 1})).unwrap();
    db.insert("decisions", json!({"b": 2})).unwrap();

    let sessions = db.since(Some("sessions"), 60).unwrap();
    assert_eq!(sessions.len(), 1);

    let decisions = db.since(Some("decisions"), 60).unwrap();
    assert_eq!(decisions.len(), 1);

    let all = db.since(None, 60).unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn timeline_returns_newest_first() {
    let (db, _dir) = temp_ts_db();
    let r1 = db.insert("sessions", json!({"n": 1})).unwrap();
    // Sleep to guarantee distinct microsecond timestamps (not just ULID ordering).
    std::thread::sleep(std::time::Duration::from_millis(2));
    let r2 = db.insert("sessions", json!({"n": 2})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let r3 = db.insert("sessions", json!({"n": 3})).unwrap();

    let timeline = db.timeline(None, 2).unwrap();
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline[0].id, r3.id);
    assert_eq!(timeline[1].id, r2.id);

    let _ = r1;
}

#[test]
fn delete_removes_from_time_index() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"data": "test"})).unwrap();

    let before = db.since(None, 60).unwrap();
    assert_eq!(before.len(), 1);

    db.delete(&r.id).unwrap();

    let after = db.since(None, 60).unwrap();
    assert!(after.is_empty());
}

#[test]
fn time_range_query() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"data": "test"})).unwrap();
    let created_us = r.created_at.timestamp_micros();

    // Range that includes the record.
    let results = db
        .time_range(None, created_us - 1_000_000, created_us + 1_000_000)
        .unwrap();
    assert_eq!(results.len(), 1);

    // Range that excludes the record.
    let results = db
        .time_range(None, created_us + 1_000_000, created_us + 2_000_000)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn query_builder_since() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"score": 10})).unwrap();
    db.insert("sessions", json!({"score": 20})).unwrap();

    let results = db.query().since(60).exec().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn query_builder_since_with_table_filter() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"score": 10})).unwrap();
    db.insert("decisions", json!({"score": 20})).unwrap();

    let results = db.query().table("sessions").since(60).exec().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["score"], 10);
}

#[test]
fn query_builder_between() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"data": "test"})).unwrap();
    let ts = r.created_at.timestamp_micros();

    let results = db
        .query()
        .between(ts - 1_000_000, ts + 1_000_000)
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn query_builder_after() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"data": "test"})).unwrap();
    let ts = r.created_at.timestamp_micros();

    // After a point before the record.
    let results = db.query().after(ts - 1_000_000).exec().unwrap();
    assert_eq!(results.len(), 1);

    // After a point after the record.
    let results = db.query().after(ts + 1_000_000).exec().unwrap();
    assert!(results.is_empty());
}

#[test]
fn query_builder_before() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"data": "test"})).unwrap();
    let ts = r.created_at.timestamp_micros();

    // Before a point after the record.
    let results = db.query().before(ts + 1_000_000).exec().unwrap();
    assert_eq!(results.len(), 1);

    // Before a point before the record.
    let results = db.query().before(ts - 1_000_000).exec().unwrap();
    assert!(results.is_empty());
}

#[test]
fn query_builder_changed_since() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "test"})).unwrap();

    // Recently inserted records should show as "changed since".
    let results = db.query().changed_since(60).exec().unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn changed_since_api() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "test"})).unwrap();

    let results = db.changed_since(None, 60).unwrap();
    assert_eq!(results.len(), 1);

    // changed_since(0) computes threshold = now, so records created
    // before now are excluded — the result should be empty.
    std::thread::sleep(std::time::Duration::from_millis(2));
    let results = db.changed_since(None, 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn changed_since_filters_by_table() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"a": 1})).unwrap();
    db.insert("decisions", json!({"b": 2})).unwrap();

    let sessions = db.changed_since(Some("sessions"), 60).unwrap();
    assert_eq!(sessions.len(), 1);

    let all = db.changed_since(None, 60).unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn time_filter_with_where_clause() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"score": 10})).unwrap();
    db.insert("sessions", json!({"score": 20})).unwrap();
    db.insert("sessions", json!({"score": 30})).unwrap();

    let results = db
        .query()
        .since(60)
        .where_field("score", axil_core::Op::Gte, json!(20))
        .exec()
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn time_filter_with_limit() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();
    db.insert("sessions", json!({"n": 2})).unwrap();
    db.insert("sessions", json!({"n": 3})).unwrap();

    let results = db.query().since(60).limit(2).exec().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn update_refreshes_timeseries_updated_at() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"v": 1})).unwrap();

    // Small delay to ensure updated_at differs from created_at.
    std::thread::sleep(std::time::Duration::from_millis(5));

    let updated = db.update(&r.id, json!({"v": 2})).unwrap();
    assert!(updated.updated_at > r.created_at);

    // changed_since should find the updated record.
    let changed = db.changed_since(None, 60).unwrap();
    assert_eq!(changed.len(), 1);
    assert_eq!(changed[0].id, r.id);
    assert_eq!(changed[0].data["v"], 2);
}

#[test]
fn update_after_delete_and_reinsert() {
    let (db, _dir) = temp_ts_db();
    let r = db.insert("sessions", json!({"v": 1})).unwrap();

    db.delete(&r.id).unwrap();

    let r2 = db.insert("sessions", json!({"v": 2})).unwrap();

    let updated = db.update(&r2.id, json!({"v": 3})).unwrap();
    assert_eq!(updated.data["v"], 3);
}

#[test]
fn empty_range_returns_nothing() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "test"})).unwrap();

    // Future range with no records.
    let now_us = chrono::Utc::now().timestamp_micros();
    let results = db
        .time_range(None, now_us + 1_000_000_000, now_us + 2_000_000_000)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn latest_with_zero_limit() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "test"})).unwrap();

    let results = db.timeline(None, 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn persistence_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");

    // Insert in first session.
    {
        let db = Axil::open(&path)
            .with_timeseries_plugin()
            .unwrap()
            .build()
            .unwrap();
        db.insert("sessions", json!({"data": "persisted"})).unwrap();
    }

    // Reopen and verify.
    {
        let db = Axil::open(&path)
            .with_timeseries_plugin()
            .unwrap()
            .build()
            .unwrap();
        let results = db.since(None, 60).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data["data"], "persisted");
    }
}

#[test]
fn time_filter_without_timeseries_index() {
    // Time filters should work via table scan even without the timeseries plugin.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();

    let r = db.insert("sessions", json!({"data": "test"})).unwrap();

    // since() on QueryBuilder should still filter by created_at.
    let results = db.query().table("sessions").since(60).exec().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, r.id);

    // Future after() should return empty.
    let future_us = r.created_at.timestamp_micros() + 1_000_000_000;
    let results = db
        .query()
        .table("sessions")
        .after(future_us)
        .exec()
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn order_by_time_sorts_on_struct_created_at() {
    let (db, _dir) = temp_ts_db();
    let r1 = db.insert("sessions", json!({"n": 1})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let r2 = db.insert("sessions", json!({"n": 2})).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let r3 = db.insert("sessions", json!({"n": 3})).unwrap();

    // Ascending — oldest first.
    let asc = db
        .query()
        .since(60)
        .order_by_time(axil_core::SortDirection::Asc)
        .exec()
        .unwrap();
    assert_eq!(asc.len(), 3);
    assert_eq!(asc[0].id, r1.id);
    assert_eq!(asc[1].id, r2.id);
    assert_eq!(asc[2].id, r3.id);

    // Descending — newest first.
    let desc = db
        .query()
        .since(60)
        .order_by_time(axil_core::SortDirection::Desc)
        .exec()
        .unwrap();
    assert_eq!(desc.len(), 3);
    assert_eq!(desc[0].id, r3.id);
    assert_eq!(desc[1].id, r2.id);
    assert_eq!(desc[2].id, r1.id);
}

#[test]
fn backfill_timeseries_on_existing_db() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");

    // Create DB without timeseries, add records.
    {
        let db = Axil::open(&path).build().unwrap();
        db.insert("sessions", json!({"n": 1})).unwrap();
        db.insert("sessions", json!({"n": 2})).unwrap();
        db.insert("decisions", json!({"n": 3})).unwrap();
    }

    // Reopen WITH timeseries and backfill.
    {
        let db = Axil::open(&path)
            .with_timeseries_plugin()
            .unwrap()
            .build()
            .unwrap();
        let count = db.backfill_timeseries().unwrap();
        assert_eq!(count, 3);

        // All records should now be in the time index.
        let results = db.since(None, 60).unwrap();
        assert_eq!(results.len(), 3);

        let sessions = db.since(Some("sessions"), 60).unwrap();
        assert_eq!(sessions.len(), 2);
    }
}

#[test]
fn info_shows_timeseries() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "test"})).unwrap();

    let info = db.info().unwrap();
    let ts_info = info.plugins.get("timeseries").unwrap();
    assert_eq!(ts_info["entries"], 1);
}
