use axil_core::{Axil, TimeBucket};
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
fn count_by_day_single_day() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();
    db.insert("sessions", json!({"n": 2})).unwrap();
    db.insert("sessions", json!({"n": 3})).unwrap();

    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(None, TimeBucket::Day, now_us - 86_400_000_000, now_us + 1)
        .unwrap();

    // All records inserted within the same second — one day bucket.
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].1, 3);
}

#[test]
fn count_by_day_with_table_filter() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();
    db.insert("sessions", json!({"n": 2})).unwrap();
    db.insert("decisions", json!({"n": 3})).unwrap();

    let now_us = chrono::Utc::now().timestamp_micros();

    let sessions = db
        .count_by_bucket(
            Some("sessions"),
            TimeBucket::Day,
            now_us - 86_400_000_000,
            now_us + 1,
        )
        .unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].1, 2);

    let decisions = db
        .count_by_bucket(
            Some("decisions"),
            TimeBucket::Day,
            now_us - 86_400_000_000,
            now_us + 1,
        )
        .unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].1, 1);
}

#[test]
fn count_by_hour() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();
    db.insert("sessions", json!({"n": 2})).unwrap();

    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(None, TimeBucket::Hour, now_us - 3_600_000_000, now_us + 1)
        .unwrap();

    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].1, 2);
}

#[test]
fn count_by_week() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();

    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(
            None,
            TimeBucket::Week,
            now_us - 7 * 86_400_000_000,
            now_us + 1,
        )
        .unwrap();

    // At least one bucket with our record.
    assert!(!buckets.is_empty());
    let total: usize = buckets.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 1);
}

#[test]
fn count_by_month() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();

    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(
            None,
            TimeBucket::Month,
            now_us - 31 * 86_400_000_000,
            now_us + 1,
        )
        .unwrap();

    assert!(!buckets.is_empty());
    let total: usize = buckets.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 1);
}

#[test]
fn count_by_bucket_empty_range() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();

    // Future range — no records.
    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(
            None,
            TimeBucket::Day,
            now_us + 1_000_000_000,
            now_us + 2_000_000_000,
        )
        .unwrap();
    assert!(buckets.is_empty());
}

#[test]
fn count_by_bucket_inverted_range() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"n": 1})).unwrap();

    // Inverted range (start > end) — should return empty.
    let buckets = db.count_by_bucket(None, TimeBucket::Day, 100, 50).unwrap();
    assert!(buckets.is_empty());
}

#[test]
fn buckets_sorted_chronologically() {
    let (db, _dir) = temp_ts_db();
    // Insert a few records — all in the same instant but that's fine.
    for i in 0..5 {
        db.insert("sessions", json!({"n": i})).unwrap();
    }

    let now_us = chrono::Utc::now().timestamp_micros();
    let buckets = db
        .count_by_bucket(None, TimeBucket::Hour, now_us - 86_400_000_000, now_us + 1)
        .unwrap();

    // Verify buckets are sorted by timestamp.
    for window in buckets.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}
