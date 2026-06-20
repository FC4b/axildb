use axil_core::Axil;
use axil_timeseries::AxilBuilderTimeSeriesExt;
use serde_json::json;

fn temp_ts_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_timeseries_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn downsample_no_old_records() {
    let (db, _dir) = temp_ts_db();
    // Insert a fresh record.
    db.insert("sessions", json!({"data": "new"})).unwrap();

    // Retain everything from the last 90 days — nothing to downsample.
    let (summaries, purged) = db.downsample(90, true).unwrap();
    assert_eq!(summaries, 0);
    assert_eq!(purged, 0);
}

#[test]
fn downsample_creates_summaries_without_purge() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();
    db.insert("sessions", json!({"data": "B"})).unwrap();
    db.insert("decisions", json!({"data": "C"})).unwrap();

    // retain_days=0 means everything is "old" (cutoff is now).
    // purge=false means originals are kept.
    let (summaries, purged) = db.downsample(0, false).unwrap();

    // Two groups: (sessions, today) and (decisions, today).
    assert_eq!(summaries, 2);
    assert_eq!(purged, 0);

    // Original records still exist.
    assert_eq!(db.list("sessions").unwrap().len(), 2);
    assert_eq!(db.list("decisions").unwrap().len(), 1);

    // Summaries were created.
    let summary_records = db.list("_summaries").unwrap();
    assert_eq!(summary_records.len(), 2);

    // Verify summary structure.
    for s in &summary_records {
        assert_eq!(s.data["type"], "daily_summary");
        assert!(s.data["count"].as_u64().unwrap() > 0);
        assert!(s.data["table"].as_str().is_some());
        assert!(s.data["date"].as_str().is_some());
    }
}

#[test]
fn downsample_with_purge() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();
    db.insert("sessions", json!({"data": "B"})).unwrap();

    let (summaries, purged) = db.downsample(0, true).unwrap();
    assert_eq!(summaries, 1);
    assert_eq!(purged, 2);

    // Original records gone.
    assert_eq!(db.list("sessions").unwrap().len(), 0);

    // Summary exists.
    let summary_records = db.list("_summaries").unwrap();
    assert_eq!(summary_records.len(), 1);
    assert_eq!(summary_records[0].data["count"], 2);
}

#[test]
fn downsample_idempotent() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();

    // First downsample.
    let (s1, _) = db.downsample(0, false).unwrap();
    assert_eq!(s1, 1);

    // Second downsample — summary already exists, should not duplicate.
    let (s2, _) = db.downsample(0, false).unwrap();
    assert_eq!(s2, 0);

    // Still just one summary with count=1 (not double-counted).
    let summaries = db.list("_summaries").unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].data["count"], 1);
}

#[test]
fn downsample_does_not_summarise_summaries() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();

    // Create summaries.
    db.downsample(0, false).unwrap();

    // Now the _summaries table has a record. Running downsample again
    // should NOT create a summary-of-summaries.
    let (summaries, _) = db.downsample(0, false).unwrap();
    assert_eq!(summaries, 0);
    assert_eq!(db.list("_summaries").unwrap().len(), 1);
}

#[test]
fn downsample_summary_count_reflects_group_size() {
    let (db, _dir) = temp_ts_db();
    for i in 0..5 {
        db.insert("sessions", json!({"n": i})).unwrap();
    }
    db.insert("decisions", json!({"n": 99})).unwrap();

    db.downsample(0, false).unwrap();

    let summaries = db.list("_summaries").unwrap();
    let sessions_summary = summaries
        .iter()
        .find(|r| r.data["table"] == "sessions")
        .unwrap();
    assert_eq!(sessions_summary.data["count"], 5);

    let decisions_summary = summaries
        .iter()
        .find(|r| r.data["table"] == "decisions")
        .unwrap();
    assert_eq!(decisions_summary.data["count"], 1);
}

#[test]
fn repeated_downsample_increments_existing_summary() {
    let (db, _dir) = temp_ts_db();

    // First batch — 3 records.
    db.insert("sessions", json!({"n": 1})).unwrap();
    db.insert("sessions", json!({"n": 2})).unwrap();
    db.insert("sessions", json!({"n": 3})).unwrap();

    let (s1, p1) = db.downsample(0, true).unwrap();
    assert_eq!(s1, 1);
    assert_eq!(p1, 3);

    // Second batch — 2 more records in the same day.
    db.insert("sessions", json!({"n": 4})).unwrap();
    db.insert("sessions", json!({"n": 5})).unwrap();

    let (s2, p2) = db.downsample(0, true).unwrap();
    // No new summary created — existing one was updated.
    assert_eq!(s2, 0);
    assert_eq!(p2, 2);

    // Summary count should reflect all 5 records.
    let summaries = db.list("_summaries").unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].data["count"], 5);
}

#[test]
fn downsample_uses_full_insert_hooks() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();

    db.downsample(0, false).unwrap();

    // Summary should be findable via timeseries since() — proves
    // it went through the full insert path with plugin hooks.
    let all = db.since(None, 60).unwrap();
    let summary_count = all.iter().filter(|r| r.table == "_summaries").count();
    assert_eq!(summary_count, 1);
}

#[test]
fn weekly_downsample_consolidates_dailies() {
    let (db, _dir) = temp_ts_db();

    // Create 5 daily summaries within one Monday-Sunday week (2025-01-06 is a Monday).
    for day in 6..=10 {
        db.insert(
            "_summaries",
            json!({
                "table": "sessions",
                "date": format!("2025-01-{:02}", day),
                "count": 10,
                "type": "daily_summary",
            }),
        )
        .unwrap();
    }

    // Weekly downsample with 0 retention = everything is old.
    let (weeklies, dailies_purged) = db.downsample_weekly(0, true).unwrap();

    // All 5 days fall in the same Mon-Sun week → 1 weekly summary.
    assert_eq!(weeklies, 1);
    assert_eq!(dailies_purged, 5);

    let summaries = db.list("_summaries").unwrap();
    let weekly = summaries
        .iter()
        .find(|r| r.data["type"] == "weekly_summary")
        .unwrap();
    assert_eq!(weekly.data["count"], 50); // 5 × 10
    assert_eq!(weekly.data["table"], "sessions");
    assert_eq!(weekly.data["week"], "2025-01-06"); // Monday
}

#[test]
fn weekly_downsample_no_old_dailies() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();
    db.downsample(0, false).unwrap();

    // Daily summary was just created (today) — 365-day retention means nothing to consolidate.
    let (weeklies, dailies_purged) = db.downsample_weekly(365, true).unwrap();
    assert_eq!(weeklies, 0);
    assert_eq!(dailies_purged, 0);
}

#[test]
fn repeated_weekly_downsample_updates_count() {
    let (db, _dir) = temp_ts_db();

    // First batch: 3 dailies in one week.
    for day in 6..=8 {
        db.insert(
            "_summaries",
            json!({
                "table": "sessions",
                "date": format!("2025-01-{:02}", day),
                "count": 10,
                "type": "daily_summary",
            }),
        )
        .unwrap();
    }
    let (w1, p1) = db.downsample_weekly(0, true).unwrap();
    assert_eq!(w1, 1);
    assert_eq!(p1, 3);

    // Second batch: 2 more dailies in the same week.
    for day in 9..=10 {
        db.insert(
            "_summaries",
            json!({
                "table": "sessions",
                "date": format!("2025-01-{:02}", day),
                "count": 5,
                "type": "daily_summary",
            }),
        )
        .unwrap();
    }
    let (w2, p2) = db.downsample_weekly(0, true).unwrap();
    // No new weekly created — existing one was updated.
    assert_eq!(w2, 0);
    assert_eq!(p2, 2);

    // Weekly count should be 30 + 10 = 40 (3×10 + 2×5).
    let summaries = db.list("_summaries").unwrap();
    let weekly = summaries
        .iter()
        .find(|r| r.data["type"] == "weekly_summary")
        .unwrap();
    assert_eq!(weekly.data["count"], 40);
}

#[test]
fn weekly_downsample_idempotent() {
    let (db, _dir) = temp_ts_db();
    db.insert(
        "_summaries",
        json!({
            "table": "sessions",
            "date": "2025-01-01",
            "count": 5,
            "type": "daily_summary",
        }),
    )
    .unwrap();

    let (w1, _) = db.downsample_weekly(0, false).unwrap();
    assert_eq!(w1, 1);

    // Second call — weekly already exists, count should not inflate.
    let (w2, _) = db.downsample_weekly(0, false).unwrap();
    assert_eq!(w2, 0);

    let summaries = db.list("_summaries").unwrap();
    let weekly = summaries
        .iter()
        .find(|r| r.data["type"] == "weekly_summary")
        .unwrap();
    assert_eq!(weekly.data["count"], 5); // not 10 (no double-counting)
}

#[test]
fn heal_runs_both_tiers() {
    let (db, _dir) = temp_ts_db();
    db.insert("sessions", json!({"data": "A"})).unwrap();
    db.insert("sessions", json!({"data": "B"})).unwrap();

    let config = axil_core::TimeseriesConfig {
        full_retention_days: 0,
        daily_summary_days: 0,
        auto_downsample: true,
    };
    let report = db.heal(&config).unwrap();

    // Should have created a daily summary and purged 2 records.
    assert_eq!(report.daily_summaries_created, 1);
    assert_eq!(report.records_purged, 2);

    // Then consolidated that daily into a weekly and purged the daily.
    assert_eq!(report.weekly_summaries_created, 1);
    assert_eq!(report.daily_summaries_purged, 1);

    // Final state: one weekly summary only.
    let summaries = db.list("_summaries").unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].data["type"], "weekly_summary");
    assert_eq!(summaries[0].data["count"], 2);
}

#[test]
fn config_defaults() {
    let config = axil_core::AxilConfig::default();
    assert_eq!(config.timeseries.full_retention_days, 90);
    assert_eq!(config.timeseries.daily_summary_days, 365);
    assert!(config.timeseries.auto_downsample);
}

#[test]
fn config_parses_from_toml() {
    let toml_str =
        "[timeseries]\nfull_retention_days = 7\ndaily_summary_days = 30\nauto_downsample = false\n";
    let config: axil_core::AxilConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.timeseries.full_retention_days, 7);
    assert_eq!(config.timeseries.daily_summary_days, 30);
    assert!(!config.timeseries.auto_downsample);
}
