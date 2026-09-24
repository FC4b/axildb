//! A bare `axil heal` downsamples: records past `full_retention_days` are
//! summarized and the originals purged. That purge is automatic deletion, so
//! `[healing] auto_compact = false` — "only an explicit compact deletes" —
//! must turn it off.

#![cfg(feature = "timeseries")]

use std::path::Path;
use std::process::Command;

use axil_core::{Axil, RecordId};
use axil_timeseries::AxilBuilderTimeSeriesExt;

fn heal(db: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("--db")
        .arg(db)
        .arg("heal")
        .env_remove("AXIL_DB")
        .current_dir(db.parent().unwrap())
        .output()
        .expect("failed to run axil");
    assert!(
        output.status.success(),
        "heal failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Three `notes` records created 200 days ago — well past the default
/// 90-day retention window — indexed in the time-series store.
fn seed_aged_records(db: &Path) -> Vec<RecordId> {
    let handle = Axil::open(db)
        .with_timeseries_engine()
        .unwrap()
        .build()
        .unwrap();
    let created = chrono::Utc::now() - chrono::Duration::days(200);
    (0..3)
        .map(|i| {
            let data = serde_json::json!({ "summary": format!("aged note number {i}") });
            handle.insert_at("notes", data, created).unwrap().id
        })
        .collect()
}

fn surviving(db: &Path, ids: &[RecordId]) -> usize {
    let handle = Axil::open(db).build().unwrap();
    ids.iter()
        .filter(|id| handle.get(id).unwrap().is_some())
        .count()
}

#[test]
fn bare_heal_keeps_aged_records_when_auto_compact_is_off() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let ids = seed_aged_records(&db);
    std::fs::write(
        dir.path().join("axil.toml"),
        "[healing]\nauto_compact = false\n",
    )
    .unwrap();

    heal(&db);

    assert_eq!(
        surviving(&db, &ids),
        ids.len(),
        "with auto_compact = false a bare heal must not purge aged records"
    );
}

#[test]
fn bare_heal_downsamples_aged_records_by_default() {
    // Control for the test above: the same setup does purge by default, so a
    // pass there is not just a downsample that never ran.
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let ids = seed_aged_records(&db);

    heal(&db);

    assert_eq!(
        surviving(&db, &ids),
        0,
        "default heal should downsample aged records"
    );
}
