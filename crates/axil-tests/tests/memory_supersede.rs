//! Integration tests for memory superseding and TTL.

use axil_core::Axil;
use axil_memory::types::{META_SUPERSEDED, META_SUPERSEDED_BY};
use axil_memory::AgentMemory;
use chrono::Duration;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

// ── TTL tests ────────────────────────────────────────────────────────────

#[test]
fn set_and_check_ttl() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = db.insert("test", json!({"data": "value"})).unwrap();

    // Set TTL to 1 hour.
    mem.ttl().set_ttl(&record.id, Duration::hours(1)).unwrap();

    let updated = db.get(&record.id).unwrap().unwrap();
    assert!(!mem.ttl().is_expired(&updated));
}

#[test]
fn expired_record_detected() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = db.insert("test", json!({"data": "value"})).unwrap();

    // Set TTL to -1 hour (already expired).
    mem.ttl()
        .set_expiry(&record.id, chrono::Utc::now() - Duration::hours(1))
        .unwrap();

    let updated = db.get(&record.id).unwrap().unwrap();
    assert!(mem.ttl().is_expired(&updated));
}

#[test]
fn clear_ttl() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = db.insert("test", json!({"data": "value"})).unwrap();

    // Set then clear TTL.
    mem.ttl().set_ttl(&record.id, Duration::hours(1)).unwrap();
    mem.ttl().clear_ttl(&record.id).unwrap();

    let updated = db.get(&record.id).unwrap().unwrap();
    assert!(!mem.ttl().is_expired(&updated));
}

#[test]
fn filter_active_removes_expired() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let r1 = db.insert("test", json!({"val": 1})).unwrap();
    let r2 = db.insert("test", json!({"val": 2})).unwrap();

    // Expire r1.
    mem.ttl()
        .set_expiry(&r1.id, chrono::Utc::now() - Duration::hours(1))
        .unwrap();

    let all = db.list("test").unwrap();
    let filtered = mem.ttl().filter_active(all);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].data["val"], 2);
}

// ── History / bi-temporal tests ────────────────────────────────────────

#[test]
fn entity_history_shows_all_versions() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic()
        .know("auth", "Uses session cookies", None)
        .unwrap();
    mem.semantic()
        .know("auth", "Migrated from sessions to JWT", None)
        .unwrap();
    mem.semantic()
        .know("auth", "Added refresh token rotation", None)
        .unwrap();

    let history = mem.semantic().history("auth").unwrap();
    assert_eq!(history.len(), 3);

    // Should be chronological.
    assert!(history[0].created_at <= history[1].created_at);
    assert!(history[1].created_at <= history[2].created_at);
}

// ── Supersede engine tests ──────────────────────────────────────────

#[test]
fn supersede_marks_old_record() {
    use axil_memory::supersede::SupersedeEngine;

    let (db, _dir) = temp_db();

    // Insert two semantically identical records in the same table.
    let old = db
        .insert(
            "_entities",
            json!({
                "entity": "auth",
                "fact": "Uses session cookies for authentication",
            }),
        )
        .unwrap();

    // Manually mark as superseded (since we don't have embeddings in this test).
    let engine = SupersedeEngine::new(&db);
    let mut old_data = old.data.clone();
    axil_memory::ttl::set_meta_field(&mut old_data, META_SUPERSEDED, json!(true));
    axil_memory::ttl::set_meta_field(&mut old_data, META_SUPERSEDED_BY, json!("new-record-id"));
    db.update(&old.id, old_data).unwrap();

    let updated = db.get(&old.id).unwrap().unwrap();
    assert!(axil_memory::ttl::is_record_superseded(&updated));

    // History should still show superseded records.
    let history = engine.history("auth", "_entities").unwrap();
    assert_eq!(history.len(), 1);
}

#[test]
fn bitemporal_metadata_set_correctly() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .semantic()
        .know("test-entity", "Some fact about it", None)
        .unwrap();

    // Bitemporal metadata should be present.
    let meta = record.data.get("_meta").expect("_meta should exist");
    assert!(
        meta.get("recorded_at").is_some(),
        "recorded_at should be set"
    );
    assert!(meta.get("valid_from").is_some(), "valid_from should be set");

    // Both should be valid RFC3339 timestamps.
    let recorded_at = meta["recorded_at"].as_str().unwrap();
    let valid_from = meta["valid_from"].as_str().unwrap();
    assert!(
        chrono::DateTime::parse_from_rfc3339(recorded_at).is_ok(),
        "recorded_at should be valid RFC3339"
    );
    assert!(
        chrono::DateTime::parse_from_rfc3339(valid_from).is_ok(),
        "valid_from should be valid RFC3339"
    );
}

#[test]
fn superseded_records_excluded_from_list() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic()
        .know("auth", "Uses JWT tokens", None)
        .unwrap();
    mem.semantic()
        .know("auth", "Uses session cookies", None)
        .unwrap();

    // Mark first as superseded.
    let all = db.list("_entities").unwrap();
    let first = &all[0];
    let mut data = first.data.clone();
    axil_memory::ttl::set_meta_field(&mut data, META_SUPERSEDED, json!(true));
    db.update(&first.id, data).unwrap();

    // list_facts should exclude superseded.
    let facts = mem.semantic().list_facts(Some("auth")).unwrap();
    assert_eq!(facts.len(), 1);

    // about() should also exclude superseded.
    let knowledge = mem.semantic().about("auth").unwrap();
    assert_eq!(knowledge.facts.len(), 1);
}

#[test]
fn filter_expired_and_superseded_combined() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let r1 = db.insert("test", json!({"val": "active"})).unwrap();
    let r2 = db.insert("test", json!({"val": "expired"})).unwrap();
    let r3 = db.insert("test", json!({"val": "superseded"})).unwrap();

    // Expire r2.
    mem.ttl()
        .set_expiry(&r2.id, chrono::Utc::now() - Duration::hours(1))
        .unwrap();

    // Supersede r3.
    let mut data = r3.data.clone();
    axil_memory::ttl::set_meta_field(&mut data, META_SUPERSEDED, json!(true));
    db.update(&r3.id, data).unwrap();

    // filter_active should remove both expired and superseded.
    let all = db.list("test").unwrap();
    let filtered = mem.ttl().filter_active(all);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].data["val"], "active");
}
