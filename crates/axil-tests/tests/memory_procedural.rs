//! Integration tests for procedural memory.

use axil_core::Axil;
use axil_memory::{AgentMemory, Outcome};
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn learn_procedure() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .procedural()
        .learn(
            "fix-timeout",
            "Check pool size first, then network config, then server load",
            None,
        )
        .unwrap();

    assert_eq!(record.data["pattern_name"], "fix-timeout");
    assert_eq!(record.data["confidence"], 0.5);
    assert_eq!(record.data["applications"], 1);
}

#[test]
fn learn_same_name_reinforces() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.procedural()
        .learn("fix-timeout", "v1: Check pool", None)
        .unwrap();
    let updated = mem
        .procedural()
        .learn("fix-timeout", "v2: Check pool and network", None)
        .unwrap();

    // Should update, not create new.
    let procedures = mem.procedural().list().unwrap();
    assert_eq!(procedures.len(), 1);

    // Confidence should have increased.
    let conf = updated
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(conf > 0.5, "confidence should increase on reinforcement");

    assert_eq!(updated.data["description"], "v2: Check pool and network");
    assert_eq!(updated.data["applications"], 2);
}

#[test]
fn confidence_adjusts_on_outcomes() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .procedural()
        .learn("test-proc", "test description", None)
        .unwrap();
    let initial_conf = record
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();

    // Success boosts confidence.
    let after_success = mem
        .procedural()
        .record_outcome(&record.id, Outcome::Success)
        .unwrap();
    let success_conf = after_success
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(success_conf > initial_conf);

    // Failure reduces confidence.
    let after_failure = mem
        .procedural()
        .record_outcome(&record.id, Outcome::Failure)
        .unwrap();
    let failure_conf = after_failure
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(failure_conf < success_conf);
}

#[test]
fn list_sorted_by_confidence() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let low = mem.procedural().learn("low-conf", "test", None).unwrap();
    let high = mem.procedural().learn("high-conf", "test", None).unwrap();

    // Boost high-conf.
    mem.procedural()
        .record_outcome(&high.id, Outcome::Success)
        .unwrap();
    mem.procedural()
        .record_outcome(&high.id, Outcome::Success)
        .unwrap();

    let list = mem.procedural().list().unwrap();
    assert_eq!(list.len(), 2);

    let first_name = list[0].data["pattern_name"].as_str().unwrap();
    assert_eq!(first_name, "high-conf");
}

#[test]
fn extract_from_successful_episode() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let episode = mem
        .episodic()
        .create(
            "Fixed timeout by increasing pool size",
            Outcome::Success,
            Some(vec!["Increased pool from 5 to 20".into()]),
            None,
        )
        .unwrap();

    let pattern = mem.procedural().extract_from_episode(&episode).unwrap();
    assert!(pattern.is_some());
    let pattern = pattern.unwrap();
    assert!(pattern.data["description"]
        .as_str()
        .unwrap()
        .contains("pool"));
}

#[test]
fn no_extract_from_failed_episode() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let episode = mem
        .episodic()
        .create("Failed attempt", Outcome::Failure, None, None)
        .unwrap();

    let pattern = mem.procedural().extract_from_episode(&episode).unwrap();
    assert!(pattern.is_none());
}

#[test]
fn find_by_name() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.procedural().learn("fix-timeout", "desc", None).unwrap();
    mem.procedural()
        .learn("add-api-route", "desc", None)
        .unwrap();

    let found = mem.procedural().find_by_name("fix-timeout").unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().data["pattern_name"], "fix-timeout");

    let not_found = mem.procedural().find_by_name("nonexistent").unwrap();
    assert!(not_found.is_none());
}

#[test]
fn partial_outcome_adjusts_confidence() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .procedural()
        .learn("test-partial", "test description", None)
        .unwrap();
    let initial_conf = record
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();

    // Partial outcome slightly decreases confidence.
    let after_partial = mem
        .procedural()
        .record_outcome(&record.id, Outcome::Partial)
        .unwrap();
    let partial_conf = after_partial
        .data
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap();

    assert!(
        partial_conf < initial_conf,
        "partial should decrease confidence"
    );
    // Partial penalty is half of failure penalty (0.075 vs 0.15).
    assert!(
        partial_conf > initial_conf - 0.15,
        "partial penalty should be less than failure penalty"
    );

    // Verify partials counter was incremented.
    let partials = after_partial
        .data
        .get("partials")
        .and_then(|v| v.as_u64())
        .unwrap();
    assert_eq!(partials, 1);
}
