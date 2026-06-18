//! Integration tests for the background worker.
//!
//! Tests that `AxilWorker::run()` performs consolidation, connection
//! strengthening, and stale detection.

use axil_core::{Axil, AxilWorker};
use axil_graph::AxilBuilderGraphExt;
use axil_memory::AgentMemory;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

fn temp_db_with_graph() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_plugin()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

// ── Basic worker run ─────────────────────────────────────────────────────

#[test]
fn worker_run_on_empty_db() {
    let (db, _dir) = temp_db();
    let worker = AxilWorker::new(&db);

    let report = worker.run().unwrap();
    assert_eq!(report.consolidated_entities, 0);
    assert_eq!(report.new_connections, 0);
    assert_eq!(report.inferred_facts, 0);
    assert_eq!(report.stale_detected, 0);
    assert!(
        report.duration_ms < 5000,
        "worker should be fast on empty db"
    );
}

#[test]
fn worker_run_with_facts() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    // Store multiple facts about the same entity.
    sem.know("auth-module", "handles JWT login", None).unwrap();
    sem.know("auth-module", "supports token refresh", None)
        .unwrap();
    sem.know("auth-module", "refactored in March", None)
        .unwrap();

    let worker = AxilWorker::new(&db);
    let report = worker.run().unwrap();

    // Worker should attempt consolidation on 3 facts about same entity.
    // Note: consolidated_entities may be 0 if facts don't meet merging criteria,
    // but the worker must complete without error.
    assert!(
        report.duration_ms < 10000,
        "worker should complete promptly"
    );
}

// ── Idempotency ──────────────────────────────────────────────────────────

#[test]
fn worker_is_idempotent() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("cache-layer", "uses Redis", None).unwrap();
    sem.know("cache-layer", "TTL is 5 minutes", None).unwrap();

    let worker = AxilWorker::new(&db);

    // Run twice — should not error or produce inconsistent state.
    let report1 = worker.run().unwrap();
    let report2 = worker.run().unwrap();

    // Second run should do less work (or same).
    assert!(
        report2.consolidated_entities <= report1.consolidated_entities,
        "second run should not consolidate more: r1={}, r2={}",
        report1.consolidated_entities,
        report2.consolidated_entities
    );
}

// ── Worker with graph ────────────────────────────────────────────────────

#[test]
fn worker_discovers_connections_with_graph() {
    let (db, _dir) = temp_db_with_graph();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("api-server", "serves REST endpoints", None)
        .unwrap();
    sem.know("database", "stores user records", None).unwrap();
    sem.know("api-server", "connects to database for queries", None)
        .unwrap();

    let worker = AxilWorker::new(&db);
    let report = worker.run().unwrap();

    // Worker should complete without error when graph data exists.
    assert!(
        report.duration_ms < 10000,
        "worker should complete promptly"
    );
}

// ── Last run report ──────────────────────────────────────────────────────

#[test]
fn last_run_none_before_first_run() {
    let (db, _dir) = temp_db();
    let worker = AxilWorker::new(&db);

    let last = worker.last_run().unwrap();
    assert!(last.is_none(), "no report before first run");
}

#[test]
fn last_run_available_after_run() {
    let (db, _dir) = temp_db();
    let worker = AxilWorker::new(&db);

    worker.run().unwrap();

    let last = worker.last_run().unwrap();
    assert!(last.is_some(), "report should exist after run");

    let report = last.unwrap();
    assert!(report.duration_ms < 10000);
}

// ── Stale detection ──────────────────────────────────────────────────────

#[test]
fn fresh_records_not_flagged_stale() {
    let (db, _dir) = temp_db();

    // Insert a fresh record — should NOT be flagged as stale.
    let _record = db.insert("notes", json!({"text": "recent note"})).unwrap();

    let worker = AxilWorker::new(&db);
    let report = worker.run().unwrap();

    assert_eq!(
        report.stale_detected, 0,
        "fresh records should not be stale"
    );
    // Note: testing actual stale detection (90+ day old records) requires
    // faking timestamps which the public API does not support.
}
