//! Cross-crate contract test — `axil-checkpoint` and `axil-memory` must
//! agree on the `_sessions` table name, the `status="active"`
//! sentinel, and the minimal row shape.
//!
//! `axil-checkpoint` deliberately stays a leaf crate (no dep on
//! `axil-memory`) so it hard-codes those strings. This test catches
//! the silent-drift bug where memory renames either side and checkpoint
//! quietly stops finding sessions.
//!
//! Strategy: write a session through `axil-memory`'s public API,
//! then assert `axil-checkpoint` reuses it instead of inserting a new
//! one. If the table name or status sentinel ever drift apart,
//! `ensure_active_session` inserts a *second* row and the assertion
//! fails — pointing the maintainer at the contract before users hit
//! it.

use axil_core::Axil;
use axil_checkpoint::{snapshot, Checkpoint};
use axil_memory::{types::TABLE_SESSIONS, WorkingMemory};
use serde_json::json;

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn checkpoint_reuses_session_started_by_memory() {
    let (db, _dir) = temp_db();

    // Memory writes a session through its public lifecycle API.
    let wm = WorkingMemory::new(&db);
    let session = wm
        .start_session(Some(json!({"task": "contract test"})))
        .expect("start_session");

    // Checkpoint sees the row and attaches to it.
    let checkpoint = Checkpoint::from_value(json!({"goal": "cross-crate"})).unwrap();
    let record = snapshot(&db, &checkpoint).expect("snapshot");

    assert_eq!(
        record.data["session_id"].as_str().unwrap(),
        session.id.to_string(),
        "axil-checkpoint must reuse axil-memory's active session — string drift in \
         table name or status sentinel will surface here",
    );
    // Sanity: exactly one session row exists. If checkpoint inserted a
    // second one because it disagreed on the table name, this fails.
    let sessions = db.list(TABLE_SESSIONS).unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "expected exactly the memory-started session; got {} rows",
        sessions.len(),
    );
}

#[test]
fn checkpoint_creates_session_with_shape_memory_recognizes() {
    let (db, _dir) = temp_db();

    // Checkpoint with no prior session — it auto-creates one.
    let checkpoint = Checkpoint::from_value(json!({"goal": "auto-create"})).unwrap();
    let _ = snapshot(&db, &checkpoint).expect("snapshot");

    // Memory's session lister must see the auto-created row through
    // its own filter (status="active"). If checkpoint's row shape drifts
    // from what memory's lister expects, this returns zero.
    let wm = WorkingMemory::new(&db);
    let active = wm
        .list_sessions(true)
        .expect("list active sessions");
    assert_eq!(
        active.len(),
        1,
        "axil-memory's list_sessions(active_only=true) must see checkpoint's \
         auto-created row — drift in the status sentinel will surface here",
    );
}
