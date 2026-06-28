//! Single-writer / lock-free-reader concurrency contract.
//!
//! Axil is single-writer. redb (3.x) takes an **exclusive** advisory OS lock on
//! the core `.axil` for the lifetime of a writable handle, so:
//!
//! * A second *writer* fails fast with the typed [`AxilError::Busy`] (callers
//!   retry or fall back) rather than a generic storage error.
//! * A read-only open requests a **shared** lock, which conflicts with the
//!   writer's exclusive lock — so a read-only open also reports `Busy` while a
//!   writer is live, and only succeeds once every writer has closed (released
//!   the lock). This is why hot read commands do a bounded busy-retry on the
//!   writer *first*: the writer is short-lived, and the read-only open is a
//!   fallback for the gap between writer sessions, not a way to read *through*
//!   a live writer.
//!
//! These tests pin that contract so a later redb bump or refactor can't quietly
//! change it.

use axil_core::{Axil, AxilError, RecordId, Storage};
use serde_json::json;

/// A second writable open of a file already held for writing fails with the
/// typed `Busy` error — not a generic storage error.
#[test]
fn second_writer_is_busy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("busy.axil");

    // First writer holds the single-writer lock for the rest of the test.
    let _writer = Axil::open(&path).build().expect("first writer opens");

    // Second writer must be rejected with the typed Busy variant.
    match Axil::open(&path).build() {
        Ok(_) => panic!("second writer must fail while the first holds the lock"),
        Err(err) => {
            assert!(err.is_busy(), "expected AxilError::Busy, got: {err:?} ({err})");
            assert!(matches!(err, AxilError::Busy));
        }
    }
}

/// A read-only open also reports `Busy` while a writer holds the exclusive
/// lock — redb's shared lock can't coexist with the writer's exclusive lock.
#[test]
fn read_only_open_is_busy_while_writer_live() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro-live.axil");

    let _writer = Axil::open(&path).build().expect("writer opens");

    match Storage::open_read_only(&path) {
        Ok(_) => panic!("read-only open must not succeed while a writer holds the exclusive lock"),
        Err(err) => assert!(
            err.is_busy(),
            "read-only open under a live writer should be Busy, got {err:?}"
        ),
    }
}

/// Once the writer has closed, a read-only open succeeds and serves the
/// committed `get`/`list` results without taking the write lock.
#[test]
fn reader_after_writer_serves_committed_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reader.axil");

    let id: RecordId = {
        // Writer commits a record, then drops (releasing the exclusive lock).
        let writer = Axil::open(&path).build().expect("writer opens");
        let rec = writer
            .insert("decisions", json!({"summary": "use redb single-writer"}))
            .expect("insert commits");
        rec.id.clone()
    };

    // A read-only open of the committed file succeeds and never takes the lock.
    let reader = Storage::open_read_only(&path).expect("read-only open after writer closed");
    assert!(reader.is_read_only());

    // get() sees the committed record.
    let got = reader.get(&id).expect("read-only get works");
    assert!(got.is_some(), "read-only handle should see the committed record");
    assert_eq!(got.unwrap().id, id);

    // list() sees the committed record in its table.
    let listed = reader
        .list("decisions", usize::MAX, 0)
        .expect("read-only list works");
    assert!(
        listed.iter().any(|r| r.id == id),
        "read-only list should include the committed record"
    );
}

/// A read-only `Storage` rejects writes with `Busy` rather than silently
/// dropping or panicking.
#[test]
fn read_only_storage_rejects_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro.axil");

    // Create the file with one committed record, then drop the writer so the
    // read-only open is the only handle.
    {
        let writer = Axil::open(&path).build().expect("writer opens");
        writer
            .insert("context", json!({"summary": "seed"}))
            .expect("insert");
    }

    let ro = Storage::open_read_only(&path).expect("read-only open");
    let rec = axil_core::Record::new("context", json!({"summary": "blocked"}));
    match ro.insert(&rec) {
        Ok(_) => panic!("a read-only handle must reject inserts"),
        Err(err) => assert!(err.is_busy(), "expected Busy on read-only write, got {err:?}"),
    }
}

/// A read-only open via the `AxilBuilder::read_only` flag serves the same
/// committed data through the full `Axil` handle once the writer has closed.
#[test]
fn read_only_builder_serves_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro-builder.axil");

    let id = {
        let writer = Axil::open(&path).build().expect("writer opens");
        let rec = writer
            .insert("errors", json!({"error": "timeout", "fix": "retry"}))
            .expect("insert");
        rec.id.clone()
    };

    // Read-only handle through the builder.
    let reader = Axil::open(&path)
        .read_only(true)
        .build()
        .expect("read-only builder open after writer closed");

    let got = reader.get(&id).expect("read-only get");
    assert!(got.is_some());
    assert_eq!(got.unwrap().id, id);
}
