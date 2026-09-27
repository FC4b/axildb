//! `Axil::resolve_error` across crates: the `resolves` graph edge it keeps in
//! step with the resolution, and the derived "Resume Here" block, whose
//! open-error list filters on the `resolved` field `resolve_error` writes.

use axil_checkpoint::derive_checkpoint_from_session;
use axil_core::{
    Axil, DecisionInput, Direction, ErrorInput, RecordId, ResolveInput, ResolveResult,
    WriteSource,
};
use axil_graph::AxilBuilderGraphExt;
use axil_memory::WorkingMemory;
use serde_json::json;
use tempfile::TempDir;

fn temp_db_with_graph() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

fn error(db: &Axil, text: &str) -> RecordId {
    db.remember_error(ErrorInput {
        error: text,
        root_cause: None,
        fix: None,
        files: None,
        agent_id: None,
        external_id: None,
        force_new: false,
        source: WriteSource::Core,
    })
    .unwrap()
    .id
}

fn decision(db: &Axil, text: &str) -> RecordId {
    db.remember_decision(DecisionInput {
        summary: text,
        reason: None,
        files: None,
        agent_id: None,
        external_id: None,
        force_new: false,
        source: WriteSource::Core,
    })
    .unwrap()
    .id
}

fn resolve(db: &Axil, id: &RecordId, by: Option<&RecordId>, reopen: bool) -> ResolveResult {
    db.resolve_error(ResolveInput {
        error_id: id,
        by,
        note: None,
        reopen,
    })
    .unwrap()
}

fn resolves_edges_into(db: &Axil, id: &RecordId) -> Vec<RecordId> {
    db.edges(id, Some("resolves"), Direction::In)
        .unwrap()
        .into_iter()
        .map(|e| e.from)
        .collect()
}

#[test]
fn resolves_edge_follows_the_resolution() {
    let (db, _dir) = temp_db_with_graph();
    let err = error(&db, "release job fails with crates.io 403");
    let first_fix = decision(&db, "re-ran the release job");
    let real_fix = decision(&db, "rotated CARGO_REGISTRY_TOKEN; 2.3.0 published");

    resolve(&db, &err, Some(&first_fix), false);
    assert_eq!(resolves_edges_into(&db, &err), vec![first_fix]);

    // Re-pointing the resolution moves the edge instead of adding a second one.
    let moved = resolve(&db, &err, Some(&real_fix), false);
    assert!(moved.changed);
    assert_eq!(moved.resolved_by, Some(real_fix.to_string()));
    assert_eq!(resolves_edges_into(&db, &err), vec![real_fix.clone()]);

    // The same fix again is a no-op and leaves exactly one edge.
    assert!(!resolve(&db, &err, Some(&real_fix), false).changed);
    assert_eq!(resolves_edges_into(&db, &err).len(), 1);

    resolve(&db, &err, None, true);
    assert!(resolves_edges_into(&db, &err).is_empty());
}

#[test]
fn resume_here_lists_only_open_errors() {
    let (db, _dir) = temp_db_with_graph();
    WorkingMemory::new(&db)
        .start_session(Some(json!({"task": "resolve contract"})))
        .unwrap();
    let fixed = error(&db, "release job fails with crates.io 403");
    error(&db, "nightly fuzz panics on a multi-byte escape");

    resolve(&db, &fixed, None, false);

    let checkpoint = derive_checkpoint_from_session(&db).expect("derived checkpoint");
    let open = &checkpoint.open_questions;
    assert!(
        open.iter().any(|q| q.contains("nightly fuzz")),
        "an error without a `resolved` field is open: {open:?}"
    );
    assert!(
        !open.iter().any(|q| q.contains("crates.io 403")),
        "a resolved error must drop out of Resume Here: {open:?}"
    );

    // Reopening puts it back.
    resolve(&db, &fixed, None, true);
    let checkpoint = derive_checkpoint_from_session(&db).expect("derived checkpoint");
    assert!(checkpoint
        .open_questions
        .iter()
        .any(|q| q.contains("crates.io 403")));
}
