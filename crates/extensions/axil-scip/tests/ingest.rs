//! Round-trip test for SCIP ingest.
//!
//! Hand-constructs a tiny SCIP Index in memory, encodes it via `prost`,
//! writes it to a temp file, calls `ingest_scip`, and asserts the
//! expected entities/edges landed in the database.

use axil_core::{Axil, Direction};
use axil_graph::AxilBuilderGraphExt;
use axil_scip::{
    ingest_scip,
    proto::{self, symbol_role},
    EDGE_CALLS, EDGE_DEFINED_IN, EDGE_IMPLEMENTS, EDGE_REFERENCES, EDGE_TYPE_OF,
};
use prost::Message;

fn build_sample_index() -> proto::Index {
    // Two files, two symbols in Rust-style paths.
    //   auth.rs defines `login`; main.rs has a non-def reference to `login`.
    let login_sym = "rust-analyzer cargo myapp 0.1.0 auth/login().";
    let caller_sym = "rust-analyzer cargo myapp 0.1.0 main/run().";
    let user_sym = "rust-analyzer cargo myapp 0.1.0 user/User#";

    let auth_doc = proto::Document {
        language: "Rust".into(),
        relative_path: "src/auth.rs".into(),
        occurrences: vec![
            // Definition of login at line 3 col 0..20.
            proto::Occurrence {
                range: vec![3, 0, 20],
                symbol: login_sym.into(),
                symbol_roles: symbol_role::DEFINITION,
                enclosing_range: vec![3, 0, 10, 0],
                ..Default::default()
            },
        ],
        symbols: vec![proto::SymbolInformation {
            symbol: login_sym.into(),
            display_name: "login".into(),
            kind: 17,
            relationships: vec![proto::Relationship {
                symbol: user_sym.into(),
                is_reference: true,
                ..Default::default()
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    let main_doc = proto::Document {
        language: "Rust".into(),
        relative_path: "src/main.rs".into(),
        occurrences: vec![
            // Definition of run at lines 1..20.
            proto::Occurrence {
                range: vec![1, 0, 10],
                symbol: caller_sym.into(),
                symbol_roles: symbol_role::DEFINITION,
                enclosing_range: vec![1, 0, 20, 0],
                ..Default::default()
            },
            // Reference to login from inside run's body (line 5).
            proto::Occurrence {
                range: vec![5, 4, 15],
                symbol: login_sym.into(),
                symbol_roles: symbol_role::READ_ACCESS,
                ..Default::default()
            },
        ],
        symbols: vec![proto::SymbolInformation {
            symbol: caller_sym.into(),
            display_name: "run".into(),
            kind: 17,
            ..Default::default()
        }],
        ..Default::default()
    };

    proto::Index {
        metadata: Some(proto::Metadata {
            version: 0,
            tool_info: Some(proto::ToolInfo {
                name: "scip-rust".into(),
                version: "0.1.0".into(),
                arguments: vec![],
            }),
            project_root: "file:///tmp/myapp".into(),
            text_document_encoding: 1,
        }),
        documents: vec![auth_doc, main_doc],
        external_symbols: vec![proto::SymbolInformation {
            symbol: user_sym.into(),
            display_name: "User".into(),
            kind: 5,
            ..Default::default()
        }],
    }
}

fn open_graph_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn ingest_emits_direct_and_heuristic_edges() {
    let (db, dir) = open_graph_db();
    let index = build_sample_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let path = dir.path().join("index.scip");
    std::fs::write(&path, &buf).unwrap();

    let report = ingest_scip(&db, &path).unwrap();
    assert!(report.applied);
    assert_eq!(report.indexer_name, "scip-rust");
    // 2 defs across 2 documents.
    assert!(
        report.defined_in_edges >= 2,
        "got {}",
        report.defined_in_edges
    );
    // 1 references relationship (login -> User).
    assert!(
        report.references_edges >= 1,
        "got {}",
        report.references_edges
    );
    // 1 heuristic call (run -> login) via enclosing_range match.
    assert!(report.calls_edges >= 1, "got {}", report.calls_edges);
    // Entities created: 3 (login, run, User) minimum.
    assert!(
        report.entities_created >= 3,
        "got {}",
        report.entities_created
    );

    // Re-running the ingest must be idempotent: same edge counts, no dupes.
    let second = ingest_scip(&db, &path).unwrap();
    // Entities already exist, so creation count drops to zero.
    assert_eq!(
        second.entities_created, 0,
        "second ingest should not re-create entities"
    );
    // Edge counts are emitted operations — the store-side guard prevents
    // actual duplicate rows. Verify by counting via graph neighbors.
    let gi = db.graph_index_ref().unwrap();
    let rows = db.list("_entities").unwrap();
    for r in &rows {
        for etype in [
            EDGE_DEFINED_IN,
            EDGE_REFERENCES,
            EDGE_CALLS,
            EDGE_IMPLEMENTS,
            EDGE_TYPE_OF,
        ] {
            let edges = gi.edges(r.id.clone(), Some(etype), Direction::Out).unwrap();
            let mut seen = std::collections::HashSet::new();
            for e in &edges {
                assert!(
                    seen.insert((e.edge_type.clone(), e.to.clone())),
                    "duplicate edge after re-ingest: {}->{}",
                    e.edge_type,
                    e.to
                );
            }
        }
    }
}

#[test]
fn dry_run_does_not_write() {
    let (db, dir) = open_graph_db();
    let index = build_sample_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let path = dir.path().join("index.scip");
    std::fs::write(&path, &buf).unwrap();

    let report =
        axil_scip::ingest_scip_opts(&db, &path, axil_scip::IngestOptions { dry_run: true })
            .unwrap();
    assert!(!report.applied);
    assert!(report.defined_in_edges >= 2);
    // Nothing written to storage.
    let entities = db.list("_entities").unwrap();
    assert!(
        entities.is_empty(),
        "dry run wrote {} entities",
        entities.len()
    );
}

#[test]
fn ingest_into_graphless_db_errors_clearly() {
    // Regression for the silent-edge-drop bug: before this guard,
    // ingest_scip on a DB without a graph plugin returned Ok with an
    // IngestReport that claimed thousands of edges, while
    // relate_once silently no-op'd them all. Library callers had no
    // signal that their DB was misconfigured. The CLI now
    // auto-attaches via open_for_scip_ingest; this guard catches
    // direct library users who skip that path.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.axil");
    let db = Axil::open(&path).build().unwrap(); // no graph plugin

    let index = build_sample_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();

    let err = ingest_scip(&db, &scip_path).expect_err("must error without graph plugin");
    let msg = format!("{err}");
    assert!(
        msg.contains("graph plugin"),
        "error should mention the graph requirement, got: {msg}",
    );
}

#[test]
fn bulk_path_publishes_canonical_ids_to_atlas() {
    // Regression for /octo:review finding: the batched flush must
    // re-emit the Atlas canonical-id publish that `Axil::insert` does
    // per-record, otherwise SCIP-created entities would be invisible
    // to cross-project recall through the Atlas control plane.
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingPublisher {
        seen: Mutex<Vec<String>>,
    }
    impl axil_core::CanonicalPublisher for RecordingPublisher {
        fn publish(&self, canonical_id: &str) {
            self.seen.lock().unwrap().push(canonical_id.to_string());
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.axil");
    let publisher = Arc::new(RecordingPublisher::default());
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .with_canonical_publisher(publisher.clone() as Arc<dyn axil_core::CanonicalPublisher>)
        .build()
        .unwrap();
    // audit_enabled defaults to false; the test wants to verify the
    // bulk path actually emits audit entries when enabled.
    db.set_audit_enabled(true);

    let index = build_sample_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();

    let report = ingest_scip(&db, &scip_path).unwrap();
    assert!(report.entities_created >= 3);

    let seen = publisher.seen.lock().unwrap().clone();
    // The fixture has three SCIP symbols (login, run, User) — each
    // should have been published at least once during flush.
    assert!(
        seen.iter().any(|c| c.contains("auth/login")),
        "expected login canonical in publish stream, got: {seen:?}",
    );
    assert!(
        seen.iter().any(|c| c.contains("main/run")),
        "expected run canonical in publish stream, got: {seen:?}",
    );
    assert!(
        seen.iter().any(|c| c.contains("user/User")),
        "expected User canonical in publish stream, got: {seen:?}",
    );

    // The audit trail should also reflect the bulk-path inserts so
    // `axil audit-log` doesn't have a SCIP-shaped hole.
    let audit = db.audit_log(None, None, Some("_entities"), Some("insert"));
    assert!(
        audit.len() >= report.entities_created,
        "expected ≥{} audit entries for _entities, got {}",
        report.entities_created,
        audit.len(),
    );
}

#[test]
fn dry_run_into_graphless_db_is_allowed() {
    // dry_run doesn't write, so the graph requirement doesn't apply —
    // useful for "what would this index produce?" probing on a fresh
    // core-only DB before deciding to wire up the graph plugin.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.axil");
    let db = Axil::open(&path).build().unwrap();

    let index = build_sample_index();
    let mut buf = Vec::new();
    index.encode(&mut buf).unwrap();
    let scip_path = dir.path().join("index.scip");
    std::fs::write(&scip_path, &buf).unwrap();

    let report =
        axil_scip::ingest_scip_opts(&db, &scip_path, axil_scip::IngestOptions { dry_run: true })
            .expect("dry_run must succeed without graph plugin");
    assert!(!report.applied);
    assert!(report.defined_in_edges >= 2);
}
