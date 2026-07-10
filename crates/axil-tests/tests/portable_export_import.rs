//! Round-trip integration tests for portable memory export/import.
//!
//! Populate a source database (records + a graph edge + a manually attached
//! vector standing in for an embedded field), export to JSONL, import into a
//! fresh database, and assert the content survives: counts match, FTS finds an
//! imported record, edge traversal works, and a second `--dedup` import is a
//! no-op. Also covers content-hash dedup across differing ids and the
//! embeddings-do-not-travel invariant.

use axil_core::{Axil, Direction, ExportOptions, ImportOptions};
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

/// A full database: vector + graph + FTS, no ONNX embedder (tests attach
/// vectors manually, so no model download is needed).
fn full_db(dims: usize) -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_vector(dims)
        .unwrap()
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn round_trip_preserves_records_edges_and_findability() {
    let (db_a, _dir_a) = full_db(3);

    // Records across two user tables.
    let decision = db_a
        .insert(
            "decisions",
            json!({"summary": "adopt JSONL export", "reason": "portable across machines"}),
        )
        .unwrap();
    let file = db_a
        .insert("files", json!({"path": "portable.rs"}))
        .unwrap();
    let err = db_a
        .insert(
            "errors",
            json!({"error": "auth timeout", "fix": "raise deadline"}),
        )
        .unwrap();

    // A graph edge between two exported records.
    db_a.relate(&decision.id, "modified", &file.id, None)
        .unwrap();

    // A manually attached vector — the "embedded field" that must NOT travel.
    db_a.add_vector(&decision.id, &[0.1, 0.2, 0.3]).unwrap();

    // Export to an in-memory buffer.
    let mut buf: Vec<u8> = Vec::new();
    let stats = axil_core::export_to_writer(&db_a, &ExportOptions::default(), &mut buf).unwrap();
    assert_eq!(stats.records, 3, "3 user records exported");
    assert_eq!(stats.edges, 1, "1 inter-record edge exported");

    let text = String::from_utf8(buf.clone()).unwrap();
    // First line is a header naming the format.
    let first = text.lines().next().unwrap();
    assert!(first.contains("\"kind\":\"header\""));
    assert!(first.contains("axil-export-jsonl"));
    // Embeddings do not travel: no raw vector floats leak into the JSONL.
    assert!(
        !text.contains("0.2,0.3") && !text.contains("\"vector\""),
        "export must not carry embedding vectors"
    );

    // Import into a fresh database.
    let (db_b, _dir_b) = full_db(3);
    let report =
        axil_core::import_from_reader(&db_b, &ImportOptions::default(), buf.as_slice()).unwrap();
    assert_eq!(report.imported, 3);
    assert_eq!(report.edges_created, 1);
    assert_eq!(report.skipped_id, 0);
    assert_eq!(report.skipped_dup, 0);

    // Counts match, per table.
    assert_eq!(db_b.count("decisions").unwrap(), 1);
    assert_eq!(db_b.count("files").unwrap(), 1);
    assert_eq!(db_b.count("errors").unwrap(), 1);

    // Ids were preserved (checkpoint refs / code_refs depend on this).
    assert!(db_b.get(&decision.id).unwrap().is_some());
    assert!(db_b.get(&file.id).unwrap().is_some());
    assert!(db_b.get(&err.id).unwrap().is_some());

    // FTS finds an imported record in B (indexed on insert during import).
    let hits = db_b.search_text("timeout", 10).unwrap();
    assert!(
        hits.iter().any(|(r, _)| r.id == err.id),
        "FTS should find the imported error record"
    );

    // Edge traversal works in B.
    let neighbors = db_b
        .neighbors(&decision.id, Some("modified"), Direction::Out)
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, file.id);

    // A second import with --dedup is a no-op: 0 records, 0 new edges.
    let opts = ImportOptions {
        dedup: true,
        dry_run: false,
    };
    let report2 = axil_core::import_from_reader(&db_b, &opts, buf.as_slice()).unwrap();
    assert_eq!(report2.imported, 0, "re-import must import nothing");
    assert_eq!(report2.skipped_id, 3, "all 3 skipped by existing id");
    assert_eq!(report2.edges_created, 0, "no duplicate edges");

    // No duplicates were created.
    assert_eq!(db_b.count("decisions").unwrap(), 1);
    assert_eq!(db_b.count("files").unwrap(), 1);
    assert_eq!(db_b.count("errors").unwrap(), 1);
}

#[test]
fn dedup_matches_content_across_differing_ids() {
    let (db_a, _dir_a) = full_db(3);
    db_a.insert("decisions", json!({"summary": "shared decision"}))
        .unwrap();

    let mut buf: Vec<u8> = Vec::new();
    axil_core::export_to_writer(&db_a, &ExportOptions::default(), &mut buf).unwrap();

    // Destination already holds the SAME content under a different id.
    let (db_b, _dir_b) = full_db(3);
    db_b.insert("decisions", json!({"summary": "shared decision"}))
        .unwrap();

    let opts = ImportOptions {
        dedup: true,
        dry_run: false,
    };
    let report = axil_core::import_from_reader(&db_b, &opts, buf.as_slice()).unwrap();
    assert_eq!(report.imported, 0);
    assert_eq!(report.skipped_dup, 1, "matched by content hash, not id");
    assert_eq!(db_b.count("decisions").unwrap(), 1);
}

#[test]
fn dry_run_reports_without_writing() {
    let (db_a, _dir_a) = full_db(3);
    db_a.insert("decisions", json!({"summary": "one"})).unwrap();
    db_a.insert("decisions", json!({"summary": "two"})).unwrap();

    let mut buf: Vec<u8> = Vec::new();
    axil_core::export_to_writer(&db_a, &ExportOptions::default(), &mut buf).unwrap();

    let (db_b, _dir_b) = full_db(3);
    let opts = ImportOptions {
        dedup: false,
        dry_run: true,
    };
    let report = axil_core::import_from_reader(&db_b, &opts, buf.as_slice()).unwrap();
    assert_eq!(report.imported, 2, "dry run reports what would import");
    assert_eq!(db_b.count("decisions").unwrap(), 0, "but nothing is written");
}

#[test]
fn since_filter_limits_exported_records() {
    let (db_a, _dir_a) = full_db(3);
    // Insert with an explicit old timestamp, and one with a recent timestamp.
    let old_ts = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    db_a.insert_at("decisions", json!({"summary": "old"}), old_ts)
        .unwrap();
    db_a.insert("decisions", json!({"summary": "new"})).unwrap();

    let cutoff = chrono::DateTime::parse_from_rfc3339("2021-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let opts = ExportOptions {
        since: Some(cutoff),
        ..Default::default()
    };
    let mut buf: Vec<u8> = Vec::new();
    let stats = axil_core::export_to_writer(&db_a, &opts, &mut buf).unwrap();
    assert_eq!(stats.records, 1, "only the record after the cutoff");
}

#[test]
fn default_export_excludes_system_tables() {
    let (db_a, _dir_a) = full_db(3);
    db_a.insert("decisions", json!({"summary": "user memory"}))
        .unwrap();
    // A system-prefixed table is written directly.
    db_a.insert("_entities", json!({"name": "auth"})).unwrap();

    let mut buf: Vec<u8> = Vec::new();
    let stats = axil_core::export_to_writer(&db_a, &ExportOptions::default(), &mut buf).unwrap();
    assert_eq!(stats.records, 1, "default export skips `_`-prefixed tables");

    // With include_system, the system table travels too.
    let opts = ExportOptions {
        include_system: true,
        ..Default::default()
    };
    let mut buf2: Vec<u8> = Vec::new();
    let stats2 = axil_core::export_to_writer(&db_a, &opts, &mut buf2).unwrap();
    assert_eq!(stats2.records, 2);
}

/// Deterministic embedder for verification tests — real ONNX is unavailable
/// in CI, and the point here is *whether* vectors land, not their quality.
struct FakeEmbedder;

impl axil_core::TextEmbedder for FakeEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let len = text.len() as f32;
        Ok(vec![len, len / 2.0, 1.0])
    }
}

/// Export two embeddable records, then import them into a destination that
/// has a working (fake) embedder attached.
fn export_two_records() -> Vec<u8> {
    let (db_a, _dir_a) = full_db(3);
    db_a.insert("decisions", json!({"summary": "adopt the fake embedder"}))
        .unwrap();
    db_a.insert("errors", json!({"error": "auth timeout on refresh"}))
        .unwrap();
    let mut buf: Vec<u8> = Vec::new();
    axil_core::export_to_writer(&db_a, &ExportOptions::default(), &mut buf).unwrap();
    // _dir_a drops here; the buffer is all we need.
    buf
}

#[test]
fn import_verifies_embeddings_when_embedder_present() {
    let buf = export_two_records();

    let dir = tempfile::tempdir().unwrap();
    let db_b = Axil::open(dir.path().join("b.axil"))
        .with_vector(3)
        .unwrap()
        .with_embedder(Box::new(FakeEmbedder))
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();

    let report =
        axil_core::import_from_reader(&db_b, &ImportOptions::default(), buf.as_slice()).unwrap();
    assert_eq!(report.imported, 2);
    assert_eq!(
        report.embeddings,
        Some(axil_core::EmbeddingVerification::Verified {
            expected: 2,
            indexed: 2,
            missing: 0
        }),
        "with a working embedder every imported record must be verified indexed"
    );
}

#[test]
fn import_flags_missing_embeddings_when_embedder_is_broken() {
    let buf = export_two_records();

    // `with_vector` attaches the VectorEngine as index AND embedder, but with
    // no ONNX model available every embed call fails — the exact
    // stored-but-unembedded state a broken runtime produces in the field.
    // The report must count those records as missing, not stay silent.
    let (db_b, _dir_b) = full_db(3);
    let report =
        axil_core::import_from_reader(&db_b, &ImportOptions::default(), buf.as_slice()).unwrap();
    assert_eq!(report.imported, 2);
    assert_eq!(
        report.embeddings,
        Some(axil_core::EmbeddingVerification::Verified {
            expected: 2,
            indexed: 0,
            missing: 2
        }),
        "a present-but-failing embedder must surface as missing embeddings"
    );
}

#[test]
fn import_reports_engine_unavailable_without_vector_engine() {
    let buf = export_two_records();

    // Destination has no vector engine at all (e.g. `[engines] disabled`).
    let dir = tempfile::tempdir().unwrap();
    let db_b = Axil::open(dir.path().join("novec.axil"))
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    let report =
        axil_core::import_from_reader(&db_b, &ImportOptions::default(), buf.as_slice()).unwrap();
    assert_eq!(report.imported, 2);
    assert_eq!(
        report.embeddings,
        Some(axil_core::EmbeddingVerification::EngineUnavailable { affected: 2 })
    );

    // A dry run writes nothing, so there is nothing to verify.
    let (db_c, _dir_c) = full_db(3);
    let dry = ImportOptions {
        dry_run: true,
        ..Default::default()
    };
    let report = axil_core::import_from_reader(&db_c, &dry, buf.as_slice()).unwrap();
    assert_eq!(
        report.embeddings,
        Some(axil_core::EmbeddingVerification::SkippedDryRun)
    );
}
