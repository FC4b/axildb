//! Integration tests for graph inference engine.
//!
//! Tests transitive inference, reasoning chains, and why-fact explanations.

use axil_core::{Axil, InferenceEngine};
use axil_graph::AxilBuilderGraphExt;
use axil_memory::types::TABLE_ENTITIES;
use axil_memory::AgentMemory;
use serde_json::json;
use tempfile::TempDir;

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

// ── Basic inference ──────────────────────────────────────────────────────

#[test]
fn infer_on_empty_db_returns_empty() {
    let (db, _dir) = temp_db_with_graph();
    let engine = InferenceEngine::new(&db);

    let facts = engine.infer(None).unwrap();
    assert!(facts.is_empty());
}

#[test]
fn transitive_inference_manages_part_of() {
    let (db, _dir) = temp_db_with_graph();

    // Create entities.
    let sarah = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "Sarah", "fact": "manages Project Atlas"}),
        )
        .unwrap();
    let atlas = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "Project Atlas", "fact": "infrastructure project"}),
        )
        .unwrap();
    let infra = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "Infrastructure", "fact": "engineering team"}),
        )
        .unwrap();

    // Create edges: Sarah ->manages-> Atlas, Atlas ->part_of-> Infrastructure.
    db.relate(&sarah.id, "manages", &atlas.id, None).unwrap();
    db.relate(&atlas.id, "part_of", &infra.id, None).unwrap();

    let engine = InferenceEngine::new(&db);
    let facts = engine.infer(None).unwrap();

    // Should infer: Sarah ->works_in-> Infrastructure.
    let has_works_in = facts.iter().any(|f| f.derived_edge == "works_in");
    assert!(
        has_works_in,
        "expected works_in inference, got: {:?}",
        facts.iter().map(|f| &f.derived_edge).collect::<Vec<_>>()
    );
}

#[test]
fn inferred_fact_has_reasoning_chain() {
    let (db, _dir) = temp_db_with_graph();

    let a = db
        .insert(TABLE_ENTITIES, json!({"entity": "A", "fact": "entity A"}))
        .unwrap();
    let b = db
        .insert(TABLE_ENTITIES, json!({"entity": "B", "fact": "entity B"}))
        .unwrap();
    let c = db
        .insert(TABLE_ENTITIES, json!({"entity": "C", "fact": "entity C"}))
        .unwrap();

    db.relate(&a.id, "manages", &b.id, None).unwrap();
    db.relate(&b.id, "part_of", &c.id, None).unwrap();

    let engine = InferenceEngine::new(&db);
    let facts = engine.infer(None).unwrap();

    let fact = facts
        .iter()
        .find(|f| f.derived_edge == "works_in")
        .expect("manages + part_of should trigger transitive_ownership → works_in");
    assert!(
        !fact.reasoning.is_empty(),
        "reasoning chain should not be empty"
    );
    assert!(fact.confidence > 0.0, "confidence should be positive");
    assert_eq!(fact.source, "inferred");
}

// ── Infer and store ──────────────────────────────────────────────────────

#[test]
fn infer_and_store_creates_records() {
    let (db, _dir) = temp_db_with_graph();

    let a = db
        .insert(TABLE_ENTITIES, json!({"entity": "A", "fact": "entity A"}))
        .unwrap();
    let b = db
        .insert(TABLE_ENTITIES, json!({"entity": "B", "fact": "entity B"}))
        .unwrap();
    let c = db
        .insert(TABLE_ENTITIES, json!({"entity": "C", "fact": "entity C"}))
        .unwrap();

    db.relate(&a.id, "manages", &b.id, None).unwrap();
    db.relate(&b.id, "part_of", &c.id, None).unwrap();

    let engine = InferenceEngine::new(&db);
    let stored = engine.infer_and_store(None).unwrap();
    assert!(
        !stored.is_empty(),
        "manages + part_of should produce inferred facts"
    );

    let all_entities = db.list(TABLE_ENTITIES).unwrap();
    let inferred: Vec<_> = all_entities
        .iter()
        .filter(|r| r.data.get("source").and_then(|v| v.as_str()) == Some("inferred"))
        .collect();
    assert!(
        !inferred.is_empty(),
        "expected stored inferred facts in _entities"
    );
}

// ── Why-fact ─────────────────────────────────────────────────────────────

#[test]
fn why_fact_returns_reasoning_for_inferred() {
    let (db, _dir) = temp_db_with_graph();

    let a = db
        .insert(TABLE_ENTITIES, json!({"entity": "X", "fact": "entity X"}))
        .unwrap();
    let b = db
        .insert(TABLE_ENTITIES, json!({"entity": "Y", "fact": "entity Y"}))
        .unwrap();
    let c = db
        .insert(TABLE_ENTITIES, json!({"entity": "Z", "fact": "entity Z"}))
        .unwrap();

    db.relate(&a.id, "manages", &b.id, None).unwrap();
    db.relate(&b.id, "part_of", &c.id, None).unwrap();

    let engine = InferenceEngine::new(&db);
    let stored = engine.infer_and_store(None).unwrap();
    assert!(!stored.is_empty(), "should produce inferred facts");

    let all = db.list(TABLE_ENTITIES).unwrap();
    let inferred_record = all
        .iter()
        .find(|r| r.data.get("source").and_then(|v| v.as_str()) == Some("inferred"))
        .expect("should find stored inferred record");

    let why = engine.why(&inferred_record.id).unwrap();
    let chain = why.expect("why() should return reasoning for inferred fact");
    assert!(!chain.is_empty(), "reasoning chain should not be empty");
}

#[test]
fn why_fact_returns_none_for_non_inferred() {
    let (db, _dir) = temp_db_with_graph();

    let record = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "manual", "fact": "manually stored"}),
        )
        .unwrap();

    let engine = InferenceEngine::new(&db);
    let why = engine.why(&record.id).unwrap();
    assert!(
        why.is_none(),
        "why() should return None for non-inferred facts"
    );
}
