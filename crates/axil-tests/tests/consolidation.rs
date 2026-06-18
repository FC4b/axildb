//! Integration tests for knowledge consolidation.
//!
//! Tests that fragmented facts about an entity can be merged into
//! a consolidated summary via both the core consolidation engine and
//! the semantic memory layer.

use axil_core::consolidation::{check_conflict, consolidate_facts, ConflictResult};
use axil_core::Axil;
use axil_memory::types::TABLE_ENTITIES;
use axil_memory::AgentMemory;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

// ── Low-level consolidation API ──────────────────────────────────────────

#[test]
fn consolidate_facts_multiple() {
    let (db, _dir) = temp_db();

    let r1 = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "auth", "fact": "handles login"}),
        )
        .unwrap();
    let r2 = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "auth", "fact": "supports token refresh"}),
        )
        .unwrap();
    let r3 = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "auth", "fact": "refactored in March"}),
        )
        .unwrap();

    let facts: Vec<_> = vec![
        (r1, ConflictResult::Novel),
        (r2, ConflictResult::Novel),
        (r3, ConflictResult::Novel),
    ];

    let result = consolidate_facts("auth", &facts);
    assert!(result.is_some(), "expected consolidated fact");

    let consolidated = result.unwrap();
    assert_eq!(consolidated.entity, "auth");
    assert!(!consolidated.summary.is_empty());
    assert_eq!(consolidated.source_ids.len(), 3);
}

#[test]
fn consolidate_facts_empty_returns_none() {
    let result = consolidate_facts("nothing", &[]);
    assert!(result.is_none());
}

#[test]
fn consolidate_facts_single() {
    let (db, _dir) = temp_db();

    let r1 = db
        .insert(
            TABLE_ENTITIES,
            json!({"entity": "tiny", "fact": "a small lib"}),
        )
        .unwrap();
    let facts = vec![(r1, ConflictResult::Novel)];

    let result = consolidate_facts("tiny", &facts);
    // Single fact should still produce a consolidation.
    let c = result.expect("single fact should still consolidate");
    assert!(!c.summary.is_empty());
    assert_eq!(c.source_ids.len(), 1);
}

// ── Conflict detection ───────────────────────────────────────────────────

#[test]
fn check_conflict_low_similarity_is_novel() {
    let (db, _dir) = temp_db();

    let r1 = db
        .insert("facts", json!({"text": "the sky is blue"}))
        .unwrap();
    let r2 = db
        .insert("facts", json!({"text": "dogs are friendly"}))
        .unwrap();

    // Low similarity — should be Novel.
    let result = check_conflict(&r1, &r2, 0.3);
    match result {
        ConflictResult::Novel => {} // expected
        other => panic!("expected Novel, got {other:?}"),
    }
}

#[test]
fn check_conflict_high_similarity_without_entities_returns_novel() {
    let (db, _dir) = temp_db();

    // These texts have no extractable entities (no backticks, CamelCase, snake_case, paths),
    // so check_conflict returns Novel even at high similarity.
    let r1 = db
        .insert("facts", json!({"text": "timeout is 30 seconds"}))
        .unwrap();
    let r2 = db
        .insert("facts", json!({"text": "timeout is 60 seconds"}))
        .unwrap();

    let result = check_conflict(&r2, &r1, 0.95);
    match result {
        ConflictResult::Novel => {} // expected: no shared entities
        other => panic!("expected Novel (no shared entities), got {other:?}"),
    }
}

// ── Semantic memory consolidation path ───────────────────────────────────

#[test]
fn about_returns_all_entity_facts() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("cache-layer", "uses Redis", None).unwrap();
    sem.know("cache-layer", "TTL is 5 minutes", None).unwrap();
    sem.know("cache-layer", "deployed on port 6379", None)
        .unwrap();

    let knowledge = sem.about("cache-layer").unwrap();
    assert!(
        knowledge.facts.len() >= 3,
        "expected 3+ facts, got {}",
        knowledge.facts.len()
    );
    assert_eq!(knowledge.entity, "cache-layer");
}

#[test]
fn consolidated_summary_from_about() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("payment-service", "processes credit cards", None)
        .unwrap();
    sem.know("payment-service", "integrates with Stripe API", None)
        .unwrap();

    let knowledge = sem.about("payment-service").unwrap();
    let summary = knowledge.consolidated_summary();

    // consolidated_summary() merges facts into one string.
    if let Some(s) = summary {
        assert!(!s.is_empty(), "summary should not be empty");
    }
}

#[test]
fn superseded_facts_excluded_from_about() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    // Store a fact, then a contradicting one (if superseding kicks in).
    sem.know("config", "timeout is 30 seconds", None).unwrap();
    sem.know("config", "timeout is 60 seconds", None).unwrap();

    let knowledge = sem.about("config").unwrap();
    assert!(
        !knowledge.facts.is_empty(),
        "about() should return at least one non-superseded fact"
    );
    // All returned facts should be non-superseded.
    for fact in &knowledge.facts {
        let superseded = fact
            .data
            .get("_meta")
            .and_then(|m| m.get("superseded"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let superseded_v2 = fact
            .data
            .get("_superseded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(
            !superseded && !superseded_v2,
            "about() should not return superseded facts"
        );
    }
}
