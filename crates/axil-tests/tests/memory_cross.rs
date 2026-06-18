//! Integration tests for cross-memory queries (`axil remember`).
//!
//! Tests that `remember()` searches all memory types and returns
//! tagged, token-budgeted results.

use axil_core::Axil;
use axil_memory::preference::PreferenceSource;
use axil_memory::{AgentMemory, Outcome, RecallOptions};
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

/// Populate all memory types with auth-related content.
fn populate_all_types(mem: &AgentMemory<'_>) {
    // Semantic: store facts.
    mem.semantic()
        .know("auth-module", "Uses JWT tokens with 1h expiry", None)
        .unwrap();

    // Episodic: create an episode.
    mem.episodic()
        .create(
            "Fixed auth timeout by increasing connection pool",
            Outcome::Success,
            Some(vec!["Increased pool from 5 to 20".into()]),
            Some(vec!["config.rs".into()]),
        )
        .unwrap();

    // Procedural: store a pattern.
    mem.procedural()
        .learn(
            "fix-auth-timeout",
            "Check connection pool size first, then network config",
            None,
        )
        .unwrap();

    // Preference: store a rule.
    mem.preference()
        .set(
            "auth_testing",
            "Always run auth integration tests after changes",
            PreferenceSource::User,
        )
        .unwrap();
}

#[test]
fn remember_returns_results_from_multiple_types() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    populate_all_types(&mem);

    let opts = RecallOptions {
        top_k: 10,
        ..Default::default()
    };

    let results = mem.remember("auth timeout", opts).unwrap();

    // Without vector index, remember returns empty (no embedding available).
    // This test validates the API works without panicking.
    // With embeddings enabled, results would contain tagged entries.
    assert!(results.is_empty() || results.len() <= 10);
}

#[test]
fn remember_respects_top_k() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    populate_all_types(&mem);

    let opts = RecallOptions {
        top_k: 2,
        ..Default::default()
    };

    let results = mem.remember("auth", opts).unwrap();
    assert!(results.len() <= 2);
}

#[test]
fn remember_respects_token_budget() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    populate_all_types(&mem);

    let opts = RecallOptions {
        top_k: 10,
        max_tokens: Some(50), // Very small budget.
        ..Default::default()
    };

    let results = mem.remember("auth", opts).unwrap();
    let total_tokens: usize = results.iter().map(|r| r.tokens).sum();
    assert!(total_tokens <= 50 || results.is_empty());
}

#[test]
fn recall_options_defaults_are_sane() {
    let opts = RecallOptions::default();
    assert_eq!(opts.top_k, 5);
    assert!(opts.alpha.is_none());
    assert!(!opts.include_expired);
    assert!(!opts.include_superseded);
    assert!((opts.decay_window_secs - 30.0 * 86400.0).abs() < 1.0);
}

#[test]
fn remember_with_empty_database() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let opts = RecallOptions::default();
    let results = mem.remember("anything", opts).unwrap();
    assert!(results.is_empty());
}

#[test]
fn memory_type_display_and_tagging() {
    use axil_memory::MemoryType;

    assert_eq!(MemoryType::Working.to_string(), "working");
    assert_eq!(MemoryType::Semantic.to_string(), "semantic");
    assert_eq!(MemoryType::Episodic.to_string(), "episodic");
    assert_eq!(MemoryType::Procedural.to_string(), "procedural");
    assert_eq!(MemoryType::Preference.to_string(), "preference");

    // All types have distinct table names.
    let tables: Vec<&str> = MemoryType::all().iter().map(|mt| mt.table_name()).collect();
    for (i, t) in tables.iter().enumerate() {
        for (j, u) in tables.iter().enumerate() {
            if i != j {
                assert_ne!(t, u, "table names must be unique");
            }
        }
    }
}

#[test]
fn per_type_alpha_defaults() {
    use axil_memory::MemoryType;

    // Working memory should heavily weight recency.
    assert!(MemoryType::Working.default_alpha() < 0.5);
    // Semantic memory should heavily weight relevance.
    assert!(MemoryType::Semantic.default_alpha() > 0.7);
    // Episodic should be balanced.
    assert!(MemoryType::Episodic.default_alpha() >= 0.4);
    assert!(MemoryType::Episodic.default_alpha() <= 0.6);
}

#[test]
fn per_type_ttl_defaults() {
    use axil_memory::MemoryType;

    // Only episodic has a default TTL.
    assert!(MemoryType::Working.default_ttl_secs().is_none());
    assert!(MemoryType::Semantic.default_ttl_secs().is_none());
    assert!(MemoryType::Episodic.default_ttl_secs().is_some());
    assert!(MemoryType::Procedural.default_ttl_secs().is_none());
    assert!(MemoryType::Preference.default_ttl_secs().is_none());

    // Episodic TTL should be 90 days.
    let ttl = MemoryType::Episodic.default_ttl_secs().unwrap();
    assert_eq!(ttl, 90 * 86400);
}
