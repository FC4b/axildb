//! Integration tests for auto-discovery of entity relationships.
//!
//! When two entities are mentioned in the same fact, `know()` auto-creates
//! `related_to` graph edges between them.

use axil_core::Axil;
use axil_graph::AxilBuilderGraphExt;
use axil_memory::AgentMemory;
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

// ── Basic auto-discovery ─────────────────────────────────────────────────

#[test]
fn co_mention_creates_related_to_edge() {
    let (db, _dir) = temp_db_with_graph();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    // Store facts about two entities.
    sem.know("auth-module", "handles JWT login", None).unwrap();
    // Mention auth-module in a fact about session-manager.
    sem.know(
        "session-manager",
        "integrates with auth-module for validation",
        None,
    )
    .unwrap();

    // Check that related entities are discovered.
    let knowledge = sem.about("session-manager").unwrap();
    assert!(
        knowledge
            .related_entities
            .iter()
            .any(|e| e.contains("auth-module")),
        "expected auth-module in related entities: {:?}",
        knowledge.related_entities
    );
}

#[test]
fn no_self_relationship() {
    let (db, _dir) = temp_db_with_graph();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    // Mention the entity in its own fact — should not create self-edge.
    sem.know("auth-module", "auth-module handles login", None)
        .unwrap();

    let knowledge = sem.about("auth-module").unwrap();
    // Should not list itself as related.
    assert!(
        !knowledge
            .related_entities
            .iter()
            .any(|e| e == "auth-module"),
        "entity should not be related to itself: {:?}",
        knowledge.related_entities
    );
}

#[test]
fn discovery_requires_graph_index() {
    // Without graph, know() still works — just no edge creation.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("auth-module", "handles login", None).unwrap();
    sem.know("session-manager", "uses auth-module", None)
        .unwrap();

    let knowledge = sem.about("session-manager").unwrap();
    // Without graph, related_entities should be empty.
    assert!(
        knowledge.related_entities.is_empty(),
        "no graph = no related entities"
    );
}

#[test]
fn multiple_co_mentions_across_facts() {
    let (db, _dir) = temp_db_with_graph();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.know("database", "stores user records", None).unwrap();
    sem.know("api-server", "connects to database", None)
        .unwrap();
    sem.know("cache-layer", "sits between api-server and database", None)
        .unwrap();

    let knowledge = sem.about("cache-layer").unwrap();
    // cache-layer should be related to at least api-server or database.
    assert!(
        !knowledge.related_entities.is_empty(),
        "expected related entities for cache-layer: {:?}",
        knowledge.related_entities
    );
}
