//! Integration tests for semantic memory (knowledge graph).

use axil_core::Axil;
use axil_memory::AgentMemory;
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn know_stores_fact() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .semantic()
        .know("auth-module", "Uses JWT tokens with 1h expiry", None)
        .unwrap();

    assert_eq!(record.data["entity"], "auth-module");
    assert_eq!(record.data["fact"], "Uses JWT tokens with 1h expiry");
    // Should have bi-temporal metadata.
    assert!(record.data.get("_meta").is_some());
}

#[test]
fn know_with_source() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    let record = mem
        .semantic()
        .know("auth-module", "fact1", Some("session-123"))
        .unwrap();

    assert_eq!(record.data["source"], "session-123");
}

#[test]
fn about_returns_entity_knowledge() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic()
        .know("auth-module", "Uses JWT tokens", None)
        .unwrap();
    mem.semantic()
        .know("auth-module", "Handles login and logout", None)
        .unwrap();
    mem.semantic()
        .know("user-table", "PostgreSQL table", None)
        .unwrap();

    let knowledge = mem.semantic().about("auth-module").unwrap();
    assert_eq!(knowledge.entity, "auth-module");
    assert_eq!(knowledge.facts.len(), 2);

    // JSON output should serialize correctly.
    let json = knowledge.to_json();
    assert_eq!(json["fact_count"], 2);
}

#[test]
fn list_entities_deduplicates() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic().know("auth", "fact1", None).unwrap();
    mem.semantic().know("db", "fact2", None).unwrap();
    mem.semantic().know("auth", "fact3", None).unwrap();

    let entities = mem.semantic().list_entities().unwrap();
    assert_eq!(entities, vec!["auth", "db"]);
}

#[test]
fn list_facts_filtered_by_entity() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic().know("auth", "fact1", None).unwrap();
    mem.semantic().know("db", "fact2", None).unwrap();

    let all = mem.semantic().list_facts(None).unwrap();
    assert_eq!(all.len(), 2);

    let auth_only = mem.semantic().list_facts(Some("auth")).unwrap();
    assert_eq!(auth_only.len(), 1);
    assert_eq!(auth_only[0].data["entity"], "auth");
}

#[test]
fn history_returns_all_versions() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);

    mem.semantic()
        .know("auth", "Uses session cookies", None)
        .unwrap();
    mem.semantic()
        .know("auth", "Migrated to JWT", None)
        .unwrap();

    let history = mem.semantic().history("auth").unwrap();
    assert_eq!(history.len(), 2);
    // Sorted by created_at ascending.
    assert_eq!(history[0].data["fact"], "Uses session cookies");
    assert_eq!(history[1].data["fact"], "Migrated to JWT");
}
