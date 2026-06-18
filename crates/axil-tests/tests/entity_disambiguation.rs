//! Integration tests for entity alias resolution and disambiguation.

use axil_core::Axil;
use axil_memory::AgentMemory;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

// ── Alias registration ──────────────────────────────────────────────────

#[test]
fn add_and_resolve_alias() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("Sarah", "the VP of Engineering").unwrap();

    let resolved = sem.resolve("the VP of Engineering").unwrap();
    assert_eq!(resolved, Some("Sarah".to_string()));
}

#[test]
fn resolve_unknown_name_returns_none() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    let resolved = sem.resolve("nobody").unwrap();
    assert_eq!(resolved, None);
}

#[test]
fn multiple_aliases_same_entity() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("Sarah", "my manager").unwrap();
    sem.add_alias("Sarah", "the VP").unwrap();

    assert_eq!(
        sem.resolve("my manager").unwrap(),
        Some("Sarah".to_string())
    );
    assert_eq!(sem.resolve("the VP").unwrap(), Some("Sarah".to_string()));

    let aliases = sem.aliases("Sarah").unwrap();
    assert_eq!(aliases.len(), 2);
    assert!(aliases.contains(&"my manager".to_string()));
    assert!(aliases.contains(&"the VP".to_string()));
}

#[test]
fn self_alias_rejected() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    let result = sem.add_alias("Sarah", "Sarah");
    assert!(result.is_err(), "self-alias should be rejected");
}

#[test]
fn conflicting_alias_rejected() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("Sarah", "the boss").unwrap();
    let result = sem.add_alias("Mike", "the boss");
    assert!(result.is_err(), "conflicting alias should be rejected");
}

#[test]
fn duplicate_alias_is_noop() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("Sarah", "VP").unwrap();
    sem.add_alias("Sarah", "VP").unwrap(); // should not error

    let aliases = sem.aliases("Sarah").unwrap();
    assert_eq!(aliases.len(), 1);
}

// ── Case-insensitive resolution ─────────────────────────────────────────

#[test]
fn resolve_case_insensitive() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("AuthModule", "auth system").unwrap();

    // Case-insensitive lookup should work.
    let resolved = sem.resolve("Auth System").unwrap();
    assert_eq!(resolved, Some("AuthModule".to_string()));
}

// ── Alias integration with know() ───────────────────────────────────────

#[test]
fn know_resolves_alias_to_canonical() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("auth-module", "the auth thing").unwrap();

    // Store a fact using the alias name.
    let record = sem
        .know("the auth thing", "handles JWT refresh", None)
        .unwrap();

    // The stored entity should be the canonical name.
    let entity = record.data.get("entity").and_then(|v| v.as_str());
    assert_eq!(entity, Some("auth-module"));
}

#[test]
fn about_returns_facts_for_canonical_name() {
    let (db, _dir) = temp_db();
    let mem = AgentMemory::new(&db);
    let sem = mem.semantic();

    sem.add_alias("auth-module", "auth service").unwrap();

    sem.know("auth service", "handles login", None).unwrap();
    sem.know("auth-module", "handles token refresh", None)
        .unwrap();

    let knowledge = sem.about("auth-module").unwrap();
    assert!(
        knowledge.facts.len() >= 2,
        "expected 2+ facts, got {}",
        knowledge.facts.len()
    );
}
