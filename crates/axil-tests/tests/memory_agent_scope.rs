//! Integration tests for agent-scoped memory over a real graph engine.
//!
//! Graph edges are writes too: an edge from one scope's record into another
//! agent's private record would surface that record to anyone traversing
//! from the first. These tests pin that auto-linking (semantic facts and
//! episodes) never crosses a private boundary.

use axil_core::{Axil, Direction, Record};
use axil_graph::AxilBuilderGraphExt;
use axil_memory::{AgentMemory, Outcome};
use tempfile::TempDir;

fn graph_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("scope.axil"))
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

fn agent_of(record: &Record) -> Option<&str> {
    record.data.get("_agent").and_then(|v| v.as_str())
}

#[test]
fn auto_links_never_reach_another_agents_private_fact() {
    let (db, _dir) = graph_db();
    let claude = AgentMemory::for_agent(&db, "claude");
    let codex = AgentMemory::for_agent(&db, "codex");

    codex
        .semantic()
        .know("secret-service", "internal only", None)
        .unwrap();
    let fact = claude
        .semantic()
        .know("gateway", "routes to secret-service", None)
        .unwrap();

    let linked = db.neighbors(&fact.id, None, Direction::Both).unwrap();
    assert!(
        linked.iter().all(|n| agent_of(n) != Some("codex")),
        "no edge may point into codex's private memory"
    );
    let about = claude.semantic().about("gateway").unwrap();
    assert!(!about
        .related_entities
        .contains(&"secret-service".to_string()));
}

#[test]
fn global_facts_never_link_to_private_facts() {
    let (db, _dir) = graph_db();
    let claude = AgentMemory::for_agent(&db, "claude");
    let codex = AgentMemory::for_agent(&db, "codex");

    claude
        .semantic()
        .know("secret-service", "internal only", None)
        .unwrap();
    let fact = AgentMemory::new(&db)
        .semantic()
        .know("gateway", "routes to secret-service", None)
        .unwrap();

    let linked = db.neighbors(&fact.id, None, Direction::Both).unwrap();
    assert!(linked.iter().all(|n| agent_of(n).is_none()));
    let about = codex.semantic().about("gateway").unwrap();
    assert!(!about
        .related_entities
        .contains(&"secret-service".to_string()));
}

#[test]
fn agent_facts_still_link_to_global_and_own_facts() {
    let (db, _dir) = graph_db();
    let claude = AgentMemory::for_agent(&db, "claude");

    AgentMemory::new(&db)
        .semantic()
        .know("postgres", "primary database", None)
        .unwrap();
    claude
        .semantic()
        .know("cache-layer", "fronts the session store", None)
        .unwrap();
    claude
        .semantic()
        .know(
            "api",
            "stores sessions in postgres behind cache-layer",
            None,
        )
        .unwrap();

    let related = claude.semantic().about("api").unwrap().related_entities;
    assert!(related.contains(&"postgres".to_string()));
    assert!(related.contains(&"cache-layer".to_string()));
}

#[test]
fn episodes_link_only_to_facts_their_agent_can_see() {
    let (db, _dir) = graph_db();
    let claude = AgentMemory::for_agent(&db, "claude");
    let codex = AgentMemory::for_agent(&db, "codex");

    codex
        .semantic()
        .know("secret-service", "internal only", None)
        .unwrap();
    AgentMemory::new(&db)
        .semantic()
        .know("postgres", "primary database", None)
        .unwrap();

    let episode = claude
        .episodic()
        .create(
            "Moved secret-service onto postgres",
            Outcome::Success,
            None,
            None,
        )
        .unwrap();

    let touched = db
        .neighbors(&episode.id, Some("touched"), Direction::Out)
        .unwrap();
    let entities: Vec<&str> = touched
        .iter()
        .filter_map(|n| n.data.get("entity").and_then(|v| v.as_str()))
        .collect();
    assert!(entities.contains(&"postgres"));
    assert!(!entities.contains(&"secret-service"));
}
