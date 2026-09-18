//! Structured agent memory types for Axil.
//!
//! Provides five distinct memory types purpose-built for AI agents:
//! - **Working** — current session context, auto-cleared on session end
//! - **Semantic** — facts, entities, relationships (knowledge graph)
//! - **Episodic** — past sessions, interactions, outcomes
//! - **Procedural** — learned patterns, strategies, tool usage
//! - **Preference** — user preferences, feedback, rules, conventions
//!
//! Plus cross-cutting features: TTL/expiry, memory superseding,
//! recency-weighted recall, and cross-memory queries.

pub mod episodic;
pub mod patterns;
pub mod preference;
pub mod procedural;
pub mod recall;
pub mod reflect;
pub mod semantic;
pub mod session;
pub mod supersede;
pub mod ttl;
pub mod types;

pub use episodic::EpisodicMemory;
pub use patterns::{Pattern, PatternEngine, PatternType};
pub use preference::PreferenceMemory;
pub use procedural::ProceduralMemory;
pub use recall::{RecallOptions, RecallResult, ScoredRecord};
pub use reflect::{ReflectEngine, ReflectReport, ReflectScope};
pub use semantic::{
    DisambiguationOptions, DisambiguationStrategy, EntityMatch, MatchMethod, SemanticMemory,
};
pub use session::WorkingMemory;
pub use supersede::SupersedeEngine;
pub use ttl::TtlEngine;
pub use types::{MemoryType, Outcome, MEMORY_TABLES};

/// The main entry point for agent memory operations.
///
/// Wraps an `Axil` handle and provides memory-type-specific APIs
/// that orchestrate the underlying storage, vector, graph, and
/// time-series plugins.
///
/// ## Multi-agent memory model
///
/// When `agent` is set (via `for_agent()`), memory types are split:
///
/// - **Per-agent** (isolated): Working memory, sessions, episodic memory
/// - **Shared** (all agents): Semantic, procedural, preference memory
///
/// This lets multiple agents share a knowledge base while keeping
/// separate session histories and working contexts.
pub struct AgentMemory<'a> {
    db: &'a axil_core::Axil,
    agent: Option<String>,
}

impl<'a> AgentMemory<'a> {
    /// Create a new agent memory layer over an existing database.
    pub fn new(db: &'a axil_core::Axil) -> Self {
        Self { db, agent: None }
    }

    /// Create an agent memory scoped to a specific agent.
    ///
    /// Working memory and sessions are isolated per agent. Semantic,
    /// procedural, and preference writes are stamped with the agent name,
    /// and reads see the agent's records plus unscoped (global) ones —
    /// memory that predates scoping stays visible to everyone.
    pub fn for_agent(db: &'a axil_core::Axil, agent: &str) -> Self {
        Self {
            db,
            agent: Some(agent.to_string()),
        }
    }

    /// Access the underlying database handle.
    pub fn db(&self) -> &axil_core::Axil {
        self.db
    }

    /// The agent name, if this memory is agent-scoped.
    pub fn agent_name(&self) -> Option<&str> {
        self.agent.as_deref()
    }

    /// Working memory (session context).
    ///
    /// Per-agent: if `agent` is set, sessions are isolated per agent.
    pub fn working(&self) -> WorkingMemory<'_> {
        match &self.agent {
            Some(name) => WorkingMemory::for_agent(self.db, name),
            None => WorkingMemory::new(self.db),
        }
    }

    /// Semantic memory (knowledge graph).
    ///
    /// Per-agent: if `agent` is set, writes are stamped with the agent and
    /// reads see the agent's facts plus unscoped (global) ones.
    pub fn semantic(&self) -> SemanticMemory<'_> {
        match &self.agent {
            Some(name) => SemanticMemory::for_agent(self.db, name),
            None => SemanticMemory::new(self.db),
        }
    }

    /// Episodic memory (past experiences).
    ///
    /// Per-agent: if `agent` is set, episodes are filtered per agent.
    pub fn episodic(&self) -> EpisodicMemory<'_> {
        match &self.agent {
            Some(name) => EpisodicMemory::for_agent(self.db, name),
            None => EpisodicMemory::new(self.db),
        }
    }

    /// Procedural memory (learned patterns).
    ///
    /// Per-agent: same stamping/visibility rules as semantic memory.
    pub fn procedural(&self) -> ProceduralMemory<'_> {
        match &self.agent {
            Some(name) => ProceduralMemory::for_agent(self.db, name),
            None => ProceduralMemory::new(self.db),
        }
    }

    /// Preference memory (rules & feedback).
    ///
    /// Per-agent: same stamping/visibility rules as semantic memory.
    pub fn preference(&self) -> PreferenceMemory<'_> {
        match &self.agent {
            Some(name) => PreferenceMemory::for_agent(self.db, name),
            None => PreferenceMemory::new(self.db),
        }
    }

    /// TTL / expiry engine.
    pub fn ttl(&self) -> TtlEngine<'_> {
        TtlEngine::new(self.db)
    }

    /// Supersede engine.
    pub fn supersede(&self) -> SupersedeEngine<'_> {
        SupersedeEngine::new(self.db)
    }

    /// Cross-memory recall: searches all memory types, returns tagged results.
    ///
    /// Agent-scoped memories restrict results to their agent's records plus
    /// unscoped (global) ones.
    pub fn remember(
        &self,
        query: &str,
        mut opts: RecallOptions,
    ) -> axil_core::Result<Vec<RecallResult>> {
        if opts.agent.is_none() {
            opts.agent = self.agent.clone();
        }
        recall::remember(self.db, query, opts)
    }
}

/// Visibility of a record under agent scoping: a scoped accessor sees its own
/// agent's records plus unscoped (global) ones — pre-existing shared memory
/// stays visible to every agent; an unscoped accessor sees everything.
pub fn agent_visible(agent: Option<&str>, data: &serde_json::Value) -> bool {
    match agent {
        None => true,
        Some(name) => data
            .get("_agent")
            .and_then(|v| v.as_str())
            .map(|a| a == name)
            .unwrap_or(true),
    }
}

/// Stamp a record's data with the owning agent (no-op for unscoped writers).
pub fn stamp_agent(data: &mut serde_json::Value, agent: Option<&str>) {
    if let Some(name) = agent {
        data["_agent"] = serde_json::json!(name);
    }
}

#[cfg(test)]
mod agent_scope_tests {
    use super::*;
    use axil_core::Axil;
    use serde_json::json;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scope.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn semantic_facts_are_isolated_per_agent_with_global_fallback() {
        let (db, _dir) = temp_db();

        let agent_a = AgentMemory::for_agent(&db, "claude");
        let agent_b = AgentMemory::for_agent(&db, "codex");
        let unscoped = AgentMemory::new(&db);

        agent_a.semantic().know("svc-a", "listens on 8080", None).unwrap();
        agent_b.semantic().know("svc-b", "listens on 9090", None).unwrap();

        // Each agent sees its own facts plus unscoped (global) ones.
        let a_entities = agent_a.semantic().list_entities().unwrap();
        assert!(a_entities.contains(&"svc-a".to_string()));
        assert!(!a_entities.contains(&"svc-b".to_string()));

        let b_entities = agent_b.semantic().list_entities().unwrap();
        assert!(b_entities.contains(&"svc-b".to_string()));
        assert!(!b_entities.contains(&"svc-a".to_string()));

        // An unscoped accessor sees everything.
        let all = unscoped.semantic().list_entities().unwrap();
        assert!(all.contains(&"svc-a".to_string()) && all.contains(&"svc-b".to_string()));
    }

    #[test]
    fn global_memory_stays_visible_to_scoped_agents() {
        let (db, _dir) = temp_db();

        AgentMemory::new(&db).semantic().know("shared", "global fact", None).unwrap();
        let agent_a = AgentMemory::for_agent(&db, "claude");
        let facts = agent_a.semantic().list_facts(Some("shared")).unwrap();
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn preferences_are_scoped_per_agent() {
        let (db, _dir) = temp_db();

        let agent_a = AgentMemory::for_agent(&db, "claude");
        let agent_b = AgentMemory::for_agent(&db, "codex");

        agent_a
            .preference()
            .set("test-runner", "cargo nextest", crate::preference::PreferenceSource::User)
            .unwrap();

        // Agent A reads its own rule back.
        assert!(agent_a.preference().get("test-runner").unwrap().is_some());
        // Agent B neither sees nor deletes it.
        assert!(agent_b.preference().get("test-runner").unwrap().is_none());
        assert!(!agent_b.preference().delete("test-runner").unwrap());
        // Unscoped sees it.
        assert!(AgentMemory::new(&db).preference().get("test-runner").unwrap().is_some());
    }

    #[test]
    fn remember_respects_agent_scope() {
        let (db, _dir) = temp_db();

        let agent_a = AgentMemory::for_agent(&db, "claude");
        let agent_b = AgentMemory::for_agent(&db, "codex");
        agent_a.semantic().know("deploy", "ship on friday", None).unwrap();
        agent_b.semantic().know("deploy-b", "ship on monday", None).unwrap();

        let opts = crate::recall::RecallOptions::default();
        let results = agent_a.remember("deploy", opts).unwrap();
        assert!(
            results
                .iter()
                .all(|r| r.scored.record.data.get("_agent").and_then(|v| v.as_str())
                    != Some("codex")),
            "agent A must never recall agent B's stamped records"
        );
    }
}
