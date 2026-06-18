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
    /// Per-agent isolation applies to working memory and sessions.
    /// Semantic, procedural, and preference memory remain shared.
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
    pub fn semantic(&self) -> SemanticMemory<'_> {
        SemanticMemory::new(self.db)
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
    pub fn procedural(&self) -> ProceduralMemory<'_> {
        ProceduralMemory::new(self.db)
    }

    /// Preference memory (rules & feedback).
    pub fn preference(&self) -> PreferenceMemory<'_> {
        PreferenceMemory::new(self.db)
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
    pub fn remember(
        &self,
        query: &str,
        opts: RecallOptions,
    ) -> axil_core::Result<Vec<RecallResult>> {
        recall::remember(self.db, query, opts)
    }
}
