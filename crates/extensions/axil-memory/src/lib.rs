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
#[cfg(test)]
mod test_support;
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
/// - **Per-agent** (isolated): Working memory, sessions, episodic memory —
///   an agent sees only its own.
/// - **Layered**: Semantic, procedural, preference memory — an agent's writes
///   are stamped as its own; it reads its own records plus unscoped (global)
///   ones, never another agent's.
///
/// Writes never cross a scope. Key-addressed writes (a preference `set` or
/// `delete`, a procedure `learn`) resolve by exact scope: an agent creates or
/// updates its own record — which then shadows the global record of the same
/// key on that agent's reads — and deletes only its own; an unscoped handle
/// touches only global records. Superseding, graph links, and entity merges
/// stay within a scope the same way.
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
    /// procedural, and preference writes are stamped with the agent name and
    /// never modify a global or another agent's record; reads see the agent's
    /// records plus unscoped (global) ones — memory that predates scoping
    /// stays visible to everyone.
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
        opts: RecallOptions,
    ) -> axil_core::Result<Vec<RecallResult>> {
        recall::remember_scoped(self.db, query, opts, self.agent.as_deref())
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

/// Ownership of a record under agent scoping — the rule every write resolves
/// by. A scoped accessor owns exactly the records stamped with its agent; an
/// unscoped accessor owns exactly the unstamped (global) ones. Reads are wider
/// (see [`agent_visible`]); a write that resolved its target through a
/// visibility read would let one scope edit or delete another's memory.
pub(crate) fn agent_owns(agent: Option<&str>, data: &serde_json::Value) -> bool {
    data.get("_agent").and_then(|v| v.as_str()) == agent
}

/// Keep the records a scoped reader should see for a keyed memory: each key
/// resolves to the reader's own record when it has one, which shadows the
/// global record of the same key. `own_keys` holds the keys the reader owns.
/// Unscoped readers see every record, so nothing is shadowed for them.
pub(crate) fn drop_shadowed<T>(
    agent: Option<&str>,
    items: Vec<T>,
    own_keys: &std::collections::HashSet<String>,
    parts: impl Fn(&T) -> (&serde_json::Value, Option<&str>),
) -> Vec<T> {
    if agent.is_none() || own_keys.is_empty() {
        return items;
    }
    items
        .into_iter()
        .filter(|item| {
            let (data, key) = parts(item);
            agent_owns(agent, data) || !key.is_some_and(|k| own_keys.contains(k))
        })
        .collect()
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

    #[test]
    fn remember_with_vectors_respects_agent_scope() {
        let (db, _mock, _dir) = crate::test_support::vector_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");
        claude
            .semantic()
            .know("deploy", "ship on friday", None)
            .unwrap();
        codex
            .semantic()
            .know("deploy", "ship on monday", None)
            .unwrap();
        AgentMemory::new(&db)
            .semantic()
            .know("deploy", "ship after review", None)
            .unwrap();

        let results = claude
            .remember("deploy ship", RecallOptions::default())
            .unwrap();
        let facts: Vec<&str> = results
            .iter()
            .filter_map(|r| r.scored.record.data.get("fact").and_then(|v| v.as_str()))
            .collect();
        assert!(facts.contains(&"ship on friday"));
        assert!(
            facts.contains(&"ship after review"),
            "global memory stays visible"
        );
        assert!(
            !facts.contains(&"ship on monday"),
            "never another agent's memory"
        );
    }

    use crate::preference::PreferenceSource;

    fn pref_value(mem: &AgentMemory<'_>, key: &str) -> Option<String> {
        mem.preference().get(key).unwrap().and_then(|r| {
            r.data
                .get("value")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
    }

    /// The unstamped (global) preference rows for `key`, read straight from
    /// the table so no accessor's resolution rules are involved.
    fn global_pref_values(db: &Axil, key: &str) -> Vec<String> {
        db.list(crate::types::TABLE_PREFERENCES)
            .unwrap()
            .into_iter()
            .filter(|r| r.data.get("key").and_then(|v| v.as_str()) == Some(key))
            .filter(|r| r.data.get("_agent").is_none())
            .filter_map(|r| {
                r.data
                    .get("value")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect()
    }

    #[test]
    fn scoped_preference_set_does_not_edit_the_global_rule() {
        let (db, _dir) = temp_db();
        let global = AgentMemory::new(&db);
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        global
            .preference()
            .set("test-runner", "cargo test", PreferenceSource::User)
            .unwrap();
        codex
            .preference()
            .set("test-runner", "nextest", PreferenceSource::User)
            .unwrap();

        assert_eq!(global_pref_values(&db, "test-runner"), vec!["cargo test"]);
        assert_eq!(
            pref_value(&claude, "test-runner").as_deref(),
            Some("cargo test")
        );
        // codex's own rule shadows the global one for codex only.
        assert_eq!(
            pref_value(&codex, "test-runner").as_deref(),
            Some("nextest")
        );
        let codex_rules = codex.preference().list().unwrap();
        assert_eq!(
            codex_rules.len(),
            1,
            "a shadowed global rule is not listed alongside the agent's own"
        );
        assert_eq!(codex_rules[0].data["value"], "nextest");
        // An unscoped reader resolves to the global rule.
        assert_eq!(
            pref_value(&global, "test-runner").as_deref(),
            Some("cargo test")
        );
    }

    #[test]
    fn scoped_preference_delete_removes_only_its_own_rule() {
        let (db, _dir) = temp_db();
        let global = AgentMemory::new(&db);
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        global
            .preference()
            .set("test-runner", "cargo test", PreferenceSource::User)
            .unwrap();

        // codex has no rule of its own: nothing to delete, global untouched.
        assert!(!codex.preference().delete("test-runner").unwrap());
        assert_eq!(
            pref_value(&claude, "test-runner").as_deref(),
            Some("cargo test")
        );

        // Deleting codex's own rule re-exposes the global one to codex.
        codex
            .preference()
            .set("test-runner", "nextest", PreferenceSource::User)
            .unwrap();
        assert!(codex.preference().delete("test-runner").unwrap());
        assert_eq!(
            pref_value(&codex, "test-runner").as_deref(),
            Some("cargo test")
        );
        assert_eq!(
            pref_value(&claude, "test-runner").as_deref(),
            Some("cargo test")
        );
        assert_eq!(global_pref_values(&db, "test-runner"), vec!["cargo test"]);
    }

    #[test]
    fn unscoped_preference_set_does_not_write_into_an_agent_rule() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        claude
            .preference()
            .set("style", "terse", PreferenceSource::User)
            .unwrap();
        AgentMemory::new(&db)
            .preference()
            .set("style", "verbose", PreferenceSource::User)
            .unwrap();

        assert_eq!(global_pref_values(&db, "style"), vec!["verbose"]);
        assert_eq!(pref_value(&codex, "style").as_deref(), Some("verbose"));
        assert_eq!(pref_value(&claude, "style").as_deref(), Some("terse"));
        // The unscoped delete removes only the global rule.
        assert!(AgentMemory::new(&db).preference().delete("style").unwrap());
        assert_eq!(pref_value(&claude, "style").as_deref(), Some("terse"));
        assert_eq!(pref_value(&codex, "style"), None);
    }

    #[test]
    fn scoped_detected_preference_does_not_shadow_a_global_user_rule() {
        let (db, _dir) = temp_db();
        let codex = AgentMemory::for_agent(&db, "codex");

        AgentMemory::new(&db)
            .preference()
            .set("style", "user choice", PreferenceSource::User)
            .unwrap();
        codex
            .preference()
            .set("style", "detected guess", PreferenceSource::Detected)
            .unwrap();

        assert_eq!(pref_value(&codex, "style").as_deref(), Some("user choice"));
        assert_eq!(global_pref_values(&db, "style"), vec!["user choice"]);
        assert_eq!(db.list(crate::types::TABLE_PREFERENCES).unwrap().len(), 1);
    }

    fn procedure_rows(db: &Axil, name: &str) -> Vec<Record> {
        db.list(crate::types::TABLE_PROCEDURES)
            .unwrap()
            .into_iter()
            .filter(|r| r.data.get("pattern_name").and_then(|v| v.as_str()) == Some(name))
            .collect()
    }

    fn confidence(r: &Record) -> f64 {
        r.data.get("confidence").and_then(|v| v.as_f64()).unwrap()
    }

    use axil_core::Record;

    #[test]
    fn scoped_learn_does_not_reinforce_the_global_procedure() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        let global = AgentMemory::new(&db)
            .procedural()
            .learn("fix-timeout", "raise the pool size", None)
            .unwrap();
        let forked = codex
            .procedural()
            .learn("fix-timeout", "raise the pool, then retry", None)
            .unwrap();

        assert_ne!(forked.id, global.id, "the agent writes its own copy");
        assert_eq!(forked.data["_agent"], "codex");
        let global_now = db.get(&global.id).unwrap().unwrap();
        assert_eq!(global_now.data["description"], "raise the pool size");
        assert_eq!(confidence(&global_now), confidence(&global));
        // The fork inherits the global track record, then is reinforced.
        assert!(confidence(&forked) > confidence(&global));

        let seen_by_claude = claude
            .procedural()
            .find_by_name("fix-timeout")
            .unwrap()
            .unwrap();
        assert_eq!(seen_by_claude.id, global.id);
        let seen_by_codex = codex
            .procedural()
            .find_by_name("fix-timeout")
            .unwrap()
            .unwrap();
        assert_eq!(seen_by_codex.id, forked.id);
        assert_eq!(codex.procedural().list().unwrap().len(), 1);
    }

    #[test]
    fn unscoped_learn_does_not_reinforce_an_agent_procedure() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        let private = claude
            .procedural()
            .learn("deploy-check", "claude's private runbook", None)
            .unwrap();
        AgentMemory::new(&db)
            .procedural()
            .learn("deploy-check", "shared runbook", None)
            .unwrap();

        let private_now = db.get(&private.id).unwrap().unwrap();
        assert_eq!(private_now.data["description"], "claude's private runbook");
        assert_eq!(confidence(&private_now), confidence(&private));
        assert_eq!(procedure_rows(&db, "deploy-check").len(), 2);
        let seen_by_codex = codex
            .procedural()
            .find_by_name("deploy-check")
            .unwrap()
            .unwrap();
        assert_eq!(seen_by_codex.data["description"], "shared runbook");
    }

    #[test]
    fn scoped_record_outcome_never_writes_outside_its_scope() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        let global = AgentMemory::new(&db)
            .procedural()
            .learn("fix-timeout", "raise the pool size", None)
            .unwrap();
        let private = claude
            .procedural()
            .learn("claude-only", "claude's private runbook", None)
            .unwrap();

        // Another agent's private procedure is not addressable at all.
        assert!(codex
            .procedural()
            .record_outcome(&private.id, crate::Outcome::Failure)
            .is_err());
        assert_eq!(
            confidence(&db.get(&private.id).unwrap().unwrap()),
            confidence(&private)
        );

        // Feedback on a shared procedure lands on the agent's own copy.
        let own = codex
            .procedural()
            .record_outcome(&global.id, crate::Outcome::Failure)
            .unwrap();
        assert_ne!(own.id, global.id);
        assert_eq!(own.data["_agent"], "codex");
        assert!(confidence(&own) < confidence(&global));
        assert_eq!(
            confidence(&db.get(&global.id).unwrap().unwrap()),
            confidence(&global)
        );

        // A second outcome reuses that copy instead of forking again.
        let again = codex
            .procedural()
            .record_outcome(&global.id, crate::Outcome::Success)
            .unwrap();
        assert_eq!(again.id, own.id);
        assert_eq!(procedure_rows(&db, "fix-timeout").len(), 2);
    }

    #[test]
    fn scoped_episode_create_is_owned_by_the_agent() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        let ep = claude
            .episodic()
            .create(
                "Fixed the flaky deploy",
                crate::Outcome::Success,
                None,
                None,
            )
            .unwrap();
        assert_eq!(ep.data["_agent"], "claude");
        assert_eq!(claude.episodic().list(None, 100).unwrap().len(), 1);
        assert!(codex.episodic().list(None, 100).unwrap().is_empty());
    }

    #[test]
    fn scoped_episode_similarity_stays_in_scope() {
        let (db, _mock, _dir) = crate::test_support::vector_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");
        claude
            .episodic()
            .create("deploy rollback drill", crate::Outcome::Success, None, None)
            .unwrap();

        assert_eq!(
            claude
                .episodic()
                .similar("deploy rollback", 5)
                .unwrap()
                .len(),
            1
        );
        assert!(codex
            .episodic()
            .similar("deploy rollback", 5)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn scoped_semantic_merge_moves_only_its_own_facts() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");

        AgentMemory::new(&db)
            .semantic()
            .know("pg", "shared fact", None)
            .unwrap();
        codex
            .semantic()
            .know("pg", "codex private fact", None)
            .unwrap();
        claude
            .semantic()
            .know("pg", "claude private fact", None)
            .unwrap();

        let moved = claude.semantic().merge("postgres", "pg").unwrap();
        assert_eq!(moved, 1, "only claude's own fact is re-pointed");

        let pg_facts: Vec<String> = db
            .list(crate::types::TABLE_ENTITIES)
            .unwrap()
            .into_iter()
            .filter(|r| r.data["entity"] == "pg")
            .filter_map(|r| r.data["fact"].as_str().map(String::from))
            .collect();
        assert!(pg_facts.contains(&"shared fact".to_string()));
        assert!(pg_facts.contains(&"codex private fact".to_string()));
        assert_eq!(codex.semantic().list_facts(Some("pg")).unwrap().len(), 2);
    }

    #[test]
    fn scoped_semantic_history_hides_other_agents() {
        let (db, _dir) = temp_db();
        let claude = AgentMemory::for_agent(&db, "claude");
        let codex = AgentMemory::for_agent(&db, "codex");
        claude
            .semantic()
            .know("deploy", "claude's plan", None)
            .unwrap();
        codex
            .semantic()
            .know("deploy", "codex's plan", None)
            .unwrap();

        let history = claude.semantic().history("deploy").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].data["fact"], "claude's plan");
    }
}
