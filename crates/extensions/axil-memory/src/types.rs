//! Core types shared across all memory modules.

use serde::{Deserialize, Serialize};

/// The five memory types an agent can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Current session context, active tasks, tool outputs.
    Working,
    /// Facts, entities, relationships (knowledge graph).
    Semantic,
    /// Past sessions, interactions, outcomes.
    Episodic,
    /// Learned patterns, strategies, tool usage.
    Procedural,
    /// User preferences, feedback, rules, conventions.
    Preference,
}

impl MemoryType {
    /// The internal table name for this memory type.
    pub fn table_name(&self) -> &'static str {
        match self {
            MemoryType::Working => TABLE_WORKING,
            MemoryType::Semantic => TABLE_ENTITIES,
            MemoryType::Episodic => TABLE_EPISODES,
            MemoryType::Procedural => TABLE_PROCEDURES,
            MemoryType::Preference => TABLE_PREFERENCES,
        }
    }

    /// Default recency weight (alpha) for this memory type.
    ///
    /// `final_score = alpha * similarity + (1 - alpha) * recency`
    pub fn default_alpha(&self) -> f32 {
        match self {
            MemoryType::Working => 0.3,    // heavily weight recency
            MemoryType::Semantic => 0.8,   // heavily weight relevance
            MemoryType::Episodic => 0.5,   // balanced
            MemoryType::Procedural => 0.7, // relevance matters more
            MemoryType::Preference => 0.9, // almost pure relevance
        }
    }

    /// Default TTL in seconds, or `None` for no expiry.
    pub fn default_ttl_secs(&self) -> Option<u64> {
        match self {
            MemoryType::Working => None, // cleared on session end; orphaned on crash
            MemoryType::Semantic => None, // facts persist
            MemoryType::Episodic => Some(90 * 86400), // 90 days
            MemoryType::Procedural => None, // patterns persist
            MemoryType::Preference => None, // rules persist
        }
    }

    /// All memory types.
    pub fn all() -> &'static [MemoryType] {
        &[
            MemoryType::Working,
            MemoryType::Semantic,
            MemoryType::Episodic,
            MemoryType::Procedural,
            MemoryType::Preference,
        ]
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Working => write!(f, "working"),
            MemoryType::Semantic => write!(f, "semantic"),
            MemoryType::Episodic => write!(f, "episodic"),
            MemoryType::Procedural => write!(f, "procedural"),
            MemoryType::Preference => write!(f, "preference"),
        }
    }
}

/// Outcome of an episode or procedure application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    Failure,
    Partial,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Success => write!(f, "success"),
            Outcome::Failure => write!(f, "failure"),
            Outcome::Partial => write!(f, "partial"),
        }
    }
}

impl std::str::FromStr for Outcome {
    type Err = axil_core::AxilError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "success" => Ok(Outcome::Success),
            "failure" => Ok(Outcome::Failure),
            "partial" => Ok(Outcome::Partial),
            other => Err(axil_core::AxilError::InvalidQuery(format!(
                "unknown outcome: {other} (expected success, failure, or partial)"
            ))),
        }
    }
}

// ── Table names (prefixed with _ to indicate system tables) ──────────────

/// Working memory: current session state.
pub const TABLE_WORKING: &str = "_working";

/// Sessions table (shared with session lifecycle).
pub const TABLE_SESSIONS: &str = "_sessions";

/// Semantic memory: entities and facts.
pub const TABLE_ENTITIES: &str = "_entities";

/// Entity alias mappings (canonical name → alias).
pub const TABLE_ENTITY_ALIASES: &str = "_entity_aliases";

/// Episodic memory: completed sessions with outcomes.
pub const TABLE_EPISODES: &str = "_episodes";

/// Procedural memory: learned patterns and strategies.
pub const TABLE_PROCEDURES: &str = "_procedures";

/// Preference memory: user directives and detected preferences.
pub const TABLE_PREFERENCES: &str = "_preferences";

/// All memory table names searchable by `remember()`.
/// `TABLE_SESSIONS` is excluded — sessions are converted to episodes on end.
pub const MEMORY_TABLES: &[&str] = &[
    TABLE_WORKING,
    TABLE_ENTITIES,
    TABLE_EPISODES,
    TABLE_PROCEDURES,
    TABLE_PREFERENCES,
];

// ── Edge types ───────────────────────────────────────────────────────────

/// Session contains a record.
pub const EDGE_SESSION_CONTAINS: &str = "session_contains";

/// Entity is related to another entity.
pub const EDGE_RELATED_TO: &str = "related_to";

/// Entity uses/depends on another entity.
pub const EDGE_USES: &str = "uses";

/// Episode touched an entity.
pub const EDGE_TOUCHED: &str = "touched";

/// Procedure was learned from an episode.
pub const EDGE_LEARNED_FROM: &str = "learned_from";

/// New record supersedes old record.
pub const EDGE_SUPERSEDES: &str = "supersedes";

// ── Input limits ────────────────────────────────────────────────────────

/// Maximum length for entity names.
pub const MAX_ENTITY_NAME_LEN: usize = 256;

/// Maximum length for fact/description text.
pub const MAX_TEXT_LEN: usize = 65_536; // 64 KB

/// Maximum length for preference values.
pub const MAX_PREFERENCE_VALUE_LEN: usize = 8_192;

/// Minimum entity name length for substring relationship matching.
pub const MIN_ENTITY_NAME_LEN_FOR_MATCH: usize = 3;

/// Validate an entity name length.
pub fn validate_entity_name(name: &str) -> axil_core::Result<()> {
    if name.is_empty() {
        return Err(axil_core::AxilError::InvalidQuery(
            "entity name cannot be empty".into(),
        ));
    }
    if name.len() > MAX_ENTITY_NAME_LEN {
        return Err(axil_core::AxilError::InvalidQuery(format!(
            "entity name exceeds maximum length of {MAX_ENTITY_NAME_LEN} bytes"
        )));
    }
    Ok(())
}

/// Validate text content length.
pub fn validate_text(text: &str, field_name: &str) -> axil_core::Result<()> {
    if text.len() > MAX_TEXT_LEN {
        return Err(axil_core::AxilError::InvalidQuery(format!(
            "{field_name} exceeds maximum length of {MAX_TEXT_LEN} bytes"
        )));
    }
    Ok(())
}

/// Metadata keys used in record metadata.
pub const META_VALID_UNTIL: &str = "valid_until";
pub const META_VALID_FROM: &str = "valid_from";
pub const META_RECORDED_AT: &str = "recorded_at";
pub const META_SUPERSEDED: &str = "superseded";
pub const META_SUPERSEDED_BY: &str = "superseded_by";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_type_roundtrip() {
        for mt in MemoryType::all() {
            let s = serde_json::to_string(mt).unwrap();
            let parsed: MemoryType = serde_json::from_str(&s).unwrap();
            assert_eq!(*mt, parsed);
        }
    }

    #[test]
    fn outcome_roundtrip() {
        for o in &[Outcome::Success, Outcome::Failure, Outcome::Partial] {
            let s = o.to_string();
            let parsed: Outcome = s.parse().unwrap();
            assert_eq!(*o, parsed);
        }
    }

    #[test]
    fn default_alphas_are_valid() {
        for mt in MemoryType::all() {
            let a = mt.default_alpha();
            assert!(a >= 0.0 && a <= 1.0, "alpha for {mt} out of range: {a}");
        }
    }
}
