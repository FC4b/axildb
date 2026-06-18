//! Semantic memory — knowledge graph of facts and entities.
//!
//! Stores facts about entities and automatically links them via graph edges.
//! Facts are auto-embedded for vector search.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Direction, Record, Result};

use crate::supersede::set_bitemporal;
use crate::types::{
    validate_entity_name, validate_text, EDGE_RELATED_TO, MIN_ENTITY_NAME_LEN_FOR_MATCH,
    TABLE_ENTITIES, TABLE_ENTITY_ALIASES,
};

/// Semantic memory — facts, entities, and their relationships.
pub struct SemanticMemory<'a> {
    db: &'a Axil,
}

impl<'a> SemanticMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Store a fact about an entity.
    ///
    /// Auto-embeds the fact for vector search and checks for superseding.
    /// If the entity name is an alias, the canonical name is used instead.
    pub fn know(&self, entity: &str, fact: &str, source: Option<&str>) -> Result<Record> {
        // Resolve alias to canonical name before storing.
        let canonical = self.resolve(entity)?;
        let entity = canonical.as_deref().unwrap_or(entity);

        validate_entity_name(entity)?;
        validate_text(fact, "fact")?;

        let mut data = json!({
            "entity": entity,
            "fact": fact,
        });

        if let Some(s) = source {
            data["source"] = json!(s);
        }

        set_bitemporal(&mut data, None);

        let record = self.db.insert(TABLE_ENTITIES, data)?;

        // Auto-embed the fact if embedder is available.
        if self.db.has_vector_index() {
            let embed_text = format!("{entity}: {fact}");
            let _ = self.db.embed_text(&record.id, &embed_text);
        }

        // Check for superseding within entity facts.
        let supersede = crate::supersede::SupersedeEngine::new(self.db);
        let _ = supersede.check_and_supersede(&record);

        // Auto-discover relationships: if this entity is mentioned in facts
        // about other entities, create related_to edges.
        self.auto_discover_relationships(&record, entity)?;

        Ok(record)
    }

    /// Get everything known about an entity.
    ///
    /// Returns all non-superseded facts, plus graph neighbors.
    pub fn about(&self, entity: &str) -> Result<EntityKnowledge> {
        // Get all facts about this entity.
        let all_records = self.db.list(TABLE_ENTITIES)?;
        let facts: Vec<Record> = all_records
            .into_iter()
            .filter(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(|e| e == entity)
                    .unwrap_or(false)
            })
            .filter(|r| !crate::ttl::is_record_superseded(r))
            .filter(|r| !crate::ttl::is_record_expired(r))
            .collect();

        // Get related entities via graph.
        let related = if self.db.has_graph_index() && !facts.is_empty() {
            let mut related = Vec::new();
            for fact in &facts {
                if let Ok(neighbors) = self.db.neighbors(
                    &fact.id,
                    None, // all edge types
                    Direction::Both,
                ) {
                    for neighbor in neighbors {
                        if neighbor.table == TABLE_ENTITIES {
                            if let Some(name) = neighbor.data.get("entity").and_then(|v| v.as_str())
                            {
                                if name != entity && !related.contains(&name.to_string()) {
                                    related.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
            related
        } else {
            Vec::new()
        };

        Ok(EntityKnowledge {
            entity: entity.to_string(),
            facts,
            related_entities: related,
        })
    }

    /// List all known entities (unique entity names).
    pub fn list_entities(&self) -> Result<Vec<String>> {
        let records = self.db.list(TABLE_ENTITIES)?;
        let mut entities: Vec<String> = records
            .iter()
            .filter_map(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();
        entities.sort();
        entities.dedup();
        Ok(entities)
    }

    /// List all facts, optionally filtered by entity.
    pub fn list_facts(&self, entity: Option<&str>) -> Result<Vec<Record>> {
        let records = self.db.list(TABLE_ENTITIES)?;
        let filtered: Vec<Record> = records
            .into_iter()
            .filter(|r| !crate::ttl::is_record_superseded(r))
            .filter(|r| !crate::ttl::is_record_expired(r))
            .filter(|r| {
                if let Some(e) = entity {
                    r.data
                        .get("entity")
                        .and_then(|v| v.as_str())
                        .map(|n| n == e)
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .collect();
        Ok(filtered)
    }

    /// Get the history of an entity — all versions including superseded.
    pub fn history(&self, entity: &str) -> Result<Vec<Record>> {
        let supersede = crate::supersede::SupersedeEngine::new(self.db);
        supersede.history(entity, TABLE_ENTITIES)
    }

    /// Register an alias for an entity.
    ///
    /// Stores a mapping so that `resolve(alias)` returns the canonical entity name.
    /// Multiple aliases can point to the same entity.
    pub fn add_alias(&self, entity: &str, alias: &str) -> Result<()> {
        validate_entity_name(entity)?;
        validate_entity_name(alias)?;

        // Don't allow aliasing to self.
        if entity.eq_ignore_ascii_case(alias) {
            return Err(axil_core::AxilError::InvalidQuery(
                "alias cannot be the same as the entity name".into(),
            ));
        }

        // Check for conflicting alias (alias already points to a different entity).
        if let Some(existing) = self.resolve(alias)? {
            if !existing.eq_ignore_ascii_case(entity) {
                return Err(axil_core::AxilError::InvalidQuery(format!(
                    "alias \"{alias}\" already points to entity \"{existing}\""
                )));
            }
            // Already registered — no-op.
            return Ok(());
        }

        self.db.insert(
            TABLE_ENTITY_ALIASES,
            json!({
                "entity": entity,
                "alias": alias,
            }),
        )?;
        Ok(())
    }

    /// Resolve a name to its canonical entity.
    ///
    /// Returns `Some(canonical_name)` if the given name is a registered alias,
    /// or `None` if it is not an alias (i.e., it is already canonical or unknown).
    /// Checks exact match first, then case-insensitive.
    pub fn resolve(&self, name: &str) -> Result<Option<String>> {
        let aliases = self.db.list(TABLE_ENTITY_ALIASES)?;

        // Exact match first.
        for r in &aliases {
            if r.data.get("alias").and_then(|v| v.as_str()) == Some(name) {
                if let Some(entity) = r.data.get("entity").and_then(|v| v.as_str()) {
                    return Ok(Some(entity.to_string()));
                }
            }
        }

        // Case-insensitive fallback.
        let name_lower = name.to_lowercase();
        for r in &aliases {
            if let Some(alias_val) = r.data.get("alias").and_then(|v| v.as_str()) {
                if alias_val.to_lowercase() == name_lower {
                    if let Some(entity) = r.data.get("entity").and_then(|v| v.as_str()) {
                        return Ok(Some(entity.to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Resolve a name with confidence scoring.
    ///
    /// Returns `(canonical_name, confidence)` where confidence is:
    /// - 1.0 for exact alias match
    /// - 0.9 for case-insensitive alias match
    /// - 0.5–0.85 for fuzzy matches against known entity names (not auto-resolved)
    ///
    /// Fuzzy matches below `auto_threshold` (default 0.8) are returned but
    /// NOT auto-resolved — they should be flagged for review.
    pub fn resolve_with_confidence(&self, name: &str) -> Result<Vec<EntityMatch>> {
        let mut matches = Vec::new();

        // 1. Check alias table (high confidence)
        let aliases = self.db.list(TABLE_ENTITY_ALIASES)?;
        for r in &aliases {
            if let (Some(alias_val), Some(entity)) = (
                r.data.get("alias").and_then(|v| v.as_str()),
                r.data.get("entity").and_then(|v| v.as_str()),
            ) {
                if alias_val == name {
                    matches.push(EntityMatch {
                        canonical: entity.to_string(),
                        confidence: 1.0,
                        method: MatchMethod::ExactAlias,
                    });
                    return Ok(matches);
                }
                if alias_val.eq_ignore_ascii_case(name) {
                    matches.push(EntityMatch {
                        canonical: entity.to_string(),
                        confidence: 0.9,
                        method: MatchMethod::CaseInsensitiveAlias,
                    });
                    return Ok(matches);
                }
            }
        }

        // 2. Collect all known canonical entity names (from facts + alias table)
        let entities = self.db.list(TABLE_ENTITIES)?;
        let mut seen = std::collections::HashSet::new();
        let mut canonical_names: Vec<String> = Vec::new();

        for r in &entities {
            if let Some(entity_name) = r.data.get("entity").and_then(|v| v.as_str()) {
                let lower = entity_name.to_lowercase();
                if seen.insert(lower) {
                    canonical_names.push(entity_name.to_string());
                }
            }
        }
        // Also include canonical names from alias table (entities with aliases but no facts yet)
        for r in &aliases {
            if let Some(entity_name) = r.data.get("entity").and_then(|v| v.as_str()) {
                let lower = entity_name.to_lowercase();
                if seen.insert(lower) {
                    canonical_names.push(entity_name.to_string());
                }
            }
        }

        let name_lower = name.to_lowercase();
        let name_words: Vec<&str> = name_lower.split_whitespace().collect();

        for entity_name in &canonical_names {
            let entity_lower = entity_name.to_lowercase();
            if entity_lower == name_lower {
                matches.push(EntityMatch {
                    canonical: entity_name.to_string(),
                    confidence: 1.0,
                    method: MatchMethod::ExactEntity,
                });
                return Ok(matches);
            }

            let sim = token_similarity(&name_lower, &name_words, &entity_lower);
            if sim >= FUZZY_MATCH_THRESHOLD {
                matches.push(EntityMatch {
                    canonical: entity_name.to_string(),
                    confidence: sim,
                    method: if sim >= AUTO_RESOLVE_THRESHOLD {
                        MatchMethod::FuzzyHigh
                    } else {
                        MatchMethod::FuzzyLow
                    },
                });
            }
        }

        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(5);
        Ok(matches)
    }

    /// List all aliases for a given entity.
    pub fn aliases(&self, entity: &str) -> Result<Vec<String>> {
        let all = self.db.list(TABLE_ENTITY_ALIASES)?;
        let entity_lower = entity.to_lowercase();
        let mut result: Vec<String> = all
            .iter()
            .filter(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(|e| e.to_lowercase() == entity_lower)
                    .unwrap_or(false)
            })
            .filter_map(|r| {
                r.data
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();
        result.sort();
        Ok(result)
    }

    /// Merge two entities: move all facts from `source` to `target`,
    /// transfer aliases, and register `source` as an alias of `target`.
    ///
    /// Returns the number of facts moved.
    pub fn merge(&self, target: &str, source: &str) -> Result<usize> {
        validate_entity_name(target)?;
        validate_entity_name(source)?;

        if target.eq_ignore_ascii_case(source) {
            return Err(axil_core::AxilError::InvalidQuery(
                "cannot merge an entity into itself".into(),
            ));
        }

        // 1. Re-point all facts from source → target.
        let all = self.db.list(TABLE_ENTITIES)?;
        let mut moved = 0usize;
        for record in &all {
            if record.data.get("entity").and_then(|v| v.as_str()) == Some(source) {
                let mut data = record.data.clone();
                data["entity"] = json!(target);
                self.db.update(&record.id, data)?;
                moved += 1;
            }
        }

        // 2. Transfer aliases: re-point source's aliases to target.
        let alias_records = self.db.list(TABLE_ENTITY_ALIASES)?;
        for record in &alias_records {
            if record.data.get("entity").and_then(|v| v.as_str()) == Some(source) {
                let mut data = record.data.clone();
                data["entity"] = json!(target);
                self.db.update(&record.id, data)?;
            }
        }

        // 3. Register source name as an alias of target.
        let _ = self.add_alias(target, source);

        Ok(moved)
    }

    /// Resolve a name with strategy-based disambiguation.
    ///
    /// Like `resolve_with_confidence`, but applies an additional strategy
    /// to boost or re-rank fuzzy matches:
    /// - **Frequency**: entities with more facts get a confidence boost
    /// - **Session**: entities mentioned in recent/active sessions get a boost
    /// - **Context**: entities whose facts contain the given context terms get a boost
    pub fn resolve_with_strategy(
        &self,
        name: &str,
        opts: &DisambiguationOptions,
    ) -> Result<Vec<EntityMatch>> {
        let mut matches = self.resolve_with_confidence(name)?;

        // Exact/alias matches don't need boosting.
        if matches.len() == 1 && matches[0].confidence >= 0.9 {
            return Ok(matches);
        }

        if opts.strategy != DisambiguationStrategy::Default {
            // Single fetch: load entities once for all strategy methods.
            let all_entities = self.db.list(TABLE_ENTITIES)?;

            match opts.strategy {
                DisambiguationStrategy::Default => unreachable!(),
                DisambiguationStrategy::Frequency => {
                    Self::boost_by_frequency(&mut matches, &all_entities);
                }
                DisambiguationStrategy::Session => {
                    self.boost_by_session(&mut matches, &all_entities, opts.session_id.as_deref())?;
                }
                DisambiguationStrategy::Context => {
                    Self::boost_by_context(&mut matches, &all_entities, &opts.context_terms);
                }
            }
        }

        // Re-sort and clamp.
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for m in &mut matches {
            m.confidence = m.confidence.min(1.0);
        }
        matches.truncate(5);
        Ok(matches)
    }

    /// Boost entities that have more facts (frequency-based disambiguation).
    fn boost_by_frequency(matches: &mut [EntityMatch], all_entities: &[Record]) {
        let mut fact_counts: HashMap<String, usize> = HashMap::new();
        for r in all_entities {
            if crate::ttl::is_record_superseded(r) {
                continue;
            }
            if let Some(name) = r.data.get("entity").and_then(|v| v.as_str()) {
                *fact_counts.entry(name.to_lowercase()).or_default() += 1;
            }
        }

        let max_count = fact_counts.values().copied().max().unwrap_or(1).max(1) as f32;
        for m in matches.iter_mut() {
            let count = fact_counts
                .get(&m.canonical.to_lowercase())
                .copied()
                .unwrap_or(0) as f32;
            let boost = 0.15 * (count / max_count);
            if boost > 0.0 {
                m.confidence += boost;
                m.method = MatchMethod::FrequencyBoosted;
            }
        }
    }

    /// Boost entities mentioned in active or recent sessions.
    fn boost_by_session(
        &self,
        matches: &mut [EntityMatch],
        all_entities: &[Record],
        session_id: Option<&str>,
    ) -> Result<()> {
        let session_entities: std::collections::HashSet<String> = if let Some(sid) = session_id {
            Self::entities_in_session(all_entities, sid, self.db)
        } else {
            // Reuse WorkingMemory for session listing instead of inline logic.
            let wm = crate::session::WorkingMemory::new(self.db);
            let active = wm.list_sessions(true)?;
            let targets: Vec<Record> = if active.is_empty() {
                let mut all = wm.list_sessions(false)?;
                all.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                all.into_iter().take(3).collect()
            } else {
                active
            };

            let mut ents = std::collections::HashSet::new();
            for session in &targets {
                ents.extend(Self::entities_in_session(
                    all_entities,
                    &session.id.to_string(),
                    self.db,
                ));
            }
            ents
        };

        for m in matches.iter_mut() {
            if session_entities.contains(&m.canonical.to_lowercase()) {
                m.confidence += 0.1;
                m.method = MatchMethod::SessionBoosted;
            }
        }
        Ok(())
    }

    /// Find entity names mentioned in a session using pre-loaded entity records.
    fn entities_in_session(
        all_entities: &[Record],
        session_id: &str,
        db: &Axil,
    ) -> std::collections::HashSet<String> {
        let mut entities = std::collections::HashSet::new();

        for r in all_entities {
            let matches_session = r
                .data
                .get("_session_id")
                .and_then(|v| v.as_str())
                .map(|s| s == session_id)
                .unwrap_or(false);
            if matches_session {
                if let Some(name) = r.data.get("entity").and_then(|v| v.as_str()) {
                    entities.insert(name.to_lowercase());
                }
            }
        }

        if db.has_graph_index() {
            if let Ok(rid) = axil_core::RecordId::from_string(session_id) {
                if let Ok(neighbors) = db.neighbors(
                    &rid,
                    Some(crate::types::EDGE_SESSION_CONTAINS),
                    Direction::Out,
                ) {
                    for n in &neighbors {
                        if n.table == TABLE_ENTITIES {
                            if let Some(name) = n.data.get("entity").and_then(|v| v.as_str()) {
                                entities.insert(name.to_lowercase());
                            }
                        }
                    }
                }
            }
        }

        entities
    }

    /// Boost entities whose facts contain the given context terms.
    fn boost_by_context(
        matches: &mut [EntityMatch],
        all_entities: &[Record],
        context_terms: &[String],
    ) {
        if context_terms.is_empty() {
            return;
        }

        let context_lower: Vec<String> = context_terms.iter().map(|t| t.to_lowercase()).collect();

        let mut context_scores: HashMap<String, f32> = HashMap::new();
        for r in all_entities {
            if crate::ttl::is_record_superseded(r) {
                continue;
            }
            let entity_name = match r.data.get("entity").and_then(|v| v.as_str()) {
                Some(n) => n.to_lowercase(),
                None => continue,
            };
            let fact = r
                .data
                .get("fact")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();

            let hits: usize = context_lower
                .iter()
                .filter(|term| fact.contains(term.as_str()))
                .count();
            if hits > 0 {
                let score = context_scores.entry(entity_name).or_default();
                *score += hits as f32;
            }
        }

        let max_score = context_scores.values().copied().fold(1.0_f32, f32::max);
        for m in matches.iter_mut() {
            if let Some(&score) = context_scores.get(&m.canonical.to_lowercase()) {
                let boost = 0.15 * (score / max_score);
                m.confidence += boost;
                m.method = MatchMethod::ContextBoosted;
            }
        }
    }

    /// Auto-discover relationships between entities.
    ///
    /// Single-pass: fetches all entity records once, builds a lookup map,
    /// then checks for substring matches in the new fact.
    fn auto_discover_relationships(&self, new_record: &Record, entity: &str) -> Result<()> {
        if !self.db.has_graph_index() {
            return Ok(());
        }

        let fact = new_record
            .data
            .get("fact")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if fact.is_empty() {
            return Ok(());
        }

        let fact_lower = fact.to_lowercase();

        // Single fetch: build lowercase_entity_name → (original_name, record) map.
        let all_records = self.db.list(TABLE_ENTITIES)?;
        let mut entity_map: HashMap<String, (String, Record)> = HashMap::new();
        for record in all_records {
            if crate::ttl::is_record_superseded(&record) {
                continue;
            }
            if let Some(name) = record.data.get("entity").and_then(|v| v.as_str()) {
                let key = name.to_lowercase();
                let entry = entity_map
                    .entry(key)
                    .or_insert_with(|| (name.to_string(), record.clone()));
                if record.created_at > entry.1.created_at {
                    *entry = (name.to_string(), record);
                }
            }
        }

        // Single fetch: get existing edges for the new record.
        let existing_edges =
            self.db
                .edges(&new_record.id, Some(EDGE_RELATED_TO), Direction::Both)?;

        let entity_lower = entity.to_lowercase();
        for (other_entity_lower, (_, other_record)) in &entity_map {
            if *other_entity_lower == entity_lower {
                continue;
            }
            if other_entity_lower.len() < MIN_ENTITY_NAME_LEN_FOR_MATCH {
                continue;
            }
            // Word-boundary-aware matching: entity name must appear as a
            // standalone word/token, not as a substring of another word
            // (e.g. "database" should not match entity "base").
            // Uses char-level boundaries for UTF-8 safety.
            let Some(pos) = fact_lower.find(other_entity_lower.as_str()) else {
                continue;
            };
            let before_ok = pos == 0
                || fact_lower[..pos]
                    .chars()
                    .next_back()
                    .map(|c| !c.is_alphanumeric())
                    .unwrap_or(true);
            let after_pos = pos + other_entity_lower.len();
            let after_ok = after_pos >= fact_lower.len()
                || fact_lower[after_pos..]
                    .chars()
                    .next()
                    .map(|c| !c.is_alphanumeric())
                    .unwrap_or(true);
            if !before_ok || !after_ok {
                continue;
            }

            let already_linked = existing_edges
                .iter()
                .any(|e| e.to == other_record.id || e.from == other_record.id);

            if !already_linked {
                let _ = self
                    .db
                    .relate(&new_record.id, EDGE_RELATED_TO, &other_record.id, None);
            }
        }

        Ok(())
    }
}

/// Everything known about a specific entity.
#[derive(Debug)]
pub struct EntityKnowledge {
    /// The entity name.
    pub entity: String,
    /// All active (non-superseded) facts.
    pub facts: Vec<Record>,
    /// Related entity names discovered via graph.
    pub related_entities: Vec<String>,
}

impl EntityKnowledge {
    /// Get a consolidated summary of all facts about this entity.
    ///
    /// Uses template-based merging from `axil_core::consolidation` to produce
    /// a single coherent profile when multiple facts exist.
    pub fn consolidated_summary(&self) -> Option<String> {
        if self.facts.is_empty() {
            return None;
        }
        let tagged: Vec<(Record, axil_core::consolidation::ConflictResult)> = self
            .facts
            .iter()
            .map(|r| (r.clone(), axil_core::consolidation::ConflictResult::Novel))
            .collect();
        axil_core::consolidation::consolidate_facts(&self.entity, &tagged).map(|c| c.summary)
    }

    /// Serialize to JSON for CLI output.
    pub fn to_json(&self) -> Value {
        let facts: Vec<Value> = self
            .facts
            .iter()
            .map(|r| {
                json!({
                    "id": r.id.to_string(),
                    "fact": r.data.get("fact").cloned().unwrap_or(Value::Null),
                    "source": r.data.get("source").cloned().unwrap_or(Value::Null),
                    "created_at": r.created_at.to_rfc3339(),
                })
            })
            .collect();

        let mut result = json!({
            "entity": self.entity,
            "facts": facts,
            "related_entities": self.related_entities,
            "fact_count": self.facts.len(),
        });

        if let Some(summary) = self.consolidated_summary() {
            result["consolidated"] = json!(summary);
        }

        result
    }
}

// ── Confidence scoring types ──────────────────────────────────────────

/// Minimum confidence for auto-resolution (no manual review needed).
const AUTO_RESOLVE_THRESHOLD: f32 = 0.8;

/// Minimum confidence to appear in fuzzy results.
const FUZZY_MATCH_THRESHOLD: f32 = 0.5;

/// Disambiguation strategy for resolving ambiguous entity names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DisambiguationStrategy {
    /// Default: token similarity only.
    #[default]
    Default,
    /// Boost entities mentioned in recent/active sessions.
    Session,
    /// Boost entities that appear more frequently (more facts).
    Frequency,
    /// Boost entities co-occurring with given context terms.
    Context,
}

impl std::str::FromStr for DisambiguationStrategy {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "frequency" => Ok(Self::Frequency),
            "session" => Ok(Self::Session),
            "context" => Ok(Self::Context),
            _ => Ok(Self::Default),
        }
    }
}

/// Options for disambiguation with strategy selection.
#[derive(Debug, Clone)]
pub struct DisambiguationOptions {
    /// Strategy to use for boosting fuzzy matches.
    pub strategy: DisambiguationStrategy,
    /// Context terms for `Context` strategy (words that should co-occur).
    pub context_terms: Vec<String>,
    /// Session ID to scope `Session` strategy (if None, uses most recent).
    pub session_id: Option<String>,
}

impl Default for DisambiguationOptions {
    fn default() -> Self {
        Self {
            strategy: DisambiguationStrategy::Default,
            context_terms: Vec::new(),
            session_id: None,
        }
    }
}

/// A scored entity match from disambiguation.
#[derive(Debug, Clone)]
pub struct EntityMatch {
    /// Canonical entity name.
    pub canonical: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// How the match was found.
    pub method: MatchMethod,
}

impl EntityMatch {
    /// Whether this match is high enough confidence to auto-resolve.
    pub fn is_auto_resolve(&self) -> bool {
        self.confidence >= AUTO_RESOLVE_THRESHOLD
    }

    /// Whether this match should be flagged for manual review.
    pub fn needs_review(&self) -> bool {
        self.confidence >= FUZZY_MATCH_THRESHOLD && self.confidence < AUTO_RESOLVE_THRESHOLD
    }

    pub fn to_json(&self) -> Value {
        json!({
            "canonical": self.canonical,
            "confidence": self.confidence,
            "method": format!("{:?}", self.method),
            "auto_resolve": self.is_auto_resolve(),
            "needs_review": self.needs_review(),
        })
    }
}

/// How an entity match was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Exact alias match (confidence: 1.0).
    ExactAlias,
    /// Case-insensitive alias match (confidence: 0.9).
    CaseInsensitiveAlias,
    /// Exact entity name match (confidence: 1.0).
    ExactEntity,
    /// Fuzzy match with high confidence (>= 0.8).
    FuzzyHigh,
    /// Fuzzy match with low confidence (< 0.8, needs review).
    FuzzyLow,
    /// Boosted by frequency (many facts about this entity).
    FrequencyBoosted,
    /// Boosted by session context (entity in active/recent session).
    SessionBoosted,
    /// Boosted by context terms (entity co-occurs with given terms).
    ContextBoosted,
}

/// Token-based similarity between two entity names.
///
/// Computes Jaccard similarity on word tokens, with a bonus for
/// prefix/substring matches. Returns 0.0–1.0.
fn token_similarity(a_lower: &str, a_words: &[&str], b_lower: &str) -> f32 {
    // Substring match gets high score
    if a_lower.contains(b_lower) || b_lower.contains(a_lower) {
        let len_ratio =
            a_lower.len().min(b_lower.len()) as f32 / a_lower.len().max(b_lower.len()) as f32;
        return 0.7 + 0.15 * len_ratio; // 0.7–0.85
    }

    // Word-level Jaccard similarity
    let b_words: Vec<&str> = b_lower.split_whitespace().collect();
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let shared = a_words.iter().filter(|w| b_words.contains(w)).count();
    let union = a_words.len() + b_words.len() - shared;
    if union == 0 {
        return 0.0;
    }

    shared as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn store_and_retrieve_fact() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        let record = sem
            .know("auth-module", "Uses JWT tokens with 1h expiry", None)
            .unwrap();
        assert_eq!(record.data["entity"], "auth-module");
        assert_eq!(record.data["fact"], "Uses JWT tokens with 1h expiry");

        let knowledge = sem.about("auth-module").unwrap();
        assert_eq!(knowledge.facts.len(), 1);
        assert_eq!(knowledge.entity, "auth-module");
    }

    #[test]
    fn list_entities() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.know("auth-module", "fact1", None).unwrap();
        sem.know("user-table", "fact2", None).unwrap();
        sem.know("auth-module", "fact3", None).unwrap();

        let entities = sem.list_entities().unwrap();
        assert_eq!(entities, vec!["auth-module", "user-table"]);
    }

    #[test]
    fn consolidated_summary_merges_facts() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.know("auth-module", "Uses JWT tokens with 1h expiry", None)
            .unwrap();
        sem.know("auth-module", "Refresh token rotation enabled", None)
            .unwrap();

        let knowledge = sem.about("auth-module").unwrap();
        assert_eq!(knowledge.facts.len(), 2);

        let summary = knowledge.consolidated_summary().unwrap();
        // Should contain content from both facts
        assert!(summary.contains("auth-module"), "summary: {summary}");

        // JSON output should include consolidated field
        let json = knowledge.to_json();
        assert!(
            json.get("consolidated").is_some(),
            "missing consolidated field"
        );
    }

    #[test]
    fn consolidated_summary_single_fact() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.know("db-module", "Uses PostgreSQL 15", None).unwrap();

        let knowledge = sem.about("db-module").unwrap();
        let summary = knowledge.consolidated_summary().unwrap();
        assert!(summary.contains("Uses PostgreSQL 15"));
    }

    #[test]
    fn add_alias_and_resolve() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("Sarah", "the VP of Engineering").unwrap();

        let resolved = sem.resolve("the VP of Engineering").unwrap();
        assert_eq!(resolved, Some("Sarah".to_string()));

        // Non-alias returns None.
        let resolved = sem.resolve("Sarah").unwrap();
        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_case_insensitive() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("Sarah", "VP of Engineering").unwrap();

        // Exact match first.
        let resolved = sem.resolve("VP of Engineering").unwrap();
        assert_eq!(resolved, Some("Sarah".to_string()));

        // Case-insensitive fallback.
        let resolved = sem.resolve("vp of engineering").unwrap();
        assert_eq!(resolved, Some("Sarah".to_string()));
    }

    #[test]
    fn list_aliases() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("Sarah", "VP of Engineering").unwrap();
        sem.add_alias("Sarah", "Sarah K.").unwrap();

        let aliases = sem.aliases("Sarah").unwrap();
        assert_eq!(aliases, vec!["Sarah K.", "VP of Engineering"]);

        // Case-insensitive entity lookup.
        let aliases = sem.aliases("sarah").unwrap();
        assert_eq!(aliases, vec!["Sarah K.", "VP of Engineering"]);
    }

    #[test]
    fn know_resolves_alias() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("auth-module", "the auth system").unwrap();

        // Store fact using alias — should resolve to canonical name.
        let record = sem
            .know("the auth system", "Uses JWT tokens", None)
            .unwrap();
        assert_eq!(record.data["entity"], "auth-module");

        // Fact should appear under canonical name.
        let knowledge = sem.about("auth-module").unwrap();
        assert_eq!(knowledge.facts.len(), 1);
    }

    #[test]
    fn alias_self_rejected() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        let err = sem.add_alias("Sarah", "Sarah").unwrap_err();
        assert!(err.to_string().contains("same as the entity name"));

        // Case-insensitive self-check.
        let err = sem.add_alias("Sarah", "sarah").unwrap_err();
        assert!(err.to_string().contains("same as the entity name"));
    }

    #[test]
    fn alias_conflict_rejected() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("Sarah", "the boss").unwrap();

        // Same alias pointing to different entity should fail.
        let err = sem.add_alias("Dave", "the boss").unwrap_err();
        assert!(err.to_string().contains("already points to"));
    }

    #[test]
    fn alias_idempotent() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.add_alias("Sarah", "the boss").unwrap();
        // Re-adding the same alias to the same entity is a no-op.
        sem.add_alias("Sarah", "the boss").unwrap();

        let aliases = sem.aliases("Sarah").unwrap();
        assert_eq!(aliases.len(), 1);
    }

    #[test]
    fn list_facts_filtered() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.know("auth", "fact1", None).unwrap();
        sem.know("db", "fact2", None).unwrap();

        let all = sem.list_facts(None).unwrap();
        assert_eq!(all.len(), 2);

        let auth_only = sem.list_facts(Some("auth")).unwrap();
        assert_eq!(auth_only.len(), 1);
    }

    #[test]
    fn resolve_with_confidence_exact_alias() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        sem.add_alias("Sarah", "VP of Engineering").unwrap();

        let matches = sem.resolve_with_confidence("VP of Engineering").unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].canonical, "Sarah");
        assert_eq!(matches[0].confidence, 1.0);
        assert!(matches[0].is_auto_resolve());
        assert!(!matches[0].needs_review());
    }

    #[test]
    fn resolve_with_confidence_case_insensitive() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        sem.add_alias("Sarah", "VP of Engineering").unwrap();

        let matches = sem.resolve_with_confidence("vp of engineering").unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].confidence, 0.9);
        assert!(matches[0].is_auto_resolve());
    }

    #[test]
    fn resolve_with_confidence_fuzzy() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        sem.know("auth module", "handles login", None).unwrap();
        sem.know("database layer", "stores records", None).unwrap();

        let matches = sem.resolve_with_confidence("auth").unwrap();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].canonical, "auth module");
        assert!(matches[0].confidence >= 0.7); // substring match
    }

    #[test]
    fn resolve_with_confidence_no_match() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        sem.know("auth module", "handles login", None).unwrap();

        let matches = sem.resolve_with_confidence("completely unrelated").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn resolve_with_confidence_alias_only_entity() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        // Entity exists only via alias, no facts stored yet
        sem.add_alias("Sarah", "VP of Engineering").unwrap();

        let matches = sem.resolve_with_confidence("Sarah").unwrap();
        assert!(
            !matches.is_empty(),
            "should find alias-only canonical entity"
        );
        assert_eq!(matches[0].canonical, "Sarah");
        assert_eq!(matches[0].confidence, 1.0);
    }

    #[test]
    fn token_similarity_basic() {
        let a = "auth module";
        let a_words: Vec<&str> = a.split_whitespace().collect();
        // Substring match
        assert!(token_similarity(a, &a_words, "auth") >= 0.7);
        // No match
        assert!(token_similarity(a, &a_words, "database") < 0.5);
    }

    #[test]
    fn frequency_strategy_boosts_common_entities() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        // auth-module has many facts, db-layer has one
        sem.know("auth-module", "Uses JWT tokens", None).unwrap();
        sem.know("auth-module", "Has rate limiting", None).unwrap();
        sem.know("auth-module", "Supports OAuth2", None).unwrap();
        sem.know("auth-layer", "Has connection pool", None).unwrap();

        let opts = DisambiguationOptions {
            strategy: DisambiguationStrategy::Frequency,
            ..Default::default()
        };
        let matches = sem.resolve_with_strategy("auth", &opts).unwrap();
        assert!(!matches.is_empty());
        // auth-module should rank higher due to frequency
        assert_eq!(matches[0].canonical, "auth-module");
    }

    #[test]
    fn context_strategy_boosts_matching_entities() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);

        sem.know("auth-module", "Uses JWT tokens for login", None)
            .unwrap();
        sem.know("auth-service", "Handles OAuth2 for API keys", None)
            .unwrap();

        let opts = DisambiguationOptions {
            strategy: DisambiguationStrategy::Context,
            context_terms: vec!["JWT".to_string(), "login".to_string()],
            ..Default::default()
        };
        let matches = sem.resolve_with_strategy("auth", &opts).unwrap();
        assert!(!matches.is_empty());
        // auth-module should rank higher because facts mention JWT and login
        assert_eq!(matches[0].canonical, "auth-module");
    }

    #[test]
    fn default_strategy_unchanged() {
        let (db, _dir) = temp_db();
        let sem = SemanticMemory::new(&db);
        sem.know("auth module", "handles login", None).unwrap();

        let opts = DisambiguationOptions::default();
        let matches = sem.resolve_with_strategy("auth", &opts).unwrap();
        // Same as resolve_with_confidence
        let baseline = sem.resolve_with_confidence("auth").unwrap();
        assert_eq!(matches.len(), baseline.len());
        assert_eq!(matches[0].canonical, baseline[0].canonical);
    }
}
