//! Memory synthesis — `axil reflect` scans all memory types and surfaces insights.
//!
//! Heuristic mode (no LLM): frequency analysis + graph traversal + pattern matching.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Direction, Record, Result};

use crate::ttl;
use crate::types::{TABLE_ENTITIES, TABLE_EPISODES, TABLE_PREFERENCES, TABLE_PROCEDURES};

/// Records older than this are excluded in `Recent` scope.
const RECENT_DAYS: i64 = 30;

/// Result of a reflection analysis.
#[derive(Debug, Serialize, Deserialize)]
pub struct ReflectReport {
    /// Topic that was analyzed (or "all").
    pub topic: String,
    /// Generated insights.
    pub insights: Vec<String>,
    /// Number of memories analyzed.
    pub memories_analyzed: usize,
    /// Approximate token count of the output.
    pub tokens: usize,
}

impl ReflectReport {
    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "topic": self.topic,
            "insights": self.insights,
            "memories_analyzed": self.memories_analyzed,
            "tokens": self.tokens,
        })
    }
}

/// Memory synthesis engine.
pub struct ReflectEngine<'a> {
    db: &'a Axil,
}

impl<'a> ReflectEngine<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Reflect on a topic or all memories.
    ///
    /// Uses heuristic analysis: entity frequency, episode patterns, graph connections.
    pub fn reflect(&self, topic: Option<&str>, scope: ReflectScope) -> Result<ReflectReport> {
        let topic_str = topic.unwrap_or("all").to_string();
        let mut insights = Vec::new();
        let mut total_analyzed = 0;
        let recency_cutoff = Utc::now() - chrono::Duration::days(RECENT_DAYS);

        // 1. Entity frequency analysis
        let entities = self.db.list(TABLE_ENTITIES).unwrap_or_default();
        let active_entities: Vec<&Record> = entities
            .iter()
            .filter(|r| !ttl::is_record_superseded(r) && !ttl::is_record_expired(r))
            .filter(|r| scope != ReflectScope::Recent || r.created_at >= recency_cutoff)
            .filter(|r| matches_topic(r, topic))
            .collect();
        total_analyzed += active_entities.len();

        // Count facts per entity
        let mut entity_facts: HashMap<String, usize> = HashMap::new();
        for r in &active_entities {
            if let Some(name) = r.data.get("entity").and_then(|v| v.as_str()) {
                *entity_facts.entry(name.to_string()).or_default() += 1;
            }
        }

        // Surface entities with many facts (knowledge hotspots)
        let mut sorted_entities: Vec<_> = entity_facts.iter().collect();
        sorted_entities.sort_by(|a, b| b.1.cmp(a.1));
        for (entity, count) in sorted_entities.iter().take(3) {
            if **count >= 3 {
                insights.push(format!(
                    "{entity} has {count} stored facts — consider consolidating into a profile",
                ));
            }
        }

        // 2. Episode analysis (skip if Entity scope — only entities matter)
        let episodes = if scope == ReflectScope::Entity {
            Vec::new()
        } else {
            self.db.list(TABLE_EPISODES).unwrap_or_default()
        };
        let relevant_episodes: Vec<&Record> = episodes
            .iter()
            .filter(|r| !ttl::is_record_expired(r))
            .filter(|r| scope != ReflectScope::Recent || r.created_at >= recency_cutoff)
            .filter(|r| matches_topic(r, topic))
            .collect();
        total_analyzed += relevant_episodes.len();

        // Count outcomes
        let mut failure_count = 0usize;
        for ep in &relevant_episodes {
            if ep.data.get("outcome").and_then(|v| v.as_str()) == Some("failure") {
                failure_count += 1;
            }
        }

        if failure_count > 0 && relevant_episodes.len() >= 3 {
            let fail_rate = failure_count as f32 / relevant_episodes.len() as f32;
            if fail_rate > 0.3 {
                insights.push(format!(
                    "{failure_count}/{} episodes ended in failure ({:.0}%) — investigate recurring issues",
                    relevant_episodes.len(),
                    fail_rate * 100.0,
                ));
            }
        }

        // 3. Graph connection analysis (if available)
        if self.db.has_graph_index() {
            let mut high_degree_entities = Vec::new();
            for r in active_entities.iter().take(50) {
                if let Ok(neighbors) = self.db.neighbors(&r.id, None, Direction::Both) {
                    if neighbors.len() >= 5 {
                        let name = r.data.get("entity").and_then(|v| v.as_str()).unwrap_or("?");
                        high_degree_entities.push((name.to_string(), neighbors.len()));
                    }
                }
            }
            high_degree_entities.sort_by(|a, b| b.1.cmp(&a.1));
            for (entity, degree) in high_degree_entities.iter().take(2) {
                insights.push(format!(
                    "{entity} is highly connected ({degree} relationships) — central to the project",
                ));
            }
        }

        // 4. Procedural memory analysis (skip if Entity scope)
        let procedures = if scope == ReflectScope::Entity {
            Vec::new()
        } else {
            self.db.list(TABLE_PROCEDURES).unwrap_or_default()
        };
        let relevant_procs: Vec<&Record> = procedures
            .iter()
            .filter(|r| scope != ReflectScope::Recent || r.created_at >= recency_cutoff)
            .filter(|r| matches_topic(r, topic))
            .collect();
        total_analyzed += relevant_procs.len();

        // Surface high-confidence procedures
        for proc in &relevant_procs {
            let confidence = proc
                .data
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let name = proc
                .data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            if confidence >= 0.8 {
                insights.push(format!(
                    "Pattern \"{name}\" has high confidence ({confidence:.1}) — reliable approach",
                ));
            }
        }

        // 5. Preference analysis
        if scope == ReflectScope::All {
            let prefs = self.db.list(TABLE_PREFERENCES).unwrap_or_default();
            total_analyzed += prefs.len();
            let user_rules = prefs
                .iter()
                .filter(|r| r.data.get("source").and_then(|v| v.as_str()) == Some("user"))
                .count();
            if user_rules > 0 {
                insights.push(format!(
                    "{user_rules} user-set rules are active — check with `axil rule list`",
                ));
            }
        }

        if insights.is_empty() {
            insights.push(format!(
                "No significant patterns found for \"{topic_str}\" across {total_analyzed} memories",
            ));
        }

        let tokens = insights
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>()
            + 20;

        Ok(ReflectReport {
            topic: topic_str,
            insights,
            memories_analyzed: total_analyzed,
            tokens,
        })
    }
}

/// Scope of reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectScope {
    /// All memory types.
    All,
    /// Only recent memories (last 30 days).
    Recent,
    /// Only a specific entity.
    Entity,
}

impl std::str::FromStr for ReflectScope {
    type Err = axil_core::AxilError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "all" => Ok(ReflectScope::All),
            "recent" => Ok(ReflectScope::Recent),
            "entity" => Ok(ReflectScope::Entity),
            other => Err(axil_core::AxilError::InvalidQuery(format!(
                "unknown scope: {other} (expected all, recent, or entity)"
            ))),
        }
    }
}

/// Check if a record's text content matches the topic query.
fn matches_topic(record: &Record, topic: Option<&str>) -> bool {
    let topic = match topic {
        Some(t) => t,
        None => return true, // no filter
    };
    let topic_lower = topic.to_lowercase();
    // Check common text fields
    for key in &[
        "summary",
        "fact",
        "description",
        "name",
        "entity",
        "content",
        "key",
    ] {
        if let Some(val) = record.data.get(*key).and_then(|v| v.as_str()) {
            if val.to_lowercase().contains(&topic_lower) {
                return true;
            }
        }
    }
    false
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
    fn reflect_empty_db() {
        let (db, _dir) = temp_db();
        let engine = ReflectEngine::new(&db);
        let report = engine.reflect(None, ReflectScope::All).unwrap();
        assert!(!report.insights.is_empty());
        assert!(report.insights[0].contains("No significant patterns"));
    }

    #[test]
    fn reflect_with_entities() {
        let (db, _dir) = temp_db();
        // Store multiple facts for one entity
        for i in 0..4 {
            db.insert(
                TABLE_ENTITIES,
                json!({
                    "entity": "auth-module",
                    "fact": format!("fact {i} about auth"),
                }),
            )
            .unwrap();
        }

        let engine = ReflectEngine::new(&db);
        let report = engine.reflect(Some("auth"), ReflectScope::All).unwrap();
        assert!(report.memories_analyzed >= 4);
        // Should detect the entity has many facts
        assert!(
            report.insights.iter().any(|i| i.contains("auth-module")),
            "insights: {:?}",
            report.insights
        );
    }

    #[test]
    fn reflect_topic_filter() {
        let (db, _dir) = temp_db();
        db.insert(
            TABLE_ENTITIES,
            json!({"entity": "auth", "fact": "uses JWT"}),
        )
        .unwrap();
        db.insert(
            TABLE_ENTITIES,
            json!({"entity": "database", "fact": "uses PostgreSQL"}),
        )
        .unwrap();

        let engine = ReflectEngine::new(&db);
        let report = engine.reflect(Some("JWT"), ReflectScope::All).unwrap();
        // Should only analyze auth-related records
        assert_eq!(report.memories_analyzed, 1);
    }
}
