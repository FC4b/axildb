//! Autonomous pattern recognition — detects recurring patterns across sessions.
//!
//! Pattern types:
//! - **Repeated failures**: same error type appearing across multiple sessions
//! - **Hot spots**: files/modules modified frequently
//! - **Knowledge gaps**: entities mentioned but without stored facts
//! - **Workflow patterns**: common sequences of actions

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Record, Result};

use crate::ttl;
use crate::types::{TABLE_ENTITIES, TABLE_EPISODES};

/// Table name for detected patterns.
pub const TABLE_PATTERNS: &str = "_patterns";

/// A detected pattern across memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique pattern identifier.
    pub name: String,
    /// Pattern type.
    pub pattern_type: PatternType,
    /// Human-readable description.
    pub description: String,
    /// How many times this pattern was observed.
    pub frequency: usize,
    /// First observation date.
    pub first_seen: String,
    /// Most recent observation date.
    pub last_seen: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Suggested action.
    pub suggestion: Option<String>,
    /// Whether the user has dismissed this pattern.
    pub dismissed: bool,
}

impl Pattern {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "pattern_type": self.pattern_type.as_str(),
            "description": self.description,
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "dismissed": self.dismissed,
        })
    }
}

/// Types of patterns the engine can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    /// Same error recurring across sessions.
    RepeatedFailure,
    /// Files/modules frequently modified.
    HotSpot,
    /// Entities mentioned without stored facts.
    KnowledgeGap,
    /// Common action sequences.
    Workflow,
}

impl PatternType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PatternType::RepeatedFailure => "repeated_failure",
            PatternType::HotSpot => "hot_spot",
            PatternType::KnowledgeGap => "knowledge_gap",
            PatternType::Workflow => "workflow",
        }
    }
}

impl std::str::FromStr for PatternType {
    type Err = axil_core::AxilError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "repeated_failure" => Ok(PatternType::RepeatedFailure),
            "hot_spot" => Ok(PatternType::HotSpot),
            "knowledge_gap" => Ok(PatternType::KnowledgeGap),
            "workflow" => Ok(PatternType::Workflow),
            other => Err(axil_core::AxilError::InvalidQuery(format!(
                "unknown pattern type: {other}"
            ))),
        }
    }
}

/// Pattern detection engine.
pub struct PatternEngine<'a> {
    db: &'a Axil,
    min_frequency: usize,
}

impl<'a> PatternEngine<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self {
            db,
            min_frequency: 3,
        }
    }

    /// Set minimum frequency threshold for pattern detection.
    pub fn with_min_frequency(mut self, min: usize) -> Self {
        self.min_frequency = min;
        self
    }

    /// Detect all patterns across the database.
    pub fn detect(&self) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();
        patterns.extend(self.detect_repeated_failures()?);
        patterns.extend(self.detect_hot_spots()?);
        patterns.extend(self.detect_knowledge_gaps()?);
        Ok(patterns)
    }

    /// Detect patterns of a specific type.
    pub fn detect_type(&self, pattern_type: PatternType) -> Result<Vec<Pattern>> {
        match pattern_type {
            PatternType::RepeatedFailure => self.detect_repeated_failures(),
            PatternType::HotSpot => self.detect_hot_spots(),
            PatternType::KnowledgeGap => self.detect_knowledge_gaps(),
            PatternType::Workflow => Ok(Vec::new()), // TODO: workflow detection
        }
    }

    /// Store detected patterns to the database.
    pub fn store_patterns(&self, patterns: &[Pattern]) -> Result<usize> {
        let existing = self.db.list(TABLE_PATTERNS)?;
        let existing_names: std::collections::HashSet<&str> = existing
            .iter()
            .filter_map(|r| r.data.get("name").and_then(|v| v.as_str()))
            .collect();

        let mut stored = 0;
        for pattern in patterns {
            if pattern.dismissed || existing_names.contains(pattern.name.as_str()) {
                continue;
            }
            self.db.insert(TABLE_PATTERNS, pattern.to_json())?;
            stored += 1;
        }
        Ok(stored)
    }

    /// List stored patterns, optionally filtered by type.
    pub fn list(&self, pattern_type: Option<PatternType>) -> Result<Vec<Pattern>> {
        let records = self.db.list(TABLE_PATTERNS)?;
        let patterns: Vec<Pattern> = records
            .iter()
            .filter_map(|r| serde_json::from_value(r.data.clone()).ok())
            .filter(|p: &Pattern| {
                if let Some(pt) = pattern_type {
                    p.pattern_type == pt
                } else {
                    true
                }
            })
            .filter(|p| !p.dismissed)
            .collect();
        Ok(patterns)
    }

    /// Dismiss a pattern by name (stop surfacing it).
    pub fn dismiss(&self, pattern_name: &str) -> Result<bool> {
        let records = self.db.list(TABLE_PATTERNS)?;
        for r in &records {
            if r.data.get("name").and_then(|v| v.as_str()) == Some(pattern_name) {
                let mut data = r.data.clone();
                data["dismissed"] = json!(true);
                self.db.update(&r.id, data)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Detect repeated failure patterns from episodes.
    fn detect_repeated_failures(&self) -> Result<Vec<Pattern>> {
        let episodes = self.db.list(TABLE_EPISODES).unwrap_or_default();
        let failures: Vec<&Record> = episodes
            .iter()
            .filter(|r| !ttl::is_record_expired(r))
            .filter(|r| r.data.get("outcome").and_then(|v| v.as_str()) == Some("failure"))
            .collect();

        if failures.len() < self.min_frequency {
            return Ok(Vec::new());
        }

        // Group failures by keywords in summary
        let mut keyword_groups: HashMap<String, Vec<&Record>> = HashMap::new();
        for f in &failures {
            let summary = f.data.get("summary").and_then(|v| v.as_str()).unwrap_or("");
            // Extract significant words (>4 chars, lowercase)
            for word in summary.split_whitespace() {
                let w = word
                    .to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
                if w.len() > 4 {
                    keyword_groups.entry(w).or_default().push(f);
                }
            }
        }

        let mut patterns = Vec::new();
        for (keyword, records) in &keyword_groups {
            if records.len() >= self.min_frequency {
                let first = records.iter().map(|r| r.created_at).min().unwrap();
                let last = records.iter().map(|r| r.created_at).max().unwrap();
                patterns.push(Pattern {
                    name: format!("repeated_failure_{keyword}"),
                    pattern_type: PatternType::RepeatedFailure,
                    description: format!(
                        "Failures mentioning \"{keyword}\" occurred {} times",
                        records.len()
                    ),
                    frequency: records.len(),
                    first_seen: first.to_rfc3339(),
                    last_seen: last.to_rfc3339(),
                    confidence: (records.len() as f32 / failures.len() as f32).min(1.0),
                    suggestion: Some(format!("Investigate recurring {keyword}-related failures")),
                    dismissed: false,
                });
            }
        }

        // Deduplicate — keep highest frequency
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns.truncate(5);
        Ok(patterns)
    }

    /// Detect hot spots — entities mentioned frequently.
    fn detect_hot_spots(&self) -> Result<Vec<Pattern>> {
        let entities = self.db.list(TABLE_ENTITIES).unwrap_or_default();
        let active: Vec<&Record> = entities
            .iter()
            .filter(|r| !ttl::is_record_superseded(r) && !ttl::is_record_expired(r))
            .collect();

        let mut entity_count: HashMap<String, usize> = HashMap::new();
        for r in &active {
            if let Some(name) = r.data.get("entity").and_then(|v| v.as_str()) {
                *entity_count.entry(name.to_string()).or_default() += 1;
            }
        }

        let mut patterns = Vec::new();
        let mut sorted: Vec<_> = entity_count.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (entity, count) in sorted.iter().take(5) {
            if **count >= self.min_frequency {
                // Find date range
                let entity_records: Vec<&&Record> = active
                    .iter()
                    .filter(|r| {
                        r.data.get("entity").and_then(|v| v.as_str()) == Some(entity.as_str())
                    })
                    .collect();
                let first = entity_records.iter().map(|r| r.created_at).min().unwrap();
                let last = entity_records.iter().map(|r| r.created_at).max().unwrap();

                patterns.push(Pattern {
                    name: format!("hot_spot_{entity}"),
                    pattern_type: PatternType::HotSpot,
                    description: format!(
                        "{entity} has {count} facts — frequently referenced",
                    ),
                    frequency: **count,
                    first_seen: first.to_rfc3339(),
                    last_seen: last.to_rfc3339(),
                    confidence: (**count as f32 / 10.0).min(1.0),
                    suggestion: Some(format!(
                        "Consider consolidating {entity} knowledge with `axil consolidate --entity {entity}`",
                    )),
                    dismissed: false,
                });
            }
        }

        Ok(patterns)
    }

    /// Detect knowledge gaps — entities mentioned in episodes but not in semantic memory.
    fn detect_knowledge_gaps(&self) -> Result<Vec<Pattern>> {
        let episodes = self.db.list(TABLE_EPISODES).unwrap_or_default();
        let entities = self.db.list(TABLE_ENTITIES).unwrap_or_default();

        // Collect known entity names
        let known_entities: std::collections::HashSet<String> = entities
            .iter()
            .filter_map(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();

        // Extract entity-like words from episode summaries
        let mut mentioned: HashMap<String, usize> = HashMap::new();
        for ep in &episodes {
            let summary = ep
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let extracted = axil_core::extract_entities(summary);
            for entity in extracted {
                if !known_entities.contains(&entity.name) {
                    *mentioned.entry(entity.name).or_default() += 1;
                }
            }
        }

        let mut patterns = Vec::new();
        let mut sorted: Vec<_> = mentioned.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (entity, count) in sorted.iter().take(5) {
            if **count >= self.min_frequency {
                patterns.push(Pattern {
                    name: format!("knowledge_gap_{entity}"),
                    pattern_type: PatternType::KnowledgeGap,
                    description: format!(
                        "{entity} mentioned in {count} episodes but has no stored facts",
                    ),
                    frequency: **count,
                    first_seen: String::new(),
                    last_seen: String::new(),
                    confidence: (**count as f32 / 5.0).min(1.0),
                    suggestion: Some(format!(
                        "Store knowledge about {entity} with `axil know \"{entity}\" \"...\"`",
                    )),
                    dismissed: false,
                });
            }
        }

        Ok(patterns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn detect_empty_db() {
        let (db, _dir) = temp_db();
        let engine = PatternEngine::new(&db);
        let patterns = engine.detect().unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn detect_hot_spots() {
        let (db, _dir) = temp_db();
        for i in 0..5 {
            db.insert(
                TABLE_ENTITIES,
                json!({
                    "entity": "auth-module",
                    "fact": format!("fact {i}"),
                }),
            )
            .unwrap();
        }

        let engine = PatternEngine::new(&db).with_min_frequency(3);
        let patterns = engine.detect_type(PatternType::HotSpot).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].pattern_type, PatternType::HotSpot);
        assert!(patterns[0].name.contains("auth-module"));
    }

    #[test]
    fn detect_repeated_failures() {
        let (db, _dir) = temp_db();
        for i in 0..4 {
            db.insert(
                TABLE_EPISODES,
                json!({
                    "summary": format!("timeout error in deploy pipeline round {i}"),
                    "outcome": "failure",
                }),
            )
            .unwrap();
        }

        let engine = PatternEngine::new(&db).with_min_frequency(3);
        let patterns = engine.detect_type(PatternType::RepeatedFailure).unwrap();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].pattern_type, PatternType::RepeatedFailure);
    }

    #[test]
    fn store_and_list_patterns() {
        let (db, _dir) = temp_db();
        let engine = PatternEngine::new(&db);

        let pattern = Pattern {
            name: "test_pattern".into(),
            pattern_type: PatternType::HotSpot,
            description: "test".into(),
            frequency: 5,
            first_seen: Utc::now().to_rfc3339(),
            last_seen: Utc::now().to_rfc3339(),
            confidence: 0.8,
            suggestion: None,
            dismissed: false,
        };

        let stored = engine.store_patterns(&[pattern]).unwrap();
        assert_eq!(stored, 1);

        let listed = engine.list(None).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "test_pattern");
    }

    #[test]
    fn dismiss_pattern() {
        let (db, _dir) = temp_db();
        let engine = PatternEngine::new(&db);

        let pattern = Pattern {
            name: "dismissable".into(),
            pattern_type: PatternType::Workflow,
            description: "test".into(),
            frequency: 3,
            first_seen: String::new(),
            last_seen: String::new(),
            confidence: 0.5,
            suggestion: None,
            dismissed: false,
        };
        engine.store_patterns(&[pattern]).unwrap();

        assert!(engine.dismiss("dismissable").unwrap());

        let listed = engine.list(None).unwrap();
        assert!(listed.is_empty(), "dismissed patterns should be hidden");
    }
}
