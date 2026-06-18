//! Graph inference engine — derive new facts from existing edges.
//!
//! No LLM required: uses 2-hop graph pattern matching to infer transitive
//! relationships, impact propagation, and temporal superseding.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::db::Axil;
use crate::error::Result;
use crate::plugin::{Direction, EdgeInfo};
use crate::record::RecordId;

/// A fact inferred by the graph inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredFact {
    /// Human-readable description of the inferred relationship.
    pub fact: String,
    /// Always `"inferred"`.
    pub source: String,
    /// Chain of edge descriptions that led to this inference.
    pub reasoning: Vec<String>,
    /// Confidence score (0.0–1.0). Higher for shorter paths and stronger rules.
    pub confidence: f32,
    /// The derived edge type (e.g. `"works_in"`, `"may_be_affected"`).
    pub derived_edge: String,
    /// Source record ID of the inferred relationship.
    pub from: RecordId,
    /// Target record ID of the inferred relationship.
    pub to: RecordId,
}

/// A pattern-based inference rule.
///
/// Matches a 2-hop path: `from -[pattern[0]]-> mid -[pattern[1]]-> to`
/// and derives a new edge `from -[derived_edge]-> to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    /// Rule name for identification and logging.
    pub name: String,
    /// Edge types and directions to follow. Each element is `(edge_type, direction)`.
    /// Currently supports exactly 2 hops.
    pub pattern: Vec<(String, Direction)>,
    /// The edge type to create when the pattern matches.
    pub derived_edge: String,
    /// Confidence assigned to facts produced by this rule (0.0–1.0).
    pub confidence: f32,
}

/// Graph inference engine that derives new facts from existing edges.
///
/// Runs pattern-based rules over the graph to discover implicit relationships.
/// All inferred facts are stored with `"source": "inferred"` and a reasoning
/// chain so they can be explained via `why()`.
pub struct InferenceEngine<'a> {
    db: &'a Axil,
    rules: Vec<InferenceRule>,
}

/// Built-in inference rules.
fn builtin_rules() -> Vec<InferenceRule> {
    vec![
        InferenceRule {
            name: "transitive_ownership".into(),
            pattern: vec![
                ("manages".into(), Direction::Out),
                ("part_of".into(), Direction::Out),
            ],
            derived_edge: "works_in".into(),
            confidence: 0.8,
        },
        InferenceRule {
            name: "impact_propagation".into(),
            pattern: vec![
                ("depends_on".into(), Direction::Out),
                ("has_issue".into(), Direction::Out),
            ],
            derived_edge: "may_be_affected".into(),
            confidence: 0.6,
        },
        InferenceRule {
            name: "temporal_superseding".into(),
            pattern: vec![("supersedes".into(), Direction::Out)],
            derived_edge: "outdated".into(),
            confidence: 0.9,
        },
    ]
}

impl<'a> InferenceEngine<'a> {
    /// Create an inference engine with built-in rules.
    pub fn new(db: &'a Axil) -> Self {
        Self {
            db,
            rules: builtin_rules(),
        }
    }

    /// Create an inference engine with custom rules (built-in rules are still included).
    pub fn with_rules(db: &'a Axil, extra_rules: Vec<InferenceRule>) -> Self {
        let mut rules = builtin_rules();
        rules.extend(extra_rules);
        Self { db, rules }
    }

    /// Run all inference rules and return newly inferred facts.
    ///
    /// If `entity_id` is `Some`, only check paths starting from that entity.
    /// If `None`, check all records in `_entities`.
    pub fn infer(&self, entity_id: Option<&RecordId>) -> Result<Vec<InferredFact>> {
        if !self.db.has_graph_index() {
            return Ok(Vec::new());
        }

        let start_ids = match entity_id {
            Some(id) => vec![id.clone()],
            None => {
                let entities = self.db.list("_entities")?;
                entities.into_iter().map(|r| r.id).collect()
            }
        };

        let mut facts = Vec::new();
        for id in &start_ids {
            for rule in &self.rules {
                let mut new_facts = self.apply_rule(id, rule)?;
                facts.append(&mut new_facts);
            }
        }

        // Deduplicate: same (from, derived_edge, to) should only appear once
        facts.dedup_by(|a, b| a.from == b.from && a.to == b.to && a.derived_edge == b.derived_edge);

        Ok(facts)
    }

    /// Run inference and store the results as records in `_entities` with
    /// `"source": "inferred"`.
    ///
    /// Returns the inferred facts that were stored. Skips facts whose derived
    /// edge already exists between the two endpoints.
    pub fn infer_and_store(&self, entity_id: Option<&RecordId>) -> Result<Vec<InferredFact>> {
        let facts = self.infer(entity_id)?;
        let mut stored = Vec::new();

        for fact in facts {
            // Check if this derived edge already exists
            if self.edge_exists(&fact.from, &fact.derived_edge, &fact.to)? {
                continue;
            }

            // Store the inferred fact as a record
            let record = self.db.insert(
                "_entities",
                json!({
                    "source": "inferred",
                    "fact": fact.fact,
                    "derived_edge": fact.derived_edge,
                    "from": fact.from.0,
                    "to": fact.to.0,
                    "confidence": fact.confidence,
                    "_inferred": true,
                    "_reasoning": fact.reasoning,
                }),
            )?;

            // Create the derived edge
            self.db.relate(
                &fact.from,
                &fact.derived_edge,
                &fact.to,
                Some(json!({
                    "source": "inferred",
                    "rule": fact.reasoning.first().unwrap_or(&String::new()),
                    "confidence": fact.confidence,
                    "fact_record": record.id.0,
                })),
            )?;

            stored.push(fact);
        }

        Ok(stored)
    }

    /// Explain why a fact was inferred. Returns the reasoning chain if the
    /// record has `"_inferred": true` and `"_reasoning"` in its data.
    pub fn why(&self, fact_id: &RecordId) -> Result<Option<Vec<String>>> {
        let record = match self.db.get(fact_id)? {
            Some(r) => r,
            None => return Ok(None),
        };

        let inferred = record
            .data
            .get("_inferred")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !inferred {
            return Ok(None);
        }

        let reasoning = record
            .data
            .get("_reasoning")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(reasoning))
    }

    /// Confirm an inferred fact — set confidence to 1.0 and source to "confirmed".
    pub fn confirm(&self, fact_id: &RecordId) -> Result<bool> {
        let record = match self.db.get(fact_id)? {
            Some(r) => r,
            None => return Ok(false),
        };

        let inferred = record
            .data
            .get("_inferred")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !inferred {
            return Ok(false);
        }

        let mut data = record.data.clone();
        data["confidence"] = serde_json::json!(1.0);
        data["source"] = serde_json::json!("confirmed");
        self.db.update(fact_id, data)?;
        Ok(true)
    }

    /// Reject an inferred fact — mark as invalid so it won't be re-inferred.
    pub fn reject(&self, fact_id: &RecordId) -> Result<bool> {
        let record = match self.db.get(fact_id)? {
            Some(r) => r,
            None => return Ok(false),
        };

        let inferred = record
            .data
            .get("_inferred")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !inferred {
            return Ok(false);
        }

        let mut data = record.data.clone();
        data["_rejected"] = serde_json::json!(true);
        data["source"] = serde_json::json!("rejected");
        self.db.update(fact_id, data)?;

        // Retract the derived graph edge created by infer_and_store().
        // The edge has a `fact_record` property matching this record's ID.
        if self.db.has_graph_index() {
            if let (Some(from_str), Some(derived_edge)) = (
                record.data.get("from").and_then(|v| v.as_str()),
                record.data.get("derived_edge").and_then(|v| v.as_str()),
            ) {
                if let Ok(from_id) = crate::RecordId::from_string(from_str) {
                    if let Ok(edges) =
                        self.db
                            .edges(&from_id, Some(derived_edge), crate::Direction::Out)
                    {
                        for edge in &edges {
                            let matches =
                                edge.properties.get("fact_record").and_then(|v| v.as_str())
                                    == Some(&fact_id.0);
                            if matches {
                                let _ = self.db.unrelate(&edge.id);
                            }
                        }
                    }
                }
            }
        }

        Ok(true)
    }

    /// Apply a single rule starting from `start_id`.
    fn apply_rule(&self, start_id: &RecordId, rule: &InferenceRule) -> Result<Vec<InferredFact>> {
        if rule.pattern.is_empty() {
            return Ok(Vec::new());
        }

        // Special case: single-hop rules (e.g. temporal superseding)
        if rule.pattern.len() == 1 {
            return self.apply_single_hop(start_id, rule);
        }

        // General case: 2-hop rules
        let (ref edge1, dir1) = rule.pattern[0];
        let (ref edge2, dir2) = rule.pattern[1];

        let mid_edges = self.db.edges(start_id, Some(edge1), dir1)?;
        let mut facts = Vec::new();

        for mid_edge in &mid_edges {
            let mid_id = neighbor_from_edge(&mid_edge, start_id);

            let end_edges = self.db.edges(&mid_id, Some(edge2), dir2)?;
            for end_edge in &end_edges {
                let end_id = neighbor_from_edge(&end_edge, &mid_id);

                // Don't create self-referencing edges
                if *start_id == end_id {
                    continue;
                }

                let reasoning = vec![
                    format!("rule: {}", rule.name),
                    format!("{} -[{}]-> {}", start_id, edge1, mid_id),
                    format!("{} -[{}]-> {}", mid_id, edge2, end_id),
                    format!(
                        "therefore: {} -[{}]-> {}",
                        start_id, rule.derived_edge, end_id
                    ),
                ];

                facts.push(InferredFact {
                    fact: format!(
                        "{} {} {} (via {} through {})",
                        start_id, rule.derived_edge, end_id, rule.name, mid_id
                    ),
                    source: "inferred".into(),
                    reasoning,
                    confidence: rule.confidence,
                    derived_edge: rule.derived_edge.clone(),
                    from: start_id.clone(),
                    to: end_id,
                });
            }
        }

        Ok(facts)
    }

    /// Apply a single-hop rule (e.g. temporal superseding: A →supersedes→ B means B is outdated).
    fn apply_single_hop(
        &self,
        start_id: &RecordId,
        rule: &InferenceRule,
    ) -> Result<Vec<InferredFact>> {
        let (ref edge_type, dir) = rule.pattern[0];
        let edges = self.db.edges(start_id, Some(edge_type), dir)?;
        let mut facts = Vec::new();

        for edge in &edges {
            let target = neighbor_from_edge(edge, start_id);

            let reasoning = vec![
                format!("rule: {}", rule.name),
                format!("{} -[{}]-> {}", start_id, edge_type, target),
                format!("therefore: {} is {}", target, rule.derived_edge),
            ];

            facts.push(InferredFact {
                fact: format!(
                    "{} is {} (superseded by {})",
                    target, rule.derived_edge, start_id
                ),
                source: "inferred".into(),
                reasoning,
                confidence: rule.confidence,
                derived_edge: rule.derived_edge.clone(),
                from: start_id.clone(),
                to: target,
            });
        }

        Ok(facts)
    }

    /// Check if an edge of the given type already exists between two records.
    fn edge_exists(&self, from: &RecordId, edge_type: &str, to: &RecordId) -> Result<bool> {
        let edges = self.db.edges(from, Some(edge_type), Direction::Out)?;
        Ok(edges.iter().any(|e| e.to == *to))
    }
}

/// Given an edge and the ID of one endpoint, return the other endpoint.
fn neighbor_from_edge(edge: &EdgeInfo, from: &RecordId) -> RecordId {
    if edge.from == *from {
        edge.to.clone()
    } else {
        edge.from.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_rules_are_valid() {
        let rules = builtin_rules();
        assert_eq!(rules.len(), 3);
        assert!(rules.iter().any(|r| r.name == "transitive_ownership"));
        assert!(rules.iter().any(|r| r.name == "impact_propagation"));
        assert!(rules.iter().any(|r| r.name == "temporal_superseding"));
        for rule in &rules {
            assert!(!rule.pattern.is_empty());
            assert!(!rule.derived_edge.is_empty());
            assert!(rule.confidence > 0.0 && rule.confidence <= 1.0);
        }
    }

    #[test]
    fn inference_rule_serialization() {
        let rule = InferenceRule {
            name: "test_rule".into(),
            pattern: vec![("mentions".into(), Direction::Out)],
            derived_edge: "related".into(),
            confidence: 0.7,
        };
        let json = serde_json::to_string(&rule).unwrap();
        let back: InferenceRule = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test_rule");
        assert_eq!(back.derived_edge, "related");
    }

    #[test]
    fn inferred_fact_serialization() {
        let fact = InferredFact {
            fact: "A works_in C".into(),
            source: "inferred".into(),
            reasoning: vec!["rule: transitive_ownership".into()],
            confidence: 0.8,
            derived_edge: "works_in".into(),
            from: RecordId("A".into()),
            to: RecordId("C".into()),
        };
        let json = serde_json::to_string(&fact).unwrap();
        assert!(json.contains("inferred"));
        assert!(json.contains("works_in"));
    }

    #[test]
    fn neighbor_from_edge_returns_other() {
        let edge = EdgeInfo {
            id: RecordId("e1".into()),
            from: RecordId("a".into()),
            to: RecordId("b".into()),
            edge_type: "test".into(),
            properties: serde_json::Value::Object(Default::default()),
            created_at: String::new(),
        };
        assert_eq!(
            neighbor_from_edge(&edge, &RecordId("a".into())),
            RecordId("b".into())
        );
        assert_eq!(
            neighbor_from_edge(&edge, &RecordId("b".into())),
            RecordId("a".into())
        );
    }

    #[test]
    fn engine_no_graph_returns_empty() {
        // Create a temp database without graph index
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let engine = InferenceEngine::new(&db);
        let facts = engine.infer(None).unwrap();
        assert!(facts.is_empty());
    }

    #[test]
    fn engine_with_custom_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let custom = vec![InferenceRule {
            name: "custom".into(),
            pattern: vec![
                ("uses".into(), Direction::Out),
                ("provides".into(), Direction::Out),
            ],
            derived_edge: "benefits_from".into(),
            confidence: 0.5,
        }];
        let engine = InferenceEngine::with_rules(&db, custom);
        assert_eq!(engine.rules.len(), 4); // 3 built-in + 1 custom
    }

    #[test]
    fn why_returns_none_for_non_inferred() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let record = db.insert("test", json!({"hello": "world"})).unwrap();
        let engine = InferenceEngine::new(&db);
        let result = engine.why(&record.id).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn why_returns_reasoning_for_inferred() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let record = db
            .insert(
                "_entities",
                json!({
                    "_inferred": true,
                    "_reasoning": ["rule: test", "A -> B", "B -> C"],
                    "source": "inferred",
                    "fact": "A related C",
                }),
            )
            .unwrap();
        let engine = InferenceEngine::new(&db);
        let result = engine.why(&record.id).unwrap();
        assert!(result.is_some());
        let chain = result.unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0], "rule: test");
    }

    #[test]
    fn why_returns_none_for_missing_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let engine = InferenceEngine::new(&db);
        let fake_id = RecordId::new();
        let result = engine.why(&fake_id).unwrap();
        assert!(result.is_none());
    }
}
