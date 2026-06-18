//! Belief system — Phase 10.6
//!
//! Beliefs are the agent's high-level understanding of the world.
//! Auto-generated from consolidated high-importance facts, or explicitly stated.
//! Updated when contradicting evidence arrives (superseding).

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{Axil, Record, Result};

const TABLE_BELIEFS: &str = "_beliefs";

/// Minimum importance for a fact to be auto-promoted to a belief.
const BELIEF_IMPORTANCE_THRESHOLD: f32 = 0.8;

/// A belief — something the agent currently holds to be true.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub id: String,
    pub statement: String,
    pub confidence: f32,
    pub source: BeliefSource,
    pub created_at: String,
    pub doubted: bool,
}

/// How a belief was formed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefSource {
    /// Explicitly stated by the agent or user.
    Explicit,
    /// Auto-generated from consolidated facts.
    Consolidated,
    /// Inferred from patterns.
    Inferred,
}

/// Belief system operations on an Axil database.
pub struct BeliefSystem<'a> {
    db: &'a Axil,
}

impl<'a> BeliefSystem<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Explicitly state a belief.
    pub fn believe(&self, statement: &str) -> Result<Record> {
        self.db.insert(
            TABLE_BELIEFS,
            json!({
                "statement": statement,
                "confidence": 1.0,
                "source": "explicit",
                "doubted": false,
                "created_at": Utc::now().to_rfc3339(),
            }),
        )
    }

    /// Mark a belief as doubted (uncertain).
    pub fn doubt(&self, belief_id: &crate::RecordId) -> Result<Record> {
        let record = self
            .db
            .get(belief_id)?
            .ok_or_else(|| crate::AxilError::NotFound(format!("belief {belief_id}")))?;
        if record.table != TABLE_BELIEFS {
            return Err(crate::AxilError::InvalidQuery(format!(
                "record {belief_id} is in table '{}', not '{TABLE_BELIEFS}'",
                record.table
            )));
        }
        let mut data = record.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("doubted".to_string(), json!(true));
            obj.insert("confidence".to_string(), json!(0.5));
        }
        self.db.update(belief_id, data)
    }

    /// Reaffirm a doubted belief.
    pub fn reaffirm(&self, belief_id: &crate::RecordId) -> Result<Record> {
        let record = self
            .db
            .get(belief_id)?
            .ok_or_else(|| crate::AxilError::NotFound(format!("belief {belief_id}")))?;
        if record.table != TABLE_BELIEFS {
            return Err(crate::AxilError::InvalidQuery(format!(
                "record {belief_id} is in table '{}', not '{TABLE_BELIEFS}'",
                record.table
            )));
        }
        let mut data = record.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("doubted".to_string(), json!(false));
            obj.insert("confidence".to_string(), json!(1.0));
        }
        self.db.update(belief_id, data)
    }

    /// List current beliefs, optionally filtering by topic.
    pub fn list(&self, topic: Option<&str>, include_doubted: bool) -> Result<Vec<Belief>> {
        let records = self.db.list(TABLE_BELIEFS)?;
        let mut beliefs: Vec<Belief> = records
            .iter()
            .filter(|r| {
                if !include_doubted {
                    let doubted = r
                        .data
                        .get("doubted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if doubted {
                        return false;
                    }
                }
                if let Some(t) = topic {
                    let stmt = r
                        .data
                        .get("statement")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    return stmt.to_lowercase().contains(&t.to_lowercase());
                }
                true
            })
            .map(|r| record_to_belief(r))
            .collect();
        beliefs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(beliefs)
    }

    /// Auto-generate beliefs from high-importance consolidated facts.
    ///
    /// Scans entity facts with importance >= threshold and creates beliefs
    /// for any that don't already have a matching belief.
    pub fn auto_generate(&self) -> Result<Vec<Record>> {
        let entities = self.db.list("_entities").unwrap_or_default();
        let existing_beliefs = self.db.list(TABLE_BELIEFS).unwrap_or_default();
        let mut seen_statements: std::collections::HashSet<String> = existing_beliefs
            .iter()
            .filter_map(|r| {
                r.data
                    .get("statement")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_lowercase())
            })
            .collect();

        let mut new_beliefs = Vec::new();
        for record in &entities {
            let importance = crate::importance::get_importance(&record.data);
            if importance < BELIEF_IMPORTANCE_THRESHOLD {
                continue;
            }

            let entity = record
                .data
                .get("entity")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let fact = record
                .data
                .get("fact")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if entity.is_empty() || fact.is_empty() {
                continue;
            }

            let statement = format!("{entity}: {fact}");
            let statement_lower = statement.to_lowercase();
            if !seen_statements.insert(statement_lower) {
                continue;
            }

            let belief = self.db.insert(
                TABLE_BELIEFS,
                json!({
                    "statement": statement,
                    "confidence": importance,
                    "source": "consolidated",
                    "doubted": false,
                    "entity": entity,
                    "created_at": Utc::now().to_rfc3339(),
                }),
            )?;
            new_beliefs.push(belief);
        }

        Ok(new_beliefs)
    }
}

fn record_to_belief(r: &Record) -> Belief {
    Belief {
        id: r.id.to_string(),
        statement: r
            .data
            .get("statement")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        confidence: r
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32,
        source: match r.data.get("source").and_then(|v| v.as_str()) {
            Some("explicit") => BeliefSource::Explicit,
            Some("consolidated") => BeliefSource::Consolidated,
            Some("inferred") => BeliefSource::Inferred,
            _ => BeliefSource::Explicit,
        },
        created_at: r
            .data
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        doubted: r
            .data
            .get("doubted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    }
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
    fn believe_and_list() {
        let (db, _dir) = temp_db();
        let bs = BeliefSystem::new(&db);
        bs.believe("auth uses JWT with 15min expiry").unwrap();
        bs.believe("database is PostgreSQL 16").unwrap();

        let beliefs = bs.list(None, false).unwrap();
        assert_eq!(beliefs.len(), 2);
        assert_eq!(beliefs[0].confidence, 1.0);
    }

    #[test]
    fn doubt_reduces_confidence() {
        let (db, _dir) = temp_db();
        let bs = BeliefSystem::new(&db);
        let record = bs.believe("cache uses Redis").unwrap();

        bs.doubt(&record.id).unwrap();
        let beliefs = bs.list(None, true).unwrap();
        let cached = beliefs
            .iter()
            .find(|b| b.statement.contains("Redis"))
            .unwrap();
        assert!(cached.doubted);
        assert_eq!(cached.confidence, 0.5);
    }

    #[test]
    fn doubted_excluded_by_default() {
        let (db, _dir) = temp_db();
        let bs = BeliefSystem::new(&db);
        let r1 = bs.believe("fact A").unwrap();
        bs.believe("fact B").unwrap();
        bs.doubt(&r1.id).unwrap();

        let active = bs.list(None, false).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].statement, "fact B");

        let all = bs.list(None, true).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn topic_filter() {
        let (db, _dir) = temp_db();
        let bs = BeliefSystem::new(&db);
        bs.believe("auth uses JWT").unwrap();
        bs.believe("database uses PostgreSQL").unwrap();

        let auth_beliefs = bs.list(Some("auth"), false).unwrap();
        assert_eq!(auth_beliefs.len(), 1);
        assert!(auth_beliefs[0].statement.contains("auth"));
    }

    #[test]
    fn reaffirm_restores_confidence() {
        let (db, _dir) = temp_db();
        let bs = BeliefSystem::new(&db);
        let r = bs.believe("fact X").unwrap();
        bs.doubt(&r.id).unwrap();
        bs.reaffirm(&r.id).unwrap();

        let beliefs = bs.list(None, false).unwrap();
        assert_eq!(beliefs.len(), 1);
        assert!(!beliefs[0].doubted);
        assert_eq!(beliefs[0].confidence, 1.0);
    }
}
