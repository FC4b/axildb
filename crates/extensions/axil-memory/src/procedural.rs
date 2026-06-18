//! Procedural memory — learned patterns and strategies.
//!
//! Stores approaches that worked (or failed), with confidence scores
//! that strengthen on success and weaken on failure.

use serde_json::json;

use axil_core::{Axil, Op, Record, RecordId, Result};

use crate::supersede::set_bitemporal;
use crate::types::{validate_text, Outcome, EDGE_LEARNED_FROM, TABLE_PROCEDURES};

/// Default initial confidence for a new procedure.
const INITIAL_CONFIDENCE: f64 = 0.5;

/// How much confidence increases on success.
const CONFIDENCE_BOOST: f64 = 0.1;

/// How much confidence decreases on failure.
const CONFIDENCE_PENALTY: f64 = 0.15;

/// Procedural memory — learned patterns and strategies.
pub struct ProceduralMemory<'a> {
    db: &'a Axil,
}

impl<'a> ProceduralMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Store a learned procedure/pattern.
    pub fn learn(
        &self,
        pattern_name: &str,
        description: &str,
        source_episode: Option<&RecordId>,
    ) -> Result<Record> {
        validate_text(pattern_name, "pattern_name")?;
        validate_text(description, "description")?;

        // Check if a procedure with this name already exists.
        if let Some(existing) = self.find_by_name(pattern_name)? {
            // Update existing procedure — boost confidence.
            return self.reinforce(&existing.id, description);
        }

        let mut data = json!({
            "pattern_name": pattern_name,
            "description": description,
            "confidence": INITIAL_CONFIDENCE,
            "applications": 1,
            "successes": 0,
            "failures": 0,
            "partials": 0,
        });

        set_bitemporal(&mut data, None);

        let record = self.db.insert(TABLE_PROCEDURES, data)?;

        if self.db.has_vector_index() {
            let embed_text = format!("{pattern_name}: {description}");
            let _ = self.db.embed_text(&record.id, &embed_text);
        }

        // Link to source episode if provided.
        if let Some(ep_id) = source_episode {
            if self.db.has_graph_index() {
                let _ = self.db.relate(&record.id, EDGE_LEARNED_FROM, ep_id, None);
            }
        }

        Ok(record)
    }

    /// Reinforce an existing procedure with updated info and boosted confidence.
    fn reinforce(&self, id: &RecordId, new_description: &str) -> Result<Record> {
        let record = self
            .db
            .get(id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("procedure {id}")))?;

        let mut data = record.data.clone();
        data["description"] = json!(new_description);

        let confidence = data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(INITIAL_CONFIDENCE);
        data["confidence"] = json!((confidence + CONFIDENCE_BOOST).min(1.0));

        let apps = data
            .get("applications")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        data["applications"] = json!(apps + 1);

        let record = self.db.update(id, data)?;

        if self.db.has_vector_index() {
            let pattern_name = record
                .data
                .get("pattern_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let embed_text = format!("{pattern_name}: {new_description}");
            let _ = self.db.embed_text(id, &embed_text);
        }

        Ok(record)
    }

    /// Record the outcome of applying a procedure.
    pub fn record_outcome(&self, id: &RecordId, outcome: Outcome) -> Result<Record> {
        let record = self
            .db
            .get(id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("procedure {id}")))?;

        let mut data = record.data.clone();
        let confidence = data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(INITIAL_CONFIDENCE);

        match outcome {
            Outcome::Success => {
                data["confidence"] = json!((confidence + CONFIDENCE_BOOST).min(1.0));
                let s = data.get("successes").and_then(|v| v.as_u64()).unwrap_or(0);
                data["successes"] = json!(s + 1);
            }
            Outcome::Failure => {
                data["confidence"] = json!((confidence - CONFIDENCE_PENALTY).max(0.0));
                let f = data.get("failures").and_then(|v| v.as_u64()).unwrap_or(0);
                data["failures"] = json!(f + 1);
            }
            Outcome::Partial => {
                // Slight decrease for partial outcomes.
                data["confidence"] = json!((confidence - CONFIDENCE_PENALTY / 2.0).max(0.0));
                let p = data.get("partials").and_then(|v| v.as_u64()).unwrap_or(0);
                data["partials"] = json!(p + 1);
            }
        }

        let apps = data
            .get("applications")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        data["applications"] = json!(apps + 1);

        self.db.update(id, data)
    }

    /// Find a procedure by name.
    pub fn find_by_name(&self, name: &str) -> Result<Option<Record>> {
        let records = self
            .db
            .query()
            .table(TABLE_PROCEDURES)
            .where_field("pattern_name", Op::Eq, json!(name))
            .limit(1)
            .exec()?;

        Ok(records.into_iter().next())
    }

    /// Find relevant procedures for a task (vector search).
    pub fn how(&self, task: &str, top_k: usize) -> Result<Vec<(Record, f32)>> {
        if !self.db.has_vector_index() {
            return Ok(Vec::new());
        }

        let results = self.db.similar_to(task, top_k * 3)?;
        let mut filtered: Vec<(Record, f32)> = results
            .into_iter()
            .filter(|(r, _)| r.table == TABLE_PROCEDURES)
            .filter(|(r, _)| !crate::ttl::is_record_expired(r))
            .filter(|(r, _)| !crate::ttl::is_record_superseded(r))
            .collect();

        // Sort by confidence-weighted similarity.
        filtered.sort_by(|a, b| {
            let ca =
                a.0.data
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
            let cb =
                b.0.data
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
            let sa = a.1 * ca;
            let sb = b.1 * cb;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        filtered.truncate(top_k);
        Ok(filtered)
    }

    /// List all procedures, sorted by confidence descending.
    pub fn list(&self) -> Result<Vec<Record>> {
        let mut records = self.db.list(TABLE_PROCEDURES)?;
        records.sort_by(|a, b| {
            let ca = a
                .data
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let cb = b
                .data
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(records)
    }

    /// Auto-extract a procedure from a successful episode.
    pub fn extract_from_episode(&self, episode: &Record) -> Result<Option<Record>> {
        let outcome = episode
            .data
            .get("outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("partial");

        if outcome != "success" {
            return Ok(None);
        }

        let summary = episode
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if summary.is_empty() {
            return Ok(None);
        }

        let decisions = episode
            .data
            .get("decisions_made")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default();

        let description = if decisions.is_empty() {
            summary.to_string()
        } else {
            format!("{summary}. Steps: {decisions}")
        };

        // Generate a pattern name from the summary.
        let pattern_name = generate_pattern_name(summary);

        let record = self.learn(&pattern_name, &description, Some(&episode.id))?;
        Ok(Some(record))
    }
}

/// Generate a concise pattern name from a summary.
fn generate_pattern_name(summary: &str) -> String {
    // Take first few significant words, hyphenate.
    let words: Vec<&str> = summary
        .split_whitespace()
        .filter(|w| w.len() > 2) // skip small words
        .take(4)
        .collect();

    if words.is_empty() {
        return "unnamed-pattern".to_string();
    }

    words
        .join("-")
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect()
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
    fn learn_and_find_procedure() {
        let (db, _dir) = temp_db();
        let proc = ProceduralMemory::new(&db);

        let record = proc
            .learn(
                "fix-timeout",
                "Check pool size first, then network config",
                None,
            )
            .unwrap();

        assert_eq!(record.data["pattern_name"], "fix-timeout");
        assert_eq!(record.data["confidence"], 0.5);

        let found = proc.find_by_name("fix-timeout").unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn reinforcement_boosts_confidence() {
        let (db, _dir) = temp_db();
        let proc = ProceduralMemory::new(&db);

        proc.learn("fix-timeout", "v1", None).unwrap();
        let updated = proc.learn("fix-timeout", "v2 improved", None).unwrap();

        assert!(
            updated
                .data
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap()
                > INITIAL_CONFIDENCE
        );
        assert_eq!(updated.data["description"], "v2 improved");
    }

    #[test]
    fn record_outcome_adjusts_confidence() {
        let (db, _dir) = temp_db();
        let proc = ProceduralMemory::new(&db);

        let record = proc.learn("test-proc", "test", None).unwrap();

        // Success boosts.
        let updated = proc.record_outcome(&record.id, Outcome::Success).unwrap();
        let conf = updated
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(conf > INITIAL_CONFIDENCE);

        // Failure penalizes.
        let updated = proc.record_outcome(&record.id, Outcome::Failure).unwrap();
        let conf2 = updated
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(conf2 < conf);
    }

    #[test]
    fn list_sorted_by_confidence() {
        let (db, _dir) = temp_db();
        let proc = ProceduralMemory::new(&db);

        proc.learn("low-conf", "test", None).unwrap();
        proc.learn("high-conf", "test", None).unwrap();
        // Boost high-conf.
        proc.record_outcome(
            &db.list(TABLE_PROCEDURES).unwrap().last().unwrap().id,
            Outcome::Success,
        )
        .unwrap();

        let list = proc.list().unwrap();
        assert_eq!(list.len(), 2);
        let first_conf = list[0]
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap();
        let second_conf = list[1]
            .data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(first_conf >= second_conf);
    }

    #[test]
    fn generate_pattern_name_works() {
        assert_eq!(
            generate_pattern_name("Fixed auth timeout bug"),
            "fixed-auth-timeout-bug"
        );
        assert_eq!(generate_pattern_name("a b"), "unnamed-pattern");
    }

    #[test]
    fn extract_from_episode() {
        use crate::types::TABLE_EPISODES;

        let (db, _dir) = temp_db();
        let proc = ProceduralMemory::new(&db);

        let episode = db
            .insert(
                TABLE_EPISODES,
                json!({
                    "summary": "Fixed timeout by increasing pool",
                    "outcome": "success",
                    "decisions_made": ["Increased pool from 5 to 20"],
                }),
            )
            .unwrap();

        let pattern = proc.extract_from_episode(&episode).unwrap();
        assert!(pattern.is_some());
        let pattern = pattern.unwrap();
        assert!(pattern.data["description"]
            .as_str()
            .unwrap()
            .contains("pool"));
    }
}
