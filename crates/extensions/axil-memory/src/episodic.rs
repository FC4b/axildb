//! Episodic memory — past sessions, interactions, and outcomes.
//!
//! Episodes are automatically created when sessions end. They store
//! summaries, outcomes, decisions made, and files touched.
//! Supports dual-granularity embeddings (summary + full-text).

use serde_json::json;

use axil_core::{Axil, Op, Record, Result};

use crate::supersede::set_bitemporal;
use crate::types::{Outcome, EDGE_TOUCHED, TABLE_ENTITIES, TABLE_EPISODES};

/// Episodic memory — completed sessions with outcomes.
pub struct EpisodicMemory<'a> {
    db: &'a Axil,
    /// Agent scope — if set, only episodes tagged with this agent are returned.
    agent: Option<String>,
}

impl<'a> EpisodicMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db, agent: None }
    }

    /// Create an episodic memory scoped to a specific agent.
    pub fn for_agent(db: &'a Axil, agent: &str) -> Self {
        Self {
            db,
            agent: Some(agent.to_string()),
        }
    }

    /// Create an episode from a completed session record.
    ///
    /// Called automatically by `WorkingMemory::end_session()`.
    /// Returns `None` if the session has no summary.
    pub fn create_from_session(&self, session: &Record) -> Result<Option<Record>> {
        let summary = session
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if summary.is_empty() {
            // No summary = no episode. The session record still exists.
            return Ok(None);
        }

        // Build full-text from user turns.
        let full_text = extract_user_turns(session);

        let outcome = session
            .data
            .get("outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("partial")
            .to_string();

        let decisions = session
            .data
            .get("decisions_made")
            .cloned()
            .unwrap_or(json!([]));

        let files = session
            .data
            .get("files_touched")
            .cloned()
            .unwrap_or(json!([]));

        let duration = session
            .data
            .get("duration_secs")
            .cloned()
            .unwrap_or(json!(0));

        let meta = session.data.get("meta").cloned().unwrap_or(json!(null));

        let mut data = json!({
            "summary": summary,
            "outcome": outcome,
            "decisions_made": decisions,
            "files_touched": files,
            "duration_secs": duration,
            "session_id": session.id.to_string(),
            "meta": meta,
        });

        if !full_text.is_empty() {
            data["full_text"] = json!(full_text);
        }

        // Propagate agent tag from session to episode for per-agent isolation.
        if let Some(agent) = session.data.get("_agent") {
            data["_agent"] = agent.clone();
        }

        set_bitemporal(&mut data, Some(session.created_at));

        let episode = self.db.insert(TABLE_EPISODES, data)?;

        if self.db.has_vector_index() {
            let _ = self.db.embed_text(&episode.id, &summary);
        }

        // Link episode to entities mentioned in the summary.
        self.link_to_entities(&episode, &summary)?;

        Ok(Some(episode))
    }

    /// Manually create an episode (not from a session).
    pub fn create(
        &self,
        summary: &str,
        outcome: Outcome,
        decisions: Option<Vec<String>>,
        files: Option<Vec<String>>,
    ) -> Result<Record> {
        let mut data = json!({
            "summary": summary,
            "outcome": outcome,
            "decisions_made": decisions.unwrap_or_default(),
            "files_touched": files.unwrap_or_default(),
        });

        set_bitemporal(&mut data, None);

        let episode = self.db.insert(TABLE_EPISODES, data)?;

        if self.db.has_vector_index() {
            let _ = self.db.embed_text(&episode.id, summary);
        }

        self.link_to_entities(&episode, summary)?;

        Ok(episode)
    }

    /// List episodes, optionally filtered by outcome.
    ///
    /// If this episodic memory is agent-scoped, only episodes for that agent
    /// are returned.
    pub fn list(&self, outcome: Option<Outcome>, limit: usize) -> Result<Vec<Record>> {
        let records = if let Some(o) = outcome {
            self.db
                .query()
                .table(TABLE_EPISODES)
                .where_field("outcome", Op::Eq, json!(o.to_string()))
                .limit(limit)
                .exec()?
        } else {
            let mut all = self.db.list(TABLE_EPISODES)?;
            all.truncate(limit);
            all
        };

        let filtered = crate::ttl::filter_expired(records);

        if let Some(ref agent) = self.agent {
            Ok(filtered
                .into_iter()
                .filter(|r| {
                    r.data
                        .get("_agent")
                        .and_then(|v| v.as_str())
                        .map(|a| a == agent)
                        .unwrap_or(false)
                })
                .collect())
        } else {
            Ok(filtered)
        }
    }

    /// Find similar past episodes using vector search.
    pub fn similar(&self, query: &str, top_k: usize) -> Result<Vec<(Record, f32)>> {
        if !self.db.has_vector_index() {
            return Ok(Vec::new());
        }

        let results = self.db.similar_to(query, top_k * 3)?;
        let mut filtered: Vec<(Record, f32)> = results
            .into_iter()
            .filter(|(r, _)| r.table == TABLE_EPISODES)
            .filter(|(r, _)| !crate::ttl::is_record_expired(r))
            .filter(|(r, _)| !crate::ttl::is_record_superseded(r))
            .collect();

        filtered.truncate(top_k);
        Ok(filtered)
    }

    /// Link an episode to entities mentioned in its text.
    fn link_to_entities(&self, episode: &Record, text: &str) -> Result<()> {
        if !self.db.has_graph_index() {
            return Ok(());
        }

        let entity_records = self.db.list(TABLE_ENTITIES)?;
        let text_lower = text.to_lowercase();

        for entity_record in &entity_records {
            // Skip superseded entity records to avoid stale graph edges.
            if crate::ttl::is_record_superseded(entity_record) {
                continue;
            }
            if let Some(name) = entity_record.data.get("entity").and_then(|v| v.as_str()) {
                if text_lower.contains(&name.to_lowercase()) {
                    let _ = self
                        .db
                        .relate(&episode.id, EDGE_TOUCHED, &entity_record.id, None);
                }
            }
        }

        Ok(())
    }
}

/// Extract user-only turns from a session record for embedding.
///
/// MemPalace research shows that assistant turns pollute search —
/// user turns are more informative for retrieval.
fn extract_user_turns(session: &Record) -> String {
    let turns = match session.data.get("turns").and_then(|v| v.as_array()) {
        Some(t) => t,
        None => return String::new(),
    };

    turns
        .iter()
        .filter(|t| t.get("role").and_then(|v| v.as_str()) == Some("user"))
        .filter_map(|t| t.get("content").and_then(|v| v.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
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
    fn create_episode() {
        let (db, _dir) = temp_db();
        let ep = EpisodicMemory::new(&db);

        let record = ep
            .create(
                "Fixed auth timeout by increasing pool",
                Outcome::Success,
                Some(vec!["Increase pool size".into()]),
                Some(vec!["config.rs".into()]),
            )
            .unwrap();

        assert_eq!(
            record.data["summary"],
            "Fixed auth timeout by increasing pool"
        );
        assert_eq!(record.data["outcome"], "success");
    }

    #[test]
    fn list_episodes_by_outcome() {
        let (db, _dir) = temp_db();
        let ep = EpisodicMemory::new(&db);

        ep.create("Success story", Outcome::Success, None, None)
            .unwrap();
        ep.create("Failure story", Outcome::Failure, None, None)
            .unwrap();

        let successes = ep.list(Some(Outcome::Success), 100).unwrap();
        assert_eq!(successes.len(), 1);

        let all = ep.list(None, 100).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn create_from_session() {
        let (db, _dir) = temp_db();
        let ep = EpisodicMemory::new(&db);

        let session = Record::new(
            "_sessions",
            json!({
                "summary": "Fixed auth bug",
                "outcome": "success",
                "turns": [
                    {"role": "user", "content": "Fix the auth timeout"},
                    {"role": "assistant", "content": "I'll check the config"},
                ],
                "duration_secs": 300,
            }),
        );
        // We need to insert the session first so it has an ID in the DB.
        let session = db.insert("_sessions", session.data.clone()).unwrap();

        let episode = ep.create_from_session(&session).unwrap();
        assert!(episode.is_some());
        let episode = episode.unwrap();
        assert_eq!(episode.data["summary"], "Fixed auth bug");
        assert!(episode.data.get("full_text").is_some());
    }

    #[test]
    fn extract_user_turns_filters() {
        let r = Record::new(
            "_sessions",
            json!({
                "turns": [
                    {"role": "user", "content": "Fix auth"},
                    {"role": "assistant", "content": "Checking..."},
                    {"role": "user", "content": "Also check pool"},
                ],
            }),
        );
        let text = extract_user_turns(&r);
        assert_eq!(text, "Fix auth\nAlso check pool");
    }

    #[test]
    fn no_episode_without_summary() {
        let (db, _dir) = temp_db();
        let ep = EpisodicMemory::new(&db);

        let session = db.insert("_sessions", json!({"status": "ended"})).unwrap();
        let episode = ep.create_from_session(&session).unwrap();
        assert!(episode.is_none());
    }
}
