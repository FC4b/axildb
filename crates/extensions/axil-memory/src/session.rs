//! Working memory — session context management.
//!
//! Working memory holds current session state: active tasks, recent tool
//! outputs, open file references, and pending decisions. It auto-transitions
//! to episodic memory when a session ends.

use chrono::Utc;
use serde_json::{json, Value};

use axil_core::{Axil, Direction, Op, Record, RecordId, Result};

use crate::episodic::EpisodicMemory;
use crate::types::{Outcome, EDGE_SESSION_CONTAINS, TABLE_SESSIONS, TABLE_WORKING};

const SESSION_ACTIVE: &str = "active";
const SESSION_ENDED: &str = "ended";

/// Working memory — manages session lifecycle and current context.
pub struct WorkingMemory<'a> {
    db: &'a Axil,
    /// Agent name for per-agent session isolation. If set, sessions are
    /// scoped to this agent. Shared memory types (semantic, procedural,
    /// preference) are NOT affected by agent scoping.
    agent: Option<String>,
}

impl<'a> WorkingMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db, agent: None }
    }

    /// Create a working memory scoped to a specific agent.
    ///
    /// Per-agent sessions are isolated: each agent sees only its own
    /// sessions and working memory. Shared memory types (semantic,
    /// procedural, preference) remain accessible to all agents.
    pub fn for_agent(db: &'a Axil, agent: &str) -> Self {
        Self {
            db,
            agent: Some(agent.to_string()),
        }
    }

    /// Start a new session. Returns the session record.
    ///
    /// If this working memory is agent-scoped, the session is tagged
    /// with `_agent` so it can be filtered per-agent.
    pub fn start_session(&self, meta: Option<Value>) -> Result<Record> {
        let now = Utc::now();
        let mut data = json!({
            "status": SESSION_ACTIVE,
            "started_at": now.to_rfc3339(),
            "record_count": 0,
            "turns": [],
        });

        if let Some(m) = meta {
            data["meta"] = m;
        }

        if let Some(ref agent) = self.agent {
            data["_agent"] = json!(agent);
        }

        self.db.insert(TABLE_SESSIONS, data)
    }

    /// Check that the session belongs to this agent (if agent-scoped).
    fn check_agent_ownership(&self, session: &Record) -> Result<()> {
        if let Some(ref agent) = self.agent {
            let session_agent = session.data.get("_agent").and_then(|v| v.as_str());
            if session_agent != Some(agent) {
                return Err(axil_core::AxilError::InvalidQuery(format!(
                    "session belongs to agent {:?}, not {:?}",
                    session_agent.unwrap_or("<none>"),
                    agent
                )));
            }
        }
        Ok(())
    }

    /// Log a record to a session.
    ///
    /// Inserts the record into the working memory table and links it
    /// to the session via a graph edge.
    pub fn log(&self, session_id: &RecordId, table: &str, data: Value) -> Result<Record> {
        // Verify session exists and is active.
        let session = self
            .db
            .get(session_id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("session {session_id}")))?;

        if session.data.get("status").and_then(|v| v.as_str()) != Some(SESSION_ACTIVE) {
            return Err(axil_core::AxilError::InvalidQuery(
                "cannot log to an ended session".into(),
            ));
        }

        self.check_agent_ownership(&session)?;

        let mut data = data;
        // Stamp session ID so records can be associated without graph.
        data["_session_id"] = json!(session_id.to_string());

        // Tag with agent name if agent-scoped.
        if let Some(ref agent) = self.agent {
            data["_agent"] = json!(agent);
        }

        let record = self.db.insert(table, data)?;

        // Link to session via graph if available.
        if self.db.has_graph_index() {
            self.db
                .relate(session_id, EDGE_SESSION_CONTAINS, &record.id, None)?;
        }

        // Increment record_count.
        let count = session
            .data
            .get("record_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            + 1;
        let mut session_data = session.data.clone();
        session_data["record_count"] = json!(count);
        self.db.update(session_id, session_data)?;

        Ok(record)
    }

    /// Log a turn (user or assistant message) to a session.
    pub fn log_turn(&self, session_id: &RecordId, role: &str, content: &str) -> Result<()> {
        let session = self
            .db
            .get(session_id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("session {session_id}")))?;

        if session.data.get("status").and_then(|v| v.as_str()) != Some(SESSION_ACTIVE) {
            return Err(axil_core::AxilError::InvalidQuery(
                "cannot log turn to an ended session".into(),
            ));
        }

        self.check_agent_ownership(&session)?;

        let mut data = session.data.clone();
        let turn = json!({
            "role": role,
            "content": content,
            "timestamp": Utc::now().to_rfc3339(),
        });

        if let Some(turns) = data.get_mut("turns").and_then(|v| v.as_array_mut()) {
            turns.push(turn);
        } else {
            data["turns"] = json!([turn]);
        }

        self.db.update(session_id, data)?;
        Ok(())
    }

    /// End a session. Transitions working memory → episodic memory.
    ///
    /// Returns the created episode record (if episodic conversion succeeds)
    /// alongside the updated session record.
    pub fn end_session(
        &self,
        session_id: &RecordId,
        summary: Option<&str>,
        outcome: Option<Outcome>,
        decisions_made: Option<Vec<String>>,
        files_touched: Option<Vec<String>>,
    ) -> Result<EndSessionResult> {
        let session = self
            .db
            .get(session_id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("session {session_id}")))?;

        self.check_agent_ownership(&session)?;

        let now = Utc::now();
        let mut data = session.data.clone();
        data["status"] = json!(SESSION_ENDED);
        data["ended_at"] = json!(now.to_rfc3339());

        if let Some(s) = summary {
            data["summary"] = json!(s);
        }

        if let Some(o) = outcome {
            data["outcome"] = json!(o);
        }

        if let Some(d) = &decisions_made {
            data["decisions_made"] = json!(d);
        }

        if let Some(f) = &files_touched {
            data["files_touched"] = json!(f);
        }

        // Count linked records.
        let record_count = if self.db.has_graph_index() {
            self.db
                .neighbors(session_id, Some(EDGE_SESSION_CONTAINS), Direction::Out)
                .map(|n| n.len())
                .unwrap_or(0)
        } else {
            data.get("record_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
        };
        data["record_count"] = json!(record_count);

        // Calculate duration.
        if let Some(started_str) = data.get("started_at").and_then(|v| v.as_str()) {
            if let Ok(started) = chrono::DateTime::parse_from_rfc3339(started_str) {
                let duration_secs = (now - started.with_timezone(&Utc)).num_seconds();
                data["duration_secs"] = json!(duration_secs);
            }
        }

        let updated_session = self.db.update(session_id, data)?;

        if summary.is_some() && self.db.has_vector_index() {
            let _ = self.db.embed_field(session_id, "summary");
        }

        // Create episode from session.
        let episodic = EpisodicMemory::new(self.db);
        let episode = episodic.create_from_session(&updated_session)?;

        // Clear working memory records linked to this session.
        self.clear_working_memory(session_id)?;

        Ok(EndSessionResult {
            session: updated_session,
            episode,
        })
    }

    /// List sessions, optionally filtering to active-only.
    ///
    /// If this working memory is agent-scoped, only sessions for that agent
    /// are returned.
    pub fn list_sessions(&self, active_only: bool) -> Result<Vec<Record>> {
        let all = if active_only {
            self.db
                .query()
                .table(TABLE_SESSIONS)
                .where_field("status", Op::Eq, json!(SESSION_ACTIVE))
                .exec()?
        } else {
            self.db.list(TABLE_SESSIONS)?
        };

        if let Some(ref agent) = self.agent {
            Ok(all
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
            Ok(all)
        }
    }

    /// Get all records linked to a session.
    pub fn session_history(&self, session_id: &RecordId) -> Result<Vec<Record>> {
        if self.db.has_graph_index() {
            self.db
                .neighbors(session_id, Some(EDGE_SESSION_CONTAINS), Direction::Out)
        } else {
            Ok(Vec::new())
        }
    }

    /// Clear working memory records linked to a session.
    ///
    /// With graph: deletes only records linked to this session via edges.
    /// Without graph: deletes working records whose `session_id` field matches.
    fn clear_working_memory(&self, session_id: &RecordId) -> Result<()> {
        let working_records = self.db.list(TABLE_WORKING)?;
        let sid_str = session_id.to_string();

        if self.db.has_graph_index() {
            let edges = self
                .db
                .edges(session_id, Some(EDGE_SESSION_CONTAINS), Direction::Out)?;
            for record in working_records {
                if edges.iter().any(|e| e.to == record.id) {
                    self.db.delete(&record.id)?;
                }
            }
        } else {
            // Without graph: match on session_id field stamped by log().
            for record in working_records {
                let belongs = record
                    .data
                    .get("_session_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s == sid_str)
                    .unwrap_or(false);
                if belongs {
                    self.db.delete(&record.id)?;
                }
            }
        }

        Ok(())
    }
}

/// Result of ending a session.
pub struct EndSessionResult {
    /// The updated (ended) session record.
    pub session: Record,
    /// The episode created from this session (if conversion succeeded).
    pub episode: Option<Record>,
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
    fn start_and_list_session() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::new(&db);

        let session = wm.start_session(Some(json!({"task": "fix auth"}))).unwrap();
        assert_eq!(session.data["status"], "active");

        let sessions = wm.list_sessions(true).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session.id);
    }

    #[test]
    fn log_to_session() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::new(&db);

        let session = wm.start_session(None).unwrap();
        let record = wm
            .log(
                &session.id,
                TABLE_WORKING,
                json!({"tool": "grep", "result": "found"}),
            )
            .unwrap();

        assert_eq!(record.table, TABLE_WORKING);

        // Session record_count should be incremented.
        let updated = db.get(&session.id).unwrap().unwrap();
        assert_eq!(updated.data["record_count"], 1);
    }

    #[test]
    fn log_turn_to_session() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::new(&db);

        let session = wm.start_session(None).unwrap();
        wm.log_turn(&session.id, "user", "Fix the auth timeout")
            .unwrap();
        wm.log_turn(&session.id, "assistant", "I'll check the pool config")
            .unwrap();

        let updated = db.get(&session.id).unwrap().unwrap();
        let turns = updated.data["turns"].as_array().unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[1]["role"], "assistant");
    }

    #[test]
    fn end_session_transitions() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::new(&db);

        let session = wm.start_session(None).unwrap();
        let result = wm
            .end_session(
                &session.id,
                Some("Fixed auth timeout"),
                Some(Outcome::Success),
                Some(vec!["Increased pool size".into()]),
                Some(vec!["config.rs".into()]),
            )
            .unwrap();

        assert_eq!(result.session.data["status"], "ended");
        assert!(result.session.data.get("ended_at").is_some());
        assert!(result.session.data.get("duration_secs").is_some());

        // Episode should be created.
        assert!(result.episode.is_some());
        let episode = result.episode.unwrap();
        assert_eq!(episode.data["outcome"], "success");
    }

    #[test]
    fn per_agent_sessions_isolated() {
        let (db, _dir) = temp_db();
        let wm_alice = WorkingMemory::for_agent(&db, "alice");
        let wm_bob = WorkingMemory::for_agent(&db, "bob");

        wm_alice.start_session(None).unwrap();
        wm_bob.start_session(None).unwrap();
        wm_bob.start_session(None).unwrap();

        // Alice sees only her session.
        let alice_sessions = wm_alice.list_sessions(false).unwrap();
        assert_eq!(alice_sessions.len(), 1);
        assert_eq!(alice_sessions[0].data["_agent"], "alice");

        // Bob sees only his sessions.
        let bob_sessions = wm_bob.list_sessions(false).unwrap();
        assert_eq!(bob_sessions.len(), 2);

        // Unscoped sees all.
        let wm_all = WorkingMemory::new(&db);
        let all_sessions = wm_all.list_sessions(false).unwrap();
        assert_eq!(all_sessions.len(), 3);
    }

    #[test]
    fn per_agent_log_tags_records() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::for_agent(&db, "agent-1");

        let session = wm.start_session(None).unwrap();
        let record = wm
            .log(&session.id, TABLE_WORKING, json!({"tool": "read"}))
            .unwrap();
        assert_eq!(record.data["_agent"], "agent-1");
    }

    #[test]
    fn cannot_log_to_ended_session() {
        let (db, _dir) = temp_db();
        let wm = WorkingMemory::new(&db);

        let session = wm.start_session(None).unwrap();
        wm.end_session(&session.id, None, None, None, None).unwrap();

        let result = wm.log(&session.id, TABLE_WORKING, json!({"test": true}));
        assert!(result.is_err());
    }

    #[test]
    fn cross_agent_log_rejected() {
        let (db, _dir) = temp_db();
        let wm_alice = WorkingMemory::for_agent(&db, "alice");
        let wm_bob = WorkingMemory::for_agent(&db, "bob");

        let alice_session = wm_alice.start_session(None).unwrap();

        // Bob should not be able to log to Alice's session.
        let result = wm_bob.log(&alice_session.id, TABLE_WORKING, json!({"hack": true}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not"));
    }

    #[test]
    fn cross_agent_end_session_rejected() {
        let (db, _dir) = temp_db();
        let wm_alice = WorkingMemory::for_agent(&db, "alice");
        let wm_bob = WorkingMemory::for_agent(&db, "bob");

        let alice_session = wm_alice.start_session(None).unwrap();

        // Bob should not be able to end Alice's session.
        let result = wm_bob.end_session(&alice_session.id, Some("hijacked"), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn unscoped_can_access_any_session() {
        let (db, _dir) = temp_db();
        let wm_alice = WorkingMemory::for_agent(&db, "alice");
        let wm_unscoped = WorkingMemory::new(&db);

        let alice_session = wm_alice.start_session(None).unwrap();

        // Unscoped working memory can log to any session.
        let result = wm_unscoped.log(&alice_session.id, TABLE_WORKING, json!({"ok": true}));
        assert!(result.is_ok());
    }
}
