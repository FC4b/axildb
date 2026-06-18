//! Intent-native write APIs — a thin layer above `Axil::insert`.
//!
//! Agents don't want to hand-craft JSON and remember which table a
//! "decision" goes into. They want one call per intent, with auto-embed,
//! idempotency, and supersede baked in. This module is that layer.
//!
//! ## Idempotency
//!
//! Every `remember_*` call produces a stable record id when callers supply
//! both `agent_id` and `external_id` — a second call with the same pair
//! returns the existing record with `is_new = false`.
//!
//! When either id is missing, Axil falls back to a content-hash on
//! (table, agent_id, summary/error text) with a 5-minute dedup window —
//! so an agent that fires the same observation twice in quick succession
//! from a retry loop won't double-write, but an intentional rewrite
//! minutes later will be allowed through.
//!
//! Callers who want to bypass both dedup paths set `force_new = true`.
//!
//! ## Metadata
//!
//! Every record written through this module carries a normalized header:
//!
//! | field              | value                                       |
//! |--------------------|---------------------------------------------|
//! | `_auto_captured`   | `false` (auto-capture sets `true`)          |
//! | `_source`          | `"axil-core" | "axil-cli" | "axil-mcp"`     |
//! | `_agent_id`        | optional agent identifier                   |
//! | `_external_id`     | optional caller-supplied idempotency key    |
//! | `_content_hash`    | blake3 of the canonical fields              |
//! | `_importance`      | computed via `importance::compute_importance` |
//!
//! The normalized metadata is what makes the idempotency + supersede
//! checks cheap: `remember_*` asks the table for records with a matching
//! `(agent_id, external_id)` or `content_hash` before it inserts.

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use crate::db::Axil;
use crate::error::Result;
use crate::record::{Record, RecordId};

/// Which surface invoked the write. Used to tag records for
/// observability — not used for routing or dedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteSource {
    Core,
    Cli,
    Mcp,
}

impl WriteSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Core => "axil-core",
            Self::Cli => "axil-cli",
            Self::Mcp => "axil-mcp",
        }
    }
}

/// Window within which an identical content-hash write is treated as a
/// dedup rather than a new record. Longer-than-5-min re-writes are
/// assumed intentional (e.g. revisiting a decision with new wording).
const CONTENT_HASH_WINDOW: Duration = Duration::minutes(5);

/// Input to `remember_decision`.
#[derive(Debug, Clone)]
pub struct DecisionInput<'a> {
    /// What was decided. Required.
    pub summary: &'a str,
    /// Why it was decided. Optional but recommended.
    pub reason: Option<&'a str>,
    /// Files affected by the decision.
    pub files: Option<&'a [&'a str]>,
    /// Agent identifier (part of the idempotency key when paired with
    /// `external_id`).
    pub agent_id: Option<&'a str>,
    /// Caller-supplied idempotency key. When paired with `agent_id`
    /// the two form a unique tuple — repeated calls return the
    /// existing record.
    pub external_id: Option<&'a str>,
    /// Bypass both idempotency paths. Use when you intentionally want a
    /// new record for text that duplicates an earlier write.
    pub force_new: bool,
    /// Which Axil surface originated the write.
    pub source: WriteSource,
}

/// Input to `remember_error`.
#[derive(Debug, Clone)]
pub struct ErrorInput<'a> {
    pub error: &'a str,
    pub root_cause: Option<&'a str>,
    pub fix: Option<&'a str>,
    pub files: Option<&'a [&'a str]>,
    pub agent_id: Option<&'a str>,
    pub external_id: Option<&'a str>,
    pub force_new: bool,
    pub source: WriteSource,
}

/// Result from any `remember_*` call.
#[derive(Debug, Clone)]
pub struct RememberResult {
    /// Record id — of the newly-inserted row OR of the pre-existing
    /// row returned by the dedup path.
    pub id: RecordId,
    /// `true` when a new record was inserted; `false` when the call was
    /// deduped onto an existing one.
    pub is_new: bool,
    /// Records that were marked superseded as a side effect of this
    /// write. The superseding machinery is shared with normal inserts —
    /// we don't run additional semantic checks on top.
    pub superseded: Vec<RecordId>,
}

impl Axil {
    /// Record an architectural / implementation decision.
    ///
    /// Writes to the `decisions` table with normalized metadata and
    /// honors the crate-level idempotency contract (see module docs).
    pub fn remember_decision(&self, input: DecisionInput<'_>) -> Result<RememberResult> {
        let hash = content_hash(&["decisions", input.agent_id.unwrap_or(""), input.summary]);

        if !input.force_new {
            if let Some(existing) =
                self.find_remember_dup("decisions", input.agent_id, input.external_id, &hash)?
            {
                return Ok(RememberResult {
                    id: existing,
                    is_new: false,
                    superseded: Vec::new(),
                });
            }
        }

        let mut data = json!({
            "summary": input.summary,
        });
        if let Some(r) = input.reason {
            data["reason"] = Value::String(r.to_string());
        }
        if let Some(files) = input.files {
            data["files"] = json!(files);
        }
        inject_metadata(
            &mut data,
            input.source,
            input.agent_id,
            input.external_id,
            &hash,
        );

        let record = self.insert("decisions", data)?;
        let superseded = self.collect_superseded_for(&record);
        Ok(RememberResult {
            id: record.id,
            is_new: true,
            superseded,
        })
    }

    /// Record an error and (optionally) what caused and fixed it.
    ///
    /// Writes to the `errors` table. Same idempotency + metadata rules
    /// as `remember_decision`.
    pub fn remember_error(&self, input: ErrorInput<'_>) -> Result<RememberResult> {
        let hash = content_hash(&["errors", input.agent_id.unwrap_or(""), input.error]);

        if !input.force_new {
            if let Some(existing) =
                self.find_remember_dup("errors", input.agent_id, input.external_id, &hash)?
            {
                return Ok(RememberResult {
                    id: existing,
                    is_new: false,
                    superseded: Vec::new(),
                });
            }
        }

        let mut data = json!({
            "error": input.error,
        });
        if let Some(c) = input.root_cause {
            data["root_cause"] = Value::String(c.to_string());
        }
        if let Some(f) = input.fix {
            data["fix"] = Value::String(f.to_string());
        }
        if let Some(files) = input.files {
            data["files"] = json!(files);
        }
        inject_metadata(
            &mut data,
            input.source,
            input.agent_id,
            input.external_id,
            &hash,
        );

        let record = self.insert("errors", data)?;
        let superseded = self.collect_superseded_for(&record);
        Ok(RememberResult {
            id: record.id,
            is_new: true,
            superseded,
        })
    }

    /// Set a user preference. Writes to `preferences`, overwriting any
    /// existing value for the same `key`.
    ///
    /// The previous value is preserved on the new record as
    /// `_previous_value` for lightweight auditability. A full history
    /// table is out of scope for this round.
    pub fn set_preference(&self, key: &str, value: Value) -> Result<RememberResult> {
        let existing = self.find_preference_by_key(key)?;
        let previous_value = existing.as_ref().and_then(|r| r.data.get("value").cloned());

        let mut data = json!({
            "key": key,
            "value": value,
        });
        if let Some(prev) = previous_value {
            data["_previous_value"] = prev;
        }
        let hash = content_hash(&["preferences", key]);
        inject_metadata(&mut data, WriteSource::Core, None, None, &hash);

        // Delete the previous preference record, if any — set-preference
        // is overwrite semantics, not append.
        if let Some(prev_rec) = existing {
            let _ = self.delete(&prev_rec.id);
        }

        let record = self.insert("preferences", data)?;
        Ok(RememberResult {
            id: record.id,
            is_new: true,
            superseded: Vec::new(),
        })
    }

    /// Mark a session as closed, optionally attaching a final summary.
    ///
    /// The session record is keyed by `id`, so a repeated call
    /// (e.g. on crash recovery) returns the existing closed session
    /// rather than creating duplicates.
    pub fn close_session(&self, id: &str, summary: Option<&str>) -> Result<RememberResult> {
        let hash = content_hash(&["sessions", id]);
        // Look for an existing session record with this id. If the
        // caller already closed it we return `is_new=false` and leave
        // the record alone.
        if let Some(existing) = self.find_session_by_external_id(id)? {
            return Ok(RememberResult {
                id: existing,
                is_new: false,
                superseded: Vec::new(),
            });
        }

        let mut data = json!({
            "session_id": id,
            "closed_at": Utc::now().to_rfc3339(),
        });
        if let Some(s) = summary {
            data["summary"] = Value::String(s.to_string());
        }
        inject_metadata(&mut data, WriteSource::Core, None, Some(id), &hash);

        let record = self.insert("sessions", data)?;
        Ok(RememberResult {
            id: record.id,
            is_new: true,
            superseded: Vec::new(),
        })
    }

    // ── Private helpers ─────────────────────────────────────────────

    fn find_remember_dup(
        &self,
        table: &str,
        agent_id: Option<&str>,
        external_id: Option<&str>,
        content_hash: &str,
    ) -> Result<Option<RecordId>> {
        let records = self.list(table).unwrap_or_default();
        let now = Utc::now();

        // Strong match: (agent_id, external_id) pair always dedupes.
        if let (Some(aid), Some(eid)) = (agent_id, external_id) {
            for record in &records {
                let stored_agent = record.data.get("_agent_id").and_then(|v| v.as_str());
                let stored_ext = record.data.get("_external_id").and_then(|v| v.as_str());
                if stored_agent == Some(aid) && stored_ext == Some(eid) {
                    return Ok(Some(record.id.clone()));
                }
            }
        }

        // Weak match: content-hash within the 5-min window.
        for record in &records {
            let stored_hash = record.data.get("_content_hash").and_then(|v| v.as_str());
            if stored_hash == Some(content_hash) && within_window(&record.created_at, &now) {
                return Ok(Some(record.id.clone()));
            }
        }

        Ok(None)
    }

    fn find_preference_by_key(&self, key: &str) -> Result<Option<Record>> {
        let records = self.list("preferences").unwrap_or_default();
        for record in records {
            if record.data.get("key").and_then(|v| v.as_str()) == Some(key) {
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    fn find_session_by_external_id(&self, external_id: &str) -> Result<Option<RecordId>> {
        let records = self.list("sessions").unwrap_or_default();
        for record in &records {
            if record.data.get("_external_id").and_then(|v| v.as_str()) == Some(external_id) {
                return Ok(Some(record.id.clone()));
            }
        }
        Ok(None)
    }

    /// Best-effort collection of records superseded as a side effect of
    /// a write. The real supersede logic runs inside `insert` /
    /// `insert_batch`; here we just surface the ids so callers can show
    /// them in the `RememberResult`. A storage-level change-set would be
    /// more precise; this is the ergonomic version.
    fn collect_superseded_for(&self, new_record: &Record) -> Vec<RecordId> {
        // Scan the same table for records marked `_superseded_by` this
        // new record. O(n) on the table — acceptable for the agent-memory
        // sizes we target.
        let Ok(records) = self.list(&new_record.table) else {
            return Vec::new();
        };
        records
            .into_iter()
            .filter(|r| {
                r.data.get("_superseded_by").and_then(|v| v.as_str())
                    == Some(new_record.id.as_str())
            })
            .map(|r| r.id)
            .collect()
    }
}

fn within_window(record_time: &DateTime<Utc>, now: &DateTime<Utc>) -> bool {
    *now - *record_time <= CONTENT_HASH_WINDOW
}

fn inject_metadata(
    data: &mut Value,
    source: WriteSource,
    agent_id: Option<&str>,
    external_id: Option<&str>,
    content_hash: &str,
) {
    let Some(obj) = data.as_object_mut() else {
        return;
    };
    obj.insert("_auto_captured".into(), json!(false));
    obj.insert("_source".into(), json!(source.as_str()));
    if let Some(aid) = agent_id {
        obj.insert("_agent_id".into(), json!(aid));
    }
    if let Some(eid) = external_id {
        obj.insert("_external_id".into(), json!(eid));
    }
    obj.insert("_content_hash".into(), json!(content_hash));
}

/// 16-byte hex content hash over the normalized inputs. Uses the stdlib
/// `DefaultHasher` to avoid pulling in a crypto dep — the hash is for
/// idempotency, not integrity, so collision resistance only needs to
/// beat 5-minute-window birthday odds.
fn content_hash(parts: &[&str]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
        // Separator prevents ("ab", "cd") colliding with ("a", "bcd").
        0u8.hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn remember_decision_inserts_to_decisions_table() {
        let (db, _dir) = temp_db();
        let result = db
            .remember_decision(DecisionInput {
                summary: "use JWT over sessions",
                reason: Some("stateless auth scales better"),
                files: Some(&["src/auth.rs"]),
                agent_id: Some("claude"),
                external_id: Some("dec-001"),
                force_new: false,
                source: WriteSource::Core,
            })
            .unwrap();

        assert!(result.is_new);
        let stored = db.get(&result.id).unwrap().unwrap();
        assert_eq!(stored.table, "decisions");
        assert_eq!(stored.data["summary"], "use JWT over sessions");
        assert_eq!(stored.data["_agent_id"], "claude");
        assert_eq!(stored.data["_source"], "axil-core");
    }

    #[test]
    fn remember_decision_dedupes_on_agent_external_id() {
        let (db, _dir) = temp_db();
        let input = DecisionInput {
            summary: "try Redis for hot path",
            reason: None,
            files: None,
            agent_id: Some("a1"),
            external_id: Some("k1"),
            force_new: false,
            source: WriteSource::Core,
        };
        let first = db.remember_decision(input.clone()).unwrap();
        let second = db.remember_decision(input).unwrap();
        assert!(first.is_new);
        assert!(!second.is_new);
        assert_eq!(first.id, second.id);
    }

    #[test]
    fn remember_decision_dedupes_on_content_hash_in_window() {
        let (db, _dir) = temp_db();
        let mk = || DecisionInput {
            summary: "adopt Axum for HTTP",
            reason: None,
            files: None,
            agent_id: None,
            external_id: None,
            force_new: false,
            source: WriteSource::Core,
        };
        let first = db.remember_decision(mk()).unwrap();
        let second = db.remember_decision(mk()).unwrap();
        assert_eq!(first.id, second.id);
        assert!(!second.is_new);
    }

    #[test]
    fn force_new_bypasses_dedup() {
        let (db, _dir) = temp_db();
        let base = DecisionInput {
            summary: "same wording",
            reason: None,
            files: None,
            agent_id: Some("a"),
            external_id: Some("k"),
            force_new: false,
            source: WriteSource::Core,
        };
        let first = db.remember_decision(base.clone()).unwrap();
        let mut forced = base;
        forced.force_new = true;
        let second = db.remember_decision(forced).unwrap();
        assert_ne!(first.id, second.id);
        assert!(first.is_new);
        assert!(second.is_new);
    }

    #[test]
    fn remember_error_inserts_to_errors_table() {
        let (db, _dir) = temp_db();
        let result = db
            .remember_error(ErrorInput {
                error: "connection refused",
                root_cause: Some("postgres container not running"),
                fix: Some("ran docker compose up"),
                files: None,
                agent_id: None,
                external_id: None,
                force_new: false,
                source: WriteSource::Cli,
            })
            .unwrap();
        let stored = db.get(&result.id).unwrap().unwrap();
        assert_eq!(stored.table, "errors");
        assert_eq!(stored.data["error"], "connection refused");
        assert_eq!(stored.data["fix"], "ran docker compose up");
    }

    #[test]
    fn set_preference_overwrites_and_keeps_previous_value() {
        let (db, _dir) = temp_db();
        let first = db.set_preference("theme", json!("dark")).unwrap();
        assert!(first.is_new);
        let second = db.set_preference("theme", json!("light")).unwrap();
        assert!(second.is_new);
        // Only one preference for `theme` exists after overwrite.
        let remaining: Vec<_> = db
            .list("preferences")
            .unwrap()
            .into_iter()
            .filter(|r| r.data.get("key").and_then(|v| v.as_str()) == Some("theme"))
            .collect();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].data["value"], "light");
        assert_eq!(remaining[0].data["_previous_value"], "dark");
    }

    #[test]
    fn close_session_is_idempotent_by_id() {
        let (db, _dir) = temp_db();
        let a = db.close_session("run-42", Some("done")).unwrap();
        let b = db.close_session("run-42", Some("done")).unwrap();
        assert!(a.is_new);
        assert!(!b.is_new);
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn content_hash_separator_prevents_collision() {
        // ("ab","cd") and ("a","bcd") must hash differently.
        let h1 = content_hash(&["ab", "cd"]);
        let h2 = content_hash(&["a", "bcd"]);
        assert_ne!(h1, h2);
    }
}
