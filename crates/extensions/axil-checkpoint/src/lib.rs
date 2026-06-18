//! `axil-checkpoint` — structured session-checkpoint memory.
//!
//! A *checkpoint* is the durable answer to "what would the next agent need
//! to pick this up?" — stored once, replayed automatically by `axil
//! boot` so a fresh session resumes instead of re-discovering. The
//! design mirrors the conversational checkpoint pattern popularised by
//! Matt Pocock's `/handoff` skill (compact, reference don't duplicate,
//! redact, tailor to focus), but the *delivery* is Axil-native: the
//! checkpoint lives in the database as a first-class record, not in a
//! one-shot temp file, and `boot` resolves any record references it
//! contains to their *current* state rather than a stale snapshot.
//!
//! ## Storage
//!
//! Checkpoints live in the `_checkpoint_records` table (prefix `_checkpoint_`)
//! and link to their owning session via the `session_checkpoint_for`
//! graph edge when a graph engine is registered. Fields:
//!
//! - `goal`           — the user's north-star intent (most-often-lost field)
//! - `state`          — 1–2 sentence "where things stand"
//! - `next_steps[]`   — ordered, the single most-valuable resume field
//! - `open_questions[]` — blockers / uncertainty
//! - `references[]`   — typed pointers (commit/PR/file/plan/record), NOT copies
//! - `summary`        — optional one-line mirror, embedded for semantic recall
//! - `session_id`     — owning session (string form)
//! - `kind`           — "snapshot" (mid-session) or "final" (at session end)
//!
//! ## Path C dispatch (Phase 17)
//!
//! [`CheckpointExtension`] implements every dispatch method on
//! [`axil_core::Extension`]: a CLI surface `axil checkpoint …`, an MCP
//! tool `checkpoint`, and a `boot_block` that renders "Resume Here" at
//! the top of `axil boot`. When no explicit checkpoint has been written,
//! the boot block falls back to a derived view assembled from the
//! latest session's `summary` / `decisions_made` / `files_touched`
//! (the internal adapter in [`derive_checkpoint_from_session`]).

mod extension;

pub use extension::CheckpointExtension;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, AxilError, Op, Record, RecordId, Result, SortDirection};

/// Table that holds explicit checkpoint records.
///
/// Prefix `_checkpoint_` follows the Phase-17 convention for new
/// Extensions (see `docs/src/extending/extensions.md`). The single
/// owned table is plural-suffixed (`_records`) so future tables
/// (`_checkpoint_resolved_refs`, etc.) slot in without collision.
pub const TABLE_CHECKPOINTS: &str = "_checkpoint_records";

/// Graph edge from a checkpoint record → its owning session record.
pub const EDGE_CHECKPOINT_FOR: &str = "session_checkpoint_for";

/// Reference kind for an in-DB record id. Boot renders this kind by
/// resolving the id to its *current* row, so checkpoints never carry
/// stale snapshots.
pub const REF_KIND_RECORD: &str = "record";
/// Reference kind for a workspace-relative file path.
pub const REF_KIND_FILE: &str = "file";

/// Where in the source `_sessions` table to look for the most recent
/// session. The constant is hard-coded rather than pulled from
/// `axil-memory` so this crate stays a leaf — checkpoint doesn't need
/// memory's full lifecycle API, just to read/append the table.
const TABLE_SESSIONS: &str = "_sessions";
const SESSION_ACTIVE: &str = "active";

/// Errors specific to checkpoint write / derive paths.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint payload must be a JSON object")]
    NotAnObject,
    #[error("checkpoint payload is empty — provide at least one of goal/state/next_steps/open_questions/summary")]
    Empty,
    #[error("axil error: {0}")]
    Axil(#[from] AxilError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// What kind of checkpoint this row represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// Mid-session snapshot — the session keeps running afterwards.
    Snapshot,
    /// Final checkpoint written at session end — the session is closed.
    Final,
}

impl CheckpointKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Snapshot => "snapshot",
            Self::Final => "final",
        }
    }
}

/// A typed pointer carried inside a checkpoint's `references[]`. Pointers
/// are resolved live at boot time when their `kind` is `record`, so a
/// superseded decision shows its *current* state rather than a stale
/// snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    /// What this pointer points at. Free-form so future ecosystems
    /// (e.g. `linear-issue`, `notion-page`) slot in without a schema
    /// bump. The boot resolver only special-cases `record`.
    pub kind: String,
    /// The reference value itself — a record id, commit sha, PR url,
    /// file path, plan-doc path, etc.
    #[serde(rename = "ref")]
    pub reference: String,
    /// Optional human note explaining why this reference matters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// In-memory representation of a checkpoint payload.
///
/// All fields are optional so partial checkpoints (e.g. "just next_steps")
/// are first-class. Construction goes through [`Checkpoint::from_value`]
/// to validate that *something* meaningful is present.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_steps: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_questions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<Reference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl Checkpoint {
    /// Parse + validate a checkpoint from raw JSON.
    pub fn from_value(value: Value) -> std::result::Result<Self, CheckpointError> {
        if !value.is_object() {
            return Err(CheckpointError::NotAnObject);
        }
        let checkpoint: Checkpoint = serde_json::from_value(value)?;
        if checkpoint.is_empty() {
            return Err(CheckpointError::Empty);
        }
        Ok(checkpoint)
    }

    /// `true` if no field carries content the boot replay would render.
    pub fn is_empty(&self) -> bool {
        self.goal.is_none()
            && self.state.is_none()
            && self.next_steps.is_empty()
            && self.open_questions.is_empty()
            && self.references.is_empty()
            && self.summary.is_none()
    }

    /// One-line text used for semantic embedding (so checkpoints are
    /// recallable). Joins goal + next_steps + open_questions — the
    /// fields most useful as a search target.
    pub fn embedding_text(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(g) = &self.goal {
            parts.push(g.clone());
        }
        if !self.next_steps.is_empty() {
            parts.push(format!("next: {}", self.next_steps.join("; ")));
        }
        if !self.open_questions.is_empty() {
            parts.push(format!("open: {}", self.open_questions.join("; ")));
        }
        if parts.is_empty() {
            self.summary.clone().unwrap_or_default()
        } else {
            parts.join(" — ")
        }
    }
}

/// Write a checkpoint for the most-recent active session (creating one
/// when none exists), stamped with `kind`. The owning session is
/// *not* ended — `CheckpointKind::Final` only signals intent. Callers
/// that want to end the session should do so separately via the
/// existing session-lifecycle API.
pub fn write_with_active_session(
    db: &Axil,
    checkpoint: &Checkpoint,
    kind: CheckpointKind,
) -> std::result::Result<Record, CheckpointError> {
    let session = ensure_active_session(db)?;
    write_checkpoint(db, &session.id, checkpoint, kind)
}

/// Back-compat alias — write a snapshot against the active session.
/// New callers should prefer [`write_with_active_session`] with an
/// explicit kind.
pub fn snapshot(db: &Axil, checkpoint: &Checkpoint) -> std::result::Result<Record, CheckpointError> {
    write_with_active_session(db, checkpoint, CheckpointKind::Snapshot)
}

/// Write a checkpoint and attach it to a specific session id, regardless
/// of whether that session is active or ended. Used by `--session`
/// overrides.
pub fn write_for_session(
    db: &Axil,
    session_id: &RecordId,
    checkpoint: &Checkpoint,
    kind: CheckpointKind,
) -> std::result::Result<Record, CheckpointError> {
    write_checkpoint(db, session_id, checkpoint, kind)
}

// TODO(retention): `_checkpoint_records` grows unbounded — one row per
// snapshot, no cleanup. Boot stays O(1) (latest_checkpoint uses an
// indexed query) but disk grows ~tens of rows/day/repo. Wire into
// axil-memory's TTL/decay machinery when that lands as a cross-
// Extension hook; until then, prune manually via `axil store delete`.
fn write_checkpoint(
    db: &Axil,
    session_id: &RecordId,
    checkpoint: &Checkpoint,
    kind: CheckpointKind,
) -> std::result::Result<Record, CheckpointError> {
    let mut data = serde_json::to_value(checkpoint)?;
    // Pre-stamp summary so the embed step is a single op rather than
    // read-modify-write-embed. We only synthesize when the user
    // didn't supply one and there's an embedder to consume it.
    let has_vector = db.has_vector_index();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("session_id".into(), json!(session_id.to_string()));
        obj.insert("kind".into(), json!(kind.as_str()));
        obj.insert("written_at".into(), json!(Utc::now().to_rfc3339()));
        if has_vector
            && obj
                .get("summary")
                .and_then(|v| v.as_str())
                .map_or(true, str::is_empty)
        {
            let embed_text = checkpoint.embedding_text();
            if !embed_text.is_empty() {
                obj.insert("summary".into(), json!(embed_text));
            }
        }
    }
    let record = db.insert(TABLE_CHECKPOINTS, data)?;

    // Link to session via graph when available, so neighbour queries
    // can walk session ↔ checkpoint in both directions.
    if db.has_graph_index() {
        let _ = db.relate(&record.id, EDGE_CHECKPOINT_FOR, session_id, None);
    }

    if has_vector {
        // Embed failure is non-fatal — recall just won't find this
        // row semantically. Caller still gets the inserted record.
        let _ = db.embed_field(&record.id, "summary");
    }

    Ok(record)
}

/// Find the most-recent active session, or start a new minimal one if
/// none exists. The created session matches the shape `axil-memory`
/// writes (status="active", started_at, record_count=0) so the
/// existing session tooling continues to see it as a normal session.
fn ensure_active_session(db: &Axil) -> std::result::Result<Record, CheckpointError> {
    let active = db
        .query()
        .table(TABLE_SESSIONS)
        .where_field("status", Op::Eq, json!(SESSION_ACTIVE))
        .exec()?;
    if let Some(latest) = active.into_iter().max_by_key(|r| r.created_at) {
        return Ok(latest);
    }
    let data = json!({
        "status": SESSION_ACTIVE,
        "started_at": Utc::now().to_rfc3339(),
        "record_count": 0,
        "turns": [],
        "_created_by": "axil-checkpoint",
    });
    Ok(db.insert(TABLE_SESSIONS, data)?)
}

/// Fetch the most recent checkpoint record (any kind), or `None` if the
/// table is empty. Used by both the boot block and the MCP read tool.
///
/// Pushes the sort + limit into the query engine so this is O(1)
/// rather than a full-table scan — boot is the hot path.
pub fn latest_checkpoint(db: &Axil) -> Result<Option<Record>> {
    // list + max_by_key reads every row, but `_checkpoint_records` grows
    // slowly (tens/day per repo) and this runs once per `axil boot`.
    // The query builder's `order_by_time(Desc).limit(1)` *should* be
    // O(log n), but it interacts badly with the table scan and can
    // return a stale row — see TODO(query-order_by_time-limit).
    // Correctness beats premature optimization here.
    Ok(db
        .list(TABLE_CHECKPOINTS)?
        .into_iter()
        .max_by_key(|r| r.created_at))
}

/// Internal adapter — derive a best-effort checkpoint from the latest
/// session record when no explicit checkpoint has been stored. Surfaces
/// the existing `summary`, `decisions_made`, and `files_touched`
/// fields that `axil-memory::end_session` already writes, plus a
/// `next_steps` list mined from unresolved error / context rows.
///
/// Returns `None` when there's nothing meaningful to show (empty DB,
/// or a brand-new session with no fields populated).
pub fn derive_checkpoint_from_session(db: &Axil) -> Option<Checkpoint> {
    let session = db
        .list(TABLE_SESSIONS)
        .ok()?
        .into_iter()
        .max_by_key(|r| r.created_at)?;

    let data = &session.data;
    let summary = data
        .get("summary")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    let state = data
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| format!("last session {s}"));

    let files = string_array(data, "files_touched");

    // next_steps stays empty in the derived path — prior `decisions_made`
    // are *completed* actions, not pending ones. Surfacing them as
    // next_steps would tell the next agent to redo finished work.
    // If the agent wants prior decisions, `axil recall` is the right tool.
    let next_steps: Vec<String> = Vec::new();
    let open_questions = derive_open_questions(db);

    let references: Vec<Reference> = files
        .into_iter()
        .map(|f| Reference {
            kind: REF_KIND_FILE.into(),
            reference: f,
            note: None,
        })
        .collect();

    let checkpoint = Checkpoint {
        goal: None,
        state,
        next_steps,
        open_questions,
        references,
        summary,
    };
    if checkpoint.is_empty() {
        None
    } else {
        Some(checkpoint)
    }
}

/// Pull recent unresolved errors as open questions, capped at 5.
///
/// Pushes the "resolved != true" filter + recency order + limit into
/// the query engine so the boot fallback path doesn't deserialize
/// every error row on a long-lived project.
fn derive_open_questions(db: &Axil) -> Vec<String> {
    // `order_by_time` sorts on intrinsic `Record.created_at` — payload
    // `created_at` may not exist on error rows.
    db.query()
        .table("errors")
        .where_field("resolved", Op::Ne, json!(true))
        .order_by_time(SortDirection::Desc)
        .limit(5)
        .exec()
        .unwrap_or_default()
        .iter()
        .filter_map(|r| r.data.get("error").and_then(|v| v.as_str()).map(String::from))
        .collect()
}

fn string_array(data: &Value, field: &str) -> Vec<String> {
    data.get(field)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Render a checkpoint (explicit or derived) into a compact "Resume Here"
/// block suitable for `axil boot` narrative output. Returns `None`
/// when the checkpoint has nothing renderable.
///
/// The renderer resolves `references[]` of `kind: "record"` to their
/// *current* row by id, surfacing the live summary instead of a stale
/// copy. Other reference kinds render their raw `ref` string.
pub fn render_resume_block(db: &Axil, checkpoint: &Checkpoint) -> Option<String> {
    if checkpoint.is_empty() {
        return None;
    }
    let mut out = String::from("## Resume Here\n");
    if let Some(s) = &checkpoint.summary {
        // Headline: short prose that the agent reads first. We keep
        // goal/state below as structured fields so the agent can
        // discriminate intent vs. status vs. summary.
        out.push_str(&format!("- {s}\n"));
    }
    if let Some(g) = &checkpoint.goal {
        out.push_str(&format!("- **Goal:** {g}\n"));
    }
    if let Some(s) = &checkpoint.state {
        out.push_str(&format!("- **State:** {s}\n"));
    }
    if !checkpoint.next_steps.is_empty() {
        out.push_str("- **Next:**\n");
        for s in &checkpoint.next_steps {
            out.push_str(&format!("  - {s}\n"));
        }
    }
    if !checkpoint.open_questions.is_empty() {
        out.push_str("- **Open:**\n");
        for q in &checkpoint.open_questions {
            out.push_str(&format!("  - {q}\n"));
        }
    }
    if !checkpoint.references.is_empty() {
        out.push_str("- **Refs:**\n");
        for r in &checkpoint.references {
            let resolved = if r.kind == REF_KIND_RECORD {
                resolve_record_reference(db, &r.reference)
            } else {
                None
            };
            let label = match resolved {
                Some(s) => format!("{} → {}", r.reference, s),
                None => r.reference.clone(),
            };
            match &r.note {
                Some(n) => out.push_str(&format!("  - [{}] {label} — {n}\n", r.kind)),
                None => out.push_str(&format!("  - [{}] {label}\n", r.kind)),
            }
        }
    }
    Some(out)
}

/// Look up a record-id reference and return a one-line live summary
/// from its current state. Returns `None` if the id is malformed, the
/// record is gone, or the record has no recognizable display field.
///
/// Delegates to [`axil_core::util::value_text_legacy`] for the field
/// walk so checkpoint stays consistent with the rest of the project's
/// "what's the display text for this record?" convention.
fn resolve_record_reference(db: &Axil, id: &str) -> Option<String> {
    let rid = RecordId::from_string(id).ok()?;
    let row = db.get(&rid).ok().flatten()?;
    let text = axil_core::util::value_text_legacy(&row.data);
    if text.is_empty() {
        // Final fall-through: some record kinds (e.g. plans, prefs) use
        // `title` and aren't covered by `value_text_legacy`.
        row.data
            .get("title")
            .and_then(|v| v.as_str())
            .map(String::from)
    } else {
        Some(text)
    }
}

/// Convenience: read the latest stored checkpoint into a [`Checkpoint`], or
/// fall back to a derived one. Used by [`CheckpointExtension::boot_block`]
/// and any MCP/CLI read paths.
///
/// **Freshness rule:** a stored checkpoint is stale once a session newer
/// than the checkpoint itself exists — the next agent started fresh work
/// without writing a checkpoint for it, so replaying the prior session's
/// "Resume Here" would point at unrelated context. We compare
/// `created_at` rather than session ids so two rows landing in the
/// same second can't make freshness order-dependent.
pub fn current_checkpoint(db: &Axil) -> Option<(Checkpoint, Source)> {
    let latest_session = latest_session_record(db);
    if let Ok(Some(row)) = latest_checkpoint(db) {
        let stale = latest_session
            .as_ref()
            .map(|s| s.created_at > row.created_at)
            .unwrap_or(false);
        if !stale {
            if let Ok(h) = serde_json::from_value::<Checkpoint>(row.data.clone()) {
                if !h.is_empty() {
                    return Some((h, Source::Stored));
                }
            }
        }
    }
    derive_checkpoint_from_session(db).map(|h| (h, Source::Derived))
}

/// Fetch the most-recent session row, for staleness comparisons.
fn latest_session_record(db: &Axil) -> Option<Record> {
    db.list(TABLE_SESSIONS)
        .ok()?
        .into_iter()
        .max_by_key(|r| r.created_at)
}

/// How `current_checkpoint` produced the returned [`Checkpoint`] — explicit
/// vs derived. Surfaced in the boot block so the agent knows whether
/// the prior session actually wrote a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Stored,
    Derived,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Derived => "derived",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn checkpoint_rejects_non_object() {
        let err = Checkpoint::from_value(json!([1, 2, 3])).unwrap_err();
        assert!(matches!(err, CheckpointError::NotAnObject));
    }

    #[test]
    fn checkpoint_rejects_empty_object() {
        let err = Checkpoint::from_value(json!({})).unwrap_err();
        assert!(matches!(err, CheckpointError::Empty));
    }

    #[test]
    fn checkpoint_parses_minimal_next_steps_only() {
        let h = Checkpoint::from_value(json!({
            "next_steps": ["land axil-checkpoint scaffold"]
        }))
        .unwrap();
        assert_eq!(h.next_steps.len(), 1);
        assert!(h.goal.is_none());
        assert!(!h.is_empty());
    }

    #[test]
    fn embedding_text_prefers_goal_and_next_steps() {
        let h = Checkpoint {
            goal: Some("ship axil-checkpoint".into()),
            next_steps: vec!["scaffold crate".into(), "wire boot block".into()],
            ..Default::default()
        };
        let t = h.embedding_text();
        assert!(t.contains("ship axil-checkpoint"));
        assert!(t.contains("scaffold crate"));
    }

    #[test]
    fn snapshot_creates_session_when_none_active() {
        let (db, _dir) = temp_db();
        let h = Checkpoint::from_value(json!({"goal": "test"})).unwrap();
        let rec = snapshot(&db, &h).unwrap();
        // Checkpoint row got stamped with session_id, kind, written_at.
        assert!(rec.data.get("session_id").is_some());
        assert_eq!(rec.data["kind"], "snapshot");
        // A matching active session exists.
        let sessions = db.list(TABLE_SESSIONS).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].data["status"], "active");
    }

    #[test]
    fn snapshot_reuses_latest_active_session() {
        let (db, _dir) = temp_db();
        // Seed an active session and a stale ended one.
        let active = db
            .insert(
                TABLE_SESSIONS,
                json!({"status": "active", "started_at": "2026-05-25T00:00:00Z"}),
            )
            .unwrap();
        db.insert(
            TABLE_SESSIONS,
            json!({"status": "ended", "started_at": "2026-05-24T00:00:00Z"}),
        )
        .unwrap();

        let h = Checkpoint::from_value(json!({"state": "midway"})).unwrap();
        let rec = snapshot(&db, &h).unwrap();
        assert_eq!(
            rec.data["session_id"].as_str().unwrap(),
            active.id.to_string()
        );
        // No new session was created.
        assert_eq!(db.list(TABLE_SESSIONS).unwrap().len(), 2);
    }

    #[test]
    fn derive_returns_none_on_empty_db() {
        let (db, _dir) = temp_db();
        assert!(derive_checkpoint_from_session(&db).is_none());
    }

    #[test]
    fn derive_pulls_summary_and_files_from_latest_session() {
        let (db, _dir) = temp_db();
        db.insert(
            TABLE_SESSIONS,
            json!({
                "status": "ended",
                "summary": "wired checkpoint scaffold",
                "decisions_made": ["chose tier-2 extension"],
                "files_touched": ["crates/extensions/axil-checkpoint/src/lib.rs"]
            }),
        )
        .unwrap();
        let h = derive_checkpoint_from_session(&db).expect("derive should produce a checkpoint");
        assert_eq!(h.summary.as_deref(), Some("wired checkpoint scaffold"));
        // Prior decisions are completed actions — they must NOT leak into
        // next_steps; surfacing them there would tell the next agent to
        // redo finished work.
        assert!(h.next_steps.is_empty());
        assert_eq!(h.references.len(), 1);
        assert_eq!(h.references[0].kind, REF_KIND_FILE);
    }

    #[test]
    fn render_resume_block_includes_each_field() {
        let (db, _dir) = temp_db();
        let h = Checkpoint {
            goal: Some("ship axil-checkpoint".into()),
            state: Some("scaffolded; tests next".into()),
            next_steps: vec!["wire boot_block dispatch".into()],
            open_questions: vec!["which CLI flag name?".into()],
            references: vec![Reference {
                kind: "file".into(),
                reference: "crates/extensions/axil-checkpoint/src/lib.rs".into(),
                note: Some("crate root".into()),
            }],
            summary: None,
        };
        let s = render_resume_block(&db, &h).unwrap();
        assert!(s.starts_with("## Resume Here"));
        assert!(s.contains("ship axil-checkpoint"));
        assert!(s.contains("wire boot_block dispatch"));
        assert!(s.contains("which CLI flag name?"));
        assert!(s.contains("crate root"));
    }

    #[test]
    fn render_resume_block_returns_none_for_empty() {
        let (db, _dir) = temp_db();
        assert!(render_resume_block(&db, &Checkpoint::default()).is_none());
    }

    #[test]
    fn current_checkpoint_prefers_stored_over_derived() {
        let (db, _dir) = temp_db();
        // Seed a session that would derive into something non-empty.
        db.insert(
            TABLE_SESSIONS,
            json!({"status": "ended", "summary": "derived summary"}),
        )
        .unwrap();
        // Write an explicit checkpoint. Stored should win.
        let h = Checkpoint::from_value(json!({"goal": "explicit goal"})).unwrap();
        snapshot(&db, &h).unwrap();
        let (got, src) = current_checkpoint(&db).expect("must have current checkpoint");
        assert_eq!(src, Source::Stored);
        assert_eq!(got.goal.as_deref(), Some("explicit goal"));
    }

    /// Regression for the cross-session staleness bug Codex caught
    /// (/octo:review on the Phase 18 commit): writing a checkpoint in
    /// session A, then later starting session B without writing a
    /// new checkpoint, must NOT replay A's checkpoint at boot — B is the
    /// current scope and its derive-from-state should win.
    #[test]
    fn stored_checkpoint_for_old_session_is_stale_when_newer_session_exists() {
        use std::thread::sleep;
        use std::time::Duration;

        let (db, _dir) = temp_db();
        // Session A + an explicit checkpoint against it.
        let h = Checkpoint::from_value(json!({"goal": "session A goal"})).unwrap();
        snapshot(&db, &h).unwrap();
        // Force a clock tick so the next session's created_at is
        // strictly greater than the checkpoint's.
        sleep(Duration::from_millis(1100));
        // Session B starts after the checkpoint — agent hasn't written one
        // yet for B.
        db.insert(
            TABLE_SESSIONS,
            json!({"status": "ended", "summary": "session B summary"}),
        )
        .unwrap();

        let (got, src) = current_checkpoint(&db).expect("must have current checkpoint");
        assert_eq!(
            src,
            Source::Derived,
            "session B is newer than session A's checkpoint — boot must derive from B, \
             not replay A's stale checkpoint",
        );
        assert_eq!(
            got.summary.as_deref(),
            Some("session B summary"),
            "derived checkpoint must reflect the newer session's summary, \
             not the stale checkpoint's text",
        );
    }
}
