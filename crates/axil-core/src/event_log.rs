//! Durable, opt-in semantic event log (`event-log` feature).
//!
//! The per-record [`audit_log`](crate::Axil::audit_log) is a lossy ring buffer
//! that skips every `_`-prefixed table and keys entries by a coarse RFC3339
//! timestamp — same-millisecond writes are indistinguishable, so it cannot back
//! a deterministic "what changed since cursor C" feed.
//!
//! This module upgrades that idea into a pull-based event tape:
//!
//! - **Curated allowlist.** Only a small set of agent-meaningful events is
//!   captured (see [`classify`]): a belief revised, a decision superseded, an
//!   error fixed, a checkpoint written. The `_`-prefixed skip is deliberately
//!   bypassed for those allowlisted internal tables (`_beliefs`,
//!   `_checkpoint_records`), and nothing else.
//! - **Monotonic ULID cursor.** Entries are keyed by a monotonic ULID
//!   ([`EventCursor`]), so two writes in the same millisecond still sort in
//!   commit order and a consumer can resume from any cursor without loss.
//! - **`agent_id` tag.** Each event records the writing agent (from the
//!   record's `_agent_id` field) so a reader can exclude its own writes.
//!
//! ## Isolation contract
//!
//! The event log surfaces **committed facts only** — it is a notification that
//! a record changed, never a back-door to read another agent's private session
//! state. It captures the same `(op, table, record_id, agent_id, kind)` tuple
//! that is already queryable via [`Axil::get`](crate::Axil::get) under normal
//! access rules; it does not embed record bodies and it does not relax
//! cross-agent session isolation. `recall_delta`'s `exclude_agent` is an
//! ergonomic filter (skip my own writes), not a security boundary.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;

use crate::record::Record;

/// One captured semantic event on the durable `_event_log` tape.
///
/// Stored as `cursor → SemanticEvent` (JSON). The cursor lives in the redb key,
/// not the body, so a range scan is the cheap path; it is mirrored here for
/// callers that read events without their keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticEvent {
    /// Monotonic ULID cursor — also the redb key. Pass the last one seen back to
    /// `recall_delta` / `events_since` to resume.
    pub cursor: String,
    /// Curated event kind: `belief-revised`, `decision-superseded`,
    /// `error-fixed`, or `checkpoint-written`.
    pub kind: String,
    /// Underlying mutation: `"insert"`, `"update"`, or `"delete"`.
    pub op: String,
    /// Table the affected record belongs to (may be `_`-prefixed for
    /// allowlisted internal events such as belief revisions).
    pub table: String,
    /// Affected record ID.
    pub record_id: String,
    /// Writing agent (`_agent_id` on the record), if tagged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}

/// Curated allowlist of semantic event kinds. A `(table, op, record)` tuple that
/// matches none of these is **not** captured — keeping the tape a high-signal
/// feed, not a full mirror of the write stream.
pub mod kind {
    /// A belief's confidence/validity changed (e.g. auto-doubt, revision).
    pub const BELIEF_REVISED: &str = "belief-revised";
    /// A decision (or other supersede-able record) was marked superseded.
    pub const DECISION_SUPERSEDED: &str = "decision-superseded";
    /// An error record gained a fix.
    pub const ERROR_FIXED: &str = "error-fixed";
    /// A session checkpoint was written.
    pub const CHECKPOINT_WRITTEN: &str = "checkpoint-written";
}

/// Table that holds session checkpoint records (owned by `axil-checkpoint`).
const CHECKPOINT_TABLE: &str = "_checkpoint_records";
/// Table that holds belief records.
const BELIEFS_TABLE: &str = "_beliefs";
/// Convention table for error memories.
const ERRORS_TABLE: &str = "errors";

/// Classify a committed write against the curated allowlist.
///
/// `op` is `"insert"`, `"update"`, or `"delete"`. Returns the event `kind` to
/// record, or `None` when the write is not an allowlisted semantic event.
///
/// The allowlist deliberately includes two `_`-prefixed internal tables
/// (`_beliefs`, `_checkpoint_records`) that the plain audit log skips, because a
/// belief revision and a checkpoint write are exactly the cross-agent signals
/// this feed exists to surface.
pub fn classify(op: &str, record: &Record) -> Option<&'static str> {
    let table = record.table.as_str();
    let data = &record.data;

    // Checkpoint written: any new checkpoint record.
    if table == CHECKPOINT_TABLE && op == "insert" {
        return Some(kind::CHECKPOINT_WRITTEN);
    }

    // Belief revised: a belief was doubted or had its confidence revised.
    if table == BELIEFS_TABLE && (op == "update" || op == "insert") {
        if bool_field(data, "doubted") || data.get("_doubt_reason").is_some() {
            return Some(kind::BELIEF_REVISED);
        }
    }

    // Decision (or any user-facing record) marked superseded.
    if bool_field(data, "_superseded") {
        return Some(kind::DECISION_SUPERSEDED);
    }

    // Error fixed: an error record carrying a non-empty `fix`.
    if table == ERRORS_TABLE && non_empty_str(data, "fix") {
        return Some(kind::ERROR_FIXED);
    }

    None
}

fn bool_field(data: &Value, key: &str) -> bool {
    data.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn non_empty_str(data: &Value, key: &str) -> bool {
    data.get(key)
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
}

/// Read the writing agent's id off a record (`_agent_id`), if tagged.
pub fn agent_id_of(record: &Record) -> Option<String> {
    record
        .data
        .get("_agent_id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Monotonic ULID cursor source for the event tape.
///
/// Wraps [`ulid::Generator`], which guarantees each generated ULID is strictly
/// greater than the previous one **even within the same millisecond** by
/// incrementing the random component. That is the property the plain audit
/// log's RFC3339 timestamp lacks, and it is what makes the cursor a reliable
/// resume point.
///
/// Held behind a `Mutex` because the generator carries mutable monotonic state;
/// the lock is only taken on an allowlisted event (a rare, high-signal write),
/// never on the per-record hot path.
pub struct EventCursor {
    generator: Mutex<ulid::Generator>,
}

impl EventCursor {
    /// Create a fresh monotonic cursor source.
    pub fn new() -> Self {
        Self {
            generator: Mutex::new(ulid::Generator::new()),
        }
    }

    /// Produce the next monotonic cursor string.
    ///
    /// On the (astronomically rare) per-millisecond random-overflow error, falls
    /// back to a fresh non-monotonic ULID — a single out-of-fine-order cursor is
    /// preferable to dropping the event, and the next call re-establishes order.
    pub fn next(&self) -> String {
        let mut gen = match self.generator.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match gen.generate() {
            Ok(ulid) => ulid.to_string(),
            Err(_) => ulid::Ulid::new().to_string(),
        }
    }
}

impl Default for EventCursor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cursor_is_monotonic_within_same_millisecond() {
        let cursor = EventCursor::new();
        // A tight loop forces same-millisecond generation; every cursor must
        // still be strictly greater than the previous one.
        let mut prev = cursor.next();
        for _ in 0..1000 {
            let next = cursor.next();
            assert!(
                next > prev,
                "cursor not monotonic: {next} should be > {prev}"
            );
            prev = next;
        }
    }

    #[test]
    fn classify_checkpoint_insert() {
        let rec = Record::new(CHECKPOINT_TABLE, json!({"goal": "ship T14"}));
        assert_eq!(classify("insert", &rec), Some(kind::CHECKPOINT_WRITTEN));
        // An update to a checkpoint is not a "written" event.
        assert_eq!(classify("update", &rec), None);
    }

    #[test]
    fn classify_belief_revision() {
        let doubted = Record::new(BELIEFS_TABLE, json!({"doubted": true}));
        assert_eq!(classify("update", &doubted), Some(kind::BELIEF_REVISED));
        let fresh = Record::new(BELIEFS_TABLE, json!({"statement": "x is y"}));
        assert_eq!(classify("insert", &fresh), None);
    }

    #[test]
    fn classify_decision_superseded() {
        let rec = Record::new("decisions", json!({"_superseded": true, "summary": "old"}));
        assert_eq!(classify("update", &rec), Some(kind::DECISION_SUPERSEDED));
    }

    #[test]
    fn classify_error_fixed() {
        let fixed = Record::new(ERRORS_TABLE, json!({"error": "boom", "fix": "patch"}));
        assert_eq!(classify("insert", &fixed), Some(kind::ERROR_FIXED));
        let open = Record::new(ERRORS_TABLE, json!({"error": "boom", "fix": ""}));
        assert_eq!(classify("insert", &open), None);
    }

    #[test]
    fn classify_ignores_ordinary_writes() {
        let rec = Record::new("notes", json!({"text": "hello"}));
        assert_eq!(classify("insert", &rec), None);
    }

    #[test]
    fn agent_id_extracted_from_record() {
        let rec = Record::new("decisions", json!({"summary": "x", "_agent_id": "agent-7"}));
        assert_eq!(agent_id_of(&rec), Some("agent-7".to_string()));
        let untagged = Record::new("decisions", json!({"summary": "x"}));
        assert_eq!(agent_id_of(&untagged), None);
    }
}
