//! Atlas sync protocol — wire DTOs.
//!
//! This crate is the **shared contract** for the Atlas sync protocol. The
//! public [`axil-sync`] client and the (private, commercial) Atlas server both
//! depend on it so they agree on the request/response shapes without either
//! side depending on the other's code. The protocol is fine to be public — it
//! is just an API; the *intelligence* behind it (cross-project compounding,
//! the registry) lives in the closed server.
//!
//! The types are **storage-agnostic**: identical whether the server is backed
//! by a per-tenant `.axil` (Stage 1) or a Postgres translator (Stage 2).
//!
//! [`axil-sync`]: https://github.com/FC4b/axildb

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Opaque tenant identifier. On the wire it is always **derived from the PAT**
/// by the server's auth middleware, never supplied in a request body.
pub type TenantId = String;

/// Which sync tier an op belongs to. `Distillate` is the enforced default;
/// `Raw` is the opt-in, end-to-end-encrypted mirror tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Distillate,
    Raw,
}

fn default_tier() -> Tier {
    Tier::Distillate
}

/// The kind of mutation an op encodes. Wire names use dots (`record.upsert`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    #[serde(rename = "record.upsert")]
    RecordUpsert,
    #[serde(rename = "record.supersede")]
    RecordSupersede,
    #[serde(rename = "record.tombstone")]
    RecordTombstone,
    #[serde(rename = "edge.add")]
    EdgeAdd,
    #[serde(rename = "edge.expire")]
    EdgeExpire,
    #[serde(rename = "canonical.publish")]
    CanonicalPublish,
}

/// Last-writer-wins metadata. `updated_at` is the client clock; the server
/// clamps the *winner* to its own receipt time and uses this only for
/// deterministic convergence among already-received ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lww {
    pub updated_at: i64,
    #[serde(default)]
    pub origin: String,
}

/// One mutation in a push batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    /// Client-derived idempotency key. Stable while the record is unchanged
    /// (so re-syncs dedup) and different when its content changes (so the
    /// server never sees a reused id with a different `content_hash`).
    pub op_id: String,
    pub kind: OpKind,
    /// The op's *logical* source table (e.g. `decisions`). The server
    /// re-derives the tier from this and rejects episodic tables.
    pub table: String,
    pub member: String,
    #[serde(default = "default_tier")]
    pub tier: Tier,
    /// Client-computed content hash; idempotent dedup + tamper check.
    pub content_hash: String,
    #[serde(default)]
    pub lww: Option<Lww>,
    /// The record/edge body.
    #[serde(default)]
    pub payload: Value,
    /// Client-computed embedding (the server never embeds). Present on
    /// `record.upsert` ops for distillate records; omitted on pull.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// `POST /v1/sync/push` body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushBatch {
    #[serde(default = "one")]
    pub protocol: u32,
    pub member: String,
    #[serde(default)]
    pub ops: Vec<Op>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

fn one() -> u32 {
    1
}

/// `POST /v1/sync/push` response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PushResponse {
    pub accepted: usize,
    pub deduped: usize,
    pub assigned: Vec<Assigned>,
    pub rejected: Vec<Rejected>,
    /// Per-table head watermark after applying this batch.
    pub cursors: BTreeMap<String, u64>,
}

/// An op that was accepted (or deduped) and the seq it was assigned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assigned {
    pub op_id: String,
    pub seq: u64,
    pub table: String,
}

/// An op the server refused. `code` is machine-readable
/// (`tier_violation` | `id_conflict`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rejected {
    pub op_id: String,
    pub code: String,
    pub msg: String,
}

/// `GET /v1/sync/pull` query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullQuery {
    pub member: String,
    /// Optional logical-table filter.
    #[serde(default)]
    pub table: Option<String>,
    #[serde(default)]
    pub since: u64,
    #[serde(default = "limit_default")]
    pub limit: usize,
    #[serde(default = "default_tier")]
    pub tier: Tier,
}

fn limit_default() -> usize {
    200
}

/// An op carried back on pull, tagged with its server-assigned seq.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqOp {
    pub seq: u64,
    #[serde(flatten)]
    pub op: Op,
}

/// `GET /v1/sync/pull` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullResponse {
    pub from: u64,
    pub ops: Vec<SeqOp>,
    pub next_cursor: u64,
    pub has_more: bool,
}

/// A workspace member as seen at registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberRef {
    pub member_id: String,
    pub label: String,
}

/// `POST /v1/workspace/register` body (tenant derived from the token).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterWorkspace {
    pub workspace_id: String,
    pub name: String,
    #[serde(default)]
    pub members: Vec<MemberRef>,
    /// The per-project gate an owner flips to allow the raw-replication tier.
    #[serde(default)]
    pub raw_opt_in: bool,
}

/// Where a canonical id is known to live within a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locator {
    pub workspace_id: String,
    pub member_id: String,
    pub canonical_id: String,
}

/// `GET /v1/compound/{topic}` query — the client embeds the topic locally and
/// sends the vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundQuery {
    pub query_vector: Vec<f32>,
    #[serde(default = "topk_default")]
    pub top_k: usize,
}

fn topk_default() -> usize {
    10
}

/// One distillate hit from a compound query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundHit {
    pub canonical_id: Option<String>,
    pub kind: Option<String>,
    pub content: Value,
    pub member: Option<String>,
    pub score: f32,
}

/// `GET /v1/compound/{topic}` response: ranked hits + `canonical_id` →
/// indices-into-`hits` clusters (the "solved in N repos" grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundResult {
    pub hits: Vec<CompoundHit>,
    pub clusters: BTreeMap<String, Vec<usize>>,
}

/// One per-(member,stream) watermark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorState {
    pub member: String,
    pub stream: String,
    pub head: u64,
}

/// `GET /v1/sync/bootstrap` payload: current materialized distillate + the
/// cursor to tail from, so a new/offline device skips full oplog replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapSnapshot {
    pub cursor: u64,
    pub distillate: Vec<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opkind_wire_names_use_dots() {
        assert_eq!(
            serde_json::to_string(&OpKind::RecordUpsert).unwrap(),
            "\"record.upsert\""
        );
        assert_eq!(
            serde_json::to_string(&OpKind::CanonicalPublish).unwrap(),
            "\"canonical.publish\""
        );
        let k: OpKind = serde_json::from_str("\"edge.add\"").unwrap();
        assert_eq!(k, OpKind::EdgeAdd);
    }

    #[test]
    fn tier_defaults_to_distillate_and_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Tier::Distillate).unwrap(), "\"distillate\"");
        // Missing `tier` on an incoming op defaults to distillate.
        let op: Op = serde_json::from_str(
            r#"{"op_id":"x","kind":"record.upsert","table":"decisions","member":"m","content_hash":"h"}"#,
        )
        .unwrap();
        assert_eq!(op.tier, Tier::Distillate);
    }

    #[test]
    fn seqop_flattens_op_fields() {
        let so = SeqOp {
            seq: 7,
            op: Op {
                op_id: "op1".into(),
                kind: OpKind::RecordUpsert,
                table: "decisions".into(),
                member: "m".into(),
                tier: Tier::Distillate,
                content_hash: "h".into(),
                lww: None,
                payload: serde_json::json!({"k": 1}),
                embedding: None,
            },
        };
        let v: Value = serde_json::to_value(&so).unwrap();
        // seq and the op's fields sit at the same level (flattened).
        assert_eq!(v["seq"], 7);
        assert_eq!(v["op_id"], "op1");
        assert_eq!(v["table"], "decisions");
    }
}
