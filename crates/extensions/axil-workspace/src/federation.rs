//! Query-time fan-out across sibling DBs.
//!
//! `axil-workspace` does not link against the plugin crates; it provides a
//! generic fan-out driver that accepts a caller-supplied "open + query"
//! closure so the CLI stays in charge of which plugins each sibling gets.
//!
//! The driver is responsible for:
//!   1. Iterating the manifest's members
//!   2. Applying `read_consent` at the *remote* (records carry their own scope)
//!   3. Fusing results with RRF + a local-boost
//!   4. Decorating each result with provenance metadata

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::consent::{MatchContext, ReadConsent};
use crate::manifest::{Member, MemberId, RoleId, WorkspaceId, WorkspaceManifest};

/// Single result row from fan-out, carrying its provenance so the caller
/// can still write back to the originating DB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Final fused score (higher = better).
    pub score: f32,
    pub source_workspace_id: WorkspaceId,
    pub source_member_id: MemberId,
    pub source_member_label: String,
    pub source_record_id: String,
    /// The raw per-sibling recall result payload (record + explanation).
    pub record: Value,
    /// True when the sibling returned text/FTS-only results because it
    /// disagreed with the local vector profile.
    pub text_only_fallback: bool,
}

/// Produced by a sibling's `recall()` path. `vector_compatible = false`
/// tells the fuser this member is on a different embedder and its scores
/// shouldn't be compared with vector hits from other members.
#[derive(Debug, Clone)]
pub struct MemberRecallBatch {
    pub member_label: String,
    pub member_id: MemberId,
    pub workspace_id: WorkspaceId,
    pub vector_compatible: bool,
    pub rows: Vec<MemberRecallRow>,
    /// Soft warnings to surface in `--trace` mode (e.g. "db missing").
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MemberRecallRow {
    pub record_id: String,
    pub record: Value,
    pub score: f32,
    pub read_consent: ReadConsent,
}

/// Parameters for a fan-out call.
pub struct FederationRequest<'a> {
    pub manifest: &'a WorkspaceManifest,
    /// Caller's workspace id + member id — for consent evaluation at the remote.
    pub caller_workspace: WorkspaceId,
    pub caller_member: MemberId,
    pub caller_roles: Vec<RoleId>,
    /// Members chosen by `--across`.
    pub members: Vec<(String, &'a Member)>,
    /// Top-K requested by the caller.
    pub top_k: usize,
    /// When true, drop `ReadConsent::Workspace` records.
    pub strict_consent: bool,
}

/// Drive the fan-out.
///
/// `recall_fn` is invoked once per member, receives the absolute DB path
/// and member metadata, and returns a `MemberRecallBatch`. Any `Err`
/// raised by the closure becomes a soft warning on the corresponding
/// member's entry — unreachable siblings must never prevent reachable
/// ones from answering.
pub fn fan_out<F>(
    req: FederationRequest<'_>,
    mut recall_fn: F,
) -> (Vec<FederatedResult>, Vec<String>)
where
    F: FnMut(
        &str, // member_label
        &Member,
        std::path::PathBuf, // abs DB path
    ) -> Result<MemberRecallBatch, String>,
{
    const RRF_K: f32 = 60.0;
    let local_boost = req.manifest.federation.local_boost;

    // Collect batches with their inputs.
    let mut batches: Vec<MemberRecallBatch> = Vec::with_capacity(req.members.len());
    for (label, member) in &req.members {
        let abs = req.manifest.member_db_abs(member);
        match recall_fn(label, member, abs.clone()) {
            Ok(batch) => batches.push(batch),
            Err(msg) => {
                // Emit an empty batch with the warning so the caller can
                // surface it in `--trace`.
                batches.push(MemberRecallBatch {
                    member_label: (*label).to_string(),
                    member_id: member.id.clone(),
                    workspace_id: req.manifest.workspace.id.clone(),
                    vector_compatible: false,
                    rows: Vec::new(),
                    warnings: vec![format!("member '{label}' unreachable: {msg}")],
                });
            }
        }
    }

    // Apply consent at the remote and stash RRF contributions.
    let mut merged: HashMap<String, FederatedResult> = HashMap::new();
    for batch in &batches {
        let source_ws = batch.workspace_id.clone();
        let source_member = batch.member_id.clone();
        let is_local = source_ws == req.caller_workspace && source_member == req.caller_member;

        for (rank, row) in batch.rows.iter().enumerate() {
            let ctx = MatchContext {
                source_workspace: &source_ws,
                source_member: &source_member,
                caller_workspace: &req.caller_workspace,
                caller_member: &req.caller_member,
                caller_roles: &req.caller_roles,
                strict: req.strict_consent,
            };
            if !row.read_consent.allows(&ctx) {
                continue;
            }

            let rrf = 1.0 / (RRF_K + rank as f32);
            let boost = if is_local { local_boost } else { 0.0 };
            let score = row.score + rrf + boost;

            let key = format!("{}:{}", batch.member_id, row.record_id);
            let entry = merged.entry(key).or_insert_with(|| FederatedResult {
                score: 0.0,
                source_workspace_id: source_ws.clone(),
                source_member_id: source_member.clone(),
                source_member_label: batch.member_label.clone(),
                source_record_id: row.record_id.clone(),
                record: row.record.clone(),
                text_only_fallback: !batch.vector_compatible,
            });
            if score > entry.score {
                entry.score = score;
            }
        }
    }

    let mut out: Vec<FederatedResult> = merged.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(req.top_k);

    let warnings: Vec<String> = batches.into_iter().flat_map(|b| b.warnings).collect();
    (out, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::MANIFEST_FILENAME;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    fn seed() -> (TempDir, WorkspaceManifest) {
        let tmp = TempDir::new().unwrap();
        let body = r#"
[workspace]
id = "ws_x"
name = "acme"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"

[members.backend]
id = "mem_be"
root = "./backend"
path = "./backend/.axil/memory.axil"
"#;
        fs::write(tmp.path().join(MANIFEST_FILENAME), body).unwrap();
        let manifest = WorkspaceManifest::load(tmp.path().join(MANIFEST_FILENAME)).unwrap();
        (tmp, manifest)
    }

    #[test]
    fn consent_filters_at_remote() {
        let (_tmp, manifest) = seed();
        let (members, _) = manifest.resolve_members_arg("*");
        let req = FederationRequest {
            manifest: &manifest,
            caller_workspace: manifest.workspace.id.clone(),
            caller_member: "mem_fe".to_string(),
            caller_roles: vec![],
            members: members
                .into_iter()
                .map(|(l, m)| (l.to_string(), m))
                .collect(),
            top_k: 10,
            strict_consent: false,
        };
        let (out, warnings) = fan_out(req, |label, member, _path| {
            let is_fe = member.id == "mem_fe";
            Ok(MemberRecallBatch {
                member_label: label.to_string(),
                member_id: member.id.clone(),
                workspace_id: "ws_x".to_string(),
                vector_compatible: true,
                rows: vec![MemberRecallRow {
                    record_id: if is_fe { "fe_1".into() } else { "be_1".into() },
                    record: json!({"summary": "x"}),
                    score: 0.5,
                    // backend record is private → should be filtered out.
                    read_consent: if is_fe {
                        ReadConsent::Private
                    } else {
                        ReadConsent::Private
                    },
                }],
                warnings: vec![],
            })
        });
        // Only the local (frontend) record survives; remote is Private.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source_record_id, "fe_1");
        assert!(warnings.is_empty());
    }

    #[test]
    fn workspace_scope_visible_but_dropped_under_strict() {
        let (_tmp, manifest) = seed();
        let (members, _) = manifest.resolve_members_arg("*");
        let make_request = |strict: bool| FederationRequest {
            manifest: &manifest,
            caller_workspace: manifest.workspace.id.clone(),
            caller_member: "mem_fe".to_string(),
            caller_roles: vec![],
            members: members.iter().map(|(l, m)| (l.to_string(), *m)).collect(),
            top_k: 10,
            strict_consent: strict,
        };

        let closure = |label: &str, member: &Member, _path: std::path::PathBuf| {
            Ok(MemberRecallBatch {
                member_label: label.to_string(),
                member_id: member.id.clone(),
                workspace_id: "ws_x".to_string(),
                vector_compatible: true,
                rows: vec![MemberRecallRow {
                    record_id: format!("rec_{}", member.id),
                    record: json!({}),
                    score: 0.5,
                    read_consent: ReadConsent::Workspace,
                }],
                warnings: vec![],
            })
        };

        let (lax, _) = fan_out(make_request(false), closure);
        let (strict, _) = fan_out(make_request(true), closure);
        assert_eq!(lax.len(), 2, "workspace scope visible to siblings");
        assert_eq!(strict.len(), 1, "strict drops workspace-scoped remotes");
    }
}
