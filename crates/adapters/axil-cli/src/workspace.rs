//! CLI handlers for Phase 14 workspace / consent / bridge commands.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use axil_core::RecordId;
use axil_workspace::bridge::{BridgeEvidence, EntityBridge};
use axil_workspace::consent::{parse_read_consent, parse_write_consent};
#[cfg(feature = "embed")]
use axil_workspace::federation::{fan_out, FederationRequest, MemberRecallBatch, MemberRecallRow};
use axil_workspace::manifest::{
    new_member_id, new_workspace_id, Federation, Member, Role, WorkspaceManifest, WorkspaceSection,
    MANIFEST_FILENAME,
};
use axil_workspace::registry::{global_registry_path, GlobalRegistry, RegistryEntry};
use axil_workspace::resolve::{discover_manifest, unbound_status};
use chrono::Utc;
use serde_json::{json, Value};

use crate::{Output, EXIT_OK};

pub fn handle_workspace(
    op: crate::WorkspaceOp,
    db_opt: &Option<PathBuf>,
    out: &Output,
) -> Result<i32> {
    match op {
        crate::WorkspaceOp::Init { name } => init_workspace(name, out),
        crate::WorkspaceOp::Status => status_workspace(db_opt, out),
        crate::WorkspaceOp::List => list_workspaces(out),
        crate::WorkspaceOp::Add { path, as_label } => add_member(path, as_label, out),
    }
}

/// Parse CLI `--read` / `--write` consent arguments into JSON values.
fn parse_consent_pair(
    read: Option<&str>,
    write: Option<&str>,
) -> Result<(Option<Value>, Option<Value>)> {
    let read_val = match read {
        Some(v) => Some(serde_json::to_value(
            parse_read_consent(v).map_err(anyhow::Error::msg)?,
        )?),
        None => None,
    };
    let write_val = match write {
        Some(v) => Some(serde_json::to_value(
            parse_write_consent(v).map_err(anyhow::Error::msg)?,
        )?),
        None => None,
    };
    Ok((read_val, write_val))
}

pub fn handle_consent(op: crate::ConsentOp, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    match op {
        crate::ConsentOp::Set {
            record_id,
            read,
            write,
        } => {
            let db_path = crate::require_db(db_opt)?;
            let db = crate::open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&record_id).context("invalid record ID")?;
            let (read_val, write_val) = parse_consent_pair(read.as_deref(), write.as_deref())?;
            let updated = db.set_record_consent(&rid, read_val, write_val)?;
            out.print(&json!({
                "id": updated.id.to_string(),
                "table": updated.table,
                "read_consent": updated.read_consent_raw(),
                "write_consent": updated.write_consent_raw(),
            }));
            Ok(EXIT_OK)
        }
        crate::ConsentOp::Show { record_id } => {
            let db_path = crate::require_db(db_opt)?;
            let db = crate::open_with_all_detected(&db_path)?;
            let rid = RecordId::from_string(&record_id).context("invalid record ID")?;
            let (read, write) = db.get_record_consent(&rid)?;
            out.print(&json!({
                "id": record_id,
                "read_consent": read,
                "write_consent": write,
            }));
            Ok(EXIT_OK)
        }
        crate::ConsentOp::Audit {
            since,
            audit_format,
        } => {
            let format = audit_format;
            let db_path = crate::require_db(db_opt)?;
            let db = crate::open_with_all_detected(&db_path)?;
            let rows = db.list("_consent_log").unwrap_or_default();
            let filtered: Vec<&axil_core::Record> = rows
                .iter()
                .filter(|r| {
                    let ts = r
                        .data
                        .get("timestamp")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    match since.as_deref() {
                        Some(after) => ts > after,
                        None => true,
                    }
                })
                .collect();
            match format.as_str() {
                "csv" => {
                    println!("timestamp,record_id,table,read_kind,write_kind");
                    for r in filtered {
                        let ts = r
                            .data
                            .get("timestamp")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let rid = r
                            .data
                            .get("record_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let table = r.data.get("table").and_then(|v| v.as_str()).unwrap_or("");
                        let read_kind = r
                            .data
                            .get("read_consent")
                            .and_then(|v| v.get("kind"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let write_kind = r
                            .data
                            .get("write_consent")
                            .and_then(|v| v.get("kind"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        println!("{ts},{rid},{table},{read_kind},{write_kind}");
                    }
                }
                _ => {
                    let values: Vec<Value> = filtered.iter().map(|r| r.data.clone()).collect();
                    out.print(&json!({"entries": values}));
                }
            }
            return Ok(EXIT_OK);
        }
        crate::ConsentOp::Default { table, read, write } => {
            let db_path = crate::require_db(db_opt)?;
            let db = crate::open_with_all_detected(&db_path)?;
            let (read_val, write_val) = parse_consent_pair(read.as_deref(), write.as_deref())?;
            db.set_consent_default(&table, read_val.clone(), write_val.clone())?;
            out.print(&json!({
                "table": table,
                "read_consent": read_val,
                "write_consent": write_val,
            }));
            Ok(EXIT_OK)
        }
    }
}

pub fn handle_bridge(op: crate::BridgeOp, db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    let db_path = crate::require_db(db_opt)?;
    let db = crate::open_with_all_detected(&db_path)?;
    match op {
        crate::BridgeOp::Add {
            local,
            to,
            evidence,
            confidence,
        } => {
            let (member, remote_canonical) = parse_bridge_target(&to)?;
            let manifest = discover_manifest(&db_path)?.ok_or_else(|| {
                anyhow!("no workspace manifest found — run `axil workspace init` first")
            })?;
            let (matched, unknown) = manifest.resolve_members_arg(&member);
            if !unknown.is_empty() {
                anyhow::bail!("unknown member '{member}'");
            }
            let (member_label, member_obj) = matched
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("member '{member}' not found"))?;

            let ev = match evidence.as_str() {
                "manual" | "manual_assert" => BridgeEvidence::ManualAssert,
                "scip" | "scip_symbol" => BridgeEvidence::ScipSymbol {
                    symbol: remote_canonical.clone(),
                },
                "shared_uri" | "uri" => BridgeEvidence::SharedUri {
                    uri: remote_canonical.clone(),
                },
                "name_and_type" => BridgeEvidence::NameAndType {
                    name: local.clone(),
                    type_name: "unknown".to_string(),
                },
                other => anyhow::bail!(
                    "unknown evidence kind '{other}'; expected manual | scip_symbol | shared_uri | name_and_type"
                ),
            };
            let conf = confidence
                .unwrap_or_else(|| ev.default_confidence())
                .clamp(0.0, 1.0);

            let bridge = EntityBridge {
                local_canonical: local.clone(),
                remote_workspace_id: manifest.workspace.id.clone(),
                remote_member_id: member_obj.id.clone(),
                remote_canonical: remote_canonical.clone(),
                confidence: conf,
                evidence: ev,
                asserted_at: Utc::now(),
                asserted_by: axil_workspace::bridge::AssertSource::Human,
                dangling: false,
            };
            let stored = db.upsert_bridge(&serde_json::to_value(&bridge)?)?;
            out.print(&json!({
                "bridge_id": stored.id.to_string(),
                "local": local,
                "member": member_label,
                "remote": remote_canonical,
                "confidence": conf,
            }));
            Ok(EXIT_OK)
        }
        crate::BridgeOp::List { local, member } => {
            let manifest = discover_manifest(&db_path)?;
            let remote_member_id = match (&manifest, member.as_deref()) {
                (Some(m), Some(label)) => {
                    let (matched, _) = m.resolve_members_arg(label);
                    matched
                        .into_iter()
                        .next()
                        .map(|(_, member)| member.id.clone())
                }
                _ => None,
            };
            let rows = db.list_bridges(local.as_deref(), remote_member_id.as_deref())?;
            let values: Vec<Value> = rows
                .into_iter()
                .map(|r| {
                    let mut data = r.data;
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("id".to_string(), json!(r.id.to_string()));
                    }
                    data
                })
                .collect();
            out.print(&json!({
                "bridges": values,
            }));
            Ok(EXIT_OK)
        }
        crate::BridgeOp::Verify => {
            let (verified, dangling) = db.verify_bridges()?;
            out.print(&json!({
                "verified": verified,
                "dangling": dangling,
            }));
            Ok(EXIT_OK)
        }
        crate::BridgeOp::Auto { members, dry_run } => {
            let manifest = discover_manifest(&db_path)?.ok_or_else(|| {
                anyhow!("no workspace manifest found — run `axil workspace init` first")
            })?;

            let (targets, unknown) = manifest.resolve_members_arg(&members);
            if !unknown.is_empty() && members != "*" {
                anyhow::bail!("unknown member(s): {}", unknown.join(","));
            }
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let caller = axil_workspace::resolve::resolve_member(&manifest, &cwd);
            let caller_id = caller
                .as_ref()
                .map(|r| r.member_id.clone())
                .unwrap_or_default();

            let local_canonicals: std::collections::HashSet<String> = db
                .list("_entities")?
                .into_iter()
                .filter_map(|r| scip_canonical_id(&r))
                .collect();

            let local_uris = collect_record_uris(&db);

            let mut plan: Vec<Value> = Vec::new();
            let mut created = 0usize;
            for (label, member) in &targets {
                if member.id == caller_id {
                    continue;
                }
                // Core-only open: the scan only reads `_entities` from the
                // core store, so skip FTS/vector/ONNX attach.
                let remote_db = match axil_core::Axil::open(manifest.member_db_abs(member)).build()
                {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let remote_canonicals: std::collections::HashSet<String> = remote_db
                    .list("_entities")
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|r| scip_canonical_id(&r))
                    .collect();
                for local_cid in &local_canonicals {
                    if remote_canonicals.contains(local_cid) {
                        let bridge = EntityBridge {
                            local_canonical: local_cid.clone(),
                            remote_workspace_id: manifest.workspace.id.clone(),
                            remote_member_id: member.id.clone(),
                            remote_canonical: local_cid.clone(),
                            confidence: 1.0,
                            evidence: BridgeEvidence::ScipSymbol {
                                symbol: local_cid.clone(),
                            },
                            asserted_at: Utc::now(),
                            asserted_by: axil_workspace::bridge::AssertSource::Scip,
                            dangling: false,
                        };
                        plan.push(json!({
                            "local_canonical": local_cid,
                            "member": label,
                            "remote_canonical": local_cid,
                            "confidence": 1.0,
                            "evidence": "scip_symbol",
                        }));
                        if !dry_run {
                            db.upsert_bridge(&serde_json::to_value(&bridge)?)?;
                            created += 1;
                        }
                    }
                }

                // URI matcher (14.5 second pass): records tagged with a
                // shared OpenAPI operationId or GraphQL type auto-bridge
                // at confidence 0.9. The bridge carries the URI as both
                // local and remote canonical so a future URI resolver
                // can find both sides from a single query.
                let remote_uris = collect_record_uris(&remote_db);
                for (uri, local_record_ids) in &local_uris {
                    let Some(remote_record_ids) = remote_uris.get(uri) else {
                        continue;
                    };
                    if local_record_ids.is_empty() || remote_record_ids.is_empty() {
                        continue;
                    }
                    let pseudo_canonical = format!("uri:{uri}");
                    let bridge = EntityBridge {
                        local_canonical: pseudo_canonical.clone(),
                        remote_workspace_id: manifest.workspace.id.clone(),
                        remote_member_id: member.id.clone(),
                        remote_canonical: pseudo_canonical.clone(),
                        confidence: 0.9,
                        evidence: BridgeEvidence::SharedUri { uri: uri.clone() },
                        asserted_at: Utc::now(),
                        asserted_by: axil_workspace::bridge::AssertSource::Heuristic,
                        dangling: false,
                    };
                    plan.push(json!({
                        "local_canonical": pseudo_canonical,
                        "member": label,
                        "remote_canonical": pseudo_canonical,
                        "confidence": 0.9,
                        "evidence": "shared_uri",
                        "uri": uri,
                    }));
                    if !dry_run {
                        db.upsert_bridge(&serde_json::to_value(&bridge)?)?;
                        created += 1;
                    }
                }
            }

            out.print(&json!({
                "dry_run": dry_run,
                "created": created,
                "plan": plan,
                "min_confidence_threshold": manifest.federation.min_bridge_confidence,
            }));
            Ok(EXIT_OK)
        }
    }
}

/// `axil trace-record <member>:<id>` — show blast-radius counters and
/// bridge rows referencing the target record. Reads from the caller's
/// `_recall_blast_radius` + `_entity_bridges`; when a workspace
/// manifest exists, also looks up the named member's metadata.
pub fn handle_trace_record(db_opt: &Option<PathBuf>, out: &Output, target: &str) -> Result<i32> {
    let db_path = crate::require_db(db_opt)?;
    let db = crate::open_with_all_detected(&db_path)?;

    let (member_label, record_id) = match target.split_once(':') {
        Some((m, r)) if !m.is_empty() && !r.is_empty() => (Some(m.to_string()), r.to_string()),
        _ => (None, target.to_string()),
    };

    let manifest = discover_manifest(&db_path)?;
    let member_id = match (&manifest, &member_label) {
        (Some(m), Some(label)) => {
            let (matched, _) = m.resolve_members_arg(label);
            matched.into_iter().next().map(|(_, mem)| mem.id.clone())
        }
        _ => None,
    };

    let blast: Vec<Value> = db
        .list("_recall_blast_radius")
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.data.get("source_record_id").and_then(|v| v.as_str()) == Some(&record_id))
        .filter(|r| match member_id.as_deref() {
            Some(mid) => r.data.get("source_member_id").and_then(|v| v.as_str()) == Some(mid),
            None => true,
        })
        .map(|r| r.data)
        .collect();

    let bridges_as_local: Vec<Value> = db
        .list_bridges(Some(&record_id), None)
        .unwrap_or_default()
        .into_iter()
        .map(|r| r.data)
        .collect();
    let bridges_as_remote: Vec<Value> = db
        .list("_entity_bridges")
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.data.get("remote_canonical").and_then(|v| v.as_str()) == Some(&record_id))
        .map(|r| r.data)
        .collect();

    out.print(&json!({
        "target": target,
        "resolved_member": member_label,
        "resolved_member_id": member_id,
        "record_id": record_id,
        "blast_radius": blast,
        "bridges_as_local_canonical": bridges_as_local,
        "bridges_as_remote_canonical": bridges_as_remote,
    }));
    Ok(EXIT_OK)
}

#[cfg(feature = "embed")]
pub fn handle_recall_across(
    db_opt: &Option<PathBuf>,
    out: &Output,
    query: &str,
    across: &str,
    top_k: usize,
    strict_consent: bool,
    trace: bool,
    oneline: bool,
) -> Result<i32> {
    let db_path = crate::require_db(db_opt)?;
    let manifest = discover_manifest(&db_path)?.ok_or_else(|| {
        anyhow!("no workspace manifest found — run `axil workspace init` first, then retry")
    })?;

    let (members, unknown) = manifest.resolve_members_arg(across);
    if !unknown.is_empty() && across != "*" {
        anyhow::bail!("unknown member(s): {}", unknown.join(","));
    }
    if members.is_empty() {
        anyhow::bail!("no members matched --across '{across}'");
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let (caller_member, caller_roles) =
        match axil_workspace::resolve::resolve_member(&manifest, &cwd) {
            Some(res) => {
                let roles = manifest
                    .members
                    .get(&res.member_label)
                    .map(|m| m.roles.clone())
                    .unwrap_or_default();
                (res.member_id, roles)
            }
            None => (String::new(), Vec::new()),
        };

    let local_dims = read_vector_dims(&db_path);
    let members_owned: Vec<(String, &Member)> = members
        .iter()
        .map(|(label, member)| ((*label).to_string(), *member))
        .collect();

    let workspace_id = manifest.workspace.id.clone();
    let req = FederationRequest {
        manifest: &manifest,
        caller_workspace: workspace_id.clone(),
        caller_member: caller_member.clone(),
        caller_roles,
        members: members_owned,
        top_k,
        strict_consent,
    };

    let mut warnings: Vec<String> = Vec::new();
    let (results, fan_out_warnings) = fan_out(req, |label, member, path| {
        sibling_recall(
            &path,
            query,
            top_k,
            local_dims,
            label,
            member,
            &workspace_id,
            &mut warnings,
        )
    });
    warnings.extend(fan_out_warnings);

    // Open the caller DB once: blast-radius write + optional trace / oneline
    // read both touch `_recall_blast_radius`.
    let local_db = crate::open_with_all_detected(&db_path)?;
    record_blast_radius(&local_db, &results, &workspace_id, &caller_member)?;
    let blast_radius_map = if trace || oneline {
        load_blast_radius_map(&local_db)
    } else {
        Default::default()
    };

    if oneline {
        for r in &results {
            println!("{}", format_oneline(r, &blast_radius_map));
        }
        for w in &warnings {
            eprintln!("warning: {w}");
        }
        return Ok(EXIT_OK);
    }

    let value = json!({
        "query": query,
        "across": across,
        "strict_consent": strict_consent,
        "results": results
            .iter()
            .map(|r| render_result_row(r, trace, &blast_radius_map))
            .collect::<Vec<_>>(),
        "warnings": warnings,
    });
    out.print(&value);
    Ok(EXIT_OK)
}

/// Format a single federated result as
/// `[member] [score=0.88] "summary"  (BR=N callers, Mq/30d)` when the
/// record has cross-project recall history. Falls back to a short
/// `(BR=Nq)` form when there's no caller data and to no tag at all
/// for unrecalled records.
fn format_oneline(
    r: &axil_workspace::federation::FederatedResult,
    blast_radius_map: &std::collections::HashMap<String, BlastRadiusStats>,
) -> String {
    let summary = axil_core::util::value_text_legacy(&r.record).replace('\n', " ");
    let truncated = axil_core::util::truncate_str(&summary, 120);
    let summary_display = if truncated.len() < summary.len() {
        format!("{truncated}…")
    } else {
        summary.clone()
    };
    let key = format!("{}|{}", r.source_member_id, r.source_record_id);
    let br_tag = match blast_radius_map.get(&key) {
        Some(s) if s.callers > 0 => {
            format!("  (BR={} callers, {}q/30d)", s.callers, s.queries_30d)
        }
        Some(s) if s.queries > 0 => format!("  (BR={}q)", s.queries),
        _ => String::new(),
    };
    let fallback_tag = if r.text_only_fallback {
        "  [text-only]"
    } else {
        ""
    };
    format!(
        "[{}] [score={:.2}] \"{}\"{}{}",
        r.source_member_label, r.score, summary_display, br_tag, fallback_tag
    )
}

fn render_result_row(
    r: &axil_workspace::federation::FederatedResult,
    trace: bool,
    blast_radius_map: &std::collections::HashMap<String, BlastRadiusStats>,
) -> Value {
    if trace {
        let key = format!("{}|{}", r.source_member_id, r.source_record_id);
        let stats = blast_radius_map.get(&key).cloned().unwrap_or_default();
        json!({
            "score": r.score,
            "member": r.source_member_label,
            "member_id": r.source_member_id,
            "record_id": r.source_record_id,
            "record": r.record,
            "text_only_fallback": r.text_only_fallback,
            "blast_radius": {
                "queries": stats.queries,
                "callers": stats.callers,
                "queries_30d": stats.queries_30d,
            },
        })
    } else {
        json!({
            "score": r.score,
            "member": r.source_member_label,
            "record_id": r.source_record_id,
            "record": r.record,
        })
    }
}

/// Bump `_recall_blast_radius` counters for every cross-member hit in a
/// single result batch. One full-table scan up front, then in-memory
/// updates via `db.update` / `db.insert`.
fn record_blast_radius(
    db: &axil_core::Axil,
    results: &[axil_workspace::federation::FederatedResult],
    workspace_id: &str,
    caller_member: &str,
) -> Result<()> {
    let remote_hits: Vec<&axil_workspace::federation::FederatedResult> = results
        .iter()
        .filter(|r| !(r.source_workspace_id == workspace_id && r.source_member_id == caller_member))
        .collect();
    if remote_hits.is_empty() {
        return Ok(());
    }

    let mut existing: std::collections::HashMap<String, (axil_core::RecordId, Value)> =
        std::collections::HashMap::new();
    for row in db.list("_recall_blast_radius").unwrap_or_default() {
        if let Some(k) = row.data.get("key").and_then(|v| v.as_str()) {
            existing.insert(k.to_string(), (row.id, row.data));
        }
    }

    let now_rfc = Utc::now().to_rfc3339();
    for r in remote_hits {
        let key = format!(
            "{caller_member}|{}|{}",
            r.source_member_id, r.source_record_id
        );
        if let Some((rid, mut data)) = existing.remove(&key) {
            let current = data.get("queries").and_then(|v| v.as_u64()).unwrap_or(0);
            if let Some(obj) = data.as_object_mut() {
                obj.insert("queries".to_string(), json!(current + 1));
                obj.insert("last_seen_at".to_string(), json!(now_rfc.clone()));
            }
            db.update(&rid, data)?;
        } else {
            db.insert(
                "_recall_blast_radius",
                json!({
                    "key": key,
                    "caller_member": caller_member,
                    "source_workspace_id": r.source_workspace_id,
                    "source_member_id": r.source_member_id,
                    "source_record_id": r.source_record_id,
                    "queries": 1,
                    "first_seen_at": now_rfc,
                    "last_seen_at": now_rfc,
                }),
            )?;
        }
    }
    Ok(())
}

/// Per-record blast-radius stats aggregated from `_recall_blast_radius`.
/// `callers` counts distinct caller members; `queries_30d` sums queries
/// recorded in the trailing 30-day window using each row's
/// `last_seen_at`.
#[derive(Debug, Clone, Default)]
struct BlastRadiusStats {
    queries: u64,
    callers: usize,
    queries_30d: u64,
}

fn load_blast_radius_map(
    db: &axil_core::Axil,
) -> std::collections::HashMap<String, BlastRadiusStats> {
    let mut acc: std::collections::HashMap<String, (u64, std::collections::HashSet<String>, u64)> =
        std::collections::HashMap::new();
    let now = chrono::Utc::now();
    let cutoff = now - chrono::Duration::days(30);

    for row in db.list("_recall_blast_radius").unwrap_or_default() {
        let Some(member_id) = row.data.get("source_member_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(record_id) = row.data.get("source_record_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let queries = row
            .data
            .get("queries")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let caller = row
            .data
            .get("caller_member")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let last_seen = row
            .data
            .get("last_seen_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&chrono::Utc));

        let key = format!("{member_id}|{record_id}");
        let entry = acc.entry(key).or_default();
        entry.0 = entry.0.saturating_add(queries);
        if !caller.is_empty() {
            entry.1.insert(caller);
        }
        if last_seen.map(|t| t >= cutoff).unwrap_or(false) {
            entry.2 = entry.2.saturating_add(queries);
        }
    }

    acc.into_iter()
        .map(|(k, (q, callers, q30))| {
            (
                k,
                BlastRadiusStats {
                    queries: q,
                    callers: callers.len(),
                    queries_30d: q30,
                },
            )
        })
        .collect()
}

/// Best-effort: open a sibling DB, run its own `recall`, translate results
/// into the federation row shape. Unreachable or unopenable siblings map
/// to `Err` so `fan_out` can surface a warning.
#[cfg(feature = "embed")]
fn sibling_recall(
    path: &Path,
    query: &str,
    top_k: usize,
    local_dims: Option<usize>,
    label: &str,
    member: &Member,
    workspace_id: &str,
    warnings: &mut Vec<String>,
) -> std::result::Result<MemberRecallBatch, String> {
    let sibling_dims = read_vector_dims(path);
    let vector_compatible = match (local_dims, sibling_dims) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    };

    let db = match crate::open_with_all_detected(path) {
        Ok(db) => db,
        Err(e) => {
            warnings.push(format!("member '{}' failed to open: {}", label, e));
            return Err(e.to_string());
        }
    };

    let rows = match db.recall(query, top_k, None) {
        Ok(r) => r,
        Err(e) => {
            warnings.push(format!("member '{}' recall failed: {}", label, e));
            return Err(e.to_string());
        }
    };

    let member_rows: Vec<MemberRecallRow> = rows
        .into_iter()
        .map(|r| {
            let read_consent: axil_workspace::consent::ReadConsent =
                serde_json::from_value(r.record.read_consent_raw()).unwrap_or_default();
            let record_id = r.record.id.to_string();
            let record = serde_json::json!({
                "id": record_id,
                "table": r.record.table,
                "summary": r.record.data.get("summary")
                    .or_else(|| r.record.data.get("fact"))
                    .or_else(|| r.record.data.get("error"))
                    .cloned()
                    .unwrap_or(r.record.data.clone()),
                "explanation": r.explanation.summary,
            });
            MemberRecallRow {
                record_id,
                record,
                score: r.score,
                read_consent,
            }
        })
        .collect();

    Ok(MemberRecallBatch {
        member_label: label.to_string(),
        member_id: member.id.clone(),
        workspace_id: workspace_id.to_string(),
        vector_compatible,
        rows: member_rows,
        warnings: Vec::new(),
    })
}

/// Scan every non-internal record for URI-ish fields that identify the
/// same logical resource across members. Covers OpenAPI `operation_id`
/// and `operationId`, GraphQL `graphql_type`, and a generic `uri` /
/// `api_uri` field. Returns `uri → [record_id…]` so the auto-bridge
/// scan can cross-match values between siblings without double-adding
/// the same URI for records within a single DB.
fn collect_record_uris(db: &axil_core::Axil) -> std::collections::HashMap<String, Vec<String>> {
    const URI_FIELDS: &[&str] = &[
        "uri",
        "api_uri",
        "operation_id",
        "operationId",
        "graphql_type",
    ];
    let mut out: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let tables = db.tables().unwrap_or_default();
    for table in &tables {
        if table.starts_with('_') {
            continue;
        }
        let Ok(records) = db.list(table) else {
            continue;
        };
        for r in records {
            let Some(obj) = r.data.as_object() else {
                continue;
            };
            for field in URI_FIELDS {
                if let Some(val) = obj.get(*field).and_then(|v| v.as_str()) {
                    let trimmed = val.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    out.entry(trimmed.to_string())
                        .or_default()
                        .push(r.id.to_string());
                }
            }
        }
    }
    out
}

/// Extract a record's SCIP canonical id, skipping `provisional:` rows
/// and SCIP `local N` symbols.
///
/// `local N` is the SCIP escape hatch for document-scoped identifiers
/// (block-locals, lambda captures, anonymous structs). Two members can
/// each have a `local 27` that refer to entirely unrelated symbols in
/// unrelated files — bridging them at confidence 1.0 produces nonsense.
/// See https://github.com/sourcegraph/scip/blob/main/scip.proto for the
/// `local <id>` grammar.
fn scip_canonical_id(record: &axil_core::Record) -> Option<String> {
    let cid = record.data.get("canonical_id")?.as_str()?;
    if cid.starts_with("provisional:") || cid.starts_with("local ") {
        return None;
    }
    Some(cid.to_string())
}

fn read_vector_dims(_db_path: &Path) -> Option<usize> {
    #[cfg(feature = "vector")]
    {
        return axil_vector::read_stored_dimensions(_db_path).ok().flatten();
    }
    #[cfg(not(feature = "vector"))]
    {
        None
    }
}

fn parse_bridge_target(raw: &str) -> Result<(String, String)> {
    let (member, canonical) = raw
        .split_once(':')
        .ok_or_else(|| anyhow!("expected <member>:<canonical_id>, got '{raw}'"))?;
    let member = member.trim();
    let canonical = canonical.trim();
    if member.is_empty() || canonical.is_empty() {
        anyhow::bail!("--to must be '<member>:<canonical_id>', both non-empty");
    }
    Ok((member.to_string(), canonical.to_string()))
}

// ── workspace init / status / list / add ─────────────────────────────

/// Directory names pruned from `workspace init`'s walk. Matches the
/// common noise across Rust, Node, Python, and framework toolchains so
/// a depth-4 descent doesn't balloon on monorepos.
const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".turbo",
    ".cache",
    ".parcel-cache",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    "vendor",
];

const ROLE_UI_TOKENS: &[&str] = &["front", "frontend", "ui", "web", "client"];
const ROLE_API_TOKENS: &[&str] = &["back", "backend", "api", "server", "service"];
const ROLE_OPS_TOKENS: &[&str] = &["infra", "ops", "deploy", "platform"];
const ROLE_MOBILE_TOKENS: &[&str] = &["mobile", "ios", "android", "flutter", "rn"];

/// Sniff a project root for marker files and return role IDs (e.g.
/// "role_mobile", "role_api", "role_ui", "role_ops"). Reads at most a few
/// kilobytes per project — cheap to run on every member during init.
///
/// Detection is conservative: only files at the root are inspected, and
/// we look at file contents only when the filename alone is ambiguous
/// (e.g. `package.json` could be web, api, or library).
fn detect_roles_from_files(root: &Path) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut out: BTreeSet<String> = BTreeSet::new();

    let exists = |name: &str| root.join(name).exists();
    let read_head = |name: &str| -> Option<String> {
        let path = root.join(name);
        std::fs::File::open(&path).ok().and_then(|f| {
            use std::io::Read;
            let mut buf = String::new();
            f.take(32 * 1024).read_to_string(&mut buf).ok().map(|_| buf)
        })
    };

    // Mobile: Flutter/Dart pubspec, or native iOS/Android scaffolds.
    if exists("pubspec.yaml") || exists("pubspec.yml") {
        out.insert("role_mobile".into());
    }
    if root.join("ios").is_dir() && root.join("android").is_dir() {
        out.insert("role_mobile".into());
    }

    // Python backend: alembic + FastAPI/Flask/Django/Starlette in deps.
    if exists("alembic.ini") {
        out.insert("role_api".into());
    }
    let py_deps = read_head("requirements.txt")
        .or_else(|| read_head("requirements-froze.txt"))
        .or_else(|| read_head("pyproject.toml"))
        .unwrap_or_default()
        .to_lowercase();
    if !py_deps.is_empty()
        && (py_deps.contains("fastapi")
            || py_deps.contains("flask")
            || py_deps.contains("django")
            || py_deps.contains("starlette")
            || py_deps.contains("uvicorn")
            || py_deps.contains("gunicorn"))
    {
        out.insert("role_api".into());
    }

    // Node: package.json — distinguish web (react/vue/next/svelte) from
    // api (express/fastify/nest/koa/hono).
    if let Some(pkg) = read_head("package.json") {
        let pkg_lc = pkg.to_lowercase();
        let has_web = pkg_lc.contains("\"react\"")
            || pkg_lc.contains("\"vue\"")
            || pkg_lc.contains("\"next\"")
            || pkg_lc.contains("\"svelte\"")
            || pkg_lc.contains("\"@angular/core\"")
            || pkg_lc.contains("\"vite\"");
        let has_api = pkg_lc.contains("\"express\"")
            || pkg_lc.contains("\"fastify\"")
            || pkg_lc.contains("\"@nestjs/core\"")
            || pkg_lc.contains("\"koa\"")
            || pkg_lc.contains("\"hono\"");
        let has_rn = pkg_lc.contains("\"react-native\"") || pkg_lc.contains("\"expo\"");
        if has_rn {
            out.insert("role_mobile".into());
        } else {
            if has_web {
                out.insert("role_ui".into());
            }
            if has_api {
                out.insert("role_api".into());
            }
        }
    }
    if exists("vite.config.ts")
        || exists("vite.config.js")
        || exists("next.config.ts")
        || exists("next.config.js")
        || exists("svelte.config.js")
        || exists("nuxt.config.ts")
    {
        out.insert("role_ui".into());
    }

    // Rust: Cargo.toml — axum/actix/rocket → api; clap-only bin → cli.
    if let Some(cargo) = read_head("Cargo.toml") {
        let lc = cargo.to_lowercase();
        if lc.contains("axum")
            || lc.contains("actix-web")
            || lc.contains("rocket")
            || lc.contains("warp =")
            || lc.contains("tower-http")
            || lc.contains("poem =")
        {
            out.insert("role_api".into());
        }
    }

    // Ops/infra: terraform, helm, k8s manifests at root.
    if exists("Chart.yaml")
        || exists("kustomization.yaml")
        || exists("kustomization.yml")
        || exists("docker-compose.yml")
        || exists("docker-compose.yaml")
        || root.join("terraform").is_dir()
        || root.join("helm").is_dir()
        || root.join("k8s").is_dir()
    {
        out.insert("role_ops".into());
    }
    // Any *.tf at root.
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|s| s.to_str()) {
                if ext == "tf" {
                    out.insert("role_ops".into());
                    break;
                }
            }
        }
    }

    out.into_iter().collect()
}

fn init_workspace(name: Option<String>, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("cannot read cwd")?;
    let manifest_path = cwd.join(MANIFEST_FILENAME);
    if manifest_path.exists() {
        anyhow::bail!(
            "manifest already exists at {} — edit directly or remove first",
            manifest_path.display()
        );
    }

    let ws_name = name
        .or_else(|| {
            cwd.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "workspace".to_string());

    // Discover `.axil/memory.axil` files anywhere up to 4 levels under
    // cwd (plus cwd itself). Walking deep is cheap compared to getting
    // it wrong — the 1-level-only version missed both root-level DBs
    // (the axildb repo itself) and workspace-member layouts like
    // `crates/<name>/.axil/` that sit two levels in. Skips common noise
    // (git, target, node_modules) and dedupes by canonical path.
    let mut members: std::collections::BTreeMap<String, Member> = std::collections::BTreeMap::new();
    let mut role_ui = false;
    let mut role_api = false;
    let mut role_ops = false;
    let mut role_mobile = false;
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let cwd_abs = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());

    let walker = walkdir::WalkDir::new(&cwd)
        .max_depth(4)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.iter().any(|d| name == *d)
        });
    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        if p.file_name().and_then(|s| s.to_str()) != Some("memory.axil") {
            continue;
        }
        if p.parent()
            .and_then(|q| q.file_name())
            .and_then(|s| s.to_str())
            != Some(".axil")
        {
            continue;
        }
        let db_abs = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        if !seen.insert(db_abs.clone()) {
            continue;
        }
        let root_abs = db_abs
            .parent()
            .and_then(|q| q.parent())
            .unwrap_or(&cwd_abs)
            .to_path_buf();
        let label = if root_abs == cwd_abs {
            "main".to_string()
        } else {
            root_abs
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("member")
                .to_string()
        };
        if members.contains_key(&label) {
            continue;
        }
        let label_lc = label.to_lowercase();
        let tokens: std::collections::HashSet<&str> = label_lc
            .split(|c: char| c == '-' || c == '_' || c == '.')
            .filter(|t| !t.is_empty())
            .collect();
        let mut role_set: std::collections::BTreeSet<String> =
            detect_roles_from_files(&root_abs).into_iter().collect();
        if ROLE_UI_TOKENS.iter().any(|t| tokens.contains(t)) {
            role_set.insert("role_ui".into());
        }
        if ROLE_API_TOKENS.iter().any(|t| tokens.contains(t)) {
            role_set.insert("role_api".into());
        }
        if ROLE_OPS_TOKENS.iter().any(|t| tokens.contains(t)) {
            role_set.insert("role_ops".into());
        }
        if ROLE_MOBILE_TOKENS.iter().any(|t| tokens.contains(t)) {
            role_set.insert("role_mobile".into());
        }
        for r in &role_set {
            match r.as_str() {
                "role_ui" => role_ui = true,
                "role_api" => role_api = true,
                "role_ops" => role_ops = true,
                "role_mobile" => role_mobile = true,
                _ => {}
            }
        }
        let roles: Vec<String> = role_set.into_iter().collect();
        members.insert(
            label,
            Member {
                id: new_member_id(),
                root: relative_to_base(&root_abs, &cwd_abs),
                path: relative_to_base(&db_abs, &cwd_abs),
                roles,
            },
        );
    }

    let mut roles: std::collections::BTreeMap<String, Role> = std::collections::BTreeMap::new();
    if role_ui {
        roles.insert("role_ui".to_string(), Role { label: "ui".into() });
    }
    if role_api {
        roles.insert(
            "role_api".to_string(),
            Role {
                label: "api".into(),
            },
        );
    }
    if role_ops {
        roles.insert(
            "role_ops".to_string(),
            Role {
                label: "ops".into(),
            },
        );
    }
    if role_mobile {
        roles.insert(
            "role_mobile".to_string(),
            Role {
                label: "mobile".into(),
            },
        );
    }

    let manifest = WorkspaceManifest {
        workspace: WorkspaceSection {
            id: new_workspace_id(),
            name: ws_name,
            version: "1".to_string(),
        },
        members,
        roles,
        federation: Federation::default(),
        manifest_path: manifest_path.clone(),
    };
    let body = manifest
        .to_toml_string()
        .context("serialize workspace manifest")?;
    std::fs::write(&manifest_path, body).context("write workspace manifest")?;

    let gitignore_path = cwd.join(".gitignore");
    let _ = crate::add_to_gitignore(&gitignore_path, ".axil-workspace.local.toml");

    // Register in the global registry.
    let reg_path = global_registry_path();
    if let Ok(mut reg) = GlobalRegistry::load(&reg_path) {
        reg.upsert(RegistryEntry {
            id: manifest.workspace.id.clone(),
            name: manifest.workspace.name.clone(),
            manifest_path: manifest_path.clone(),
            last_seen: Some(Utc::now()),
        });
        let _ = reg.save(&reg_path);
    }

    out.print(&json!({
        "manifest_path": manifest_path.display().to_string(),
        "workspace_id": manifest.workspace.id,
        "workspace_name": manifest.workspace.name,
        "members": manifest
            .members
            .iter()
            .map(|(label, m)| json!({"label": label, "id": m.id}))
            .collect::<Vec<_>>(),
    }));
    Ok(EXIT_OK)
}

fn status_workspace(db_opt: &Option<PathBuf>, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("cannot read cwd")?;
    let start = db_opt.clone().unwrap_or_else(|| cwd.clone());
    let manifest = match discover_manifest(&start)? {
        Some(m) => m,
        None => {
            out.print(&json!({
                "workspace": null,
                "message": "no .axil-workspace.toml found in any ancestor",
            }));
            return Ok(EXIT_OK);
        }
    };
    let status = unbound_status(&manifest, &cwd);
    let value = serde_json::to_value(&status).context("serialize status")?;
    out.print(&value);
    Ok(EXIT_OK)
}

fn list_workspaces(out: &Output) -> Result<i32> {
    let reg_path = global_registry_path();
    let mut registry = GlobalRegistry::load(&reg_path).unwrap_or_default();
    let pruned = registry.prune();
    if pruned > 0 {
        let _ = registry.save(&reg_path);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let local = discover_manifest(&cwd)?;
    let local_value = local.as_ref().map(|m| {
        json!({
            "workspace_id": m.workspace.id,
            "name": m.workspace.name,
            "manifest_path": m.manifest_path.display().to_string(),
        })
    });
    out.print(&json!({
        "registry_path": reg_path.display().to_string(),
        "cwd_workspace": local_value,
        "known": registry.workspaces,
    }));
    Ok(EXIT_OK)
}

fn add_member(path: PathBuf, as_label: Option<String>, out: &Output) -> Result<i32> {
    let cwd = std::env::current_dir().context("cannot read cwd")?;
    let cwd_abs = cwd.canonicalize().context("canonicalize cwd")?;
    let manifest_path = cwd.join(MANIFEST_FILENAME);
    if !manifest_path.exists() {
        anyhow::bail!(
            "no manifest at {} — run `axil workspace init` first",
            manifest_path.display()
        );
    }

    let path_abs = path
        .canonicalize()
        .with_context(|| format!("cannot resolve {}", path.display()))?;

    // Accept either a directory containing `.axil/memory.axil` or the file directly.
    let (member_root, member_db) = if path_abs.is_dir() {
        let inner = path_abs.join(".axil").join("memory.axil");
        if !inner.exists() {
            anyhow::bail!(
                "{} does not contain a .axil/memory.axil",
                path_abs.display()
            );
        }
        (path_abs.clone(), inner)
    } else if path_abs.ends_with("memory.axil") {
        let root = path_abs
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or(path_abs.clone());
        (root, path_abs.clone())
    } else {
        anyhow::bail!(
            "expected a .axil directory or memory.axil file: {}",
            path_abs.display()
        );
    };

    let label = as_label.unwrap_or_else(|| {
        member_root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("member")
            .to_string()
    });

    let mut manifest = WorkspaceManifest::load(&manifest_path)?;
    if manifest.members.contains_key(&label) {
        anyhow::bail!("member label '{label}' already exists in manifest");
    }

    let rel_root = relative_to_base(&member_root, &cwd_abs);
    let rel_db = relative_to_base(&member_db, &cwd_abs);

    manifest.members.insert(
        label.clone(),
        Member {
            id: new_member_id(),
            root: rel_root,
            path: rel_db,
            roles: Vec::new(),
        },
    );

    let body = manifest
        .to_toml_string()
        .context("serialize manifest after add")?;
    std::fs::write(&manifest_path, body).context("write manifest after add")?;

    out.print(&json!({
        "label": label,
        "manifest_path": manifest_path.display().to_string(),
    }));
    Ok(EXIT_OK)
}

/// Strip the Windows extended-length prefix so the resulting path is
/// still a valid absolute path on Windows and portable when possible.
/// `\\?\UNC\server\share\…` must become `\\server\share\…` — leaving
/// the bare `UNC\…` would be neither a drive nor a share. No-op on
/// non-Windows.
#[cfg(windows)]
fn strip_unc_prefix(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(unc_tail) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{unc_tail}"));
    }
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(stripped);
    }
    p.to_path_buf()
}

#[cfg(not(windows))]
fn strip_unc_prefix(p: &Path) -> PathBuf {
    p.to_path_buf()
}

/// Express `target` relative to `base` using forward slashes so the
/// result is portable across Windows and Unix checkouts. Both inputs
/// MUST already be absolute + canonicalized; this function is pure (no
/// filesystem I/O). Falls back to the absolute target (minus any
/// Windows extended-length prefix) when `target` is not a descendant
/// of `base`.
fn relative_to_base(target: &Path, base: &Path) -> PathBuf {
    match target.strip_prefix(base) {
        Ok(rel) => {
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if rel_str.is_empty() {
                PathBuf::from(".")
            } else {
                PathBuf::from(format!("./{rel_str}"))
            }
        }
        Err(_) => strip_unc_prefix(target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entity(canonical: &str) -> axil_core::Record {
        axil_core::Record::new("_entities", json!({ "canonical_id": canonical }))
    }

    #[test]
    fn scip_canonical_id_skips_local_and_provisional() {
        // Real SCIP symbols pass through.
        assert_eq!(
            scip_canonical_id(&entity("rust-analyzer cargo axil-core 0.6.0 db/Axil#")),
            Some("rust-analyzer cargo axil-core 0.6.0 db/Axil#".to_string()),
        );

        // SCIP `local N` is document-scoped — must never bridge across
        // sibling DBs (each member's `local 27` is a different symbol).
        assert_eq!(scip_canonical_id(&entity("local 0")), None);
        assert_eq!(scip_canonical_id(&entity("local 42")), None);

        // Phase 13 provisional rows are pre-canonical and must not bridge.
        assert_eq!(scip_canonical_id(&entity("provisional:abc123def")), None,);

        // Records with no canonical_id, or non-string canonical_id, are skipped.
        let no_canon = axil_core::Record::new("_entities", json!({ "name": "foo" }));
        assert_eq!(scip_canonical_id(&no_canon), None);
        let int_canon = axil_core::Record::new("_entities", json!({ "canonical_id": 7 }));
        assert_eq!(scip_canonical_id(&int_canon), None);
    }

    #[test]
    fn scip_canonical_id_does_not_match_local_substring() {
        // Symbols whose name happens to contain "local" but don't START
        // with the SCIP `local ` prefix must still bridge normally.
        let cid = "rust-analyzer cargo myapp 0.1.0 cache/local_store/impl#";
        assert_eq!(scip_canonical_id(&entity(cid)), Some(cid.to_string()),);
    }
}
