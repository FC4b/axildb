//! Boot contract — the stable agent wake-up schema.
//!
//! Axil's agent clients (Claude Code via CLI, Cursor via MCP, embedded
//! Rust users) all need "what does the agent need to know right now?"
//! served as a deterministic, token-budgeted struct. This module is the
//! single source of truth; CLI/MCP serializers wrap `BootContext` rather
//! than re-assembling their own shapes.
//!
//! ## Schema
//!
//! The returned struct carries a `schema_version` ("1" for now) and a
//! fixed, ordered `sections` list. Sections MAY be absent (e.g. an empty
//! DB has no `recent_decisions`) but never reordered.
//!
//! ```text
//! CurrentScope ─► Constraints ─► RecentDecisions ─► ActiveFailures
//!               ─► OpenThreads ─► Preferences ─► ConfidenceNotes
//! ```
//!
//! ## Token budget
//!
//! Callers pass `token_budget`. We estimate usage per section (4 chars ≈
//! 1 token, good-enough for a BPE-averaged value), accumulate into
//! `token_budget_used`, and drop sections in reverse priority order once
//! we exceed it. The top four sections (scope/constraints/decisions/
//! failures) are never dropped — those are load-bearing for planning.
//! Drops are reported in `dropped_sections` so the caller can show "we
//! omitted X to stay in budget".

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::db::Axil;
use crate::error::Result;

/// Stable schema version. Bumped when the `sections` layout changes in a
/// way that breaks downstream parsers.
pub const BOOT_SCHEMA_VERSION: &str = "1";

/// Default budget when the caller doesn't specify one. Picked to fit
/// comfortably in a 4k-window prompt with room for the user's turn.
pub const DEFAULT_TOKEN_BUDGET: usize = 2000;

/// Chars-to-tokens divisor. 4.0 is a widely-cited average for English BPE
/// tokenizers and is within ±15% across cl100k, tiktoken, and BGE's
/// XLM-R. We don't need precision — only enough accuracy to prevent
/// boot context from blowing through a generous budget.
const CHARS_PER_TOKEN: f64 = 4.0;

/// Per-table caps on how many rows to include in each section before
/// budget shaping. Prevents a chat-heavy DB from dumping 500 decisions
/// into boot.
const MAX_DECISIONS: usize = 10;
const MAX_FAILURES: usize = 10;
const MAX_THREADS: usize = 10;
const MAX_PREFERENCES: usize = 20;

/// Options passed by the caller. All fields are optional; `Default`
/// gives a sensible baseline.
#[derive(Debug, Clone, Default)]
pub struct BootOptions {
    /// Token budget for the entire boot context. `0` or `None` semantics
    /// use `DEFAULT_TOKEN_BUDGET`.
    pub token_budget: Option<usize>,
    /// Optional topic — if set, a topic-focused recall runs and its
    /// results feed the `RecentDecisions` section's head.
    pub topic: Option<String>,
    /// Scope filter passed through to recall.
    pub scope: Option<Vec<String>>,
}

/// A single section in the boot context.
///
/// Serde-tagged via `kind` so downstream JSON consumers can branch on
/// section type without peeking at the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BootSection {
    CurrentScope { content: Value },
    Constraints { content: Value },
    RecentDecisions { content: Vec<Value> },
    ActiveFailures { content: Vec<Value> },
    OpenThreads { content: Vec<Value> },
    Preferences { content: Vec<Value> },
    ConfidenceNotes { content: Value },
}

impl BootSection {
    /// Short stable name for tooling / diagnostics / drop tracking.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::CurrentScope { .. } => "current_scope",
            Self::Constraints { .. } => "constraints",
            Self::RecentDecisions { .. } => "recent_decisions",
            Self::ActiveFailures { .. } => "active_failures",
            Self::OpenThreads { .. } => "open_threads",
            Self::Preferences { .. } => "preferences",
            Self::ConfidenceNotes { .. } => "confidence_notes",
        }
    }

    /// Rank: lower number = higher priority, never dropped.
    /// Budget-driven drops walk from highest rank down.
    fn priority(&self) -> u8 {
        match self {
            Self::CurrentScope { .. } => 0,
            Self::Constraints { .. } => 1,
            Self::RecentDecisions { .. } => 2,
            Self::ActiveFailures { .. } => 3,
            Self::OpenThreads { .. } => 4,
            Self::Preferences { .. } => 5,
            Self::ConfidenceNotes { .. } => 6,
        }
    }
}

/// Returned from `Axil::boot()`. Deterministic order, stable schema,
/// token-budget aware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootContext {
    pub schema_version: &'static str,
    pub generated_at: DateTime<Utc>,
    pub token_budget: usize,
    pub token_budget_used: usize,
    pub sections: Vec<BootSection>,
    /// Kinds that were dropped to fit the budget. Empty when the full
    /// boot fit.
    pub dropped_sections: Vec<String>,
}

impl Axil {
    /// Assemble a boot context from current DB state.
    ///
    /// Returns a `BootContext` with sections in fixed order. When the
    /// estimated token usage exceeds `opts.token_budget`, lower-priority
    /// sections are dropped (ConfidenceNotes → Preferences → OpenThreads)
    /// until we fit. The top four sections (scope, constraints, recent
    /// decisions, active failures) are never dropped.
    pub fn boot(&self, opts: BootOptions) -> Result<BootContext> {
        let budget = opts.token_budget.unwrap_or(DEFAULT_TOKEN_BUDGET);

        // ── Assemble sections in fixed priority order. ───────────
        let mut sections: Vec<BootSection> = Vec::new();

        // 0. current scope: most recent session + what files touched.
        let scope_content = self.build_current_scope(&opts);
        sections.push(BootSection::CurrentScope {
            content: scope_content,
        });

        // 1. constraints: user-set rules + pinned / high-importance facts.
        let constraints_content = self.build_constraints();
        sections.push(BootSection::Constraints {
            content: constraints_content,
        });

        // 2. recent decisions: focused recall when --topic given, otherwise top-N by importance.
        let decisions = self.boot_records("decisions", MAX_DECISIONS, &opts);
        sections.push(BootSection::RecentDecisions { content: decisions });

        // 3. active failures: unresolved errors.
        let failures = self.boot_records("errors", MAX_FAILURES, &opts);
        sections.push(BootSection::ActiveFailures { content: failures });

        // 4. open threads: in-flight context items.
        let threads = self.boot_records("context", MAX_THREADS, &opts);
        sections.push(BootSection::OpenThreads { content: threads });

        // 5. preferences: user-set key/value pairs.
        let prefs = self.list_preferences_truncated(MAX_PREFERENCES);
        sections.push(BootSection::Preferences { content: prefs });

        // 6. confidence notes: how fresh/stale the DB is.
        let confidence = self.build_confidence_notes();
        sections.push(BootSection::ConfidenceNotes {
            content: confidence,
        });

        // ── Budget discipline: drop low-priority sections until we
        // fit. Never drop priority < 4 (scope/constraints/decisions/
        // failures). ─────────────────────────────────────────────
        let (sections, dropped_sections, used) = apply_budget(sections, budget);

        Ok(BootContext {
            schema_version: BOOT_SCHEMA_VERSION,
            generated_at: Utc::now(),
            token_budget: budget,
            token_budget_used: used,
            sections,
            dropped_sections,
        })
    }

    fn build_current_scope(&self, opts: &BootOptions) -> Value {
        let latest_session = self
            .list("sessions")
            .unwrap_or_default()
            .into_iter()
            .filter(|r| record_in_scope(r, opts.scope.as_deref()))
            .max_by_key(|r| r.created_at);
        let session_id = latest_session
            .as_ref()
            .and_then(|r| r.data.get("session_id").cloned())
            .unwrap_or(Value::Null);
        let mut out = serde_json::Map::new();
        out.insert("latest_session_id".to_string(), session_id);
        out.insert(
            "generated_at".to_string(),
            Value::String(Utc::now().to_rfc3339()),
        );
        if let Some(scope) = opts.scope.as_deref() {
            out.insert("scope_filter".to_string(), json!(scope));
        }
        if let Some(topic) = opts.topic.as_deref() {
            out.insert("topic".to_string(), Value::String(topic.to_string()));
        }

        // Surface registered Extensions' `boot_block` contributions
        // in the top-priority, never-dropped section. Backward-
        // compatible: consumers that don't know about `extension_blocks`
        // just ignore the new key.
        //
        // Shape is `Array<{id, text}>`, not `Object` — a serde_json::Map
        // is BTreeMap-backed by default and would silently sort blocks
        // alphabetically, breaking the registration-order contract on
        // `collect_extension_blocks`.
        let blocks = collect_extension_blocks(self);
        if !blocks.is_empty() {
            let blocks_arr: Vec<Value> = blocks
                .into_iter()
                .map(|(id, text)| json!({ "id": id, "text": text }))
                .collect();
            out.insert("extension_blocks".to_string(), Value::Array(blocks_arr));
        }

        // Advise when this is a code repo indexed without a precise graph.
        if let Some(hint) = self.code_graph_hint() {
            out.insert("code_graph_hint".to_string(), json!(hint));
        }
        Value::Object(out)
    }

    /// Advisory for a code repo that has structural proxies but no precise,
    /// SCIP-grounded call graph. A plain `axil index` builds `_idx_code_proxies`
    /// but no SCIP edges; only SCIP ingest produces precise
    /// `calls`/`references`/`implements`/`type_of` edges.
    ///
    /// The presence signal is the `_scip_aliases` table ([`SCIP_ALIAS_TABLE`]),
    /// which is written *exclusively* by SCIP ingest (`register_entity_alias`).
    /// We deliberately do NOT key off `_entities`: that table is also populated
    /// by algorithmic entity extraction, auto-linking, inference, and beliefs
    /// (`entity.rs`, `worker.rs`, `inference.rs`, …), so a repo that auto-linked
    /// without SCIP would carry `_entities` rows and wrongly suppress this
    /// advisory in exactly the structural-only case it exists to catch.
    ///
    /// Returns `None` for non-code repos (no proxies) and for repos that already
    /// have a precise graph.
    pub fn code_graph_hint(&self) -> Option<String> {
        if self.count("_idx_code_proxies").unwrap_or(0) == 0 {
            return None; // not a code repo, or not indexed yet
        }
        if self.count(crate::SCIP_ALIAS_TABLE).unwrap_or(0) > 0 {
            return None; // precise (SCIP-grounded) graph already ingested
        }
        Some(
            "No precise call graph for this code repo — only structural proxies \
             are indexed. Run `axil scip refresh` (needs rust-analyzer / scip-* on \
             PATH) to add precise calls/references/implements edges."
                .to_string(),
        )
    }

    /// Pick the top-N records for a section, honoring `topic` (semantic
    /// recall scoped to `table`) and `scope` (filter by `_scope` field).
    /// Falls back to importance ranking when no topic is set or recall
    /// returns nothing usable.
    fn boot_records(&self, table: &str, n: usize, opts: &BootOptions) -> Vec<Value> {
        if let Some(topic) = opts.topic.as_deref() {
            let cfg = crate::scoring::RecallConfig {
                scope_filter: opts.scope.clone().unwrap_or_default(),
                ..Default::default()
            };
            // Over-fetch then filter by table to mimic top_n_by_importance's
            // table-scoped ranking under a topic-driven query.
            let fetch = n.saturating_mul(8).max(40);
            if let Ok(results) = self.recall(topic, fetch, Some(cfg)) {
                let filtered: Vec<Value> = results
                    .into_iter()
                    .filter(|r| r.record.table == table)
                    .take(n)
                    .map(|r| {
                        json!({
                            "id": r.record.id.to_string(),
                            "data": r.record.data,
                            "created_at": r.record.created_at.to_rfc3339(),
                            "score": r.score,
                        })
                    })
                    .collect();
                if !filtered.is_empty() {
                    return filtered;
                }
            }
        }
        // No topic or recall miss: fall back to importance ranking, still
        // honoring scope.
        self.top_n_by_importance_scoped(table, n, opts.scope.as_deref())
    }

    fn top_n_by_importance_scoped(
        &self,
        table: &str,
        n: usize,
        scope: Option<&[String]>,
    ) -> Vec<Value> {
        let mut records: Vec<_> = self
            .list(table)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| record_in_scope(r, scope))
            .collect();
        records.sort_by(|a, b| {
            let ia = a
                .data
                .get("_effective_importance")
                .or_else(|| a.data.get("_importance"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let ib = b
                .data
                .get("_effective_importance")
                .or_else(|| b.data.get("_importance"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
        });
        records.truncate(n);
        records
            .into_iter()
            .map(|r| {
                json!({
                    "id": r.id.to_string(),
                    "data": r.data,
                    "created_at": r.created_at.to_rfc3339(),
                })
            })
            .collect()
    }

    fn build_constraints(&self) -> Value {
        // Pinned + importance=1.0 records in the `rules` table, if any.
        let rules = self.list("rules").unwrap_or_default();
        let items: Vec<Value> = rules
            .into_iter()
            .filter(|r| {
                crate::importance::is_pinned(&r.data)
                    || crate::importance::get_importance(&r.data) >= 0.9
            })
            .map(|r| {
                json!({
                    "id": r.id.to_string(),
                    "rule": r.data.get("rule").cloned().unwrap_or_default(),
                })
            })
            .collect();
        json!({ "rules": items })
    }

    fn list_preferences_truncated(&self, n: usize) -> Vec<Value> {
        let mut prefs = self.list("preferences").unwrap_or_default();
        prefs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        prefs.truncate(n);
        prefs
            .into_iter()
            .map(|r| {
                json!({
                    "key": r.data.get("key").cloned().unwrap_or_default(),
                    "value": r.data.get("value").cloned().unwrap_or_default(),
                })
            })
            .collect()
    }

    fn build_confidence_notes(&self) -> Value {
        // Surface how "fresh" the DB looks: total records, newest record
        // age in days. Agents use this to decide if the memory is still
        // relevant or if the DB has been sitting stale.
        let tables = self.tables().unwrap_or_default();
        let mut total_records = 0usize;
        let mut newest: Option<DateTime<Utc>> = None;
        for t in &tables {
            if t.starts_with('_') {
                continue;
            }
            if let Ok(records) = self.list(t) {
                total_records += records.len();
                if let Some(latest) = records.iter().map(|r| r.created_at).max() {
                    newest = Some(newest.map_or(latest, |cur| cur.max(latest)));
                }
            }
        }
        let newest_age_days = newest
            .map(|t| (Utc::now() - t).num_days())
            .unwrap_or(i64::MAX);
        json!({
            "total_records": total_records,
            "newest_age_days": newest_age_days,
        })
    }
}

/// Collect non-empty `boot_block` contributions from every registered
/// Extension into an ordered (id, text) list. Registration order is
/// preserved so consumers can render deterministically. Extensions
/// whose `boot_block` returns `None` are skipped.
///
/// Exposed as `pub` so the CLI Adapter's flat-JSON boot path can share
/// the same collection without re-implementing the loop.
pub fn collect_extension_blocks(db: &Axil) -> Vec<(String, String)> {
    db.extensions()
        .iter()
        .filter_map(|ext| {
            ext.boot_block(db).map(|text| (ext.id().to_string(), text))
        })
        .collect()
}

/// Estimate token cost of serialized JSON content. Rough but stable:
/// total_chars / CHARS_PER_TOKEN.
/// Returns true when `record` matches `scope` (or `scope` is None).
/// A record is in scope when its `_scope` field equals one of the
/// caller-supplied scopes, or when neither side declares a scope.
fn record_in_scope(record: &crate::record::Record, scope: Option<&[String]>) -> bool {
    let Some(scope) = scope.filter(|s| !s.is_empty()) else {
        return true;
    };
    let record_scope = record
        .data
        .get("_scope")
        .and_then(|v| v.as_str())
        .unwrap_or("project");
    scope.iter().any(|s| s == record_scope)
}

fn estimate_tokens(v: &Value) -> usize {
    let s = v.to_string();
    (s.len() as f64 / CHARS_PER_TOKEN).ceil() as usize
}

/// Approximate tokens used by a section's serialized form.
fn section_cost(s: &BootSection) -> usize {
    match s {
        BootSection::CurrentScope { content }
        | BootSection::Constraints { content }
        | BootSection::ConfidenceNotes { content } => estimate_tokens(content),
        BootSection::RecentDecisions { content }
        | BootSection::ActiveFailures { content }
        | BootSection::OpenThreads { content }
        | BootSection::Preferences { content } => content.iter().map(estimate_tokens).sum(),
    }
}

/// Walk sections in decreasing priority (highest priority number first
/// = lowest-importance section first) and drop until the total fits
/// under `budget`. Never drop sections with `priority() < 4`.
fn apply_budget(
    sections: Vec<BootSection>,
    budget: usize,
) -> (Vec<BootSection>, Vec<String>, usize) {
    let mut costs: Vec<usize> = sections.iter().map(section_cost).collect();
    let mut kept: Vec<bool> = vec![true; sections.len()];

    // We drop in priority-descending order: highest priority number
    // (lowest importance) goes first.
    let mut order: Vec<usize> = (0..sections.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sections[i].priority()));

    let total = |kept: &[bool], costs: &[usize]| -> usize {
        kept.iter()
            .zip(costs.iter())
            .filter(|(k, _)| **k)
            .map(|(_, c)| *c)
            .sum()
    };

    let mut dropped = Vec::new();
    for idx in order {
        if total(&kept, &costs) <= budget {
            break;
        }
        // Never drop load-bearing sections.
        if sections[idx].priority() < 4 {
            continue;
        }
        kept[idx] = false;
        costs[idx] = 0;
        dropped.push(sections[idx].kind_str().to_string());
    }

    let used = total(&kept, &costs);
    let out: Vec<BootSection> = sections
        .into_iter()
        .zip(kept.iter())
        .filter_map(|(s, k)| if *k { Some(s) } else { None })
        .collect();
    (out, dropped, used)
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
    fn boot_returns_schema_v1_with_fixed_order() {
        let (db, _dir) = temp_db();
        let ctx = db.boot(BootOptions::default()).unwrap();
        assert_eq!(ctx.schema_version, "1");
        let kinds: Vec<&str> = ctx.sections.iter().map(BootSection::kind_str).collect();
        assert_eq!(
            kinds,
            vec![
                "current_scope",
                "constraints",
                "recent_decisions",
                "active_failures",
                "open_threads",
                "preferences",
                "confidence_notes",
            ]
        );
    }

    #[test]
    fn code_graph_hint_fires_only_for_code_repo_without_precise_graph() {
        let (db, _dir) = temp_db();
        // No code proxies → not a code repo → no hint.
        assert!(db.code_graph_hint().is_none());

        // Code proxies present, no SCIP aliases → structural-only → hint fires.
        db.insert("_idx_code_proxies", json!({"path": "src/x.rs", "kind": "file"}))
            .unwrap();
        let hint = db.code_graph_hint();
        assert!(hint.is_some(), "code repo without precise graph should warn");
        assert!(hint.unwrap().contains("axil scip refresh"));

        // `_entities` rows alone (e.g. from auto-linking / entity extraction,
        // not SCIP) must NOT suppress the hint — the no-precise-graph case.
        db.insert("_entities", json!({"canonical_id": "natural-language-entity"}))
            .unwrap();
        assert!(
            db.code_graph_hint().is_some(),
            "non-SCIP _entities rows must not suppress the advisory"
        );

        // SCIP alias rows present → precise graph ingested → hint suppressed.
        db.insert(crate::SCIP_ALIAS_TABLE, json!({"alias": "y", "canonical_id": "scip-rust ... y()."}))
            .unwrap();
        assert!(db.code_graph_hint().is_none());
    }

    #[test]
    fn empty_db_still_produces_full_section_list() {
        let (db, _dir) = temp_db();
        let ctx = db.boot(BootOptions::default()).unwrap();
        assert_eq!(ctx.sections.len(), 7);
        assert!(ctx.dropped_sections.is_empty());
        assert!(
            ctx.token_budget_used > 0,
            "empty boot still costs some tokens"
        );
    }

    #[test]
    fn tiny_budget_drops_low_priority_sections_first() {
        let (db, _dir) = temp_db();
        // Seed enough prefs / threads that those sections have real cost.
        for i in 0..20 {
            db.insert(
                "preferences",
                json!({ "key": format!("k{i}"), "value": format!("v{i}") }),
            )
            .unwrap();
        }
        for i in 0..20 {
            db.insert(
                "context",
                json!({ "summary": format!("open thread #{i} with enough words to cost tokens") }),
            )
            .unwrap();
        }

        // Budget tiny enough to force drops.
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(50),
                ..Default::default()
            })
            .unwrap();

        // Dropped sections list ordered as we dropped them (lowest
        // priority first).
        let first_drop = ctx.dropped_sections.first().cloned();
        assert_eq!(
            first_drop.as_deref(),
            Some("confidence_notes"),
            "confidence_notes (lowest priority) must be dropped first; dropped={:?}",
            ctx.dropped_sections
        );

        // Top-priority sections must always survive.
        let kept: std::collections::HashSet<&str> =
            ctx.sections.iter().map(BootSection::kind_str).collect();
        for required in [
            "current_scope",
            "constraints",
            "recent_decisions",
            "active_failures",
        ] {
            assert!(
                kept.contains(required),
                "load-bearing section {required} must never be dropped"
            );
        }
    }

    #[test]
    fn budget_discipline_reports_usage() {
        let (db, _dir) = temp_db();
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(500),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(ctx.token_budget, 500);
        assert!(
            ctx.token_budget_used <= ctx.token_budget + 50,
            // Small slop: a single section over-budget by a hair won't
            // trigger a drop since we only drop when the *total* exceeds
            // the cap. Keep the test robust to that.
            "used {} should be ≤ budget {} (plus small slop)",
            ctx.token_budget_used,
            ctx.token_budget
        );
    }

    #[test]
    fn load_bearing_sections_never_dropped_even_at_zero_budget() {
        let (db, _dir) = temp_db();
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(1),
                ..Default::default()
            })
            .unwrap();
        let kinds: Vec<&str> = ctx.sections.iter().map(BootSection::kind_str).collect();
        for required in [
            "current_scope",
            "constraints",
            "recent_decisions",
            "active_failures",
        ] {
            assert!(kinds.contains(&required), "missing {required}");
        }
    }

    // ---- follow-up — Extension boot_block integration ----

    /// Stub Extension that always emits a known boot_block — used to
    /// pin the wiring from `Extension::boot_block` → `CurrentScope`'s
    /// `extension_blocks` sub-key.
    struct StubBootBlockExt;
    impl crate::Extension for StubBootBlockExt {
        fn id(&self) -> &str {
            "stub-boot-block"
        }
        fn boot_block(&self, _db: &Axil) -> Option<String> {
            Some("## Stub Block\n- hello from a stub extension\n".into())
        }
    }

    /// Stub Extension that returns None — used to assert silent
    /// Extensions don't leak empty entries into `extension_blocks`.
    struct SilentExt;
    impl crate::Extension for SilentExt {
        fn id(&self) -> &str {
            "silent-ext"
        }
    }

    #[test]
    fn collect_extension_blocks_skips_silent_extensions() {
        let dir = tempdir().unwrap();
        let db = Axil::open(dir.path().join("test.axil"))
            .with_extension(SilentExt)
            .with_extension(StubBootBlockExt)
            .build()
            .unwrap();
        let blocks = collect_extension_blocks(&db);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "stub-boot-block");
        assert!(blocks[0].1.starts_with("## Stub Block"));
    }

    #[test]
    fn current_scope_carries_extension_blocks() {
        let dir = tempdir().unwrap();
        let db = Axil::open(dir.path().join("test.axil"))
            .with_extension(StubBootBlockExt)
            .build()
            .unwrap();
        let ctx = db.boot(BootOptions::default()).unwrap();
        let scope = ctx
            .sections
            .iter()
            .find_map(|s| match s {
                BootSection::CurrentScope { content } => Some(content),
                _ => None,
            })
            .expect("CurrentScope must be present");
        let blocks = scope
            .get("extension_blocks")
            .and_then(|v| v.as_array())
            .expect("extension_blocks should be a non-empty Array when an Extension contributes one");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["id"], "stub-boot-block");
        let text = blocks[0]["text"].as_str().unwrap();
        assert!(text.contains("hello from a stub extension"));
    }

    /// Two stub Extensions registered in a deliberate order — assert
    /// the rendered `extension_blocks` array preserves registration
    /// order, not alphabetical order. This is the regression gate for
    /// the `serde_json::Map` ordering bug Codex caught.
    #[test]
    fn extension_blocks_preserve_registration_order() {
        // "z-…" deliberately sorts after "a-…" alphabetically, so if
        // any code path round-trips through a BTreeMap-backed Map,
        // this test fails.
        struct ZebraExt;
        impl crate::Extension for ZebraExt {
            fn id(&self) -> &str {
                "z-zebra"
            }
            fn boot_block(&self, _db: &Axil) -> Option<String> {
                Some("z text".into())
            }
        }
        struct AlphaExt;
        impl crate::Extension for AlphaExt {
            fn id(&self) -> &str {
                "a-alpha"
            }
            fn boot_block(&self, _db: &Axil) -> Option<String> {
                Some("a text".into())
            }
        }
        let dir = tempdir().unwrap();
        // Registration order: zebra first, alpha second. Alphabetical
        // order would flip them.
        let db = Axil::open(dir.path().join("test.axil"))
            .with_extension(ZebraExt)
            .with_extension(AlphaExt)
            .build()
            .unwrap();
        let ctx = db.boot(BootOptions::default()).unwrap();
        let scope = ctx
            .sections
            .iter()
            .find_map(|s| match s {
                BootSection::CurrentScope { content } => Some(content),
                _ => None,
            })
            .expect("CurrentScope must be present");
        let blocks = scope["extension_blocks"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["id"], "z-zebra", "registration order must be preserved");
        assert_eq!(blocks[1]["id"], "a-alpha", "registration order must be preserved");
    }

    #[test]
    fn current_scope_omits_extension_blocks_when_none_contribute() {
        let (db, _dir) = temp_db();
        let ctx = db.boot(BootOptions::default()).unwrap();
        let scope = ctx
            .sections
            .iter()
            .find_map(|s| match s {
                BootSection::CurrentScope { content } => Some(content),
                _ => None,
            })
            .expect("CurrentScope must be present");
        assert!(
            scope.get("extension_blocks").is_none(),
            "extension_blocks should be absent (not just empty) when no Extension contributed"
        );
    }

    /// Regression gate for review finding #6: each call to
    /// `db.boot()` must invoke `Extension::boot_block` exactly once per
    /// registered Extension.
    ///
    /// Scope is intentionally narrow — this test proves the in-process
    /// `db.boot()` pipeline (used by `axil boot --schema v1` and the
    /// MCP `boot` tool) doesn't double-fire. It does *not* cover:
    ///   - The CLI's legacy flat-JSON `axil boot` path, which calls
    ///     `collect_extension_blocks` directly outside `db.boot()`.
    ///     That site is a single explicit call (no implicit pipeline
    ///     replay risk), so the regression surface is narrower.
    ///   - Concurrent `db.boot()` calls. `&self` makes them safe; two
    ///     concurrent boots fire `boot_block` twice per Extension
    ///     (once each), which is the correct sequential count summed.
    #[test]
    fn boot_fires_extension_boot_block_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct CountingExt {
            calls: Arc<AtomicUsize>,
        }
        impl crate::Extension for CountingExt {
            fn id(&self) -> &str {
                "counting-ext"
            }
            fn boot_block(&self, _db: &Axil) -> Option<String> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Some("block".into())
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let dir = tempdir().unwrap();
        let db = Axil::open(dir.path().join("test.axil"))
            .with_extension(CountingExt {
                calls: calls.clone(),
            })
            .build()
            .unwrap();

        let _ = db.boot(BootOptions::default()).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "db.boot() must call Extension::boot_block exactly once per registered Extension"
        );

        // A second boot() also fires exactly once — proves the count is
        // per-invocation, not cumulative-across-process, and that no
        // hidden caller in the pipeline replays the collection.
        let _ = db.boot(BootOptions::default()).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "second db.boot() should bring the total to exactly 2"
        );
    }
}
