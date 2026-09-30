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
//! The returned struct carries a `schema_version` ([`BOOT_SCHEMA_VERSION`])
//! and a fixed, ordered `sections` list. Sections MAY be absent (a
//! lower-priority section the budget had no room for) but never reordered.
//!
//! ```text
//! CurrentScope ─► Constraints ─► RecentDecisions ─► ActiveFailures
//!               ─► OpenThreads ─► Preferences ─► ConfidenceNotes
//! ```
//!
//! ## Rows
//!
//! Every row (rules, decisions, failures, threads, preferences) is one
//! string: `id · age · status · summary` (see [`boot_row`]). The summary is
//! clipped, so a row costs a few dozen tokens instead of a whole record;
//! `axil get <id>` (MCP `get`) expands it. Schema 1 carried whole records
//! (`{id, data, created_at}`) instead; that shape change is why the version
//! is 2.
//!
//! ## Token budget
//!
//! Callers pass `token_budget`, in tokens estimated by
//! [`crate::token::DEFAULT_TOKEN_ESTIMATOR`] (`ceil(bytes / 4)`, a heuristic)
//! over the serialized context. [`fill_by_priority`] picks the rows in
//! strict priority order — the Resume Here block (scope), constraints,
//! active failures, recent decisions, then open threads, preferences and
//! confidence notes — row by row. The first row that does not fit is
//! clipped to the space left and ends the fill, so no row is ever shown
//! while a higher-priority one is left out.
//! The four load-bearing sections are never dropped, only trimmed; a
//! lower-priority section left without a row is dropped and named in
//! `dropped_sections`. Cut rows are counted in `omitted_items` and, per
//! section, in `omitted_by_section`. Every candidate is checked against the
//! real serialized context, so `token_budget_used` stays within the budget
//! unless the envelope and empty section skeletons alone exceed it.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::db::Axil;
use crate::error::Result;
use crate::record::Record;
use crate::token::TokenEstimator;

/// Stable schema version. Bumped when the `sections` layout changes in a
/// way that breaks downstream parsers: version 2 made every row a one-line
/// string where version 1 carried the whole record.
pub const BOOT_SCHEMA_VERSION: &str = "2";

/// Default boot budget, in tokens, when the caller doesn't pass one — for
/// this schema and for the CLI's `axil boot` formats alike.
///
/// Tokens here are always an estimate: `ceil(bytes / 4)` of the rendered
/// output ([`crate::token::CHARS_PER_TOKEN`]), not a tokenizer count. Boot is
/// injected at the start of every session, so its default is kept small; a
/// caller that wants more passes a larger budget.
pub const DEFAULT_TOKEN_BUDGET: usize = 1000;

/// Separator between the fields of a boot row.
pub const BOOT_ROW_SEP: &str = " · ";

/// Longest summary a boot row carries, in chars. The rest of the record is
/// one `axil get <id>` away.
pub const BOOT_ROW_SUMMARY_CHARS: usize = 160;

/// Longest rule text a boot row carries, in chars. Rules are always-apply
/// constraints, so they get more room than other rows before clipping.
pub const BOOT_RULE_CHARS: usize = 400;

/// Fewest summary chars a row clipped to the budget keeps. Shorter than
/// this a summary no longer says what the row is about, so the row is left
/// out instead.
pub const BOOT_CLIP_MIN_CHARS: usize = 24;

/// Per-table caps on how many rows to include in each section before
/// budget shaping. Prevents a chat-heavy DB from dumping 500 decisions
/// into boot.
const MAX_DECISIONS: usize = 10;
/// Open errors shown by boot, in every format. They fill before decisions,
/// so this cap decides how much room decisions get: at the SessionStart
/// hook's 800-token budget, five open errors left space for a single
/// clipped decision on this repo's own memory; three leave room for several.
pub const MAX_OPEN_ERRORS: usize = 3;
const MAX_THREADS: usize = 10;
const MAX_PREFERENCES: usize = 20;

/// Options passed by the caller. All fields are optional; `Default`
/// gives a sensible baseline.
#[derive(Debug, Clone, Default)]
pub struct BootOptions {
    /// Token budget for the entire boot context. `0` or `None` use
    /// [`DEFAULT_TOKEN_BUDGET`].
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

    /// Fill rank: lower number = filled first. This is the budget order,
    /// not the display order: open failures fill before decisions because
    /// an unresolved error changes what the agent should do next.
    fn priority(&self) -> u8 {
        match self {
            Self::CurrentScope { .. } => 0,
            Self::Constraints { .. } => 1,
            Self::ActiveFailures { .. } => 2,
            Self::RecentDecisions { .. } => 3,
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
    /// Kinds that were dropped to fit the budget, lowest priority first.
    /// Empty when every section kept at least one row.
    pub dropped_sections: Vec<String>,
    /// Rows, and lines of extension blocks, cut to fit the budget. Omitted
    /// when nothing was cut.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted_items: usize,
    /// `omitted_items` per section kind (an extension block by its id), so
    /// a reader can tell which sections lost rows. Omitted when nothing was
    /// cut.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub omitted_by_section: BTreeMap<String, usize>,
    /// Engines found on disk that this handle could not attach (see
    /// [`Axil::degraded_engines`]). Omitted when every Engine is attached.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<crate::diagnostics::DegradedEngine>,
}

impl Axil {
    /// Assemble a boot context from current DB state.
    ///
    /// Returns a `BootContext` with sections in fixed order and rows as
    /// one-liners. Rows fill in strict priority order within
    /// `opts.token_budget` (see [`fill_by_priority`]): scope, constraints,
    /// active failures, recent decisions, open threads, preferences,
    /// confidence notes. The first row that does not fit is clipped and
    /// ends the fill; a lower-priority section left without a row is
    /// dropped. The top four sections are never dropped.
    pub fn boot(&self, opts: BootOptions) -> Result<BootContext> {
        let budget = opts
            .token_budget
            .filter(|&b| b > 0)
            .unwrap_or(DEFAULT_TOKEN_BUDGET);
        let now = Utc::now();

        // Resolve the decay config once for the whole boot. `boot_records`
        // runs for several tables and each used to re-read + re-parse
        // `axil.toml` from disk; a single load here yields identical per-table
        // half-lives (a missing/unreadable config maps to an empty
        // `DecayConfig`, whose `half_life_for` returns the default for every
        // table — the same fallback the old per-call path took).
        let decay = std::env::current_dir()
            .ok()
            .and_then(|cwd| crate::config::load_config_from(&cwd).ok())
            .map(|c| c.decay)
            .unwrap_or_default();

        // ── Assemble sections in fixed priority order. ───────────
        let mut sections: Vec<BootSection> = Vec::new();

        // 0. current scope: most recent session + what files touched.
        let scope_content = self.build_current_scope(&opts);
        sections.push(BootSection::CurrentScope {
            content: scope_content,
        });

        // 1. constraints: user-set rules + pinned / high-importance facts.
        let constraints_content = self.build_constraints(now);
        sections.push(BootSection::Constraints {
            content: constraints_content,
        });

        // 2. recent decisions: focused recall when --topic given, otherwise top-N by importance.
        let decisions = self.boot_records("decisions", MAX_DECISIONS, &opts, &decay, now, |_| true);
        sections.push(BootSection::RecentDecisions { content: decisions });

        // 3. active failures: unresolved errors only. Resolved errors stay
        // recallable as lessons, but they are not failures to act on.
        let failures = self.boot_records("errors", MAX_OPEN_ERRORS, &opts, &decay, now, |r| {
            is_open_error(&r.data)
        });
        sections.push(BootSection::ActiveFailures { content: failures });

        // 4. open threads: in-flight context items.
        let threads = self.boot_records("context", MAX_THREADS, &opts, &decay, now, |_| true);
        sections.push(BootSection::OpenThreads { content: threads });

        // 5. preferences: user-set key/value pairs.
        let prefs = self.list_preferences_truncated(MAX_PREFERENCES, now);
        sections.push(BootSection::Preferences { content: prefs });

        // 6. confidence notes: how fresh/stale the DB is.
        let confidence = self.build_confidence_notes();
        sections.push(BootSection::ConfidenceNotes {
            content: confidence,
        });

        // ── Budget discipline: fill rows by priority, cutting rows (and
        // dropping only droppable sections) once the budget is spent. ──
        let mut ctx = BootContext {
            schema_version: BOOT_SCHEMA_VERSION,
            generated_at: now,
            token_budget: budget,
            token_budget_used: 0,
            sections: Vec::new(),
            dropped_sections: Vec::new(),
            omitted_items: 0,
            omitted_by_section: BTreeMap::new(),
            degraded: self.degraded_engines().to_vec(),
        };
        fit_to_budget(&mut ctx, sections, &crate::token::DEFAULT_TOKEN_ESTIMATOR);
        Ok(ctx)
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
    /// recall scoped to `table`) and `scope` (filter by `_scope` field),
    /// as one-line rows. `keep` narrows the candidates (e.g. to open
    /// errors) before the top N are taken. Falls back to importance
    /// ranking when no topic is set or recall returns nothing usable.
    fn boot_records(
        &self,
        table: &str,
        n: usize,
        opts: &BootOptions,
        decay: &crate::config::DecayConfig,
        now: DateTime<Utc>,
        keep: impl Fn(&Record) -> bool,
    ) -> Vec<Value> {
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
                    .filter(|r| r.record.table == table && keep(&r.record))
                    .take(n)
                    .map(|r| Value::String(record_row(&r.record, now)))
                    .collect();
                if !filtered.is_empty() {
                    return filtered;
                }
            }
        }
        // No topic or recall miss: fall back to importance ranking, still
        // honoring scope.
        self.top_n_by_importance_scoped(table, n, opts.scope.as_deref(), decay, now, keep)
    }

    fn top_n_by_importance_scoped(
        &self,
        table: &str,
        n: usize,
        scope: Option<&[String]>,
        decay: &crate::config::DecayConfig,
        now: DateTime<Utc>,
        keep: impl Fn(&Record) -> bool,
    ) -> Vec<Value> {
        let mut records: Vec<_> = self
            .list(table)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| record_in_scope(r, scope) && keep(r))
            .collect();
        // The decay config is resolved once per boot and threaded in, so this
        // hot per-table loop doesn't re-read `axil.toml` from disk.
        sort_by_effective_importance(&mut records, decay.half_life_for(table), now);
        records.truncate(n);
        records
            .iter()
            .map(|r| Value::String(record_row(r, now)))
            .collect()
    }

    fn build_constraints(&self, now: DateTime<Utc>) -> Value {
        // Pinned + importance=1.0 records in the `rules` table, if any.
        let rules = self.list("rules").unwrap_or_default();
        let items: Vec<Value> = rules
            .iter()
            .filter(|r| {
                crate::importance::is_pinned(&r.data)
                    || crate::importance::get_importance(&r.data) >= 0.9
            })
            .map(|r| Value::String(record_row(r, now)))
            .collect();
        json!({ "rules": items })
    }

    fn list_preferences_truncated(&self, n: usize, now: DateTime<Utc>) -> Vec<Value> {
        let mut prefs = self.list("preferences").unwrap_or_default();
        prefs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        prefs.truncate(n);
        prefs
            .iter()
            .map(|r| Value::String(preference_row(r, now)))
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

/// Collect non-empty boot blocks for the never-dropped scope section, in a
/// fixed order: every registered Extension's `boot_block` contribution first
/// (registration order preserved so consumers render deterministically;
/// `None`-returning Extensions skipped), then any core-synthesized advisories.
///
/// Routing core advisories (like the [`Axil::code_graph_hint`] SCIP nudge)
/// through here — rather than a bespoke top-level key + render branch per
/// adapter — keeps them a single shape across the CLI, MCP, and embedded boot
/// surfaces, rendered by the one generic block loop.
///
/// Exposed as `pub` so the CLI Adapter's flat-JSON boot path can share
/// the same collection without re-implementing the loop.
pub fn collect_extension_blocks(db: &Axil) -> Vec<(String, String)> {
    let mut blocks: Vec<(String, String)> = db
        .extensions()
        .iter()
        .filter_map(|ext| ext.boot_block(db).map(|text| (ext.id().to_string(), text)))
        .collect();
    if let Some(hint) = db.code_graph_hint() {
        blocks.push(("code_graph".to_string(), format!("## Code Graph\n- ⚠️ {hint}")));
    }
    blocks
}

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

/// Sort `records` highest effective importance first: stored importance
/// decayed by age with `half_life_days` ([`crate::importance::effective_importance`]),
/// the order boot ranks every table by.
///
/// Effective importance is computed here from age and half-life (a pure
/// function), so the order reflects current decay without relying on a
/// background sweep having stamped `_effective_importance`.
pub fn sort_by_effective_importance(
    records: &mut [Record],
    half_life_days: f64,
    now: DateTime<Utc>,
) {
    // Clamp age at 0 so a future-dated (clock-skewed) record can't get a >1
    // decay factor and jump to the top of the ranking.
    let score = |r: &Record| {
        let age_days = ((now - r.created_at).num_seconds() as f64 / 86400.0).max(0.0);
        crate::importance::effective_importance(&r.data, age_days, half_life_days)
    };
    records.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// True unless an `errors` record is marked resolved (`resolved: true`, set
/// by [`Axil::resolve_error`]). Open-error views filter on this; lessons
/// views keep resolved errors on purpose, since an error with its fix is
/// still worth knowing.
pub fn is_open_error(data: &Value) -> bool {
    data.get("resolved").and_then(Value::as_bool) != Some(true)
}

/// Compact age of `then` as seen at `now`: `"<1h"`, `"5h"`, `"3d"`. A
/// future timestamp (clock skew) reads as `"<1h"`.
pub fn boot_age(then: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let secs = (now - then).num_seconds().max(0);
    if secs < 3_600 {
        "<1h".to_string()
    } else if secs < 86_400 {
        format!("{}h", secs / 3_600)
    } else {
        format!("{}d", secs / 86_400)
    }
}

/// Flatten `s` to one line (every whitespace run becomes one space) and
/// clip it to `max_chars` chars, ending in `…` when anything was cut.
pub fn clip_one_line(s: &str, max_chars: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let mut out: String = flat.chars().take(max_chars.saturating_sub(1)).collect();
    out.truncate(out.trim_end().len());
    out.push('…');
    out
}

/// One boot row: `id · age · status · summary`, with `summary` flattened
/// and clipped to `max_chars` chars by [`clip_one_line`].
pub fn boot_row(id: &str, age: &str, status: &str, summary: &str, max_chars: usize) -> String {
    let summary = clip_one_line(summary, max_chars);
    format!("{id}{BOOT_ROW_SEP}{age}{BOOT_ROW_SEP}{status}{BOOT_ROW_SEP}{summary}")
}

/// The text a record's boot row leads with: `rule` for a rules record,
/// otherwise the first non-empty of `summary`, `error`, `rule`,
/// `statement`, `fact`, `description`, `name`, `title`, `text`, `content`.
/// A record with none of those shows its non-internal fields as compact
/// JSON.
pub fn record_headline(record: &Record) -> String {
    const KEYS: &[&str] = &[
        "summary",
        "error",
        "rule",
        "statement",
        "fact",
        "description",
        "name",
        "title",
        "text",
        "content",
    ];
    let lead: &[&str] = if record.table == "rules" {
        &["rule"]
    } else {
        &[]
    };
    for key in lead.iter().chain(KEYS) {
        if let Some(s) = record.data.get(*key).and_then(Value::as_str) {
            if !s.trim().is_empty() {
                return s.to_string();
            }
        }
    }
    match record.data.as_object() {
        Some(obj) => {
            let visible: serde_json::Map<String, Value> = obj
                .iter()
                .filter(|(k, _)| !k.starts_with('_'))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Value::Object(visible).to_string()
        }
        None => record.data.to_string(),
    }
}

/// The status word on a record's boot row: `superseded` for a superseded
/// record; otherwise `open`/`resolved` for errors, `pinned`/`rule` for
/// rules, the `type` facet (or `context`) for context notes, and `active`
/// for anything else.
pub fn record_status(record: &Record) -> String {
    let data = &record.data;
    if data.get("_superseded").and_then(Value::as_bool) == Some(true) {
        return "superseded".to_string();
    }
    match record.table.as_str() {
        "errors" if is_open_error(data) => "open",
        "errors" => "resolved",
        "rules" if crate::importance::is_pinned(data) => "pinned",
        "rules" => "rule",
        "context" => data
            .get("type")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|t| !t.is_empty() && t.len() <= 24 && !t.contains(char::is_whitespace))
            .unwrap_or("context"),
        _ => "active",
    }
    .to_string()
}

/// A record as one boot row, `id · age · status · summary`
/// ([`boot_row`]): rules keep up to [`BOOT_RULE_CHARS`] chars of their
/// text, other records [`BOOT_ROW_SUMMARY_CHARS`].
pub fn record_row(record: &Record, now: DateTime<Utc>) -> String {
    let max_chars = if record.table == "rules" {
        BOOT_RULE_CHARS
    } else {
        BOOT_ROW_SUMMARY_CHARS
    };
    boot_row(
        &record.id.to_string(),
        &boot_age(record.created_at, now),
        &record_status(record),
        &record_headline(record),
        max_chars,
    )
}

/// A `preferences` record as one boot row: `id · age · preference · key =
/// value`, a string value shown as-is and any other value as compact JSON.
pub fn preference_row(record: &Record, now: DateTime<Utc>) -> String {
    let key = record
        .data
        .get("key")
        .and_then(Value::as_str)
        .unwrap_or("?");
    let value = match record.data.get("value") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    boot_row(
        &record.id.to_string(),
        &boot_age(record.created_at, now),
        "preference",
        &format!("{key} = {value}"),
        BOOT_ROW_SUMMARY_CHARS,
    )
}

/// One section offered to [`fill_by_priority`].
#[derive(Debug, Clone)]
pub struct FillSlot {
    /// Name reported when rows are left out (`omitted_by_section`).
    pub name: String,
    /// Rows in display order: one-line records, or a prose block's lines.
    pub rows: Vec<String>,
    /// Whether a row that does not fit whole may be shortened to fit
    /// ([`clip_row`]). Off for a row that is structured data, not text.
    pub clippable: bool,
}

impl FillSlot {
    /// A clippable slot.
    pub fn new(name: impl Into<String>, rows: Vec<String>) -> Self {
        Self {
            name: name.into(),
            rows,
            clippable: true,
        }
    }
}

/// What [`fill_by_priority`] shows of each slot: `None` leaves the section
/// out, `Some(rows)` shows it with those rows (the last one possibly
/// clipped). An empty slot can be shown with no rows.
pub type FillSelection = Vec<Option<Vec<String>>>;

/// Choose the rows of `slots` (given highest priority first) to show within
/// a budget that `fits` checks on the real rendering of a selection.
///
/// Rows fill in strict priority order: every row of a slot before any row
/// of the next slot, and a slot with no rows at all is offered as the
/// section's bare presence. The first row that does not fit whole is
/// clipped to fit when its slot allows it, and ends the fill either way,
/// so no row is ever shown while a higher-priority one is left out or
/// shortened. `fits` is called on exactly the selection returned, so the
/// result fits unless the empty selection alone does not.
pub fn fill_by_priority(
    slots: &[FillSlot],
    mut fits: impl FnMut(&[Option<Vec<String>>]) -> bool,
) -> FillSelection {
    let everything: FillSelection = slots.iter().map(|s| Some(s.rows.clone())).collect();
    if fits(&everything) {
        return everything;
    }
    let mut sel: FillSelection = vec![None; slots.len()];
    for (i, slot) in slots.iter().enumerate() {
        if slot.rows.is_empty() {
            sel[i] = Some(Vec::new());
            if fits(&sel) {
                continue;
            }
            sel[i] = None;
            return sel;
        }
        for row in &slot.rows {
            let mut fits_with = |candidate: &str| {
                sel[i]
                    .get_or_insert_with(Vec::new)
                    .push(candidate.to_string());
                let ok = fits(&sel);
                if let Some(rows) = &mut sel[i] {
                    rows.pop();
                    if rows.is_empty() {
                        sel[i] = None;
                    }
                }
                ok
            };
            let shown = if slot.clippable {
                clip_row(row, &mut fits_with)
            } else {
                fits_with(row).then(|| row.clone())
            };
            let Some(shown) = shown else { return sel };
            let whole = shown == *row;
            sel[i].get_or_insert_with(Vec::new).push(shown);
            if !whole {
                return sel;
            }
        }
    }
    sel
}

/// Rows of each slot that `selection` leaves out, as `(name, count)` for
/// the names that lost any, in slot order; slots sharing a name (two
/// extension blocks with one id) are summed. A clipped row counts as shown.
pub fn omitted_by_slot(
    slots: &[FillSlot],
    selection: &[Option<Vec<String>>],
) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    for (slot, shown) in slots.iter().zip(selection) {
        let cut = slot.rows.len() - shown.as_ref().map_or(0, Vec::len);
        if cut == 0 {
            continue;
        }
        match out.iter_mut().find(|(name, _)| *name == slot.name) {
            Some((_, total)) => *total += cut,
            None => out.push((slot.name.clone(), cut)),
        }
    }
    out
}

/// `row` shortened until `fits` accepts it: its summary — the text after
/// the third field separator, after the last one when it has fewer, or
/// the whole line when it has none — is cut on a char boundary and ends in
/// `…`, so the leading `id · age · status` stay whole. Returns `row` itself
/// when it already fits, and `None` when even [`BOOT_CLIP_MIN_CHARS`] chars
/// of summary do not.
pub fn clip_row(row: &str, mut fits: impl FnMut(&str) -> bool) -> Option<String> {
    if fits(row) {
        return Some(row.to_string());
    }
    let head_len = row
        .match_indices(BOOT_ROW_SEP)
        .take(3)
        .last()
        .map_or(0, |(i, sep)| i + sep.len());
    let (head, summary) = row.split_at(head_len);
    let ends: Vec<usize> = summary.char_indices().map(|(i, _)| i).collect();
    if ends.len() <= BOOT_CLIP_MIN_CHARS {
        return None;
    }
    // `ends[k]` is where the first `k` chars end; the clipped form's cost
    // never falls as `k` grows, so the longest fit is a binary search.
    let clipped = |k: usize| format!("{head}{}…", summary[..ends[k]].trim_end());
    let (mut lo, mut hi) = (BOOT_CLIP_MIN_CHARS, ends.len() - 1);
    if !fits(&clipped(lo)) {
        return None;
    }
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(&clipped(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(clipped(lo))
}

/// Where a fill slot's rows go back in the context.
enum SlotHome {
    /// Lines of the `block`-th extension block of the `CurrentScope` section.
    Block(usize),
    /// Rows of the section at this index.
    Rows(usize),
    /// The whole content of the section at this index, as one row.
    Whole(usize),
}

/// A section's rows as the strings [`fill_by_priority`] works on.
fn row_strings(rows: &[Value]) -> Vec<String> {
    rows.iter()
        .map(|v| v.as_str().map_or_else(|| v.to_string(), String::from))
        .collect()
}

/// Fill `sections` into `ctx` within `ctx.token_budget` by
/// [`fill_by_priority`], then record what was dropped and cut and the
/// estimated size of the whole serialized context.
fn fit_to_budget(
    ctx: &mut BootContext,
    sections: Vec<BootSection>,
    estimator: &dyn TokenEstimator,
) {
    let mut order: Vec<usize> = (0..sections.len()).collect();
    order.sort_by_key(|&i| sections[i].priority());

    let mut slots: Vec<FillSlot> = Vec::new();
    let mut homes: Vec<SlotHome> = Vec::new();
    for &i in &order {
        let s = &sections[i];
        match s {
            BootSection::CurrentScope { content } => {
                let blocks = content.get("extension_blocks").and_then(Value::as_array);
                for (b, block) in blocks.into_iter().flatten().enumerate() {
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("extension");
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                    let lines = text.lines().map(String::from).collect();
                    slots.push(FillSlot::new(id, lines));
                    homes.push(SlotHome::Block(b));
                }
            }
            BootSection::Constraints { content } => {
                let rules = content.get("rules").and_then(Value::as_array);
                let rows = rules.map_or_else(Vec::new, |r| row_strings(r));
                slots.push(FillSlot::new(s.kind_str(), rows));
                homes.push(SlotHome::Rows(i));
            }
            BootSection::ActiveFailures { content }
            | BootSection::RecentDecisions { content }
            | BootSection::OpenThreads { content }
            | BootSection::Preferences { content } => {
                slots.push(FillSlot::new(s.kind_str(), row_strings(content)));
                homes.push(SlotHome::Rows(i));
            }
            BootSection::ConfidenceNotes { content } => {
                slots.push(FillSlot {
                    clippable: false,
                    ..FillSlot::new(s.kind_str(), vec![content.to_string()])
                });
                homes.push(SlotHome::Whole(i));
            }
        }
    }

    let original_blocks: Vec<Value> = sections
        .iter()
        .find_map(|s| match s {
            BootSection::CurrentScope { content } => {
                content.get("extension_blocks")?.as_array().cloned()
            }
            _ => None,
        })
        .unwrap_or_default();

    let render = |sel: &[Option<Vec<String>>]| -> BootContext {
        let mut blocks: Vec<Value> = Vec::new();
        let mut rows: Vec<Option<Vec<Value>>> = vec![None; sections.len()];
        let mut whole = vec![false; sections.len()];
        for ((slot, home), kept) in slots.iter().zip(&homes).zip(sel) {
            let Some(kept) = kept else { continue };
            match *home {
                SlotHome::Block(b) => {
                    let mut block = original_blocks[b].clone();
                    // A block shown whole keeps its exact text (trailing
                    // newline included); a cut one is its kept lines.
                    if *kept != slot.rows {
                        block["text"] = Value::String(kept.join("\n"));
                    }
                    blocks.push(block);
                }
                SlotHome::Rows(i) => {
                    rows[i] = Some(kept.iter().cloned().map(Value::String).collect());
                }
                SlotHome::Whole(i) => whole[i] = true,
            }
        }

        // Scope, constraints, failures and decisions are load-bearing for
        // planning: always shown, only their rows trimmed. The rest are
        // shown only when the fill kept them.
        let mut shown: Vec<BootSection> = Vec::new();
        let mut dropped: Vec<usize> = Vec::new();
        for (i, section) in sections.iter().enumerate() {
            let kept = rows[i].take();
            let fitted = match section {
                BootSection::CurrentScope { content } => {
                    let mut content = content.clone();
                    if let Some(obj) = content.as_object_mut() {
                        obj.remove("extension_blocks");
                        if !blocks.is_empty() {
                            let blocks = std::mem::take(&mut blocks);
                            obj.insert("extension_blocks".into(), Value::Array(blocks));
                        }
                    }
                    Some(BootSection::CurrentScope { content })
                }
                BootSection::Constraints { content } => {
                    let mut content = content.clone();
                    content["rules"] = Value::Array(kept.unwrap_or_default());
                    Some(BootSection::Constraints { content })
                }
                BootSection::ActiveFailures { .. } => Some(BootSection::ActiveFailures {
                    content: kept.unwrap_or_default(),
                }),
                BootSection::RecentDecisions { .. } => Some(BootSection::RecentDecisions {
                    content: kept.unwrap_or_default(),
                }),
                BootSection::OpenThreads { .. } => {
                    kept.map(|content| BootSection::OpenThreads { content })
                }
                BootSection::Preferences { .. } => {
                    kept.map(|content| BootSection::Preferences { content })
                }
                BootSection::ConfidenceNotes { content } => {
                    whole[i].then(|| BootSection::ConfidenceNotes {
                        content: content.clone(),
                    })
                }
            };
            match fitted {
                Some(section) => shown.push(section),
                None => dropped.push(i),
            }
        }
        dropped.sort_by_key(|&i| std::cmp::Reverse(sections[i].priority()));

        let omitted = omitted_by_slot(&slots, sel);
        let mut out = ctx.clone();
        out.sections = shown;
        out.dropped_sections = dropped
            .iter()
            .map(|&i| sections[i].kind_str().to_string())
            .collect();
        out.omitted_items = omitted.iter().map(|(_, n)| n).sum();
        out.omitted_by_section = omitted.into_iter().collect();
        // Sized with the widest value `token_budget_used` can end up with,
        // so the final context is never bigger than the one checked.
        out.token_budget_used = out.token_budget;
        out
    };
    let size =
        |c: &BootContext| estimator.estimate_tokens(&serde_json::to_string(c).unwrap_or_default());
    let budget = ctx.token_budget;
    let selection = fill_by_priority(&slots, |sel| size(&render(sel)) <= budget);
    *ctx = render(&selection);
    // Twice: the first pass sizes the context with a placeholder count, the
    // second with the count itself.
    for _ in 0..2 {
        ctx.token_budget_used = size(ctx);
    }
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
    fn boot_returns_schema_2_with_fixed_order() {
        let (db, _dir) = temp_db();
        let ctx = db.boot(BootOptions::default()).unwrap();
        assert_eq!(ctx.schema_version, "2");
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

    // ---- one-line rows and the budget inside load-bearing sections ----

    fn section<'a>(ctx: &'a BootContext, kind: &str) -> Option<&'a BootSection> {
        ctx.sections.iter().find(|s| s.kind_str() == kind)
    }

    fn rows<'a>(ctx: &'a BootContext, kind: &str) -> &'a [Value] {
        match section(ctx, kind) {
            Some(BootSection::RecentDecisions { content })
            | Some(BootSection::ActiveFailures { content })
            | Some(BootSection::OpenThreads { content })
            | Some(BootSection::Preferences { content }) => content,
            _ => &[],
        }
    }

    #[test]
    fn rows_are_one_liners_with_id_age_status_summary() {
        let (db, _dir) = temp_db();
        let rec = db
            .insert(
                "decisions",
                json!({ "summary": "Use redb\nfor core storage", "reason": "ACID, pure Rust" }),
            )
            .unwrap();
        db.insert(
            "rules",
            json!({ "rule": "Never push to main", "_importance": 1.0 }),
        )
        .unwrap();
        let ctx = db.boot(BootOptions::default()).unwrap();

        let row = rows(&ctx, "recent_decisions")[0]
            .as_str()
            .expect("rows are strings");
        assert_eq!(
            row,
            format!("{} · <1h · active · Use redb for core storage", rec.id)
        );

        let Some(BootSection::Constraints { content }) = section(&ctx, "constraints") else {
            panic!("constraints section");
        };
        let rule = content["rules"][0].as_str().unwrap();
        assert!(
            rule.ends_with(" · <1h · rule · Never push to main"),
            "{rule}"
        );
    }

    #[test]
    fn active_failures_list_only_open_errors() {
        let (db, _dir) = temp_db();
        db.insert("errors", json!({ "error": "still broken" }))
            .unwrap();
        db.insert(
            "errors",
            json!({ "error": "fixed already", "resolved": true }),
        )
        .unwrap();
        let ctx = db.boot(BootOptions::default()).unwrap();
        let failures = rows(&ctx, "active_failures");
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0]
            .as_str()
            .unwrap()
            .ends_with(" · open · still broken"));
    }

    #[test]
    fn budget_holds_inside_load_bearing_sections() {
        let (db, _dir) = temp_db();
        let long = "a decision with plenty of words to cost tokens ".repeat(20);
        for i in 0..30 {
            db.insert("decisions", json!({ "summary": format!("{i} {long}") }))
                .unwrap();
            db.insert("errors", json!({ "error": format!("{i} {long}") }))
                .unwrap();
            db.insert("context", json!({ "summary": format!("{i} {long}") }))
                .unwrap();
        }
        for budget in [150, 300, 600, 1000] {
            let ctx = db
                .boot(BootOptions {
                    token_budget: Some(budget),
                    ..Default::default()
                })
                .unwrap();
            let serialized = serde_json::to_string(&ctx).unwrap();
            let estimate = crate::token::DEFAULT_TOKEN_ESTIMATOR.estimate_tokens(&serialized);
            assert!(
                estimate <= budget,
                "budget {budget}: serialized estimate {estimate}"
            );
            assert!(
                ctx.token_budget_used <= budget,
                "budget {budget}: used {}",
                ctx.token_budget_used
            );
            assert!(ctx.omitted_items > 0, "budget {budget} should cut rows");
            for kind in [
                "current_scope",
                "constraints",
                "recent_decisions",
                "active_failures",
            ] {
                assert!(
                    section(&ctx, kind).is_some(),
                    "budget {budget}: {kind} kept"
                );
            }
        }
    }

    #[test]
    fn open_failures_fill_before_decisions() {
        let (db, _dir) = temp_db();
        let long = "words that cost tokens ".repeat(6);
        for i in 0..10 {
            db.insert(
                "decisions",
                json!({ "summary": format!("decision {i} {long}") }),
            )
            .unwrap();
            db.insert("errors", json!({ "error": format!("error {i} {long}") }))
                .unwrap();
        }
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(400),
                ..Default::default()
            })
            .unwrap();
        let failures = rows(&ctx, "active_failures").len();
        let decisions = rows(&ctx, "recent_decisions").len();
        assert!(failures > 0, "some failures fit");
        assert!(
            failures >= decisions,
            "failures {failures} fill before decisions {decisions}"
        );
        // Display order is unchanged by the fill order.
        let kinds: Vec<&str> = ctx.sections.iter().map(BootSection::kind_str).collect();
        let pos = |k: &str| kinds.iter().position(|x| *x == k).unwrap();
        assert!(pos("recent_decisions") < pos("active_failures"));
    }

    #[test]
    fn an_oversized_extension_block_keeps_its_leading_lines() {
        struct LongBlock;
        impl crate::Extension for LongBlock {
            fn id(&self) -> &str {
                "long"
            }
            fn boot_block(&self, _db: &Axil) -> Option<String> {
                let mut text = String::from("## Resume Here\n");
                for i in 0..200 {
                    text.push_str(&format!("- step {i} of a very long resume block\n"));
                }
                Some(text)
            }
        }
        let dir = tempdir().unwrap();
        let db = Axil::open(dir.path().join("test.axil"))
            .with_extension(LongBlock)
            .build()
            .unwrap();
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(500),
                ..Default::default()
            })
            .unwrap();
        let Some(BootSection::CurrentScope { content }) = section(&ctx, "current_scope") else {
            panic!("current scope");
        };
        let text = content["extension_blocks"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("## Resume Here\n- step 0 "), "{text}");
        assert!(!text.contains("step 199"), "the tail is cut");
        assert!(ctx.omitted_items > 0);
        assert!(ctx.token_budget_used <= 500);
    }

    #[test]
    fn zero_budget_means_the_default() {
        let (db, _dir) = temp_db();
        let ctx = db
            .boot(BootOptions {
                token_budget: Some(0),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(ctx.token_budget, DEFAULT_TOKEN_BUDGET);
    }

    #[test]
    fn clip_one_line_flattens_and_clips_on_chars() {
        assert_eq!(clip_one_line("a\n  b\tc", 10), "a b c");
        assert_eq!(clip_one_line("abcdef", 4), "abc…");
        // Multibyte text clips on char boundaries, never mid-character.
        assert_eq!(clip_one_line("résumé — naïve", 8), "résumé…");
        assert_eq!(clip_one_line("abc", 0), "…");
    }

    #[test]
    fn boot_age_is_compact() {
        let now = Utc::now();
        assert_eq!(boot_age(now, now), "<1h");
        assert_eq!(boot_age(now - chrono::Duration::hours(5), now), "5h");
        assert_eq!(boot_age(now - chrono::Duration::days(3), now), "3d");
        assert_eq!(boot_age(now + chrono::Duration::days(1), now), "<1h");
    }

    #[test]
    fn clip_row_keeps_the_leading_fields() {
        let row = "01ABC · 2d · open · the summary that is far too long to fit whole here";
        let clipped = clip_row(row, |r| r.len() <= 50).unwrap();
        assert!(
            clipped.starts_with("01ABC · 2d · open · the summary"),
            "{clipped}"
        );
        assert!(clipped.ends_with('…') && clipped.len() <= 50, "{clipped}");
        // Fits whole: returned as is.
        assert_eq!(clip_row(row, |_| true).as_deref(), Some(row));
        // Too little room for the fields plus a readable summary.
        assert_eq!(clip_row(row, |r| r.len() <= 30), None);
        // Prose without separators clips from its start, on char boundaries.
        let prose = "- **State:** résumé — naïve text that goes on and on and on";
        let clipped = clip_row(prose, |r| r.len() <= 40).unwrap();
        assert!(
            prose.starts_with(clipped.trim_end_matches('…')),
            "{clipped}"
        );
    }

    fn slot(name: &str, rows: &[&str]) -> FillSlot {
        FillSlot::new(name, rows.iter().map(|r| r.to_string()).collect())
    }

    /// Bytes of a selection rendered as one line per row, the simplest
    /// stand-in for a real renderer.
    fn bytes(sel: &[Option<Vec<String>>]) -> usize {
        sel.iter().flatten().flatten().map(|r| r.len() + 1).sum()
    }

    #[test]
    fn fill_stops_at_the_first_row_that_misses() {
        let long = "x".repeat(200);
        let slots = [
            slot("rules", &[&format!("R1 · 1d · pinned · {long}")]),
            slot(
                "decisions",
                &["D1 · 1d · active · short", "D2 · 1d · active · short"],
            ),
        ];
        let sel = fill_by_priority(&slots, |s| bytes(s) <= 120);
        // The rule is clipped to the room left, and the shorter decisions
        // after it are not shown in its place.
        let rules = sel[0].as_ref().expect("the rule is shown, clipped");
        assert!(rules[0].starts_with("R1 · 1d · pinned · xxx") && rules[0].ends_with('…'));
        assert!(sel[1].is_none(), "{sel:?}");
        assert_eq!(
            omitted_by_slot(&slots, &sel),
            vec![("decisions".to_string(), 2)]
        );
    }

    /// At every budget the selection is a prefix of all rows in fill
    /// order: every row before the last one shown is whole, and only the
    /// last may be clipped.
    #[test]
    fn fill_shows_a_prefix_of_the_rows_in_priority_order() {
        let long = |p: &str| format!("{p} · 1d · open · {}", "words that run on ".repeat(6));
        let (e1, e2, e3) = (long("E1"), long("E2"), long("E3"));
        let slots = [
            slot(
                "checkpoint",
                &["## Resume Here", "- **State:** mid-refactor"],
            ),
            slot("rules", &["R1 · 9d · pinned · always ask first"]),
            slot("errors", &[&e1, &e2, &e3]),
            slot(
                "decisions",
                &["D1 · 1d · active · a", "D2 · 1d · active · b"],
            ),
            slot("recent_sessions", &["S1 · 1d · recent · a"]),
        ];
        let all: Vec<&String> = slots.iter().flat_map(|s| &s.rows).collect();
        for budget in 0..700 {
            let sel = fill_by_priority(&slots, |s| bytes(s) <= budget);
            let shown: Vec<&String> = sel.iter().flatten().flatten().collect();
            assert!(bytes(&sel) <= budget, "budget {budget}");
            for (k, row) in shown.iter().enumerate() {
                if k + 1 < shown.len() {
                    assert_eq!(*row, all[k], "budget {budget}: row {k} out of order or cut");
                } else if *row != all[k] {
                    let kept = row.strip_suffix('…').expect("a clipped row ends in …");
                    assert!(
                        all[k].starts_with(kept.trim_end()),
                        "budget {budget}: {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn fill_returns_everything_when_it_fits_and_shows_empty_slots() {
        let slots = [
            slot("errors", &["E1 · 1d · open · a"]),
            slot("topic_recall", &[]),
        ];
        let sel = fill_by_priority(&slots, |_| true);
        assert_eq!(
            sel,
            vec![Some(vec!["E1 · 1d · open · a".to_string()]), Some(vec![])]
        );
        // Budget for the row but not the empty slot's key.
        let sel = fill_by_priority(&slots, |s| s[1].is_none());
        assert_eq!(
            sel,
            vec![Some(vec!["E1 · 1d · open · a".to_string()]), None]
        );
    }

    /// Long pinned rules and open errors, short decisions, threads and
    /// preferences: at every budget, a section shows rows only when every
    /// row of every higher-priority section is shown whole, and confidence
    /// notes (lowest priority) only when nothing was cut.
    #[test]
    fn a_small_budget_never_hides_a_higher_priority_row() {
        let (db, _dir) = temp_db();
        for i in 0..2 {
            let rule = format!("Rule {i}: always check the rule text. ").repeat(9);
            db.insert("rules", json!({ "rule": rule, "_importance_pinned": true }))
                .unwrap();
        }
        for i in 0..3 {
            let error = format!("Open error {i} keeps happening in the boot path. ").repeat(4);
            db.insert("errors", json!({ "error": error })).unwrap();
        }
        for i in 0..6 {
            db.insert("decisions", json!({ "summary": format!("decision {i}") }))
                .unwrap();
        }
        for i in 0..3 {
            db.insert("context", json!({ "summary": format!("thread {i}") }))
                .unwrap();
        }
        for i in 0..2 {
            db.insert(
                "preferences",
                json!({ "key": format!("k{i}"), "value": "v" }),
            )
            .unwrap();
        }
        let boot = |budget: usize| {
            db.boot(BootOptions {
                token_budget: Some(budget),
                ..Default::default()
            })
            .unwrap()
        };
        let fill_order = [
            "constraints",
            "active_failures",
            "recent_decisions",
            "open_threads",
            "preferences",
        ];
        let rows_of = |ctx: &BootContext, kind: &str| -> Vec<Value> {
            match section(ctx, kind) {
                Some(BootSection::Constraints { content }) => {
                    content["rules"].as_array().cloned().unwrap_or_default()
                }
                _ => rows(ctx, kind).to_vec(),
            }
        };
        let full = boot(100_000);
        let full_rows: Vec<Vec<Value>> = fill_order.iter().map(|k| rows_of(&full, k)).collect();
        let counts: Vec<usize> = full_rows.iter().map(Vec::len).collect();
        assert_eq!(counts, vec![2, 3, 6, 3, 2]);
        assert!(section(&full, "confidence_notes").is_some());

        for budget in (60..1500).step_by(7) {
            let ctx = boot(budget);
            let mut complete = true;
            for (kind, full) in fill_order.iter().zip(&full_rows) {
                let shown = rows_of(&ctx, kind);
                if !complete {
                    assert!(
                        shown.is_empty(),
                        "budget {budget}: {kind} shows a row while a higher-priority row is cut"
                    );
                    continue;
                }
                // Rows of equal importance may swap places between boots,
                // so a row counts as whole when the full boot has it at all.
                for (j, row) in shown.iter().enumerate() {
                    if !full.contains(row) {
                        assert_eq!(j + 1, shown.len(), "budget {budget}: {kind} row {j}: {row}");
                        assert!(row.as_str().unwrap().ends_with('…'), "{row}");
                        complete = false;
                    }
                }
                complete &= shown.len() == full.len();
            }
            if section(&ctx, "confidence_notes").is_some() {
                assert!(complete && ctx.omitted_items == 0, "budget {budget}");
            }
            let per_section: usize = ctx.omitted_by_section.values().sum();
            assert_eq!(per_section, ctx.omitted_items, "budget {budget}");
            // Any row shown was checked against the budget; with none shown
            // the envelope and empty skeletons alone may overshoot it.
            if rows_of(&ctx, "constraints").is_empty() {
                continue;
            }
            assert!(
                ctx.token_budget_used <= budget,
                "budget {budget}: used {}",
                ctx.token_budget_used
            );
        }
    }

    #[test]
    fn preferences_are_one_line_rows() {
        let (db, _dir) = temp_db();
        let rec = db
            .insert("preferences", json!({ "key": "editor", "value": "helix" }))
            .unwrap();
        let ctx = db.boot(BootOptions::default()).unwrap();
        assert_eq!(
            rows(&ctx, "preferences")[0].as_str().unwrap(),
            format!("{} · <1h · preference · editor = helix", rec.id)
        );
    }

    #[test]
    fn effective_importance_lets_recent_work_outrank_old() {
        let now = Utc::now();
        let mut old = Record::new("decisions", json!({ "summary": "old", "_importance": 1.0 }));
        old.created_at = now - chrono::Duration::days(180);
        let recent = Record::new(
            "decisions",
            json!({ "summary": "recent", "_importance": 0.6 }),
        );
        let mut pinned = Record::new(
            "decisions",
            json!({ "summary": "pinned", "_importance_pinned": true }),
        );
        pinned.created_at = now - chrono::Duration::days(365);
        let mut records = vec![old, recent, pinned];
        sort_by_effective_importance(&mut records, 90.0, now);
        let order: Vec<&str> = records
            .iter()
            .map(|r| r.data["summary"].as_str().unwrap())
            .collect();
        assert_eq!(order, ["pinned", "recent", "old"]);
    }

    #[test]
    fn record_status_reports_lifecycle() {
        let (db, _dir) = temp_db();
        let open = db.insert("errors", json!({ "error": "e" })).unwrap();
        let fixed = db
            .insert("errors", json!({ "error": "e", "resolved": true }))
            .unwrap();
        let pinned = db
            .insert("rules", json!({ "rule": "r", "_importance_pinned": true }))
            .unwrap();
        let arch = db
            .insert("context", json!({ "summary": "s", "type": "architecture" }))
            .unwrap();
        let old = db
            .insert("decisions", json!({ "summary": "s", "_superseded": true }))
            .unwrap();
        assert_eq!(record_status(&open), "open");
        assert_eq!(record_status(&fixed), "resolved");
        assert_eq!(record_status(&pinned), "pinned");
        assert_eq!(record_status(&arch), "architecture");
        assert_eq!(record_status(&old), "superseded");
    }
}
