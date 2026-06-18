//! Query intent detection and smart routing for `axil ask`.
//!
//! Detects what KIND of query an agent is asking and routes to the
//! appropriate backend(s) — vector, graph, FTS, temporal, or a blend.

use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Direction, Record};

use crate::recall::RecallResult;
use crate::token;

// ── Compiled regexes ───────────────────────────────────────────────

static RE_TEMPORAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:last|past|since)\s+\d+\s*(?:day|hour|minute|week|month)").unwrap()
});

static RE_DURATION_N: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:last|past)\s+(\d+)\s*(day|hour|minute|week|month)s?").unwrap()
});

static RE_DURATION_SINGLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:last|past)\s+(day|hour|minute|week|month)").unwrap());

// ── Intent ──────────────────────────────────────────────────────────

/// The detected intent of a natural-language query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// "similar to X", "like X", "related to X"
    VectorSearch,
    /// "connected to X", "depends on X", "what uses X"
    GraphTraversal,
    /// "why did we", "what led to", "decision about"
    Causality,
    /// "since yesterday", "last N days", "changed recently"
    Temporal,
    /// "always", "never", "rule about", "convention for"
    RuleLookup,
    /// "exact error", "error message", "find the string"
    TextSearch,
    /// Ambiguous — blend multiple strategies.
    Combined,
}

impl std::fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VectorSearch => write!(f, "vector_search"),
            Self::GraphTraversal => write!(f, "graph_traversal"),
            Self::Causality => write!(f, "causality"),
            Self::Temporal => write!(f, "temporal"),
            Self::RuleLookup => write!(f, "rule_lookup"),
            Self::TextSearch => write!(f, "text_search"),
            Self::Combined => write!(f, "combined"),
        }
    }
}

// ── Result ──────────────────────────────────────────────────────────

/// Result of an `ask` query, including routing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskResult {
    /// Detected query intent.
    pub intent: QueryIntent,
    /// Original query string.
    pub query: String,
    /// Merged results as JSON values.
    pub results: Vec<Value>,
    /// Approximate token cost of the result set.
    pub tokens: usize,
    /// Which retrieval strategies were actually used.
    pub strategies_used: Vec<String>,
    /// The query plan if multi-step decomposition was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<QueryPlan>,
}

// ── Query Planner ──────────────────────────────────────────────────

/// A single step in a query plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStep {
    pub step: usize,
    #[serde(rename = "type")]
    pub query_type: String,
    pub query: String,
    /// If set, this step uses results from a previous step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
}

/// A multi-step query plan decomposed from a complex question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub steps: Vec<QueryStep>,
}

/// Decompose a complex query into a multi-step plan.
///
/// Detects when a query contains multiple intent signals and creates
/// a pipeline of steps where each feeds into the next.
pub fn plan_query(query: &str) -> QueryPlan {
    let q = query.to_lowercase();
    let mut steps = Vec::new();
    let mut step_num = 1;

    // Detect which signals are present in the query.
    let has_similarity = contains_any(&q, &["similar", "like", "related", "resembles"]);
    let has_temporal =
        has_temporal_pattern(&q) || contains_any(&q, &["recently", "yesterday", "today"]);
    let has_graph = contains_any(&q, &["connected", "depends", "uses", "imports", "affected"]);
    let has_causality = contains_any(
        &q,
        &["why did", "what led to", "decision", "decided", "reason"],
    );
    let has_fts = contains_any(&q, &["exact", "error message", "find the string", "grep"]);
    let has_rules = contains_any(&q, &["always", "never", "rule", "convention", "should"]);

    // Build steps based on detected signals.
    // FTS/vector first (find the entities), then graph (expand), then time (filter).
    if has_fts {
        steps.push(QueryStep {
            step: step_num,
            query_type: "fts".to_string(),
            query: query.to_string(),
            from: None,
        });
        step_num += 1;
    }

    if has_similarity && !has_fts {
        steps.push(QueryStep {
            step: step_num,
            query_type: "vector".to_string(),
            query: query.to_string(),
            from: None,
        });
        step_num += 1;
    }

    if has_causality || has_graph {
        let from = if step_num > 1 {
            Some(format!("step{}_results", step_num - 1))
        } else {
            None
        };
        let traverse = if has_causality {
            "->decided_by->"
        } else {
            "->depends_on->"
        };
        steps.push(QueryStep {
            step: step_num,
            query_type: "graph".to_string(),
            query: traverse.to_string(),
            from,
        });
        step_num += 1;
    }

    if has_rules {
        steps.push(QueryStep {
            step: step_num,
            query_type: "rules".to_string(),
            query: query.to_string(),
            from: None,
        });
        step_num += 1;
    }

    if has_temporal {
        steps.push(QueryStep {
            step: step_num,
            query_type: "time_filter".to_string(),
            query: "desc".to_string(),
            from: if step_num > 1 {
                Some(format!("step{}_results", step_num - 1))
            } else {
                None
            },
        });
        step_num += 1;
    }

    // Fallback: if no signals detected, single combined step.
    if steps.is_empty() {
        steps.push(QueryStep {
            step: 1,
            query_type: "combined".to_string(),
            query: query.to_string(),
            from: None,
        });
    }

    let _ = step_num; // suppress unused warning
    QueryPlan { steps }
}

/// Execute a query plan, running each step sequentially.
///
/// Results from earlier steps are passed to later steps that reference them
/// via the `from` field. The final merged result set is returned.
pub fn execute_plan(
    db: &Axil,
    plan: &QueryPlan,
    query: &str,
    top_k: usize,
) -> axil_core::Result<AskResult> {
    let mut all_results: Vec<Value> = Vec::new();
    let mut strategies: Vec<String> = Vec::new();
    // Map from "stepN_results" → record IDs produced by that step.
    let mut step_ids: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for step in &plan.steps {
        let step_key = format!("step{}_results", step.step);
        let mut step_results: Vec<Value> = Vec::new();

        // Get seed IDs from a previous step if referenced.
        let seed_ids: Vec<String> = step
            .from
            .as_ref()
            .and_then(|key| step_ids.get(key))
            .cloned()
            .unwrap_or_default();

        match step.query_type.as_str() {
            "vector" => {
                if let Ok(hits) = db.similar_to(query, top_k) {
                    strategies.push("vector".to_string());
                    for (record, score) in hits {
                        step_results.push(record_to_value(&record, score, "vector"));
                    }
                }
            }
            "fts" => {
                if let Ok(hits) = db.search_text(query, top_k) {
                    strategies.push("fts".to_string());
                    for (record, score) in hits {
                        step_results.push(record_to_value(&record, score, "fts"));
                    }
                }
            }
            "graph" => {
                strategies.push("graph".to_string());
                if !seed_ids.is_empty() {
                    // Traverse from seed IDs produced by the previous step.
                    let mut seen = std::collections::HashSet::new();
                    for id_str in &seed_ids {
                        let rid = match axil_core::RecordId::from_string(id_str) {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        if let Ok(neighbors) = db.neighbors(&rid, None, Direction::Both) {
                            for n in neighbors {
                                if seen.insert(n.id.to_string()) {
                                    step_results.push(record_to_value(&n, 0.7, "graph"));
                                }
                            }
                        }
                        if step_results.len() >= top_k {
                            break;
                        }
                    }
                } else {
                    // No seeds — find seeds via vector/FTS, then traverse.
                    let seeds = find_seed_records(db, query, 3);
                    let mut seen = std::collections::HashSet::new();
                    for (seed, score) in &seeds {
                        if seen.insert(seed.id.to_string()) {
                            step_results.push(record_to_value(seed, *score, "graph_seed"));
                        }
                        if let Ok(neighbors) = db.neighbors(&seed.id, None, Direction::Both) {
                            for n in neighbors {
                                if seen.insert(n.id.to_string()) {
                                    step_results.push(record_to_value(&n, score * 0.8, "graph"));
                                }
                            }
                        }
                        if step_results.len() >= top_k {
                            break;
                        }
                    }
                }
                step_results.truncate(top_k);
            }
            "rules" => {
                if let Ok(rules) = crate::rules::list_rules(db) {
                    strategies.push("rules".to_string());
                    for rule in &rules {
                        let tokens = token::estimate_tokens(&rule.rule);
                        step_results.push(json!({
                            "id": rule.key,
                            "table": crate::rules::TABLE_RULES,
                            "summary": rule.rule,
                            "tokens": tokens,
                            "strategy": "rule",
                        }));
                    }
                }
            }
            "time_filter" => {
                strategies.push("time_filter".to_string());
                let duration = parse_duration_from_query(query).unwrap_or(7 * 86400);

                if !seed_ids.is_empty() {
                    // Filter accumulated results to only those within the time window.
                    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(duration as i64);
                    let cutoff_str = cutoff.to_rfc3339();
                    all_results.retain(|v| {
                        v.get("created_at")
                            .and_then(|t| t.as_str())
                            .map(|t| t >= cutoff_str.as_str())
                            .unwrap_or(true)
                    });
                } else if let Ok(records) = db.since(None, duration) {
                    for record in records.into_iter().take(top_k) {
                        step_results.push(record_to_value(&record, 1.0, "temporal"));
                    }
                }
                // Sort all accumulated results by created_at descending.
                all_results.sort_by(|a, b| {
                    let ta = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
                    let tb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
                    tb.cmp(ta)
                });
            }
            _ => {
                // Fallback: run combined vector+FTS.
                let mut dummy_strats = Vec::new();
                run_vector(db, query, top_k, &mut step_results, &mut dummy_strats);
                run_text_search(db, query, top_k, &mut step_results, &mut dummy_strats);
                strategies.extend(dummy_strats);
            }
        }

        // Record this step's result IDs for downstream steps.
        let ids: Vec<String> = step_results
            .iter()
            .filter_map(|v| v.get("id").and_then(|id| id.as_str()).map(String::from))
            .collect();
        step_ids.insert(step_key, ids);

        all_results.extend(step_results);
    }

    // Deduplicate by ID, keeping first occurrence (earlier steps are higher priority).
    let mut seen = std::collections::HashSet::new();
    all_results.retain(|v| {
        let id = v.get("id").and_then(|i| i.as_str()).unwrap_or("");
        seen.insert(id.to_string())
    });
    all_results.truncate(top_k);

    let tokens: usize = all_results.iter().map(token::estimate_json_tokens).sum();

    Ok(AskResult {
        intent: QueryIntent::Combined,
        query: query.to_string(),
        results: all_results,
        tokens,
        strategies_used: strategies,
        plan: Some(plan.clone()),
    })
}

// ── Intent Detection ────────────────────────────────────────────────

/// Detect the intent of a natural-language query using keyword/pattern
/// matching (no LLM required).
pub fn detect_intent(query: &str) -> QueryIntent {
    let q = query.to_lowercase();

    // Order matters — more specific patterns first.

    // TextSearch — exact/literal searches
    if contains_any(
        &q,
        &[
            "exact",
            "error message",
            "find the string",
            "search for",
            "grep",
            "literal",
            "verbatim",
        ],
    ) {
        return QueryIntent::TextSearch;
    }

    // Causality — "why" / "what led to" / decisions
    if contains_any(
        &q,
        &[
            "why did",
            "what led to",
            "decision about",
            "decided",
            "reason for",
            "rationale",
            "motivation for",
            "caused by",
        ],
    ) {
        return QueryIntent::Causality;
    }

    // GraphTraversal — relationships / dependencies
    if contains_any(
        &q,
        &[
            "connected to",
            "depends on",
            "what uses",
            "imports",
            "imported by",
            "affected by",
            "links to",
            "references",
            "who calls",
            "called by",
            "dependency",
            "dependents",
        ],
    ) {
        return QueryIntent::GraphTraversal;
    }

    // Temporal — time-based queries
    if contains_any(
        &q,
        &[
            "since yesterday",
            "last week",
            "last month",
            "recently",
            "changed recently",
            "new today",
            "today",
        ],
    ) || has_temporal_pattern(&q)
    {
        return QueryIntent::Temporal;
    }

    // RuleLookup — conventions / rules
    if contains_any(
        &q,
        &[
            "always",
            "never",
            "rule about",
            "rule for",
            "convention for",
            "convention about",
            "should i",
            "should we",
            "must",
            "best practice",
            "guideline",
        ],
    ) {
        return QueryIntent::RuleLookup;
    }

    // VectorSearch — similarity
    if contains_any(
        &q,
        &[
            "similar to",
            "like",
            "related to",
            "resembles",
            "closest to",
            "analogous",
        ],
    ) {
        return QueryIntent::VectorSearch;
    }

    // Fallback — blend everything available.
    QueryIntent::Combined
}

/// Check if `text` contains any of the given `patterns`.
fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| text.contains(p))
}

/// Check for patterns like "last 3 days", "past 2 hours", "since 4/1".
fn has_temporal_pattern(text: &str) -> bool {
    RE_TEMPORAL.is_match(text)
}

// ── Duration Parsing ────────────────────────────────────────────────

/// Parse a duration (in seconds) from natural language in a query string.
///
/// Returns `None` if no recognizable duration is found.
///
/// # Examples
/// - "last 3 days" → `Some(259200)`
/// - "since yesterday" → `Some(86400)`
/// - "last hour" → `Some(3600)`
/// - "recently" → `Some(604800)` (default: 1 week)
pub fn parse_duration_from_query(query: &str) -> Option<u64> {
    let q = query.to_lowercase();

    if let Some(caps) = RE_DURATION_N.captures(&q) {
        let n: u64 = caps[1].parse().ok()?;
        let secs = match &caps[2] {
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            "week" => 7 * 86400,
            "month" => 30 * 86400,
            _ => return None,
        };
        return Some(n * secs);
    }

    if let Some(caps) = RE_DURATION_SINGLE.captures(&q) {
        let secs = match &caps[1] {
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            "week" => 7 * 86400,
            "month" => 30 * 86400,
            _ => return None,
        };
        return Some(secs);
    }

    // Keywords
    if q.contains("yesterday") {
        return Some(86400);
    }
    if q.contains("today") {
        return Some(86400);
    }
    if q.contains("recently") || q.contains("recent") {
        return Some(7 * 86400);
    }

    None
}

// ── Main Ask Function ───────────────────────────────────────────────

/// Execute a smart query against the database.
///
/// Detects the query intent and routes to the appropriate backend(s),
/// returning merged, token-counted results.
pub fn ask(db: &Axil, query: &str, top_k: usize) -> axil_core::Result<AskResult> {
    let intent = detect_intent(query);
    let mut results: Vec<Value> = Vec::new();
    let mut strategies: Vec<String> = Vec::new();

    match intent {
        QueryIntent::VectorSearch => {
            run_vector(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::GraphTraversal => {
            run_graph_traversal(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::Causality => {
            run_causality(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::Temporal => {
            run_temporal(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::RuleLookup => {
            run_rule_lookup(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::TextSearch => {
            run_text_search(db, query, top_k, &mut results, &mut strategies);
        }
        QueryIntent::Combined => {
            run_combined(db, query, top_k, &mut results, &mut strategies);
        }
    }

    // Global fallback: if primary strategy produced nothing, use text-matching recall
    if results.is_empty() {
        if let Ok(recall_results) = crate::recall::recall(db, query, top_k) {
            if !strategies.contains(&"text_match".to_string()) {
                strategies.push("text_match".to_string());
            }
            results.extend(recall_results.iter().map(recall_result_to_value));
        }
    }

    let tokens: usize = results.iter().map(token::estimate_json_tokens).sum();

    Ok(AskResult {
        intent,
        query: query.to_string(),
        results,
        tokens,
        strategies_used: strategies,
        plan: None,
    })
}

// ── Parallel Multi-Strategy Retrieval ───────────────────────────────

/// A single strategy's results before fusion.
#[derive(Debug, Clone)]
struct StrategyResult {
    name: String,
    items: Vec<Value>,
}

/// Run multiple retrieval strategies in parallel and fuse with Reciprocal Rank Fusion.
///
/// `allowed_strategies` limits which strategies run. Pass `None` to run all.
/// Returns fused results with RRF scores, sorted descending.
pub async fn ask_parallel(
    db: Arc<Axil>,
    query: &str,
    top_k: usize,
    allowed_strategies: Option<Vec<String>>,
) -> axil_core::Result<AskResult> {
    let q = query.to_string();
    let should_run = |name: &str| -> bool {
        allowed_strategies
            .as_ref()
            .map(|a| a.iter().any(|s| s == name))
            .unwrap_or(true)
    };

    let mut handles = Vec::new();
    let mut strategy_names = Vec::new();

    // Spawn each strategy as a blocking task.
    if should_run("vector") {
        let db = Arc::clone(&db);
        let q = q.clone();
        strategy_names.push("vector".to_string());
        handles.push(tokio::task::spawn_blocking(move || {
            let mut items = Vec::new();
            if let Ok(hits) = db.similar_to(&q, top_k) {
                for (record, score) in hits {
                    items.push(record_to_value(&record, score, "vector"));
                }
            }
            StrategyResult {
                name: "vector".to_string(),
                items,
            }
        }));
    }

    if should_run("fts") {
        let db = Arc::clone(&db);
        let q = q.clone();
        strategy_names.push("fts".to_string());
        handles.push(tokio::task::spawn_blocking(move || {
            let mut items = Vec::new();
            if let Ok(hits) = db.search_text(&q, top_k) {
                for (record, score) in hits {
                    items.push(record_to_value(&record, score, "fts"));
                }
            }
            StrategyResult {
                name: "fts".to_string(),
                items,
            }
        }));
    }

    if should_run("graph") {
        let db = Arc::clone(&db);
        let q = q.clone();
        strategy_names.push("graph".to_string());
        handles.push(tokio::task::spawn_blocking(move || {
            let mut items = Vec::new();
            let seeds = find_seed_records(&db, &q, 3);
            let mut seen = std::collections::HashSet::new();
            for (seed, score) in &seeds {
                if seen.insert(seed.id.to_string()) {
                    items.push(record_to_value(seed, *score, "graph_seed"));
                }
                if let Ok(neighbors) = db.neighbors(&seed.id, None, Direction::Both) {
                    for n in neighbors {
                        if seen.insert(n.id.to_string()) {
                            items.push(record_to_value(&n, score * 0.8, "graph_neighbor"));
                        }
                    }
                }
                if items.len() >= top_k {
                    break;
                }
            }
            items.truncate(top_k);
            StrategyResult {
                name: "graph".to_string(),
                items,
            }
        }));
    }

    if should_run("time") {
        let db = Arc::clone(&db);
        let q = q.clone();
        strategy_names.push("time".to_string());
        handles.push(tokio::task::spawn_blocking(move || {
            let duration = parse_duration_from_query(&q).unwrap_or(7 * 86400);
            let mut items = Vec::new();
            if let Ok(records) = db.since(None, duration) {
                for record in records.into_iter().take(top_k) {
                    items.push(record_to_value(&record, 1.0, "temporal"));
                }
            }
            StrategyResult {
                name: "time".to_string(),
                items,
            }
        }));
    }

    // Collect all results.
    let mut strategy_results: Vec<StrategyResult> = Vec::new();
    for handle in handles {
        if let Ok(result) = handle.await {
            strategy_results.push(result);
        }
    }

    // Apply Reciprocal Rank Fusion.
    let fused = reciprocal_rank_fusion(&strategy_results, top_k);

    let strategies_used: Vec<String> = strategy_results
        .iter()
        .filter(|s| !s.items.is_empty())
        .map(|s| s.name.clone())
        .collect();

    let tokens: usize = fused.iter().map(token::estimate_json_tokens).sum();

    Ok(AskResult {
        intent: QueryIntent::Combined,
        query: query.to_string(),
        results: fused,
        tokens,
        strategies_used,
        plan: None,
    })
}

/// Reciprocal Rank Fusion: merge ranked lists from multiple strategies.
///
/// For each item across all strategy lists, its RRF score is:
///   `sum over strategies of 1 / (k + rank_in_strategy)`
/// where `k` is a constant (default 60).
fn reciprocal_rank_fusion(strategy_results: &[StrategyResult], top_k: usize) -> Vec<Value> {
    const K: f64 = 60.0;
    let mut scores: std::collections::HashMap<String, (f64, Value)> =
        std::collections::HashMap::new();

    for strategy in strategy_results {
        for (rank, item) in strategy.items.iter().enumerate() {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rrf_score = 1.0 / (K + rank as f64 + 1.0);

            let entry = scores.entry(id).or_insert((0.0, item.clone()));
            entry.0 += rrf_score;
        }
    }

    let mut fused: Vec<(f64, Value)> = scores.into_values().collect();
    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);

    fused
        .into_iter()
        .map(|(rrf_score, mut item)| {
            if let Some(o) = item.as_object_mut() {
                o.insert(
                    "rrf_score".to_string(),
                    json!((rrf_score * 10000.0).round() / 10000.0),
                );
            }
            item
        })
        .collect()
}

// ── Strategy Runners ────────────────────────────────────────────────

fn run_vector(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    if let Ok(hits) = db.similar_to(query, top_k) {
        strategies.push("vector".to_string());
        for (record, score) in hits {
            results.push(record_to_value(&record, score, "vector"));
        }
    }
}

fn run_text_search(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    if let Ok(hits) = db.search_text(query, top_k) {
        strategies.push("fts".to_string());
        for (record, score) in hits {
            results.push(record_to_value(&record, score, "fts"));
        }
    }
}

fn run_temporal(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    let duration = parse_duration_from_query(query).unwrap_or(7 * 86400);

    if let Ok(records) = db.since(None, duration) {
        strategies.push("temporal".to_string());
        for record in records.into_iter().take(top_k) {
            results.push(record_to_value(&record, 1.0, "temporal"));
        }
    } else {
        // Fallback: list all records and filter by created_at.
        fallback_text_or_vector(db, query, top_k, results, strategies);
    }
}

fn run_graph_traversal(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    // Step 1: Find the entity the user is asking about via vector or FTS.
    let seed_records = find_seed_records(db, query, 3);

    if seed_records.is_empty() {
        // No seeds found — fall back to combined.
        fallback_text_or_vector(db, query, top_k, results, strategies);
        return;
    }

    strategies.push("graph".to_string());

    // Step 2: For each seed, get neighbors.
    let mut seen = std::collections::HashSet::new();
    for (seed, seed_score) in &seed_records {
        // Include the seed itself.
        if seen.insert(seed.id.to_string()) {
            results.push(record_to_value(seed, *seed_score, "graph_seed"));
        }
        if let Ok(neighbors) = db.neighbors(&seed.id, None, Direction::Both) {
            for neighbor in neighbors {
                if seen.insert(neighbor.id.to_string()) {
                    results.push(record_to_value(
                        &neighbor,
                        seed_score * 0.8,
                        "graph_neighbor",
                    ));
                }
            }
        }
        if results.len() >= top_k {
            break;
        }
    }

    results.truncate(top_k);
}

fn run_causality(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    // Try FTS first for causality queries (keyword-rich).
    if let Ok(hits) = db.search_text(query, top_k) {
        if !hits.is_empty() {
            strategies.push("fts".to_string());
            for (record, score) in &hits {
                results.push(record_to_value(record, *score, "fts"));
            }
        }
    }

    // Then traverse causal graph edges from seed records.
    let seeds = find_seed_records(db, query, 2);
    for (seed, _score) in &seeds {
        for edge_type in &["decided_by", "caused_by", "led_to", "supersedes"] {
            if let Ok(neighbors) = db.neighbors(&seed.id, Some(edge_type), Direction::Both) {
                if !neighbors.is_empty() {
                    strategies.push(format!("graph:{edge_type}"));
                    for neighbor in neighbors {
                        results.push(record_to_value(
                            &neighbor,
                            0.7,
                            &format!("graph:{edge_type}"),
                        ));
                    }
                }
            }
        }
        if results.len() >= top_k {
            break;
        }
    }

    // If nothing found yet, fall back to vector.
    if results.is_empty() {
        run_vector(db, query, top_k, results, strategies);
    }

    results.truncate(top_k);
}

fn run_rule_lookup(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    // Use the rules module to list all rules and filter by query terms.
    if let Ok(rules) = crate::rules::list_rules(db) {
        if !rules.is_empty() {
            strategies.push("rules".to_string());
            let query_lower = query.to_lowercase();
            let terms: Vec<&str> = query_lower.split_whitespace().collect();

            for rule in &rules {
                let text = format!("{} {}", rule.key, rule.rule).to_lowercase();
                let matches = terms.iter().filter(|t| text.contains(**t)).count();
                if matches > 0 {
                    let score = matches as f32 / terms.len() as f32;
                    let tokens = token::estimate_tokens(&rule.rule);
                    results.push(json!({
                        "id": rule.key,
                        "table": crate::rules::TABLE_RULES,
                        "score": (score * 1000.0).round() / 1000.0,
                        "summary": rule.rule,
                        "tokens": tokens,
                        "strategy": "rule",
                        "source": rule.source,
                    }));
                }
            }
        }
    }

    // Also try FTS and vector as supplementary.
    if results.len() < top_k {
        fallback_text_or_vector(db, query, top_k - results.len(), results, strategies);
    }

    sort_by_score(results);
    results.truncate(top_k);
}

fn run_combined(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    let mut vector_results = Vec::new();
    run_vector(db, query, top_k, &mut vector_results, strategies);

    let mut fts_results = Vec::new();
    run_text_search(db, query, top_k, &mut fts_results, strategies);

    // Merge: deduplicate by id, keep highest score per id.
    let mut best: std::collections::HashMap<String, (f64, Value)> =
        std::collections::HashMap::new();

    for item in vector_results.into_iter().chain(fts_results) {
        let id = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let score = item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let entry = best.entry(id).or_insert((score, item.clone()));
        if score > entry.0 {
            *entry = (score, item);
        }
    }

    results.extend(best.into_values().map(|(_, v)| v));

    // Sort by score descending.
    sort_by_score(results);
    results.truncate(top_k);
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Find seed records for graph traversal by trying vector then FTS.
fn find_seed_records(db: &Axil, query: &str, limit: usize) -> Vec<(Record, f32)> {
    // Try vector first.
    if let Ok(hits) = db.similar_to(query, limit) {
        if !hits.is_empty() {
            return hits;
        }
    }
    // Fall back to FTS.
    if let Ok(hits) = db.search_text(query, limit) {
        return hits;
    }
    Vec::new()
}

/// Fallback helper: try FTS, then vector.
fn fallback_text_or_vector(
    db: &Axil,
    query: &str,
    top_k: usize,
    results: &mut Vec<Value>,
    strategies: &mut Vec<String>,
) {
    if let Ok(hits) = db.search_text(query, top_k) {
        if !hits.is_empty() {
            strategies.push("fts_fallback".to_string());
            for (record, score) in hits {
                results.push(record_to_value(&record, score, "fts_fallback"));
            }
            return;
        }
    }
    if let Ok(hits) = db.similar_to(query, top_k) {
        strategies.push("vector_fallback".to_string());
        for (record, score) in hits {
            results.push(record_to_value(&record, score, "vector_fallback"));
        }
    }
}

/// Convert a `Record` + score into a compact JSON value for the result set.
fn record_to_value(record: &Record, score: f32, strategy: &str) -> Value {
    let summary = record
        .data
        .get("summary")
        .and_then(|v| v.as_str())
        .or_else(|| record.data.get("name").and_then(|v| v.as_str()))
        .or_else(|| record.data.get("title").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();

    let tokens = token::estimate_json_tokens(&record.data);

    json!({
        "id": record.id.to_string(),
        "table": record.table,
        "score": (score * 1000.0).round() / 1000.0,
        "summary": summary,
        "tokens": tokens,
        "strategy": strategy,
        "created_at": record.created_at.to_rfc3339(),
    })
}

/// Convert a `RecallResult` into a compact JSON value for the result set.
fn recall_result_to_value(r: &RecallResult) -> Value {
    json!({
        "id": r.id,
        "score": r.score,
        "summary": r.summary,
        "source": r.source,
        "path": r.path,
        "kind": r.kind,
        "tokens": r.tokens,
    })
}

/// Sort a result set by score descending.
fn sort_by_score(results: &mut [Value]) {
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_intent ───────────────────────────────────────────

    #[test]
    fn intent_vector_search() {
        assert_eq!(
            detect_intent("find something similar to auth"),
            QueryIntent::VectorSearch
        );
        assert_eq!(
            detect_intent("related to the login flow"),
            QueryIntent::VectorSearch
        );
        assert_eq!(
            detect_intent("what resembles the cache layer"),
            QueryIntent::VectorSearch
        );
    }

    #[test]
    fn intent_graph_traversal() {
        assert_eq!(
            detect_intent("what depends on axil-core"),
            QueryIntent::GraphTraversal
        );
        assert_eq!(
            detect_intent("what uses the Record type"),
            QueryIntent::GraphTraversal
        );
        assert_eq!(
            detect_intent("files imported by main.rs"),
            QueryIntent::GraphTraversal
        );
        assert_eq!(
            detect_intent("modules affected by this change"),
            QueryIntent::GraphTraversal
        );
    }

    #[test]
    fn intent_causality() {
        assert_eq!(
            detect_intent("why did we switch to redb"),
            QueryIntent::Causality
        );
        assert_eq!(
            detect_intent("what led to the plugin redesign"),
            QueryIntent::Causality
        );
        assert_eq!(
            detect_intent("decision about the storage engine"),
            QueryIntent::Causality
        );
        assert_eq!(
            detect_intent("reason for using ULID"),
            QueryIntent::Causality
        );
    }

    #[test]
    fn intent_temporal() {
        assert_eq!(
            detect_intent("what changed since yesterday"),
            QueryIntent::Temporal
        );
        assert_eq!(
            detect_intent("show records from last 3 days"),
            QueryIntent::Temporal
        );
        assert_eq!(
            detect_intent("anything new recently"),
            QueryIntent::Temporal
        );
        assert_eq!(detect_intent("files changed today"), QueryIntent::Temporal);
    }

    #[test]
    fn intent_rule_lookup() {
        assert_eq!(
            detect_intent("should I use anyhow or thiserror"),
            QueryIntent::RuleLookup
        );
        assert_eq!(
            detect_intent("we always use &str in function params"),
            QueryIntent::RuleLookup
        );
        assert_eq!(
            detect_intent("convention for error types"),
            QueryIntent::RuleLookup
        );
        assert_eq!(
            detect_intent("never do X in this project"),
            QueryIntent::RuleLookup
        );
    }

    #[test]
    fn intent_text_search() {
        assert_eq!(
            detect_intent("search for InvalidQuery"),
            QueryIntent::TextSearch
        );
        assert_eq!(
            detect_intent("find the string TODO"),
            QueryIntent::TextSearch
        );
        assert_eq!(detect_intent("grep for panics"), QueryIntent::TextSearch);
        assert_eq!(
            detect_intent("exact error message connection refused"),
            QueryIntent::TextSearch
        );
    }

    #[test]
    fn intent_combined_fallback() {
        assert_eq!(
            detect_intent("how does the query engine work"),
            QueryIntent::Combined
        );
        assert_eq!(
            detect_intent("tell me about authentication"),
            QueryIntent::Combined
        );
    }

    // ── parse_duration_from_query ───────────────────────────────

    #[test]
    fn duration_last_n_days() {
        assert_eq!(parse_duration_from_query("last 3 days"), Some(3 * 86400));
        assert_eq!(
            parse_duration_from_query("show me the last 7 days of activity"),
            Some(7 * 86400)
        );
    }

    #[test]
    fn duration_last_n_hours() {
        assert_eq!(parse_duration_from_query("last 2 hours"), Some(2 * 3600));
        assert_eq!(parse_duration_from_query("past 1 hour"), Some(3600));
    }

    #[test]
    fn duration_last_n_weeks() {
        assert_eq!(
            parse_duration_from_query("last 2 weeks"),
            Some(2 * 7 * 86400)
        );
    }

    #[test]
    fn duration_last_unit_implicit_one() {
        assert_eq!(parse_duration_from_query("last hour"), Some(3600));
        assert_eq!(parse_duration_from_query("last day"), Some(86400));
        assert_eq!(parse_duration_from_query("past week"), Some(7 * 86400));
    }

    #[test]
    fn duration_yesterday() {
        assert_eq!(parse_duration_from_query("since yesterday"), Some(86400));
    }

    #[test]
    fn duration_recently() {
        assert_eq!(
            parse_duration_from_query("what changed recently"),
            Some(7 * 86400)
        );
    }

    #[test]
    fn duration_today() {
        assert_eq!(parse_duration_from_query("anything new today"), Some(86400));
    }

    #[test]
    fn duration_no_match() {
        assert_eq!(parse_duration_from_query("how does auth work"), None);
    }

    #[test]
    fn duration_last_n_minutes() {
        assert_eq!(parse_duration_from_query("last 30 minutes"), Some(30 * 60));
    }

    #[test]
    fn duration_last_month() {
        assert_eq!(parse_duration_from_query("last month"), Some(30 * 86400));
    }

    // ── RRF tests ──────────────────────────────────────────────

    #[test]
    fn rrf_single_strategy() {
        let results = vec![StrategyResult {
            name: "vector".to_string(),
            items: vec![
                json!({"id": "a", "score": 0.9}),
                json!({"id": "b", "score": 0.7}),
            ],
        }];
        let fused = reciprocal_rank_fusion(&results, 5);
        assert_eq!(fused.len(), 2);
        // First item should have higher RRF score.
        let s0 = fused[0]["rrf_score"].as_f64().unwrap();
        let s1 = fused[1]["rrf_score"].as_f64().unwrap();
        assert!(s0 > s1);
    }

    #[test]
    fn rrf_two_strategies_boost_overlap() {
        let results = vec![
            StrategyResult {
                name: "vector".to_string(),
                items: vec![
                    json!({"id": "a", "score": 0.9}),
                    json!({"id": "b", "score": 0.7}),
                ],
            },
            StrategyResult {
                name: "fts".to_string(),
                items: vec![
                    json!({"id": "b", "score": 0.95}),
                    json!({"id": "c", "score": 0.6}),
                ],
            },
        ];
        let fused = reciprocal_rank_fusion(&results, 5);
        assert_eq!(fused.len(), 3);
        // "b" appears in both lists, so it should be ranked highest by RRF.
        assert_eq!(fused[0]["id"], "b");
    }

    #[test]
    fn rrf_respects_top_k() {
        let results = vec![StrategyResult {
            name: "vector".to_string(),
            items: (0..10)
                .map(|i| json!({"id": format!("item_{i}"), "score": 0.5}))
                .collect(),
        }];
        let fused = reciprocal_rank_fusion(&results, 3);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_empty_strategies() {
        let results: Vec<StrategyResult> = vec![];
        let fused = reciprocal_rank_fusion(&results, 5);
        assert!(fused.is_empty());
    }

    // ── Query planner tests ────────────────────────────────────

    #[test]
    fn plan_simple_vector() {
        let plan = plan_query("find something similar to auth");
        assert!(plan.steps.iter().any(|s| s.query_type == "vector"));
    }

    #[test]
    fn plan_multi_signal() {
        let plan = plan_query("what similar auth bugs have I fixed recently");
        let types: Vec<&str> = plan.steps.iter().map(|s| s.query_type.as_str()).collect();
        assert!(types.contains(&"vector"), "should have vector step");
        assert!(types.contains(&"time_filter"), "should have time step");
    }

    #[test]
    fn plan_causality_chain() {
        let plan = plan_query("what decisions led to the JWT middleware design");
        let types: Vec<&str> = plan.steps.iter().map(|s| s.query_type.as_str()).collect();
        assert!(types.contains(&"graph"), "should have graph step");
        // The graph step should reference a previous step.
        let graph_step = plan.steps.iter().find(|s| s.query_type == "graph").unwrap();
        assert!(graph_step.from.is_some() || graph_step.step == 1);
    }

    #[test]
    fn plan_fallback_combined() {
        let plan = plan_query("tell me about the database");
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].query_type, "combined");
    }

    #[test]
    fn plan_steps_are_sequential() {
        let plan = plan_query("what similar auth bugs have I fixed recently");
        for (i, step) in plan.steps.iter().enumerate() {
            assert_eq!(step.step, i + 1);
        }
    }

    #[test]
    fn plan_fts_with_graph_and_time() {
        let plan = plan_query("grep for the exact error connected to auth recently");
        let types: Vec<&str> = plan.steps.iter().map(|s| s.query_type.as_str()).collect();
        assert!(types.contains(&"fts"));
        assert!(types.contains(&"graph"));
        assert!(types.contains(&"time_filter"));
        // Graph should reference FTS results.
        let graph_step = plan.steps.iter().find(|s| s.query_type == "graph").unwrap();
        assert!(graph_step.from.is_some());
    }
}
