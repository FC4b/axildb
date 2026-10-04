//! Usage tracking and token analytics for agent sessions.
//!
//! Stores query logs in the `_analytics` table and provides
//! aggregated statistics over configurable time periods.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Result};

// ── Table name ──────────────────────────────────────────────────────

/// Reserved table for analytics records.
pub const TABLE_ANALYTICS: &str = "_analytics";

/// Label for the `tokens_served` total in [`get_analytics`]: each logged
/// count comes from [`crate::token::estimate_json_tokens`] over the results a
/// query returned, not from a tokenizer.
pub const TOKENS_SERVED_BASIS: &str = "estimate: ~4 chars/token over the JSON each query returned";

// ── Types ───────────────────────────────────────────────────────────

/// A single analytics record stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsRecord {
    /// Type of query: "vector", "graph", "fts", "time", "rules", etc.
    pub query_type: String,
    /// The raw query string.
    pub query: String,
    /// Number of results returned.
    pub results_count: usize,
    /// Estimated tokens served to the agent.
    pub tokens_served: usize,
    /// ISO-8601 timestamp.
    pub timestamp: String,
}

// ── Public API ──────────────────────────────────────────────────────

/// Log a query event to the analytics table.
pub fn log_query(
    db: &Axil,
    query_type: &str,
    query: &str,
    results_count: usize,
    tokens_served: usize,
) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    db.insert(
        TABLE_ANALYTICS,
        json!({
            "query_type": query_type,
            "query": query,
            "results_count": results_count,
            "tokens_served": tokens_served,
            "timestamp": now,
        }),
    )?;
    Ok(())
}

/// Return aggregated analytics for the given period.
///
/// The returned JSON includes:
/// - `period` — the number of days covered
/// - `total_queries` — count of logged queries in the period
/// - `tokens_served` — total tokens served, an estimate (see
///   [`TOKENS_SERVED_BASIS`])
/// - `tokens_served_basis` — how `tokens_served` was counted
/// - `most_recalled` — top 5 query terms by frequency
/// - `query_types` — breakdown by query type
/// - `stale_hits` — queries that returned zero results
///
/// There is deliberately no "tokens saved" figure: a query's raw baseline
/// (what the agent would have read without Axil) is never measured, so any
/// multiplier over `tokens_served` would be invented.
pub fn get_analytics(db: &Axil, period_days: u64) -> Result<Value> {
    let all = db.list(TABLE_ANALYTICS)?;

    let cutoff = Utc::now() - chrono::Duration::days(period_days as i64);
    let cutoff_str = cutoff.to_rfc3339();

    // Filter to records within the period.
    let records: Vec<&axil_core::Record> = all
        .iter()
        .filter(|r| {
            r.data
                .get("timestamp")
                .and_then(|v| v.as_str())
                .map(|ts| ts >= cutoff_str.as_str())
                .unwrap_or(false)
        })
        .collect();

    let total_queries = records.len();

    let mut tokens_served: usize = 0;
    let mut stale_hits: usize = 0;
    let mut query_types: HashMap<String, usize> = HashMap::new();
    let mut term_freq: HashMap<String, usize> = HashMap::new();

    for record in &records {
        let data = &record.data;

        // Accumulate tokens served.
        tokens_served += data
            .get("tokens_served")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Count stale hits (zero results).
        let count = data
            .get("results_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if count == 0 {
            stale_hits += 1;
        }

        // Query type breakdown.
        if let Some(qt) = data.get("query_type").and_then(|v| v.as_str()) {
            *query_types.entry(qt.to_string()).or_insert(0) += 1;
        }

        // Term frequency.
        if let Some(q) = data.get("query").and_then(|v| v.as_str()) {
            for term in q.split_whitespace() {
                let t = term.to_lowercase();
                if t.len() > 1 {
                    *term_freq.entry(t).or_insert(0) += 1;
                }
            }
        }
    }

    // Top 5 most recalled terms.
    let mut terms: Vec<(String, usize)> = term_freq.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1));
    terms.truncate(5);
    let most_recalled: Vec<Value> = terms
        .into_iter()
        .map(|(term, count)| json!({"term": term, "count": count}))
        .collect();

    Ok(json!({
        "period": period_days,
        "total_queries": total_queries,
        "tokens_served": tokens_served,
        "tokens_served_basis": TOKENS_SERVED_BASIS,
        "most_recalled": most_recalled,
        "query_types": query_types,
        "stale_hits": stale_hits,
    }))
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn log_and_retrieve_analytics() {
        let (db, _dir) = open_temp_db();

        log_query(&db, "vector", "auth timeout", 3, 120).unwrap();
        log_query(&db, "fts", "login error", 5, 200).unwrap();
        log_query(&db, "graph", "user relations", 0, 0).unwrap();

        let analytics = get_analytics(&db, 1).unwrap();

        assert_eq!(analytics["total_queries"], 3);
        assert_eq!(analytics["tokens_served"], 320);
        assert_eq!(analytics["stale_hits"], 1);
    }

    #[test]
    fn empty_analytics() {
        let (db, _dir) = open_temp_db();

        let analytics = get_analytics(&db, 7).unwrap();

        assert_eq!(analytics["total_queries"], 0);
        assert_eq!(analytics["tokens_served"], 0);
        assert_eq!(analytics["stale_hits"], 0);
    }

    #[test]
    fn query_type_breakdown() {
        let (db, _dir) = open_temp_db();

        log_query(&db, "vector", "search one", 2, 50).unwrap();
        log_query(&db, "vector", "search two", 1, 30).unwrap();
        log_query(&db, "fts", "text query", 4, 100).unwrap();

        let analytics = get_analytics(&db, 1).unwrap();
        let types = analytics["query_types"].as_object().unwrap();

        assert_eq!(types["vector"], 2);
        assert_eq!(types["fts"], 1);
    }

    #[test]
    fn most_recalled_terms() {
        let (db, _dir) = open_temp_db();

        log_query(&db, "vector", "auth timeout", 2, 50).unwrap();
        log_query(&db, "vector", "auth login", 3, 80).unwrap();
        log_query(&db, "fts", "auth middleware", 1, 40).unwrap();

        let analytics = get_analytics(&db, 1).unwrap();
        let most = analytics["most_recalled"].as_array().unwrap();

        // "auth" should be the top term (3 occurrences).
        assert!(!most.is_empty());
        assert_eq!(most[0]["term"], "auth");
        assert_eq!(most[0]["count"], 3);
    }

    #[test]
    fn served_tokens_are_labeled_and_no_savings_figure_is_invented() {
        let (db, _dir) = open_temp_db();

        log_query(&db, "vector", "query", 5, 100).unwrap();

        let analytics = get_analytics(&db, 1).unwrap();
        assert_eq!(analytics["tokens_served"], 100);
        assert_eq!(analytics["tokens_served_basis"], TOKENS_SERVED_BASIS);
        assert!(TOKENS_SERVED_BASIS.starts_with("estimate"));
        // No raw baseline is measured, so no savings figure may appear.
        let keys: Vec<&String> = analytics.as_object().unwrap().keys().collect();
        assert!(
            keys.iter().all(|k| !k.contains("saved")),
            "unexpected savings field in {keys:?}"
        );
    }
}
