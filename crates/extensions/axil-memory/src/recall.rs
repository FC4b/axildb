//! Recency-weighted recall and cross-memory queries.
//!
//! The `recall` function combines vector similarity with recency scoring:
//! `final_score = alpha * similarity + (1 - alpha) * recency`
//!
//! The `remember` function searches all memory types and returns tagged results.

use chrono::Utc;
use serde::{Deserialize, Serialize};

use axil_core::{Axil, Record, Result};

use crate::types::MemoryType;

/// Options for recall and remember queries.
#[derive(Debug, Clone)]
pub struct RecallOptions {
    /// Number of results to return.
    pub top_k: usize,
    /// Recency weight: `alpha * similarity + (1 - alpha) * recency`.
    /// `None` means use the memory type's default.
    pub alpha: Option<f32>,
    /// Maximum tokens in the response (for budget-aware recall).
    /// `None` means no limit.
    pub max_tokens: Option<usize>,
    /// Include expired records.
    pub include_expired: bool,
    /// Include superseded records.
    pub include_superseded: bool,
    /// Decay window in seconds for recency calculation.
    /// Default: 30 days (2_592_000 seconds).
    pub decay_window_secs: f64,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            alpha: None,
            max_tokens: None,
            include_expired: false,
            include_superseded: false,
            decay_window_secs: 30.0 * 86400.0, // 30 days
        }
    }
}

/// A single result from a recall or remember query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRecord {
    /// The record.
    pub record: ScoredRecordData,
    /// Vector similarity score (0.0 to 1.0).
    pub similarity: f32,
    /// Recency score (0.0 to 1.0, 1.0 = just created).
    pub recency: f32,
    /// Final blended score.
    pub final_score: f32,
}

/// Serializable record data for scored results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRecordData {
    pub id: String,
    pub table: String,
    pub data: serde_json::Value,
    pub created_at: String,
}

impl From<&Record> for ScoredRecordData {
    fn from(r: &Record) -> Self {
        Self {
            id: r.id.to_string(),
            table: r.table.clone(),
            data: r.data.clone(),
            created_at: r.created_at.to_rfc3339(),
        }
    }
}

/// A tagged result from a cross-memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// Which memory type this result came from.
    pub memory_type: MemoryType,
    /// The scored record.
    #[serde(flatten)]
    pub scored: ScoredRecord,
    /// Estimated token count for this result.
    pub tokens: usize,
}

/// Perform recency-weighted recall on a specific table.
pub fn recall(
    db: &Axil,
    query: &str,
    table: &str,
    opts: &RecallOptions,
    memory_type: MemoryType,
) -> Result<Vec<ScoredRecord>> {
    if !db.has_vector_index() {
        return Ok(Vec::new());
    }

    let alpha = opts
        .alpha
        .unwrap_or_else(|| memory_type.default_alpha())
        .clamp(0.0, 1.0);

    // Fetch extra candidates for re-ranking.
    let fetch_k = opts.top_k.saturating_mul(3).max(20);
    let results = db.similar_to(query, fetch_k)?;

    let now = Utc::now();

    let mut scored: Vec<ScoredRecord> = results
        .into_iter()
        .filter(|(r, _)| r.table == table)
        .filter(|(r, _)| opts.include_expired || !crate::ttl::is_record_expired(r))
        .filter(|(r, _)| opts.include_superseded || !crate::ttl::is_record_superseded(r))
        .map(|(record, similarity)| {
            let age_secs = (now - record.created_at).num_seconds().max(0) as f64;
            let recency = (1.0 - (age_secs / opts.decay_window_secs).min(1.0)).max(0.0) as f32;
            let final_score = alpha * similarity + (1.0 - alpha) * recency;
            ScoredRecord {
                record: ScoredRecordData::from(&record),
                similarity,
                recency,
                final_score,
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(opts.top_k);

    Ok(scored)
}

/// Cross-memory query: searches all memory types, returns tagged results.
pub fn remember(db: &Axil, query: &str, opts: RecallOptions) -> Result<Vec<RecallResult>> {
    let mut all_results = Vec::new();

    // Use a larger per-type limit so the global merge has enough candidates.
    let num_types = MemoryType::all().len();
    let per_type_opts = RecallOptions {
        top_k: opts.top_k.saturating_mul(num_types).max(20),
        ..opts.clone()
    };

    for memory_type in MemoryType::all() {
        let table = memory_type.table_name();

        if db.count(table).unwrap_or(0) == 0 {
            continue;
        }

        let results = recall(db, query, table, &per_type_opts, *memory_type)?;

        for scored in results {
            let tokens = estimate_tokens(&scored.record.data);
            all_results.push(RecallResult {
                memory_type: *memory_type,
                scored,
                tokens,
            });
        }
    }

    // Sort all results by final_score descending.
    all_results.sort_by(|a, b| {
        b.scored
            .final_score
            .partial_cmp(&a.scored.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply top_k limit.
    all_results.truncate(opts.top_k);

    // Apply token budget if specified.
    if let Some(max_tokens) = opts.max_tokens {
        let mut total = 0usize;
        all_results.retain(|r| {
            if total.saturating_add(r.tokens) > max_tokens {
                return false;
            }
            total += r.tokens;
            true
        });
    }

    Ok(all_results)
}

/// Rough token estimate: ~4 chars per token for JSON.
fn estimate_tokens(data: &serde_json::Value) -> usize {
    let json_str = serde_json::to_string(data).unwrap_or_default();
    json_str.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_options_defaults() {
        let opts = RecallOptions::default();
        assert_eq!(opts.top_k, 5);
        assert!(opts.alpha.is_none());
        assert!(!opts.include_expired);
    }

    #[test]
    fn estimate_tokens_works() {
        let data = serde_json::json!({"summary": "Fixed the authentication timeout bug by increasing connection pool size"});
        let tokens = estimate_tokens(&data);
        assert!(tokens > 0);
        assert!(tokens < 100);
    }

    #[test]
    fn scored_record_serialization() {
        let sr = ScoredRecord {
            record: ScoredRecordData {
                id: "test".to_string(),
                table: "_episodes".to_string(),
                data: serde_json::json!({"summary": "test"}),
                created_at: "2026-01-01T00:00:00Z".to_string(),
            },
            similarity: 0.95,
            recency: 0.8,
            final_score: 0.9,
        };
        let json = serde_json::to_string(&sr).unwrap();
        assert!(json.contains("similarity"));
        assert!(json.contains("final_score"));
    }
}
