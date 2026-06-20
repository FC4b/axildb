//! Cross-encoder reranking stage for the Axil recall pipeline.
//!
//! inserts a learned (query, passage)
//! scorer between RRF fusion and token-budget trim. Replaces the prior
//! in-tree reranker module (formerly `axil_indexer::rerank`) which was wired
//! only via the CLI, never into [`axil_core::query::QueryBuilder`].
//!
//! Pipeline position:
//! ```text
//!   vector + FTS + graph + recency + feedback  →  RRF fusion  →
//!     RERANK (top-50 → top-10)  →  token-budget trim  →  return
//! ```
//!
//! Status (2026-05-16): **opt-in only.** was trimmed after four
//! benchmark sweeps (`01KPT2TDH6GDGZYBWK9BKWEYCC`, `01KPTFNP0WQJ0EDAK4SJWNW6KB`,
//! `01KPTA6AA0YRKYPFKXTKAX53PF`, `01KPTA769A3NCX1YHBJ3GGPJ35`) showed the
//! reranker did not clear the ≥+5 recall@K gate on LongMemEval-S. The crate
//! stays in tree behind the `rerank` Cargo feature for future re-evaluation
//! on different datasets (e.g. dep-docs in ). The original gate
//! threshold — ≥+5 recall@K over reranker-off, p95 ≤80ms — is the bar to
//! clear before any future default-on flip.
//!
//! ## Models
//!
//! - [`RerankModel::AnswerAiColbertSmall`] — answerdotai/answerai-colbert-small-v1,
//!   33M params, Apache-2.0, ONNX shipped. **Default** for new installs.
//! - [`RerankModel::MsMarcoMiniLm`] — cross-encoder/ms-marco-MiniLM-L-6-v2,
//!   22M, Apache-2.0. Backwards compat with the previous in-tree reranker.
//!
//! ## Features
//!
//! - default — no ONNX runtime; only the trait, config, model registry,
//!   and a [`NoOpReranker`] for callers that just want the interface.
//! - `onnx` — pulls `ort` + `tokenizers`; enables [`OnnxReranker`].

pub mod config;
pub mod download;
pub mod models;

#[cfg(feature = "onnx")]
mod runtime;

pub use config::RerankConfig;
pub use models::{RerankModel, RerankModelMeta};

use serde::{Deserialize, Serialize};

/// Outcome of scoring a single (query, passage) pair.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RerankScore {
    /// Position in the input list — preserved so callers can map back to
    /// their candidate records.
    pub index: usize,
    /// Cross-encoder score (higher = more relevant). Range depends on the
    /// model; for normalisation use [`Self::sigmoid`].
    pub score: f32,
}

impl RerankScore {
    /// Map raw logit to (0,1) via sigmoid. Useful when blending with
    /// already-normalised fused scores from RRF.
    pub fn sigmoid(&self) -> f32 {
        let s = self.score;
        1.0 / (1.0 + (-s).exp())
    }
}

/// A single candidate handed to the reranker. Borrowed so callers can
/// rerank without cloning every passage out of their result vec.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    /// Index in the caller's source array — written back into
    /// [`RerankScore::index`] so the caller can map results back.
    pub index: usize,
    /// Passage text to score against the query.
    pub passage: &'a str,
}

/// Reranking strategy. Implementations score (query, passages) and return
/// scores in input order. Sorting is the caller's responsibility — keeps
/// the trait pure so tests can assert order-stability.
pub trait Reranker: Send + Sync {
    /// Score every candidate against `query`. Output length == input length.
    fn score_batch(
        &self,
        query: &str,
        candidates: &[Candidate<'_>],
    ) -> Result<Vec<RerankScore>, RerankError>;

    /// Human-readable name, used in diagnostics + `axil bench`.
    fn name(&self) -> &str;
}

/// No-op reranker. Used when the crate is built without the `onnx` feature
/// or when the user disables reranking via config — passes scores through
/// as 0.0 so blending leaves the fused score untouched.
pub struct NoOpReranker;

impl Reranker for NoOpReranker {
    fn score_batch(
        &self,
        _query: &str,
        candidates: &[Candidate<'_>],
    ) -> Result<Vec<RerankScore>, RerankError> {
        Ok(candidates
            .iter()
            .map(|c| RerankScore {
                index: c.index,
                score: 0.0,
            })
            .collect())
    }

    fn name(&self) -> &str {
        "noop"
    }
}

#[cfg(feature = "onnx")]
pub use runtime::OnnxReranker;

// ── axil-core::Rerank bridge ──────────────────────────────────────────────
//
// Implements axil_core::query::Rerank for our Reranker trait so callers
// can wire any axil-rerank impl into [`axil_core::query::QueryBuilder`]
// via `.with_reranker(&adapter)`. Use [`bridge`] to obtain a
// `&dyn axil_core::query::Rerank` from any [`Reranker`].
pub use bridge::{bridge, RerankBridge};

mod bridge {
    use super::{apply, RerankConfig, Reranker};
    use axil_core::query::Rerank as CoreRerank;
    use axil_core::record::Record;
    use serde_json::Value;

    /// Adapter that wraps a [`Reranker`] and implements [`axil_core::query::Rerank`].
    pub struct RerankBridge<'r, R: Reranker + ?Sized> {
        inner: &'r R,
        cfg: RerankConfig,
    }

    impl<'r, R: Reranker + ?Sized> RerankBridge<'r, R> {
        pub fn new(reranker: &'r R) -> Self {
            Self {
                inner: reranker,
                cfg: RerankConfig::enabled_default(),
            }
        }

        pub fn with_config(mut self, cfg: RerankConfig) -> Self {
            self.cfg = cfg;
            self
        }
    }

    /// Free-function convenience: borrow `r` as a `dyn CoreRerank`.
    pub fn bridge<'r, R: Reranker + ?Sized>(r: &'r R) -> RerankBridge<'r, R> {
        RerankBridge::new(r)
    }

    impl<'r, R: Reranker + ?Sized> CoreRerank for RerankBridge<'r, R> {
        fn rerank_records(
            &self,
            query: &str,
            records: &mut Vec<Record>,
            top_k_in: usize,
            top_k_out: usize,
        ) -> Result<usize, String> {
            // Project Records → JSON, run the standard apply() pipeline,
            // then project back. Avoids duplicating the score-blend logic.
            let mut as_json: Vec<Value> = records
                .iter()
                .map(|r| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("id".into(), Value::String(r.id.to_string()));
                    obj.insert("table".into(), Value::String(r.table.clone()));
                    obj.insert("data".into(), r.data.clone());
                    Value::Object(obj)
                })
                .collect();

            let mut cfg = self.cfg.clone();
            cfg.top_k_in = top_k_in.min(records.len());
            cfg.top_k_out = top_k_out;
            cfg.enabled = true;

            let report =
                apply(self.inner, query, &mut as_json, &cfg).map_err(|e| format!("rerank: {e}"))?;

            // Build new Records list in reranked order. We rebuild
            // from the original `records` via id lookup to preserve
            // exact field state (timestamps, importance, etc.).
            use std::collections::HashMap;
            let by_id: HashMap<String, Record> =
                records.drain(..).map(|r| (r.id.to_string(), r)).collect();
            let mut rebuilt = Vec::with_capacity(as_json.len());
            for v in &as_json {
                if let Some(id) = v.get("id").and_then(|x| x.as_str()) {
                    if let Some(rec) = by_id.get(id) {
                        rebuilt.push(rec.clone());
                    }
                }
            }
            *records = rebuilt;
            Ok(report.scored)
        }

        fn name(&self) -> &str {
            self.inner.name()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::NoOpReranker;
        use axil_core::query::Rerank;
        use axil_core::record::Record;
        use serde_json::json;

        #[test]
        fn bridge_reranks_records_via_core_trait() {
            let r = NoOpReranker;
            let adapter = bridge(&r);
            let mut records = vec![
                Record::new("decisions", json!({"summary":"first"})),
                Record::new("decisions", json!({"summary":"second"})),
            ];
            let n = (&adapter as &dyn Rerank)
                .rerank_records("q", &mut records, 2, 2)
                .unwrap();
            assert_eq!(n, 2);
            assert_eq!(records.len(), 2);
        }
    }
}

/// Errors surfaced by the reranker layer.
#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("model not found: {0}. Run `axil rerank download {0}` first.")]
    ModelMissing(String),

    #[error("tokenizer error: {0}")]
    Tokenize(String),

    #[error("ONNX inference error: {0}")]
    Inference(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl From<String> for RerankError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

/// Apply the reranker to a fused result list in-place. Splits the list at
/// `cfg.top_k_in`, reranks the prefix, and stitches it back in front of
/// the unranked tail. Each reranked entry gets a `rerank_score` field
/// injected for downstream observability + score blending.
///
/// Score blending: the final order is determined by
/// `cfg.weight * sigmoid(rerank_score) + (1 - cfg.weight) * fused_score`,
/// where `fused_score` is read from the existing `_score` field if present
/// (otherwise 0.0). Callers that want pure reranker order set
/// `cfg.weight = 1.0`.
pub fn apply<R: Reranker + ?Sized>(
    reranker: &R,
    query: &str,
    results: &mut Vec<serde_json::Value>,
    cfg: &RerankConfig,
) -> Result<RerankReport, RerankError> {
    let started = std::time::Instant::now();
    let mut report = RerankReport::default();
    if !cfg.enabled || results.is_empty() {
        return Ok(report);
    }

    let rerank_count = results.len().min(cfg.top_k_in);
    if rerank_count == 0 {
        return Ok(report);
    }

    let passages: Vec<String> = results[..rerank_count]
        .iter()
        .map(|v| extract_passage(v))
        .collect();

    let candidates: Vec<Candidate<'_>> = passages
        .iter()
        .enumerate()
        .map(|(i, p)| Candidate {
            index: i,
            passage: p.as_str(),
        })
        .collect();

    let scores = reranker.score_batch(query, &candidates)?;
    report.scored = scores.len();

    // Blend rerank with fused score.
    let mut blended: Vec<(usize, f32, f32)> = scores
        .iter()
        .map(|rs| {
            let fused = results[rs.index]
                .get("_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let r_norm = rs.sigmoid();
            let final_score = cfg.weight * r_norm + (1.0 - cfg.weight) * fused;
            (rs.index, rs.score, final_score)
        })
        .collect();
    blended.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Stitch reranked prefix + tail.
    let tail: Vec<serde_json::Value> = results.drain(rerank_count..).collect();
    let mut head: Vec<serde_json::Value> = blended
        .iter()
        .map(|(idx, raw, blended_score)| {
            let mut v = results[*idx].clone();
            if let Some(obj) = v.as_object_mut() {
                obj.insert("rerank_score".to_string(), serde_json::json!(*raw as f64));
                obj.insert(
                    "rerank_blended_score".to_string(),
                    serde_json::json!(*blended_score as f64),
                );
            }
            v
        })
        .collect();
    let truncated_to = head.len().min(cfg.top_k_out);
    head.truncate(truncated_to);
    head.extend(tail);
    *results = head;

    report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(report)
}

/// Pull a text passage out of a result row. Falls back across the common
/// field names used by axil-core record shapes (summary > content > text >
/// data.summary > flat data).
fn extract_passage(v: &serde_json::Value) -> String {
    let direct = v
        .get("summary")
        .or_else(|| v.get("content"))
        .or_else(|| v.get("text"))
        .and_then(|s| s.as_str());
    if let Some(s) = direct {
        return s.to_string();
    }
    if let Some(data) = v.get("data") {
        if let Some(s) = data
            .get("summary")
            .or_else(|| data.get("content"))
            .or_else(|| data.get("text"))
            .and_then(|s| s.as_str())
        {
            return s.to_string();
        }
        if data.is_string() {
            return data.as_str().unwrap_or("").to_string();
        }
    }
    String::new()
}

/// Diagnostics surfaced from [`apply`] — observability for the gate.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct RerankReport {
    pub scored: usize,
    pub elapsed_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn apply_noop_when_disabled() {
        let r = NoOpReranker;
        let cfg = RerankConfig::default();
        let mut results = vec![json!({"id": "a", "_score": 0.9, "summary": "foo"})];
        let before = results.clone();
        apply(&r, "q", &mut results, &cfg).unwrap();
        assert_eq!(results, before);
    }

    #[test]
    fn apply_noop_keeps_input_order_when_enabled() {
        // NoOpReranker scores everything 0.0 — final order should be
        // determined by fused score because rerank contribution is constant.
        let r = NoOpReranker;
        let cfg = RerankConfig {
            enabled: true,
            weight: 0.0, // force pure fused-score ordering
            top_k_in: 10,
            top_k_out: 10,
            ..Default::default()
        };
        let mut results = vec![
            json!({"id": "a", "_score": 0.1, "summary": "x"}),
            json!({"id": "b", "_score": 0.9, "summary": "y"}),
        ];
        apply(&r, "q", &mut results, &cfg).unwrap();
        assert_eq!(results[0]["id"], "b");
        assert_eq!(results[1]["id"], "a");
    }

    #[test]
    fn apply_emits_rerank_score_fields() {
        let r = NoOpReranker;
        let cfg = RerankConfig {
            enabled: true,
            ..Default::default()
        };
        let mut results = vec![json!({"id": "a", "_score": 0.5, "summary": "x"})];
        apply(&r, "q", &mut results, &cfg).unwrap();
        assert!(results[0].get("rerank_score").is_some());
        assert!(results[0].get("rerank_blended_score").is_some());
    }

    #[test]
    fn extract_passage_falls_back_through_field_names() {
        assert_eq!(extract_passage(&json!({"summary":"a"})), "a");
        assert_eq!(extract_passage(&json!({"content":"b"})), "b");
        assert_eq!(extract_passage(&json!({"text":"c"})), "c");
        assert_eq!(extract_passage(&json!({"data":{"summary":"d"}})), "d");
        assert_eq!(extract_passage(&json!({"data":"e"})), "e");
        assert_eq!(extract_passage(&json!({"id":"x"})), "");
    }

    #[test]
    fn rerank_score_sigmoid_bounds() {
        let r = RerankScore {
            index: 0,
            score: 100.0,
        };
        assert!(r.sigmoid() > 0.99);
        let r = RerankScore {
            index: 0,
            score: -100.0,
        };
        assert!(r.sigmoid() < 0.01);
    }
}
