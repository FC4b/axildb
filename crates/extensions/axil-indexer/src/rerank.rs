//! Cross-encoder reranking for improved retrieval accuracy.
//!
//! After initial retrieval (vector, FTS, etc.), reranks the top-K candidates
//! using a cross-encoder model that scores (query, passage) pairs jointly.
//!
//! Requires the `rerank` feature flag and a compatible ONNX cross-encoder model.
//! Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22MB ONNX).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "rerank")]
use ort::value::Tensor;
#[cfg(feature = "rerank")]
use std::cell::RefCell;
#[cfg(feature = "rerank")]
use std::collections::HashMap;

const MAX_RERANK_TOKENS: usize = 512;

/// Char budget per passage window. ~450 BPE tokens for english text, leaving
/// headroom for the query + special tokens under `MAX_RERANK_TOKENS`.
const WINDOW_MAX_BYTES: usize = 1800;
const WINDOW_OVERLAP_BYTES: usize = 200;
/// Passages shorter than this are scored as a single window (skip chunking).
const WINDOW_SKIP_BELOW_BYTES: usize = 2000;

#[cfg(feature = "rerank")]
struct RerankRuntime {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
    /// Whether the loaded model declares a `token_type_ids` input.
    /// BERT-family models (MS-MARCO MiniLM) do; RoBERTa-family (BGE reranker,
    /// XLM-R) do not. We must only bind inputs the model declares — ONNX
    /// rejects extras with "Invalid input name".
    accepts_token_type_ids: bool,
}

#[cfg(feature = "rerank")]
thread_local! {
    static RERANK_RUNTIMES: RefCell<HashMap<String, RerankRuntime>> = RefCell::new(HashMap::new());
}

/// Configuration for the reranker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// Enable cross-encoder reranking.
    pub enabled: bool,
    /// Path to the ONNX cross-encoder model file.
    pub model: String,
    /// Only rerank the top K candidates (reduces latency).
    pub top_k_rerank: usize,
    /// Share of the cross-encoder in the final order, in `[0, 1]`. Below
    /// 1.0 each scored candidate is ranked by
    /// `w · cross-encoder + (1 − w) · incoming score`, both min-max
    /// normalized over the scored candidates; 0.0 keeps the incoming order
    /// and 1.0 replaces it with the cross-encoder's. The default, 0.3, ranked
    /// best on the dogfood eval and on a LongMemEval sample, where the
    /// cross-encoder's order alone ranked below no rerank at all
    /// (`benchmarks/results/dogfood-recall-2026-10-14-rerank.json`,
    /// `benchmarks/results/longmemeval-s60-rerank-2026-10-14.json`).
    pub blend_weight: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "models/cross-encoder.onnx".to_string(),
            top_k_rerank: 20,
            blend_weight: 0.3,
        }
    }
}

/// A scored (query, passage) pair from the cross-encoder.
#[derive(Debug, Clone)]
pub struct RerankScore {
    pub index: usize,
    pub score: f32,
}

fn truncate_encoding_triplet(
    ids: &mut Vec<i64>,
    attention: &mut Vec<i64>,
    type_ids: &mut Vec<i64>,
    max_tokens: usize,
) {
    let keep = ids
        .len()
        .min(attention.len())
        .min(type_ids.len())
        .min(max_tokens);
    ids.truncate(keep);
    attention.truncate(keep);
    type_ids.truncate(keep);
}

/// The text the cross-encoder scores for one candidate: a top-level
/// `summary`, else `data` — a string, or a record object (what the CLI
/// passes), read through [`axil_core::util::searchable_text`].
#[cfg(any(feature = "rerank", test))]
fn passage_text(candidate: &Value) -> String {
    if let Some(summary) = candidate.get("summary").and_then(Value::as_str) {
        if !summary.trim().is_empty() {
            return summary.to_string();
        }
    }
    match candidate.get("data") {
        Some(Value::String(text)) => text.clone(),
        Some(data @ Value::Object(_)) => axil_core::util::searchable_text(data),
        _ => String::new(),
    }
}

/// The score each candidate arrived with (its `score` field), which a blend
/// weighs the cross-encoder against. When any candidate lacks one, the
/// incoming order stands in for it: position `i` scores `-i`.
fn incoming_scores(candidates: &[Value]) -> Vec<f32> {
    let scores: Option<Vec<f32>> = candidates
        .iter()
        .map(|c| c.get("score").and_then(Value::as_f64).map(|s| s as f32))
        .collect();
    scores.unwrap_or_else(|| (0..candidates.len()).map(|i| -(i as f32)).collect())
}

/// Min-max normalize into `[0, 1]`. Equal values all map to 0, leaving the
/// order to the other signal.
fn min_max(values: &[f32]) -> Vec<f32> {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = hi - lo;
    values
        .iter()
        .map(|v| if span > 0.0 { (v - lo) / span } else { 0.0 })
        .collect()
}

/// Order the scored candidates best-first, each with its sort key: the
/// cross-encoder score when `blend_weight` is 1 or more (or not a number),
/// else `w · ce + (1 − w) · incoming` over min-max normalized scores.
/// `incoming` is indexed by candidate index. Ties keep the incoming order.
fn order_scored(
    scores: &[RerankScore],
    incoming: &[f32],
    blend_weight: f32,
) -> Vec<(RerankScore, f32)> {
    let mut keyed: Vec<(RerankScore, f32)> = if blend_weight < 1.0 {
        let w = blend_weight.max(0.0);
        let ce: Vec<f32> = scores.iter().map(|rs| rs.score).collect();
        let fused: Vec<f32> = scores.iter().map(|rs| incoming[rs.index]).collect();
        let (ce, fused) = (min_max(&ce), min_max(&fused));
        scores
            .iter()
            .zip(ce.iter().zip(&fused))
            .map(|(rs, (c, f))| (rs.clone(), w * c + (1.0 - w) * f))
            .collect()
    } else {
        scores.iter().map(|rs| (rs.clone(), rs.score)).collect()
    };
    keyed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.index.cmp(&b.0.index))
    });
    keyed
}

/// Rerank a set of candidate results using a cross-encoder model.
///
/// Takes the top `config.top_k_rerank` candidates, scores each (query, passage)
/// pair with the cross-encoder, and returns results sorted by cross-encoder
/// score, or by its blend with their incoming `score` when
/// `config.blend_weight` is below 1.
///
/// When the `rerank` feature is not enabled, returns results unchanged.
#[cfg(feature = "rerank")]
pub fn rerank(query: &str, results: &mut Vec<Value>, config: &RerankConfig) -> Result<(), String> {
    if !config.enabled || results.is_empty() {
        return Ok(());
    }
    let scores = cross_encoder_scores(query, results, config)?;
    apply_scores(results, &scores, config);
    Ok(())
}

/// Score the first `config.top_k_rerank` candidates against `query` with the
/// cross-encoder, one [`RerankScore`] per candidate that has text (a long
/// passage scores as its best window). [`apply_scores`] orders the list by
/// them; scoring once and applying several blend weights compares the
/// weights on one set of scores.
#[cfg(feature = "rerank")]
pub fn cross_encoder_scores(
    query: &str,
    results: &[Value],
    config: &RerankConfig,
) -> Result<Vec<RerankScore>, String> {
    let model_path = std::path::Path::new(&config.model);
    if !model_path.exists() {
        return Err(format!(
            "cross-encoder model not found: {}. Download it first.",
            config.model
        ));
    }

    // Only rerank top K.
    let rerank_count = results.len().min(config.top_k_rerank);

    let passages: Vec<String> = results[..rerank_count].iter().map(passage_text).collect();

    // Score each (query, passage) pair.
    let mut scores: Vec<RerankScore> = Vec::with_capacity(rerank_count);

    RERANK_RUNTIMES.with(|cache| -> Result<(), String> {
        let mut cache = cache.borrow_mut();
        if !cache.contains_key(&config.model) {
            let runtime = load_runtime(model_path)?;
            cache.insert(config.model.clone(), runtime);
        }
        let runtime = cache
            .get_mut(&config.model)
            .ok_or_else(|| "rerank runtime cache miss".to_string())?;

        for (i, passage) in passages.iter().enumerate() {
            // Nothing to score: an empty passage takes no rank from a real one.
            if passage.trim().is_empty() {
                continue;
            }
            let windows = passage_windows(passage);
            let mut best: f32 = f32::NEG_INFINITY;
            for window in &windows {
                let score = score_pair(runtime, query, window)?;
                if score > best {
                    best = score;
                }
            }
            if !best.is_finite() {
                continue;
            }

            scores.push(RerankScore {
                index: i,
                score: best,
            });
        }

        Ok(())
    })?;
    Ok(scores)
}

/// Reorder `results` by `scores` from [`cross_encoder_scores`] over the same
/// list and `config`: the scored candidates best-first (by cross-encoder
/// score, or by its blend with their incoming `score` when
/// `config.blend_weight` is below 1), then the unscorable ones (no text) in
/// their original order, then the candidates past `config.top_k_rerank`.
/// Each scored candidate gains `rerank_score`, and `rerank_blend` when blended.
pub fn apply_scores(results: &mut Vec<Value>, scores: &[RerankScore], config: &RerankConfig) {
    let rerank_count = results.len().min(config.top_k_rerank);
    let incoming = incoming_scores(&results[..rerank_count]);
    let ordered = order_scored(scores, &incoming, config.blend_weight);
    let blending = config.blend_weight < 1.0;

    let mut reranked: Vec<Value> = Vec::with_capacity(results.len());
    for (rs, key) in &ordered {
        let mut item = results[rs.index].clone();
        if let Some(obj) = item.as_object_mut() {
            obj.insert("rerank_score".to_string(), serde_json::json!(rs.score));
            if blending {
                obj.insert("rerank_blend".to_string(), serde_json::json!(key));
            }
        }
        reranked.push(item);
    }
    let scored: std::collections::HashSet<usize> = scores.iter().map(|rs| rs.index).collect();
    for (i, item) in results[..rerank_count].iter().enumerate() {
        if !scored.contains(&i) {
            reranked.push(item.clone());
        }
    }
    // Append items that weren't reranked.
    for item in results.iter().skip(rerank_count) {
        reranked.push(item.clone());
    }

    *results = reranked;
}

/// Split a passage into overlapping byte-safe windows. Each window is sized so
/// the tokenized `(query, window)` pair fits under `MAX_RERANK_TOKENS` for
/// typical english text, avoiding silent truncation of the answer content.
fn passage_windows(passage: &str) -> Vec<String> {
    if passage.is_empty() {
        return Vec::new();
    }
    if passage.len() <= WINDOW_SKIP_BELOW_BYTES {
        return vec![passage.to_string()];
    }
    axil_core::util::overlapping_chunks(passage, WINDOW_MAX_BYTES, WINDOW_OVERLAP_BYTES)
}

#[cfg(feature = "rerank")]
fn score_pair(runtime: &mut RerankRuntime, query: &str, passage: &str) -> Result<f32, String> {
    let encoding = runtime
        .tokenizer
        .encode((query, passage), true)
        .map_err(|e| format!("tokenize error: {e}"))?;

    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mut attention: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let mut type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
    truncate_encoding_triplet(&mut ids, &mut attention, &mut type_ids, MAX_RERANK_TOKENS);
    let len = ids.len();

    let input_ids = Tensor::<i64>::from_array(([1, len], ids))
        .map_err(|e| format!("input_ids tensor error: {e}"))?;
    let attn_mask = Tensor::<i64>::from_array(([1, len], attention))
        .map_err(|e| format!("attention_mask tensor error: {e}"))?;

    // Only bind `token_type_ids` when the model declares it. BGE reranker
    // (XLM-R backbone) omits this input, and ONNX rejects unrecognized names.
    let outputs = if runtime.accepts_token_type_ids {
        let token_type = Tensor::<i64>::from_array(([1, len], type_ids))
            .map_err(|e| format!("token_type_ids tensor error: {e}"))?;
        runtime.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attn_mask,
            "token_type_ids" => token_type,
        ])
    } else {
        runtime.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attn_mask,
        ])
    }
    .map_err(|e| format!("inference error: {e}"))?;

    let output = outputs.iter().next().ok_or("model produced no outputs")?.1;
    let (shape, data): (_, &[f32]) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("output extract error: {e}"))?;
    // For 2-label classifier exports (`[1, 2]` = `[neg_logit, pos_logit]`),
    // taking `data[0]` would sort on the negative class. Single-logit
    // exports (`[1, 1]`) take `data[0]`.
    let score = if shape.len() == 2 && shape[1] == 2 && data.len() >= 2 {
        data[1]
    } else {
        data.first().copied().unwrap_or(0.0)
    };
    Ok(score)
}

#[cfg(feature = "rerank")]
fn load_runtime(model_path: &std::path::Path) -> Result<RerankRuntime, String> {
    // Try CUDA execution provider first — cross-encoder scoring is the
    // latency bottleneck at inference time (dozens of (query, passage) pairs
    // per query). On CPU, BGE-reranker-base runs ~1–2 s per pair; on CUDA,
    // ~30 ms. Fall back to CPU if the CUDA EP fails to build.
    let session = {
        use ort::execution_providers::cuda::CUDAExecutionProvider;
        let cuda_ep = CUDAExecutionProvider::default().build();
        let base = ort::session::Session::builder()
            .map_err(|e| format!("ONNX session error: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("optimization error: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| format!("thread error: {e}"))?;
        match base.with_execution_providers([cuda_ep]) {
            Ok(gpu_builder) => {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                axil_core::util::log_once_if_verbose(
                    &LOGGED,
                    "[rerank] CUDA execution provider enabled",
                );
                gpu_builder
                    .commit_from_file(model_path)
                    .map_err(|e| format!("model load error: {e}"))?
            }
            Err(_) => ort::session::Session::builder()
                .map_err(|e| format!("ONNX session error: {e}"))?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(|e| format!("optimization error: {e}"))?
                .with_intra_threads(2)
                .map_err(|e| format!("thread error: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| format!("model load error: {e}"))?,
        }
    };

    let tokenizer_path = model_path
        .parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| std::path::PathBuf::from("models/tokenizer.json"));
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer load error: {e}"))?;

    // Probe declared inputs so we know which tensors to bind later.
    let accepts_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

    Ok(RerankRuntime {
        session,
        tokenizer,
        accepts_token_type_ids,
    })
}

/// No-op reranker when the `rerank` feature is not enabled.
#[cfg(not(feature = "rerank"))]
pub fn rerank(
    _query: &str,
    _results: &mut Vec<Value>,
    _config: &RerankConfig,
) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CLI hands the reranker records whose `data` is an object; reading
    /// only string fields made every passage empty.
    #[test]
    fn passage_text_reads_record_objects() {
        let record = serde_json::json!({"id": "A", "data": {"summary": "pool size is 32", "files": ["a.rs"]}});
        assert_eq!(passage_text(&record), "pool size is 32");
        let flat =
            serde_json::json!({"summary": "top-level summary", "data": {"summary": "inner"}});
        assert_eq!(passage_text(&flat), "top-level summary");
        assert_eq!(
            passage_text(&serde_json::json!({"data": "plain text"})),
            "plain text"
        );
        assert_eq!(passage_text(&serde_json::json!({"id": "B"})), "");
    }
    use serde_json::json;

    #[test]
    fn rerank_noop_when_disabled() {
        let config = RerankConfig::default(); // enabled: false
        let mut results = vec![
            json!({"id": "a", "summary": "first"}),
            json!({"id": "b", "summary": "second"}),
        ];
        let original = results.clone();
        rerank("test query", &mut results, &config).unwrap();
        assert_eq!(
            results, original,
            "disabled reranker should not modify results"
        );
    }

    #[test]
    fn rerank_noop_empty_results() {
        let config = RerankConfig {
            enabled: true,
            ..Default::default()
        };
        let mut results: Vec<Value> = vec![];
        rerank("test", &mut results, &config).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn rerank_config_defaults() {
        let config = RerankConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.top_k_rerank, 20);
        assert_eq!(config.model, "models/cross-encoder.onnx");
        assert_eq!(config.blend_weight, 0.3);
    }

    fn ce(scores: &[(usize, f32)]) -> Vec<RerankScore> {
        scores
            .iter()
            .map(|&(index, score)| RerankScore { index, score })
            .collect()
    }

    fn order(ranked: &[(RerankScore, f32)]) -> Vec<usize> {
        ranked.iter().map(|(rs, _)| rs.index).collect()
    }

    // Recall order 0, 1, 2; the cross-encoder prefers 2, then 0, then 1.
    const INCOMING: [f32; 3] = [0.9, 0.8, 0.1];
    const CE: [(usize, f32); 3] = [(0, 1.0), (1, -4.0), (2, 6.0)];

    #[test]
    fn full_weight_is_the_cross_encoder_order() {
        let ranked = order_scored(&ce(&CE), &INCOMING, 1.0);
        assert_eq!(order(&ranked), vec![2, 0, 1]);
        // The key is the raw cross-encoder score, as before blending existed.
        assert_eq!(ranked[0].1, 6.0);
    }

    #[test]
    fn zero_weight_is_the_incoming_order() {
        assert_eq!(
            order(&order_scored(&ce(&CE), &INCOMING, 0.0)),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn a_blend_weighs_both_normalized_scores() {
        // Normalized: ce = [0.5, 0.0, 1.0], incoming = [1.0, 0.875, 0.0].
        // w = 0.5 → [0.75, 0.4375, 0.5]: candidate 0 keeps first place, and
        // the cross-encoder lifts 2 over 1.
        let ranked = order_scored(&ce(&CE), &INCOMING, 0.5);
        assert_eq!(order(&ranked), vec![0, 2, 1]);
        assert!((ranked[0].1 - 0.75).abs() < 1e-6);
    }

    #[test]
    fn a_flat_signal_leaves_the_order_to_the_other_one() {
        let flat = ce(&[(0, 3.0), (1, 3.0), (2, 3.0)]);
        assert_eq!(order(&order_scored(&flat, &INCOMING, 0.9)), vec![0, 1, 2]);
        // Ties keep the incoming order.
        assert_eq!(order(&order_scored(&flat, &INCOMING, 1.0)), vec![0, 1, 2]);
    }

    #[test]
    fn apply_scores_orders_the_window_then_the_unscored_then_the_rest() {
        let mut results = vec![
            json!({"id": "a", "score": 0.9}),
            json!({"id": "b", "score": 0.8}),
            json!({"id": "c", "score": 0.7}),
            json!({"id": "d", "score": 0.6}),
        ];
        // A window of three; "b" had no text to score. The cross-encoder's
        // order alone, so the window's order is the scores'.
        let config = RerankConfig {
            top_k_rerank: 3,
            blend_weight: 1.0,
            ..Default::default()
        };
        let scores = ce(&[(0, -2.0), (2, 5.0)]);
        apply_scores(&mut results, &scores, &config);
        let ids: Vec<&str> = results.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["c", "a", "b", "d"]);
        assert_eq!(results[0]["rerank_score"], json!(5.0));
        assert!(results[0].get("rerank_blend").is_none());
        assert!(results[2].get("rerank_score").is_none());
    }

    #[test]
    fn incoming_scores_fall_back_to_position() {
        let scored = [json!({"score": 0.5}), json!({"score": 0.25})];
        assert_eq!(incoming_scores(&scored), vec![0.5, 0.25]);
        let partly = [json!({"score": 0.5}), json!({"id": "b"})];
        assert_eq!(incoming_scores(&partly), vec![0.0, -1.0]);
    }

    #[test]
    fn passage_windows_short_passage_single_window() {
        let out = passage_windows("short passage");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], "short passage");
    }

    #[test]
    fn passage_windows_empty_no_windows() {
        let out = passage_windows("");
        assert!(out.is_empty());
    }

    #[test]
    fn passage_windows_long_passage_overlapping_chunks() {
        let long: String = "a".repeat(5000);
        let out = passage_windows(&long);
        assert!(out.len() > 1, "long passage must produce multiple windows");
        for w in &out {
            assert!(w.len() <= WINDOW_MAX_BYTES);
        }
        // Overlap: consecutive windows share the trailing/leading region.
        let step = WINDOW_MAX_BYTES - WINDOW_OVERLAP_BYTES;
        let expected_windows = (long.len() + step - 1) / step;
        assert!(out.len() >= expected_windows.saturating_sub(1));
    }

    #[test]
    fn truncate_encoding_triplet_caps_lengths() {
        let mut ids = (0..600).collect::<Vec<i64>>();
        let mut attention = vec![1_i64; 620];
        let mut type_ids = vec![0_i64; 610];
        truncate_encoding_triplet(&mut ids, &mut attention, &mut type_ids, 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(attention.len(), 512);
        assert_eq!(type_ids.len(), 512);
    }

    #[cfg(feature = "rerank")]
    #[test]
    fn rerank_errors_on_missing_model() {
        let config = RerankConfig {
            enabled: true,
            model: "/nonexistent/model.onnx".to_string(),
            top_k_rerank: 5,
            ..Default::default()
        };
        let mut results = vec![json!({"id": "a", "summary": "test"})];
        let err = rerank("query", &mut results, &config);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("not found"));
    }
}
