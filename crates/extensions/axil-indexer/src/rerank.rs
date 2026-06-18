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
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "models/cross-encoder.onnx".to_string(),
            top_k_rerank: 20,
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

/// Rerank a set of candidate results using a cross-encoder model.
///
/// Takes the top `config.top_k_rerank` candidates, scores each (query, passage)
/// pair with the cross-encoder, and returns results sorted by cross-encoder score.
///
/// When the `rerank` feature is not enabled, returns results unchanged.
#[cfg(feature = "rerank")]
pub fn rerank(query: &str, results: &mut Vec<Value>, config: &RerankConfig) -> Result<(), String> {
    if !config.enabled || results.is_empty() {
        return Ok(());
    }

    let model_path = std::path::Path::new(&config.model);
    if !model_path.exists() {
        return Err(format!(
            "cross-encoder model not found: {}. Download it first.",
            config.model
        ));
    }

    // Only rerank top K.
    let rerank_count = results.len().min(config.top_k_rerank);

    // Extract passage text from each candidate.
    let passages: Vec<String> = results[..rerank_count]
        .iter()
        .map(|v| {
            v.get("summary")
                .and_then(|s| s.as_str())
                .or_else(|| v.get("data").and_then(|d| d.as_str()))
                .unwrap_or("")
                .to_string()
        })
        .collect();

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
            let windows = passage_windows(passage);
            let mut best: f32 = f32::NEG_INFINITY;
            for window in &windows {
                let score = score_pair(runtime, query, window)?;
                if score > best {
                    best = score;
                }
            }
            // Empty passage → no windows → neutral score.
            if !best.is_finite() {
                best = 0.0;
            }

            scores.push(RerankScore {
                index: i,
                score: best,
            });
        }

        Ok(())
    })?;

    // Sort by cross-encoder score descending.
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Rebuild the results: reranked portion first, then the rest unchanged.
    let mut reranked: Vec<Value> = Vec::with_capacity(results.len());
    for rs in &scores {
        let mut item = results[rs.index].clone();
        if let Some(obj) = item.as_object_mut() {
            obj.insert("rerank_score".to_string(), serde_json::json!(rs.score));
        }
        reranked.push(item);
    }
    // Append items that weren't reranked.
    for item in results.iter().skip(rerank_count) {
        reranked.push(item.clone());
    }

    *results = reranked;
    Ok(())
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
        };
        let mut results = vec![json!({"id": "a", "summary": "test"})];
        let err = rerank("query", &mut results, &config);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("not found"));
    }
}
