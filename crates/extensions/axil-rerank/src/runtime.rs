//! ONNX execution backend. Gated by the `onnx` feature so the crate can
//! be depended on without dragging `ort` + `tokenizers` everywhere.
//!
//! Direct successor to the prior `axil_indexer::rerank` module, with two
//! upgrades:
//! 1. Pluggable model registry (answerai-colbert + ms-marco-minilm built-in).
//! 2. `Reranker` trait so callers can swap reranker impls in tests.

use std::path::Path;

use parking_lot::Mutex;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::models::RerankModel;
use crate::{Candidate, RerankError, RerankScore, Reranker};

/// Byte budget per passage window. ~450 BPE tokens of English text, leaving
/// headroom for the query + special tokens under the 512-token cross-
/// encoder limit. Long passages are split into overlapping windows and
/// scored window-by-window — the *max* window score is the passage score.
/// Without this, an 8 KB conversation truncates to its first 512 tokens
/// and the model never sees the turn where the answer fact was stated.
const WINDOW_MAX_BYTES: usize = 1800;
const WINDOW_OVERLAP_BYTES: usize = 200;
/// Passages at or below this size are scored as a single window.
const WINDOW_SKIP_BELOW_BYTES: usize = 2000;

/// Split a passage into overlapping byte-safe windows for cross-encoder
/// scoring. Short passages return a single window unchanged.
fn passage_windows(passage: &str) -> Vec<String> {
    if passage.is_empty() {
        return Vec::new();
    }
    if passage.len() <= WINDOW_SKIP_BELOW_BYTES {
        return vec![passage.to_string()];
    }
    axil_core::util::overlapping_chunks(passage, WINDOW_MAX_BYTES, WINDOW_OVERLAP_BYTES)
}

/// ONNX-backed cross-encoder. Owns its tokenizer + session and is cheap
/// to call concurrently — the inner [`Session`] is wrapped in a [`Mutex`]
/// because `ort::session::Session::run` takes `&mut self`.
pub struct OnnxReranker {
    model_name: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// BERT-family models declare `token_type_ids`; XLM-R-family don't.
    /// Binding an undeclared input causes `Invalid input name`.
    accepts_token_type_ids: bool,
    max_tokens: usize,
}

impl OnnxReranker {
    /// Build a runtime from a [`RerankModel`]. Returns
    /// [`RerankError::ModelMissing`] when the on-disk files aren't there
    /// yet — caller should run `crate::download::download(&model)` first.
    pub fn load(model: &RerankModel) -> Result<Self, RerankError> {
        let dir = model
            .model_dir()
            .ok_or_else(|| RerankError::Other("cannot resolve model dir".into()))?;
        let model_path = match model {
            RerankModel::Custom(p) => p.clone(),
            _ => dir.join("model.onnx"),
        };
        if !model_path.exists() {
            return Err(RerankError::ModelMissing(model.name().to_string()));
        }
        let tokenizer_path = match model {
            RerankModel::Custom(p) => p
                .parent()
                .map(|x| x.join("tokenizer.json"))
                .unwrap_or_else(|| dir.join("tokenizer.json")),
            _ => dir.join("tokenizer.json"),
        };
        if !tokenizer_path.exists() {
            return Err(RerankError::ModelMissing(format!(
                "tokenizer for {}",
                model.name()
            )));
        }

        let session = build_session(&model_path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RerankError::Other(format!("tokenizer load: {e}")))?;
        let accepts_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        Ok(Self {
            model_name: model.name().to_string(),
            session: Mutex::new(session),
            tokenizer,
            accepts_token_type_ids,
            max_tokens: model.meta().max_seq_len,
        })
    }

    fn score_pair(&self, query: &str, passage: &str) -> Result<f32, RerankError> {
        let encoding = self
            .tokenizer
            .encode((query, passage), true)
            .map_err(|e| RerankError::Tokenize(e.to_string()))?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mut attn: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let mut types: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let keep = ids
            .len()
            .min(attn.len())
            .min(types.len())
            .min(self.max_tokens);
        ids.truncate(keep);
        attn.truncate(keep);
        types.truncate(keep);
        let len = ids.len();
        if len == 0 {
            return Ok(0.0);
        }

        let input_ids = Tensor::<i64>::from_array(([1usize, len], ids))
            .map_err(|e| RerankError::Inference(format!("input_ids tensor: {e}")))?;
        let attention_mask = Tensor::<i64>::from_array(([1usize, len], attn))
            .map_err(|e| RerankError::Inference(format!("attention tensor: {e}")))?;

        let mut session = self.session.lock();
        let outputs = if self.accepts_token_type_ids {
            let type_ids = Tensor::<i64>::from_array(([1usize, len], types))
                .map_err(|e| RerankError::Inference(format!("type_ids tensor: {e}")))?;
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => type_ids,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
        }
        .map_err(|e| RerankError::Inference(e.to_string()))?;

        let first = outputs
            .iter()
            .next()
            .ok_or_else(|| RerankError::Inference("model produced no outputs".into()))?
            .1;
        let (shape, data): (_, &[f32]) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| RerankError::Inference(format!("extract tensor: {e}")))?;
        // [1,2] = [neg_logit, pos_logit] for 2-class exports → take pos.
        // [1,1] single-logit ranking export → take data[0].
        let score = if shape.len() == 2 && shape[1] == 2 && data.len() >= 2 {
            data[1]
        } else {
            data.first().copied().unwrap_or(0.0)
        };
        Ok(score)
    }
}

impl Reranker for OnnxReranker {
    fn score_batch(
        &self,
        query: &str,
        candidates: &[Candidate<'_>],
    ) -> Result<Vec<RerankScore>, RerankError> {
        // No batching across pairs yet — ONNX cross-encoders accept
        // variable-length inputs that don't pad cleanly to a single
        // tensor without dropping accuracy. Loop is the safe default.
        //
        // Each passage is split into overlapping windows so a long
        // conversation isn't silently truncated to its first 512 tokens.
        // The passage score is the max over its windows.
        let mut out = Vec::with_capacity(candidates.len());
        for c in candidates {
            let windows = passage_windows(c.passage);
            let mut best = f32::NEG_INFINITY;
            for window in &windows {
                let s = self.score_pair(query, window)?;
                if s > best {
                    best = s;
                }
            }
            // Empty passage → no windows → neutral score.
            if !best.is_finite() {
                best = 0.0;
            }
            out.push(RerankScore {
                index: c.index,
                score: best,
            });
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

fn build_session(path: &Path) -> Result<Session, RerankError> {
    // CUDA EP first, fall back to CPU. Matches the prior axil-indexer
    // behaviour so existing GPU users don't lose acceleration.
    use ort::execution_providers::cuda::CUDAExecutionProvider;
    use ort::session::builder::GraphOptimizationLevel;

    let base = Session::builder()
        .map_err(|e| RerankError::Inference(format!("session builder: {e}")))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| RerankError::Inference(format!("opt level: {e}")))?
        .with_intra_threads(2)
        .map_err(|e| RerankError::Inference(format!("threads: {e}")))?;

    let cuda_ep = CUDAExecutionProvider::default().build();
    match base.with_execution_providers([cuda_ep]) {
        Ok(b) => b
            .commit_from_file(path)
            .map_err(|e| RerankError::Inference(format!("model load (cuda): {e}"))),
        Err(_) => Session::builder()
            .map_err(|e| RerankError::Inference(format!("session builder cpu: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::Inference(format!("opt level cpu: {e}")))?
            .with_intra_threads(2)
            .map_err(|e| RerankError::Inference(format!("threads cpu: {e}")))?
            .commit_from_file(path)
            .map_err(|e| RerankError::Inference(format!("model load (cpu): {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_returns_missing_when_files_absent() {
        let model = RerankModel::Custom(PathBuf::from("/tmp/__nope__/model.onnx"));
        match OnnxReranker::load(&model) {
            Err(RerankError::ModelMissing(_)) => {}
            Err(other) => panic!("expected ModelMissing, got {other:?}"),
            Ok(_) => panic!("expected load failure for nonexistent model"),
        }
    }

    #[test]
    fn passage_windows_short_passage_single_window() {
        let out = passage_windows("a short conversation");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn passage_windows_empty_no_windows() {
        assert!(passage_windows("").is_empty());
    }

    #[test]
    fn passage_windows_long_passage_splits() {
        let long = "x".repeat(10_000);
        let out = passage_windows(&long);
        assert!(out.len() > 1, "10KB passage must produce multiple windows");
        for w in &out {
            assert!(w.len() <= WINDOW_MAX_BYTES);
        }
    }
}
