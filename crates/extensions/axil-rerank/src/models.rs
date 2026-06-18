//! Built-in cross-encoder model registry.
//!
//! Each variant maps to a Hugging Face repository + the file paths needed
//! to run inference via [`ort`]. Adding a new model is a 3-step process:
//! enum variant → `repo_files()` → optional registry entry in `runtime.rs`
//! if the model needs special tokenizer handling.

use std::path::PathBuf;

/// Supported cross-encoder reranker models. All are Apache-2.0 so the model
/// weights stay redistributable under both Axil's noncommercial license and its
/// commercial license (jina-reranker-v2 is CC-BY-NC — noncommercial-only — and is
/// explicitly skipped for that reason).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankModel {
    /// answerdotai/answerai-colbert-small-v1 — 33M params, Apache-2.0.
    /// BEIR 53.79 NDCG@10 (beats bge-base 53.25 at ~7× smaller). Despite
    /// the ColBERT name, the ONNX export runs as a single-vector cross-
    /// encoder for our purposes. **Default** for Phase 15 P0.3.
    AnswerAiColbertSmall,
    /// cross-encoder/ms-marco-MiniLM-L-6-v2 — 22M params, Apache-2.0.
    /// MS-MARCO-trained, widely tested. Kept for backward compatibility
    /// with the previous in-tree reranker (formerly in axil-indexer).
    MsMarcoMiniLm,
    /// User-provided ONNX cross-encoder. Path must contain `model.onnx`
    /// and `tokenizer.json` in the same directory.
    Custom(PathBuf),
}

/// Metadata captured for diagnostics + the LongMemEval gate annotations.
#[derive(Debug, Clone)]
pub struct RerankModelMeta {
    pub name: &'static str,
    pub params_millions: f32,
    pub license: &'static str,
    pub max_seq_len: usize,
    pub approx_size: &'static str,
}

impl RerankModel {
    /// Parse a string id from `axil.toml` or `--model` CLI flag.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "answerai-colbert-small" | "answerai-colbert-small-v1" | "colbert-small" => {
                Some(Self::AnswerAiColbertSmall)
            }
            "ms-marco-minilm" | "ms-marco-minilm-l-6-v2" | "minilm" => Some(Self::MsMarcoMiniLm),
            _ => None,
        }
    }

    /// Stable id for serialisation, diagnostics, and on-disk model dir.
    pub fn name(&self) -> &str {
        match self {
            Self::AnswerAiColbertSmall => "answerai-colbert-small-v1",
            Self::MsMarcoMiniLm => "ms-marco-MiniLM-L-6-v2",
            Self::Custom(path) => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom-reranker"),
        }
    }

    /// Metadata bundle for tooling.
    pub fn meta(&self) -> RerankModelMeta {
        match self {
            Self::AnswerAiColbertSmall => RerankModelMeta {
                name: "answerai-colbert-small-v1",
                params_millions: 33.0,
                license: "Apache-2.0",
                max_seq_len: 512,
                approx_size: "~130MB",
            },
            Self::MsMarcoMiniLm => RerankModelMeta {
                name: "ms-marco-MiniLM-L-6-v2",
                params_millions: 22.0,
                license: "Apache-2.0",
                max_seq_len: 512,
                approx_size: "~90MB",
            },
            Self::Custom(_) => RerankModelMeta {
                name: "custom",
                params_millions: 0.0,
                license: "user-provided",
                max_seq_len: 512,
                approx_size: "unknown",
            },
        }
    }

    /// HuggingFace `(repo, remote_path, local_filename)` triples to fetch.
    /// Empty for [`Self::Custom`] (no auto-download — user supplies files).
    pub fn repo_files(&self) -> Vec<(&'static str, &'static str, &'static str)> {
        match self {
            Self::AnswerAiColbertSmall => vec![
                (
                    "answerdotai/answerai-colbert-small-v1",
                    "onnx/model.onnx",
                    "model.onnx",
                ),
                (
                    "answerdotai/answerai-colbert-small-v1",
                    "tokenizer.json",
                    "tokenizer.json",
                ),
            ],
            Self::MsMarcoMiniLm => vec![
                (
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "onnx/model.onnx",
                    "model.onnx",
                ),
                (
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "tokenizer.json",
                    "tokenizer.json",
                ),
            ],
            Self::Custom(_) => vec![],
        }
    }

    /// Resolve the directory holding `model.onnx` + `tokenizer.json` for
    /// this model. For built-ins: `~/.axil/rerank-models/<name>/`.
    pub fn model_dir(&self) -> Option<PathBuf> {
        match self {
            Self::Custom(path) => path.parent().map(|p| p.to_path_buf()),
            _ => home_axil_dir().map(|base| base.join("rerank-models").join(self.name())),
        }
    }
}

fn home_axil_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".axil"))
}

impl std::fmt::Display for RerankModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for RerankModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_name(s).ok_or_else(|| format!("unknown reranker model: {s}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_answerai_colbert_aliases() {
        for n in [
            "answerai-colbert-small",
            "answerai-colbert-small-v1",
            "colbert-small",
            "COLBERT-SMALL",
        ] {
            assert!(matches!(
                RerankModel::from_name(n),
                Some(RerankModel::AnswerAiColbertSmall)
            ));
        }
    }

    #[test]
    fn parses_minilm_aliases() {
        for n in ["ms-marco-minilm", "minilm", "MS-MARCO-MINILM-L-6-V2"] {
            assert!(matches!(
                RerankModel::from_name(n),
                Some(RerankModel::MsMarcoMiniLm)
            ));
        }
    }

    #[test]
    fn rejects_unknown_model() {
        assert!(RerankModel::from_name("not-a-real-thing").is_none());
    }

    #[test]
    fn meta_carries_license() {
        assert_eq!(
            RerankModel::AnswerAiColbertSmall.meta().license,
            "Apache-2.0"
        );
        assert_eq!(RerankModel::MsMarcoMiniLm.meta().license, "Apache-2.0");
    }

    #[test]
    fn repo_files_present_for_builtins() {
        assert!(!RerankModel::AnswerAiColbertSmall.repo_files().is_empty());
        assert!(!RerankModel::MsMarcoMiniLm.repo_files().is_empty());
        assert!(RerankModel::Custom(PathBuf::from("/tmp/m.onnx"))
            .repo_files()
            .is_empty());
    }
}
