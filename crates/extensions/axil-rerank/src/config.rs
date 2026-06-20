//! Configuration for the rerank pipeline stage.

use serde::{Deserialize, Serialize};

/// Knob set for the rerank stage. Serialised under `[rerank]` in `axil.toml`.
///
/// Defaults match the memory-quality tuning:
/// - `top_k_in = 50`  — candidates fed to the reranker after fusion
/// - `top_k_out = 10` — kept after rerank, before the token-budget trim
/// - `weight = 0.7`   — blend toward the reranker's score
/// - `enabled = false` in v0.7; default-on after the LongMemEval gate passes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// Master switch — when off, the stage is a true no-op (zero token /
    /// latency cost). Flip to true to engage the trait impl. Default off
    /// until the LongMemEval gate proves a ≥+5 recall@K lift.
    pub enabled: bool,
    /// Built-in model id, or arbitrary path/name forwarded to a custom
    /// loader. Defaults to [`crate::RerankModel::AnswerAiColbertSmall`].
    pub model: String,
    /// Candidates to rerank per query (the fusion stage typically returns
    /// 50–200; reranking more than ~100 is wasted CPU since the tail
    /// almost never makes the cut).
    pub top_k_in: usize,
    /// Final list size returned to the caller after rerank. The token-budget
    /// trim runs after this, so keep this generous (≥ top_k caller asks for).
    pub top_k_out: usize,
    /// Score-blend weight: `final = weight * sigmoid(rerank) + (1-weight) * fused`.
    /// 1.0 = pure reranker order, 0.0 = pure fused order (NoOp behaviour).
    pub weight: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "answerai-colbert-small-v1".to_string(),
            top_k_in: 50,
            top_k_out: 10,
            weight: 0.7,
        }
    }
}

impl RerankConfig {
    /// Convenience for tests: enabled config pointing at the default model.
    pub fn enabled_default() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_phase15_spec() {
        let c = RerankConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.top_k_in, 50);
        assert_eq!(c.top_k_out, 10);
        assert!((c.weight - 0.7).abs() < 1e-6);
        assert_eq!(c.model, "answerai-colbert-small-v1");
    }

    #[test]
    fn enabled_default_flips_enabled() {
        assert!(RerankConfig::enabled_default().enabled);
    }
}
