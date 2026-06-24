//! Token estimation utilities.
//!
//! Approximates token counts without requiring a tokenizer model. These
//! helpers delegate to [`axil_core::TokenEstimator`] so the indexer's budget
//! paths share one estimation seam with the rest of Axil: the default
//! [`axil_core::CharsPerTokenEstimator`] keeps the historical ~4 chars/token
//! heuristic (accurate within ~10% for English text and code), and a caller
//! that wants precision can substitute a tokenizer-backed estimator at the
//! core seam.

use axil_core::{TokenEstimator, DEFAULT_TOKEN_ESTIMATOR};

/// Estimate the number of tokens in a string (~4 chars per token, rounded up).
pub fn estimate_tokens(text: &str) -> usize {
    DEFAULT_TOKEN_ESTIMATOR.estimate_tokens(text)
}

/// Estimate tokens for a JSON value (serialized form).
pub fn estimate_json_tokens(value: &serde_json::Value) -> usize {
    let s = serde_json::to_string(value).unwrap_or_default();
    estimate_tokens(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn short_text() {
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn code_snippet() {
        let code = "fn validate_token(token: &str) -> Result<Claims, AuthError>";
        let tokens = estimate_tokens(code);
        assert!(tokens >= 10 && tokens <= 20);
    }
}
