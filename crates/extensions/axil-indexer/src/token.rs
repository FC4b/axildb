//! Token estimation utilities.
//!
//! Approximates token counts without requiring a tokenizer model.
//! Uses the ~4 characters per token heuristic (accurate within ~10%
//! for English text and code with GPT/Claude tokenizers).

/// Estimate the number of tokens in a string (~4 chars per token, rounded up).
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.len();
    if chars == 0 {
        return 0;
    }
    chars.div_ceil(4)
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
