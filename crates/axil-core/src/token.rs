//! Token estimation — a pluggable seam for sizing context against a budget.
//!
//! Boot context assembly and budget-shaped recall need to know roughly how
//! many tokens a chunk of text will cost a model before deciding what to keep.
//! Historically this was a hardcoded `chars / 4` heuristic scattered across
//! the budget paths. That heuristic is cheap and dependency-free but admits
//! 10–30% error against real BPE tokenizers, and it left no way to opt into a
//! precise count when accuracy matters.
//!
//! [`TokenEstimator`] is that seam. The default [`CharsPerTokenEstimator`]
//! reproduces the old `chars / 4` math byte-for-byte, so swapping the call
//! sites onto the trait changes no numbers. Callers that want precision (and
//! can afford a tokenizer model) can substitute [`TokenizersEstimator`] behind
//! the `real-tokenizer` feature, or any custom impl.

/// Chars-to-tokens divisor. 4.0 is a widely-cited average for English BPE
/// tokenizers and is within ±15% across cl100k, tiktoken, and BGE's XLM-R.
/// The default estimator needs only enough accuracy to keep boot context from
/// blowing through a generous budget, not exactness.
pub const CHARS_PER_TOKEN: f64 = 4.0;

/// Estimates the token cost of a piece of text.
///
/// Implementations trade accuracy against cost: the default
/// [`CharsPerTokenEstimator`] is a dependency-free heuristic, while a
/// tokenizer-backed estimator returns exact counts at the price of loading a
/// model. Budget and truncation paths take a `&dyn TokenEstimator` so the
/// estimator can be swapped without touching the shaping logic.
pub trait TokenEstimator: Send + Sync {
    /// Estimate the number of tokens `text` would tokenize into.
    fn estimate_tokens(&self, text: &str) -> usize;
}

/// The dependency-free default: `ceil(byte_len / 4)`.
///
/// This reproduces the historical `chars / 4` accounting exactly — it counts
/// bytes (`str::len`), not Unicode scalar values, because that is what the
/// original budget code did and changing it would shift every budget decision.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharsPerTokenEstimator;

impl TokenEstimator for CharsPerTokenEstimator {
    fn estimate_tokens(&self, text: &str) -> usize {
        (text.len() as f64 / CHARS_PER_TOKEN).ceil() as usize
    }
}

/// A cheap, shareable instance of the default estimator.
///
/// `CharsPerTokenEstimator` is zero-sized, so this costs nothing and lets call
/// sites borrow a `&'static dyn TokenEstimator` without constructing one.
pub static DEFAULT_TOKEN_ESTIMATOR: CharsPerTokenEstimator = CharsPerTokenEstimator;

#[cfg(feature = "real-tokenizer")]
mod real {
    use super::TokenEstimator;
    use std::path::Path;
    use tokenizers::Tokenizer;

    /// A precise estimator backed by a Hugging Face `tokenizers` model.
    ///
    /// Construct it from an on-disk `tokenizer.json` (the file shipped
    /// alongside an embedding model); it performs no download and touches no
    /// network. Use this on budget paths where the `chars / 4` heuristic's
    /// 10–30% error would cause boot context to over- or under-fill the
    /// model's window.
    pub struct TokenizersEstimator {
        tokenizer: Tokenizer,
    }

    impl TokenizersEstimator {
        /// Load a tokenizer from a `tokenizer.json` file on disk.
        ///
        /// # Errors
        /// Returns an error if the file is missing or not a valid tokenizer
        /// definition.
        pub fn from_file(path: impl AsRef<Path>) -> Result<Self, super::TokenError> {
            let tokenizer = Tokenizer::from_file(path.as_ref())
                .map_err(|e| super::TokenError::TokenizerLoad(e.to_string()))?;
            Ok(Self { tokenizer })
        }
    }

    impl TokenEstimator for TokenizersEstimator {
        fn estimate_tokens(&self, text: &str) -> usize {
            // A failed encode falls back to the heuristic rather than
            // panicking on a budget path; an over/under count is preferable
            // to aborting boot assembly.
            match self.tokenizer.encode(text, false) {
                Ok(encoding) => encoding.len(),
                Err(_) => super::CharsPerTokenEstimator.estimate_tokens(text),
            }
        }
    }
}

#[cfg(feature = "real-tokenizer")]
pub use real::TokenizersEstimator;

/// Errors raised while building a token estimator.
#[cfg(feature = "real-tokenizer")]
#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    /// A tokenizer model could not be loaded from disk.
    #[error("failed to load tokenizer: {0}")]
    TokenizerLoad(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // The legacy math both budget paths used, kept verbatim as an oracle so
    // the default impl is proven byte-for-byte identical, not merely "close".
    fn legacy_chars_per_4(text: &str) -> usize {
        (text.len() as f64 / 4.0).ceil() as usize
    }

    #[test]
    fn default_matches_legacy_chars_per_4() {
        let cases = [
            "",
            "a",
            "abc",
            "abcd",
            "abcde",
            "hello",
            "fn validate_token(token: &str) -> Result<Claims, AuthError>",
            "a longer sentence with several words and some punctuation!",
            "café — naïve façade", // multi-byte: len() counts bytes, on purpose
        ];
        for case in cases {
            assert_eq!(
                CharsPerTokenEstimator.estimate_tokens(case),
                legacy_chars_per_4(case),
                "default estimator diverged from legacy chars/4 on {case:?}"
            );
        }
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(CharsPerTokenEstimator.estimate_tokens(""), 0);
    }

    #[test]
    fn rounds_up() {
        // 5 bytes / 4 = 1.25 -> 2
        assert_eq!(CharsPerTokenEstimator.estimate_tokens("hello"), 2);
        // 4 bytes / 4 = 1.0 -> 1
        assert_eq!(CharsPerTokenEstimator.estimate_tokens("abcd"), 1);
    }

    #[test]
    fn static_default_instance_is_usable() {
        let est: &dyn TokenEstimator = &DEFAULT_TOKEN_ESTIMATOR;
        assert_eq!(est.estimate_tokens("hello"), 2);
    }

    #[test]
    fn custom_estimator_can_be_substituted() {
        // A custom impl that returns a fixed cost, proving the trait object
        // dispatches to whatever the caller supplies.
        struct FixedEstimator(usize);
        impl TokenEstimator for FixedEstimator {
            fn estimate_tokens(&self, _text: &str) -> usize {
                self.0
            }
        }
        let est: &dyn TokenEstimator = &FixedEstimator(42);
        assert_eq!(est.estimate_tokens("anything"), 42);
        assert_eq!(est.estimate_tokens(""), 42);
    }
}
