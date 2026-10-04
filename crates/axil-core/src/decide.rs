//! Pair judgements from a small local model.
//!
//! A [`PairJudge`] answers one fixed question about two texts, such as "does
//! the second update or contradict the first?", with a calibrated probability
//! and a yes/no cut at the judge's threshold. Only the trait lives in core, so
//! the ONNX and tokenizer machinery stays out of `axil-core` (as with
//! [`crate::query::Rerank`]); the shipped implementation is the
//! `PairClassifier` in the `axil-decide` crate.

use crate::error::Result;

/// The judgement on one `(premise, hypothesis)` pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairVerdict {
    /// Calibrated probability that the judged relation holds.
    pub probability: f64,
    /// Whether `probability` reached the judge's threshold.
    pub flagged: bool,
}

/// A model that judges one relation between two texts.
///
/// Implementations must be thread-safe (`Send + Sync`) and may block on
/// inference; callers should not hold locks across [`PairJudge::judge`].
pub trait PairJudge: Send + Sync {
    /// Judge each `(premise, hypothesis)` pair. Returns one verdict per pair,
    /// in input order.
    fn judge(&self, pairs: &[(&str, &str)]) -> Result<Vec<PairVerdict>>;

    /// The probability at or above which a verdict is flagged.
    fn threshold(&self) -> f64;

    /// The model behind the verdicts, for provenance on anything they create.
    fn name(&self) -> &str;
}
