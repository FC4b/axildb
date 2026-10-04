//! Small local models for Axil's judgement calls.
//!
//! The first one is a generic ONNX *pair classifier*: a cross-encoder that
//! reads two texts together and returns one logit per label. A
//! [`Calibration`] picks the label, turns its logit into a score calibrated
//! against the decision's own labels with a fitted temperature, and cuts it
//! at a threshold.
//! [`PairClassifier`] implements [`axil_core::PairJudge`], the seam core
//! code calls.
//!
//! Models are opt-in downloads pinned by Hugging Face revision and sha256
//! ([`PairModel`]); nothing is bundled.
//!
//! The U1 judge is [`NLI_DEBERTA_V3_XSMALL_QINT8`] with [`Calibration::u1`],
//! fitted by `benchmarks/typed-decisions/calibrate_u1.py`. Its score is a
//! contradiction-model score calibrated against U1 labels (replace, update or
//! contradict): the NLI model's contradiction logit for (older, newer), with a
//! temperature fitted against pairs labelled as the newer text replacing,
//! updating or contradicting the older. It is a review signal, not an update
//! detector: at the shipped threshold it flagged 5 of the 46 plain updates in
//! the U1 test set, so most updates pass unflagged, and a flag is a
//! suggestion to review, never grounds to retire a record.
//!
//! ```no_run
//! use axil_decide::{Calibration, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
//!
//! let dir = U1.default_dir().expect("home directory");
//! // Verifies what is there, fetches what is missing, repairs what is corrupt.
//! U1.download(&dir)?;
//! let judge = PairClassifier::load(&U1, &dir, Calibration::u1())?;
//! let old = "Auth tokens never expire.";
//! let new = "Auth tokens expire after 24 hours.";
//! let scores = judge.score_pairs(&[(old, new)])?;
//! println!("p = {:.3}, flagged = {}", scores[0].probability, scores[0].flagged);
//! # Ok::<(), axil_decide::DecideError>(())
//! ```

mod calibration;
mod classifier;
mod model;

pub use calibration::Calibration;
pub use classifier::{LoadOptions, PairClassifier, PairScore};
pub use model::{ModelFile, PairModel, NLI_DEBERTA_V3_XSMALL_QINT8};

/// Errors from loading, downloading or running a pair classifier.
#[derive(Debug, thiserror::Error)]
pub enum DecideError {
    /// Model files are missing, failed to download, or fail verification.
    #[error("model: {0}")]
    Model(String),
    /// A calibration file is malformed or does not fit the model.
    #[error("calibration: {0}")]
    Calibration(String),
    /// Tokenization or ONNX inference failed.
    #[error("inference: {0}")]
    Inference(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, DecideError>;
