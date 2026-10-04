//! Small local models for Axil's judgement calls.
//!
//! The first one is a generic ONNX *pair classifier*: a cross-encoder that
//! reads two texts together and returns one logit per label. A
//! [`Calibration`] picks the label, turns its logit into a calibrated
//! probability with a fitted temperature, and cuts it at a threshold.
//! [`PairClassifier`] implements [`axil_core::PairJudge`], the seam core
//! code calls.
//!
//! Models are opt-in downloads pinned by Hugging Face revision and sha256
//! ([`PairModel`]); nothing is bundled. The U1 judge ("does NEW update,
//! replace or contradict OLD?") is [`NLI_DEBERTA_V3_XSMALL_QINT8`] with
//! [`Calibration::u1`], fitted by `benchmarks/typed-decisions/calibrate_u1.py`.
//!
//! ```no_run
//! use axil_decide::{Calibration, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
//!
//! let dir = U1.default_dir().expect("home directory");
//! if !U1.is_available(&dir) {
//!     U1.download(&dir)?;
//! }
//! let judge = PairClassifier::load(&U1, &dir, Calibration::u1())?;
//! let scores = judge.score_pairs(&[("Deploys run at 5pm.", "Deploys now run at 6pm.")])?;
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
