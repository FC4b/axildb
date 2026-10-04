//! Turning a classifier's logits into a calibrated probability and a verdict.

use serde::Deserialize;

use crate::model::PairModel;
use crate::{DecideError, Result};

/// The U1 calibration, fitted by `benchmarks/typed-decisions/calibrate_u1.py`.
const U1_JSON: &str = include_str!("../calibration/u1-nli-deberta-v3-xsmall-qint8.json");

/// How one decision reads a pair classifier: which label it asks about, the
/// temperature that calibrates that label's probability, and the threshold
/// that turns the probability into a verdict.
///
/// The probability is `sigmoid((z_label - logsumexp(z_other)) / temperature)`.
/// At a temperature of 1 that is the label's softmax probability; for any
/// temperature it ranks pairs the same way, so it changes what the
/// probability means but never which pair scores higher. Dividing every logit
/// by the temperature before the softmax would move mass between the other
/// labels and reorder pairs.
///
/// A calibration only fits the scores it was fitted on, and those depend on
/// more than the model's name: the graph and tokenizer files, the truncation
/// length, the onnxruntime version, the graph optimisation level and whether
/// pairs were batched all move them. It records each of these, and
/// [`crate::PairClassifier::load`] refuses a model or runtime that differs in
/// any (see [`Calibration::check_fit`]).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Calibration {
    /// Model name the calibration was fitted for (see [`crate::PairModel::name`]).
    pub model: String,
    /// Hugging Face revision of that model.
    pub revision: String,
    /// The label whose probability is the verdict, e.g. `contradiction`.
    pub label: String,
    /// Divides the label's one-vs-rest logit; above 1 softens an
    /// over-confident model.
    pub temperature: f64,
    /// Calibrated probability at or above which a pair is flagged.
    pub threshold: f64,
    /// sha256 of the ONNX graph the fitted scores came from.
    pub onnx_sha256: String,
    /// sha256 of the tokenizer they came from.
    pub tokenizer_sha256: String,
    /// Truncation length they were tokenized with.
    pub max_len: usize,
    /// onnxruntime version that produced them, e.g. `1.22.0`.
    pub onnxruntime: String,
    /// Graph optimisation level that produced them, e.g. `ORT_ENABLE_ALL`.
    pub optimization_level: String,
    /// Whether each pair ran alone (no batching, no padding). When true the
    /// runtime scores one pair per call, even for a model marked
    /// batch-invariant: batched scores still differ slightly (up to 1.95e-6
    /// in probability on the fp32 U1 export).
    pub one_pair_per_call: bool,
}

impl Calibration {
    /// The U1 judge: P(contradiction | premise = OLD, hypothesis = NEW) from
    /// [`crate::NLI_DEBERTA_V3_XSMALL_QINT8`], cut at a threshold chosen for a
    /// low false-positive rate. A possible-contradiction signal for review,
    /// not an update detector: it flags few plain updates. Fit and test
    /// numbers are in
    /// `benchmarks/results/typed-decisions-u1-calibration-2026-10-04.json`.
    pub fn u1() -> Self {
        Self::from_json(U1_JSON).expect("the bundled U1 calibration parses")
    }

    /// Parse a calibration file (unknown fields are ignored) and check it.
    pub fn from_json(text: &str) -> Result<Self> {
        let calibration: Self =
            serde_json::from_str(text).map_err(|e| DecideError::Calibration(e.to_string()))?;
        calibration.validate()?;
        Ok(calibration)
    }

    /// Check the values are usable: a finite temperature above 0 and a
    /// threshold strictly between 0 and 1. [`Calibration::from_json`] and
    /// every [`crate::PairClassifier`] load run it, so a calibration built
    /// field by field is checked too.
    pub fn validate(&self) -> Result<()> {
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(DecideError::Calibration(format!(
                "temperature must be a positive number, got {}",
                self.temperature
            )));
        }
        if !(self.threshold > 0.0 && self.threshold < 1.0) {
            return Err(DecideError::Calibration(format!(
                "threshold must be strictly between 0 and 1, got {}",
                self.threshold
            )));
        }
        Ok(())
    }

    /// Check that this calibration was fitted on scores `model` produces
    /// under the loaded onnxruntime, version `runtime` (e.g. `1.22.0`), at
    /// `optimization_level`; returns the index of its label in the model's
    /// logits.
    ///
    /// Refuses a different model, revision, graph or tokenizer file,
    /// truncation length or optimisation level; an onnxruntime whose
    /// major.minor differs from the one the scores were measured on (the
    /// int8 U1 model's verdicts change between 1.22 and 1.30); and a
    /// calibration fitted on batched scores of a model whose scores change
    /// in a batch.
    pub fn check_fit(
        &self,
        model: &PairModel,
        runtime: &str,
        optimization_level: &str,
    ) -> Result<usize> {
        let differs = |what: &str, fitted: &dyn std::fmt::Display, here: &dyn std::fmt::Display| {
            Err(DecideError::Calibration(format!(
                "fitted with {what} {fitted}, but this model or runtime has {here}"
            )))
        };
        if self.model != model.name || self.revision != model.revision {
            let fitted = format!("{}@{}", self.model, self.revision);
            return differs(
                "model",
                &fitted,
                &format!("{}@{}", model.name, model.revision),
            );
        }
        if self.onnx_sha256 != model.onnx.sha256 {
            return differs("graph sha256", &self.onnx_sha256, &model.onnx.sha256);
        }
        if self.tokenizer_sha256 != model.tokenizer.sha256 {
            return differs(
                "tokenizer sha256",
                &self.tokenizer_sha256,
                &model.tokenizer.sha256,
            );
        }
        if self.max_len != model.max_len {
            return differs("max_len", &self.max_len, &model.max_len);
        }
        match (major_minor(&self.onnxruntime), major_minor(runtime)) {
            (Some(fitted), Some(loaded)) if fitted == loaded => {}
            _ => return differs("onnxruntime", &self.onnxruntime, &runtime),
        }
        if self.optimization_level != optimization_level {
            return differs(
                "optimization level",
                &self.optimization_level,
                &optimization_level,
            );
        }
        if !self.one_pair_per_call && !model.batch_invariant {
            return differs(
                "batched scores",
                &"of a model whose scores change in a batch",
                &"one pair per call",
            );
        }
        model.label_index(&self.label).ok_or_else(|| {
            DecideError::Calibration(format!(
                "label {:?} is not one of {:?}",
                self.label, model.labels
            ))
        })
    }

    /// Calibrated probability of the label at `label_index` in `logits`.
    pub fn probability(&self, logits: &[f32], label_index: usize) -> f64 {
        sigmoid(one_vs_rest_logit(logits, label_index) / self.temperature)
    }

    /// Whether a calibrated probability reaches the threshold.
    pub fn flags(&self, probability: f64) -> bool {
        probability >= self.threshold
    }
}

/// `(major, minor)` of a version such as `1.22.0`; `None` if it has no
/// numeric major and minor.
fn major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// `z[label] - logsumexp(z[others])`: the log-odds of the label under a
/// softmax over `logits`. Computed in f64 for the same result as the Python
/// calibration, which works on float64.
pub(crate) fn one_vs_rest_logit(logits: &[f32], label_index: usize) -> f64 {
    let rest_max = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != label_index)
        .map(|(_, &z)| z as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let rest_sum: f64 = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != label_index)
        .map(|(_, &z)| (z as f64 - rest_max).exp())
        .sum();
    logits[label_index] as f64 - (rest_max + rest_sum.ln())
}

/// Logistic function, without overflow for large negative inputs.
pub(crate) fn sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax(z: &[f32]) -> Vec<f64> {
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = z.iter().map(|&x| (x as f64 - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    }

    fn calibration(temperature: f64, threshold: f64) -> Calibration {
        Calibration {
            model: "m".into(),
            revision: "r".into(),
            label: "contradiction".into(),
            temperature,
            threshold,
            onnx_sha256: "g".into(),
            tokenizer_sha256: "t".into(),
            max_len: 512,
            onnxruntime: "1.22.0".into(),
            optimization_level: "ORT_ENABLE_ALL".into(),
            one_pair_per_call: true,
        }
    }

    #[test]
    fn temperature_one_is_the_softmax_probability() {
        let c = calibration(1.0, 0.5);
        for z in [[3.36f32, -4.64, 0.85], [-2.0, 1.0, 0.5], [0.0, 0.0, 0.0]] {
            for i in 0..3 {
                assert!((c.probability(&z, i) - softmax(&z)[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn temperature_keeps_the_order_of_pairs() {
        // Same contradiction logit, the rest of the mass spread differently.
        let a = [0.0f32, -1.0, -10.0];
        let b = [0.0f32, -3.0, -3.0];
        for t in [0.5, 1.0, 5.28, 20.0] {
            let c = calibration(t, 0.5);
            let (pa, pb) = (c.probability(&a, 0), c.probability(&b, 0));
            assert!(pb > pa, "T = {t}: {pb} <= {pa}");
        }
        // Dividing every logit by T before the softmax swaps them.
        let scaled = |z: &[f32; 3]| softmax(&z.map(|x| x / 10.0))[0];
        assert!(softmax(&b)[0] > softmax(&a)[0]);
        assert!(scaled(&a) > scaled(&b));
    }

    #[test]
    fn higher_temperature_moves_toward_one_half() {
        let z = [4.0f32, -3.0, 0.5];
        let sharp = calibration(1.0, 0.5).probability(&z, 0);
        let soft = calibration(5.0, 0.5).probability(&z, 0);
        assert!(sharp > soft && soft > 0.5);
    }

    #[test]
    fn extreme_logits_stay_finite() {
        let c = calibration(1.0, 0.5);
        assert_eq!(c.probability(&[-1e4, 1e4, 0.0], 0), 0.0);
        assert_eq!(c.probability(&[1e4, -1e4, 0.0], 0), 1.0);
        assert!(sigmoid(-800.0).is_finite() && sigmoid(800.0) == 1.0);
    }

    #[test]
    fn flags_at_or_above_the_threshold() {
        let c = calibration(1.0, 0.7);
        assert!(c.flags(0.7));
        assert!(c.flags(0.71));
        assert!(!c.flags(0.699_999));
    }

    #[test]
    fn bundled_u1_calibration_parses() {
        let c = Calibration::u1();
        assert_eq!(c.model, "nli-deberta-v3-xsmall-qint8");
        assert_eq!(c.label, "contradiction");
        assert!(c.temperature > 1.0, "the raw model is over-confident");
        assert!(c.threshold > 0.5 && c.threshold < 1.0);
    }

    #[test]
    fn bundled_u1_calibration_fits_the_u1_model_and_the_loaded_runtime() {
        let runtime = crate::classifier::loaded_runtime_version().unwrap();
        let index = Calibration::u1()
            .check_fit(
                &crate::NLI_DEBERTA_V3_XSMALL_QINT8,
                &runtime,
                crate::classifier::OPTIMIZATION_LEVEL,
            )
            .unwrap();
        assert_eq!(index, 0);
    }

    #[test]
    fn major_minor_parses_versions() {
        assert_eq!(major_minor("1.22.0"), Some((1, 22)));
        assert_eq!(major_minor("1.22"), Some((1, 22)));
        assert_eq!(major_minor(" 1.30.1 "), Some((1, 30)));
        assert_eq!(major_minor("1"), None);
        assert_eq!(major_minor("main"), None);
        assert_eq!(major_minor(""), None);
    }

    #[test]
    fn check_fit_refuses_every_mismatch() {
        let u1 = crate::NLI_DEBERTA_V3_XSMALL_QINT8;
        let all = "ORT_ENABLE_ALL";
        let ok = Calibration::u1();
        assert!(ok.check_fit(&u1, "1.22.0", all).is_ok());
        assert!(ok.check_fit(&u1, "1.22.1", all).is_ok(), "patch release");

        // Same name and revision, different files or settings.
        let other_graph = PairModel {
            onnx: crate::ModelFile {
                sha256: "7105da41f625c42eca24e9465ec99150d02a80e644659d7a1daa93a6357155d4",
                ..u1.onnx
            },
            ..u1
        };
        let other_tokenizer = PairModel {
            tokenizer: crate::ModelFile {
                sha256: "0000000000000000000000000000000000000000000000000000000000000000",
                ..u1.tokenizer
            },
            ..u1
        };
        let shorter = PairModel { max_len: 256, ..u1 };
        for model in [other_graph, other_tokenizer, shorter] {
            let err = ok.check_fit(&model, "1.22.0", all).unwrap_err();
            assert!(matches!(err, DecideError::Calibration(_)), "{err}");
        }
        // A newer runtime that still serves API 22 is not the one measured.
        for runtime in ["1.23.0", "1.30.0", "1.21.4", "2.22.0", "unknown", ""] {
            let err = ok.check_fit(&u1, runtime, all).unwrap_err();
            assert!(err.to_string().contains("onnxruntime"), "{runtime}: {err}");
        }
        assert!(ok.check_fit(&u1, "1.22.0", "ORT_ENABLE_BASIC").is_err());

        let batched = Calibration {
            one_pair_per_call: false,
            ..Calibration::u1()
        };
        assert!(
            batched.check_fit(&u1, "1.22.0", all).is_err(),
            "batched fit"
        );
        let invariant = PairModel {
            batch_invariant: true,
            ..u1
        };
        assert!(batched.check_fit(&invariant, "1.22.0", all).is_ok());

        let unknown_label = Calibration {
            label: "relevant".into(),
            ..Calibration::u1()
        };
        assert!(unknown_label.check_fit(&u1, "1.22.0", all).is_err());
    }

    #[test]
    fn rejects_bad_values() {
        let base = r#""model":"m","revision":"r","label":"contradiction","onnx_sha256":"g",
            "tokenizer_sha256":"t","max_len":512,"onnxruntime":"1.22.0",
            "optimization_level":"ORT_ENABLE_ALL","one_pair_per_call":true"#;
        for bad in [
            format!(r#"{{{base},"temperature":0,"threshold":0.5}}"#),
            format!(r#"{{{base},"temperature":-1,"threshold":0.5}}"#),
            format!(r#"{{{base},"temperature":1,"threshold":1.0}}"#),
            format!(r#"{{{base},"temperature":1,"threshold":0}}"#),
            format!(r#"{{{base},"temperature":1}}"#),
        ] {
            assert!(Calibration::from_json(&bad).is_err(), "{bad}");
        }
        let ok = format!(r#"{{{base},"temperature":2,"threshold":0.6,"extra":true}}"#);
        assert_eq!(Calibration::from_json(&ok).unwrap(), calibration(2.0, 0.6));
    }
}
