//! Turning a classifier's logits into a calibrated probability and a verdict.

use serde::Deserialize;

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
/// A calibration is fitted on the scores of one model file under one
/// onnxruntime version and graph optimisation level; it records the model
/// and revision it belongs to, and [`crate::PairClassifier::load`] refuses a
/// mismatch.
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
}

impl Calibration {
    /// The U1 judge: P(contradiction | premise = OLD, hypothesis = NEW) from
    /// [`crate::NLI_DEBERTA_V3_XSMALL_QINT8`], cut at a threshold chosen for a
    /// low false-positive rate. Fit and test numbers are in
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

    fn validate(&self) -> Result<()> {
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

    /// Calibrated probability of the label at `label_index` in `logits`.
    pub fn probability(&self, logits: &[f32], label_index: usize) -> f64 {
        sigmoid(one_vs_rest_logit(logits, label_index) / self.temperature)
    }

    /// Whether a calibrated probability reaches the threshold.
    pub fn flags(&self, probability: f64) -> bool {
        probability >= self.threshold
    }
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
    fn rejects_bad_values() {
        let base = r#""model":"m","revision":"r","label":"contradiction""#;
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
