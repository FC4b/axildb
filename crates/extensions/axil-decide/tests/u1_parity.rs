//! Parity of the Rust pair classifier with Python onnxruntime.
//!
//! These tests need model downloads (87 MB, and 284 MB for the fp32 one) and
//! the repo's `benchmarks/typed-decisions/sets`, so they are ignored by
//! default. Run them with
//! `cargo test --release -p axil-decide --test u1_parity -- --ignored --nocapture`.
//! `AXIL_DECIDE_MODEL_DIR` / `AXIL_DECIDE_FP32_DIR` point them at existing
//! model directories; otherwise the files download to `~/.axil/models/`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use axil_decide::{
    Calibration, ModelFile, PairClassifier, PairModel, NLI_DEBERTA_V3_XSMALL_QINT8 as U1,
};
use serde_json::Value;

/// The fixture's logits and probabilities come from Python onnxruntime 1.22.0
/// (`benchmarks/typed-decisions/calibrate_u1.py`).
const FIXTURE: &str = include_str!("fixtures/u1_parity.json");
const TOLERANCE: f64 = 1e-4;

/// `(old, new)` text of every U1 pair, by id.
fn u1_texts() -> HashMap<String, (String, String)> {
    let sets =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../benchmarks/typed-decisions/sets");
    let mut out = HashMap::new();
    for name in ["u1_lme.jsonl", "u1_repo.jsonl"] {
        let text = std::fs::read_to_string(sets.join(name)).expect("U1 set");
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line).unwrap();
            out.insert(
                row["id"].as_str().unwrap().to_string(),
                (
                    row["old"].as_str().unwrap().to_string(),
                    row["new"].as_str().unwrap().to_string(),
                ),
            );
        }
    }
    out
}

fn ready(model: &PairModel, env: &str) -> PathBuf {
    let dir = std::env::var_os(env)
        .map(PathBuf::from)
        .or_else(|| model.default_dir())
        .expect("a model directory");
    if !model.is_available(&dir) {
        model.download(&dir).expect("model download");
    }
    dir
}

fn argmax(xs: &[f64]) -> usize {
    (0..xs.len())
        .max_by(|&a, &b| xs[a].total_cmp(&xs[b]))
        .unwrap()
}

#[test]
#[ignore = "downloads an 87 MB model; run with --ignored"]
fn u1_scores_match_python_onnxruntime() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let calibration = Calibration::u1();
    assert_eq!(fixture["model"].as_str(), Some(U1.name));
    assert_eq!(fixture["revision"].as_str(), Some(U1.revision));
    assert_eq!(
        fixture["temperature"].as_f64(),
        Some(calibration.temperature)
    );
    assert_eq!(fixture["threshold"].as_f64(), Some(calibration.threshold));

    let judge =
        PairClassifier::load(&U1, &ready(&U1, "AXIL_DECIDE_MODEL_DIR"), calibration).unwrap();
    let texts = u1_texts();
    let pairs = fixture["pairs"].as_array().unwrap();
    let inputs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|p| {
            let (old, new) = &texts[p["id"].as_str().unwrap()];
            (old.as_str(), new.as_str())
        })
        .collect();

    let one_by_one: Vec<_> = inputs
        .iter()
        .map(|&pair| judge.score_pairs(&[pair]).unwrap().remove(0))
        .collect();
    // Scoring all pairs in one call must give the same answers: this model is
    // not batch-invariant, so the runtime still runs one pair per call.
    assert_eq!(judge.score_pairs(&inputs).unwrap(), one_by_one);

    let (mut max_logit, mut max_p, mut flagged) = (0f64, 0f64, 0usize);
    for (want, got) in pairs.iter().zip(&one_by_one) {
        let id = want["id"].as_str().unwrap();
        let logits: Vec<f64> = want["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got_logits: Vec<f64> = got.logits.iter().map(|&x| x as f64).collect();
        assert_eq!(
            got.tokens as u64,
            want["tokens"].as_u64().unwrap(),
            "{id}: tokens"
        );
        assert_eq!(argmax(&got_logits), argmax(&logits), "{id}: argmax");
        assert_eq!(
            got.flagged,
            want["flag"].as_bool().unwrap(),
            "{id}: verdict"
        );
        for (g, w) in got_logits.iter().zip(&logits) {
            max_logit = max_logit.max((g - w).abs());
        }
        max_p = max_p.max((got.probability - want["p"].as_f64().unwrap()).abs());
        flagged += usize::from(got.flagged);
    }
    println!(
        "{} pairs: max |logit diff| {max_logit:.2e}, max |p diff| {max_p:.2e}, {flagged} flagged",
        pairs.len()
    );
    assert!(max_p <= TOLERANCE, "max |p diff| {max_p:e} > {TOLERANCE:e}");
}

/// The fp32 export of the same model, which is batch-invariant: the only
/// model here that exercises the padded-batch path.
const FP32: PairModel = PairModel {
    name: "nli-deberta-v3-xsmall-fp32",
    onnx: ModelFile {
        remote: "onnx/model.onnx",
        local: "model.onnx",
        sha256: "7105da41f625c42eca24e9465ec99150d02a80e644659d7a1daa93a6357155d4",
        size: 284_200_859,
    },
    batch_invariant: true,
    ..U1
};

#[test]
#[ignore = "downloads a 284 MB model; run with --ignored"]
fn padded_batches_match_single_pairs() {
    let calibration = Calibration {
        model: FP32.name.into(),
        temperature: 1.0,
        ..Calibration::u1()
    };
    let judge =
        PairClassifier::load(&FP32, &ready(&FP32, "AXIL_DECIDE_FP32_DIR"), calibration).unwrap();
    let texts = u1_texts();
    let mut ids: Vec<&String> = texts.keys().collect();
    ids.sort();
    let inputs: Vec<(&str, &str)> = ids
        .iter()
        .take(40)
        .map(|id| (texts[*id].0.as_str(), texts[*id].1.as_str()))
        .collect();

    let batched = judge.score_pairs(&inputs).unwrap();
    let mut max_p = 0f64;
    for (pair, b) in inputs.iter().zip(&batched) {
        let single = judge.score_pairs(&[*pair]).unwrap().remove(0);
        assert_eq!(single.tokens, b.tokens);
        max_p = max_p.max((single.probability - b.probability).abs());
    }
    println!(
        "{} pairs: max |p diff| batched vs single {max_p:.2e}",
        inputs.len()
    );
    assert!(max_p <= TOLERANCE, "max |p diff| {max_p:e} > {TOLERANCE:e}");
}
