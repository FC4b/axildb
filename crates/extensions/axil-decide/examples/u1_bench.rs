//! Load time, per-pair latency and Python parity of the U1 judge.
//!
//! ```text
//! cargo run --release -p axil-decide --example u1_bench -- [--model-dir DIR] [--threads N] [--out FILE.json]
//! ```
//!
//! Scores every U1 pair in `benchmarks/typed-decisions/sets` one at a time
//! (tokenize, run, calibrate), as the runtime does for this model, and
//! compares the test pairs with the Python onnxruntime fixture. Downloads the
//! pinned model if `--model-dir` (default `~/.axil/models/...`) lacks it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use axil_decide::{Calibration, LoadOptions, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
use serde_json::{json, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dir = flag("--model-dir")
        .map(PathBuf::from)
        .or_else(|| U1.default_dir())
        .expect("a model directory");
    if !U1.is_available(&dir) {
        U1.download(&dir).expect("model download");
    }

    let threads = flag("--threads").map_or(4, |n| n.parse().expect("--threads N"));
    let options = LoadOptions {
        intra_threads: threads,
    };
    // A load hashes the model files first; time that part on its own too.
    let started = Instant::now();
    U1.verify(&dir).expect("pinned files");
    let verify_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let judge = PairClassifier::load_with(&U1, &dir, Calibration::u1(), options).expect("load");
    let load_ms = started.elapsed().as_secs_f64() * 1e3;

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut rows: Vec<Value> = Vec::new();
    for name in ["u1_lme.jsonl", "u1_repo.jsonl"] {
        let text = std::fs::read_to_string(root.join("benchmarks/typed-decisions/sets").join(name))
            .expect("U1 set");
        rows.extend(
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str::<Value>(l).unwrap()),
        );
    }
    let pairs: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r["old"].as_str().unwrap(), r["new"].as_str().unwrap()))
        .collect();

    for &pair in pairs.iter().take(5) {
        judge.score_pairs(&[pair]).unwrap();
    }
    let mut latencies = Vec::with_capacity(pairs.len());
    let mut scores = Vec::with_capacity(pairs.len());
    for &pair in &pairs {
        let t = Instant::now();
        scores.push(judge.score_pairs(&[pair]).unwrap().remove(0));
        latencies.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = latencies.clone();
    sorted.sort_by(f64::total_cmp);
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() - 1) as f64 * 0.95) as usize];

    let fixture: Value =
        serde_json::from_str(include_str!("../tests/fixtures/u1_parity.json")).unwrap();
    let by_id: std::collections::HashMap<&str, usize> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r["id"].as_str().unwrap(), i))
        .collect();
    let (mut max_logit, mut max_p, mut agree, mut flagged) = (0f64, 0f64, 0usize, 0usize);
    let expected = fixture["pairs"].as_array().unwrap();
    for want in expected {
        let got = &scores[by_id[want["id"].as_str().unwrap()]];
        for (g, w) in got.logits.iter().zip(want["logits"].as_array().unwrap()) {
            max_logit = max_logit.max((*g as f64 - w.as_f64().unwrap()).abs());
        }
        max_p = max_p.max((got.probability - want["p"].as_f64().unwrap()).abs());
        agree += usize::from(got.flagged == want["flag"].as_bool().unwrap());
        flagged += usize::from(got.flagged);
    }

    let report = json!({
        "benchmark": "typed-decisions-u1-rust-runtime",
        "machine": format!("{} {}", std::env::consts::ARCH, std::env::consts::OS),
        "runtime": "axil-decide PairClassifier, ort 2.0.0-rc.10 (onnxruntime 1.22.0), CPU, Level3",
        "model": format!("{}@{}", U1.name, U1.revision),
        "threads": options.intra_threads,
        "load_ms": load_ms,
        "verify_ms": verify_ms,
        "load_note": "load_ms includes hashing the pinned files (verify_ms, timed separately before it)",
        "latency": {
            "pairs": pairs.len(),
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": latencies.iter().sum::<f64>() / latencies.len() as f64,
            "what": "one pair per call: tokenize + run + calibrate, after 5 warm-up pairs",
        },
        "parity": {
            "against": format!("Python onnxruntime {}", fixture["onnxruntime"].as_str().unwrap()),
            "pairs": expected.len(),
            "max_abs_logit_diff": max_logit,
            "max_abs_probability_diff": max_p,
            "verdicts_agree": agree,
            "flagged": flagged,
        },
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    println!("{text}");
    if let Some(out) = flag("--out") {
        std::fs::write(&out, text + "\n").expect("write --out");
    }
}
