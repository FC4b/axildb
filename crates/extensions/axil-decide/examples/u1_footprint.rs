//! Where the U1 judge can run: the cost of loading it, the memory it holds
//! and the time per pair, as a one-shot process and as a long-lived host.
//!
//! ```text
//! cargo run --release -p axil-decide --example u1_footprint -- one-shot [--model-dir DIR] [--threads N]
//! cargo run --release -p axil-decide --example u1_footprint -- host [--rounds R] [--model-dir DIR] [--threads N]
//! ```
//!
//! `one-shot` is what a CLI process pays when it judges: load (hash check,
//! session, tokenizer), score one pair, exit. `host` is a process that keeps
//! the model: load once, then score every U1 pair one at a time, `--rounds`
//! times, reading resident memory after each round. Prints one JSON object.
//! Resident memory is read with `ps` (kilobytes), so it works the same on
//! macOS and Linux. `benchmarks/typed-decisions/measure_u1_footprint.py`
//! drives both modes and adds the process's wall time and peak memory.
//!
//! Needs the pinned model already in `--model-dir` (default
//! `~/.axil/models/...`); unlike `u1_bench` it never downloads, so a timing
//! never includes a transfer.

use std::path::{Path, PathBuf};
use std::time::Instant;

use axil_decide::{Calibration, LoadOptions, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
use serde_json::{json, Value};

/// This process's resident set size in kilobytes, from `ps`.
fn rss_kb() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// `p`-th percentile (0..=1) of sorted values, nearest-rank below.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

/// Every U1 pair, as `(old, new)` texts.
fn u1_pairs() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut pairs = Vec::new();
    for name in ["u1_lme.jsonl", "u1_repo.jsonl"] {
        let text = std::fs::read_to_string(root.join("benchmarks/typed-decisions/sets").join(name))
            .expect("U1 set");
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line).expect("U1 row");
            pairs.push((
                row["old"].as_str().unwrap().to_string(),
                row["new"].as_str().unwrap().to_string(),
            ));
        }
    }
    pairs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let mode = args.get(1).cloned().unwrap_or_default();
    let dir = flag("--model-dir")
        .map(PathBuf::from)
        .or_else(|| U1.default_dir())
        .expect("a model directory");
    let threads = flag("--threads").map_or(4, |n| n.parse().expect("--threads N"));
    let rounds: usize = flag("--rounds").map_or(2, |n| n.parse().expect("--rounds N"));
    let options = LoadOptions {
        intra_threads: threads,
    };

    let rss_start_kb = rss_kb();
    let started = Instant::now();
    let judge = PairClassifier::load_with(&U1, &dir, Calibration::u1(), options)
        .expect("load (is the pinned model in --model-dir?)");
    let load_ms = ms(started);
    let rss_loaded_kb = rss_kb();
    let pairs = u1_pairs();

    let mut report = match mode.as_str() {
        "one-shot" => {
            let (old, new) = &pairs[0];
            let t = Instant::now();
            let score = judge.score_pairs(&[(old, new)]).unwrap().remove(0);
            let first_pair_ms = ms(t);
            json!({
                "mode": "one-shot",
                "load_ms": load_ms,
                "first_pair_ms": first_pair_ms,
                "load_and_first_pair_ms": load_ms + first_pair_ms,
                "rss_start_kb": rss_start_kb,
                "rss_loaded_kb": rss_loaded_kb,
                "rss_end_kb": rss_kb(),
                "probability": score.probability,
            })
        }
        "host" => {
            for (old, new) in pairs.iter().take(5) {
                judge.score_pairs(&[(old, new)]).unwrap();
            }
            let mut per_round = Vec::new();
            for round in 1..=rounds {
                let mut latencies = Vec::with_capacity(pairs.len());
                let t_round = Instant::now();
                for (old, new) in &pairs {
                    let t = Instant::now();
                    judge.score_pairs(&[(old, new)]).unwrap();
                    latencies.push(ms(t));
                }
                let wall_ms = ms(t_round);
                latencies.sort_by(f64::total_cmp);
                per_round.push(json!({
                    "round": round,
                    "pairs": latencies.len(),
                    "p50_ms": percentile(&latencies, 0.5),
                    "p95_ms": percentile(&latencies, 0.95),
                    "mean_ms": latencies.iter().sum::<f64>() / latencies.len() as f64,
                    "wall_ms": wall_ms,
                    "rss_after_kb": rss_kb(),
                }));
            }
            json!({
                "mode": "host",
                "load_ms": load_ms,
                "rss_start_kb": rss_start_kb,
                "rss_loaded_kb": rss_loaded_kb,
                "warm_up_pairs": 5,
                "rounds": per_round,
            })
        }
        other => panic!("mode must be one-shot or host, got {other:?}"),
    };
    report["threads"] = json!(options.intra_threads);
    report["model"] = json!(format!("{}@{}", U1.name, U1.revision));
    report["machine"] = json!(format!(
        "{} {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    ));
    println!("{}", serde_json::to_string(&report).unwrap());
}
