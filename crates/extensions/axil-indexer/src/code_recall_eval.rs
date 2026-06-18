//! `code-recall-eval` — Phase 13b.9 evaluation harness.
//!
//! Runs a small set of known-answer queries against an indexed Axil
//! database and reports hit-rate / MRR / FP / token / latency metrics for
//! one or more recall strategies. The fixture is checked in (small,
//! deterministic) and the harness can also point at the Axil repo itself
//! for dogfood tracking.
//!
//! Strategies are deliberately compared *against the same DB*: we only
//! filter the recall result list, so we never have to spin up multiple
//! Axil instances per run.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use axil_core::Axil;

use crate::indexer::{TABLE_FILES, TABLE_SYMBOLS};
use crate::proxy::TABLE_CODE_PROXIES;
use crate::recall::{recall, recall_with_related, RecallResult};

/// One known-answer eval case. Expected answers are structured pointers,
/// not free-form text — this is what makes the harness usable as a
/// regression gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub query: String,
    pub expected: Vec<ExpectedPointer>,
}

/// One acceptable answer for an `EvalCase`. A pointer matches a result
/// when every populated field matches the result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpectedPointer {
    /// Path of the source file or markdown doc.
    pub path: Option<String>,
    /// Symbol name, when the expected answer is a specific function/type.
    pub symbol: Option<String>,
    /// `"file" | "symbol" | "section"`, when the expected answer kind
    /// matters.
    pub kind: Option<String>,
}

/// Strategy under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Pre-13b baseline: recall over `_idx_files` / `_idx_symbols` only.
    BaselineIndexer,
    /// Phase 13b.2/3: recall over `_idx_code_proxies` proxies.
    StructuralProxies,
    /// 13b.4: proxies + memories whose `code_refs` point at proxy hits.
    ProxiesPlusPointerMemories,
}

impl Strategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BaselineIndexer => "baseline_indexer",
            Self::StructuralProxies => "structural_proxies",
            Self::ProxiesPlusPointerMemories => "proxies_plus_pointer_memories",
        }
    }
}

/// Per-strategy aggregate metrics for a benchmark run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy: String,
    pub queries: usize,
    pub top1_file_hit_rate: f32,
    pub top3_symbol_hit_rate: f32,
    /// Mean reciprocal rank of the first correct pointer (0..1).
    pub mrr: f32,
    /// Fraction of queries where the top-3 list contained at least one
    /// wrong code pointer.
    pub fp_at_3: f32,
    /// Mean tokens emitted by recall context block per query.
    pub mean_context_tokens: f32,
    /// Estimated raw source tokens avoided by the proxy layer (vs always
    /// returning the full file).
    pub mean_raw_tokens_avoided: f32,
    /// Latency p50/p95 for the recall call itself (ms).
    pub p50_ms: f32,
    pub p95_ms: f32,
}

/// Output of a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub corpus: CorpusStats,
    pub strategies: Vec<StrategyMetrics>,
    pub indexed_at: String,
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusStats {
    pub indexed_files: usize,
    pub indexed_symbols: usize,
    pub indexed_proxies: usize,
}

/// Default eval fixture used both for the in-tree test and the dogfood
/// run. Points at structures that always exist in Axil's own repo when
/// indexed.
pub fn axil_dogfood_cases() -> Vec<EvalCase> {
    vec![
        EvalCase {
            query: "Where is recall scoring implemented?".into(),
            expected: vec![
                ExpectedPointer {
                    path: Some("crates/axil-core/src/scoring.rs".into()),
                    kind: Some("file".into()),
                    ..Default::default()
                },
                ExpectedPointer {
                    path: Some("crates/axil-core/src/db.rs".into()),
                    symbol: Some("recall".into()),
                    kind: Some("symbol".into()),
                },
            ],
        },
        EvalCase {
            query: "What handles vector search?".into(),
            expected: vec![ExpectedPointer {
                path: Some("crates/axil-vector/src/lib.rs".into()),
                kind: Some("file".into()),
                ..Default::default()
            }],
        },
        EvalCase {
            query: "Where are graph edges created?".into(),
            expected: vec![ExpectedPointer {
                path: Some("crates/axil-graph/src/edge.rs".into()),
                kind: Some("file".into()),
                ..Default::default()
            }],
        },
        EvalCase {
            query: "What task file defines SCIP ingestion?".into(),
            expected: vec![ExpectedPointer {
                path: Some("tasks/phase-13-code-graph.md".into()),
                kind: Some("section".into()),
                ..Default::default()
            }],
        },
        EvalCase {
            query: "Which tests cover memory superseding?".into(),
            expected: vec![ExpectedPointer {
                path: Some("crates/axil-tests/tests/memory_supersede.rs".into()),
                kind: Some("file".into()),
                ..Default::default()
            }],
        },
    ]
}

/// Run all strategies on the given DB and return a report.
///
/// `top_k` controls how many results we ask for per query — ranking
/// metrics use up to top_k; "top-1" and "top-3" derive from the same
/// list.
pub fn run_bench(db: &Axil, cases: &[EvalCase], top_k: usize) -> BenchReport {
    let strategies = [
        Strategy::BaselineIndexer,
        Strategy::StructuralProxies,
        Strategy::ProxiesPlusPointerMemories,
    ];
    let metrics: Vec<StrategyMetrics> = strategies
        .iter()
        .map(|s| run_strategy(db, *s, cases, top_k))
        .collect();

    let proxies = db.list(TABLE_CODE_PROXIES).map(|r| r.len()).unwrap_or(0);
    let files = db.list(TABLE_FILES).map(|r| r.len()).unwrap_or(0);
    let symbols = db.list(TABLE_SYMBOLS).map(|r| r.len()).unwrap_or(0);

    BenchReport {
        corpus: CorpusStats {
            indexed_files: files,
            indexed_symbols: symbols,
            indexed_proxies: proxies,
        },
        strategies: metrics,
        indexed_at: chrono::Utc::now().to_rfc3339(),
        git_commit: std::env::var("AXIL_BENCH_COMMIT")
            .ok()
            .or_else(detect_git_commit),
    }
}

/// Compare a `current` run against a saved `baseline` and return a list
/// of regression descriptions. An empty list means the gate passes.
///
/// Regressions:
/// 1. `top3_symbol_hit_rate` decreased for any strategy.
/// 2. `mean_context_tokens` for the `structural_proxies` strategy grew
///    by more than 10% over baseline. (Other strategies are advisory.)
pub fn compare_for_gate(baseline: &BenchReport, current: &BenchReport) -> Vec<String> {
    let mut out = Vec::new();
    for cur in &current.strategies {
        let Some(base) = baseline
            .strategies
            .iter()
            .find(|s| s.strategy == cur.strategy)
        else {
            continue;
        };
        if cur.top3_symbol_hit_rate + 1e-6 < base.top3_symbol_hit_rate {
            out.push(format!(
                "{}: top-3 symbol hit rate dropped {:.1}% → {:.1}%",
                cur.strategy,
                base.top3_symbol_hit_rate * 100.0,
                cur.top3_symbol_hit_rate * 100.0,
            ));
        }
        if cur.strategy == Strategy::StructuralProxies.as_str() && base.mean_context_tokens > 0.0 {
            let growth =
                (cur.mean_context_tokens - base.mean_context_tokens) / base.mean_context_tokens;
            if growth > 0.10 {
                out.push(format!(
                    "{}: mean context tokens grew {:.1}% ({} → {})",
                    cur.strategy,
                    growth * 100.0,
                    base.mean_context_tokens as u64,
                    cur.mean_context_tokens as u64,
                ));
            }
        }
    }
    out
}

fn detect_git_commit() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn run_strategy(
    db: &Axil,
    strategy: Strategy,
    cases: &[EvalCase],
    top_k: usize,
) -> StrategyMetrics {
    let mut top1_files = 0usize;
    let mut top3_symbols = 0usize;
    let mut mrr_sum = 0.0f32;
    let mut fp3 = 0usize;
    let mut latencies: Vec<f32> = Vec::with_capacity(cases.len());
    let mut ctx_tokens_sum = 0usize;
    let mut raw_tokens_avoided_sum = 0usize;

    for case in cases {
        let start = Instant::now();
        let results = match strategy {
            Strategy::BaselineIndexer => filter_to_baseline(
                recall(db, &case.query, top_k * 3).unwrap_or_default(),
                top_k,
            ),
            Strategy::StructuralProxies => filter_to_proxies(
                recall(db, &case.query, top_k * 3).unwrap_or_default(),
                top_k,
            ),
            Strategy::ProxiesPlusPointerMemories => {
                let rwr = recall_with_related(db, &case.query, top_k * 3, 5)
                    .unwrap_or_else(|_| crate::recall::RecallWithRelated::default());
                // Keep proxies first (the actionable pointers), then
                // append pointer-attached memories so the top-3 metric is
                // computed against pointers, not against memories that
                // simply *reference* a proxy.
                let mut out: Vec<RecallResult> = rwr
                    .primary
                    .into_iter()
                    .filter(|r| r.source == "proxy")
                    .collect();
                out.extend(rwr.related);
                out.truncate(top_k);
                out
            }
        };
        let elapsed = start.elapsed();
        latencies.push(duration_ms(elapsed));
        // Token accounting: one line per result, ~16 tokens each. Raw
        // tokens avoided estimated by file-level token counts.
        ctx_tokens_sum += estimated_context_tokens(&results);
        raw_tokens_avoided_sum += estimated_raw_tokens_avoided(db, &results);

        if first_match_rank(&case.expected, &results, |e, r| match_file(e, r)).is_some() {
            top1_files += 1;
        }
        let symbol_rank = first_match_rank(&case.expected, &results, |e, r| {
            match_symbol_or_section(e, r)
        });
        if let Some(r) = symbol_rank {
            if r < 3 {
                top3_symbols += 1;
            }
            mrr_sum += 1.0 / (r as f32 + 1.0);
        }
        if has_top3_false_positive(&case.expected, &results) {
            fp3 += 1;
        }
    }

    let n = cases.len().max(1) as f32;
    let p50 = percentile(&mut latencies.clone(), 50.0);
    let p95 = percentile(&mut latencies.clone(), 95.0);
    StrategyMetrics {
        strategy: strategy.as_str().to_string(),
        queries: cases.len(),
        top1_file_hit_rate: top1_files as f32 / n,
        top3_symbol_hit_rate: top3_symbols as f32 / n,
        mrr: mrr_sum / n,
        fp_at_3: fp3 as f32 / n,
        mean_context_tokens: ctx_tokens_sum as f32 / n,
        mean_raw_tokens_avoided: raw_tokens_avoided_sum as f32 / n,
        p50_ms: p50,
        p95_ms: p95,
    }
}

fn filter_to_baseline(mut results: Vec<RecallResult>, top_k: usize) -> Vec<RecallResult> {
    results.retain(|r| r.source == "file" || r.source == "symbol");
    results.truncate(top_k);
    results
}

fn filter_to_proxies(mut results: Vec<RecallResult>, top_k: usize) -> Vec<RecallResult> {
    results.retain(|r| r.source == "proxy");
    results.truncate(top_k);
    results
}

fn match_file(expected: &ExpectedPointer, result: &RecallResult) -> bool {
    let path_ok = match (&expected.path, &result.path) {
        (Some(e), Some(r)) => e == r,
        (None, _) => true,
        _ => false,
    };
    let kind_ok = match (&expected.kind, &result.kind) {
        (Some(e), Some(r)) => e == r,
        (None, _) => true,
        _ => true,
    };
    path_ok && kind_ok
}

fn match_symbol_or_section(expected: &ExpectedPointer, result: &RecallResult) -> bool {
    let path_ok = match (&expected.path, &result.path) {
        (Some(e), Some(r)) => e == r,
        (None, _) => true,
        _ => false,
    };
    let symbol_ok = match (&expected.symbol, &result.symbol) {
        (Some(e), Some(r)) => e == r,
        (None, _) => true,
        _ => false,
    };
    path_ok && symbol_ok
}

fn first_match_rank<F>(
    expected: &[ExpectedPointer],
    results: &[RecallResult],
    pred: F,
) -> Option<usize>
where
    F: Fn(&ExpectedPointer, &RecallResult) -> bool,
{
    for (rank, r) in results.iter().enumerate() {
        if expected.iter().any(|e| pred(e, r)) {
            return Some(rank);
        }
    }
    None
}

fn has_top3_false_positive(expected: &[ExpectedPointer], results: &[RecallResult]) -> bool {
    for r in results.iter().take(3) {
        let any_match = expected
            .iter()
            .any(|e| match_file(e, r) || match_symbol_or_section(e, r));
        if !any_match {
            return true;
        }
    }
    false
}

fn estimated_context_tokens(results: &[RecallResult]) -> usize {
    // Conservative estimate matching the context-block formatter
    // (`path:line symbol — why` ≈ 16 tokens per hit).
    results.len().saturating_mul(16)
}

fn estimated_raw_tokens_avoided(db: &Axil, results: &[RecallResult]) -> usize {
    // Sum estimated source tokens for the files we *did not* return.
    let files = db.list(TABLE_FILES).unwrap_or_default();
    let returned_paths: std::collections::HashSet<&str> =
        results.iter().filter_map(|r| r.path.as_deref()).collect();
    files
        .iter()
        .filter_map(|f| {
            let path = f.data.get("path").and_then(|v| v.as_str())?;
            if returned_paths.contains(path) {
                return None;
            }
            f.data
                .get("size_bytes")
                .and_then(|v| v.as_u64())
                .map(|b| (b as usize).div_ceil(4))
        })
        .sum()
}

fn duration_ms(d: Duration) -> f32 {
    (d.as_secs_f64() * 1000.0) as f32
}

fn percentile(latencies: &mut [f32], pct: f32) -> f32 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (latencies.len() - 1) as f32).round() as usize;
    latencies[idx.min(latencies.len() - 1)]
}

/// Render a `BenchReport` as a markdown comparison table — the format the
/// task spec asks the bench to emit.
pub fn render_markdown_table(report: &BenchReport) -> String {
    let mut out = String::new();
    out.push_str("| Strategy | Top-1 File | Top-3 Symbol | MRR | FP@3 | Ctx Tokens | Raw Avoided | p95 ms |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for s in &report.strategies {
        out.push_str(&format!(
            "| {} | {:.1}% | {:.1}% | {:.2} | {:.2} | {:.0} | {:.0} | {:.0} |\n",
            s.strategy,
            s.top1_file_hit_rate * 100.0,
            s.top3_symbol_hit_rate * 100.0,
            s.mrr,
            s.fp_at_3,
            s.mean_context_tokens,
            s.mean_raw_tokens_avoided,
            s.p95_ms,
        ));
    }
    out
}

/// Render a `BenchReport` as a plain-text comparison table for stdout.
pub fn render_plain_table(report: &BenchReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<32} {:>10} {:>13} {:>5} {:>5} {:>11} {:>13} {:>7}\n",
        "Strategy",
        "Top-1 File",
        "Top-3 Symbol",
        "MRR",
        "FP@3",
        "Ctx Tokens",
        "Raw Avoided",
        "p95 ms"
    ));
    for s in &report.strategies {
        out.push_str(&format!(
            "{:<32} {:>9.1}% {:>12.1}% {:>5.2} {:>5.2} {:>11.0} {:>13.0} {:>7.0}\n",
            s.strategy,
            s.top1_file_hit_rate * 100.0,
            s.top3_symbol_hit_rate * 100.0,
            s.mrr,
            s.fp_at_3,
            s.mean_context_tokens,
            s.mean_raw_tokens_avoided,
            s.p95_ms,
        ));
    }
    out
}

/// JSON form for persistence (`--json`).
pub fn report_to_json(report: &BenchReport) -> Value {
    serde_json::to_value(report).unwrap_or_else(|_| Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ProjectIndexer;
    use axil_core::Axil;
    use axil_core::IndexConfig;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    fn fixture(root: &std::path::Path) {
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname=\"f\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
        )
        .unwrap();
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("scoring.rs"),
            "//! Recall scoring helpers.\npub fn score() -> u32 { 0 }\n",
        )
        .unwrap();
        std::fs::write(
            src.join("vector.rs"),
            "//! Vector search backend.\npub fn search() -> u32 { 0 }\n",
        )
        .unwrap();
    }

    #[test]
    fn bench_runs_against_tiny_fixture() {
        let (db, dir) = temp_db();
        let root = dir.path();
        fixture(root);
        ProjectIndexer::new(&db, IndexConfig::default())
            .index_full(root)
            .unwrap();

        let cases = vec![
            EvalCase {
                query: "recall scoring".into(),
                expected: vec![ExpectedPointer {
                    path: Some("src/scoring.rs".into()),
                    kind: Some("file".into()),
                    ..Default::default()
                }],
            },
            EvalCase {
                query: "vector search".into(),
                expected: vec![ExpectedPointer {
                    path: Some("src/vector.rs".into()),
                    kind: Some("file".into()),
                    ..Default::default()
                }],
            },
        ];
        let report = run_bench(&db, &cases, 5);
        assert_eq!(report.strategies.len(), 3);
        assert!(report.corpus.indexed_proxies > 0);
        let md = render_markdown_table(&report);
        assert!(md.contains("structural_proxies"));
        let plain = render_plain_table(&report);
        assert!(plain.contains("structural_proxies"));
    }
}
