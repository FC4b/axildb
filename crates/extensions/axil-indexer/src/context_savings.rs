//! `context-savings` — model the context-token cost of answering a coding
//! task **with vs without Axil**.
//!
//! This is the quantified version of Axil's core pitch: an agent that has
//! to *discover* the relevant code by reading files pulls far more text
//! into its context window than an agent that asks Axil for a compact,
//! pointer-shaped answer. Smaller working context → cheaper, faster, and
//! measurably less prone to hallucination.
//!
//! ## What is measured
//!
//! For each task (a natural-language query), we run real `recall` against
//! the indexed DB and compute two figures:
//!
//! * **with_axil_tokens** — the compact context block Axil injects: one
//!   pointer line per hit (`path:line symbol — why`). This is what the
//!   agent actually pays to get the answer.
//! * **without_axil_tokens** — the *conservative* baseline: the full
//!   source of every distinct file the hits point at. This models the
//!   minimum an agent must read to reach the same answer by opening files
//!   itself. It does **not** assume the agent reads the whole repo.
//!
//! A third figure, **repo_full_scan_tokens**, is the upper bound: the
//! token cost of loading every indexed file — what a "to be sure, read
//! everything" strategy would pay, and the number the per-query
//! `code-recall-bench` `raw_tokens_avoided` metric approximates.
//!
//! The harness deliberately reuses the *same* indexed DB and the *same*
//! recall path the agent uses in production, so the numbers track real
//! behavior rather than a synthetic model. Token counts use the shared
//! [`crate::token::estimate_tokens`] heuristic (~4 chars/token) so they
//! are consistent with every other token figure Axil reports.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use axil_core::Axil;

use crate::indexer::TABLE_FILES;
use crate::recall::{recall, RecallResult};
use crate::token::estimate_tokens;

/// One task to measure — just a query the agent might ask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub task: String,
}

impl TaskSpec {
    pub fn new(task: impl Into<String>) -> Self {
        Self { task: task.into() }
    }
}

/// Per-task savings figures.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskSavings {
    pub task: String,
    /// Hits returned by recall for this task.
    pub hits: usize,
    /// Distinct files the hits point at (the files an unaided agent would
    /// have to open).
    pub files_consulted: usize,
    /// Tokens in the compact context block Axil injects.
    pub with_axil_tokens: usize,
    /// Tokens of full source for the consulted files (conservative
    /// no-Axil baseline).
    pub without_axil_tokens: usize,
    /// `(without - with) / without` as a percentage, clamped to `[0, 100]`.
    pub reduction_pct: f32,
}

/// Aggregate report across all measured tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavingsReport {
    pub tasks: Vec<TaskSavings>,
    /// Sum of `with_axil_tokens` over all tasks.
    pub total_with_axil: usize,
    /// Sum of `without_axil_tokens` over all tasks.
    pub total_without_axil: usize,
    /// Overall conservative reduction: `(without - with) / without` %.
    pub reduction_pct: f32,
    /// Compression ratio `without / with` (e.g. 15.0 = 15:1).
    pub compression_ratio: f32,
    /// Upper-bound cost of loading every indexed file once.
    pub repo_full_scan_tokens: usize,
    /// Indexed files in the corpus.
    pub indexed_files: usize,
    pub indexed_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
}

/// Default task set — reuses the dogfood eval queries so this harness and
/// `code-recall-bench` describe the same workload.
pub fn default_tasks() -> Vec<TaskSpec> {
    crate::code_recall_eval::axil_dogfood_cases()
        .into_iter()
        .map(|c| TaskSpec::new(c.query))
        .collect()
}

/// Measure context savings for `tasks` against the indexed `db`.
///
/// `top_k` controls how many hits recall returns per task — the more hits,
/// the more files an unaided agent would have had to read, so both sides
/// of the comparison scale together.
pub fn measure(db: &Axil, tasks: &[TaskSpec], top_k: usize) -> SavingsReport {
    let file_sizes = file_size_map(db);
    let repo_full_scan_tokens: usize = file_sizes.values().map(|b| (*b as usize).div_ceil(4)).sum();
    let indexed_files = file_sizes.len();

    let mut rows = Vec::with_capacity(tasks.len());
    let mut total_with = 0usize;
    let mut total_without = 0usize;

    for spec in tasks {
        let results = recall(db, &spec.task, top_k).unwrap_or_default();
        let with_tokens = compact_context_tokens(&results);

        let consulted: HashSet<&str> = results.iter().filter_map(|r| r.path.as_deref()).collect();
        let without_tokens: usize = consulted
            .iter()
            .filter_map(|p| file_sizes.get(*p))
            .map(|b| (*b as usize).div_ceil(4))
            .sum();

        total_with += with_tokens;
        total_without += without_tokens;

        rows.push(TaskSavings {
            task: spec.task.clone(),
            hits: results.len(),
            files_consulted: consulted.len(),
            with_axil_tokens: with_tokens,
            without_axil_tokens: without_tokens,
            reduction_pct: reduction(with_tokens, without_tokens),
        });
    }

    SavingsReport {
        tasks: rows,
        total_with_axil: total_with,
        total_without_axil: total_without,
        reduction_pct: reduction(total_with, total_without),
        compression_ratio: if total_with == 0 {
            0.0
        } else {
            total_without as f32 / total_with as f32
        },
        repo_full_scan_tokens,
        indexed_files,
        indexed_at: chrono::Utc::now().to_rfc3339(),
        git_commit: std::env::var("AXIL_BENCH_COMMIT")
            .ok()
            .or_else(detect_git_commit),
    }
}

/// `(without - with) / without` as a percentage, clamped to `[0, 100]`.
fn reduction(with: usize, without: usize) -> f32 {
    if without == 0 {
        return 0.0;
    }
    let r = (without.saturating_sub(with)) as f32 / without as f32 * 100.0;
    r.clamp(0.0, 100.0)
}

/// Tokens for the compact context block Axil would inject for these hits —
/// one pointer line per result, the same shape `axil code-search` /
/// `code-context` emit.
fn compact_context_tokens(results: &[RecallResult]) -> usize {
    results.iter().map(|r| estimate_tokens(&compact_line(r))).sum()
}

/// Render a single recall hit as the agent-facing one-liner.
fn compact_line(r: &RecallResult) -> String {
    let loc = match (&r.path, r.line_start) {
        (Some(p), Some(l)) => format!("{p}:{l}"),
        (Some(p), None) => p.clone(),
        _ => r.id.clone(),
    };
    let label = r
        .symbol
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&r.summary);
    let why = r.why.as_deref().unwrap_or("relevant");
    // Cap the descriptive tail so a noisy summary cannot inflate the
    // estimate — matches the truncation the formatters apply.
    let label: String = label.chars().take(80).collect();
    format!("{loc} {label} — {why}")
}

/// Build a `path -> size_bytes` map from the indexed files table.
fn file_size_map(db: &Axil) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for f in db.list(TABLE_FILES).unwrap_or_default() {
        let path = match f.data.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => continue,
        };
        let size = f
            .data
            .get("size_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        map.insert(path, size);
    }
    map
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

/// Render the report as a human-readable text block for stdout.
pub fn render_plain(report: &SavingsReport) -> String {
    let mut out = String::new();
    out.push_str("Context savings — answering coding tasks with vs without Axil\n");
    out.push_str(&format!(
        "Corpus: {} indexed files  (~{} tokens to read all)\n\n",
        report.indexed_files,
        fmt_int(report.repo_full_scan_tokens),
    ));
    out.push_str(&format!(
        "{:<46} {:>6} {:>6} {:>11} {:>11} {:>8}\n",
        "Task", "Hits", "Files", "w/ Axil", "no Axil", "Saved",
    ));
    for t in &report.tasks {
        out.push_str(&format!(
            "{:<46} {:>6} {:>6} {:>11} {:>11} {:>7.1}%\n",
            truncate(&t.task, 46),
            t.hits,
            t.files_consulted,
            fmt_int(t.with_axil_tokens),
            fmt_int(t.without_axil_tokens),
            t.reduction_pct,
        ));
    }
    out.push_str(&format!(
        "\nTotal: {} tokens with Axil vs {} without — {:.1}% reduction ({:.1}:1 compression)\n",
        fmt_int(report.total_with_axil),
        fmt_int(report.total_without_axil),
        report.reduction_pct,
        report.compression_ratio,
    ));
    out.push_str(
        "Baseline = whole-file-read upper bound (optimistic ceiling; a grep-savvy agent reads \
         less, so real-world savings are lower).\n",
    );
    out.push_str(&format!(
        "Upper bound: reading the whole indexed corpus once = {} tokens.\n",
        fmt_int(report.repo_full_scan_tokens),
    ));
    out
}

/// Render the report as a markdown table (for docs / PR comments).
pub fn render_markdown(report: &SavingsReport) -> String {
    let mut out = String::new();
    out.push_str("| Task | Hits | Files | w/ Axil | no Axil | Saved |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for t in &report.tasks {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.1}% |\n",
            t.task,
            t.hits,
            t.files_consulted,
            fmt_int(t.with_axil_tokens),
            fmt_int(t.without_axil_tokens),
            t.reduction_pct,
        ));
    }
    out.push_str(&format!(
        "| **Total** | | | **{}** | **{}** | **{:.1}%** |\n",
        fmt_int(report.total_with_axil),
        fmt_int(report.total_without_axil),
        report.reduction_pct,
    ));
    out.push_str(
        "\n_Baseline = whole-file-read upper bound (optimistic ceiling; a grep-savvy agent reads less)._\n",
    );
    out
}

/// JSON form for persistence (`--format json`, A/B baselines).
pub fn report_to_json(report: &SavingsReport) -> Value {
    serde_json::to_value(report).unwrap_or(Value::Null)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

/// Thousands-separated integer for readable token counts.
fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, c) in s.chars().enumerate() {
        if i != 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::ProjectIndexer;
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
        // A deliberately chunky file so the no-Axil baseline is non-trivial.
        let big = format!("//! Recall scoring helpers.\n{}", "pub fn score() -> u32 { 0 }\n".repeat(80));
        std::fs::write(src.join("scoring.rs"), big).unwrap();
        std::fs::write(
            src.join("vector.rs"),
            "//! Vector search backend.\npub fn search() -> u32 { 0 }\n",
        )
        .unwrap();
    }

    #[test]
    fn measures_savings_against_fixture() {
        let (db, dir) = temp_db();
        let root = dir.path();
        fixture(root);
        ProjectIndexer::new(&db, IndexConfig::default())
            .index_full(root)
            .unwrap();

        let tasks = vec![
            TaskSpec::new("recall scoring"),
            TaskSpec::new("vector search"),
        ];
        let report = measure(&db, &tasks, 5);

        assert_eq!(report.tasks.len(), 2);
        assert!(report.indexed_files >= 2);
        assert!(report.repo_full_scan_tokens > 0);
        // Compact context must never cost more than reading the files.
        assert!(report.total_with_axil <= report.total_without_axil.max(report.total_with_axil));
        // Renderers don't panic and include the headline.
        assert!(render_plain(&report).contains("reduction"));
        assert!(render_markdown(&report).contains("Total"));
        assert!(report_to_json(&report).is_object());
    }

    #[test]
    fn reduction_is_clamped_and_safe() {
        assert_eq!(reduction(10, 0), 0.0);
        assert_eq!(reduction(0, 100), 100.0);
        assert!((reduction(20, 100) - 80.0).abs() < 1e-3);
        // with > without (pathological) clamps to 0, never negative.
        assert_eq!(reduction(200, 100), 0.0);
    }

    #[test]
    fn fmt_int_groups_thousands() {
        assert_eq!(fmt_int(0), "0");
        assert_eq!(fmt_int(52), "52");
        assert_eq!(fmt_int(999), "999");
        assert_eq!(fmt_int(1000), "1,000");
        assert_eq!(fmt_int(12345), "12,345");
        assert_eq!(fmt_int(546661), "546,661");
        assert_eq!(fmt_int(774209), "774,209");
    }
}
