//! MemEfficiency metric and competitor comparison types.
//!
//! Provides the `BenchmarkResult` type for tracking benchmark accuracy,
//! latency, and token usage, plus `ComparisonReport` for ranking against
//! known competitor baselines.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Result of running a recall-quality benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Human-readable benchmark name (e.g. "LongMemEval-single-session").
    pub benchmark_name: String,
    /// Overall accuracy as a percentage (0.0 – 100.0).
    pub accuracy: f64,
    /// Average context tokens returned per query.
    pub avg_context_tokens: usize,
    /// Median latency in milliseconds.
    pub latency_p50_ms: f64,
    /// 99th-percentile latency in milliseconds.
    pub latency_p99_ms: f64,
    /// Composite efficiency: `accuracy / tokens * 1000`.
    pub mem_efficiency: f64,
    /// Per-category accuracy breakdown (category name → accuracy %).
    pub per_category: BTreeMap<String, f64>,
    /// ISO-8601 timestamp of when the benchmark was run.
    pub timestamp: String,
    /// Axil version string used for the run.
    pub axil_version: String,
}

/// Compute the MemEfficiency metric.
///
/// Formula: `accuracy / avg_tokens * 1000`
///
/// Returns 0.0 when `avg_tokens` is zero to avoid division by zero.
pub fn compute_mem_efficiency(accuracy: f64, avg_tokens: usize) -> f64 {
    if avg_tokens == 0 {
        return 0.0;
    }
    accuracy / avg_tokens as f64 * 1000.0
}

/// Format the compact MemScore string used in reports and CLI output.
///
/// Example output: `"85% / 12ms / 950tok"`
pub fn format_memscore(accuracy: f64, latency_ms: f64, tokens: usize) -> String {
    format!("{:.0}% / {:.0}ms / {}tok", accuracy, latency_ms, tokens)
}

/// A known competitor's baseline numbers for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorResult {
    /// Competitor name.
    pub name: String,
    /// Reported accuracy (0.0 – 100.0).
    pub accuracy: f64,
    /// Average context tokens per query.
    pub avg_tokens: usize,
    /// Computed efficiency (`accuracy / avg_tokens * 1000`).
    pub efficiency: f64,
}

/// Side-by-side comparison of Axil against known competitors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Axil's benchmark result.
    pub axil: BenchmarkResult,
    /// Pre-populated competitor baselines.
    pub competitors: Vec<CompetitorResult>,
}

/// Return pre-populated competitor baselines sourced from public benchmarks.
pub fn competitor_baselines() -> Vec<CompetitorResult> {
    vec![
        CompetitorResult {
            name: "MemPalace".to_string(),
            accuracy: 96.6,
            avg_tokens: 8000,
            efficiency: compute_mem_efficiency(96.6, 8000),
        },
        CompetitorResult {
            name: "Memvid".to_string(),
            accuracy: 85.7,
            avg_tokens: 4000,
            efficiency: compute_mem_efficiency(85.7, 4000),
        },
        CompetitorResult {
            name: "Mem0".to_string(),
            accuracy: 68.4,
            avg_tokens: 6000,
            efficiency: compute_mem_efficiency(68.4, 6000),
        },
        CompetitorResult {
            name: "Zep".to_string(),
            accuracy: 66.0,
            avg_tokens: 5000,
            efficiency: compute_mem_efficiency(66.0, 5000),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mem_efficiency() {
        let eff = compute_mem_efficiency(85.0, 1000);
        assert!((eff - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_mem_efficiency_zero_tokens() {
        assert_eq!(compute_mem_efficiency(90.0, 0), 0.0);
    }

    #[test]
    fn test_format_memscore() {
        let s = format_memscore(85.0, 12.0, 950);
        assert_eq!(s, "85% / 12ms / 950tok");
    }

    #[test]
    fn test_competitor_baselines_count() {
        let baselines = competitor_baselines();
        assert_eq!(baselines.len(), 4);
        assert_eq!(baselines[0].name, "MemPalace");
    }

    #[test]
    fn test_competitor_efficiency_values() {
        let baselines = competitor_baselines();
        // MemPalace: 96.6 / 8000 * 1000 = 12.075
        assert!((baselines[0].efficiency - 12.075).abs() < 0.001);
        // Memvid: 85.7 / 4000 * 1000 = 21.425
        assert!((baselines[1].efficiency - 21.425).abs() < 0.001);
    }
}
