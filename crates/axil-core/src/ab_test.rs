//! A/B testing harness for comparing recall configurations.
//!
//! Allows side-by-side comparison of two `AbTestConfig` variants,
//! producing a human-readable winner determination.

use serde::{Deserialize, Serialize};

use crate::bench_metrics::BenchmarkResult;

/// A recall configuration to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestConfig {
    /// Human-readable name for this configuration variant.
    pub name: String,
    /// Number of results to return per query.
    pub top_k: usize,
    /// Whether full-text search is enabled.
    pub use_fts: bool,
    /// Whether graph traversal is enabled.
    pub use_graph: bool,
    /// Whether re-ranking is enabled.
    pub use_rerank: bool,
}

/// Result of an A/B comparison between two configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestResult {
    /// First configuration.
    pub config_a: AbTestConfig,
    /// Second configuration.
    pub config_b: AbTestConfig,
    /// Benchmark result for configuration A.
    pub result_a: BenchmarkResult,
    /// Benchmark result for configuration B.
    pub result_b: BenchmarkResult,
    /// Winner summary, e.g. `"config_b (+3.7% accuracy, +14ms latency)"`.
    pub winner: String,
    /// Multi-line human-readable comparison.
    pub summary: String,
}

/// Generate a human-readable comparison of two benchmark results.
///
/// Reports the accuracy delta, latency delta, token delta, and
/// efficiency delta, then declares a winner based on efficiency.
pub fn compare_configs(a: &BenchmarkResult, b: &BenchmarkResult) -> String {
    let acc_diff = b.accuracy - a.accuracy;
    let lat_diff = b.latency_p50_ms - a.latency_p50_ms;
    let tok_diff = b.avg_context_tokens as i64 - a.avg_context_tokens as i64;
    let eff_diff = b.mem_efficiency - a.mem_efficiency;

    let winner = if eff_diff > 0.0 {
        format!(
            "{} ({:+.1}% accuracy, {:+.0}ms latency)",
            b.benchmark_name, acc_diff, lat_diff
        )
    } else if eff_diff < 0.0 {
        format!(
            "{} ({:+.1}% accuracy, {:+.0}ms latency)",
            a.benchmark_name, -acc_diff, -lat_diff
        )
    } else {
        "Tie (identical efficiency)".to_string()
    };

    let mut lines = Vec::new();
    lines.push(format!(
        "A: {} — accuracy {:.1}%, p50 {:.1}ms, {} tokens, efficiency {:.2}",
        a.benchmark_name, a.accuracy, a.latency_p50_ms, a.avg_context_tokens, a.mem_efficiency
    ));
    lines.push(format!(
        "B: {} — accuracy {:.1}%, p50 {:.1}ms, {} tokens, efficiency {:.2}",
        b.benchmark_name, b.accuracy, b.latency_p50_ms, b.avg_context_tokens, b.mem_efficiency
    ));
    lines.push(String::new());
    lines.push(format!("Accuracy delta: {:+.1}%", acc_diff));
    lines.push(format!("Latency delta:  {:+.1}ms", lat_diff));
    lines.push(format!("Token delta:    {:+}", tok_diff));
    lines.push(format!("Efficiency delta: {:+.2}", eff_diff));
    lines.push(String::new());
    lines.push(format!("Winner: {}", winner));

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn make_result(
        name: &str,
        accuracy: f64,
        tokens: usize,
        p50: f64,
        efficiency: f64,
    ) -> BenchmarkResult {
        BenchmarkResult {
            benchmark_name: name.to_string(),
            accuracy,
            avg_context_tokens: tokens,
            latency_p50_ms: p50,
            latency_p99_ms: p50 * 2.0,
            mem_efficiency: efficiency,
            per_category: BTreeMap::new(),
            timestamp: "2026-04-11T00:00:00Z".to_string(),
            axil_version: "0.1.0".to_string(),
        }
    }

    #[test]
    fn test_compare_b_wins() {
        let a = make_result("baseline", 80.0, 2000, 10.0, 40.0);
        let b = make_result("with_graph", 85.0, 1800, 14.0, 47.22);
        let report = compare_configs(&a, &b);
        assert!(report.contains("Winner: with_graph"));
        assert!(report.contains("+5.0% accuracy"));
    }

    #[test]
    fn test_compare_a_wins() {
        let a = make_result("optimized", 90.0, 1000, 8.0, 90.0);
        let b = make_result("default", 80.0, 2000, 10.0, 40.0);
        let report = compare_configs(&a, &b);
        assert!(report.contains("Winner: optimized"));
    }

    #[test]
    fn test_compare_tie() {
        let a = make_result("alpha", 80.0, 2000, 10.0, 40.0);
        let b = make_result("beta", 80.0, 2000, 10.0, 40.0);
        let report = compare_configs(&a, &b);
        assert!(report.contains("Tie"));
    }
}
