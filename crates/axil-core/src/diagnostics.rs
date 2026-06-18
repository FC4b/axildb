use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Severity level for a diagnostic check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warning,
    Error,
}

impl Severity {
    /// Returns the more severe of two levels.
    pub fn max(self, other: Self) -> Self {
        match (self, other) {
            (Severity::Error, _) | (_, Severity::Error) => Severity::Error,
            (Severity::Warning, _) | (_, Severity::Warning) => Severity::Warning,
            _ => Severity::Ok,
        }
    }
}

/// Result of a single health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: Severity,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

/// Full doctor report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub status: Severity,
    pub checks: Vec<CheckResult>,
}

impl DoctorReport {
    /// Create a report from a list of checks, computing overall status.
    pub fn from_checks(checks: Vec<CheckResult>) -> Self {
        let status = checks.iter().fold(Severity::Ok, |acc, c| acc.max(c.status));
        Self { status, checks }
    }

    /// Add a check result and update the overall status.
    pub fn add_check(&mut self, check: CheckResult) {
        self.status = self.status.max(check.status);
        self.checks.push(check);
    }

    /// Exit code: 0 = all ok, 1 = warnings, 2 = errors.
    pub fn exit_code(&self) -> i32 {
        match self.status {
            Severity::Ok => 0,
            Severity::Warning => 1,
            Severity::Error => 2,
        }
    }
}

/// Comprehensive database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub database: DatabaseMeta,
    pub records: RecordStats,
    pub indexes: IndexStats,
    pub performance: Value,
}

/// Database metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMeta {
    pub path: String,
    pub size_bytes: u64,
    pub size_human: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_write: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_read: Option<String>,
}

/// Record counts and per-table breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordStats {
    pub total: usize,
    pub tables: Value,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub vectors: usize,
    pub vector_dimensions: usize,
    pub fts_enabled: bool,
    pub edges: usize,
    pub timeseries_entries: usize,
}

/// Benchmark result for a single test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub name: String,
    pub ops_per_sec: f64,
    pub avg_ms: f64,
    pub iterations: usize,
}

/// Full benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub benchmarks: Vec<BenchResult>,
    pub system: SystemInfo,
}

/// System information for benchmark context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
}

impl SystemInfo {
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }
}

// ── Self-Healing Types (Phase 5c) ──────────────────────────────────────────

/// Report from a compaction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactReport {
    pub compacted: bool,
    pub purged_expired: usize,
    pub purged_superseded: usize,
    pub cleaned_orphaned_edges: usize,
    pub cleaned_orphaned_vectors: usize,
    pub cleaned_orphaned_fts: usize,
    pub freed_estimate_bytes: u64,
    pub duration_ms: f64,
}

/// Report from a vector index rebuild.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRebuildReport {
    pub rebuilt: bool,
    pub reason: String,
    pub old_size: usize,
    pub new_size: usize,
    pub deleted_removed: usize,
    pub duration_ms: f64,
}

/// A single action taken during a heal operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealAction {
    pub action: String,
    pub result: String,
}

/// Full heal report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfHealReport {
    pub healed: bool,
    pub actions: Vec<HealAction>,
    pub duration_ms: f64,
}

/// A detected problem with optional auto-fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDetection {
    pub detector: String,
    pub severity: Severity,
    pub message: String,
    pub recommendation: String,
    pub auto_fixable: bool,
}

/// Comprehensive health + recommendations report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub generated_at: String,
    pub overall_health: String,
    pub score: u32,
    pub summary: String,
    pub sections: HealthSections,
    pub recommendations: Vec<Recommendation>,
}

/// Sub-sections of the health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSections {
    pub storage: StorageSection,
    pub performance: PerformanceSection,
    pub indexes: IndexSection,
    pub data_quality: DataQualitySection,
}

/// Storage health section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSection {
    pub status: Severity,
    pub size: String,
    pub record_count: usize,
    pub table_count: usize,
}

/// Performance health section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSection {
    pub status: Severity,
    pub slow_queries_count: usize,
}

/// Index health section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSection {
    pub status: Severity,
    pub vectors: usize,
    pub edges: usize,
    pub orphaned_edges: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_deletion_ratio: Option<f64>,
}

/// Data quality section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualitySection {
    pub status: Severity,
    pub expired_records: usize,
    pub superseded_records: usize,
    pub live_ratio: f64,
}

/// A recommendation from the health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub priority: String,
    pub action: String,
    pub auto_fixable: bool,
    pub command: String,
}

/// Trend data for a single metric over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTrend {
    pub start: f64,
    pub end: f64,
    pub change: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alert: Option<String>,
}

/// Full trend report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendReport {
    pub period: String,
    pub snapshots: usize,
    pub trends: std::collections::BTreeMap<String, MetricTrend>,
}

/// A point-in-time metrics snapshot stored in _axil_metrics_history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistoryEntry {
    pub timestamp: String,
    pub record_count: usize,
    pub file_size_bytes: u64,
    pub vector_count: usize,
    pub edge_count: usize,
    pub live_ratio: f64,
}

/// Format bytes into a human-readable string.
pub fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
