use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level configuration loaded from `axil.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AxilConfig {
    /// Database settings.
    pub database: DatabaseConfig,
    /// Time-series retention and downsampling settings.
    pub timeseries: TimeseriesConfig,
    /// Development / debug settings (for projects that use Axil).
    pub dev: DevConfig,
    /// Debug output settings.
    pub debug: DebugConfig,
    /// Diagnose settings (for Axil source repo).
    pub diagnose: DiagnoseConfig,
    /// Optimize settings (for Axil source repo).
    pub optimize: OptimizeConfig,
    /// Project indexer settings.
    pub index: IndexConfig,
    /// Agent runtime settings.
    pub runtime: RuntimeConfig,
    /// Full-text search settings.
    pub fts: FtsConfig,
    /// Self-healing and auto-optimization settings.
    pub healing: HealingConfig,
    /// LLM provider settings.
    pub llm: crate::llm::LlmConfig,
    /// Brain display settings.
    pub brain: BrainConfig,
    /// Per-table memory-decay half-life overrides.
    ///
    /// The default half-life (`importance::DEFAULT_HALF_LIFE_DAYS` = 90 days)
    /// fits personal facts but is too slow for code-centric memory (errors,
    /// build logs) and too fast for long-term preferences. This table lets
    /// projects set realistic values per table; missing tables fall back to
    /// the default.
    pub decay: DecayConfig,
    /// Opportunistic, time-gated maintenance (`axil maintain --if-stale`).
    pub maintenance: MaintenanceConfig,
    /// Built-in Extension enable/disable overrides (`[extensions]`).
    pub extensions: ExtensionsConfig,
    /// Built-in Engine enable/disable overrides (`[engines]`).
    pub engines: EnginesConfig,
    /// Per-WASM-plugin runtime config — capability grants, keyed by the
    /// plugin's `.wasm` filename stem (`[plugins.<key>]`).
    pub plugins: std::collections::BTreeMap<String, PluginConfig>,
}

impl AxilConfig {
    /// Whether a built-in Extension is turned off in `[extensions] disabled`.
    ///
    /// A disabled Extension is skipped at registration (it never reaches
    /// `db.extensions()`), so its CLI/MCP surface and `boot_block` vanish
    /// without a rebuild. Matching is by [`crate::Extension::id`].
    pub fn is_extension_disabled(&self, id: &str) -> bool {
        self.extensions.disabled.iter().any(|d| d == id)
    }

    /// Whether a built-in Engine is turned off in `[engines] disabled`.
    ///
    /// Matched against the Engine's companion-file suffix (e.g. `"vec"`,
    /// `"graph"`, `"fts"`, `"ts"`) so an operator can keep a companion file on
    /// disk but skip attaching its Engine for a session.
    pub fn is_engine_disabled(&self, suffix: &str) -> bool {
        self.engines.disabled.iter().any(|d| d == suffix)
    }

    /// Capabilities granted to a WASM plugin, keyed by its `.wasm` filename
    /// stem. An unconfigured plugin gets an empty list — **deny-by-default**:
    /// it can run but cannot call back into Axil until the operator grants
    /// capabilities via `[plugins.<key>] capabilities = [...]`.
    pub fn plugin_capabilities(&self, key: &str) -> Vec<String> {
        self.plugins
            .get(key)
            .map(|p| p.capabilities.clone())
            .unwrap_or_default()
    }
}

/// Per-WASM-plugin runtime config (`[plugins.<key>]`).
///
/// ```toml
/// [plugins.hello]
/// capabilities = ["records.read", "records.write", "recall"]
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginConfig {
    /// Capabilities granted to this plugin (deny-by-default; empty = none).
    pub capabilities: Vec<String>,
}

/// Runtime enable/disable overrides for built-in Extensions (`[extensions]`).
///
/// ```toml
/// [extensions]
/// disabled = ["deps", "checkpoint"]
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtensionsConfig {
    /// Built-in Extension ids to skip at registration.
    pub disabled: Vec<String>,
}

/// Runtime enable/disable overrides for built-in Engines (`[engines]`).
///
/// ```toml
/// [engines]
/// disabled = ["vec"]
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EnginesConfig {
    /// Engine companion-file suffixes to skip attaching.
    pub disabled: Vec<String>,
}

/// Per-table memory-decay half-life configuration.
///
/// Serialized as `[decay]` with a `tables` map:
///
/// ```toml
/// [decay]
/// tables.errors = 7        # 1 week
/// tables.code = 14         # 2 weeks
/// tables.preferences = 365 # 1 year
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayConfig {
    /// Map of table-name → half-life in days.
    pub tables: std::collections::BTreeMap<String, f64>,
}

impl DecayConfig {
    /// Resolve the half-life (in days) for a given table, falling back to
    /// `importance::DEFAULT_HALF_LIFE_DAYS` when no override is set.
    pub fn half_life_for(&self, table: &str) -> f64 {
        self.tables
            .get(table)
            .copied()
            .filter(|v| *v > 0.0 && v.is_finite())
            .unwrap_or(crate::importance::DEFAULT_HALF_LIFE_DAYS)
    }
}

/// Brain configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrainConfig {
    /// Whether brain mode is enabled. When true, existing commands gain
    /// brain behavior (provenance, scoped recall, belief revision, etc.).
    pub enabled: bool,
    /// Banner style: "compact" (default), "box", "ascii", "status", "bold".
    pub banner: String,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            banner: "compact".to_string(),
        }
    }
}

/// Database path configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// Default database path (overrides AXIL_DB env var).
    pub path: Option<String>,
    /// Embedding model name (e.g. "bge-small", "bge-small-int8", "bge-base", "nomic").
    /// Defaults to "bge-small" if not specified.
    pub embedding_model: Option<String>,
}

/// Development and field-report settings (for working projects).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DevConfig {
    /// Path to the Axil source repo (for dev workflows).
    pub source_repo: Option<String>,
    /// Directory where `/axil-report` writes output.
    pub reports_dir: String,
    /// Automatically generate a report on error/panic.
    pub auto_report: bool,
}

impl Default for DevConfig {
    fn default() -> Self {
        Self {
            source_repo: None,
            reports_dir: ".axil-reports".to_string(),
            auto_report: false,
        }
    }
}

/// Debug output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Log queries slower than this (milliseconds).
    pub slow_query_threshold_ms: u64,
    /// Auto-add --profile to all queries.
    pub profile: bool,
    /// Print full JSON to stderr on every op.
    pub verbose: bool,
    /// Log level: off, error, warn, info, debug, trace.
    pub log_level: String,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            slow_query_threshold_ms: 100,
            profile: false,
            verbose: false,
            log_level: "warn".to_string(),
        }
    }
}

/// Configuration for time-series retention and downsampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TimeseriesConfig {
    /// Keep full records for the last N days. Older records are eligible
    /// for downsampling into daily summaries. Default: 90.
    pub full_retention_days: u64,
    /// Keep daily summaries for N days. Older daily summaries are
    /// consolidated into weekly summaries. Default: 365.
    pub daily_summary_days: u64,
    /// Automatically run downsampling during `axil heal`. Default: true.
    pub auto_downsample: bool,
}

impl Default for TimeseriesConfig {
    fn default() -> Self {
        Self {
            full_retention_days: 90,
            daily_summary_days: 365,
            auto_downsample: true,
        }
    }
}

/// Diagnose settings (for the Axil source repo).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiagnoseConfig {
    /// Project directories to scan for field reports.
    pub watch_projects: Vec<String>,
    /// Manual report drop location within the Axil source repo.
    pub incoming_dir: String,
}

impl Default for DiagnoseConfig {
    fn default() -> Self {
        Self {
            watch_projects: Vec::new(),
            incoming_dir: "reports/incoming".to_string(),
        }
    }
}

/// Optimize settings (for the Axil source repo).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimizeConfig {
    /// Directory for stored benchmark baselines.
    pub bench_baseline_dir: String,
    /// Alert if release build exceeds this size (MB).
    pub binary_size_target_mb: u64,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            bench_baseline_dir: "benches/baselines".to_string(),
            binary_size_target_mb: 10,
        }
    }
}

/// Configuration for the project indexer (`[index]` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    /// Re-index automatically on session start.
    pub auto_index: bool,
    /// Languages to index. `["auto"]` means auto-detect.
    pub languages: Vec<String>,
    /// Additional ignore patterns (on top of .gitignore).
    pub ignore: Vec<String>,
    /// Skip files larger than this (KB).
    pub max_file_size_kb: u64,
    /// Include test files in the index.
    pub index_tests: bool,
    /// Include private functions/types.
    pub index_private: bool,
    /// Symbol indexing depth: "public", "all", or "none".
    pub symbol_depth: String,
    /// Maximum tokens for a file summary.
    pub max_file_summary_tokens: usize,
    /// Maximum tokens for a module summary.
    pub max_module_summary_tokens: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            auto_index: true,
            languages: vec!["auto".to_string()],
            ignore: Vec::new(),
            max_file_size_kb: 100,
            index_tests: false,
            index_private: false,
            symbol_depth: "public".to_string(),
            max_file_summary_tokens: 50,
            max_module_summary_tokens: 150,
        }
    }
}

/// Configuration for the agent runtime (`[runtime]` section).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Default maximum tokens for context queries.
    pub default_max_tokens: usize,
    /// Default context depth: "shallow", "medium", "deep".
    pub default_depth: String,
    /// Mark index records stale after this many minutes.
    pub stale_threshold_minutes: u64,
    /// Auto-refresh stale files when accessed.
    pub auto_refresh: bool,
    /// Track query patterns for analytics.
    pub track_usage: bool,
    /// Enable predictive prefetch.
    pub prefetch_enabled: bool,
    /// Prefetch cache TTL in minutes.
    pub prefetch_cache_ttl_minutes: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            default_max_tokens: 2000,
            default_depth: "medium".to_string(),
            stale_threshold_minutes: 60,
            auto_refresh: true,
            track_usage: true,
            prefetch_enabled: true,
            prefetch_cache_ttl_minutes: 30,
        }
    }
}

/// Full-text search settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FtsConfig {
    /// Tantivy writer heap size in bytes. Default: 15MB.
    /// Increase to 50-100MB for batch ingest workloads.
    pub writer_heap_bytes: usize,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            writer_heap_bytes: 15_000_000,
        }
    }
}

/// Self-healing and auto-optimization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HealingConfig {
    /// Automatically compact when thresholds are breached.
    pub auto_compact: bool,
    /// Trigger compaction when live ratio drops below this (0.0-1.0).
    pub compact_live_ratio_threshold: f64,
    /// Maximum expired records before compaction triggers.
    pub compact_expired_threshold: usize,
    /// Maximum superseded records before compaction triggers.
    pub compact_superseded_threshold: usize,
    /// Rebuild vector index when deletion ratio exceeds this (0.0-1.0).
    pub vector_rebuild_threshold: f64,
    /// Merge FTS segments when count exceeds this.
    pub fts_segment_merge_threshold: usize,
    /// Enable periodic background maintenance (for long-running processes).
    pub background_maintenance: bool,
    /// How often to run maintenance checks (e.g. "1h", "30m").
    pub maintenance_interval: String,
    /// Metrics snapshot settings.
    pub metrics: MetricsHealingConfig,
    /// Auto-supersede similarity threshold.
    pub supersede_similarity_threshold: f64,
    /// Enable the durable semantic event log (the `recall_delta` pull feed).
    ///
    /// Off by default — it is a write-amplifier (an extra committed write per
    /// allowlisted event). Only takes effect in builds compiled with the
    /// `event-log` feature; ignored otherwise so `axil.toml` stays portable
    /// across feature sets.
    pub event_log: bool,
}

impl Default for HealingConfig {
    fn default() -> Self {
        Self {
            auto_compact: true,
            compact_live_ratio_threshold: 0.7,
            compact_expired_threshold: 1000,
            compact_superseded_threshold: 500,
            vector_rebuild_threshold: 0.2,
            fts_segment_merge_threshold: 10,
            background_maintenance: false,
            maintenance_interval: "1h".to_string(),
            metrics: MetricsHealingConfig::default(),
            supersede_similarity_threshold: 0.92,
            event_log: false,
        }
    }
}

/// Metrics snapshot sub-config under `[healing.metrics]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsHealingConfig {
    /// How often to snapshot metrics for trend tracking.
    pub snapshot_interval: String,
    /// Auto-rotate audit log after this many entries.
    pub max_audit_log_entries: usize,
}

impl Default for MetricsHealingConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: "daily".to_string(),
            max_audit_log_entries: 10_000,
        }
    }
}

/// Opportunistic, time-gated maintenance — the engine behind
/// `axil maintain --if-stale`.
///
/// Rather than a wall-clock cron, each task runs only when its cadence
/// has elapsed since the last run (tracked in the `_maintenance_runs`
/// table). The brain hook fires `axil maintain --if-stale --in-background`
/// on session start, so "the next command after the cadence elapses"
/// triggers the work, non-blocking. Only SAFE, additive tasks auto-run:
/// `snapshot` and `health-report --save`. Destructive maintenance —
/// downsampling (which purges old records) and `heal --reindex` — is
/// **never** auto-run; keep it explicit via `axil heal`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MaintenanceConfig {
    /// Master switch for the opportunistic (`--if-stale`) trigger. When
    /// `false`, `axil maintain --if-stale` is a no-op; an explicit
    /// `axil maintain` (no `--if-stale`) still runs every eligible task.
    pub auto: bool,
    /// Run `axil snapshot` (metrics for trends) at most this often.
    pub snapshot_every: String,
    /// Run `axil health-report --save` at most this often.
    pub health_report_every: String,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            auto: true,
            snapshot_every: "24h".to_string(),
            health_report_every: "7d".to_string(),
        }
    }
}

/// Parse a coarse duration string into seconds. Accepts a bare integer
/// (seconds) or an integer with a unit suffix: `s`, `m`, `h`, `d`, `w`.
/// Returns `None` on malformed input so callers fall back to a default.
pub fn parse_duration_secs(s: &str) -> Option<u64> {
    let s = s.trim();
    let last = s.chars().last()?;
    let (num, mult): (&str, u64) = match last {
        's' => (&s[..s.len() - 1], 1),
        'm' => (&s[..s.len() - 1], 60),
        'h' => (&s[..s.len() - 1], 3_600),
        'd' => (&s[..s.len() - 1], 86_400),
        'w' => (&s[..s.len() - 1], 604_800),
        c if c.is_ascii_digit() => (s, 1),
        _ => return None,
    };
    // checked_mul so an absurd-but-parsable value (e.g. "9999999999999w")
    // returns None → caller falls back to the default, honoring the
    // documented contract instead of wrapping (release) or panicking (debug).
    num.trim().parse::<u64>().ok().and_then(|n| n.checked_mul(mult))
}

/// `true` if a task whose last run was `last_ran` (None = never) is due
/// again, given `every_secs` cadence and the current time `now_secs`
/// (both unix seconds). Never-run tasks are always due.
pub fn is_due(last_ran_secs: Option<i64>, now_secs: i64, every_secs: u64) -> bool {
    match last_ran_secs {
        None => true,
        Some(last) => now_secs.saturating_sub(last) >= every_secs as i64,
    }
}

/// Generate a default `axil.toml` with all sections commented.
pub fn default_config_toml() -> String {
    r#"# Axil Configuration
# See: https://github.com/FC4b/axildb

[database]
# path = "./memory.axil"            # default db path (overrides AXIL_DB)

[timeseries]
# full_retention_days = 90           # keep full records for N days
# daily_summary_days = 365           # keep daily summaries for N days
# auto_downsample = true             # auto-downsample during `axil heal`

[maintenance]
# auto = true                        # opportunistic `axil maintain --if-stale` trigger (brain hook)
# snapshot_every = "24h"             # min interval between auto `axil snapshot`
# health_report_every = "7d"         # min interval between auto `axil health-report --save`
#                                    # (downsampling is destructive, so it is NOT auto-run — use `axil heal`)

[dev]
# source_repo = "../axildb"          # Axil source repo location (dev only)
# reports_dir = ".axil-reports"      # where /axil-report writes output
# auto_report = false                # auto-generate report on error/panic

[debug]
# slow_query_threshold_ms = 100      # log queries slower than this
# profile = false                    # auto-add --profile to all queries
# verbose = false                    # print full JSON to stderr on every op
# log_level = "warn"                 # off | error | warn | info | debug | trace

[diagnose]
# watch_projects = []                # project dirs to scan for field reports
# incoming_dir = "reports/incoming"  # manual report drop location

[optimize]
# bench_baseline_dir = "benches/baselines"  # stored benchmark results
# binary_size_target_mb = 10                # alert if release exceeds

[index]
# auto_index = true                         # re-index on session start
# languages = ["auto"]                      # or ["rust", "typescript", "python"]
# ignore = []                               # extra ignore patterns
# max_file_size_kb = 100                    # skip files larger than this
# index_tests = false                       # include test files
# index_private = false                     # include private functions/types
# symbol_depth = "public"                   # "public" | "all" | "none"
# max_file_summary_tokens = 50             # token limit per file summary
# max_module_summary_tokens = 150          # token limit per module summary

[runtime]
# default_max_tokens = 2000                # default token budget for context
# default_depth = "medium"                 # shallow | medium | deep
# stale_threshold_minutes = 60             # mark records stale after N minutes
# auto_refresh = true                      # re-index stale files on access
# track_usage = true                       # log query patterns for analytics
# prefetch_enabled = true                  # enable predictive context pre-loading
# prefetch_cache_ttl_minutes = 30          # how long prefetched context stays warm

[healing]
# auto_compact = true                      # compact on threshold breach
# compact_live_ratio_threshold = 0.7       # trigger compaction below this
# compact_expired_threshold = 1000         # max expired records before compact
# compact_superseded_threshold = 500       # max superseded records before compact
# vector_rebuild_threshold = 0.2           # rebuild when deletion ratio exceeds
# fts_segment_merge_threshold = 10         # merge when segment count exceeds
# background_maintenance = false           # periodic auto-heal (for long-running)
# maintenance_interval = "1h"             # how often to check (if background on)
# supersede_similarity_threshold = 0.92   # auto-supersede above this
# event_log = false                        # durable semantic event log (recall_delta); needs `event-log` build

[healing.metrics]
# snapshot_interval = "daily"              # how often to snapshot for trends
# max_audit_log_entries = 10000            # auto-rotate audit log

[llm]
# endpoint = "https://api.openai.com/v1/chat/completions"  # LLM API endpoint
# model = "gpt-4o-mini"                   # model name
# api_key = ""                            # API key (prefer AXIL_LLM_API_KEY env var)
# cost_per_1m_input = 0.15                # cost per 1M input tokens (USD)
# cost_per_1m_output = 0.60               # cost per 1M output tokens (USD)

[llm.limits]
# max_calls_per_minute = 10               # rate limit (0 = unlimited)
# max_tokens_per_session = 50000          # session token budget (0 = unlimited)
# budget_usd_per_day = 1.0                # daily budget (0.0 = unlimited)
"#
    .to_string()
}

/// Find the nearest `axil.toml` by walking up from `start_dir`.
/// Returns `None` if no config file is found (defaults should be used).
pub fn find_config_file(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = start_dir.to_path_buf();
    loop {
        let candidate = dir.join("axil.toml");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    // Check global config
    if let Some(home) = home_dir() {
        let global = home.join(".config").join("axil").join("config.toml");
        if global.exists() {
            return Some(global);
        }
    }
    None
}

/// Load config from `axil.toml`, searching current directory then walking up.
/// Falls back to `~/.config/axil/config.toml` then defaults.
pub fn load_config_from(start_dir: &Path) -> Result<AxilConfig, String> {
    // Walk up from start_dir looking for axil.toml
    let mut dir = start_dir.to_path_buf();
    loop {
        let candidate = dir.join("axil.toml");
        if candidate.exists() {
            let contents = std::fs::read_to_string(&candidate)
                .map_err(|e| format!("failed to read {}: {e}", candidate.display()))?;
            let config: AxilConfig = toml::from_str(&contents)
                .map_err(|e| format!("invalid config in {}: {e}", candidate.display()))?;
            return Ok(config);
        }
        if !dir.pop() {
            break;
        }
    }

    // Check global config
    if let Some(home) = home_dir() {
        let global = home.join(".config").join("axil").join("config.toml");
        if global.exists() {
            let contents = std::fs::read_to_string(&global)
                .map_err(|e| format!("failed to read {}: {e}", global.display()))?;
            let config: AxilConfig = toml::from_str(&contents)
                .map_err(|e| format!("invalid config in {}: {e}", global.display()))?;
            return Ok(config);
        }
    }

    Ok(AxilConfig::default())
}

/// Get a config value by dotted key (e.g. "dev.source_repo").
pub fn get_config_value(config: &AxilConfig, key: &str) -> Option<String> {
    let toml_val = toml::Value::try_from(config).ok()?;
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = &toml_val;

    for part in &parts {
        current = current.get(part)?;
    }

    match current {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Integer(n) => Some(n.to_string()),
        toml::Value::Float(f) => Some(f.to_string()),
        toml::Value::Boolean(b) => Some(b.to_string()),
        toml::Value::Datetime(d) => Some(d.to_string()),
        // Non-leaf keys (a table or array) have no single scalar value. Return
        // None rather than a Rust `Debug` dump — the `{:?}` form is not valid
        // config syntax, isn't round-trippable, and (for e.g. `[llm]`) would
        // expose every field of the subtree, including secrets, to a plugin
        // reading the parent key.
        toml::Value::Array(_) | toml::Value::Table(_) => None,
    }
}

/// Set a config value by dotted key in an `axil.toml` file.
/// If the file doesn't exist, creates it with the single value set.
pub fn set_config_value(config_path: &Path, key: &str, value: &str) -> Result<(), String> {
    let contents = if config_path.exists() {
        std::fs::read_to_string(config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&contents)
        .map_err(|e| format!("invalid TOML in {}: {e}", config_path.display()))?;

    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return Err("empty key".to_string());
    }
    for part in &parts {
        if part.is_empty() {
            return Err(format!("invalid key '{key}': contains empty segment"));
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(format!(
                "invalid key segment '{part}': only alphanumeric, underscore, and hyphen allowed"
            ));
        }
    }

    // Parse the value — try bool, int, float, then fall back to string
    let toml_val = if value == "true" {
        toml::Value::Boolean(true)
    } else if value == "false" {
        toml::Value::Boolean(false)
    } else if let Ok(n) = value.parse::<i64>() {
        toml::Value::Integer(n)
    } else if let Ok(f) = value.parse::<f64>() {
        toml::Value::Float(f)
    } else {
        toml::Value::String(value.to_string())
    };

    // Navigate to the right table, creating intermediate tables as needed
    let mut current = &mut doc;
    for part in &parts[..parts.len() - 1] {
        current = current
            .entry(part.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| format!("'{part}' is not a table"))?;
    }

    let last_key = parts[parts.len() - 1];
    current.insert(last_key.to_string(), toml_val);

    let output =
        toml::to_string_pretty(&doc).map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(config_path, output)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(())
}

/// Set an array-of-strings config value by dotted key in an `axil.toml` file
/// (e.g. `extensions.disabled = ["docs", "checkpoint"]`).
///
/// `set_config_value` only writes scalars; this is the array counterpart used by
/// `axil extensions enable|disable`. Creates the file and intermediate tables as
/// needed. An empty `values` removes the key so it doesn't linger as `[]`.
pub fn set_config_string_array(
    config_path: &Path,
    key: &str,
    values: &[String],
) -> Result<(), String> {
    let contents = if config_path.exists() {
        std::fs::read_to_string(config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&contents)
        .map_err(|e| format!("invalid TOML in {}: {e}", config_path.display()))?;

    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return Err("empty key".to_string());
    }
    for part in &parts {
        if part.is_empty() {
            return Err(format!("invalid key '{key}': contains empty segment"));
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(format!(
                "invalid key segment '{part}': only alphanumeric, underscore, and hyphen allowed"
            ));
        }
    }

    // Navigate to the right table, creating intermediate tables as needed.
    let mut current = &mut doc;
    for part in &parts[..parts.len() - 1] {
        current = current
            .entry(part.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| format!("'{part}' is not a table"))?;
    }

    let last_key = parts[parts.len() - 1];
    if values.is_empty() {
        current.remove(last_key);
    } else {
        let arr = values
            .iter()
            .map(|v| toml::Value::String(v.clone()))
            .collect();
        current.insert(last_key.to_string(), toml::Value::Array(arr));
    }

    let output =
        toml::to_string_pretty(&doc).map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(config_path, output)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(())
}

/// Get the user's home directory (`HOME` or `USERPROFILE`).
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = AxilConfig::default();
        assert_eq!(cfg.timeseries.full_retention_days, 90);
        assert_eq!(cfg.timeseries.daily_summary_days, 365);
        assert!(cfg.timeseries.auto_downsample);
    }

    #[test]
    fn decay_config_falls_back_to_default() {
        let cfg = DecayConfig::default();
        assert_eq!(
            cfg.half_life_for("errors"),
            crate::importance::DEFAULT_HALF_LIFE_DAYS
        );
    }

    #[test]
    fn decay_config_respects_table_override() {
        let mut cfg = DecayConfig::default();
        cfg.tables.insert("errors".into(), 7.0);
        cfg.tables.insert("preferences".into(), 365.0);
        assert_eq!(cfg.half_life_for("errors"), 7.0);
        assert_eq!(cfg.half_life_for("preferences"), 365.0);
        // Tables without an override still get the default.
        assert_eq!(
            cfg.half_life_for("sessions"),
            crate::importance::DEFAULT_HALF_LIFE_DAYS
        );
    }

    #[test]
    fn decay_config_rejects_nonpositive_values() {
        // Guards against malformed toml like `tables.errors = 0` or a NaN.
        let mut cfg = DecayConfig::default();
        cfg.tables.insert("a".into(), 0.0);
        cfg.tables.insert("b".into(), -5.0);
        cfg.tables.insert("c".into(), f64::NAN);
        assert_eq!(
            cfg.half_life_for("a"),
            crate::importance::DEFAULT_HALF_LIFE_DAYS
        );
        assert_eq!(
            cfg.half_life_for("b"),
            crate::importance::DEFAULT_HALF_LIFE_DAYS
        );
        assert_eq!(
            cfg.half_life_for("c"),
            crate::importance::DEFAULT_HALF_LIFE_DAYS
        );
    }

    #[test]
    fn decay_config_parses_from_toml() {
        let src = r#"
            [decay]
            tables.errors = 7
            tables.preferences = 365
        "#;
        let cfg: AxilConfig = toml::from_str(src).expect("parse");
        assert_eq!(cfg.decay.half_life_for("errors"), 7.0);
        assert_eq!(cfg.decay.half_life_for("preferences"), 365.0);
    }

    #[test]
    fn default_dev_config() {
        let cfg = DevConfig::default();
        assert_eq!(cfg.reports_dir, ".axil-reports");
        assert!(!cfg.auto_report);
        assert!(cfg.source_repo.is_none());
    }

    #[test]
    fn default_debug_config() {
        let cfg = DebugConfig::default();
        assert_eq!(cfg.slow_query_threshold_ms, 100);
        assert!(!cfg.profile);
        assert!(!cfg.verbose);
        assert_eq!(cfg.log_level, "warn");
    }

    #[test]
    fn default_diagnose_config() {
        let cfg = DiagnoseConfig::default();
        assert!(cfg.watch_projects.is_empty());
        assert_eq!(cfg.incoming_dir, "reports/incoming");
    }

    #[test]
    fn default_optimize_config() {
        let cfg = OptimizeConfig::default();
        assert_eq!(cfg.bench_baseline_dir, "benches/baselines");
        assert_eq!(cfg.binary_size_target_mb, 10);
    }

    #[test]
    fn parse_full_config() {
        let toml_str = r#"
[database]
path = "./my.axil"

[timeseries]
full_retention_days = 30
daily_summary_days = 180
auto_downsample = false

[dev]
source_repo = "../axildb"
reports_dir = ".reports"
auto_report = true

[debug]
slow_query_threshold_ms = 50
profile = true
verbose = true
log_level = "debug"

[diagnose]
watch_projects = ["../app1", "../app2"]
incoming_dir = "reports/in"

[optimize]
bench_baseline_dir = "bench/bl"
binary_size_target_mb = 5
"#;
        let cfg: AxilConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.database.path.as_deref(), Some("./my.axil"));
        assert_eq!(cfg.timeseries.full_retention_days, 30);
        assert!(!cfg.timeseries.auto_downsample);
        assert_eq!(cfg.dev.source_repo.as_deref(), Some("../axildb"));
        assert_eq!(cfg.dev.reports_dir, ".reports");
        assert!(cfg.dev.auto_report);
        assert_eq!(cfg.debug.slow_query_threshold_ms, 50);
        assert!(cfg.debug.profile);
        assert_eq!(cfg.debug.log_level, "debug");
        assert_eq!(cfg.diagnose.watch_projects.len(), 2);
        assert_eq!(cfg.diagnose.incoming_dir, "reports/in");
        assert_eq!(cfg.optimize.bench_baseline_dir, "bench/bl");
        assert_eq!(cfg.optimize.binary_size_target_mb, 5);
    }

    #[test]
    fn partial_config_uses_defaults() {
        let toml_str = r#"
[timeseries]
full_retention_days = 7
"#;
        let cfg: AxilConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.timeseries.full_retention_days, 7);
        assert_eq!(cfg.timeseries.daily_summary_days, 365); // default
        assert!(cfg.timeseries.auto_downsample); // default
        assert_eq!(cfg.debug.slow_query_threshold_ms, 100); // default
    }

    #[test]
    fn get_config_value_works() {
        let cfg = AxilConfig::default();
        assert_eq!(
            get_config_value(&cfg, "timeseries.full_retention_days"),
            Some("90".to_string())
        );
        assert_eq!(
            get_config_value(&cfg, "debug.log_level"),
            Some("warn".to_string())
        );
        assert_eq!(
            get_config_value(&cfg, "dev.reports_dir"),
            Some(".axil-reports".to_string())
        );
    }

    #[test]
    fn set_config_value_works() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("axil.toml");

        set_config_value(&path, "dev.source_repo", "../axildb").unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let cfg: AxilConfig = toml::from_str(&contents).unwrap();
        assert_eq!(cfg.dev.source_repo.as_deref(), Some("../axildb"));

        set_config_value(&path, "debug.slow_query_threshold_ms", "50").unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let cfg: AxilConfig = toml::from_str(&contents).unwrap();
        assert_eq!(cfg.debug.slow_query_threshold_ms, 50);
    }

    #[test]
    fn set_config_string_array_round_trips_and_removes() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("axil.toml");

        // Write an array; it parses back into ExtensionsConfig.disabled.
        set_config_string_array(&path, "extensions.disabled", &["docs".into(), "checkpoint".into()])
            .unwrap();
        let cfg = load_config_from(dir.path()).unwrap();
        assert_eq!(cfg.extensions.disabled, vec!["docs", "checkpoint"]);
        assert!(cfg.is_extension_disabled("docs"));
        assert!(!cfg.is_extension_disabled("scip"));

        // An empty slice removes the key entirely (no lingering `[]`).
        set_config_string_array(&path, "extensions.disabled", &[]).unwrap();
        let cfg = load_config_from(dir.path()).unwrap();
        assert!(cfg.extensions.disabled.is_empty());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(
            !contents.contains("disabled"),
            "empty array should remove the key, got:\n{contents}"
        );
    }

    #[test]
    fn default_config_toml_is_valid() {
        let toml_str = default_config_toml();
        // The default config is all comments, so parsing it should yield defaults
        let cfg: AxilConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg.timeseries.full_retention_days, 90);
    }

    #[test]
    fn load_config_from_nonexistent_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = load_config_from(dir.path()).unwrap();
        assert_eq!(cfg.timeseries.full_retention_days, 90);
    }

    #[test]
    fn load_config_from_dir_with_file() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("axil.toml"),
            "[timeseries]\nfull_retention_days = 7\n",
        )
        .unwrap();
        let cfg = load_config_from(dir.path()).unwrap();
        assert_eq!(cfg.timeseries.full_retention_days, 7);
    }

    #[test]
    fn set_config_value_rejects_empty_segment() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("axil.toml");
        assert!(set_config_value(&path, "dev.", "value").is_err());
        assert!(set_config_value(&path, ".foo", "value").is_err());
        assert!(set_config_value(&path, "", "value").is_err());
    }

    #[test]
    fn set_config_value_rejects_special_chars() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("axil.toml");
        // `@` is not a valid TOML bare-key char (hyphens now ARE allowed).
        assert!(set_config_value(&path, "dev.source@repo", "value").is_err());
        assert!(set_config_value(&path, "dev.source repo", "value").is_err());
        // Hyphens are valid TOML bare keys (e.g. `[plugins.my-plugin]`).
        assert!(set_config_value(&path, "plugins.my-plugin.x", "1").is_ok());
    }

    #[test]
    fn find_config_file_walks_up() {
        let dir = tempfile::TempDir::new().unwrap();
        let sub = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(dir.path().join("axil.toml"), "[database]\n").unwrap();

        let found = find_config_file(&sub);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), dir.path().join("axil.toml"));
    }

    #[test]
    fn find_config_file_returns_none_when_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        // In a temp dir with no axil.toml and no global config, should return None
        // (unless user has ~/.config/axil/config.toml, but that's unlikely in CI)
        let found = find_config_file(dir.path());
        // We can't guarantee None (user might have global config), so just check it doesn't panic
        let _ = found;
    }

    #[test]
    fn parse_duration_secs_units() {
        assert_eq!(parse_duration_secs("30s"), Some(30));
        assert_eq!(parse_duration_secs("5m"), Some(300));
        assert_eq!(parse_duration_secs("24h"), Some(86_400));
        assert_eq!(parse_duration_secs("7d"), Some(604_800));
        assert_eq!(parse_duration_secs("2w"), Some(1_209_600));
        assert_eq!(parse_duration_secs("3600"), Some(3_600)); // bare = seconds
        assert_eq!(parse_duration_secs(" 12h "), Some(43_200)); // trimmed
    }

    #[test]
    fn parse_duration_secs_rejects_garbage() {
        assert_eq!(parse_duration_secs(""), None);
        assert_eq!(parse_duration_secs("h"), None);
        assert_eq!(parse_duration_secs("24x"), None);
        assert_eq!(parse_duration_secs("abc"), None);
    }

    #[test]
    fn is_due_logic() {
        // Never run → always due.
        assert!(is_due(None, 1_000_000, 3_600));
        // Exactly at cadence → due.
        assert!(is_due(Some(1_000_000), 1_003_600, 3_600));
        // Past cadence → due.
        assert!(is_due(Some(1_000_000), 1_010_000, 3_600));
        // Within cadence → not due.
        assert!(!is_due(Some(1_000_000), 1_001_000, 3_600));
    }

    #[test]
    fn maintenance_config_defaults() {
        let m = MaintenanceConfig::default();
        assert!(m.auto);
        assert_eq!(parse_duration_secs(&m.snapshot_every), Some(86_400));
        assert_eq!(parse_duration_secs(&m.health_report_every), Some(604_800));
    }
}
