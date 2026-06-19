//! # Axil core — storage engine, record types, and the extensibility SPI.
//!
//! ## Stability surface (the 1.0 contract)
//!
//! Axil deliberately splits its API into a **stable outer SPI** that third
//! parties build against, and an **unstable inner Engine API** that core owns.
//! This asymmetry is what lets Axil add, drop, or swap storage Engines freely
//! while third-party Extensions/Adapters keep compiling.
//!
//! **Stable (semver-locked at 1.0) — extend Axil through these:**
//! - [`Extension`] + support types: [`CliSurface`], [`CliSubcommand`], [`CliArg`],
//!   [`CliInvocation`], [`CliOutput`], [`McpSurface`], [`McpTool`], [`McpCall`],
//!   [`Dispatch`], [`Hit`], [`RefreshOpts`], [`RefreshReport`] (see [`extension`]).
//! - [`Adapter`], [`Protocol`], [`compose_cli_surface`], [`compose_mcp_surface`],
//!   [`dispatch_cli`], [`dispatch_mcp`] (see [`adapter`]).
//! - The [`Axil`] builder + query/recall methods Extensions and Adapters call.
//!
//! The structs third parties construct or receive are `#[non_exhaustive]` with
//! constructors/builders, so fields can grow without a breaking change.
//!
//! **Unstable (no semver guarantee — upstream-or-fork):** everything in
//! [`plugin`] — [`Plugin`], [`VectorIndex`], [`GraphIndex`], [`SearchIndex`],
//! [`TimeSeriesIndex`], [`TextEmbedder`], [`Capability`]. These are the Engine
//! (Tier-1) substrate the master coordinator drives directly.
//!
//! See `docs/src/extending/overview.md` for the three-tier taxonomy.

pub mod ab_test;
pub mod activation;
pub mod adapter;
pub mod auto_capture;
pub mod beliefs;
pub mod bench_metrics;
pub mod boot;
pub mod brain;
pub mod branch;
pub mod code_refs;
pub mod config;
pub mod consolidation;
pub mod db;
pub mod detectors;
pub mod diagnostics;
pub mod entity;
pub mod error;
pub mod extension;
pub mod feedback;
pub mod importance;
pub mod inference;
pub mod llm;
#[cfg(feature = "llm-http")]
pub mod llm_http;
pub mod metrics;
pub mod otel;
pub mod plugin;
pub mod prefetch;
pub mod query;
pub mod record;
pub mod remember;
pub mod scoring;
pub mod simhash;
pub mod snapshot;
pub mod storage;
pub mod temporal;
pub mod tiering;
pub mod util;
pub mod worker;
pub mod write_buffer;

// Re-export primary types for convenience.
pub use boot::{
    collect_extension_blocks, BootContext, BootOptions, BootSection, BOOT_SCHEMA_VERSION,
    DEFAULT_TOKEN_BUDGET,
};
pub use config::{
    default_config_toml, find_config_file, get_config_value, home_dir, load_config_from,
    set_config_value, AxilConfig, DatabaseConfig, DebugConfig, DecayConfig, DevConfig,
    DiagnoseConfig, FtsConfig, HealingConfig, IndexConfig, MetricsHealingConfig, OptimizeConfig,
    RuntimeConfig, TimeseriesConfig,
};
pub use db::{
    companion_path, AutoLinkReport, Axil, AxilBuilder, CanonicalPublisher, DatabaseInfo,
    HealReport, WarmUpReport, SCIP_ALIAS_TABLE,
};
pub use diagnostics::{
    human_bytes, BenchReport, BenchResult, CheckResult, CompactReport, DataQualitySection,
    DatabaseMeta, DatabaseStats, DoctorReport, HealAction, HealthReport, HealthSections,
    IndexSection, IndexStats as DiagIndexStats, MetricTrend, MetricsHistoryEntry,
    PerformanceSection, ProblemDetection, Recommendation, RecordStats, SelfHealReport, Severity,
    StorageSection, SystemInfo, TrendReport, VectorRebuildReport,
};
pub use error::{AxilError, Result};
pub use metrics::{
    AuditEntry, LatencyPercentiles, Metrics, MetricsSnapshot, OpType, SlowQueryEntry,
};
pub use plugin::{
    parse_path, Capability, Direction, EdgeInfo, GraphIndex, Plugin, SearchIndex, TextEmbedder,
    TimeBucket, TimeSeriesIndex, TraversalStep, VectorIndex,
};

// Phase 17 — extensibility contracts (Tier 2 + Tier 3).
pub use adapter::{
    compose_cli_surface, compose_mcp_surface, dispatch_cli, dispatch_mcp, Adapter, Protocol,
};
pub use extension::{
    CliArg, CliInvocation, CliOutput, CliSubcommand, CliSurface, Dispatch, Extension, Hit, McpCall,
    McpSurface, McpTool, RefreshOpts, RefreshReport,
};
pub use query::{
    graph_boost, EstimatedCost, Op, PlanStep, ProfileStep, QueryBuilder, QueryPlan, QueryProfile,
    SortDirection,
};
pub use record::{Record, RecordId};
pub use remember::{DecisionInput, ErrorInput, RememberResult, WriteSource};
pub use scoring::{
    DedupConfig, QtcConfig, RecallConfig, RecallResult, ScoreExplanation, ScoreWeights,
    SignalValues,
};
pub use storage::Storage;

// Phase 5f re-exports
pub use llm::{
    LlmConfig, LlmLimits, LlmProvider, LlmRateLimiter, LlmResponse, LlmUsage, LlmUsageTracker,
    NoLlm,
};
#[cfg(feature = "llm-http")]
pub use llm_http::HttpLlm;

// Phase 5e re-exports
pub use consolidation::{
    check_conflict, compute_confidence, consolidate_facts, detect_conflict_confidence,
    ConfidenceScore, ConflictConfidence, ConflictResult, ConsolidatedFact,
};
pub use entity::{extract_entities, Entity, EntityType};
pub use feedback::{FeedbackEntry, FeedbackStore};
pub use inference::{InferenceEngine, InferenceRule, InferredFact};
pub use prefetch::{MaterializedRecall, PrefetchEngine, QueryLogEntry, QueryPattern};
pub use temporal::{parse_temporal, temporal_boost, TemporalTarget};

// Phase 6 re-exports — benchmark metrics and A/B testing
pub use ab_test::{compare_configs, AbTestConfig, AbTestResult};
pub use bench_metrics::{
    competitor_baselines, compute_mem_efficiency, format_memscore,
    BenchmarkResult as RecallBenchmarkResult, ComparisonReport, CompetitorResult,
};

// Phase 8b re-exports — activation-level scoring
pub use activation::{
    activation_boost, compute_bump, compute_stats as compute_activation_stats, decayed_activation,
    get_activation, get_last_accessed, ActivationConfig, ActivationDistribution, ActivationStats,
    DEFAULT_ACTIVATION, DEFAULT_ACTIVATION_WEIGHT, DEFAULT_BOOST, DEFAULT_HALF_LIFE_DAYS,
};

// Phase 8b re-exports — write buffer (deferred indexing)
pub use write_buffer::{WriteBuffer, WriteBufferConfig};

// Phase 8b re-exports — tiered memory
pub use tiering::{classify_tier, tier_distribution, MemoryTier, TierConfig, TierStats};

// Phase 8b re-exports — snapshots
pub use snapshot::{create_snapshot, list_snapshots, restore_snapshot, SnapshotMeta};

// Phase 11 re-exports — brain (full agent brain)
pub use brain::{
    belief_history, classify_trust, doubt_record, extract_provenance, migrate_provenance_all,
    migrate_provenance_record, project_model_generate, project_model_set, project_model_show,
    remember, revise_beliefs, run_brain_eval, run_needle_eval, self_note, self_profile,
    user_contract_list, user_contract_set, verify_record, why_recalled, why_remembered,
    why_revised, BeliefRevisionAction, BeliefRevisionResult, BrainEvalReport, MemoryScope,
    MemorySource, MemoryType, NeedleEvalReport, NeedleResult, Observation, PipelineAction,
    PipelineOutcome, Provenance, TrustTier, WhyRecalled,
    WhyRemembered, WhyRevised,
};

// Phase 5d re-exports — worker, branching
pub use branch::{
    branch_create, branch_delete, branch_diff, branch_list, branch_merge, branch_switch,
    BranchDiff, MergeReport, MergeStrategy,
};
pub use detectors::{run_all_detectors, DetectorResult};
pub use worker::{AxilWorker, MaintenanceThread, WorkerReport};
