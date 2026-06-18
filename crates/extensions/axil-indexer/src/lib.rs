//! Project indexer for Axil — token-efficient codebase understanding for agents.
//!
//! Scans a project directory, parses source files using language-specific
//! heuristics, and stores compact summaries in an Axil database. Agents
//! query these summaries via `recall` and `context` instead of re-reading
//! raw source files every session.
//!
//! # Tables
//!
//! The indexer stores records in reserved tables:
//! - `_idx_project` — single project overview record
//! - `_idx_files` — one record per source file
//! - `_idx_modules` — one record per directory/module
//! - `_idx_symbols` — public functions, types, traits
//! - `_idx_deps` — external dependencies

pub mod analytics;
pub mod ask;
pub mod code_recall_eval;
pub mod config;
pub mod config_sections;
pub mod context_savings;
pub mod freshness;
pub mod distill;
pub mod impact;
pub mod indexer;
pub mod markdown;
pub mod parser;
pub mod prefetch;
pub mod progress;
pub mod proxy;
pub mod recall;
pub mod rerank;
pub mod rules;
pub mod scanner;
pub mod token;

// Re-export IndexConfig from core for convenience.
pub use analytics::{get_analytics, log_query, AnalyticsRecord, TABLE_ANALYTICS};
pub use ask::{
    ask, detect_intent, execute_plan, parse_duration_from_query, plan_query, AskResult,
    QueryIntent, QueryPlan, QueryStep,
};
pub use axil_core::IndexConfig;
pub use config_sections::{split_json_sections, split_toml_sections, split_yaml_sections};
pub use freshness::{check_freshness, FreshnessReport, FreshnessStatus};
pub use impact::{impact, reverse_impact, why_connected, ImpactReport};
pub use indexer::{IndexResult, ProjectIndexer};
pub use markdown::{section_canonical_id, split_sections, ParsedSection};
pub use prefetch::{
    load_cached, prefetch, prefetch_file, save_cache, PrefetchResult, PrefetchSection,
};
pub use progress::{IndexProgress, NoopProgress};
pub use proxy::{
    build_proxy, build_proxy_id, build_proxy_text, code_ref_anchor_keys, proxy_to_record,
    CodeProxy, ProxyInput, ProxyKind, DEFAULT_PROXY_TOKEN_BUDGET, TABLE_CODE_PROXIES,
    TABLE_CODE_REFS_INDEX,
};
pub use recall::{ContextDepth, ContextOptions, RecallResult};
pub use rules::{
    auto_extract_rules, delete_rule, get_rule, list_rules, set_rule, Rule, TABLE_RULES,
};
pub use scanner::{ProjectInfo, ProjectType};
pub use token::estimate_tokens;
