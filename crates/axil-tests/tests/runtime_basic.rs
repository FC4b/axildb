//! Integration tests for Phase 4e: Agent Runtime.

use serde_json::json;
use std::fs;

use axil_core::Axil;
use axil_indexer::{IndexConfig, ProjectIndexer};

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

fn create_and_index_rust_project(db: &Axil) -> tempfile::TempDir {
    let project_dir = tempfile::tempdir().unwrap();
    let src = project_dir.path().join("src");
    fs::create_dir_all(&src).unwrap();

    fs::write(
        project_dir.path().join("Cargo.toml"),
        r#"
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"
[dependencies]
serde = "1"
tokio = "1"
"#,
    )
    .unwrap();

    fs::write(
        src.join("main.rs"),
        r#"//! Main entry point.
fn main() { println!("Hello"); }
"#,
    )
    .unwrap();

    fs::write(
        src.join("auth.rs"),
        r#"//! Authentication module — JWT tokens and middleware.
/// Validates a JWT token.
pub fn validate_token(token: &str) -> bool { true }
/// Authentication error types.
pub enum AuthError { Expired, Invalid }
pub struct Claims { pub user_id: String }
"#,
    )
    .unwrap();

    fs::write(
        src.join("db.rs"),
        r#"//! Database connection pool and queries.
use crate::auth::Claims;
/// Get a database connection from the pool.
pub fn get_connection() -> Result<(), String> { Ok(()) }
pub struct Pool { pub size: usize }
"#,
    )
    .unwrap();

    // Add a CLAUDE.md for rule extraction
    fs::write(
        project_dir.path().join("CLAUDE.md"),
        r#"
# Coding Conventions
- Use `thiserror` for error types in library crates
- Prefer `&str` over `String` in function params
- Never commit secrets to the repository
- Always run tests before committing
"#,
    )
    .unwrap();

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(db, config);
    indexer.index_full(project_dir.path()).unwrap();
    project_dir
}

// ── 4e.1: Query intent detection ──────────────────────────────────

#[test]
fn ask_detects_vector_intent() {
    let intent = axil_indexer::detect_intent("find something similar to auth bugs");
    assert_eq!(intent, axil_indexer::QueryIntent::VectorSearch);
}

#[test]
fn ask_detects_temporal_intent() {
    let intent = axil_indexer::detect_intent("what changed in the last 3 days");
    assert_eq!(intent, axil_indexer::QueryIntent::Temporal);
}

#[test]
fn ask_detects_rule_intent() {
    let intent = axil_indexer::detect_intent("what should I always do");
    assert_eq!(intent, axil_indexer::QueryIntent::RuleLookup);
}

#[test]
fn ask_detects_text_search_intent() {
    let intent = axil_indexer::detect_intent("find the exact error message timeout");
    assert_eq!(intent, axil_indexer::QueryIntent::TextSearch);
}

#[test]
fn ask_detects_graph_intent() {
    let intent = axil_indexer::detect_intent("what depends on the auth module");
    assert_eq!(intent, axil_indexer::QueryIntent::GraphTraversal);
}

#[test]
fn ask_returns_results() {
    let (db, _db_dir) = temp_db();
    let _project = create_and_index_rust_project(&db);

    let result = axil_indexer::ask::ask(&db, "authentication", 5).unwrap();
    assert!(!result.results.is_empty(), "ask should return results");
    assert!(!result.strategies_used.is_empty());
    assert!(result.tokens > 0);
}

#[test]
fn parse_duration_from_query() {
    assert_eq!(
        axil_indexer::parse_duration_from_query("last 3 days"),
        Some(3 * 86400)
    );
    assert_eq!(
        axil_indexer::parse_duration_from_query("since yesterday"),
        Some(86400)
    );
    assert_eq!(
        axil_indexer::parse_duration_from_query("last hour"),
        Some(3600)
    );
    // No duration pattern → returns None
    assert!(axil_indexer::parse_duration_from_query("no duration here").is_none());
}

// ── 4e.3: Rules store ─────────────────────────────────────────────

#[test]
fn rule_crud() {
    let (db, _dir) = temp_db();

    // Set
    axil_indexer::set_rule(&db, "error_handling", "Use thiserror in libs", "user").unwrap();

    // Get
    let rule = axil_indexer::get_rule(&db, "error_handling").unwrap();
    assert!(rule.is_some());
    let rule = rule.unwrap();
    assert_eq!(rule.key, "error_handling");
    assert_eq!(rule.rule, "Use thiserror in libs");
    assert_eq!(rule.source, "user");

    // List
    let rules = axil_indexer::list_rules(&db).unwrap();
    assert_eq!(rules.len(), 1);

    // Delete
    let deleted = axil_indexer::delete_rule(&db, "error_handling").unwrap();
    assert!(deleted);

    let rules = axil_indexer::list_rules(&db).unwrap();
    assert!(rules.is_empty());
}

#[test]
fn rule_get_missing_returns_none() {
    let (db, _dir) = temp_db();
    let rule = axil_indexer::get_rule(&db, "nonexistent").unwrap();
    assert!(rule.is_none());
}

#[test]
fn auto_extract_rules_from_claude_md() {
    let (db, _dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();

    fs::write(
        project_dir.path().join("CLAUDE.md"),
        r#"
# Conventions
- Use `thiserror` for error types
- Never store secrets in code
- Always write tests for new features
"#,
    )
    .unwrap();

    let extracted = axil_indexer::auto_extract_rules(&db, project_dir.path()).unwrap();
    assert!(
        !extracted.is_empty(),
        "should extract at least one rule from CLAUDE.md"
    );

    let rules = axil_indexer::list_rules(&db).unwrap();
    assert!(rules.iter().any(|r| r.source == "detected"));
}

// ── 4e.5: Impact analysis ─────────────────────────────────────────

#[test]
fn impact_without_graph_returns_graceful_message() {
    let (db, _dir) = temp_db();
    let _project = create_and_index_rust_project(&db);

    // No graph plugin loaded, so impact should return gracefully
    let report = axil_indexer::impact(&db, "src/auth.rs").unwrap();
    assert!(report.suggestion.contains("graph") || report.direct_dependents.is_empty());
}

#[test]
fn why_connected_without_graph() {
    let (db, _dir) = temp_db();
    let _project = create_and_index_rust_project(&db);

    let chain = axil_indexer::why_connected(&db, "src/auth.rs", "src/db.rs").unwrap();
    // Without graph plugin, should return empty (no connection found)
    assert!(chain.is_empty());
}

// ── 4e.8: Analytics ───────────────────────────────────────────────

#[test]
fn analytics_logs_and_retrieves() {
    let (db, _dir) = temp_db();

    // Log some queries
    axil_indexer::log_query(&db, "vector", "auth bugs", 3, 150).unwrap();
    axil_indexer::log_query(&db, "fts", "error message", 1, 50).unwrap();
    axil_indexer::log_query(&db, "vector", "db queries", 2, 100).unwrap();

    let analytics = axil_indexer::get_analytics(&db, 7).unwrap();
    let queries = analytics
        .get("total_queries")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(queries, 3);

    let tokens = analytics
        .get("tokens_served")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(tokens, 300);
}

// ── 4e.9: Prefetch ────────────────────────────────────────────────

#[test]
fn prefetch_returns_sections() {
    let (db, _dir) = temp_db();
    let _project = create_and_index_rust_project(&db);

    let result = axil_indexer::prefetch(&db, "fix auth timeout", 2000).unwrap();
    assert!(result.ready);
    assert!(result.total_tokens > 0);
    assert!(
        !result.sections.is_empty(),
        "prefetch should return at least one section"
    );
}

// ── Combined: full agent session ──────────────────────────────────

#[test]
fn full_agent_session_workflow() {
    let (db, _db_dir) = temp_db();
    let _project = create_and_index_rust_project(&db);

    // 1. Load context
    let opts = axil_indexer::ContextOptions {
        max_tokens: 2000,
        ..Default::default()
    };
    let context = axil_indexer::recall::context(&db, &opts).unwrap();
    assert!(context.get("project").is_some());

    // 2. Ask a question
    let ask_result = axil_indexer::ask::ask(&db, "how does authentication work", 5).unwrap();
    assert!(!ask_result.results.is_empty());

    // 3. Set a rule
    axil_indexer::set_rule(&db, "test_convention", "Always test auth changes", "user").unwrap();

    // 4. Check rules
    let rules = axil_indexer::list_rules(&db).unwrap();
    assert!(!rules.is_empty());

    // 5. Check analytics
    let analytics = axil_indexer::get_analytics(&db, 7).unwrap();
    // Analytics may or may not have entries depending on whether ask logs queries
    assert!(analytics.get("period").is_some());

    // 6. Prefetch for a task
    let prefetch = axil_indexer::prefetch(&db, "fix auth bug", 1000).unwrap();
    assert!(prefetch.ready);
}
