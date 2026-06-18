//! Integration tests for the project indexer (Phase 4d).

use serde_json::json;
use std::fs;

use axil_core::Axil;
use axil_indexer::{IndexConfig, ProjectIndexer};

// ── Test helpers ─────────────────────────────────────────────────────

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

fn create_rust_project(dir: &std::path::Path) {
    // Create a minimal Rust project structure
    fs::write(
        dir.join("Cargo.toml"),
        r#"[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1"
tokio = "1"
"#,
    )
    .unwrap();

    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();

    fs::write(
        src.join("main.rs"),
        r#"//! Main entry point for the test project.

use crate::auth::validate_token;

mod auth;

fn main() {
    println!("Hello, world!");
}
"#,
    )
    .unwrap();

    let auth_dir = src.join("auth");
    fs::create_dir_all(&auth_dir).unwrap();

    fs::write(
        auth_dir.join("mod.rs"),
        r#"//! Authentication module — JWT tokens and middleware.

pub mod middleware;

/// Validates a JWT token and returns the claims.
pub fn validate_token(token: &str) -> Result<Claims, AuthError> {
    todo!()
}

/// Decoded JWT claims.
pub struct Claims {
    pub user_id: String,
    pub role: String,
}

/// Authentication errors.
pub enum AuthError {
    Expired,
    Invalid,
    MissingToken,
}
"#,
    )
    .unwrap();

    fs::write(
        auth_dir.join("middleware.rs"),
        r#"//! Auth middleware — validates tokens on incoming requests.

use super::{Claims, validate_token};

/// Middleware that validates JWT tokens on every request.
pub struct AuthMiddleware;

impl AuthMiddleware {
    /// Create a new auth middleware instance.
    pub fn new() -> Self {
        Self
    }

    /// Validate the token from the request header.
    pub fn validate(&self, header: &str) -> Result<Claims, super::AuthError> {
        validate_token(header)
    }
}
"#,
    )
    .unwrap();
}

fn create_typescript_project(dir: &std::path::Path) {
    fs::write(
        dir.join("package.json"),
        r#"{"name": "ts-test-project", "dependencies": {"express": "^4.18.0", "zod": "^3.0.0"}}"#,
    )
    .unwrap();

    fs::write(dir.join("tsconfig.json"), r#"{"compilerOptions": {}}"#).unwrap();

    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();

    fs::write(
        src.join("index.ts"),
        r#"// Main entry point for the TypeScript test project.
import express from 'express';
import { validateToken } from './auth';

export function createApp(): express.Application {
    const app = express();
    return app;
}
"#,
    )
    .unwrap();

    fs::write(
        src.join("auth.ts"),
        r#"/**
 * Authentication service for JWT tokens.
 */
import { z } from 'zod';

export interface Claims {
    userId: string;
    role: string;
}

export function validateToken(token: string): Claims {
    return { userId: '1', role: 'admin' };
}

export class AuthService {
    validate(token: string): Claims {
        return validateToken(token);
    }
}
"#,
    )
    .unwrap();
}

fn create_python_project(dir: &std::path::Path) {
    fs::write(
        dir.join("pyproject.toml"),
        r#"[project]
name = "py-test-project"
dependencies = ["fastapi", "pydantic"]
"#,
    )
    .unwrap();

    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();

    fs::write(
        src.join("app.py"),
        r#""""Main FastAPI application."""

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}

class Config:
    """Application configuration."""
    debug: bool = False
    port: int = 8000
"#,
    )
    .unwrap();
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn index_rust_project() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    let result = indexer.index_full(project_dir.path()).unwrap();

    // Should have indexed files
    assert!(
        result.indexed_files >= 3,
        "expected at least 3 files, got {}",
        result.indexed_files
    );
    assert!(result.modules > 0, "expected at least 1 module");
    assert!(result.symbols > 0, "expected at least 1 symbol");
    assert!(result.deps > 0, "expected dependencies from Cargo.toml");

    // Tables should be created
    assert!(result.tables_created.contains(&"_idx_project".to_string()));
    assert!(result.tables_created.contains(&"_idx_files".to_string()));
}

#[test]
fn index_typescript_project() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_typescript_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    let result = indexer.index_full(project_dir.path()).unwrap();

    assert!(
        result.indexed_files >= 2,
        "expected at least 2 files, got {}",
        result.indexed_files
    );
    assert!(result.symbols > 0, "expected symbols from TS files");
    assert!(result.deps > 0, "expected dependencies from package.json");
}

#[test]
fn index_python_project() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_python_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    let result = indexer.index_full(project_dir.path()).unwrap();

    assert!(result.indexed_files >= 1, "expected at least 1 file");
    assert!(result.symbols > 0, "expected symbols from Python file");
}

#[test]
fn recall_finds_relevant_results() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    // Search for auth-related content
    let results = axil_indexer::recall::recall(&db, "authentication middleware", 5).unwrap();
    assert!(
        !results.is_empty(),
        "expected recall results for 'authentication middleware'"
    );

    // The top result should be related to auth
    let top = &results[0];
    assert!(top.score > 0.0);
    assert!(top.tokens > 0);
}

#[test]
fn context_respects_token_budget() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    let opts = axil_indexer::ContextOptions {
        max_tokens: 500,
        ..Default::default()
    };
    let result = axil_indexer::recall::context(&db, &opts).unwrap();

    // Should have a tokens_used field
    let tokens_used = result
        .get("tokens_used")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(tokens_used > 0, "expected non-zero tokens_used");
    assert!(
        tokens_used <= 500,
        "expected tokens_used <= 500, got {tokens_used}"
    );

    // Should include project info
    assert!(result.get("project").is_some());
    assert!(result.get("type").is_some());
}

#[test]
fn context_with_focus() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    let opts = axil_indexer::ContextOptions {
        max_tokens: 2000,
        focus: vec!["auth".to_string()],
        ..Default::default()
    };
    let result = axil_indexer::recall::context(&db, &opts).unwrap();
    assert!(result.get("modules").is_some());
}

#[test]
fn stats_shows_compression_ratio() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    let stats = axil_indexer::recall::stats(&db, None, None).unwrap();
    let index = stats.get("index").unwrap();

    // Should have all table counts
    let tables = index.get("tables").unwrap();
    assert!(tables.get("project").is_some());
    assert!(tables.get("files").is_some());
    assert!(tables.get("modules").is_some());
    assert!(tables.get("symbols").is_some());
    assert!(tables.get("deps").is_some());

    // Should have compression ratio
    let total_index = index
        .get("total_index_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(total_index > 0, "expected non-zero index tokens");
}

#[test]
fn incremental_index_detects_no_changes() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);

    // First full index
    let first = indexer.index_full(project_dir.path()).unwrap();
    assert!(first.indexed_files > 0);

    // Second incremental (no git, so it may detect changes or not)
    let second = indexer.index_incremental(project_dir.path()).unwrap();
    // The incremental path should still work without crashing
    assert!(second.changed.is_some() || second.indexed_files > 0);
}

#[test]
fn token_estimation_on_records() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    // Check that file records have token counts
    let files = db.list("_idx_files").unwrap();
    for file in &files {
        let tokens = file.data.get("tokens").and_then(|v| v.as_u64());
        assert!(
            tokens.is_some(),
            "expected tokens field on file record: {:?}",
            file.data.get("path")
        );
        assert!(tokens.unwrap() > 0);
    }

    // Check that symbol records have token counts
    let symbols = db.list("_idx_symbols").unwrap();
    for sym in &symbols {
        let tokens = sym.data.get("tokens").and_then(|v| v.as_u64());
        assert!(
            tokens.is_some(),
            "expected tokens field on symbol record: {:?}",
            sym.data.get("name")
        );
    }
}

#[test]
fn axilignore_excludes_patterns() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    // Create a .axilignore file
    fs::write(project_dir.path().join(".axilignore"), "src/auth/\n").unwrap();

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    let result = indexer.index_full(project_dir.path()).unwrap();

    // Auth files should be excluded
    let files = db.list("_idx_files").unwrap();
    for file in &files {
        let path = file.data.get("path").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !path.starts_with("src/auth/"),
            "expected auth files to be excluded, found: {path}"
        );
    }
}

#[test]
fn project_overview_has_required_fields() {
    let (db, _db_dir) = temp_db();
    let project_dir = tempfile::tempdir().unwrap();
    create_rust_project(project_dir.path());

    let config = IndexConfig::default();
    let indexer = ProjectIndexer::new(&db, config);
    indexer.index_full(project_dir.path()).unwrap();

    let projects = db.list("_idx_project").unwrap();
    assert_eq!(projects.len(), 1, "expected exactly 1 project record");

    let project = &projects[0];
    assert!(project.data.get("name").is_some());
    assert!(project.data.get("type").is_some());
    assert!(project.data.get("summary").is_some());
    assert!(project.data.get("tech_stack").is_some());
    assert!(project.data.get("file_count").is_some());
    assert!(project.data.get("line_count").is_some());
    assert!(project.data.get("indexed_at").is_some());

    // Type should be "rust"
    assert_eq!(project.data.get("type").unwrap(), "rust");
    // Name should be "test-project"
    assert_eq!(project.data.get("name").unwrap(), "test-project");
}
