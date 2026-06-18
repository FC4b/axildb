//! Agent memory example — session lifecycle, store/recall, memory types.
//!
//! Run: `cargo run --example agent_memory --features memory`
//!
//! Demonstrates how an AI agent uses Axil as cognitive memory:
//! store decisions, learn facts, detect patterns, and recall context.

use anyhow::Result;
use axil_core::Axil;
use serde_json::json;

fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("agent.axil");

    let db = Axil::open(&db_path).build()?;
    let mem = axil_memory::AgentMemory::new(&db);
    println!("Agent memory initialized at {}\n", db_path.display());

    // ── Session start: recall context ─────────────────────────────
    println!("=== Session Start ===");
    println!("(First session — no prior context)\n");

    // ── Store decisions ───────────────────────────────────────────
    println!("--- Storing decisions ---");
    db.insert(
        "decisions",
        json!({
            "summary": "Use JWT for auth tokens instead of sessions",
            "reason": "Stateless, scales horizontally, easy to revoke via short TTL",
        }),
    )?;
    db.insert(
        "decisions",
        json!({
            "summary": "Switch from REST to GraphQL for internal APIs",
            "reason": "Reduces over-fetching, better for mobile clients",
        }),
    )?;
    println!("  Stored 2 decisions");

    // ── Learn facts about entities ────────────────────────────────
    println!("\n--- Learning facts ---");
    mem.semantic()
        .know("auth-module", "Uses JWT with 15-minute expiry", None)?;
    mem.semantic().know(
        "auth-module",
        "Rate limits login to 5 attempts per minute",
        None,
    )?;
    mem.semantic()
        .know("database", "PostgreSQL 16 with pgvector extension", None)?;
    mem.semantic()
        .know("cache", "Redis cluster with 3 nodes", None)?;
    println!("  Learned 4 facts about 3 entities");

    // ── Add entity aliases ────────────────────────────────────────
    println!("\n--- Setting up aliases ---");
    mem.semantic().add_alias("auth-module", "the auth system")?;
    mem.semantic().add_alias("auth-module", "login service")?;
    println!("  'the auth system' -> auth-module");
    println!("  'login service' -> auth-module");

    // ── Resolve aliases ───────────────────────────────────────────
    let resolved = mem.semantic().resolve("the auth system")?;
    println!("\n  Resolve 'the auth system' -> {:?}", resolved);

    // ── Entity knowledge ─────────────────────────────────────────
    println!("\n--- Entity knowledge ---");
    let knowledge = mem.semantic().about("auth-module")?;
    println!("  Entity: {}", knowledge.entity);
    println!("  Facts: {}", knowledge.facts.len());
    for fact in &knowledge.facts {
        println!("    - {}", fact.data["fact"]);
    }
    let aliases = mem.semantic().aliases("auth-module")?;
    println!("  Aliases: {:?}", aliases);

    // ── Store an error ────────────────────────────────────────────
    println!("\n--- Recording error ---");
    db.insert(
        "errors",
        json!({
            "error": "Connection pool exhausted under load",
            "root_cause": "Default pool size too small for concurrent requests",
            "fix": "Increased pool size from 10 to 50, added connection timeout",
        }),
    )?;
    println!("  Stored 1 error");

    // ── Create an episode (episodic memory) ─────────────────────
    println!("\n--- Recording episode ---");
    mem.episodic().create(
        "Investigated auth performance, fixed connection pool sizing",
        axil_memory::Outcome::Success,
        Some(vec!["Use larger connection pool".into()]),
        Some(vec!["src/db/pool.rs".into()]),
    )?;
    println!("  Episode recorded with outcome: success");

    // ── Recall context (simulating next session) ──────────────────
    println!("\n=== Next Session — Recall ===");
    let decisions = db.list("decisions")?;
    println!("Prior decisions: {}", decisions.len());
    for d in &decisions {
        println!("  - {}", d.data["summary"]);
    }

    let errors = db.list("errors")?;
    println!("Prior errors: {}", errors.len());
    for e in &errors {
        println!("  - {}: {}", e.data["error"], e.data["fix"]);
    }

    // ── Reflect on patterns ───────────────────────────────────────
    println!("\n--- Reflection ---");
    let engine = axil_memory::ReflectEngine::new(&db);
    let report = engine.reflect(None, axil_memory::ReflectScope::All)?;
    println!("  Memories analyzed: {}", report.memories_analyzed);
    for insight in &report.insights {
        println!("  Insight: {insight}");
    }

    println!("\nDone.");
    Ok(())
}
