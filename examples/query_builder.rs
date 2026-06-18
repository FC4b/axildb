//! Query builder examples — filters, sorting, pagination, and explain.
//!
//! Run: `cargo run --example query_builder`

use anyhow::Result;
use axil_core::{Axil, Op, SortDirection};
use serde_json::json;

fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("queries.axil");
    let db = Axil::open(&db_path).build()?;

    // ── Seed data ──────────────────────────────────────────────────
    let tasks = vec![
        json!({"title": "Implement auth middleware",  "project": "backend",  "priority": 1, "status": "done"}),
        json!({"title": "Add rate limiting",          "project": "backend",  "priority": 2, "status": "in_progress"}),
        json!({"title": "Fix CSS overflow bug",       "project": "frontend", "priority": 3, "status": "open"}),
        json!({"title": "Refactor database layer",    "project": "backend",  "priority": 1, "status": "open"}),
        json!({"title": "Write API documentation",    "project": "docs",     "priority": 2, "status": "open"}),
        json!({"title": "Add dark mode toggle",       "project": "frontend", "priority": 3, "status": "done"}),
        json!({"title": "Set up CI pipeline",         "project": "infra",    "priority": 1, "status": "done"}),
        json!({"title": "Optimize query planner",     "project": "backend",  "priority": 2, "status": "in_progress"}),
        json!({"title": "Add search autocomplete",    "project": "frontend", "priority": 2, "status": "open"}),
        json!({"title": "Migrate to Rust 2024",       "project": "infra",    "priority": 3, "status": "open"}),
    ];
    db.insert_batch("tasks", tasks)?;
    println!("Seeded 10 tasks.\n");

    // ── Equality filter ────────────────────────────────────────────
    println!("=== Tasks with status == 'open' ===");
    let results = db
        .query()
        .table("tasks")
        .where_field("status", Op::Eq, json!("open"))
        .exec()?;
    for r in &results {
        println!("  [{}] {}", r.data["project"], r.data["title"]);
    }

    // ── Greater-than filter ────────────────────────────────────────
    println!("\n=== Tasks with priority > 1 (lower priority) ===");
    let results = db
        .query()
        .table("tasks")
        .where_field("priority", Op::Gt, json!(1))
        .exec()?;
    for r in &results {
        println!("  priority={} — {}", r.data["priority"], r.data["title"]);
    }

    // ── Less-than filter ───────────────────────────────────────────
    println!("\n=== Tasks with priority < 3 (higher priority) ===");
    let results = db
        .query()
        .table("tasks")
        .where_field("priority", Op::Lt, json!(3))
        .exec()?;
    for r in &results {
        println!("  priority={} — {}", r.data["priority"], r.data["title"]);
    }

    // ── Contains filter ────────────────────────────────────────────
    println!("\n=== Tasks whose title contains 'auth' ===");
    let results = db
        .query()
        .table("tasks")
        .where_field("title", Op::Contains, json!("auth"))
        .exec()?;
    for r in &results {
        println!("  {}", r.data["title"]);
    }

    // ── Combined filters ───────────────────────────────────────────
    println!("\n=== Backend tasks that are open ===");
    let results = db
        .query()
        .table("tasks")
        .where_field("project", Op::Eq, json!("backend"))
        .where_field("status", Op::Eq, json!("open"))
        .exec()?;
    for r in &results {
        println!("  {} (priority={})", r.data["title"], r.data["priority"]);
    }

    // ── Sorting ascending ──────────────────────────────────────────
    println!("\n=== All tasks sorted by priority (ascending) ===");
    let results = db
        .query()
        .table("tasks")
        .order_by("priority", SortDirection::Asc)
        .exec()?;
    for r in &results {
        println!("  priority={} — {}", r.data["priority"], r.data["title"]);
    }

    // ── Sorting descending ─────────────────────────────────────────
    println!("\n=== All tasks sorted by title (descending) ===");
    let results = db
        .query()
        .table("tasks")
        .order_by("title", SortDirection::Desc)
        .exec()?;
    for r in &results {
        println!("  {}", r.data["title"]);
    }

    // ── Time-based sorting ─────────────────────────────────────────
    println!("\n=== Tasks sorted by creation time (newest first) ===");
    let results = db
        .query()
        .table("tasks")
        .order_by_time(SortDirection::Desc)
        .limit(5)
        .exec()?;
    println!("  (showing top 5)");
    for r in &results {
        println!(
            "  {} — {}",
            r.created_at.format("%H:%M:%S%.3f"),
            r.data["title"]
        );
    }

    // ── Pagination ─────────────────────────────────────────────────
    println!("\n=== Pagination: 3 per page ===");
    let page_size = 3;
    let total = db.count("tasks")?;
    let pages = (total + page_size - 1) / page_size;
    for page in 0..pages {
        let results = db
            .query()
            .table("tasks")
            .order_by("priority", SortDirection::Asc)
            .limit(page_size)
            .offset(page * page_size)
            .exec()?;
        println!("  Page {} ({} results):", page + 1, results.len());
        for r in &results {
            println!("    priority={} — {}", r.data["priority"], r.data["title"]);
        }
    }

    // ── Explain mode ───────────────────────────────────────────────
    println!("\n=== Query plan (explain) ===");
    let plan = db
        .query()
        .table("tasks")
        .where_field("project", Op::Eq, json!("backend"))
        .where_field("priority", Op::Lt, json!(3))
        .order_by("priority", SortDirection::Asc)
        .limit(5)
        .explain();
    println!("  Estimated cost: {:?}", plan.estimated_cost);
    for step in &plan.plan {
        println!("  Step {}: {} — {}", step.step, step.step_type, step.params);
    }

    println!("\nDone.");
    Ok(())
}
