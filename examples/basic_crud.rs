//! Basic CRUD operations with Axil.
//!
//! Run: `cargo run --example basic_crud`

use anyhow::Result;
use axil_core::Axil;
use serde_json::json;

fn main() -> Result<()> {
    // Create a temporary database that is cleaned up when `dir` is dropped.
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("demo.axil");

    // ── Open / create ──────────────────────────────────────────────
    let db = Axil::open(&db_path).build()?;
    println!("Opened database at {}", db_path.display());

    // ── Insert records ─────────────────────────────────────────────
    let session = db.insert(
        "sessions",
        json!({
            "summary": "Fixed auth timeout bug in login flow",
            "project": "my-app",
            "priority": 2
        }),
    )?;
    println!("\nInserted session: {}", session.id);

    let decision = db.insert(
        "decisions",
        json!({
            "title": "Use JWT for session tokens",
            "rationale": "Stateless, easy to revoke via short TTL",
            "status": "accepted"
        }),
    )?;
    println!("Inserted decision: {}", decision.id);

    // ── Get by ID ──────────────────────────────────────────────────
    if let Some(record) = db.get(&session.id)? {
        println!("\nGet by ID:");
        println!("  table:   {}", record.table);
        println!("  data:    {}", record.data);
        println!("  created: {}", record.created_at);
    }

    // ── Update a record ────────────────────────────────────────────
    let updated = db.update(
        &session.id,
        json!({
            "summary": "Fixed auth timeout bug in login flow",
            "project": "my-app",
            "priority": 1,
            "resolved": true
        }),
    )?;
    println!("\nUpdated session:");
    println!("  data:       {}", updated.data);
    println!("  updated_at: {}", updated.updated_at);

    // ── Batch insert ───────────────────────────────────────────────
    let patterns = vec![
        json!({"name": "retry-with-backoff", "language": "rust", "uses": 12}),
        json!({"name": "builder-pattern",    "language": "rust", "uses": 34}),
        json!({"name": "singleton",          "language": "java", "uses": 5}),
        json!({"name": "observer",           "language": "rust", "uses": 19}),
    ];
    let batch = db.insert_batch("patterns", patterns)?;
    println!("\nBatch inserted {} patterns", batch.len());
    for r in &batch {
        println!("  {} — {}", r.id, r.data["name"]);
    }

    // ── List all records in a table ────────────────────────────────
    let all_patterns = db.list("patterns")?;
    println!("\nAll patterns ({} total):", all_patterns.len());
    for r in &all_patterns {
        println!("  {} ({})", r.data["name"], r.data["language"]);
    }

    // ── Database info ──────────────────────────────────────────────
    let tables = db.tables()?;
    println!("\nTables: {:?}", tables);

    let total = db.total_records()?;
    println!("Total records: {total}");

    for t in &tables {
        let count = db.count(t)?;
        println!("  {t}: {count} records");
    }

    // ── Delete a record ────────────────────────────────────────────
    let deleted = db.delete(&decision.id)?;
    println!("\nDeleted decision {}: {deleted}", decision.id);

    // Verify deletion.
    let gone = db.get(&decision.id)?;
    println!("Get after delete: {:?}", gone);

    println!("\nTotal records after delete: {}", db.total_records()?);

    // Temp directory and database files are cleaned up automatically.
    println!("\nDone.");
    Ok(())
}
