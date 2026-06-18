//! Vector search example — embed text fields and search by similarity.
//!
//! Run: `cargo run --example vector_search --features embed`
//!
//! Requires: bge-small model (auto-downloaded on first run).

use anyhow::Result;
use axil_core::Axil;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("demo.axil");

    // Open with vector search (384-dim BGE-small, auto-downloaded).
    let db = Axil::open(&db_path)
        .with_embedder_model(axil_vector::models::EmbeddingModel::BgeSmall)?
        .build()?;
    println!(
        "Opened database with vector search at {}",
        db_path.display()
    );

    // ── Insert and embed records ──────────────────────────────────
    let memories = vec![
        json!({"summary": "Fixed authentication timeout bug in login flow", "project": "web-app"}),
        json!({"summary": "Deployed new caching layer with Redis", "project": "backend"}),
        json!({"summary": "Refactored database migration scripts", "project": "backend"}),
        json!({"summary": "Added rate limiting to API endpoints", "project": "web-app"}),
        json!({"summary": "Investigated memory leak in worker process", "project": "backend"}),
    ];

    for data in memories {
        let record = db.insert("sessions", data)?;
        db.embed_field(&record.id, "summary")?;
        println!("  Embedded: {}", record.data["summary"]);
    }
    println!();

    // ── Semantic search ───────────────────────────────────────────
    let query = "auth error";
    println!("Search: \"{query}\"");
    let results = db.similar_to(query, 3)?;
    for (record, score) in &results {
        println!("  [{score:.3}] {}", record.data["summary"]);
    }
    println!();

    // ── Another search ────────────────────────────────────────────
    let query = "database performance";
    println!("Search: \"{query}\"");
    let results = db.similar_to(query, 3)?;
    for (record, score) in &results {
        println!("  [{score:.3}] {}", record.data["summary"]);
    }

    println!("\nDone.");
    Ok(())
}
