//! Graph queries example — relate records and traverse relationships.
//!
//! Run: `cargo run --example graph_queries --features graph`

use anyhow::Result;
use axil_core::{Axil, Direction};
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

fn main() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("demo.axil");

    // Open with graph plugin.
    let db = Axil::open(&db_path).with_graph_plugin()?.build()?;
    println!("Opened database with graph at {}", db_path.display());

    // ── Create entities ───────────────────────────────────────────
    let auth = db.insert("modules", json!({"name": "auth", "language": "rust"}))?;
    let db_mod = db.insert("modules", json!({"name": "database", "language": "rust"}))?;
    let api = db.insert(
        "modules",
        json!({"name": "api-gateway", "language": "rust"}),
    )?;
    let cache = db.insert("modules", json!({"name": "cache", "language": "rust"}))?;
    println!("Created 4 modules");

    // ── Create relationships ──────────────────────────────────────
    db.relate(&api.id, "depends_on", &auth.id, None)?;
    db.relate(&api.id, "depends_on", &db_mod.id, None)?;
    db.relate(&api.id, "depends_on", &cache.id, None)?;
    db.relate(&auth.id, "uses", &db_mod.id, None)?;
    db.relate(&cache.id, "caches", &db_mod.id, None)?;
    println!("Created 5 relationships\n");

    // ── Query neighbors ───────────────────────────────────────────
    println!("API gateway depends on:");
    let deps = db.neighbors(&api.id, Some("depends_on"), Direction::Out)?;
    for record in &deps {
        println!("  -> {}", record.data["name"]);
    }
    println!();

    // ── Reverse query (who depends on auth?) ──────────────────────
    println!("What depends on auth:");
    let dependents = db.neighbors(&auth.id, None, Direction::In)?;
    for record in &dependents {
        println!("  <- {}", record.data["name"]);
    }
    println!();

    // ── Traverse a path ───────────────────────────────────────────
    // Path syntax: ->edge_type (outgoing), <-edge_type (incoming)
    println!("Traverse: api-gateway ->depends_on-> * ->uses-> *");
    let path = db.traverse(&api.id, "->depends_on->uses")?;
    for record in &path {
        println!("  {} ({})", record.data["name"], record.table);
    }

    println!("\nDone.");
    Ok(())
}
