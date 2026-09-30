//! Time the index freshness scan the way `axil recall` and `axil boot` run it.
//!
//! Every one-shot recall calls [`stale_file_paths`] and every boot calls
//! [`check_freshness`]; both list the stored file hashes, walk the project and
//! re-hash every file on disk. This prints how long that takes on a real
//! database and tree, split into its three parts, so the cost is known before
//! deciding whether a long-lived process should cache it.
//!
//! ```text
//! cargo run --release -p axil-indexer --example freshness_timing -- \
//!     <db path> <project root> [iterations]
//! ```
//!
//! Prints one JSON object: millisecond timings of the first (cold) call and
//! p50/p95 over the rest. Open the database on a copy: opening may migrate it.

use std::path::PathBuf;
use std::time::Instant;

use axil_indexer::freshness::{check_freshness, stale_file_paths};
use axil_indexer::indexer::{hash_content, TABLE_FILES};
use axil_indexer::scanner::scan_files;
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: freshness_timing <db path> <project root> [iterations]");
        std::process::exit(2);
    }
    let db_path = PathBuf::from(&args[0]);
    let root = PathBuf::from(&args[1]);
    let iterations: usize = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(20);

    let db = axil_core::Axil::open(&db_path)
        .build()
        .unwrap_or_else(|e| panic!("open {}: {e}", db_path.display()));
    // The same config lookup `axil recall` does for its project root.
    let config = axil_core::load_config_from(&root)
        .unwrap_or_else(|e| panic!("config: {e}"))
        .index;

    let mut stale_ms = Vec::new();
    let mut check_ms = Vec::new();
    let mut list_ms = Vec::new();
    let mut walk_ms = Vec::new();
    let mut hash_ms = Vec::new();
    let mut stale_count = 0;
    let mut indexed = 0;
    let mut on_disk = 0;
    let mut hashed_bytes = 0u64;

    for _ in 0..iterations.max(1) {
        let t = Instant::now();
        stale_count = stale_file_paths(&db, &root, &config).len();
        stale_ms.push(ms(t));

        let t = Instant::now();
        let report = check_freshness(&db, &root, &config);
        check_ms.push(ms(t));
        indexed = report.indexed_files;
        on_disk = report.disk_files;

        let t = Instant::now();
        let stored = db.list(TABLE_FILES).map(|rows| rows.len()).unwrap_or(0);
        list_ms.push(ms(t));
        std::hint::black_box(stored);

        let t = Instant::now();
        let files = scan_files(&root, &config);
        walk_ms.push(ms(t));

        let t = Instant::now();
        hashed_bytes = 0;
        for file in &files {
            if let Ok(source) = std::fs::read_to_string(&file.path) {
                hashed_bytes += source.len() as u64;
                std::hint::black_box(hash_content(&source));
            }
        }
        hash_ms.push(ms(t));
    }

    let out = json!({
        "iterations": iterations,
        "indexed_files": indexed,
        "disk_files": on_disk,
        "stale_files": stale_count,
        "hashed_bytes": hashed_bytes,
        "stale_file_paths_ms": summary(&stale_ms),
        "check_freshness_ms": summary(&check_ms),
        "parts_ms": {
            "list_stored_hashes": summary(&list_ms),
            "walk_project": summary(&walk_ms),
            "read_and_hash": summary(&hash_ms),
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// First call alone (cold caches), then p50/p95 over the warm calls.
fn summary(samples: &[f64]) -> serde_json::Value {
    let first = samples.first().copied().unwrap_or(0.0);
    let mut warm: Vec<f64> = samples.iter().skip(1).copied().collect();
    if warm.is_empty() {
        warm.push(first);
    }
    warm.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| {
        let idx = ((warm.len() as f64 * q).ceil() as usize).clamp(1, warm.len()) - 1;
        (warm[idx] * 10.0).round() / 10.0
    };
    json!({
        "first": (first * 10.0).round() / 10.0,
        "p50": pick(0.50),
        "p95": pick(0.95),
    })
}
