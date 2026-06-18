//! Head-to-head benchmark: Axil vector plugin vs SQLite + sqlite-vec.
//!
//! Both engines receive the same synthetic dataset (deterministic vectors)
//! and are queried with the same set of query vectors. Latency percentiles
//! and insert throughput are reported side by side.
//!
//! Usage:
//!   cargo run --release -p sqlite-compare-bench -- [OPTIONS]

use std::time::{Duration, Instant};

use clap::Parser;
use rusqlite::{ffi::sqlite3_auto_extension, Connection};
use serde::Serialize;
use sqlite_vec::sqlite3_vec_init;
use tempfile::TempDir;
use zerocopy::IntoBytes;

use axil_core::plugin::VectorIndex;
use axil_core::record::RecordId;
use axil_vector::VectorPlugin;

#[derive(Parser, Debug)]
#[command(name = "sqlite-compare", about = "Axil vs SQLite+sqlite-vec benchmark")]
struct Args {
    /// Dataset size.
    #[arg(long, default_value = "100000")]
    n: usize,

    /// Vector dimensions.
    #[arg(long, default_value = "384")]
    dims: usize,

    /// Number of queries to time.
    #[arg(long, default_value = "500")]
    queries: usize,

    /// Top-k for ANN search.
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Warmup queries before timing.
    #[arg(long, default_value = "50")]
    warmup: usize,

    /// Batch size for sqlite-vec inserts (wrapped in a transaction).
    #[arg(long, default_value = "1000")]
    sqlite_batch: usize,

    /// Output format: "markdown" (default) or "json".
    #[arg(long, default_value = "markdown")]
    format: String,
}

#[derive(Debug, Serialize)]
struct EngineResult {
    engine: String,
    insert_ms: u128,
    insert_throughput_per_sec: f64,
    build_ms: u128,
    queries: usize,
    top_k: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
    qps: f64,
    disk_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    benchmark: String,
    dataset_size: usize,
    dimensions: usize,
    axil: EngineResult,
    sqlite: EngineResult,
}

fn make_vector(dims: usize, seed: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dims);
    for i in 0..dims {
        let val = ((seed.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503)) % 10_000) as f32
            / 10_000.0;
        v.push(val - 0.5);
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)].as_secs_f64() * 1_000_000.0
}

fn dir_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    for entry in walkdir(path) {
        if let Ok(meta) = std::fs::metadata(&entry) {
            if meta.is_file() {
                total += meta.len();
            }
        }
    }
    total
}

fn walkdir(path: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(p) = stack.pop() {
        if let Ok(meta) = std::fs::metadata(&p) {
            if meta.is_dir() {
                if let Ok(rd) = std::fs::read_dir(&p) {
                    for entry in rd.flatten() {
                        stack.push(entry.path());
                    }
                }
            } else {
                out.push(p);
            }
        }
    }
    out
}

// ── Axil side ───────────────────────────────────────────────────────

fn run_axil(args: &Args) -> EngineResult {
    eprintln!("[axil] inserting {} vectors...", args.n);
    let dir = TempDir::new().expect("tmpdir");
    let db_path = dir.path().join("bench.axil");
    let plugin = VectorPlugin::open(&db_path, args.dims).expect("open plugin");

    let insert_start = Instant::now();
    for i in 0..args.n {
        let id = RecordId::new();
        let v = make_vector(args.dims, i);
        plugin.add(id, &v).expect("add");
        if (i + 1) % 10_000 == 0 {
            eprintln!("       {}/{}", i + 1, args.n);
        }
    }
    let insert_ms = insert_start.elapsed().as_millis();

    eprintln!("[axil] rebuilding HNSW...");
    let build_start = Instant::now();
    plugin.rebuild().expect("rebuild");
    let build_ms = build_start.elapsed().as_millis();

    eprintln!("[axil] running {} queries (top_k={})...", args.queries, args.top_k);
    let queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|i| make_vector(args.dims, 1_000_000 + i))
        .collect();

    for q in queries.iter().take(args.warmup) {
        let _ = plugin.search(q, args.top_k).unwrap();
    }

    let mut lat: Vec<Duration> = Vec::with_capacity(queries.len());
    for q in &queries {
        let s = Instant::now();
        let _ = plugin.search(q, args.top_k).unwrap();
        lat.push(s.elapsed());
    }
    lat.sort();

    let sum_us: f64 = lat.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum();
    let mean_us = sum_us / lat.len() as f64;
    let disk_bytes = dir_bytes(dir.path());

    // Drop plugin before computing dir_bytes? Already open — but the files are
    // already flushed. We measure disk usage with the plugin still open.

    EngineResult {
        engine: "axil-vector (HNSW)".to_string(),
        insert_ms,
        insert_throughput_per_sec: if insert_ms > 0 {
            (args.n as f64) * 1000.0 / insert_ms as f64
        } else {
            0.0
        },
        build_ms,
        queries: lat.len(),
        top_k: args.top_k,
        mean_us,
        p50_us: percentile(&lat, 0.50),
        p95_us: percentile(&lat, 0.95),
        p99_us: percentile(&lat, 0.99),
        max_us: lat.last().map(|d| d.as_secs_f64() * 1_000_000.0).unwrap_or(0.0),
        qps: if mean_us > 0.0 { 1_000_000.0 / mean_us } else { 0.0 },
        disk_bytes,
    }
}

// ── SQLite side ─────────────────────────────────────────────────────

fn run_sqlite(args: &Args) -> EngineResult {
    // Register sqlite-vec auto-extension once; safe to call repeatedly.
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    eprintln!("[sqlite] inserting {} vectors...", args.n);
    let dir = TempDir::new().expect("tmpdir");
    let db_path = dir.path().join("bench.sqlite");
    let conn = Connection::open(&db_path).expect("sqlite open");

    // Pragmas commonly used for bulk insert speed; reasonable defaults.
    conn.pragma_update(None, "journal_mode", "WAL").ok();
    conn.pragma_update(None, "synchronous", "NORMAL").ok();

    let create_sql = format!(
        "CREATE VIRTUAL TABLE vecs USING vec0(embedding FLOAT[{}])",
        args.dims
    );
    conn.execute(&create_sql, []).expect("create vec0");

    let insert_start = Instant::now();
    let mut i = 0usize;
    while i < args.n {
        let end = (i + args.sqlite_batch).min(args.n);
        let txn = conn.unchecked_transaction().expect("tx");
        {
            let mut stmt = txn
                .prepare("INSERT INTO vecs(rowid, embedding) VALUES (?, ?)")
                .expect("prepare");
            for k in i..end {
                let v = make_vector(args.dims, k);
                let bytes: &[u8] = v.as_bytes();
                stmt.execute(rusqlite::params![k as i64, bytes])
                    .expect("insert");
            }
        }
        txn.commit().expect("commit");
        i = end;
        if i % 10_000 == 0 {
            eprintln!("       {}/{}", i, args.n);
        }
    }
    let insert_ms = insert_start.elapsed().as_millis();

    // vec0 uses flat brute-force search — no separate index build phase.
    let build_ms = 0u128;

    eprintln!("[sqlite] running {} queries (top_k={})...", args.queries, args.top_k);
    let queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|i| make_vector(args.dims, 1_000_000 + i))
        .collect();

    let query_sql =
        "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? ORDER BY distance LIMIT ?";
    let mut stmt = conn.prepare(query_sql).expect("prepare query");

    for q in queries.iter().take(args.warmup) {
        let bytes: &[u8] = q.as_bytes();
        let rows = stmt
            .query_map(rusqlite::params![bytes, args.top_k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .expect("warmup query");
        let _ = rows.count();
    }

    let mut lat: Vec<Duration> = Vec::with_capacity(queries.len());
    for q in &queries {
        let bytes: &[u8] = q.as_bytes();
        let s = Instant::now();
        let rows = stmt
            .query_map(rusqlite::params![bytes, args.top_k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .expect("query");
        let _ = rows.count();
        lat.push(s.elapsed());
    }
    lat.sort();

    let sum_us: f64 = lat.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum();
    let mean_us = sum_us / lat.len() as f64;

    // Drop statement + close connection so WAL is checkpointed before sizing.
    drop(stmt);
    drop(conn);
    let disk_bytes = dir_bytes(dir.path());

    EngineResult {
        engine: "sqlite-vec (vec0, brute force)".to_string(),
        insert_ms,
        insert_throughput_per_sec: if insert_ms > 0 {
            (args.n as f64) * 1000.0 / insert_ms as f64
        } else {
            0.0
        },
        build_ms,
        queries: lat.len(),
        top_k: args.top_k,
        mean_us,
        p50_us: percentile(&lat, 0.50),
        p95_us: percentile(&lat, 0.95),
        p99_us: percentile(&lat, 0.99),
        max_us: lat.last().map(|d| d.as_secs_f64() * 1_000_000.0).unwrap_or(0.0),
        qps: if mean_us > 0.0 { 1_000_000.0 / mean_us } else { 0.0 },
        disk_bytes,
    }
}

fn main() {
    let args = Args::parse();

    eprintln!(
        "sqlite-compare: n={} dims={} queries={} top_k={}",
        args.n, args.dims, args.queries, args.top_k
    );

    let axil = run_axil(&args);
    let sqlite = run_sqlite(&args);

    let report = Report {
        benchmark: "axil-vs-sqlite-vec".to_string(),
        dataset_size: args.n,
        dimensions: args.dims,
        axil,
        sqlite,
    };

    match args.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        _ => print_markdown(&report),
    }
}

fn print_markdown(r: &Report) {
    println!(
        "## Axil vs SQLite+sqlite-vec ({}k vectors, {} dims, top_k={})\n",
        r.dataset_size / 1000,
        r.dimensions,
        r.axil.top_k
    );
    println!("| metric | axil-vector (HNSW) | sqlite-vec (brute force) |");
    println!("|--------|---------------------:|-------------------------:|");
    println!(
        "| insert time | {} ms | {} ms |",
        r.axil.insert_ms, r.sqlite.insert_ms
    );
    println!(
        "| insert throughput | {:.0} vec/s | {:.0} vec/s |",
        r.axil.insert_throughput_per_sec, r.sqlite.insert_throughput_per_sec
    );
    println!(
        "| index build time | {} ms | {} ms (flat — no build) |",
        r.axil.build_ms, r.sqlite.build_ms
    );
    println!(
        "| search mean | {:.1} us | {:.1} us |",
        r.axil.mean_us, r.sqlite.mean_us
    );
    println!(
        "| search p50 | {:.1} us | {:.1} us |",
        r.axil.p50_us, r.sqlite.p50_us
    );
    println!(
        "| search p95 | {:.1} us | {:.1} us |",
        r.axil.p95_us, r.sqlite.p95_us
    );
    println!(
        "| search p99 | {:.1} us | {:.1} us |",
        r.axil.p99_us, r.sqlite.p99_us
    );
    println!(
        "| qps | {:.0} | {:.0} |",
        r.axil.qps, r.sqlite.qps
    );
    println!(
        "| disk usage | {:.1} MB | {:.1} MB |",
        r.axil.disk_bytes as f64 / 1_048_576.0,
        r.sqlite.disk_bytes as f64 / 1_048_576.0
    );
    println!();
    println!("_Axil uses HNSW (approximate); sqlite-vec's vec0 is exact brute force. Recall differs — this compares raw latency + insert cost on equivalent workload, not equal algorithms._");
}
