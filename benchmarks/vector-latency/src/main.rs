//! Vector latency benchmark for Axil.
//!
//! Measures insert throughput, index build time, and search latency
//! percentiles (p50/p95/p99) at configurable dataset scale.
//!
//! Usage:
//!   cargo run --release -p vector-latency-bench -- [OPTIONS]
//!
//! Defaults target the 100k vector search latency goal.

use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use tempfile::TempDir;

use axil_core::plugin::VectorIndex;
use axil_core::record::RecordId;
use axil_vector::VectorEngine;

#[derive(Parser, Debug)]
#[command(name = "vector-latency", about = "Vector latency benchmark for Axil")]
struct Args {
    /// Dataset size (vectors to populate the index with).
    #[arg(long, default_value = "100000")]
    n: usize,

    /// Vector dimensions.
    #[arg(long, default_value = "384")]
    dims: usize,

    /// Number of queries to time.
    #[arg(long, default_value = "1000")]
    queries: usize,

    /// Top-k values to benchmark (comma separated).
    #[arg(long, default_value = "1,10,100")]
    top_k: String,

    /// Warmup queries before timing begins.
    #[arg(long, default_value = "50")]
    warmup: usize,

    /// top_k for the cold-recall-after-single-store phase.
    #[arg(long, default_value = "10")]
    top_k_cold: usize,

    /// Output format: "markdown" (default) or "json".
    #[arg(long, default_value = "markdown")]
    format: String,
}

#[derive(Debug, Serialize)]
struct TopKReport {
    top_k: usize,
    queries: usize,
    mean_us: f64,
    min_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
    qps: f64,
}

#[derive(Debug, Serialize)]
struct ColdRecallReport {
    /// How many store-then-search iterations were timed.
    iterations: usize,
    /// top_k used for the cold-recall search.
    top_k: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    benchmark: String,
    dataset_size: usize,
    dimensions: usize,
    insert_ms: u128,
    insert_throughput_per_sec: f64,
    rebuild_ms: u128,
    search: Vec<TopKReport>,
    /// Cost of the agent's mandated store-then-recall loop: each iteration
    /// adds exactly one vector and times the immediately following search.
    /// With a full-rebuild backend this pays O(n); with an incremental graph
    /// it pays O(log n).
    cold_recall_after_single_store: ColdRecallReport,
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

fn run_topk(plugin: &VectorEngine, queries: &[Vec<f32>], warmup: usize, top_k: usize) -> TopKReport {
    for q in queries.iter().take(warmup) {
        let _ = plugin.search(q, top_k).unwrap();
    }

    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    for q in queries {
        let start = Instant::now();
        let _ = plugin.search(q, top_k).unwrap();
        latencies.push(start.elapsed());
    }
    latencies.sort();

    let sum_us: f64 = latencies.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum();
    let mean_us = sum_us / latencies.len() as f64;
    let qps = if mean_us > 0.0 { 1_000_000.0 / mean_us } else { 0.0 };

    TopKReport {
        top_k,
        queries: latencies.len(),
        mean_us,
        min_us: latencies.first().map(|d| d.as_secs_f64() * 1_000_000.0).unwrap_or(0.0),
        p50_us: percentile(&latencies, 0.50),
        p95_us: percentile(&latencies, 0.95),
        p99_us: percentile(&latencies, 0.99),
        max_us: latencies.last().map(|d| d.as_secs_f64() * 1_000_000.0).unwrap_or(0.0),
        qps,
    }
}

/// Time the store-then-recall loop: add one new vector, then immediately
/// search, repeated `iterations` times. Seeds are disjoint from the populated
/// corpus and the timed queries so every iteration adds a genuinely new vector.
fn run_cold_recall(
    plugin: &VectorEngine,
    dims: usize,
    populated: usize,
    iterations: usize,
    top_k: usize,
) -> ColdRecallReport {
    let mut latencies: Vec<Duration> = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let seed = 2_000_000 + populated + i;
        let v = make_vector(dims, seed);
        let id = RecordId::new();
        plugin.add(id, &v).expect("add");
        let q = make_vector(dims, seed);
        let start = Instant::now();
        let _ = plugin.search(&q, top_k).expect("search");
        latencies.push(start.elapsed());
    }
    latencies.sort();
    let sum_us: f64 = latencies.iter().map(|d| d.as_secs_f64() * 1_000_000.0).sum();
    let mean_us = if latencies.is_empty() {
        0.0
    } else {
        sum_us / latencies.len() as f64
    };
    ColdRecallReport {
        iterations: latencies.len(),
        top_k,
        mean_us,
        p50_us: percentile(&latencies, 0.50),
        p95_us: percentile(&latencies, 0.95),
        p99_us: percentile(&latencies, 0.99),
        max_us: latencies.last().map(|d| d.as_secs_f64() * 1_000_000.0).unwrap_or(0.0),
    }
}

fn main() {
    let args = Args::parse();

    let top_ks: Vec<usize> = args
        .top_k
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    eprintln!("vector-latency: n={} dims={} queries={} top_k={:?}",
        args.n, args.dims, args.queries, top_ks);

    let dir = TempDir::new().expect("tmpdir");
    let db_path = dir.path().join("bench.axil");
    let plugin = VectorEngine::open(&db_path, args.dims).expect("open vector plugin");

    // 1. Insert phase.
    eprintln!("[1/3] inserting {} vectors...", args.n);
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
    let insert_throughput_per_sec = if insert_ms > 0 {
        (args.n as f64) * 1000.0 / insert_ms as f64
    } else {
        0.0
    };

    // 2. Explicit rebuild (so per-query search doesn't pay the rebuild cost).
    eprintln!("[2/3] rebuilding HNSW index...");
    let rebuild_start = Instant::now();
    plugin.rebuild().expect("rebuild");
    let rebuild_ms = rebuild_start.elapsed().as_millis();

    // 3. Search phase — generate query vectors once, reuse across top_k runs.
    eprintln!("[3/3] running {} queries per top_k...", args.queries);
    let queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|i| make_vector(args.dims, 1_000_000 + i))
        .collect();

    let search_reports: Vec<TopKReport> = top_ks
        .iter()
        .map(|k| {
            let r = run_topk(&plugin, &queries, args.warmup, *k);
            eprintln!(
                "       top_k={:>3}  p50={:>8.1}us  p95={:>8.1}us  p99={:>8.1}us  qps={:>7.0}",
                r.top_k, r.p50_us, r.p95_us, r.p99_us, r.qps
            );
            r
        })
        .collect();

    // 4. Cold-recall after single store — the agent's mandated store-then-recall
    //    loop. Each iteration adds one fresh vector and times the immediately
    //    following search, so a full-rebuild backend pays O(n) per loop while an
    //    incremental graph pays O(log n).
    eprintln!("[4/4] cold-recall after single store ({} iterations)...", args.queries);
    let cold = run_cold_recall(&plugin, args.dims, args.n, args.queries, args.top_k_cold);
    eprintln!(
        "       cold-recall top_k={:>3}  p50={:>8.1}us  p95={:>8.1}us  p99={:>8.1}us",
        cold.top_k, cold.p50_us, cold.p95_us, cold.p99_us
    );

    let report = BenchReport {
        benchmark: "vector-latency".to_string(),
        dataset_size: args.n,
        dimensions: args.dims,
        insert_ms,
        insert_throughput_per_sec,
        rebuild_ms,
        search: search_reports,
        cold_recall_after_single_store: cold,
    };

    match args.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        _ => print_markdown(&report),
    }
}

fn print_markdown(r: &BenchReport) {
    println!("## Vector latency benchmark ({}k vectors, {} dims)\n",
        r.dataset_size / 1000, r.dimensions);
    println!("**Insert:** {} vectors in {} ms ({:.0} vec/s)  ",
        r.dataset_size, r.insert_ms, r.insert_throughput_per_sec);
    println!("**HNSW rebuild:** {} ms\n", r.rebuild_ms);
    println!("| top_k | queries | mean (us) | p50 (us) | p95 (us) | p99 (us) | max (us) | qps |");
    println!("|------:|--------:|----------:|---------:|---------:|---------:|---------:|-----:|");
    for s in &r.search {
        println!("| {:>5} | {:>7} | {:>9.1} | {:>8.1} | {:>8.1} | {:>8.1} | {:>8.1} | {:>4.0} |",
            s.top_k, s.queries, s.mean_us, s.p50_us, s.p95_us, s.p99_us, s.max_us, s.qps);
    }
    let c = &r.cold_recall_after_single_store;
    println!("\n**Cold recall after single store** (add 1 vector, then search; top_k={}, {} iters)  ",
        c.top_k, c.iterations);
    println!("mean={:.1}us  p50={:.1}us  p95={:.1}us  p99={:.1}us  max={:.1}us",
        c.mean_us, c.p50_us, c.p95_us, c.p99_us, c.max_us);
}
