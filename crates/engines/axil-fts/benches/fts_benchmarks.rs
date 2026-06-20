use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;

use axil_core::plugin::SearchIndex;
use axil_core::record::RecordId;
use axil_fts::FtsEngine;

/// Sample sentences for indexing.
const CORPUS: &[&str] = &[
    "Fixed authentication timeout in the login flow",
    "Refactored the database connection pooling layer",
    "Added retry logic for failed API requests",
    "Updated the CI pipeline to run integration tests",
    "Migrated user sessions from Redis to PostgreSQL",
    "Implemented rate limiting on the public API",
    "Fixed memory leak in the websocket handler",
    "Added OpenTelemetry tracing to all HTTP endpoints",
    "Redesigned the permission model for team workspaces",
    "Optimized SQL queries for the analytics dashboard",
];

fn open_fts() -> (FtsEngine, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bench.axil");
    let plugin = FtsEngine::open(&path).unwrap();
    (plugin, dir)
}

fn populated_fts(n: usize) -> (FtsEngine, TempDir) {
    let (plugin, dir) = open_fts();
    for i in 0..n {
        let id = RecordId::new();
        let text = CORPUS[i % CORPUS.len()];
        plugin.index_text(&id, "summary", text).unwrap();
    }
    (plugin, dir)
}

// ---------------------------------------------------------------------------
// Index throughput
// ---------------------------------------------------------------------------

fn bench_index_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("fts_index_text");
    for size in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let (plugin, _dir) = open_fts();
                for i in 0..n {
                    let id = RecordId::new();
                    let text = CORPUS[i % CORPUS.len()];
                    black_box(plugin.index_text(&id, "summary", text).unwrap());
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Search latency
// ---------------------------------------------------------------------------

fn bench_search_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("fts_search_text");
    let queries = ["authentication", "database connection", "API rate limiting"];

    for size in [1_000, 10_000] {
        let (plugin, _dir) = populated_fts(size);
        for query in &queries {
            let label = format!("{size}_{}", query.replace(' ', "_"));
            group.bench_function(&label, |b| {
                b.iter(|| {
                    black_box(plugin.search_text(query, 10).unwrap());
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Fuzzy search
// ---------------------------------------------------------------------------

fn bench_search_fuzzy(c: &mut Criterion) {
    let mut group = c.benchmark_group("fts_search_fuzzy");
    let (plugin, _dir) = populated_fts(10_000);

    group.bench_function("fuzzy_d1_10k", |b| {
        b.iter(|| {
            black_box(plugin.search_fuzzy("autentication", 1, 10).unwrap());
        });
    });

    group.bench_function("fuzzy_d2_10k", |b| {
        b.iter(|| {
            black_box(plugin.search_fuzzy("autntication", 2, 10).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_index_text,
    bench_search_text,
    bench_search_fuzzy
);
criterion_main!(benches);
