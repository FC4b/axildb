use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;

use axil_core::plugin::{Engine, VectorIndex};
use axil_core::record::RecordId;
use axil_vector::VectorEngine;

/// Generate a random-ish vector of given dimensions (deterministic for reproducibility).
fn make_vector(dims: usize, seed: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dims);
    for i in 0..dims {
        let val = ((seed * 7 + i * 13) % 1000) as f32 / 1000.0;
        v.push(val);
    }
    // L2 normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn open_vector_engine(dims: usize) -> (VectorEngine, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bench.axil");
    let plugin = VectorEngine::open(&path, dims).unwrap();
    (plugin, dir)
}

fn prepopulated_vector(dims: usize, n: usize) -> (VectorEngine, Vec<RecordId>, TempDir) {
    let (plugin, dir) = open_vector_engine(dims);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = RecordId::new();
        plugin.add(id.clone(), &make_vector(dims, i)).unwrap();
        ids.push(id);
    }
    // Force rebuild so searches use a clean index.
    plugin.rebuild().unwrap();
    (plugin, ids, dir)
}

// ---------------------------------------------------------------------------
// Vector add throughput
// ---------------------------------------------------------------------------

fn bench_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");
    let dims = 384;
    for size in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let (plugin, _dir) = open_vector_engine(dims);
                for i in 0..n {
                    let id = RecordId::new();
                    black_box(plugin.add(id, &make_vector(dims, i)).unwrap());
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Vector search (top-5 and top-50)
// ---------------------------------------------------------------------------

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    let dims = 384;

    for (dataset_size, top_k) in [(1_000, 5), (1_000, 50), (10_000, 5), (10_000, 50)] {
        let label = format!("{dataset_size}_top{top_k}");
        let (plugin, _ids, _dir) = prepopulated_vector(dims, dataset_size);
        let query = make_vector(dims, 999_999);

        group.bench_function(&label, |b| {
            b.iter(|| {
                black_box(plugin.search(&query, top_k).unwrap());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Vector delete
// ---------------------------------------------------------------------------

fn bench_vector_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_delete");
    let dims = 384;
    let n = 1_000;

    group.bench_function("delete_1000", |b| {
        b.iter_with_setup(
            || prepopulated_vector(dims, n),
            |(plugin, ids, _dir)| {
                for id in &ids {
                    black_box(plugin.on_record_delete(id).unwrap());
                }
            },
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_vector_add,
    bench_vector_search,
    bench_vector_delete
);
criterion_main!(benches);
