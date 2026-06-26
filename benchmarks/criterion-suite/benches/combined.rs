//! Combined benchmarks: real-world hot paths using multiple plugins together.
//!
//! Tests the recall query pattern: insert records, embed, create graph edges,
//! index FTS, then query using vector similarity + graph traversal + recency.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::json;
use tempfile::TempDir;

use axil_core::plugin::Direction;
use axil_core::record::RecordId;
use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use axil_vector::AxilBuilderVectorExt;

const DIMS: usize = 384;

/// Generate a deterministic pseudo-random vector.
fn make_vector(dims: usize, seed: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dims);
    for i in 0..dims {
        let val = ((seed * 7 + i * 13) % 1000) as f32 / 1000.0;
        v.push(val);
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn open_full_db() -> (Axil, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("combined.axil");
    let db = Axil::open(&path)
        .with_vector(DIMS)
        .unwrap()
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

/// Populate a full database with n records, each with vector + graph edges + FTS.
fn populated_full_db(n: usize) -> (Axil, Vec<RecordId>, TempDir) {
    let (db, dir) = open_full_db();
    let mut ids = Vec::with_capacity(n);

    for i in 0..n {
        let record = db
            .insert(
                "sessions",
                json!({
                    "summary": format!("Session {i}: working on feature {}", i % 20),
                    "project": format!("project-{}", i % 5),
                    "priority": i % 5,
                }),
            )
            .unwrap();
        let id = record.id.clone();

        // Add vector
        db.add_vector(&id, &make_vector(DIMS, i)).unwrap();

        // FTS auto-indexed on insert via plugin

        // Graph: link sequential sessions
        if i > 0 {
            db.relate(&id, "follows", &ids[i - 1], None).unwrap();
        }
        // Link to a "project entity" (reuse earlier IDs as entity stand-ins)
        if i >= 5 {
            db.relate(&id, "mentions", &ids[i % 5], None).unwrap();
        }

        ids.push(id);
    }
    (db, ids, dir)
}

// ---------------------------------------------------------------------------
// Combined insert (record + vector + graph edge + FTS)
// ---------------------------------------------------------------------------

fn bench_combined_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_insert");

    for size in [100, 1_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let (db, _dir) = open_full_db();
                let mut prev_id: Option<RecordId> = None;
                for i in 0..n {
                    let record = db
                        .insert(
                            "sessions",
                            json!({
                                "summary": format!("Session {i}: test data"),
                                "project": "bench",
                            }),
                        )
                        .unwrap();
                    let id = record.id.clone();
                    db.add_vector(&id, &make_vector(DIMS, i)).unwrap();
                    if let Some(ref prev) = prev_id {
                        db.relate(&id, "follows", prev, None).unwrap();
                    }
                    prev_id = Some(id);
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Recall query: vector search + filter + graph traversal (the hot path)
// ---------------------------------------------------------------------------

fn bench_recall_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_query");

    for size in [1_000, 10_000] {
        let (db, _ids, _dir) = populated_full_db(size);
        let query_vec = make_vector(DIMS, 999_999);

        // Pure vector search
        group.bench_function(format!("vector_only_{size}"), |b| {
            b.iter(|| {
                black_box(db.similar_to_vector(&query_vec, 10).unwrap());
            });
        });

        // Vector search + field filter
        group.bench_function(format!("vector_filter_{size}"), |b| {
            b.iter(|| {
                let vector_hits = db.similar_to_vector(&query_vec, 50).unwrap();
                let mut results = Vec::new();
                for (record, score) in &vector_hits {
                    if record.data.get("priority").and_then(|v| v.as_u64()) == Some(1) {
                        results.push((record.id.clone(), *score));
                    }
                    if results.len() >= 10 {
                        break;
                    }
                }
                black_box(results);
            });
        });

        // Vector + graph traversal (follow "mentions" edges from top hits)
        group.bench_function(format!("vector_graph_{size}"), |b| {
            b.iter(|| {
                let vector_hits = db.similar_to_vector(&query_vec, 10).unwrap();
                let mut related = Vec::new();
                for (record, _score) in &vector_hits {
                    if let Ok(neighbors) =
                        db.neighbors(&record.id, Some("mentions"), Direction::Out)
                    {
                        related.extend(neighbors);
                    }
                }
                black_box(related);
            });
        });

        // FTS search
        group.bench_function(format!("fts_only_{size}"), |b| {
            b.iter(|| {
                black_box(db.search_text("feature working", 10).unwrap());
            });
        });

        // Combined: vector top-50 -> graph expand -> FTS rerank (full recall pipeline)
        group.bench_function(format!("full_recall_{size}"), |b| {
            b.iter(|| {
                // Step 1: Vector similarity
                let vector_hits = db.similar_to_vector(&query_vec, 50).unwrap();

                // Step 2: Graph expansion
                let mut expanded: Vec<RecordId> =
                    vector_hits.iter().map(|(r, _)| r.id.clone()).collect();
                for (record, _) in vector_hits.iter().take(10) {
                    if let Ok(neighbors) =
                        db.neighbors(&record.id, None, Direction::Out)
                    {
                        for r in neighbors {
                            expanded.push(r.id.clone());
                        }
                    }
                }

                // Step 3: FTS scoring (would normally be part of RRF)
                let fts_hits = db.search_text("session feature", 20).unwrap();
                for (r, _) in fts_hits {
                    expanded.push(r.id.clone());
                }

                black_box(expanded);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Delete cascade (record + vector + graph edges)
// ---------------------------------------------------------------------------

fn bench_combined_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_delete");

    group.bench_function("delete_with_cascade_100", |b| {
        b.iter_with_setup(
            || populated_full_db(100),
            |(db, ids, _dir)| {
                for id in ids.iter().take(50) {
                    black_box(db.delete(id).unwrap());
                }
            },
        );
    });

    group.finish();
}

criterion_group!(benches, bench_combined_insert, bench_recall_query, bench_combined_delete);
criterion_main!(benches);
