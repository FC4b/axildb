use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::json;
use tempfile::TempDir;

use axil_core::{Axil, Op, SortDirection};

/// Open a fresh Axil database in a temporary directory.
fn open_temp_db() -> (Axil, TempDir) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let db_path = dir.path().join("bench.axil");
    let db = Axil::open(&db_path).build().expect("failed to open db");
    (db, dir)
}

/// Pre-populate a database with `n` records and return it.
fn prepopulated_db(n: usize) -> (Axil, TempDir) {
    let (db, dir) = open_temp_db();
    for i in 0..n {
        db.insert(
            "items",
            json!({
                "title": format!("Record {i}"),
                "priority": i % 5,
                "category": format!("cat-{}", i % 10),
                "value": i * 7,
            }),
        )
        .expect("insert failed");
    }
    (db, dir)
}

// ---------------------------------------------------------------------------
// a) Insert throughput
// ---------------------------------------------------------------------------

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");
    for size in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let (db, _dir) = open_temp_db();
                for i in 0..n {
                    black_box(
                        db.insert(
                            "items",
                            json!({
                                "title": format!("Record {i}"),
                                "priority": i % 5,
                                "category": format!("cat-{}", i % 10),
                            }),
                        )
                        .expect("insert failed"),
                    );
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// b) Get latency
// ---------------------------------------------------------------------------

fn bench_get_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_latency");
    for size in [100, 1_000, 10_000] {
        // Pre-populate outside the measured section.
        let (db, _dir) = prepopulated_db(size);
        let ids: Vec<_> = db
            .list("items")
            .expect("list failed")
            .iter()
            .map(|r| r.id.clone())
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let mut idx = 0usize;
            b.iter(|| {
                let id = &ids[idx % ids.len()];
                idx += 1;
                black_box(db.get(id).expect("get failed"))
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// c) Batch insert vs individual inserts
// ---------------------------------------------------------------------------

fn bench_batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_individual");
    let n = 1_000usize;

    group.bench_function("individual_1000", |b| {
        b.iter(|| {
            let (db, _dir) = open_temp_db();
            for i in 0..n {
                black_box(
                    db.insert(
                        "items",
                        json!({
                            "title": format!("Record {i}"),
                            "priority": i % 5,
                        }),
                    )
                    .expect("insert failed"),
                );
            }
        });
    });

    group.bench_function("batch_1000", |b| {
        b.iter(|| {
            let (db, _dir) = open_temp_db();
            let data: Vec<_> = (0..n)
                .map(|i| {
                    json!({
                        "title": format!("Record {i}"),
                        "priority": i % 5,
                    })
                })
                .collect();
            black_box(db.insert_batch("items", data).expect("batch insert failed"));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// d) Query filter (where_field)
// ---------------------------------------------------------------------------

fn bench_query_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_filter");
    let (db, _dir) = prepopulated_db(1_000);

    group.bench_function("where_eq_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("category", Op::Eq, json!("cat-3"))
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.bench_function("where_gt_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("priority", Op::Gt, json!(3))
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.bench_function("where_contains_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("title", Op::Contains, json!("Record 5"))
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// e) Combined query (where_field + order_by)
// ---------------------------------------------------------------------------

fn bench_combined_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_query");
    let (db, _dir) = prepopulated_db(1_000);

    group.bench_function("where_eq_order_asc_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("category", Op::Eq, json!("cat-3"))
                    .order_by("priority", SortDirection::Asc)
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.bench_function("where_gt_order_desc_limit_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("value", Op::Gt, json!(500))
                    .order_by("value", SortDirection::Desc)
                    .limit(10)
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.bench_function("multi_filter_order_1000", |b| {
        b.iter(|| {
            black_box(
                db.query()
                    .table("items")
                    .where_field("priority", Op::Gte, json!(2))
                    .where_field("value", Op::Lt, json!(3000))
                    .order_by("value", SortDirection::Asc)
                    .limit(20)
                    .exec()
                    .expect("query failed"),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_throughput,
    bench_get_latency,
    bench_batch_vs_individual,
    bench_query_filter,
    bench_combined_query,
);
criterion_main!(benches);
