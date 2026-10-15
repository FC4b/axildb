//! Per-write cost against table size.
//!
//! A table index that keeps each table's ids as one JSON array makes every
//! insert and delete read, dedup and rewrite the whole array, so a single
//! write slows down as its table grows; the keyed index makes it a few point
//! writes. This fills one table to each target size, then times single writes
//! into it interleaved with the same writes into a small control table in the
//! same database. Both pay redb's commit; only the big table pays for its
//! size, so the gap between the two is the per-table cost.
//!
//! With `--migration`, it then hands each database back to the JSON-array
//! layout (`Storage::downgrade_table_index`) and times the first writable
//! open, which migrates it, against a second open, which has nothing to do,
//! and records the file's length in both layouts.
//!
//! Usage:
//!   cargo run --release -p axil-core --example write_cost -- \
//!     [--sizes 0,1000,10000,30000,85000] [--samples 200] [--migration]
//!
//! Prints one JSON object on stdout; progress goes to stderr.

use std::path::Path;
use std::time::Instant;

use axil_core::{Axil, Storage};
use serde_json::{json, Value};

const CONTROL_ROWS: usize = 100;
const FILL_CHUNK: usize = 5_000;
const WARMUP: usize = 10;

fn row(i: usize) -> Value {
    json!({
        "title": format!("Record {i}"),
        "priority": i % 5,
        "category": format!("cat-{}", i % 10),
    })
}

/// p50 / p95 / mean in microseconds.
fn summary(mut us: Vec<f64>) -> Value {
    us.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| us[((us.len() - 1) as f64 * q).round() as usize];
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    json!({ "p50_us": pick(0.5).round(), "p95_us": pick(0.95).round(), "mean_us": mean.round() })
}

fn time_us(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e6
}

/// File length, and on unix the disk allocated to it (redb grows a file
/// without writing the new part, which stays sparse).
fn file_bytes(path: &Path) -> Value {
    let meta = std::fs::metadata(path).expect("metadata");
    #[cfg(unix)]
    let allocated = {
        use std::os::unix::fs::MetadataExt;
        Some(meta.blocks() * 512)
    };
    #[cfg(not(unix))]
    let allocated: Option<u64> = None;
    json!({ "length": meta.len(), "allocated": allocated })
}

/// Hand the database back to the JSON-array layout, then time the open that
/// migrates it and an open after that.
fn measure_migration(path: &Path) -> Value {
    Storage::open(path)
        .expect("open")
        .downgrade_table_index()
        .expect("downgrade");
    let legacy = file_bytes(path);
    let t = Instant::now();
    drop(Storage::open(path).expect("migrating open"));
    let migrating_ms = t.elapsed().as_secs_f64() * 1e3;
    let keyed = file_bytes(path);
    let t = Instant::now();
    drop(Storage::open(path).expect("second open"));
    let second_ms = t.elapsed().as_secs_f64() * 1e3;
    json!({
        "migrating_open_ms": (migrating_ms * 10.0).round() / 10.0,
        "second_open_ms": (second_ms * 10.0).round() / 10.0,
        "file_bytes_legacy": legacy,
        "file_bytes_keyed": keyed,
    })
}

fn measure(size: usize, samples: usize, migration: bool) -> Value {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let path = dir.path().join("cost.axil");
    let db = Axil::open(&path).build().expect("open");

    let fill = Instant::now();
    let mut done = 0;
    while done < size {
        let n = FILL_CHUNK.min(size - done);
        db.insert_batch("big", (done..done + n).map(row).collect())
            .expect("fill big");
        done += n;
    }
    db.insert_batch("ctl", (0..CONTROL_ROWS).map(row).collect())
        .expect("fill control");
    let fill_s = fill.elapsed().as_secs_f64();
    let id_list_bytes: usize = db.list("big").expect("list").iter().map(|r| r.id.as_str().len() + 3).sum();

    for i in 0..WARMUP {
        db.insert("big", row(size + i)).expect("warm big");
        db.insert("ctl", row(CONTROL_ROWS + i)).expect("warm ctl");
    }

    let (mut ins_big, mut ins_ctl) = (Vec::new(), Vec::new());
    let (mut big_ids, mut ctl_ids) = (Vec::new(), Vec::new());
    for i in 0..samples {
        ins_big.push(time_us(|| big_ids.push(db.insert("big", row(i)).expect("insert big").id)));
        ins_ctl.push(time_us(|| ctl_ids.push(db.insert("ctl", row(i)).expect("insert ctl").id)));
    }
    let (mut upd_big, mut upd_ctl) = (Vec::new(), Vec::new());
    for (b, c) in big_ids.iter().zip(&ctl_ids) {
        upd_big.push(time_us(|| {
            db.update(b, row(7)).expect("update big");
        }));
        upd_ctl.push(time_us(|| {
            db.update(c, row(7)).expect("update ctl");
        }));
    }
    let (mut del_big, mut del_ctl) = (Vec::new(), Vec::new());
    for (b, c) in big_ids.iter().zip(&ctl_ids) {
        del_big.push(time_us(|| {
            db.delete(b).expect("delete big");
        }));
        del_ctl.push(time_us(|| {
            db.delete(c).expect("delete ctl");
        }));
    }

    eprintln!("size {size}: filled in {fill_s:.1}s, timed {samples} writes of each kind");
    let mut result = json!({
        "table_rows": size,
        "id_list_bytes_approx": id_list_bytes,
        "fill_seconds": (fill_s * 10.0).round() / 10.0,
        "insert": { "big": summary(ins_big), "control": summary(ins_ctl) },
        "update": { "big": summary(upd_big), "control": summary(upd_ctl) },
        "delete": { "big": summary(del_big), "control": summary(del_ctl) },
    });
    if migration {
        drop(db);
        result["migration"] = measure_migration(&path);
    }
    result
}

fn main() {
    let mut sizes = vec![0, 1_000, 10_000, 30_000, 85_000];
    let mut samples = 200;
    let mut migration = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sizes" => {
                sizes = it
                    .next()
                    .expect("--sizes needs a value")
                    .split(',')
                    .map(|s| s.trim().parse().expect("size"))
                    .collect()
            }
            "--samples" => samples = it.next().expect("--samples needs a value").parse().expect("samples"),
            "--migration" => migration = true,
            other => panic!("unknown argument {other}"),
        }
    }

    let results: Vec<Value> = sizes.iter().map(|&s| measure(s, samples, migration)).collect();
    let out = json!({
        "benchmark": "write-cost-vs-table-size",
        "samples_per_kind": samples,
        "control_table_rows": CONTROL_ROWS,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out).expect("json"));
}
