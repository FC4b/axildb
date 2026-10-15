//! Per-write cost against table size.
//!
//! Every insert, update and delete reads, dedups and rewrites its table's whole
//! id list (one JSON array in `table_index`), so a single write may slow down
//! as its table grows. This fills one table to each target size, then times
//! single writes into it interleaved with the same writes into a small control
//! table in the same database. Both pay redb's commit; only the big table pays
//! for a long id list, so the gap between the two is the per-table cost.
//!
//! Usage:
//!   cargo run --release -p axil-core --example write_cost -- \
//!     [--sizes 0,1000,10000,30000,85000] [--samples 200]
//!
//! Prints one JSON object on stdout; progress goes to stderr.

use std::time::Instant;

use axil_core::Axil;
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

fn measure(size: usize, samples: usize) -> Value {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let db = Axil::open(dir.path().join("cost.axil")).build().expect("open");

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
    json!({
        "table_rows": size,
        "id_list_bytes_approx": id_list_bytes,
        "fill_seconds": (fill_s * 10.0).round() / 10.0,
        "insert": { "big": summary(ins_big), "control": summary(ins_ctl) },
        "update": { "big": summary(upd_big), "control": summary(upd_ctl) },
        "delete": { "big": summary(del_big), "control": summary(del_ctl) },
    })
}

fn main() {
    let mut sizes = vec![0, 1_000, 10_000, 30_000, 85_000];
    let mut samples = 200;
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
            other => panic!("unknown argument {other}"),
        }
    }

    let results: Vec<Value> = sizes.iter().map(|&s| measure(s, samples)).collect();
    let out = json!({
        "benchmark": "write-cost-vs-table-size",
        "samples_per_kind": samples,
        "control_table_rows": CONTROL_ROWS,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out).expect("json"));
}
