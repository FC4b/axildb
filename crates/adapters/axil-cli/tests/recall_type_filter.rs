//! `axil recall --type` must fill `--top-k` from records ranked below the
//! unfiltered cut, not just filter the unfiltered top-k.
//!
//! Drives the real binary against a temp DB (needs the local embedding model,
//! so it only builds with the `embed` feature).
#![cfg(feature = "embed")]

use std::path::Path;
use std::process::Command;

use serde_json::Value;

fn axil(db: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("--db")
        .arg(db)
        .args(args)
        .output()
        .expect("run axil");
    assert!(
        out.status.success(),
        "axil {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn type_filter_fills_top_k_from_below_the_unfiltered_cut() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("t.axil");
    let init = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("init")
        .arg(&db)
        .output()
        .expect("run axil init");
    assert!(init.status.success(), "axil init failed");

    // Five `architecture` records that rank *below* eight `gotcha` records for
    // the query, so the unfiltered top-3 holds no architecture record at all.
    for i in 1..=5 {
        let data = format!(
            r#"{{"type":"architecture","summary":"zzqquux module layout overview part {i}"}}"#
        );
        axil(&db, &["store", "context", &data]);
    }
    for i in 1..=8 {
        let data =
            format!(r#"{{"type":"gotcha","summary":"zzqquux taxonomy pitfall number {i}"}}"#);
        axil(&db, &["store", "context", &data]);
    }

    let recall = |extra: &[&str]| -> Vec<Value> {
        let mut args = vec![
            "recall",
            "zzqquux taxonomy pitfall",
            "--top-k",
            "3",
            "--no-dedup",
            "--recall-format",
            "full",
        ];
        args.extend_from_slice(extra);
        let stdout = axil(&db, &args);
        serde_json::from_str::<Value>(&stdout)
            .unwrap_or_else(|e| panic!("recall output is not JSON ({e}): {stdout}"))
            .as_array()
            .cloned()
            .expect("recall returns a JSON array")
    };

    let unfiltered = recall(&[]);
    assert!(
        unfiltered.iter().all(|r| r["data"]["type"] == "gotcha"),
        "fixture premise: the unfiltered top-3 is all gotcha, got {unfiltered:?}"
    );

    let arch = recall(&["--type", "architecture"]);
    assert_eq!(
        arch.len(),
        3,
        "--type must over-fetch so lower-ranked matches fill top-k, got {arch:?}"
    );
    assert!(arch.iter().all(|r| r["data"]["type"] == "architecture"));
}
