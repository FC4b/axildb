//! `axil recall --rerank` must hand the reranker a pool wider than
//! `--top-k`, then print only `--top-k`; without `--rerank` nothing changes.
//!
//! Drives the real binary against a temp DB (needs the local embedding model,
//! so it only builds with the `embed` feature). `--rerank llm` with no LLM
//! configured stands in for a reranker: it keeps recall's order, so the test
//! needs no reranker model.
#![cfg(feature = "embed")]

use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;

fn run(db: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("--db")
        .arg(db)
        .args(args)
        .output()
        .expect("run axil")
}

fn axil(db: &Path, args: &[&str]) -> Output {
    let out = run(db, args);
    assert!(
        out.status.success(),
        "axil {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn hits(out: &Output) -> Vec<Value> {
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str::<Value>(&stdout)
        .unwrap_or_else(|e| panic!("recall output is not JSON ({e}): {stdout}"))
        .as_array()
        .cloned()
        .expect("recall returns a JSON array")
}

fn init_db(dir: &Path) -> std::path::PathBuf {
    let db = dir.join("t.axil");
    let init = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("init")
        .arg(&db)
        .output()
        .expect("run axil init");
    assert!(init.status.success(), "axil init failed");
    db
}

#[test]
fn rerank_reorders_a_wider_pool_and_prints_top_k() {
    let dir = tempfile::tempdir().unwrap();
    let db = init_db(dir.path());

    let topics = [
        "connection pool",
        "retry budget",
        "cache eviction",
        "index rebuild",
        "schema migration",
        "lock timeout",
        "batch insert",
        "query planner",
        "vector store",
        "token budget",
        "hook latency",
        "graph edges",
        "fts commit",
        "snapshot copy",
        "model download",
        "daemon socket",
        "entity regex",
        "dedup window",
        "decay half-life",
        "boot payload",
        "scope filter",
        "checkpoint replay",
        "branch publish",
        "slow query log",
        "recall cascade",
    ];
    for (i, topic) in topics.iter().enumerate() {
        let data = format!(r#"{{"summary":"zzqquux note {i}: the {topic} setting changed"}}"#);
        axil(&db, &["store", "context", &data]);
    }

    let recall = |extra: &[&str]| -> Output {
        let mut args = vec![
            "recall",
            "zzqquux setting changed",
            "--top-k",
            "3",
            "--no-widen",
            "--no-dedup",
            "--recall-format",
            "full",
            "--profile",
        ];
        args.extend_from_slice(extra);
        axil(&db, &args)
    };

    let plain = recall(&[]);
    assert_eq!(hits(&plain).len(), 3);
    assert!(
        !String::from_utf8_lossy(&plain.stderr).contains("rerank pool"),
        "no pool without --rerank"
    );
    // `--rerank off` is the same path as no flag.
    assert_eq!(recall(&["--rerank", "off"]).stdout, plain.stdout);

    let reranked = recall(&["--rerank", "llm"]);
    let stderr = String::from_utf8_lossy(&reranked.stderr);
    assert!(
        stderr.contains("rerank pool=20 "),
        "--top-k 3 reranks a pool of max(4 × 3, 20) = 20, got: {stderr}"
    );
    assert_eq!(
        hits(&reranked).len(),
        3,
        "the pool is cut back to --top-k after the rerank"
    );
}

#[test]
fn rerank_weight_outside_zero_to_one_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let db = init_db(dir.path());

    for bad in ["1.5", "-0.1", "NaN"] {
        let flag = format!("--rerank-weight={bad}");
        let out = run(&db, &["recall", "anything", &flag]);
        assert!(!out.status.success(), "--rerank-weight {bad} must fail");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("--rerank-weight"),
            "error names the flag for {bad}"
        );
    }
}
