//! `axil compact` compacts the `.graph` file after purging records. The open
//! database holds the file's only writable handle, so the command has to
//! close it first; these tests drive the built binary to check it does.

#![cfg(feature = "graph")]

use std::path::Path;
use std::process::Command;

use axil_core::plugin::Direction;
use axil_core::{Axil, RecordId};
use axil_graph::AxilBuilderGraphExt;
use serde_json::{json, Value};

fn compact(db: &Path) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("--db")
        .arg(db)
        .arg("compact")
        .env_remove("AXIL_DB")
        .current_dir(db.parent().unwrap())
        .output()
        .expect("failed to run axil");
    assert!(
        output.status.success(),
        "compact failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("compact prints JSON")
}

fn open(db: &Path) -> Axil {
    Axil::open(db).with_graph_engine().unwrap().build().unwrap()
}

/// A database with a `keeps` edge from `a` to `b`, and 3000 padded edges
/// from a third record, `hub`, to `b`. Their endpoints exist, so `compact`
/// keeps them as they are; deleting `hub` removes them in one transaction,
/// leaving their pages free. Returns `a` and `b`.
fn seed(db: &Path, delete_hub: bool) -> (RecordId, RecordId) {
    let handle = open(db);
    let a = handle
        .insert("notes", json!({ "summary": "kept source" }))
        .unwrap()
        .id;
    let b = handle
        .insert("notes", json!({ "summary": "kept target" }))
        .unwrap()
        .id;
    let hub = handle
        .insert("notes", json!({ "summary": "hub" }))
        .unwrap()
        .id;
    handle.relate(&a, "keeps", &b, None).unwrap();
    let spokes = (0..3000)
        .map(|i| {
            (
                hub.clone(),
                "spoke".to_string(),
                b.clone(),
                json!({ "pad": "x".repeat(400), "n": i }),
            )
        })
        .collect();
    handle
        .graph_index_ref()
        .unwrap()
        .relate_batch(spokes)
        .unwrap();
    if delete_hub {
        handle.delete(&hub).unwrap();
    }
    (a, b)
}

fn kept_targets(db: &Path, a: &RecordId) -> Vec<RecordId> {
    open(db)
        .neighbors(a, Some("keeps"), Direction::Out)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect()
}

#[test]
fn compact_gives_back_the_pages_of_deleted_edges() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let (a, b) = seed(&db, true);

    let report = compact(&db);
    let graph = &report["graph_file"];
    assert_eq!(graph["compacted"], true, "{report}");
    let bytes = |key: &str| graph[key].as_u64().unwrap();
    assert!(
        bytes("disk_bytes_after") < bytes("disk_bytes_before"),
        "{report}"
    );
    assert!(
        bytes("size_bytes_after") <= bytes("size_bytes_before"),
        "{report}"
    );
    assert_eq!(kept_targets(&db, &a), vec![b]);
}

#[test]
fn compact_leaves_a_mostly_full_graph_file_alone() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let (a, b) = seed(&db, false);

    let report = compact(&db);
    let graph = &report["graph_file"];
    assert_eq!(graph["compacted"], false, "{report}");
    assert!(graph["skipped"].is_string(), "{report}");
    assert_eq!(kept_targets(&db, &a), vec![b]);
}
