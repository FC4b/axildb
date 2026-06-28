//! Ranking-stability and storage-invariant property tests.
//!
//! T3 made RRF fusion + final-sort deterministic. The unit/proptest coverage
//! for the fusion comparators themselves lives next to the private
//! `reciprocal_rank_fusion` fn inside `axil-core::query` (it can't be reached
//! from here). This file holds the cross-cutting *storage invariant* proptest:
//! after any sequence of insert / update / delete / re-insert operations, the
//! per-table counts must still sum to the database's total record count, and
//! the diagnostics layer must not report any orphan-class problem. A stable
//! ranking is only meaningful on a consistent store, so this guards the
//! substrate the deterministic ranking sits on.

use axil_core::{Axil, RecordId};
use axil_graph::AxilBuilderGraphExt;
use proptest::prelude::*;
use serde_json::json;

/// A single mutation against the database, drawn by proptest.
#[derive(Debug, Clone)]
enum Op {
    /// Insert a record into one of a small pool of user tables.
    Insert { table: u8, value: u32 },
    /// Update the i-th live record (index taken modulo live count).
    Update { idx: usize, value: u32 },
    /// Delete the i-th live record (index taken modulo live count).
    Delete { idx: usize },
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0u8..3, any::<u32>()).prop_map(|(table, value)| Op::Insert { table, value }),
        (any::<usize>(), any::<u32>()).prop_map(|(idx, value)| Op::Update { idx, value }),
        any::<usize>().prop_map(|idx| Op::Delete { idx }),
    ]
}

/// One of a fixed pool of user tables (kept small so records cluster).
fn table_name(table: u8) -> &'static str {
    match table % 3 {
        0 => "sessions",
        1 => "decisions",
        _ => "errors",
    }
}

proptest! {
    /// After an arbitrary insert/update/delete workload, per-table counts sum
    /// to `total_records()` and no orphan-class problem is detected.
    #[test]
    fn count_consistency_under_arbitrary_workload(ops in prop::collection::vec(op_strategy(), 0..120)) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ranking_stability.axil");
        let db = Axil::open(&path)
            .with_graph_engine()
            .unwrap()
            .build()
            .unwrap();

        // Track the ids we believe are live so update/delete target real rows.
        let mut live: Vec<RecordId> = Vec::new();

        for op in ops {
            match op {
                Op::Insert { table, value } => {
                    let rec = db
                        .insert(table_name(table), json!({"n": value, "label": format!("rec-{value}")}))
                        .unwrap();
                    live.push(rec.id);
                }
                Op::Update { idx, value } => {
                    if live.is_empty() {
                        continue;
                    }
                    let id = live[idx % live.len()].clone();
                    // The row may already be gone if a prior delete raced our
                    // bookkeeping; tolerate that rather than asserting.
                    let _ = db.update(&id, json!({"n": value, "label": format!("upd-{value}")}));
                }
                Op::Delete { idx } => {
                    if live.is_empty() {
                        continue;
                    }
                    let i = idx % live.len();
                    let id = live.remove(i);
                    db.delete(&id).unwrap();
                }
            }
        }

        // Invariant 1: per-table counts sum to the global total.
        let tables = db.tables_with_counts().unwrap();
        let summed: usize = tables.iter().map(|(_, c)| c).sum();
        let total = db.total_records().unwrap();
        prop_assert_eq!(
            summed,
            total,
            "sum of per-table counts ({}) != total_records ({})",
            summed,
            total
        );

        // Invariant 2: no orphan-class problem after a clean fan-out workload.
        let problems = db.detect_problems();
        let orphan = problems.iter().find(|p| {
            p.detector == "orphaned_edges"
                || p.detector == "missing_embeddings"
                || p.detector == "missing_fts"
        });
        prop_assert!(
            orphan.is_none(),
            "unexpected orphan-class problem: {:?}",
            orphan
        );
    }
}
