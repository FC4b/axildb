//! Times the exact scan against the HNSW graph, the numbers behind
//! `EXACT_SCAN_MAX`:
//!
//! ```text
//! cargo run --release -p axil-vector --example exact_scan_timing
//! ```
//!
//! Vectors are random unit vectors, the hardest case for the graph; real
//! embeddings cluster, and the graph builds several times faster on them.

use std::collections::HashMap;
use std::time::Instant;

use axil_core::RecordId;
use axil_vector::hnsw::HnswIndex;

/// Recall asks the index for 80 neighbours at its default top-k.
const K: usize = 80;

fn corpus(n: usize, dims: usize, seed: u64) -> HashMap<RecordId, Vec<f32>> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dims).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            (RecordId::new(), v.iter().map(|x| x / norm).collect())
        })
        .collect()
}

fn ms_per_query(index: &HnswIndex, query: &[f32], reps: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        index.search_clean(query, K).unwrap();
    }
    start.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

fn main() {
    for dims in [384usize, 768] {
        let query: Vec<f32> = corpus(1, dims, 99).into_values().next().unwrap();
        for n in [4_400usize, 20_000, 50_000, 100_000, 200_000] {
            let index = HnswIndex::from_vectors(dims, corpus(n, dims, n as u64))
                .with_exact_scan_max(usize::MAX);
            index.search_clean(&query, K).unwrap();
            println!(
                "dims {dims:4}  n {n:7}  exact scan  {:7.2} ms/query",
                ms_per_query(&index, &query, 20)
            );
        }
        let n = 4_400;
        let index = HnswIndex::from_vectors(dims, corpus(n, dims, n as u64)).with_exact_scan_max(0);
        let start = Instant::now();
        index.search_clean(&query, K).unwrap();
        let build = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "dims {dims:4}  n {n:7}  graph build + first search {build:8.1} ms, then {:6.2} ms/query",
            ms_per_query(&index, &query, 20)
        );
    }
}
