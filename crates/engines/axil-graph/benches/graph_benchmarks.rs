use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::json;
use tempfile::TempDir;

use axil_core::plugin::{Direction, GraphIndex};
use axil_core::record::RecordId;
use axil_graph::GraphEngine;

fn open_graph() -> (GraphEngine, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bench.axil");
    let plugin = GraphEngine::open(&path).unwrap();
    (plugin, dir)
}

/// Create a chain: n0 ->knows-> n1 ->knows-> ... ->knows-> n_{count-1}
fn chain_graph(count: usize) -> (GraphEngine, Vec<RecordId>, TempDir) {
    let (plugin, dir) = open_graph();
    let nodes: Vec<RecordId> = (0..count).map(|_| RecordId::new()).collect();
    for i in 0..count - 1 {
        plugin
            .relate(nodes[i].clone(), "knows", nodes[i + 1].clone(), json!({}))
            .unwrap();
    }
    (plugin, nodes, dir)
}

/// Create a dense graph: each node has edges to `density` random other nodes.
fn dense_graph(node_count: usize, density: usize) -> (GraphEngine, Vec<RecordId>, TempDir) {
    let (plugin, dir) = open_graph();
    let nodes: Vec<RecordId> = (0..node_count).map(|_| RecordId::new()).collect();
    for (i, from) in nodes.iter().enumerate() {
        for j in 0..density {
            let to_idx = (i + j + 1) % node_count;
            plugin
                .relate(from.clone(), "linked", nodes[to_idx].clone(), json!({}))
                .unwrap();
        }
    }
    (plugin, nodes, dir)
}

// ---------------------------------------------------------------------------
// Edge creation throughput
// ---------------------------------------------------------------------------

fn bench_relate(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_relate");
    for count in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.iter(|| {
                let (plugin, _dir) = open_graph();
                let nodes: Vec<RecordId> = (0..n + 1).map(|_| RecordId::new()).collect();
                for i in 0..n {
                    black_box(
                        plugin
                            .relate(nodes[i].clone(), "knows", nodes[i + 1].clone(), json!({}))
                            .unwrap(),
                    );
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Neighbor lookup
// ---------------------------------------------------------------------------

fn bench_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_neighbors");
    for density in [5, 20, 50] {
        let label = format!("density_{density}");
        let (plugin, nodes, _dir) = dense_graph(1_000, density);
        group.bench_function(&label, |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let node = &nodes[idx % nodes.len()];
                idx += 1;
                black_box(
                    plugin
                        .neighbors(node.clone(), None, Direction::Out)
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Traversal depth 1-5
// ---------------------------------------------------------------------------

fn bench_traverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_traverse");
    // Dense graph for multi-hop traversal
    let (plugin, nodes, _dir) = dense_graph(500, 10);

    for depth in [1, 2, 3, 5] {
        let label = format!("depth_{depth}");
        let steps: Vec<axil_core::plugin::TraversalStep> = (0..depth)
            .map(|_| axil_core::plugin::TraversalStep {
                edge_type: "linked".to_string(),
                direction: Direction::Out,
            })
            .collect();

        group.bench_function(&label, |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let start = &nodes[idx % nodes.len()];
                idx += 1;
                black_box(plugin.traverse(start.clone(), &steps).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_relate, bench_neighbors, bench_traverse);
criterion_main!(benches);
