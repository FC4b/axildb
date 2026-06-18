//! PersonalPageRank for graph-boosted recall (8b.10).
//!
//! Lightweight PageRank computed over the adjacency index at consolidation time.
//! Records with more/better connections score higher, boosting recall ranking.

use std::collections::HashMap;

use axil_core::RecordId;

/// Compute PageRank scores for all nodes in the graph.
///
/// Uses the standard power-iteration algorithm with configurable damping factor
/// and iteration count. Runs at consolidation/heal time, not per-query.
///
/// - `adjacency`: for each node, list of outgoing neighbor node IDs
/// - `damping`: probability of following a link (typically 0.85)
/// - `iterations`: number of power iterations (typically 20-50)
pub fn compute_pagerank(
    adjacency: &HashMap<RecordId, Vec<RecordId>>,
    damping: f32,
    iterations: usize,
) -> HashMap<RecordId, f32> {
    if adjacency.is_empty() {
        return HashMap::new();
    }

    // Collect all nodes (both sources and targets)
    let mut all_nodes: std::collections::HashSet<RecordId> = std::collections::HashSet::new();
    for (src, targets) in adjacency {
        all_nodes.insert(src.clone());
        for t in targets {
            all_nodes.insert(t.clone());
        }
    }

    let n = all_nodes.len() as f32;
    let init_score = 1.0 / n;
    let random_jump = (1.0 - damping) / n;

    let mut scores: HashMap<RecordId, f32> = all_nodes
        .iter()
        .map(|id| (id.clone(), init_score))
        .collect();

    for _ in 0..iterations {
        let mut new_scores: HashMap<RecordId, f32> = all_nodes
            .iter()
            .map(|id| (id.clone(), random_jump))
            .collect();

        for (src, targets) in adjacency {
            if targets.is_empty() {
                continue;
            }
            let contrib = scores.get(src).copied().unwrap_or(0.0) * damping / targets.len() as f32;
            for t in targets {
                *new_scores.entry(t.clone()).or_default() += contrib;
            }
        }

        scores = new_scores;
    }

    scores
}

/// Incremental PageRank update for a k-hop neighborhood around affected nodes.
///
/// More efficient than full recomputation when only a few edges changed.
/// Falls back to full computation if affected set is large.
pub fn incremental_pagerank_update(
    adjacency: &HashMap<RecordId, Vec<RecordId>>,
    current_scores: &HashMap<RecordId, f32>,
    affected_ids: &[RecordId],
    damping: f32,
    hops: usize,
) -> HashMap<RecordId, f32> {
    // If affected set is large (>20% of graph), just recompute fully
    if affected_ids.len() * 5 > adjacency.len() {
        return compute_pagerank(adjacency, damping, 20);
    }

    // Build reverse adjacency once for O(1) incoming-edge lookups
    let mut incoming: HashMap<RecordId, Vec<RecordId>> = HashMap::new();
    for (src, targets) in adjacency {
        for t in targets {
            incoming.entry(t.clone()).or_default().push(src.clone());
        }
    }

    // Collect k-hop neighborhood using both forward and reverse adjacency
    let mut neighborhood: std::collections::HashSet<RecordId> =
        affected_ids.iter().cloned().collect();
    let mut frontier: Vec<RecordId> = affected_ids.to_vec();

    for _ in 0..hops {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            if let Some(neighbors) = adjacency.get(node) {
                for n in neighbors {
                    if neighborhood.insert(n.clone()) {
                        next_frontier.push(n.clone());
                    }
                }
            }
            if let Some(sources) = incoming.get(node) {
                for s in sources {
                    if neighborhood.insert(s.clone()) {
                        next_frontier.push(s.clone());
                    }
                }
            }
        }
        frontier = next_frontier;
    }

    // Run PageRank only on the neighborhood
    let mut scores = current_scores.clone();
    let n = adjacency.len().max(1) as f32;
    let random_jump = (1.0 - damping) / n;

    for _ in 0..10 {
        for node in &neighborhood {
            let mut incoming_contrib = 0.0f32;
            if let Some(sources) = incoming.get(node) {
                for src in sources {
                    let out_degree = adjacency.get(src).map(|t| t.len()).unwrap_or(1);
                    let src_score = scores.get(src).copied().unwrap_or(0.0);
                    incoming_contrib += src_score * damping / out_degree as f32;
                }
            }
            scores.insert(node.clone(), random_jump + incoming_contrib);
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(s: &str) -> RecordId {
        RecordId::from_string(s).unwrap_or_else(|_| RecordId::new())
    }

    #[test]
    fn simple_chain() {
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone()]);
        adj.insert(b.clone(), vec![c.clone()]);

        let scores = compute_pagerank(&adj, 0.85, 20);

        // c should have highest score (sink node gets link juice)
        assert!(
            scores[&c] > scores[&a],
            "c={}, a={}",
            scores[&c],
            scores[&a]
        );
    }

    #[test]
    fn empty_graph() {
        let adj: HashMap<RecordId, Vec<RecordId>> = HashMap::new();
        let scores = compute_pagerank(&adj, 0.85, 20);
        assert!(scores.is_empty());
    }

    #[test]
    fn scores_sum_to_approximately_one() {
        let a = RecordId::new();
        let b = RecordId::new();
        let c = RecordId::new();

        let mut adj = HashMap::new();
        adj.insert(a.clone(), vec![b.clone(), c.clone()]);
        adj.insert(b.clone(), vec![c.clone()]);
        adj.insert(c.clone(), vec![a.clone()]);

        let scores = compute_pagerank(&adj, 0.85, 50);
        let sum: f32 = scores.values().sum();
        assert!((sum - 1.0).abs() < 0.05, "sum={sum}");
    }
}
