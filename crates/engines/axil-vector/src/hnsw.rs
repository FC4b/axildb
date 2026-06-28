use std::collections::HashMap;

use hnsw_rs::prelude::{DistCosine, Hnsw};

use axil_core::RecordId;

/// Graph fan-out per layer (HNSW `M`). Large enough for good recall without
/// inflating build cost.
const MAX_NB_CONNECTION: usize = 16;
/// Number of layers in the navigable small-world hierarchy.
const MAX_LAYER: usize = 16;
/// Candidate-list width during insertion. Higher = better graph, slower build.
const EF_CONSTRUCTION: usize = 200;
/// Allocation hint for the graph's internal tables. Inserts beyond this still
/// succeed — it only sizes the initial allocation.
const ALLOC_HINT: usize = 16_384;
/// Below this live-vector count an exact scan is cheaper and — unlike the
/// OS-RNG-seeded HNSW graph — deterministic, so search bypasses the graph.
const BRUTE_FORCE_MAX: usize = 128;
/// Minimum search-time candidate-list width (HNSW `ef`). Floors recall for
/// small `top_k` queries where `top_k * 4` alone would be too narrow.
const EF_SEARCH_MIN: usize = 64;

/// Cosine similarity between two f32 slices (may have different lengths — uses min).
/// Returns raw similarity in [-1.0, 1.0] — NOT clamped, so HNSW distance
/// computation preserves full geometric information for graph construction.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Build an empty live HNSW graph over normalized-or-raw f32 vectors using
/// cosine distance.
fn new_graph() -> Hnsw<'static, f32, DistCosine> {
    let mut graph = Hnsw::<f32, DistCosine>::new(
        MAX_NB_CONNECTION,
        ALLOC_HINT,
        MAX_LAYER,
        EF_CONSTRUCTION,
        DistCosine {},
    );
    // Keep pruned candidates so small corpora still return the full `top_k`
    // the caller asked for (Navarro's pruning can otherwise drop neighbours).
    graph.set_keeping_pruned(true);
    graph
}

/// Incremental HNSW approximate nearest-neighbour index.
///
/// The `hnsw_rs` graph supports `O(log n)` insertion, so `add` links the new
/// vector into the *live* graph — there is no dirty flag and store-then-recall
/// never triggers a full rebuild. `remove` tombstones the record (drops it from
/// the live id map so search skips it) and leaves the graph node in place; the
/// node is reclaimed lazily by `rebuild_if_needed` (compaction), driven off the
/// write path by the background worker once the tombstone ratio is high enough.
///
/// `vectors` is the source of truth; the graph indexes integer slots that map
/// back to `RecordId`s via `slot_to_id` (live slots only).
pub struct HnswIndex {
    dimensions: usize,
    vectors: HashMap<RecordId, Vec<f32>>,
    /// Live navigable graph over integer slots. Always current — no dirty flag.
    graph: Hnsw<'static, f32, DistCosine>,
    /// RecordId → graph slot for the live (non-tombstoned) vectors.
    id_to_slot: HashMap<RecordId, usize>,
    /// Graph slot → RecordId for live vectors. A tombstoned slot is absent here
    /// but still resides in the graph until the next compaction.
    slot_to_id: HashMap<usize, RecordId>,
    /// Monotonic slot allocator — never reused within a graph generation so a
    /// tombstoned node can never be confused with a fresh insert.
    next_slot: usize,
    /// Graph nodes that are tombstoned (removed or superseded) but still
    /// physically present in the graph. Drives search over-fetch and the
    /// compaction ratio.
    tombstones: usize,
    /// Count of deletes since last full rebuild (for deletion ratio tracking).
    deletes_since_rebuild: usize,
    /// Vector count at last rebuild (for deletion ratio computation).
    count_at_last_rebuild: usize,
}

impl HnswIndex {
    /// Create a new empty index with the given vector dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            vectors: HashMap::new(),
            graph: new_graph(),
            id_to_slot: HashMap::new(),
            slot_to_id: HashMap::new(),
            next_slot: 0,
            tombstones: 0,
            deletes_since_rebuild: 0,
            count_at_last_rebuild: 0,
        }
    }

    /// Create an index pre-loaded with vectors (e.g. from persistence).
    ///
    /// Each loaded vector is inserted into the live graph on open. Load is
    /// already `O(n)` (every vector is read from disk), so building the graph
    /// here adds no asymptotic cost and leaves the index immediately
    /// searchable with no first-search rebuild stall.
    pub fn from_vectors(dimensions: usize, vectors: HashMap<RecordId, Vec<f32>>) -> Self {
        let mut index = Self::new(dimensions);
        for (id, vec) in vectors {
            // Vectors came from our own persistence, so dims already match and
            // values are finite; insert directly into the live graph.
            let slot = index.next_slot;
            index.next_slot += 1;
            index.graph.insert((&vec, slot));
            index.slot_to_id.insert(slot, id.clone());
            index.id_to_slot.insert(id.clone(), slot);
            index.vectors.insert(id, vec);
        }
        index.count_at_last_rebuild = index.vectors.len();
        index
    }

    /// Insert a vector for a record. Rejects NaN/Infinity values.
    ///
    /// Links the vector into the live graph in `O(log n)` — there is no dirty
    /// flag and no rebuild is scheduled. Re-adding an existing id tombstones the
    /// old graph node and inserts the new vector under a fresh slot.
    pub fn add(&mut self, id: RecordId, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }
        if vector.iter().any(|v| !v.is_finite()) {
            return Err("vector contains NaN or Infinity".into());
        }
        // Re-add of an existing id: tombstone the stale graph node so search
        // can never surface the old vector, then index the new one fresh.
        if let Some(old_slot) = self.id_to_slot.remove(&id) {
            self.slot_to_id.remove(&old_slot);
            self.tombstones += 1;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        self.graph.insert((&vector, slot));
        self.slot_to_id.insert(slot, id.clone());
        self.id_to_slot.insert(id.clone(), slot);
        self.vectors.insert(id, vector);
        Ok(())
    }

    /// Remove a vector by record ID. Returns true if it existed.
    ///
    /// Tombstones the record: it is dropped from the live id maps (so search
    /// skips it immediately) but its graph node is reclaimed lazily by the next
    /// compaction, keeping deletes off the write-latency path.
    pub fn remove(&mut self, id: &RecordId) -> bool {
        if self.vectors.remove(id).is_some() {
            if let Some(slot) = self.id_to_slot.remove(id) {
                self.slot_to_id.remove(&slot);
            }
            self.tombstones += 1;
            self.deletes_since_rebuild += 1;
            true
        } else {
            false
        }
    }

    /// Number of deletions since last rebuild.
    pub fn deletes_since_rebuild(&self) -> usize {
        self.deletes_since_rebuild
    }

    /// Vector count at last rebuild (for computing deletion ratio).
    pub fn count_at_last_rebuild(&self) -> usize {
        self.count_at_last_rebuild
    }

    /// Search with automatic rebuild if needed (`&mut self`).
    ///
    /// Convenience method for standalone use. When behind a `RwLock`,
    /// prefer `needs_rebuild()` + `rebuild_if_needed()` + `search_clean()`.
    pub fn search(&mut self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>, String> {
        self.rebuild_if_needed();
        self.search_clean(query, top_k)
    }

    /// Search the live graph (`&self`).
    ///
    /// Over-fetches `top_k + tombstones` candidates so that, after skipping any
    /// tombstoned graph nodes, at least `top_k` live results remain when the
    /// corpus holds them. Distances are cosine distances; converted back to
    /// similarity as `1 - distance`.
    pub fn search_clean(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(RecordId, f32)>, String> {
        if query.len() != self.dimensions {
            return Err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            ));
        }

        if self.vectors.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // hnsw_rs seeds level assignment from the OS RNG, so on a tiny corpus
        // the navigable graph can occasionally miss a reachable node. Below a
        // small threshold an exact scan is both cheap and deterministic, so
        // prefer it there (and as the safety net when the graph under-delivers).
        if self.vectors.len() <= BRUTE_FORCE_MAX {
            return Ok(self.brute_force(query, top_k));
        }

        // Over-fetch by the tombstone count so filtering them out still leaves
        // a full top_k. Cap at the physical graph population so we never ask
        // for more than exists.
        let physical = self.vectors.len() + self.tombstones;
        let knbn = top_k.saturating_add(self.tombstones).min(physical);
        // Candidate-list width drives recall. Keep it comfortably wider than
        // both `knbn` and a fixed floor so recall@10 stays well above target.
        let ef = knbn.saturating_mul(4).max(EF_SEARCH_MIN);

        let neighbours = self.graph.search(query, knbn, ef);

        let mut results: Vec<(RecordId, f32)> = Vec::with_capacity(top_k);
        for n in neighbours {
            // Tombstoned slots are absent from `slot_to_id` — skip them.
            if let Some(id) = self.slot_to_id.get(&n.d_id) {
                results.push((id.clone(), 1.0 - n.distance));
                if results.len() >= top_k {
                    break;
                }
            }
        }

        // Safety net: if the graph returned fewer live hits than asked for
        // while more live vectors exist, the ANN walk missed reachable nodes —
        // fall back to an exact scan so callers always get the true top_k.
        if results.len() < top_k && results.len() < self.vectors.len() {
            return Ok(self.brute_force(query, top_k));
        }

        Ok(results)
    }

    /// Exact top-k over the live vectors by cosine similarity. Used for tiny
    /// corpora and as the safety net when the ANN walk under-delivers.
    fn brute_force(&self, query: &[f32], top_k: usize) -> Vec<(RecordId, f32)> {
        let mut scored: Vec<(RecordId, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.clone(), cosine_sim(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Matryoshka search: HNSW coarse retrieval, re-rank at full dims.
    ///
    /// HNSW graph search for 4*k candidates at full dimensions.
    /// re-ranks candidates using truncated `search_dims` cosine similarity,
    /// then final re-rank at full dimensions. This leverages HNSW speed for
    /// candidate retrieval while MRL truncation provides a diversity signal.
    ///
    /// Only useful with MRL-compatible models (nomic, etc.) where first N dims
    /// are meaningful.
    pub fn search_mrl(
        &self,
        query: &[f32],
        top_k: usize,
        search_dims: usize,
    ) -> Result<Vec<(RecordId, f32)>, String> {
        if search_dims >= self.dimensions || search_dims == 0 {
            return self.search_clean(query, top_k);
        }
        if query.len() < search_dims {
            return Err(format!(
                "query vector length {} is shorter than search_dims {}",
                query.len(),
                search_dims
            ));
        }

        // HNSW graph search for coarse candidates at full dimensions.
        let coarse_k = top_k.saturating_mul(4).min(self.vectors.len());
        let candidates = self.search_clean(query, coarse_k)?;

        // Re-rank candidates using truncated dimensions as a diversity signal,
        // blending full-dim and truncated-dim similarity.
        const FULL_DIM_WEIGHT: f32 = 0.7;
        const TRUNC_DIM_WEIGHT: f32 = 0.3;
        let query_trunc = &query[..search_dims];
        let mut reranked: Vec<(RecordId, f32)> = candidates
            .iter()
            .filter_map(|(id, full_sim)| {
                self.vectors.get(id).map(|v| {
                    let trunc_sim = cosine_sim(query_trunc, &v[..search_dims.min(v.len())]);
                    let blended = FULL_DIM_WEIGHT * full_sim + TRUNC_DIM_WEIGHT * trunc_sim;
                    (id.clone(), blended)
                })
            })
            .collect();
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(top_k);

        Ok(reranked)
    }

    /// Whether the graph would benefit from compaction.
    ///
    /// The live graph is always searchable, so this never gates a search; it
    /// only reports that tombstoned nodes have accumulated and a compaction
    /// (`rebuild_if_needed`) would reclaim them. `add`/`remove` never set it.
    pub fn needs_rebuild(&self) -> bool {
        self.tombstones > 0
    }

    /// Compact the graph if tombstones have accumulated.
    ///
    /// Rebuilds the navigable graph from the live `vectors`, dropping every
    /// tombstoned node and resetting slot bookkeeping. Off the write path —
    /// invoked by the background worker, not by `add`/`remove`.
    pub fn rebuild_if_needed(&mut self) {
        if self.needs_rebuild() {
            self.rebuild();
        }
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Configured dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Reference to all stored vectors (for persistence).
    pub fn vectors(&self) -> &HashMap<RecordId, Vec<f32>> {
        &self.vectors
    }

    fn rebuild(&mut self) {
        let graph = new_graph();
        let mut id_to_slot = HashMap::with_capacity(self.vectors.len());
        let mut slot_to_id = HashMap::with_capacity(self.vectors.len());
        let mut next_slot = 0usize;

        for (id, vec) in &self.vectors {
            let slot = next_slot;
            next_slot += 1;
            graph.insert((vec, slot));
            slot_to_id.insert(slot, id.clone());
            id_to_slot.insert(id.clone(), slot);
        }

        self.graph = graph;
        self.id_to_slot = id_to_slot;
        self.slot_to_id = slot_to_id;
        self.next_slot = next_slot;
        self.tombstones = 0;
        self.deletes_since_rebuild = 0;
        self.count_at_last_rebuild = self.vectors.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal xorshift64* PRNG so the oracle is fully seeded and deterministic
    /// without pulling `rand` into the dependency graph.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            // Avoid the zero fixed-point of xorshift.
            Rng(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// Uniform f32 in [-1.0, 1.0).
        fn next_f32(&mut self) -> f32 {
            // Top 24 bits → [0,1), then map to [-1,1).
            let bits = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            bits * 2.0 - 1.0
        }
    }

    /// Generate `n` deterministic vectors with stable, lexically sortable ids.
    fn make_vectors(n: usize, dims: usize, seed: u64) -> Vec<(RecordId, Vec<f32>)> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
                (RecordId(format!("v{i:08}")), v)
            })
            .collect()
    }

    /// Exact brute-force top-k oracle: rank all candidates by `cosine_sim`,
    /// breaking ties deterministically by id so the order is reproducible.
    fn brute_force_topk(
        corpus: &[(RecordId, Vec<f32>)],
        query: &[f32],
        top_k: usize,
    ) -> Vec<RecordId> {
        let mut scored: Vec<(&RecordId, f32)> = corpus
            .iter()
            .map(|(id, v)| (id, cosine_sim(query, v)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0 .0.cmp(&b.0 .0))
        });
        scored.into_iter().take(top_k).map(|(id, _)| id.clone()).collect()
    }

    /// Fraction of brute-force top-k ids that also appear in the approximate top-k.
    fn recall_overlap(approx: &[RecordId], exact: &[RecordId]) -> f32 {
        if exact.is_empty() {
            return 1.0;
        }
        let approx_set: std::collections::HashSet<&RecordId> = approx.iter().collect();
        let hits = exact.iter().filter(|id| approx_set.contains(id)).count();
        hits as f32 / exact.len() as f32
    }

    /// Oracle scale, overridable for nightly runs via `AXIL_ORACLE_N`.
    fn oracle_n(default: usize) -> usize {
        std::env::var("AXIL_ORACLE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    /// HNSW recall@10 floor. The hnsw_rs graph at N~2k/dims=64 measures ~0.95+;
    /// pinned below first observation to absorb graph-construction variance.
    const RECALL_FLOOR_K10: f32 = 0.90;

    #[test]
    fn hnsw_recall_matches_brute_force() {
        let n = oracle_n(2000);
        let dims = 64;
        let top_k = 10;
        let queries = 50;

        let corpus = make_vectors(n, dims, 0xA11CE);
        let mut index = HnswIndex::new(dims);
        for (id, v) in &corpus {
            index.add(id.clone(), v.clone()).unwrap();
        }
        index.rebuild_if_needed();

        let mut query_rng = Rng::new(0xB0B);
        let mut total = 0.0f32;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dims).map(|_| query_rng.next_f32()).collect();
            let approx: Vec<RecordId> = index
                .search_clean(&q, top_k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let exact = brute_force_topk(&corpus, &q, top_k);
            total += recall_overlap(&approx, &exact);
        }
        let mean = total / queries as f32;
        assert!(
            mean >= RECALL_FLOOR_K10,
            "HNSW mean recall@{top_k} {mean:.4} below floor {RECALL_FLOOR_K10} (N={n})"
        );
    }

    #[test]
    fn recall_correct_after_deletes_without_rebuild() {
        let n = oracle_n(1500);
        let dims = 64;
        let top_k = 10;
        let queries = 50;

        let corpus = make_vectors(n, dims, 0xDE1E7E);
        let mut index = HnswIndex::new(dims);
        for (id, v) in &corpus {
            index.add(id.clone(), v.clone()).unwrap();
        }
        index.rebuild_if_needed();

        // Remove ~20% deterministically without a manual rebuild — `search`
        // must auto-rebuild and never surface a deleted id.
        let mut del_rng = Rng::new(0xCAFE);
        let mut removed: std::collections::HashSet<RecordId> = std::collections::HashSet::new();
        let target = n / 5;
        while removed.len() < target {
            let i = (del_rng.next_u64() as usize) % n;
            let id = corpus[i].0.clone();
            if removed.insert(id.clone()) {
                index.remove(&id);
            }
        }

        let survivors: Vec<(RecordId, Vec<f32>)> = corpus
            .iter()
            .filter(|(id, _)| !removed.contains(id))
            .cloned()
            .collect();

        let mut query_rng = Rng::new(0xF00D);
        let mut total = 0.0f32;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dims).map(|_| query_rng.next_f32()).collect();
            let approx: Vec<RecordId> = index
                .search(&q, top_k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for id in &approx {
                assert!(
                    !removed.contains(id),
                    "removed id {id:?} reappeared in search results after delete"
                );
            }
            let exact = brute_force_topk(&survivors, &q, top_k);
            total += recall_overlap(&approx, &exact);
        }
        let mean = total / queries as f32;
        assert!(
            mean >= RECALL_FLOOR_K10,
            "survivor mean recall@{top_k} {mean:.4} below floor {RECALL_FLOOR_K10} after deletes"
        );
    }

    #[test]
    fn new_empty_index() {
        let index = HnswIndex::new(3);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.dimensions(), 3);
    }

    #[test]
    fn add_and_search() {
        let mut index = HnswIndex::new(3);

        let id1 = RecordId::new();
        let id2 = RecordId::new();
        let id3 = RecordId::new();

        index.add(id1.clone(), vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), vec![0.9, 0.1, 0.0]).unwrap();
        index.add(id3.clone(), vec![0.0, 0.0, 1.0]).unwrap();

        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn remove_vector() {
        let mut index = HnswIndex::new(3);
        let id = RecordId::new();
        index.add(id.clone(), vec![1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);

        assert!(index.remove(&id));
        assert_eq!(index.len(), 0);
        assert!(!index.remove(&id));
    }

    #[test]
    fn dimension_mismatch() {
        let mut index = HnswIndex::new(3);
        let id = RecordId::new();
        assert!(index.add(id, vec![1.0, 0.0]).is_err());
        assert!(index.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn search_empty_index() {
        let mut index = HnswIndex::new(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_after_remove() {
        let mut index = HnswIndex::new(3);
        let id1 = RecordId::new();
        let id2 = RecordId::new();

        index.add(id1.clone(), vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), vec![0.0, 1.0, 0.0]).unwrap();

        index.remove(&id1);

        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id2);
    }

    #[test]
    fn from_vectors() {
        let id1 = RecordId::new();
        let id2 = RecordId::new();
        let mut vecs = HashMap::new();
        vecs.insert(id1.clone(), vec![1.0, 0.0, 0.0]);
        vecs.insert(id2.clone(), vec![0.0, 1.0, 0.0]);

        let mut index = HnswIndex::from_vectors(3, vecs);
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn rejects_nan_vector() {
        let mut index = HnswIndex::new(3);
        let id = RecordId::new();
        assert!(index.add(id, vec![1.0, f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn rejects_infinity_vector() {
        let mut index = HnswIndex::new(3);
        let id = RecordId::new();
        assert!(index.add(id, vec![1.0, f32::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn add_does_not_dirty() {
        // Incremental insert links into the live graph: no compaction is
        // scheduled and the new vector is immediately searchable with no
        // rebuild between the store and the recall.
        let mut index = HnswIndex::new(3);
        let id1 = RecordId::new();
        index.add(id1.clone(), vec![1.0, 0.0, 0.0]).unwrap();
        assert!(!index.needs_rebuild(), "first add must not require rebuild");

        let id2 = RecordId::new();
        index.add(id2.clone(), vec![0.0, 1.0, 0.0]).unwrap();
        assert!(!index.needs_rebuild(), "add must not require rebuild");

        // search_clean works without any rebuild_if_needed call.
        let results = index.search_clean(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id2);
    }

    #[test]
    fn tombstone_excluded_from_search() {
        // A removed id is dropped from search results immediately — before any
        // compaction — because its slot leaves the live id map.
        let mut index = HnswIndex::new(3);
        let id1 = RecordId::new();
        let id2 = RecordId::new();
        index.add(id1.clone(), vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), vec![0.9, 0.1, 0.0]).unwrap();

        assert!(index.remove(&id1));
        assert!(index.needs_rebuild(), "a tombstone should flag compaction");

        // Query closest to the removed id; it must not surface pre-compaction.
        let results = index.search_clean(&[1.0, 0.0, 0.0], 2).unwrap();
        assert!(
            results.iter().all(|(id, _)| id != &id1),
            "removed id reappeared in search before compaction"
        );
        assert!(results.iter().any(|(id, _)| id == &id2));
    }

    #[test]
    fn over_fetch_returns_full_topk() {
        // With many tombstones interleaved, search must still return the full
        // top_k of *live* results by over-fetching past the tombstones.
        let dims = 16;
        // Keep > BRUTE_FORCE_MAX live so the graph over-fetch path (not the
        // exact-scan fallback) is the thing under test.
        let total = 900usize;
        let mut index = HnswIndex::new(dims);

        let mut rng = Rng::new(0x5EED);
        let mut live_ids = Vec::new();
        // Interleave live and dead inserts so tombstones are scattered through
        // the graph rather than clustered at the end. Every other vector is
        // kept live; the rest are tombstoned.
        for i in 0..total {
            let v: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
            let id = RecordId(format!("v{i:04}"));
            index.add(id.clone(), v).unwrap();
            if i % 2 == 0 {
                live_ids.push(id);
            } else {
                index.remove(&id);
            }
        }
        assert!(index.len() > BRUTE_FORCE_MAX, "need graph path, not brute-force");
        assert!(index.tombstones > 0, "expected scattered tombstones");

        let top_k = 10;
        let q: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
        let results = index.search_clean(&q, top_k).unwrap();
        assert_eq!(
            results.len(),
            top_k,
            "over-fetch failed to fill top_k despite {} live vectors",
            index.len()
        );
        // Every returned id must be a live (non-tombstoned) one.
        let live_set: std::collections::HashSet<&RecordId> = live_ids.iter().collect();
        for (id, _) in &results {
            assert!(
                live_set.contains(id),
                "search returned a tombstoned id {id:?}"
            );
        }
    }

    #[test]
    fn incremental_graph_recall_matches_brute_force() {
        // Engine-layer correctness: a graph grown purely by incremental `add`s
        // (never compacted) must still find the true nearest neighbors. The
        // reference is the DETERMINISTIC brute-force top-k — comparing against a
        // second rayon-built HNSW graph is flaky under parallel test load, since
        // both graphs are approximate and independently constructed.
        let n = oracle_n(800);
        let dims = 48;
        let queries = 100;
        let corpus = make_vectors(n, dims, 0x1AC3E5);

        // Incremental: add one at a time, never compact.
        let mut incremental = HnswIndex::new(dims);
        for (id, v) in &corpus {
            incremental.add(id.clone(), v.clone()).unwrap();
        }
        // The "not just fast" half: incremental adds never dirty the index.
        assert!(!incremental.needs_rebuild());

        let mut query_rng = Rng::new(0xA11);
        let mut total = 0.0f32;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dims).map(|_| query_rng.next_f32()).collect();
            let exact = brute_force_topk(&corpus, &q, 10);
            let got: Vec<RecordId> = incremental
                .search_clean(&q, 10)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += recall_overlap(&got, &exact);
        }
        let recall = total / queries as f32;
        assert!(
            recall >= 0.90,
            "incremental graph recall@10 {recall:.3} below 0.90"
        );
    }
}
