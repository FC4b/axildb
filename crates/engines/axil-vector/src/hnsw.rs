use std::collections::HashMap;

use instant_distance::{Builder, HnswMap, Search};

use axil_core::RecordId;

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

/// A point in vector space for HNSW indexing.
#[derive(Clone)]
struct VectorPoint(Vec<f32>);

impl instant_distance::Point for VectorPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Use consistent cosine similarity, then convert to distance.
        1.0 - cosine_sim(&self.0, &other.0)
    }
}

/// HNSW-based approximate nearest neighbor index.
///
/// Vectors are stored in a `HashMap` and the HNSW graph is rebuilt lazily.
/// instant-distance's `HnswMap` is immutable after construction, so any
/// mutation (add/remove) marks the graph as dirty and triggers a full
/// rebuild before the next search.
pub struct HnswIndex {
    dimensions: usize,
    vectors: HashMap<RecordId, Vec<f32>>,
    index: Option<HnswMap<VectorPoint, RecordId>>,
    dirty: bool,
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
            index: None,
            deletes_since_rebuild: 0,
            count_at_last_rebuild: 0,
            dirty: false,
        }
    }

    /// Create an index pre-loaded with vectors (e.g. from persistence).
    pub fn from_vectors(dimensions: usize, vectors: HashMap<RecordId, Vec<f32>>) -> Self {
        let count = vectors.len();
        let dirty = !vectors.is_empty();
        Self {
            dimensions,
            vectors,
            index: None,
            dirty,
            deletes_since_rebuild: 0,
            count_at_last_rebuild: count,
        }
    }

    /// Insert a vector for a record. Rejects NaN/Infinity values.
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
        self.vectors.insert(id, vector);
        self.dirty = true;
        Ok(())
    }

    /// Remove a vector by record ID. Returns true if it existed.
    pub fn remove(&mut self, id: &RecordId) -> bool {
        if self.vectors.remove(id).is_some() {
            self.dirty = true;
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

    /// Search an already-built index (`&self`).
    ///
    /// Returns an error if the index needs rebuilding. Callers behind a
    /// `RwLock` should check `needs_rebuild()` first and call
    /// `rebuild_if_needed()` under a write lock if true.
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

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let map = self
            .index
            .as_ref()
            .ok_or("index needs rebuild — call rebuild_if_needed() first")?;

        let query_point = VectorPoint(query.to_vec());
        let mut search = Search::default();

        let results: Vec<(RecordId, f32)> = map
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| {
                let similarity = 1.0 - item.distance;
                (item.value.clone(), similarity)
            })
            .collect();

        Ok(results)
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

    /// Whether the HNSW graph needs to be rebuilt before searching.
    pub fn needs_rebuild(&self) -> bool {
        self.dirty || (self.index.is_none() && !self.vectors.is_empty())
    }

    /// Rebuild the HNSW graph if data has changed since the last build.
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
        if self.vectors.is_empty() {
            self.index = None;
            self.dirty = false;
            self.deletes_since_rebuild = 0;
            self.count_at_last_rebuild = 0;
            return;
        }

        let mut points = Vec::with_capacity(self.vectors.len());
        let mut values = Vec::with_capacity(self.vectors.len());

        for (id, vec) in &self.vectors {
            points.push(VectorPoint(vec.clone()));
            values.push(id.clone());
        }

        let map = Builder::default().build(points, values);
        self.index = Some(map);
        self.dirty = false;
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

    /// HNSW recall@10 floor. instant-distance at N~2k/dims=64 measures ~0.95+;
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
}
