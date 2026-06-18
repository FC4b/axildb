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

    /// Matryoshka search (8b.6): HNSW coarse retrieval, re-rank at full dims.
    ///
    /// Phase 1: HNSW graph search for 4*k candidates at full dimensions.
    /// Phase 2: re-ranks candidates using truncated `search_dims` cosine similarity,
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

        // Phase 1: HNSW graph search for coarse candidates at full dimensions.
        let coarse_k = top_k.saturating_mul(4).min(self.vectors.len());
        let candidates = self.search_clean(query, coarse_k)?;

        // Phase 2: Re-rank candidates using truncated dimensions as a diversity signal,
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
