use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

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
/// Maximum search-time `ef`. Recall gains flatten while latency keeps climbing,
/// so the population-adaptive widening below is capped here.
const EF_SEARCH_MAX: usize = 512;
/// Per-population-decade growth factor for the search `ef` (see [`adaptive_ef`]).
/// Tuned so recall@10 stays at/above the 0.90 oracle floor from a few thousand
/// up to tens of thousands of vectors.
const EF_POP_GROWTH: f64 = 2.5;
/// Largest element magnitude the graph takes as-is. `DistCosine` multiplies
/// element pairs in f32 before widening, so two elements at this bound (1e36)
/// stay clear of f32 overflow.
const GRAPH_MAX_ABS: f32 = 1e18;
/// Smallest largest-element magnitude the graph takes as-is. Below it the
/// pairwise products fall into the f32 subnormal range, whose coarse rounding
/// can break Cauchy-Schwarz by more than `DistCosine` tolerates.
const GRAPH_MIN_ABS: f32 = 1e-10;
/// Smallest squared norm [`cosine_sim`] trusts from its f32 accumulation. At or
/// above it the result is the plain f32 cosine; below it (or on overflow) the
/// f32 sums have lost the magnitude, so it recomputes in f64.
const F32_COSINE_MIN_SQ: f32 = 1e-6;

/// Why a vector cannot enter the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorDefect {
    /// Length differs from the index's configured dimensions.
    Dimensions { expected: usize, got: usize },
    /// Contains NaN or an infinity.
    NonFinite,
    /// Every element is zero: no direction, so cosine similarity is undefined.
    Zero,
}

impl fmt::Display for VectorDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimensions { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NonFinite => f.write_str("vector contains NaN or Infinity"),
            Self::Zero => f.write_str(
                "vector is all zeros — it has no direction, so cosine similarity is undefined",
            ),
        }
    }
}

/// The single definition of an indexable vector, shared by [`HnswIndex::add`],
/// the engine's pre-persist validation, and the load path — so nothing that
/// passes one can be rejected (or panic) in another.
///
/// Magnitude is otherwise unrestricted: cosine is scale-invariant, and both
/// search paths evaluate any finite, non-zero vector exactly (see
/// [`cosine_sim`] and [`graph_view`]).
pub(crate) fn check_vector(vector: &[f32], dimensions: usize) -> Result<(), VectorDefect> {
    if vector.len() != dimensions {
        return Err(VectorDefect::Dimensions {
            expected: dimensions,
            got: vector.len(),
        });
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(VectorDefect::NonFinite);
    }
    if vector.iter().all(|v| *v == 0.0) {
        return Err(VectorDefect::Zero);
    }
    Ok(())
}

/// The vector as the graph should see it, or `None` for a zero vector.
///
/// `DistCosine` multiplies element pairs in f32 and asserts the resulting
/// distance is `>= -2e-5`. Magnitudes past ~1e19 overflow those products to
/// infinity (inf/inf is NaN) and magnitudes small enough to make them
/// subnormal round badly enough to break Cauchy-Schwarz — either one panics
/// inside the graph. A vector outside the safe range is therefore rescaled so
/// its largest element is ±1: same direction, so the same cosine. Vectors
/// already in range, which is every real embedding, pass through untouched.
fn graph_view(v: &[f32]) -> Option<Cow<'_, [f32]>> {
    let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max_abs == 0.0 {
        None
    } else if (GRAPH_MIN_ABS..=GRAPH_MAX_ABS).contains(&max_abs) {
        Some(Cow::Borrowed(v))
    } else {
        Some(Cow::Owned(v.iter().map(|x| x / max_abs).collect()))
    }
}

/// Search-time `ef` (HNSW candidate-list width) for a `population`-node graph
/// returning `knbn` neighbours.
///
/// A FIXED `ef` loses recall as the graph grows — a constant beam can't navigate
/// enough of a larger graph — so widen `ef` with the population (log-scaled in
/// decades above 1k). This keeps recall@10 above the oracle floor from a few
/// thousand to tens of thousands of vectors at a modest, bounded latency cost;
/// small graphs are unaffected (they keep the `EF_SEARCH_MIN` base).
fn adaptive_ef(knbn: usize, population: usize) -> usize {
    let base = knbn.saturating_mul(4).max(EF_SEARCH_MIN);
    // Decades of population above 1k: 0 at <=1k, 1 at 10k, ~1.3 at 20k, 2 at 100k.
    let decades = ((population.max(1) as f64) / 1000.0).max(1.0).log10();
    let widened = (base as f64 * (1.0 + EF_POP_GROWTH * decades)).round() as usize;
    // Cap the population widening at EF_SEARCH_MAX — but never below `base`: a
    // large `knbn` (over-fetching past many tombstones) legitimately needs `base`
    // candidates even when that exceeds the cap, and `clamp(base, MAX)` would
    // otherwise panic when base > MAX.
    widened.min(base.max(EF_SEARCH_MAX))
}

/// Cosine similarity between two f32 slices (may have different lengths — uses min).
/// Returns raw similarity in [-1.0, 1.0] — NOT clamped, so HNSW distance
/// computation preserves full geometric information for graph construction.
/// `0.0` when either side is a zero vector.
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
    if dot.is_finite()
        && na.is_finite()
        && nb.is_finite()
        && na >= F32_COSINE_MIN_SQ
        && nb >= F32_COSINE_MIN_SQ
    {
        return dot / (na.sqrt() * nb.sqrt());
    }
    // Huge magnitudes overflowed the f32 sums, or tiny ones underflowed them.
    // Every f32 product is exact in f64, so recompute there.
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..len {
        let (x, y) = (f64::from(a[i]), f64::from(b[i]));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())) as f32
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

/// The navigable graph plus the slot bookkeeping that maps it back to records.
///
/// Slots are integers because `hnsw_rs` ids are `usize`; `slot_to_id` holds
/// live slots only, so a tombstoned node still in the graph maps to nothing.
struct LiveGraph {
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// RecordId → graph slot for the live (non-tombstoned) vectors.
    id_to_slot: HashMap<RecordId, usize>,
    /// Graph slot → RecordId for live vectors. A tombstoned slot is absent here
    /// but still resides in the graph until the next compaction.
    slot_to_id: HashMap<usize, RecordId>,
    /// Monotonic slot allocator — never reused within a graph generation so a
    /// tombstoned node can never be confused with a fresh insert.
    next_slot: usize,
    /// Nodes physically in this graph that are tombstoned (removed or
    /// superseded). Drives the search over-fetch.
    dead: usize,
}

impl LiveGraph {
    /// Build a graph over exactly the given live vectors.
    fn build(vectors: &HashMap<RecordId, Vec<f32>>) -> Self {
        let mut graph = Self {
            hnsw: new_graph(),
            id_to_slot: HashMap::with_capacity(vectors.len()),
            slot_to_id: HashMap::with_capacity(vectors.len()),
            next_slot: 0,
            dead: 0,
        };
        for (id, vec) in vectors {
            graph.insert(id, vec);
        }
        graph
    }

    /// Link a vector into the graph under a fresh slot, tombstoning the node an
    /// earlier vector for the same id occupied.
    fn insert(&mut self, id: &RecordId, vector: &[f32]) {
        self.remove(id);
        // Every indexed vector passed `check_vector`, so it is non-zero.
        let Some(view) = graph_view(vector) else {
            return;
        };
        let slot = self.next_slot;
        self.next_slot += 1;
        self.hnsw.insert((&view, slot));
        self.slot_to_id.insert(slot, id.clone());
        self.id_to_slot.insert(id.clone(), slot);
    }

    /// Tombstone the id's node: search skips it from now on, and the next
    /// compaction drops it from the graph.
    fn remove(&mut self, id: &RecordId) {
        if let Some(slot) = self.id_to_slot.remove(id) {
            self.slot_to_id.remove(&slot);
            self.dead += 1;
        }
    }
}

/// Incremental HNSW approximate nearest-neighbour index.
///
/// The graph is built lazily — once, on the first search that needs it (above
/// the exact-scan threshold) — not when the index is loaded: most processes
/// that open a store only write to it, and a full graph build costs far more
/// than reading the vectors. Once built, the `hnsw_rs` graph takes `O(log n)`
/// insertions, so `add` links the new vector into the *live* graph and
/// store-then-recall never triggers a full rebuild. `remove` tombstones the
/// record (drops it from the live id map so search skips it) and leaves the
/// graph node in place; the node is reclaimed lazily by `rebuild_if_needed`
/// (compaction), driven off the write path by the background worker once the
/// tombstone ratio is high enough.
///
/// `vectors` is the source of truth; the graph indexes integer slots that map
/// back to `RecordId`s.
pub struct HnswIndex {
    dimensions: usize,
    vectors: HashMap<RecordId, Vec<f32>>,
    /// Navigable graph over the live vectors, absent until a search needs it.
    /// Once built it is kept current by every `add`/`remove`.
    graph: OnceLock<LiveGraph>,
    /// Removes and re-adds of an indexed id since the last compaction — the
    /// reclaimable work the compactor gates on. Counted whether or not the
    /// graph is built, so the compaction ratio does not depend on whether this
    /// process happened to search.
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
            graph: OnceLock::new(),
            tombstones: 0,
            deletes_since_rebuild: 0,
            count_at_last_rebuild: 0,
        }
    }

    /// Create an index pre-loaded with vectors (e.g. from persistence).
    ///
    /// Cheap: the vectors are only taken over. The graph is built on the first
    /// search that needs it, so a caller that never searches never pays for
    /// it. The vectors must already satisfy the index's validity rules (the
    /// engine's load path filters with the same check `add` applies).
    pub fn from_vectors(dimensions: usize, vectors: HashMap<RecordId, Vec<f32>>) -> Self {
        let mut index = Self::new(dimensions);
        index.count_at_last_rebuild = vectors.len();
        index.vectors = vectors;
        index
    }

    /// Insert a vector for a record. Rejects wrong dimensions, NaN/Infinity
    /// values, and the all-zero vector.
    ///
    /// Links the vector into the live graph in `O(log n)` when the graph is
    /// built (otherwise the eventual build picks it up) — there is no dirty
    /// flag and no rebuild is scheduled. Re-adding an existing id tombstones the
    /// old graph node and inserts the new vector under a fresh slot.
    pub fn add(&mut self, id: RecordId, vector: Vec<f32>) -> Result<(), String> {
        check_vector(&vector, self.dimensions).map_err(|d| d.to_string())?;
        // Re-add of an existing id: its old graph node is now stale, so search
        // must never surface it; count it toward compaction either way.
        if self.vectors.contains_key(&id) {
            self.tombstones += 1;
        }
        if let Some(graph) = self.graph.get_mut() {
            graph.insert(&id, &vector);
        }
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
            if let Some(graph) = self.graph.get_mut() {
                graph.remove(id);
            }
            self.tombstones += 1;
            self.deletes_since_rebuild += 1;
            true
        } else {
            false
        }
    }

    /// Whether the navigable graph has been built. It is built on the first
    /// search above the exact-scan threshold, never by loading, inserting or
    /// deleting.
    pub fn is_graph_built(&self) -> bool {
        self.graph.get().is_some()
    }

    /// The live graph, building it from the current vectors on first use.
    /// Thread-safe and one-shot: concurrent first searches wait for a single
    /// build instead of racing their own.
    fn graph(&self) -> &LiveGraph {
        self.graph.get_or_init(|| LiveGraph::build(&self.vectors))
    }

    /// Number of deletions since last rebuild.
    pub fn deletes_since_rebuild(&self) -> usize {
        self.deletes_since_rebuild
    }

    /// Total tombstones since the last rebuild — counting both removed ids and
    /// the stale graph nodes left by re-adding an existing id.
    ///
    /// This, not `deletes_since_rebuild`, is what the compactor must consult: a
    /// re-add (every update / re-embed) bumps `tombstones` but NOT
    /// `deletes_since_rebuild`, so gating compaction on the latter would let an
    /// update-heavy workload grow the graph unbounded while the search
    /// over-fetch (`top_k + tombstones`) keeps widening.
    pub fn tombstones(&self) -> usize {
        self.tombstones
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

    /// Search the live graph (`&self`), building it first if this is the
    /// first search that needs it.
    ///
    /// Over-fetches `top_k + tombstones` candidates so that, after skipping any
    /// tombstoned graph nodes, at least `top_k` live results remain when the
    /// corpus holds them. Distances are cosine distances; converted back to
    /// similarity as `1 - distance`. A zero query has no direction and scores
    /// `0.0` against every vector on either path; a non-finite one is rejected.
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
        if query.iter().any(|v| !v.is_finite()) {
            return Err("query vector contains NaN or Infinity".into());
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
        // `DistCosine` calls a zero vector distance 0 (similarity 1.0) from
        // everything; the exact scan scores it 0.0, consistently.
        let Some(query_view) = graph_view(query) else {
            return Ok(self.brute_force(query, top_k));
        };

        let graph = self.graph();
        // Over-fetch by the tombstone count so filtering them out still leaves
        // a full top_k. Cap at the physical graph population so we never ask
        // for more than exists.
        let physical = self.vectors.len() + graph.dead;
        let knbn = top_k.saturating_add(graph.dead).min(physical);
        // Candidate-list width drives recall, and a fixed `ef` decays as the
        // graph grows — so widen it with the live population (see `adaptive_ef`).
        let ef = adaptive_ef(knbn, physical);

        let neighbours = graph.hnsw.search(&query_view, knbn, ef);

        let mut results: Vec<(RecordId, f32)> = Vec::with_capacity(top_k);
        for n in neighbours {
            // Tombstoned slots are absent from `slot_to_id` — skip them.
            if let Some(id) = graph.slot_to_id.get(&n.d_id) {
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
        // Tie-break equal similarities by RecordId so results are byte-stable
        // across runs (self.vectors is a HashMap with nondeterministic order) —
        // the same determinism guarantee the fusion path carries.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
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
    /// Rebuilds a built navigable graph from the live `vectors`, dropping every
    /// tombstoned node and resetting slot bookkeeping; an unbuilt graph has no
    /// nodes to reclaim, so only the counters reset. Off the write path —
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
        // A graph in use is rebuilt now, keeping the cost off the next search;
        // one never built stays unbuilt (the eventual build starts clean).
        if self.graph.get().is_some() {
            self.graph = OnceLock::from(LiveGraph::build(&self.vectors));
        }
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
    fn adaptive_ef_widens_with_population_without_panicking() {
        let base = (10usize * 4).max(EF_SEARCH_MIN); // knbn=10 -> 64
        assert_eq!(adaptive_ef(10, 500), base, "<=1k graph keeps the base width");
        let at10k = adaptive_ef(10, 10_000);
        let at20k = adaptive_ef(10, 20_000);
        assert!(at10k > base && at20k >= at10k, "ef widens monotonically with N");
        assert!(at20k <= EF_SEARCH_MAX, "population widening is capped at EF_SEARCH_MAX");
        // A large knbn (over-fetch past many tombstones) makes `base` exceed the
        // cap; ef must equal that base, not panic on clamp(base, MAX).
        assert_eq!(
            adaptive_ef(460, 900),
            (460 * 4).max(EF_SEARCH_MIN),
            "base over-fetch width is honored even above the cap"
        );
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
        // Build the (empty) graph up front so every insert and tombstone lands
        // in it; a lazily built graph would hold only the survivors.
        index.graph();

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
        assert!(index.graph().dead > 0, "expected scattered tombstones");

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
    fn brute_force_breaks_ties_by_id() {
        // Two records share an identical vector → identical similarity to any
        // query, so only the RecordId tie-break decides their order. Inserted
        // in descending id order to defeat any incidental insertion ordering.
        let mut index = HnswIndex::new(4);
        index
            .add(RecordId("id-c".to_string()), vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .add(RecordId("id-b".to_string()), vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .add(RecordId("id-a".to_string()), vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();
        let res = index.brute_force(&[1.0, 0.0, 0.0, 0.0], 2);
        let ids: Vec<&str> = res.iter().map(|(id, _)| id.0.as_str()).collect();
        assert_eq!(
            ids,
            vec!["id-b", "id-c"],
            "tied similarities must order by ascending RecordId"
        );
    }

    #[test]
    fn extreme_magnitudes_search_by_direction_without_panicking() {
        // The graph's `DistCosine` multiplies element pairs in f32: at 1e20 the
        // products overflow (inf/inf = NaN), at 1e-25 they go subnormal and
        // break Cauchy-Schwarz — both used to panic on its `dist >= -2e-5`
        // assertion. Cosine is scale-invariant, so every vector must still rank
        // by direction, on the graph path and the exact path alike.
        let dims = 8;
        let mut corpus = make_vectors(300, dims, 0xE77E);
        for (i, (_, v)) in corpus.iter_mut().enumerate() {
            let scale = match i % 4 {
                0 => 1e20_f32,
                1 => 1e-25,
                2 => 1e30,
                _ => 1.0,
            };
            for x in v.iter_mut() {
                *x *= scale;
            }
        }
        let mut index = HnswIndex::new(dims);
        for (id, v) in &corpus {
            index.add(id.clone(), v.clone()).unwrap();
        }
        assert!(index.len() > BRUTE_FORCE_MAX, "need the graph path");

        let mut total = 0.0f32;
        for (id, v) in corpus.iter().step_by(7) {
            let hits = index.search_clean(v, 10).unwrap();
            assert_eq!(&hits[0].0, id, "a stored vector must be its own top hit");
            assert!(
                (hits[0].1 - 1.0).abs() < 1e-4,
                "self-similarity must be 1.0 at any magnitude, got {}",
                hits[0].1
            );
            let got: Vec<RecordId> = hits.into_iter().map(|(id, _)| id).collect();
            total += recall_overlap(&got, &brute_force_topk(&corpus, v, 10));
            // The exact path scores the same vectors identically.
            let exact = index.brute_force(v, 1);
            assert_eq!(&exact[0].0, id);
            assert!((exact[0].1 - 1.0).abs() < 1e-4);
        }
        let queries = corpus.iter().step_by(7).count() as f32;
        assert!(total / queries >= RECALL_FLOOR_K10);
    }

    #[test]
    fn zero_vector_is_rejected() {
        // A zero vector has no direction: the graph would call it distance 0
        // (similarity 1.0) to everything while the exact scan scores it 0.0.
        let mut index = HnswIndex::new(3);
        let err = index.add(RecordId::new(), vec![0.0, -0.0, 0.0]).unwrap_err();
        assert!(err.contains("zero"), "unexpected error: {err}");
        assert!(index.is_empty());
    }

    #[test]
    fn zero_query_scores_zero_on_graph_path_like_exact_path() {
        let dims = 8;
        let mut index = HnswIndex::new(dims);
        for (id, v) in make_vectors(200, dims, 0x2E80) {
            index.add(id, v).unwrap();
        }
        assert!(index.len() > BRUTE_FORCE_MAX, "need the graph path");
        let hits = index.search_clean(&[0.0; 8], 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(
            hits.iter().all(|(_, s)| *s == 0.0),
            "a direction-less query must not score 1.0 against anything: {hits:?}"
        );
    }

    #[test]
    fn non_finite_query_is_rejected() {
        let mut index = HnswIndex::new(3);
        index.add(RecordId::new(), vec![1.0, 0.0, 0.0]).unwrap();
        assert!(index.search_clean(&[f32::NAN, 0.0, 0.0], 1).is_err());
        assert!(index.search_clean(&[f32::INFINITY, 0.0, 0.0], 1).is_err());
    }

    #[test]
    fn graph_is_built_only_by_a_search_that_needs_it() {
        let dims = 16;
        let corpus = make_vectors(300, dims, 0x1A2B);
        let mut index = HnswIndex::from_vectors(dims, corpus.iter().cloned().collect());
        assert!(!index.is_graph_built(), "loading must not build the graph");

        index
            .add(RecordId("extra".into()), corpus[0].1.clone())
            .unwrap();
        assert!(index.remove(&corpus[1].0));
        index.add(corpus[2].0.clone(), corpus[3].1.clone()).unwrap();
        assert!(!index.is_graph_built(), "insert/delete must not build the graph");
        // Compaction accounting does not depend on whether a graph exists.
        assert_eq!(index.tombstones(), 2, "one delete + one re-add");
        index.rebuild_if_needed();
        assert!(!index.is_graph_built(), "compaction must not build the graph");
        assert_eq!(index.tombstones(), 0);

        index.search_clean(&corpus[5].1, 5).unwrap();
        assert!(index.is_graph_built(), "a search above the exact-scan threshold builds it");

        // A small corpus is served by the exact scan and never needs a graph.
        let small = HnswIndex::from_vectors(
            dims,
            make_vectors(BRUTE_FORCE_MAX, dims, 0x5A11).into_iter().collect(),
        );
        small.search_clean(&corpus[5].1, 5).unwrap();
        assert!(!small.is_graph_built());
    }

    #[test]
    fn lazy_graph_returns_the_same_results_as_an_eager_one() {
        // The same mutation sequence against an index whose graph exists from
        // the start (inserts and tombstones land in it, the pre-lazy behavior)
        // and one that only builds its graph at the first search.
        fn mutate(index: &mut HnswIndex, corpus: &[(RecordId, Vec<f32>)]) {
            for (id, _) in corpus.iter().step_by(5) {
                assert!(index.remove(id));
            }
            for (i, (id, _)) in corpus.iter().enumerate().skip(1).step_by(9) {
                index.add(id.clone(), corpus[i - 1].1.clone()).unwrap();
            }
            let extra = make_vectors(20, corpus[0].1.len(), 0xE4);
            for (i, (_, v)) in extra.into_iter().enumerate() {
                index.add(RecordId(format!("x{i:04}")), v).unwrap();
            }
        }
        let dims = 24;
        for n in [100usize, 600] {
            let corpus = make_vectors(n, dims, 0x5EED + n as u64);
            let mut eager = HnswIndex::new(dims);
            eager.graph();
            for (id, v) in &corpus {
                eager.add(id.clone(), v.clone()).unwrap();
            }
            mutate(&mut eager, &corpus);
            let mut lazy = HnswIndex::from_vectors(dims, corpus.iter().cloned().collect());
            mutate(&mut lazy, &corpus);
            assert_eq!(eager.vectors(), lazy.vectors());
            assert_eq!(eager.tombstones(), lazy.tombstones());

            let live: Vec<(RecordId, Vec<f32>)> = lazy
                .vectors()
                .iter()
                .map(|(id, v)| (id.clone(), v.clone()))
                .collect();
            let ids = |r: &[(RecordId, f32)]| -> Vec<RecordId> {
                r.iter().map(|(id, _)| id.clone()).collect()
            };
            let mut query_rng = Rng::new(0xFACE);
            let (mut recall_eager, mut recall_lazy) = (0.0f32, 0.0f32);
            let queries = 20;
            for q in 0..queries {
                // Half random queries, half exact copies of a stored vector.
                let copy = q % 2 == 1;
                let query: Vec<f32> = if copy {
                    live[q * 7 % live.len()].1.clone()
                } else {
                    (0..dims).map(|_| query_rng.next_f32()).collect()
                };
                let a = eager.search_clean(&query, 10).unwrap();
                let b = lazy.search_clean(&query, 10).unwrap();
                if n <= BRUTE_FORCE_MAX {
                    // Exact path: byte-identical, scores included.
                    assert_eq!(a, b);
                    continue;
                }
                // Graph path: two independently seeded ANN graphs, so hold
                // each to the exact answer rather than to each other.
                let exact = brute_force_topk(&live, &query, 10);
                recall_eager += recall_overlap(&ids(&a), &exact);
                recall_lazy += recall_overlap(&ids(&b), &exact);
                if copy {
                    assert!((a[0].1 - 1.0).abs() < 1e-5 && (b[0].1 - 1.0).abs() < 1e-5);
                }
            }
            if n > BRUTE_FORCE_MAX {
                assert!(lazy.is_graph_built());
                assert!(recall_eager / queries as f32 >= RECALL_FLOOR_K10);
                assert!(recall_lazy / queries as f32 >= RECALL_FLOOR_K10);
            } else {
                assert!(!lazy.is_graph_built());
            }
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

        // Incremental: add one at a time into a live graph, never compact.
        let mut incremental = HnswIndex::new(dims);
        incremental.graph();
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
