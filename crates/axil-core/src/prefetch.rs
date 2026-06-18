//! Predictive pre-fetch — detect recurring query patterns and pre-compute results.
//!
//! Stores a query log and detects daily/weekly recurring patterns using
//! time-of-day clustering. When a pattern is detected, results can be
//! pre-computed (materialized) and cached.

use chrono::{DateTime, Datelike, Timelike, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::record::RecordId;

/// Maximum query log entries to retain.
const MAX_QUERY_LOG: usize = 5_000;

/// Minimum pattern occurrences to consider it recurring.
const MIN_PATTERN_COUNT: usize = 3;

/// Time window in hours for clustering queries as "same time of day".
const HOUR_WINDOW: u32 = 2;

/// A logged query for pattern detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLogEntry {
    /// The query embedding.
    pub embedding: Vec<f32>,
    /// When the query was executed.
    pub timestamp: DateTime<Utc>,
    /// Number of results returned.
    pub result_count: usize,
    /// Day of week (0=Mon, 6=Sun).
    pub day_of_week: u32,
    /// Hour of day (0–23).
    pub hour_of_day: u32,
}

/// A detected recurring query pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Representative embedding for this pattern.
    pub embedding: Vec<f32>,
    /// Day of week this pattern occurs (None = daily).
    pub day_of_week: Option<u32>,
    /// Hour of day this pattern occurs.
    pub hour_of_day: u32,
    /// Number of times this pattern has been observed.
    pub occurrence_count: usize,
    /// When this pattern was last observed.
    pub last_seen: DateTime<Utc>,
}

/// Cached recall results for a predicted query.
#[derive(Debug, Clone)]
pub struct MaterializedRecall {
    /// The query pattern this cache serves.
    pub pattern_embedding: Vec<f32>,
    /// Cached result record IDs.
    pub result_ids: Vec<RecordId>,
    /// When this cache was last refreshed.
    pub refreshed_at: DateTime<Utc>,
    /// Whether this cache has been invalidated.
    pub valid: bool,
}

/// Query pattern detector and cache manager.
pub struct PrefetchEngine {
    /// Query log for pattern detection.
    query_log: RwLock<Vec<QueryLogEntry>>,
    /// Detected patterns.
    patterns: RwLock<Vec<QueryPattern>>,
    /// Materialized recall caches, keyed by stable pattern hash.
    caches: RwLock<HashMap<u64, MaterializedRecall>>,
}

impl PrefetchEngine {
    /// Create a new empty prefetch engine.
    pub fn new() -> Self {
        Self {
            query_log: RwLock::new(Vec::new()),
            patterns: RwLock::new(Vec::new()),
            caches: RwLock::new(HashMap::new()),
        }
    }

    /// Load from serialized bytes.
    pub fn from_bytes(data: &[u8]) -> crate::error::Result<Self> {
        let state: PrefetchState = serde_json::from_slice(data)
            .map_err(|e| crate::error::AxilError::Serialization(Box::new(e)))?;
        Ok(Self {
            query_log: RwLock::new(state.query_log),
            patterns: RwLock::new(state.patterns),
            caches: RwLock::new(HashMap::new()),
        })
    }

    /// Serialize to bytes for persistence.
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        let state = PrefetchState {
            query_log: self.query_log.read().clone(),
            patterns: self.patterns.read().clone(),
        };
        serde_json::to_vec(&state).map_err(|e| crate::error::AxilError::Serialization(Box::new(e)))
    }

    /// Log a query execution for pattern detection.
    pub fn log_query(&self, embedding: &[f32], result_count: usize) {
        let now = Utc::now();
        let entry = QueryLogEntry {
            embedding: embedding.to_vec(),
            timestamp: now,
            result_count,
            day_of_week: now.weekday().num_days_from_monday(),
            hour_of_day: now.hour(),
        };

        let mut log = self.query_log.write();
        log.push(entry);
        if log.len() > MAX_QUERY_LOG {
            let excess = log.len() - MAX_QUERY_LOG;
            log.drain(..excess);
        }
    }

    /// Detect recurring patterns in the query log.
    ///
    /// Groups similar queries by time-of-day and day-of-week, then
    /// identifies clusters that occur at least `MIN_PATTERN_COUNT` times.
    pub fn detect_patterns(&self) -> Vec<QueryPattern> {
        // Clone log data and release lock before expensive O(N²) clustering
        let log_snapshot: Vec<QueryLogEntry> = {
            let log = self.query_log.read();
            if log.len() < MIN_PATTERN_COUNT {
                return Vec::new();
            }
            log.clone()
        };

        // Group queries by (hour_window, similar_embedding)
        let mut clusters: Vec<QueryCluster> = Vec::new();

        for entry in log_snapshot.iter() {
            let mut found = false;
            for cluster in clusters.iter_mut() {
                let hour_match =
                    hour_distance(entry.hour_of_day, cluster.hour_of_day) <= HOUR_WINDOW;
                let sim = cosine_similarity(&entry.embedding, &cluster.centroid);
                if hour_match && sim > 0.85 {
                    cluster.entries.push(entry.clone());
                    cluster.update_centroid();
                    found = true;
                    break;
                }
            }
            if !found {
                clusters.push(QueryCluster {
                    centroid: entry.embedding.clone(),
                    hour_of_day: entry.hour_of_day,
                    entries: vec![entry.clone()],
                });
            }
        }

        // Convert qualifying clusters to patterns
        let mut patterns = Vec::new();
        for cluster in &clusters {
            if cluster.entries.len() >= MIN_PATTERN_COUNT {
                // Check if it's weekly (same day of week) or daily
                let days: Vec<u32> = cluster.entries.iter().map(|e| e.day_of_week).collect();
                let most_common_day = mode(&days);
                let day_match_ratio = days.iter().filter(|&&d| d == most_common_day).count() as f64
                    / days.len() as f64;

                let day_of_week = if day_match_ratio > 0.7 {
                    Some(most_common_day) // Weekly pattern
                } else {
                    None // Daily pattern
                };

                patterns.push(QueryPattern {
                    embedding: cluster.centroid.clone(),
                    day_of_week,
                    hour_of_day: cluster.hour_of_day,
                    occurrence_count: cluster.entries.len(),
                    last_seen: cluster
                        .entries
                        .iter()
                        .map(|e| e.timestamp)
                        .max()
                        .unwrap_or_else(Utc::now),
                });
            }
        }

        // Store detected patterns
        *self.patterns.write() = patterns.clone();
        patterns
    }

    /// Get cached results for a query if a matching pattern exists.
    pub fn get_cached(&self, embedding: &[f32]) -> Option<Vec<RecordId>> {
        let patterns = self.patterns.read();
        let caches = self.caches.read();

        for pattern in patterns.iter() {
            let sim = cosine_similarity(embedding, &pattern.embedding);
            if sim > 0.90 {
                let key = embedding_hash(&pattern.embedding);
                if let Some(cache) = caches.get(&key) {
                    if cache.valid {
                        return Some(cache.result_ids.clone());
                    }
                }
            }
        }
        None
    }

    /// Store materialized results for a pattern using a stable hash key.
    pub fn cache_results(&self, _pattern_idx: usize, embedding: &[f32], result_ids: Vec<RecordId>) {
        let key = embedding_hash(embedding);
        let mut caches = self.caches.write();
        caches.insert(
            key,
            MaterializedRecall {
                pattern_embedding: embedding.to_vec(),
                result_ids,
                refreshed_at: Utc::now(),
                valid: true,
            },
        );
    }

    /// Invalidate all caches (call when underlying records change).
    pub fn invalidate_caches(&self) {
        let mut caches = self.caches.write();
        for cache in caches.values_mut() {
            cache.valid = false;
        }
    }

    /// Get detected patterns.
    pub fn patterns(&self) -> Vec<QueryPattern> {
        self.patterns.read().clone()
    }

    /// Get the number of logged queries.
    pub fn query_log_size(&self) -> usize {
        self.query_log.read().len()
    }
}

impl Default for PrefetchEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal cluster for grouping similar queries.
struct QueryCluster {
    centroid: Vec<f32>,
    hour_of_day: u32,
    entries: Vec<QueryLogEntry>,
}

impl QueryCluster {
    fn update_centroid(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let dim = self.centroid.len();
        let mut sum = vec![0.0f32; dim];
        for entry in &self.entries {
            for (i, v) in entry.embedding.iter().enumerate() {
                if i < dim {
                    sum[i] += v;
                }
            }
        }
        let n = self.entries.len() as f32;
        self.centroid = sum.iter().map(|v| v / n).collect();

        // Update hour to mode
        let hours: Vec<u32> = self.entries.iter().map(|e| e.hour_of_day).collect();
        self.hour_of_day = mode(&hours);
    }
}

/// Serializable state for persistence.
#[derive(Serialize, Deserialize)]
struct PrefetchState {
    query_log: Vec<QueryLogEntry>,
    patterns: Vec<QueryPattern>,
}

/// Stable hash of an embedding vector for use as cache key.
/// Uses the first 16 floats (or fewer) to produce a u64 hash.
fn embedding_hash(embedding: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &f in embedding.iter().take(16) {
        let bits = f.to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Circular hour distance (handles wrap-around at 24).
fn hour_distance(a: u32, b: u32) -> u32 {
    let diff = a.abs_diff(b);
    diff.min(24 - diff)
}

/// Find the mode (most common value) in a slice.
fn mode(values: &[u32]) -> u32 {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for v in values {
        *counts.entry(*v).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

/// Cosine similarity between two vectors — delegates to shared utility.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::util::cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..8).map(|i| (seed + i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn log_query_stores_entry() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);
        engine.log_query(&emb, 5);
        assert_eq!(engine.query_log_size(), 1);
    }

    #[test]
    fn log_query_enforces_max() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);
        for _ in 0..MAX_QUERY_LOG + 100 {
            engine.log_query(&emb, 5);
        }
        assert_eq!(engine.query_log_size(), MAX_QUERY_LOG);
    }

    #[test]
    fn detect_patterns_empty_log() {
        let engine = PrefetchEngine::new();
        let patterns = engine.detect_patterns();
        assert!(patterns.is_empty());
    }

    #[test]
    fn detect_patterns_finds_recurring() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);
        // Log same query 5 times (enough for MIN_PATTERN_COUNT)
        for _ in 0..5 {
            engine.log_query(&emb, 5);
        }
        let patterns = engine.detect_patterns();
        assert!(!patterns.is_empty());
        assert!(patterns[0].occurrence_count >= MIN_PATTERN_COUNT);
    }

    #[test]
    fn cache_and_retrieve() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);

        // First detect a pattern
        for _ in 0..5 {
            engine.log_query(&emb, 5);
        }
        let patterns = engine.detect_patterns();
        assert!(!patterns.is_empty());

        // Cache results using the pattern's centroid embedding
        let ids = vec![RecordId::new(), RecordId::new()];
        engine.cache_results(0, &patterns[0].embedding, ids.clone());

        // Retrieve
        let cached = engine.get_cached(&emb);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 2);
    }

    #[test]
    fn invalidate_clears_cache() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);
        for _ in 0..5 {
            engine.log_query(&emb, 5);
        }
        let patterns = engine.detect_patterns();
        engine.cache_results(0, &patterns[0].embedding, vec![RecordId::new()]);

        engine.invalidate_caches();
        assert!(engine.get_cached(&emb).is_none());
    }

    #[test]
    fn hour_distance_wraps() {
        assert_eq!(hour_distance(23, 1), 2);
        assert_eq!(hour_distance(1, 23), 2);
        assert_eq!(hour_distance(10, 14), 4);
    }

    #[test]
    fn serialization_round_trip() {
        let engine = PrefetchEngine::new();
        let emb = make_embedding(1.0);
        engine.log_query(&emb, 5);

        let bytes = engine.to_bytes().unwrap();
        let engine2 = PrefetchEngine::from_bytes(&bytes).unwrap();
        assert_eq!(engine2.query_log_size(), 1);
    }
}
