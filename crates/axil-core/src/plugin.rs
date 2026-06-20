//! Engine (Tier-1) traits — the internal storage SPI.
//!
//! # Stability
//!
//! **This is an internal Engine API with NO semver guarantee — it may change
//! in any release.** The `Engine`, [`VectorIndex`], [`GraphIndex`],
//! [`SearchIndex`], [`TimeSeriesIndex`], and [`TextEmbedder`] traits, and the
//! [`Capability`] enum, are the substrate the master coordinator drives
//! directly; keeping them unstable is what gives Axil freedom to add, drop, or
//! swap storage Engines without a breaking change. Third parties extend Axil
//! through the *stable* [`crate::extension::Extension`] / [`crate::adapter::Adapter`]
//! SPI (Tier 2 / Tier 3), not these traits. The supported posture for a custom
//! Engine is upstream-or-fork.

use crate::error::Result;
use crate::record::{Record, RecordId};
use serde_json::Value;

/// Describes the capabilities a plugin provides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capability {
    /// HNSW / ANN vector similarity search.
    VectorSearch,
    /// Graph edge traversal.
    GraphTraversal,
    /// Full-text search (tantivy / BM25).
    FullTextSearch,
    /// Time-series range queries.
    TimeSeries,
}

/// Base trait every plugin must implement.
pub trait Engine: Send + Sync {
    /// Human-readable plugin name.
    fn name(&self) -> &str;

    /// Capabilities this plugin provides.
    fn capabilities(&self) -> Vec<Capability>;

    /// Called after a record is inserted.
    fn on_record_insert(&self, record: &Record) -> Result<()>;

    /// Called after a record is updated.
    ///
    /// Default is a no-op. Plugins that track mutable state (e.g. time-series
    /// `updated_at` index) should override this.
    ///
    /// **Note:** Vector re-embedding is NOT automatic on update. Callers must
    /// call `embed_field()` explicitly after updating records with embedded
    /// fields (this matches Path A in the design where the agent orchestrates
    /// the embed pipeline).
    fn on_record_update(&self, _record: &Record) -> Result<()> {
        Ok(())
    }

    /// Called after a record is deleted.
    fn on_record_delete(&self, id: &RecordId) -> Result<()>;
}

/// Direction for graph traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Outgoing edges.
    Out,
    /// Incoming edges.
    In,
    /// Both directions.
    Both,
}

/// A step in a graph traversal path.
#[derive(Debug, Clone)]
pub struct TraversalStep {
    /// Edge type to follow.
    pub edge_type: String,
    /// Direction to traverse.
    pub direction: Direction,
}

/// Engine that provides vector similarity search (pure ANN operations).
///
/// Handles adding vectors and searching by similarity. Text-to-vector
/// conversion is handled separately by [`TextEmbedder`].
pub trait VectorIndex: Engine {
    /// Add a vector for a record.
    fn add(&self, id: RecordId, vector: &[f32]) -> Result<()>;

    /// Search for the top-k most similar vectors.
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>>;

    /// Number of vectors currently indexed.
    fn count(&self) -> usize;

    /// Configured vector dimensions.
    fn dimensions(&self) -> usize;

    /// Number of deleted vectors still in the index (tombstones).
    /// Returns 0 if not tracked.
    fn deleted_count(&self) -> usize {
        0
    }

    /// All vector IDs currently in the index.
    fn all_ids(&self) -> Result<Vec<RecordId>> {
        Ok(Vec::new())
    }

    /// Look up a stored vector by record id.
    ///
    /// Enables consumers (e.g. `db.recall()` with QTC enabled) to retrieve
    /// pre-computed chunk embeddings instead of re-running the embedder at
    /// query time. Default `None` so backends that only support approximate
    /// search (no direct retrieval) keep compiling without behavioural change.
    fn get_vector(&self, _id: &RecordId) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Rebuild the vector index from scratch (compact tombstones).
    /// Returns the number of vectors in the rebuilt index.
    fn rebuild(&self) -> Result<usize> {
        Ok(self.count())
    }
}

/// Converts text into embedding vectors.
///
/// Separated from [`VectorIndex`] so that ANN-only plugins don't need
/// to stub out embedding, and so embedding can be configured independently.
pub trait TextEmbedder: Send + Sync {
    /// Embed text into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in a single batch (8b.3).
    ///
    /// Default implementation falls back to sequential embedding.
    /// Implementations backed by ONNX can override for 5-10x speedup
    /// by batching tokenization and running a single inference call.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Lightweight edge descriptor returned by `GraphIndex::edges()`.
/// Contains enough information to display and target edges without
/// depending on the concrete `Edge` type from `axil-graph`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeInfo {
    /// Edge ID.
    pub id: RecordId,
    /// Source record ID.
    pub from: RecordId,
    /// Target record ID.
    pub to: RecordId,
    /// Edge type label.
    pub edge_type: String,
    /// Edge properties.
    pub properties: Value,
    /// When the edge was created (ISO 8601 string to avoid chrono dep in core).
    pub created_at: String,
}

/// Engine that provides graph traversal.
pub trait GraphIndex: Engine {
    /// Create a directed edge between two records.
    fn relate(
        &self,
        from: RecordId,
        edge_type: &str,
        to: RecordId,
        props: Value,
    ) -> Result<RecordId>;

    /// Create many directed edges in a single transaction.
    ///
    /// The default implementation calls `relate` once per edge and is
    /// purely a convenience — implementations that persist edges in
    /// their own transaction (`GraphEngine`) override this to amortize
    /// transaction overhead across the whole batch. SCIP ingest writes
    /// hundreds of thousands of edges per workspace; per-edge txns
    /// turned a 10 MB index into a 15-minute redb-lock-holding job.
    fn relate_batch(
        &self,
        edges: Vec<(RecordId, String, RecordId, Value)>,
    ) -> Result<Vec<RecordId>> {
        let mut ids = Vec::with_capacity(edges.len());
        for (from, edge_type, to, props) in edges {
            ids.push(self.relate(from, &edge_type, to, props)?);
        }
        Ok(ids)
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    fn unrelate(&self, edge_id: &RecordId) -> Result<bool>;

    /// Traverse a path starting from a record, returning the IDs reached.
    fn traverse(&self, start: RecordId, path: &[TraversalStep]) -> Result<Vec<RecordId>>;

    /// Get IDs of immediate neighbors of a record.
    fn neighbors(
        &self,
        id: RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<RecordId>>;

    /// List edges attached to a record in the given direction.
    fn edges(
        &self,
        id: RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<EdgeInfo>>;

    /// Total number of edges in the graph index.
    fn edge_count(&self) -> usize {
        0
    }

    /// List all edge IDs in the graph index (for diagnostic checks).
    fn all_edge_ids(&self) -> Result<Vec<(RecordId, RecordId, RecordId)>> {
        Ok(Vec::new())
    }
}

/// Maximum traversal depth to prevent infinite loops.
const MAX_DEPTH: usize = 50;

/// Maximum byte length of a traversal path expression.
const MAX_PATH_BYTES: usize = 4096;

/// Maximum byte length of a single edge type label.
const MAX_EDGE_TYPE_LEN: usize = 256;

/// Find the start of the next arrow token (`->`, `<-`, or `<->`) in a string,
/// returning the byte offset. Correctly handles `<->` as a single token so that
/// `find("->")` doesn't match the `->` inside `<->`.
fn find_next_arrow(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // `<->` or `<-`
            if i + 1 < bytes.len() && bytes[i + 1] == b'-' {
                return i;
            }
        } else if bytes[i] == b'-' {
            // `->`
            if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                return i;
            }
        }
        i += 1;
    }
    s.len()
}

/// Parse a path expression like `->modified->file` into traversal steps.
///
/// Syntax:
/// - `->edge_type` — outgoing edge
/// - `<-edge_type` — incoming edge
/// - `<->edge_type` — both directions
pub fn parse_path(path: &str) -> Result<Vec<TraversalStep>> {
    if path.is_empty() {
        return Err(crate::error::AxilError::InvalidQuery(
            "empty traversal path".into(),
        ));
    }
    if path.len() > MAX_PATH_BYTES {
        return Err(crate::error::AxilError::InvalidQuery(format!(
            "traversal path exceeds {MAX_PATH_BYTES} byte limit ({} bytes)",
            path.len()
        )));
    }

    let mut steps = Vec::new();
    let mut remaining = path;

    while !remaining.is_empty() {
        let (direction, rest) = if let Some(rest) = remaining.strip_prefix("<->") {
            (Direction::Both, rest)
        } else if let Some(rest) = remaining.strip_prefix("->") {
            (Direction::Out, rest)
        } else if let Some(rest) = remaining.strip_prefix("<-") {
            (Direction::In, rest)
        } else {
            return Err(crate::error::AxilError::InvalidQuery(format!(
                "expected '->', '<-', or '<->' at: {remaining}"
            )));
        };

        let end = find_next_arrow(rest);

        let edge_type = &rest[..end];
        if edge_type.is_empty() {
            return Err(crate::error::AxilError::InvalidQuery(
                "empty edge type in traversal path".into(),
            ));
        }
        if edge_type.len() > MAX_EDGE_TYPE_LEN {
            let preview: String = edge_type.chars().take(32).collect();
            return Err(crate::error::AxilError::InvalidQuery(format!(
                "edge type exceeds {MAX_EDGE_TYPE_LEN} byte limit: '{preview}'"
            )));
        }
        if edge_type.bytes().any(|b| b < 0x20 || b == 0x7F) {
            return Err(crate::error::AxilError::InvalidQuery(
                "edge type must not contain control characters".into(),
            ));
        }

        steps.push(TraversalStep {
            edge_type: edge_type.to_string(),
            direction,
        });

        remaining = &rest[end..];
    }

    if steps.len() > MAX_DEPTH {
        return Err(crate::error::AxilError::InvalidQuery(format!(
            "traversal path exceeds max depth of {MAX_DEPTH}"
        )));
    }

    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_outgoing() {
        let steps = parse_path("->modified").unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].edge_type, "modified");
        assert_eq!(steps[0].direction, Direction::Out);
    }

    #[test]
    fn parse_multi_hop() {
        let steps = parse_path("->modified->file").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].edge_type, "modified");
        assert_eq!(steps[1].edge_type, "file");
    }

    #[test]
    fn parse_incoming() {
        let steps = parse_path("<-authored").unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].direction, Direction::In);
    }

    #[test]
    fn parse_both_direction() {
        let steps = parse_path("<->related").unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].edge_type, "related");
        assert_eq!(steps[0].direction, Direction::Both);
    }

    #[test]
    fn parse_bidirectional_after_outgoing() {
        let steps = parse_path("->author<->mentions").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].edge_type, "author");
        assert_eq!(steps[0].direction, Direction::Out);
        assert_eq!(steps[1].edge_type, "mentions");
        assert_eq!(steps[1].direction, Direction::Both);
    }

    #[test]
    fn parse_mixed_directions() {
        let steps = parse_path("->modified<-authored").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].direction, Direction::Out);
        assert_eq!(steps[1].direction, Direction::In);
    }

    #[test]
    fn parse_hyphenated_edge_type() {
        let steps = parse_path("->depends-on->blocks").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].edge_type, "depends-on");
        assert_eq!(steps[1].edge_type, "blocks");
    }

    #[test]
    fn parse_empty_fails() {
        assert!(parse_path("").is_err());
    }

    #[test]
    fn parse_no_arrow_fails() {
        assert!(parse_path("modified").is_err());
    }

    #[test]
    fn parse_empty_edge_type_fails() {
        assert!(parse_path("->").is_err());
    }

    #[test]
    fn parse_at_max_depth() {
        let path = "->x".repeat(MAX_DEPTH);
        let steps = parse_path(&path).unwrap();
        assert_eq!(steps.len(), MAX_DEPTH);
    }

    #[test]
    fn parse_exceeds_max_depth() {
        let path = "->x".repeat(MAX_DEPTH + 1);
        assert!(parse_path(&path).is_err());
    }

    #[test]
    fn parse_exceeds_max_path_bytes() {
        let long_type = "a".repeat(MAX_PATH_BYTES);
        let path = format!("->{long_type}");
        assert!(parse_path(&path).is_err());
    }

    #[test]
    fn parse_exceeds_max_edge_type_len() {
        let long_type = "a".repeat(MAX_EDGE_TYPE_LEN + 1);
        let path = format!("->{long_type}");
        assert!(parse_path(&path).is_err());
    }
}

/// Engine that provides full-text search.
pub trait SearchIndex: Engine {
    /// Index a text field for a record.
    fn index_text(&self, id: &RecordId, field: &str, text: &str) -> Result<()>;

    /// Index a batch of records' auto-extracted text fields in one pass.
    ///
    /// Default implementation calls [`Engine::on_record_insert`] per record.
    /// Implementations backed by a write buffer (e.g. Tantivy) should
    /// override this to defer the commit until the whole batch is buffered —
    /// turning N expensive commits into 1.
    fn index_records_batch(&self, records: &[Record]) -> Result<()> {
        for record in records {
            self.on_record_insert(record)?;
        }
        Ok(())
    }

    /// Index the same named field across many records with a single commit.
    ///
    /// Default implementation calls [`Self::index_text`] per entry — one
    /// commit each. Buffer-backed implementations (e.g. Tantivy) should
    /// override this to defer the commit until the whole batch is buffered.
    fn index_field_batch(&self, field: &str, entries: &[(&RecordId, &str)]) -> Result<()> {
        for &(id, text) in entries {
            self.index_text(id, field, text)?;
        }
        Ok(())
    }

    /// Search indexed text across all fields, returning scored results.
    fn search_text(&self, query: &str, limit: usize) -> Result<Vec<(RecordId, f32)>>;

    /// Search indexed text within a specific field only.
    ///
    /// Default implementation falls back to `search_text` (ignoring field scope).
    /// Implementations with per-field indexing should override this.
    fn search_field(&self, query: &str, field: &str, limit: usize) -> Result<Vec<(RecordId, f32)>> {
        let _ = field;
        self.search_text(query, limit)
    }

    /// Fuzzy search with Levenshtein distance tolerance (1-2).
    ///
    /// Default implementation falls back to exact `search_text`.
    fn search_fuzzy(
        &self,
        query: &str,
        distance: u8,
        limit: usize,
    ) -> Result<Vec<(RecordId, f32)>> {
        let _ = distance;
        self.search_text(query, limit)
    }

    /// Search with snippet/highlight generation.
    ///
    /// Returns `(RecordId, score, snippet_html)`. Default returns empty snippets.
    fn search_with_snippets(
        &self,
        query: &str,
        limit: usize,
        max_chars: usize,
    ) -> Result<Vec<(RecordId, f32, String)>> {
        let _ = max_chars;
        let results = self.search_text(query, limit)?;
        Ok(results
            .into_iter()
            .map(|(id, score)| (id, score, String::new()))
            .collect())
    }

    /// Return all unique record IDs in the search index.
    ///
    /// Used by orphan cleanup during compaction. Default returns empty.
    fn all_indexed_ids(&self) -> Result<Vec<RecordId>> {
        Ok(Vec::new())
    }

    /// Commit pending writes and optimize the index.
    ///
    /// Default: no-op. Implementations with write buffers should flush and
    /// trigger segment merging.
    fn optimize(&self) -> Result<()> {
        Ok(())
    }
}

/// Time bucket granularity for aggregation queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBucket {
    /// One-hour buckets.
    Hour,
    /// One-day buckets.
    Day,
    /// One-week buckets (Monday-aligned).
    Week,
    /// One-month buckets.
    Month,
}

impl TimeBucket {
    /// Truncate a microsecond timestamp to the start of its bucket.
    pub fn truncate_us(&self, us: i64) -> i64 {
        // Hour and Day are fixed-duration — use integer arithmetic.
        match self {
            TimeBucket::Hour => us - us.rem_euclid(3_600_000_000),
            TimeBucket::Day => us - us.rem_euclid(86_400_000_000),
            TimeBucket::Week | TimeBucket::Month => {
                use chrono::{Datelike, TimeZone, Utc};
                let dt = Utc.timestamp_micros(us).single().unwrap_or_default();
                let date = match self {
                    TimeBucket::Week => {
                        let days_since_monday = dt.weekday().num_days_from_monday();
                        (dt - chrono::Duration::days(days_since_monday as i64)).date_naive()
                    }
                    TimeBucket::Month => dt.date_naive().with_day(1).unwrap_or(dt.date_naive()),
                    _ => unreachable!(),
                };
                date.and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp_micros()
            }
        }
    }
}

/// Engine that provides time-series range queries.
///
/// Records are automatically indexed by `created_at` on insert.
/// A secondary index on `updated_at` supports change tracking.
pub trait TimeSeriesIndex: Engine {
    /// Get record IDs in a time range `[start_us, end_us]` (microseconds since epoch).
    /// If `table` is `None`, searches across all tables.
    fn range(&self, table: Option<&str>, start_us: i64, end_us: i64) -> Result<Vec<RecordId>>;

    /// Get record IDs created within the last `duration_secs` seconds.
    fn since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<RecordId>>;

    /// Get the most recent `limit` record IDs, ordered newest first.
    fn latest(&self, table: Option<&str>, limit: usize) -> Result<Vec<RecordId>>;

    /// Get record IDs updated (not just created) within the last `duration_secs` seconds.
    fn changed_since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<RecordId>>;

    /// Get record IDs updated at or after an absolute timestamp (microseconds since epoch).
    /// Avoids double-clock-call skew when the caller already computed a threshold.
    fn changed_since_absolute(
        &self,
        table: Option<&str>,
        threshold_us: i64,
    ) -> Result<Vec<RecordId>>;

    /// Count records grouped by time bucket within a range.
    ///
    /// Returns `(bucket_start_us, count)` pairs sorted chronologically.
    fn count_by_bucket(
        &self,
        table: Option<&str>,
        bucket: TimeBucket,
        start_us: i64,
        end_us: i64,
    ) -> Result<Vec<(i64, usize)>>;

    /// Total number of indexed time entries.
    fn entry_count(&self) -> usize;
}
