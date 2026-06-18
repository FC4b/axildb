use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AxilError, Result};
use crate::plugin::{
    GraphIndex, SearchIndex, TextEmbedder, TimeSeriesIndex, TraversalStep, VectorIndex,
};
use crate::record::{Record, RecordId};
use crate::storage::Storage;

/// A single step in a query plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step: usize,
    #[serde(rename = "type")]
    pub step_type: String,
    pub params: Value,
}

/// Estimated cost level for a query plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EstimatedCost {
    Low,
    Medium,
    High,
}

/// Query plan returned by `explain()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub plan: Vec<PlanStep>,
    pub estimated_cost: EstimatedCost,
}

/// Timing for a single profiled step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStep {
    pub step: String,
    pub ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<usize>,
}

/// Profile result returned alongside query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    pub total_ms: f64,
    pub steps: Vec<ProfileStep>,
    pub bottleneck: Option<String>,
}

/// Comparison operators for field filtering.
#[derive(Debug, Clone)]
pub enum Op {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
}

impl std::str::FromStr for Op {
    type Err = AxilError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "=" | "==" => Ok(Op::Eq),
            "!=" => Ok(Op::Ne),
            ">" => Ok(Op::Gt),
            "<" => Ok(Op::Lt),
            ">=" => Ok(Op::Gte),
            "<=" => Ok(Op::Lte),
            "contains" => Ok(Op::Contains),
            other => Err(AxilError::InvalidQuery(format!(
                "unknown operator: {other}"
            ))),
        }
    }
}

/// Sort direction.
#[derive(Debug, Clone)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// A single where-clause filter.
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub field: String,
    pub op: Op,
    pub value: Value,
}

/// Cross-encoder rerank stage (Phase 15 P0.3). Implementations live in the
/// `axil-rerank` crate so this seam stays small and dep-free at the core
/// level. The trait is intentionally minimal: take a query + the resolved
/// records and reorder them in place. Errors are reported via a return
/// value so the pipeline can continue with the unreranked list on failure
/// — the reranker is a quality bonus, not a correctness requirement.
pub trait Rerank: Send + Sync {
    /// Reorder `records` against `query`, keeping at most `top_k_out`
    /// entries from the reranked prefix (the rest of the list is left
    /// untouched and may be discarded by downstream limit-trimming).
    /// Returns the count of records that were actually scored, useful for
    /// QueryProfile annotations.
    fn rerank_records(
        &self,
        query: &str,
        records: &mut Vec<Record>,
        top_k_in: usize,
        top_k_out: usize,
    ) -> std::result::Result<usize, String>;

    /// Human-readable identifier for diagnostics + profiling.
    fn name(&self) -> &str;
}

/// Time-range filter for queries that use the time-series index.
#[derive(Debug, Clone)]
struct TimeFilter {
    /// Microseconds since epoch. `None` = unbounded.
    after_us: Option<i64>,
    before_us: Option<i64>,
    /// If set, use the `changed_since` (updated_at) index instead of created_at.
    changed_only: bool,
}

/// Builder for composing queries against the storage layer.
pub struct QueryBuilder<'a> {
    storage: &'a Storage,
    vector_index: Option<&'a dyn VectorIndex>,
    embedder: Option<&'a dyn TextEmbedder>,
    graph_index: Option<&'a dyn GraphIndex>,
    timeseries_index: Option<&'a dyn TimeSeriesIndex>,
    fts_index: Option<&'a dyn SearchIndex>,
    table: Option<String>,
    wheres: Vec<WhereClause>,
    limit: usize,
    offset: usize,
    order_by: Option<(String, SortDirection)>,
    time_sort: Option<SortDirection>,
    vector_text_query: Option<(String, usize)>,
    vector_query: Option<(Vec<f32>, usize)>,
    traversal: Option<Vec<TraversalStep>>,
    traversal_error: Option<AxilError>,
    time_filter: Option<TimeFilter>,
    fts_query: Option<String>,
    reranker: Option<&'a dyn Rerank>,
    rerank_top_k_in: usize,
    rerank_top_k_out: usize,
}

impl<'a> QueryBuilder<'a> {
    /// Create a new query builder.
    pub fn new(
        storage: &'a Storage,
        vector_index: Option<&'a dyn VectorIndex>,
        embedder: Option<&'a dyn TextEmbedder>,
    ) -> Self {
        Self {
            storage,
            vector_index,
            embedder,
            graph_index: None,
            timeseries_index: None,
            fts_index: None,
            table: None,
            wheres: Vec::new(),
            limit: 100,
            offset: 0,
            order_by: None,
            time_sort: None,
            vector_text_query: None,
            vector_query: None,
            traversal: None,
            traversal_error: None,
            time_filter: None,
            fts_query: None,
            reranker: None,
            rerank_top_k_in: 50,
            rerank_top_k_out: 10,
        }
    }

    /// Attach a cross-encoder reranker stage (Phase 15 P0.3). When set, it
    /// runs after RRF fusion + record resolution but before sort+limit.
    /// `top_k_in` candidates are scored; the reranked prefix is kept up
    /// to `top_k_out`. Defaults: 50 in, 10 out. Pass `top_k_out` larger
    /// than the final `limit()` if downstream filters might trim.
    pub fn with_reranker(mut self, reranker: &'a dyn Rerank) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Override the rerank window (default 50 in, 10 out). Useful for
    /// benchmarks comparing different sizes — see scripts/longmemeval-gate.sh.
    pub fn rerank_window(mut self, top_k_in: usize, top_k_out: usize) -> Self {
        self.rerank_top_k_in = top_k_in;
        self.rerank_top_k_out = top_k_out;
        self
    }

    /// Set the graph index for traversal queries.
    pub fn with_graph(mut self, graph: &'a dyn GraphIndex) -> Self {
        self.graph_index = Some(graph);
        self
    }

    /// Set the timeseries index for time-range queries.
    pub fn with_timeseries(mut self, ts: &'a dyn TimeSeriesIndex) -> Self {
        self.timeseries_index = Some(ts);
        self
    }

    /// Set the FTS index for full-text search queries.
    pub fn with_fts(mut self, fts: &'a dyn SearchIndex) -> Self {
        self.fts_index = Some(fts);
        self
    }

    /// Filter by table name.
    pub fn table(mut self, name: &str) -> Self {
        self.table = Some(name.to_string());
        self
    }

    /// Filter by memory scope (Phase 11.3).
    ///
    /// Only return records whose `_scope` field matches the given scope.
    /// Can be called multiple times to allow multiple scopes.
    pub fn scope(self, scope: &str) -> Self {
        self.where_field("_scope", Op::Eq, serde_json::json!(scope))
    }

    /// Add a field filter.
    pub fn where_field(mut self, field: &str, op: Op, value: Value) -> Self {
        self.wheres.push(WhereClause {
            field: field.to_string(),
            op,
            value,
        });
        self
    }

    /// Set maximum number of results.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Set the offset for pagination.
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    /// Sort results by a JSON payload field.
    ///
    /// Mutually exclusive with `order_by_time()` — setting one clears the other.
    /// When combined with `traverse()`, sorting applies to the traversal
    /// *endpoints*, not the starting records.
    pub fn order_by(mut self, field: &str, direction: SortDirection) -> Self {
        self.order_by = Some((field.to_string(), direction));
        self.time_sort = None;
        self
    }

    /// Add a semantic text search to the query.
    ///
    /// The text is embedded using the configured model and the top-k most
    /// similar records are returned. When combined with `where_field`,
    /// field filters are applied as post-filters on the vector results.
    pub fn similar_to(mut self, text: &str, top_k: usize) -> Self {
        self.vector_text_query = Some((text.to_string(), top_k));
        self
    }

    /// Add a raw vector search to the query.
    pub fn similar_to_vector(mut self, vector: Vec<f32>, top_k: usize) -> Self {
        self.vector_query = Some((vector, top_k));
        self
    }

    /// Add a full-text search to the query chain.
    ///
    /// When combined with `similar_to`, scores from both vector similarity
    /// and FTS relevance are fused (averaged) to produce a blended ranking.
    /// When used standalone, returns records ranked by BM25 relevance.
    pub fn search_text(mut self, query: &str) -> Self {
        self.fts_query = Some(query.to_string());
        self
    }

    /// Add a graph traversal to the query chain.
    ///
    /// When combined with `similar_to`, the vector search results are used
    /// as starting points for the traversal. When used standalone, requires
    /// `table()` to select starting records. In both cases, the returned
    /// records are the traversal *endpoints*, not the starting records.
    pub fn traverse(mut self, path: &str) -> Self {
        match crate::plugin::parse_path(path) {
            Ok(steps) => self.traversal = Some(steps),
            Err(e) => self.traversal_error = Some(e),
        }
        self
    }

    /// Filter to records created within the last `duration_secs` seconds.
    /// Uses the time-series index when available for O(log n) lookups.
    pub fn since(mut self, duration_secs: u64) -> Self {
        let now_us = chrono::Utc::now().timestamp_micros();
        let delta_us = i64::try_from(duration_secs)
            .ok()
            .and_then(|s| s.checked_mul(1_000_000))
            .unwrap_or(i64::MAX);
        let start_us = now_us.saturating_sub(delta_us);
        let tf = self.time_filter.get_or_insert(TimeFilter {
            after_us: None,
            before_us: None,
            changed_only: false,
        });
        tf.after_us = Some(start_us);
        self
    }

    /// Filter to records created after the given timestamp (microseconds since epoch).
    pub fn after(mut self, timestamp_us: i64) -> Self {
        let tf = self.time_filter.get_or_insert(TimeFilter {
            after_us: None,
            before_us: None,
            changed_only: false,
        });
        tf.after_us = Some(timestamp_us);
        self
    }

    /// Filter to records created before the given timestamp (microseconds since epoch).
    pub fn before(mut self, timestamp_us: i64) -> Self {
        let tf = self.time_filter.get_or_insert(TimeFilter {
            after_us: None,
            before_us: None,
            changed_only: false,
        });
        tf.before_us = Some(timestamp_us);
        self
    }

    /// Filter to records in a time range (microseconds since epoch).
    pub fn between(mut self, start_us: i64, end_us: i64) -> Self {
        self.time_filter = Some(TimeFilter {
            after_us: Some(start_us),
            before_us: Some(end_us),
            changed_only: false,
        });
        self
    }

    /// Sort results by `Record.created_at` timestamp.
    ///
    /// Unlike `.order_by(field)` which sorts on JSON payload fields,
    /// this sorts on the struct-level `created_at` timestamp.
    /// Mutually exclusive with `order_by()` — setting one clears the other.
    pub fn order_by_time(mut self, direction: SortDirection) -> Self {
        self.time_sort = Some(direction);
        self.order_by = None;
        self
    }

    /// Filter to records updated (not just created) within the last `duration_secs`.
    ///
    /// Preserves any previously set `before_us` constraint (e.g. from `.before()`).
    pub fn changed_since(mut self, duration_secs: u64) -> Self {
        let now_us = chrono::Utc::now().timestamp_micros();
        let delta_us = i64::try_from(duration_secs)
            .ok()
            .and_then(|s| s.checked_mul(1_000_000))
            .unwrap_or(i64::MAX);
        let start_us = now_us.saturating_sub(delta_us);
        let tf = self.time_filter.get_or_insert(TimeFilter {
            after_us: None,
            before_us: None,
            changed_only: false,
        });
        tf.after_us = Some(start_us);
        tf.changed_only = true;
        self
    }

    /// Show the query plan without executing.
    pub fn explain(&self) -> QueryPlan {
        let mut steps = Vec::new();
        let mut step_num = 0;
        let mut cost = EstimatedCost::Low;

        // Vector search step
        if let Some((ref text, top_k)) = self.vector_text_query {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "vector_search".to_string(),
                params: serde_json::json!({"query": text, "top_k": top_k}),
            });
            cost = EstimatedCost::Medium;
        } else if let Some((_, top_k)) = self.vector_query {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "vector_search".to_string(),
                params: serde_json::json!({"raw_vector": true, "top_k": top_k}),
            });
            cost = EstimatedCost::Medium;
        }

        // FTS step
        if let Some(ref query) = self.fts_query {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "fts_search".to_string(),
                params: serde_json::json!({"query": query}),
            });
            if cost == EstimatedCost::Low {
                cost = EstimatedCost::Medium;
            }
        }

        // Score fusion — RRF when 2+ search sources, legacy label for single combo
        if self.search_source_count() >= 2 {
            step_num += 1;
            let sources: Vec<&str> = [
                if self.vector_text_query.is_some() || self.vector_query.is_some() {
                    Some("vector")
                } else {
                    None
                },
                if self.fts_query.is_some() {
                    Some("fts")
                } else {
                    None
                },
                if self.time_filter.is_some() && self.timeseries_index.is_some() {
                    Some("timeseries")
                } else {
                    None
                },
            ]
            .into_iter()
            .flatten()
            .collect();
            steps.push(PlanStep {
                step: step_num,
                step_type: "rrf_fusion".to_string(),
                params: serde_json::json!({"strategy": "reciprocal_rank_fusion", "k": 60, "sources": sources}),
            });
            cost = EstimatedCost::High;
        }

        // Graph traversal step
        if let Some(ref trav) = self.traversal {
            step_num += 1;
            let path_desc: Vec<String> = trav
                .iter()
                .map(|s| {
                    let arrow = match s.direction {
                        crate::plugin::Direction::Out => "->",
                        crate::plugin::Direction::In => "<-",
                        crate::plugin::Direction::Both => "<->",
                    };
                    format!("{}{}", arrow, s.edge_type)
                })
                .collect();
            steps.push(PlanStep {
                step: step_num,
                step_type: "graph_traverse".to_string(),
                params: serde_json::json!({"path": path_desc.join("")}),
            });
            cost = EstimatedCost::High;
        }

        // Time filter step
        if let Some(ref tf) = self.time_filter {
            step_num += 1;
            let mut params = serde_json::Map::new();
            if let Some(after) = tf.after_us {
                params.insert("after_us".to_string(), serde_json::json!(after));
            }
            if let Some(before) = tf.before_us {
                params.insert("before_us".to_string(), serde_json::json!(before));
            }
            if tf.changed_only {
                params.insert("changed_only".to_string(), serde_json::json!(true));
            }
            let uses_index = self.timeseries_index.is_some();
            params.insert("uses_index".to_string(), serde_json::json!(uses_index));
            steps.push(PlanStep {
                step: step_num,
                step_type: "time_filter".to_string(),
                params: Value::Object(params),
            });
        }

        // Table scan / field filter step
        if let Some(ref table) = self.table {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "table_scan".to_string(),
                params: serde_json::json!({"table": table}),
            });
        }

        // Where filters
        if !self.wheres.is_empty() {
            step_num += 1;
            let filters: Vec<Value> = self
                .wheres
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "field": w.field,
                        "op": format!("{:?}", w.op),
                        "value": w.value
                    })
                })
                .collect();
            steps.push(PlanStep {
                step: step_num,
                step_type: "field_filter".to_string(),
                params: serde_json::json!({"filters": filters}),
            });
        }

        // Sort step
        if let Some((ref field, ref dir)) = self.order_by {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "sort".to_string(),
                params: serde_json::json!({"field": field, "direction": format!("{dir:?}")}),
            });
        } else if let Some(ref dir) = self.time_sort {
            step_num += 1;
            steps.push(PlanStep {
                step: step_num,
                step_type: "sort".to_string(),
                params: serde_json::json!({"field": "created_at", "direction": format!("{dir:?}")}),
            });
        }

        // Limit step
        step_num += 1;
        steps.push(PlanStep {
            step: step_num,
            step_type: "limit".to_string(),
            params: serde_json::json!({"limit": self.limit, "offset": self.offset}),
        });

        QueryPlan {
            plan: steps,
            estimated_cost: cost,
        }
    }

    /// Execute the query with per-step profiling.
    pub fn exec_profiled(self) -> Result<(Vec<Record>, QueryProfile)> {
        let total_start = std::time::Instant::now();
        let mut profile_steps = Vec::new();

        // Determine the execution path and profile it.
        if let Some(e) = self.traversal_error {
            return Err(e);
        }

        // Unified pipeline with per-step profiling when 2+ search types.
        if self.search_source_count() >= 2 {
            return self.exec_unified_profiled();
        }

        // Vector search
        if self.vector_text_query.is_some() || self.vector_query.is_some() {
            let has_traversal = self.traversal.is_some();
            let results = self.exec_vector()?;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            let mut step_name = "vector_search".to_string();
            if has_traversal {
                step_name = "vector_search+traverse".to_string();
            }
            profile_steps.push(ProfileStep {
                step: step_name,
                ms: total_ms,
                candidates: Some(results.len()),
            });
            let profile = build_profile(total_ms, profile_steps);
            return Ok((results, profile));
        }

        // FTS
        if self.fts_query.is_some() {
            let results = self.exec_fts()?;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            profile_steps.push(ProfileStep {
                step: "fts_search".to_string(),
                ms: total_ms,
                candidates: Some(results.len()),
            });
            let profile = build_profile(total_ms, profile_steps);
            return Ok((results, profile));
        }

        // Traverse
        if self.traversal.is_some() {
            let results = self.exec_traverse()?;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            profile_steps.push(ProfileStep {
                step: "graph_traverse".to_string(),
                ms: total_ms,
                candidates: Some(results.len()),
            });
            let profile = build_profile(total_ms, profile_steps);
            return Ok((results, profile));
        }

        // Time-series
        if self.time_filter.is_some() && self.timeseries_index.is_some() {
            let results = self.exec_timeseries()?;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            profile_steps.push(ProfileStep {
                step: "timeseries_query".to_string(),
                ms: total_ms,
                candidates: Some(results.len()),
            });
            let profile = build_profile(total_ms, profile_steps);
            return Ok((results, profile));
        }

        // Default table scan
        let table = self
            .table
            .as_deref()
            .ok_or_else(|| AxilError::InvalidQuery("table is required".into()))?;

        let scan_start = std::time::Instant::now();
        let results =
            if self.wheres.is_empty() && self.order_by.is_none() && self.time_filter.is_none() {
                self.storage.list(table, self.limit, self.offset)?
            } else {
                let records = self.storage.list(table, usize::MAX, 0)?;
                let mut filtered: Vec<Record> = records
                    .into_iter()
                    .filter(|r| self.matches_all(r) && self.matches_time_filter(r))
                    .collect();
                self.apply_sort(&mut filtered);
                filtered
                    .into_iter()
                    .skip(self.offset)
                    .take(self.limit)
                    .collect()
            };
        let scan_ms = scan_start.elapsed().as_secs_f64() * 1000.0;
        profile_steps.push(ProfileStep {
            step: "table_scan".to_string(),
            ms: scan_ms,
            candidates: Some(results.len()),
        });

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let profile = build_profile(total_ms, profile_steps);
        Ok((results, profile))
    }

    /// Execute the query and return matching records.
    pub fn exec(self) -> Result<Vec<Record>> {
        if let Some(e) = self.traversal_error {
            return Err(e);
        }

        // Unified pipeline: activated when 2+ search types are combined.
        // This handles vector+FTS, vector+timeseries, FTS+timeseries,
        // and all-three, with optional graph traversal on top.
        if self.search_source_count() >= 2 {
            return self.exec_unified();
        }

        // Handle vector search queries (may include chained traversal).
        if self.vector_text_query.is_some() || self.vector_query.is_some() {
            return self.exec_vector();
        }

        // Handle standalone FTS queries (may include chained traversal).
        if self.fts_query.is_some() {
            return self.exec_fts();
        }

        // Handle standalone graph traversal from a table result set.
        // When a time filter + timeseries index are available alongside a
        // traversal, use the timeseries index to narrow the starting set
        // (O(log n)) instead of falling back to a full table scan.
        if self.traversal.is_some() {
            return self.exec_traverse();
        }

        // Handle time-series index queries when a time filter is set and
        // the timeseries index is available.
        if self.time_filter.is_some() && self.timeseries_index.is_some() {
            return self.exec_timeseries();
        }

        let table = self
            .table
            .as_deref()
            .ok_or_else(|| AxilError::InvalidQuery("table is required".into()))?;

        // When no filters or sorting, push pagination down to storage.
        if self.wheres.is_empty() && self.order_by.is_none() && self.time_filter.is_none() {
            return self.storage.list(table, self.limit, self.offset);
        }

        // Otherwise load all records for filtering/sorting.
        let records = self.storage.list(table, usize::MAX, 0)?;

        let mut filtered: Vec<Record> = records
            .into_iter()
            .filter(|r| self.matches_all(r) && self.matches_time_filter(r))
            .collect();

        self.apply_sort(&mut filtered);

        let result = filtered
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Execute a vector-based query with optional post-filtering.
    ///
    /// When post-filters (table or where clauses) are present, the index is
    /// queried with an over-fetch factor (4x) to compensate for records that
    /// will be discarded by the filters. This avoids returning fewer results
    /// than the caller expects.
    fn exec_vector(self) -> Result<Vec<Record>> {
        let vi = self
            .vector_index
            .ok_or_else(|| AxilError::plugin("no vector index configured"))?;

        let has_post_filter =
            self.table.is_some() || !self.wheres.is_empty() || self.time_filter.is_some();

        // Get vector results — over-fetch when post-filters will discard some,
        // and over-fetch *more* when a reranker is attached so the rerank
        // window has full top_k_in candidates to score.
        let over_fetch_factor = if has_post_filter { 4 } else { 1 };

        let scored = if let Some((ref text, top_k)) = self.vector_text_query {
            let embedder = self
                .embedder
                .ok_or_else(|| AxilError::plugin("no embedder configured for text search"))?;
            let vec = embedder.embed(text)?;
            let mut fetch_k = (top_k + self.offset).saturating_mul(over_fetch_factor);
            if self.reranker.is_some() {
                // Phase 15 P0.3: rerank window must dominate fetch sizing or
                // the reranker only sees the prefix the vector already ranked.
                fetch_k = fetch_k.max(self.rerank_top_k_in + self.offset);
            }
            vi.search(&vec, fetch_k)?
        } else if let Some((ref vec, top_k)) = self.vector_query {
            let mut fetch_k = (top_k + self.offset).saturating_mul(over_fetch_factor);
            if self.reranker.is_some() {
                fetch_k = fetch_k.max(self.rerank_top_k_in + self.offset);
            }
            vi.search(vec, fetch_k)?
        } else {
            unreachable!()
        };

        // Fetch records and apply post-filters.
        // When traversal follows, collect all matching seeds (no early-exit)
        // because there is no 1:1 relationship between seeds and endpoints.
        // When a reranker is attached, widen the cap so it sees the full
        // rerank window (mirrors exec_unified_profiled).
        let seed_cap = if self.traversal.is_some() {
            usize::MAX
        } else if self.reranker.is_some() {
            (self.rerank_top_k_in + self.offset).max(self.limit + self.offset)
        } else {
            self.limit + self.offset
        };

        let mut results = Vec::new();
        for (rid, _score) in &scored {
            if let Some(record) = self.storage.get(rid)? {
                // Apply table filter if set.
                if let Some(ref table) = self.table {
                    if record.table != *table {
                        continue;
                    }
                }
                // Apply where and time filters.
                if self.matches_all(&record) && self.matches_time_filter(&record) {
                    results.push(record);
                    if results.len() >= seed_cap {
                        break;
                    }
                }
            }
        }

        // Apply graph traversal if configured.
        if let Some(ref steps) = self.traversal {
            let gi = self
                .graph_index
                .ok_or_else(|| AxilError::plugin("no graph index configured for traversal"))?;

            results =
                fan_out_traversal(gi, self.storage, &results, steps, self.limit + self.offset)?;
        }

        // Phase 15 P0.3 rerank stage. Same shape as exec_unified_profiled —
        // runs only when a reranker is attached and we have query text.
        if let Some(reranker) = self.reranker {
            let query_text: Option<&str> = self
                .vector_text_query
                .as_ref()
                .map(|(t, _)| t.as_str())
                .or_else(|| self.fts_query.as_deref());
            if let Some(qt) = query_text {
                if !results.is_empty() && !qt.is_empty() {
                    if let Err(e) = reranker.rerank_records(
                        qt,
                        &mut results,
                        self.rerank_top_k_in,
                        self.rerank_top_k_out,
                    ) {
                        eprintln!(
                            "[rerank] {}: {} — falling back to fused order",
                            reranker.name(),
                            e
                        );
                    }
                }
            }
        }

        self.apply_sort(&mut results);

        // Apply pagination.
        let result = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Execute a standalone graph traversal starting from a table result set.
    ///
    /// Records are loaded from `table` (with optional where-clause filters),
    /// then each matching record is used as a starting point for traversal.
    /// The final result is the union of all traversal endpoints.
    fn exec_traverse(self) -> Result<Vec<Record>> {
        let gi = self
            .graph_index
            .ok_or_else(|| AxilError::plugin("no graph index configured for traversal"))?;
        let steps = self.traversal.as_ref().ok_or_else(|| {
            AxilError::InvalidQuery("exec_traverse called without traversal steps".into())
        })?;

        // Gather starting records. When a timeseries index and time filter
        // are both available, use the index for O(log n) lookups instead of
        // scanning the entire table.
        let starting = if let (Some(tsi), Some(ref tf)) = (self.timeseries_index, &self.time_filter)
        {
            let now_us = chrono::Utc::now().timestamp_micros();
            let start = tf.after_us.unwrap_or(0);
            let end = tf.before_us.unwrap_or(now_us);
            let ids = if tf.changed_only {
                tsi.changed_since_absolute(self.table.as_deref(), start)?
            } else {
                tsi.range(self.table.as_deref(), start, end)?
            };
            let mut records = Vec::new();
            for id in &ids {
                if let Some(record) = self.storage.get(id)? {
                    if self.matches_all(&record) && self.matches_time_filter(&record) {
                        records.push(record);
                    }
                }
            }
            records
        } else if let Some(ref table) = self.table {
            let records = self.storage.list(table, usize::MAX, 0)?;
            records
                .into_iter()
                .filter(|r| self.matches_all(r) && self.matches_time_filter(r))
                .collect::<Vec<_>>()
        } else {
            return Err(AxilError::InvalidQuery(
                "traverse() requires a table() to select starting records".into(),
            ));
        };

        let result_cap = self.limit + self.offset;
        let mut results = fan_out_traversal(gi, self.storage, &starting, steps, result_cap)?;

        self.apply_sort(&mut results);

        let result = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Execute a time-series index query with optional post-filtering.
    fn exec_timeseries(self) -> Result<Vec<Record>> {
        let tsi = self
            .timeseries_index
            .ok_or_else(|| AxilError::plugin("no timeseries index configured"))?;
        let tf = self.time_filter.as_ref().ok_or_else(|| {
            AxilError::InvalidQuery("exec_timeseries called without time filter".into())
        })?;

        let now_us = chrono::Utc::now().timestamp_micros();
        let start = tf.after_us.unwrap_or(0);
        let end = tf.before_us.unwrap_or(now_us);

        let ids = if tf.changed_only {
            // Use absolute threshold to avoid double-clock-call skew.
            tsi.changed_since_absolute(self.table.as_deref(), start)?
        } else {
            tsi.range(self.table.as_deref(), start, end)?
        };

        // Resolve and apply post-filters (where-clauses and time bounds).
        // The time post-filter catches bounds the index didn't enforce
        // (e.g. before_us on changed_since queries).
        let mut results = Vec::new();
        for id in &ids {
            if let Some(record) = self.storage.get(id)? {
                if self.matches_all(&record) && self.matches_time_filter(&record) {
                    results.push(record);
                }
            }
        }

        self.apply_sort(&mut results);

        let result = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Execute a standalone FTS query with optional post-filtering and traversal.
    fn exec_fts(self) -> Result<Vec<Record>> {
        let fi = self
            .fts_index
            .ok_or_else(|| AxilError::plugin("no FTS index configured"))?;
        let query = self
            .fts_query
            .as_ref()
            .ok_or_else(|| AxilError::InvalidQuery("exec_fts called without FTS query".into()))?;

        let has_post_filter =
            self.table.is_some() || !self.wheres.is_empty() || self.time_filter.is_some();
        let over_fetch = if has_post_filter { 4 } else { 1 };
        let fetch_limit = (self.limit + self.offset).saturating_mul(over_fetch);

        let scored = fi.search_text(query, fetch_limit)?;

        let seed_cap = if self.traversal.is_some() {
            usize::MAX
        } else {
            self.limit + self.offset
        };

        let mut results = Vec::new();
        for (rid, _score) in &scored {
            if let Some(record) = self.storage.get(rid)? {
                if let Some(ref table) = self.table {
                    if record.table != *table {
                        continue;
                    }
                }
                if self.matches_all(&record) && self.matches_time_filter(&record) {
                    results.push(record);
                    if results.len() >= seed_cap {
                        break;
                    }
                }
            }
        }

        // Apply graph traversal if configured.
        if let Some(ref steps) = self.traversal {
            let gi = self
                .graph_index
                .ok_or_else(|| AxilError::plugin("no graph index configured for traversal"))?;
            results =
                fan_out_traversal(gi, self.storage, &results, steps, self.limit + self.offset)?;
        }

        self.apply_sort(&mut results);

        let result = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Count how many independent search sources are active.
    fn search_source_count(&self) -> usize {
        let has_vector = self.vector_text_query.is_some() || self.vector_query.is_some();
        let has_fts = self.fts_query.is_some();
        let has_time = self.time_filter.is_some() && self.timeseries_index.is_some();
        has_vector as usize + has_fts as usize + has_time as usize
    }

    /// Unified execution pipeline that handles arbitrary combinations of
    /// vector + FTS + time-series + graph traversal + field filters.
    ///
    /// Uses cascaded filtering (8b.1): runs cheapest filters first (timeseries,
    /// FTS) before expensive ones (vector). If an early filter produces high-
    /// confidence results (top score > 0.95), skips remaining expensive indexes.
    fn exec_unified(self) -> Result<Vec<Record>> {
        let has_vector = self.vector_text_query.is_some() || self.vector_query.is_some();
        let has_fts = self.fts_query.is_some();
        let has_time = self.time_filter.is_some() && self.timeseries_index.is_some();

        let min_fetch = (self.limit + self.offset).saturating_mul(4);
        let mut ranked_lists: Vec<Vec<(RecordId, f32)>> = Vec::new();

        // ── Cascaded filtering (8b.1): run cheapest filters first ──
        // Cost order: timeseries (cheap scan) → FTS (inverted index) → vector (HNSW)
        // After each step, check if we can short-circuit.

        // Step 1a: Time-series (cheapest — simple range scan)
        if has_time {
            let tsi = self.timeseries_index.unwrap();
            let tf = self.time_filter.as_ref().unwrap();
            let now_us = chrono::Utc::now().timestamp_micros();
            let start = tf.after_us.unwrap_or(0);
            let end = tf.before_us.unwrap_or(now_us);

            let ids = if tf.changed_only {
                tsi.changed_since_absolute(self.table.as_deref(), start)?
            } else {
                tsi.range(self.table.as_deref(), start, end)?
            };

            let count = ids.len();
            let ts_scored: Vec<(RecordId, f32)> = ids
                .into_iter()
                .rev()
                .enumerate()
                .map(|(i, id)| {
                    let score = if count > 0 {
                        1.0 - (i as f32 / count as f32)
                    } else {
                        0.0
                    };
                    (id, score)
                })
                .collect();
            ranked_lists.push(ts_scored);
        }

        // Step 1b: FTS (medium cost — inverted index lookup)
        if has_fts {
            let fi = self
                .fts_index
                .ok_or_else(|| AxilError::plugin("no FTS index configured"))?;
            let query = self.fts_query.as_ref().unwrap();
            let scored = fi.search_text(query, min_fetch)?;
            ranked_lists.push(scored);
        }

        // Score threshold cutoff (8b.1): skip vector search if FTS already
        // found high-confidence results. Timeseries pseudo-scores (always 1.0
        // for the most recent hit) are excluded — they don't indicate relevance.
        let skip_vector = has_fts && {
            // Only check FTS scores (last list if FTS ran after timeseries)
            let fts_list_idx = if has_time { 1 } else { 0 };
            ranked_lists
                .get(fts_list_idx)
                .and_then(|list| list.first().map(|(_, s)| *s > 0.95))
                .unwrap_or(false)
        };

        // Step 1c: Vector search (most expensive — HNSW traversal)
        if has_vector && !skip_vector {
            let vi = self
                .vector_index
                .ok_or_else(|| AxilError::plugin("no vector index configured"))?;

            let scored = if let Some((ref text, top_k)) = self.vector_text_query {
                let embedder = self
                    .embedder
                    .ok_or_else(|| AxilError::plugin("no embedder configured for text search"))?;
                let vec = embedder.embed(text)?;
                let fetch_k = (top_k + self.offset).saturating_mul(4).max(min_fetch);
                vi.search(&vec, fetch_k)?
            } else if let Some((ref vec, top_k)) = self.vector_query {
                let fetch_k = (top_k + self.offset).saturating_mul(4).max(min_fetch);
                vi.search(vec, fetch_k)?
            } else {
                unreachable!()
            };
            ranked_lists.push(scored);
        }

        // ── Step 2: Score Fusion via adaptive RRF (8b.2) ──
        let fused = if ranked_lists.is_empty() {
            Vec::new()
        } else {
            reciprocal_rank_fusion(&ranked_lists, 60)
        };

        // ── Step 3 & 4: Record Resolution + Post-Filtering ──
        let seed_cap = if self.traversal.is_some() {
            usize::MAX
        } else {
            self.limit + self.offset
        };

        let mut results = Vec::new();
        for (rid, _score) in &fused {
            if let Some(record) = self.storage.get(rid)? {
                // Table filter
                if let Some(ref table) = self.table {
                    if record.table != *table {
                        continue;
                    }
                }
                // Where clauses + time filter (post-filter for records not
                // perfectly pre-filtered by the timeseries index bounds).
                if self.matches_all(&record) && self.matches_time_filter(&record) {
                    results.push(record);
                    if results.len() >= seed_cap {
                        break;
                    }
                }
            }
        }

        // ── Step 5: Graph Traversal ──
        if let Some(ref steps) = self.traversal {
            let gi = self
                .graph_index
                .ok_or_else(|| AxilError::plugin("no graph index configured for traversal"))?;
            results =
                fan_out_traversal(gi, self.storage, &results, steps, self.limit + self.offset)?;
        }

        // ── Step 6: Sort + Limit ──
        self.apply_sort(&mut results);

        let result = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        Ok(result)
    }

    /// Unified execution pipeline with per-step profiling.
    fn exec_unified_profiled(self) -> Result<(Vec<Record>, QueryProfile)> {
        let total_start = std::time::Instant::now();
        let mut profile_steps = Vec::new();

        let has_vector = self.vector_text_query.is_some() || self.vector_query.is_some();
        let has_fts = self.fts_query.is_some();
        let has_time = self.time_filter.is_some() && self.timeseries_index.is_some();

        let min_fetch = (self.limit + self.offset).saturating_mul(4);
        let mut ranked_lists: Vec<Vec<(RecordId, f32)>> = Vec::new();

        // Cascaded order (matches exec_unified): timeseries → FTS → vector

        // Step 1a: Time-series (cheapest)
        if has_time {
            let step_start = std::time::Instant::now();
            let tsi = self.timeseries_index.unwrap();
            let tf = self.time_filter.as_ref().unwrap();
            let now_us = chrono::Utc::now().timestamp_micros();
            let start = tf.after_us.unwrap_or(0);
            let end = tf.before_us.unwrap_or(now_us);

            let ids = if tf.changed_only {
                tsi.changed_since_absolute(self.table.as_deref(), start)?
            } else {
                tsi.range(self.table.as_deref(), start, end)?
            };

            let count = ids.len();
            let ts_scored: Vec<(RecordId, f32)> = ids
                .into_iter()
                .rev()
                .enumerate()
                .map(|(i, id)| {
                    let score = if count > 0 {
                        1.0 - (i as f32 / count as f32)
                    } else {
                        0.0
                    };
                    (id, score)
                })
                .collect();
            ranked_lists.push(ts_scored);
            profile_steps.push(ProfileStep {
                step: "timeseries_filter".to_string(),
                ms: step_start.elapsed().as_secs_f64() * 1000.0,
                candidates: Some(count),
            });
        }

        // Step 1b: FTS search
        if has_fts {
            let step_start = std::time::Instant::now();
            let fi = self
                .fts_index
                .ok_or_else(|| AxilError::plugin("no FTS index configured"))?;
            let query = self.fts_query.as_ref().unwrap();
            let scored = fi.search_text(query, min_fetch)?;
            let count = scored.len();
            ranked_lists.push(scored);
            profile_steps.push(ProfileStep {
                step: "fts_search".to_string(),
                ms: step_start.elapsed().as_secs_f64() * 1000.0,
                candidates: Some(count),
            });
        }

        // Step 1c: Vector search (skip if FTS already has high-confidence results)
        let skip_vector = has_fts && {
            let fts_list_idx = if has_time { 1 } else { 0 };
            ranked_lists
                .get(fts_list_idx)
                .and_then(|list| list.first().map(|(_, s)| *s > 0.95))
                .unwrap_or(false)
        };
        if has_vector && !skip_vector {
            let step_start = std::time::Instant::now();
            let vi = self
                .vector_index
                .ok_or_else(|| AxilError::plugin("no vector index configured"))?;
            let scored = if let Some((ref text, top_k)) = self.vector_text_query {
                let embedder = self
                    .embedder
                    .ok_or_else(|| AxilError::plugin("no embedder configured for text search"))?;
                let vec = embedder.embed(text)?;
                let fetch_k = (top_k + self.offset).saturating_mul(4).max(min_fetch);
                vi.search(&vec, fetch_k)?
            } else if let Some((ref vec, top_k)) = self.vector_query {
                let fetch_k = (top_k + self.offset).saturating_mul(4).max(min_fetch);
                vi.search(vec, fetch_k)?
            } else {
                unreachable!()
            };
            let count = scored.len();
            ranked_lists.push(scored);
            profile_steps.push(ProfileStep {
                step: "vector_search".to_string(),
                ms: step_start.elapsed().as_secs_f64() * 1000.0,
                candidates: Some(count),
            });
        } else if has_vector && skip_vector {
            profile_steps.push(ProfileStep {
                step: "vector_search_skipped".to_string(),
                ms: 0.0,
                candidates: Some(0),
            });
        }

        // Step 2: RRF fusion
        let fusion_start = std::time::Instant::now();
        let fused = reciprocal_rank_fusion(&ranked_lists, 60);
        profile_steps.push(ProfileStep {
            step: "rrf_fusion".to_string(),
            ms: fusion_start.elapsed().as_secs_f64() * 1000.0,
            candidates: Some(fused.len()),
        });

        // Step 3: Record resolution + post-filter
        let resolve_start = std::time::Instant::now();
        let seed_cap = if self.traversal.is_some() {
            usize::MAX
        } else if self.reranker.is_some() {
            // Phase 15 P0.3: when a reranker is attached, the resolve
            // stage must hand it the full window — capping at
            // limit+offset would turn rerank into a no-op (it'd only
            // see the prefix that already won by fused score).
            (self.rerank_top_k_in + self.offset).max(self.limit + self.offset)
        } else {
            self.limit + self.offset
        };

        let mut results = Vec::new();
        for (rid, _score) in &fused {
            if let Some(record) = self.storage.get(rid)? {
                if let Some(ref table) = self.table {
                    if record.table != *table {
                        continue;
                    }
                }
                if self.matches_all(&record) && self.matches_time_filter(&record) {
                    results.push(record);
                    if results.len() >= seed_cap {
                        break;
                    }
                }
            }
        }
        profile_steps.push(ProfileStep {
            step: "resolve_and_filter".to_string(),
            ms: resolve_start.elapsed().as_secs_f64() * 1000.0,
            candidates: Some(results.len()),
        });

        // Step 4: Graph traversal
        if let Some(ref steps) = self.traversal {
            let trav_start = std::time::Instant::now();
            let gi = self
                .graph_index
                .ok_or_else(|| AxilError::plugin("no graph index configured for traversal"))?;
            results =
                fan_out_traversal(gi, self.storage, &results, steps, self.limit + self.offset)?;
            profile_steps.push(ProfileStep {
                step: "graph_traverse".to_string(),
                ms: trav_start.elapsed().as_secs_f64() * 1000.0,
                candidates: Some(results.len()),
            });
        }

        // Step 4b: Cross-encoder rerank (Phase 15 P0.3). Runs only when a
        // reranker is attached AND the query carries a textual signal —
        // pure structural queries (graph-only / time-only) have no string
        // to score against, so the stage is skipped.
        if let Some(reranker) = self.reranker {
            let query_text: Option<&str> = self
                .vector_text_query
                .as_ref()
                .map(|(t, _)| t.as_str())
                .or_else(|| self.fts_query.as_deref());
            if let Some(qt) = query_text {
                if !results.is_empty() && !qt.is_empty() {
                    let rerank_start = std::time::Instant::now();
                    let top_k_in = self.rerank_top_k_in;
                    let top_k_out = self.rerank_top_k_out;
                    let scored =
                        match reranker.rerank_records(qt, &mut results, top_k_in, top_k_out) {
                            Ok(n) => n,
                            Err(e) => {
                                // Reranker is best-effort: log & fall through
                                // with the un-reranked list. The gate measures
                                // recall, so a silent failure is visible.
                                eprintln!(
                                    "[rerank] {}: {} — falling back to fused order",
                                    reranker.name(),
                                    e
                                );
                                0
                            }
                        };
                    profile_steps.push(ProfileStep {
                        step: format!("rerank:{}", reranker.name()),
                        ms: rerank_start.elapsed().as_secs_f64() * 1000.0,
                        candidates: Some(scored),
                    });
                }
            }
        }

        // Step 5: Sort + limit
        self.apply_sort(&mut results);
        let result: Vec<Record> = results
            .into_iter()
            .skip(self.offset)
            .take(self.limit)
            .collect();

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let profile = build_profile(total_ms, profile_steps);
        Ok((result, profile))
    }

    /// Apply `order_by` and/or `order_by_time` sorting to a result set.
    fn apply_sort(&self, results: &mut [Record]) {
        if let Some((ref field, ref dir)) = self.order_by {
            results.sort_by(|a, b| {
                let cmp = compare_json_values(a.data.get(field), b.data.get(field));
                match dir {
                    SortDirection::Asc => cmp,
                    SortDirection::Desc => cmp.reverse(),
                }
            });
        }
        if let Some(ref dir) = self.time_sort {
            results.sort_by(|a, b| {
                let cmp = a.created_at.cmp(&b.created_at);
                match dir {
                    SortDirection::Asc => cmp,
                    SortDirection::Desc => cmp.reverse(),
                }
            });
        }
    }

    fn matches_all(&self, record: &Record) -> bool {
        self.wheres.iter().all(|w| matches_where(record, w))
    }

    /// Check whether a record passes the time filter using record timestamps
    /// directly (no index needed). Used as a post-filter when the timeseries
    /// index is unavailable or when time filters are combined with vector/graph.
    fn matches_time_filter(&self, record: &Record) -> bool {
        let tf = match &self.time_filter {
            Some(tf) => tf,
            None => return true,
        };

        let record_us = if tf.changed_only {
            record.updated_at.timestamp_micros()
        } else {
            record.created_at.timestamp_micros()
        };

        if let Some(after) = tf.after_us {
            if record_us < after {
                return false;
            }
        }
        if let Some(before) = tf.before_us {
            if record_us > before {
                return false;
            }
        }
        true
    }
}

/// Build a QueryProfile from step data.
fn build_profile(total_ms: f64, steps: Vec<ProfileStep>) -> QueryProfile {
    let bottleneck = steps
        .iter()
        .max_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap_or(std::cmp::Ordering::Equal))
        .map(|s| s.step.clone());
    QueryProfile {
        total_ms: (total_ms * 100.0).round() / 100.0,
        steps,
        bottleneck,
    }
}

/// Reciprocal Rank Fusion: combines multiple ranked lists into a single ranking.
///
/// Each document's RRF score is the sum of `1 / (k + rank_i)` across all lists
/// where the document appears.
///
/// Adaptive k (8b.2): Instead of fixed k=60, computes per-source k based on
/// score distribution spread. Tight spread → higher k (flatten rankings),
/// wide spread → lower k (preserve source ranking).
///
/// Documents absent from a list receive no contribution from that list (not
/// penalized), which naturally handles asymmetric result sets.
fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<(RecordId, f32)>],
    _base_k: usize,
) -> Vec<(RecordId, f32)> {
    let total: usize = ranked_lists.iter().map(|l| l.len()).sum();
    let mut scores: HashMap<RecordId, f32> = HashMap::with_capacity(total);

    for list in ranked_lists {
        // Compute adaptive k for this source based on score spread (8b.2)
        let k = adaptive_rrf_k(list);

        for (rank_0, (rid, _original_score)) in list.iter().enumerate() {
            let rrf_contrib = 1.0 / (k as f32 + rank_0 as f32 + 1.0);
            *scores.entry(rid.clone()).or_default() += rrf_contrib;
        }
    }

    let mut fused: Vec<(RecordId, f32)> = scores.into_iter().collect();

    // Sort descending by RRF score.
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Compute adaptive k for a ranked list based on score distribution spread (8b.2).
///
/// - Tight spread (low std dev) → higher k (60-120) — flatten rankings
/// - Wide spread (high std dev) → lower k (20-60) — preserve source ranking
fn adaptive_rrf_k(ranked_list: &[(RecordId, f32)]) -> usize {
    if ranked_list.len() < 2 {
        return 60; // fallback to standard k
    }

    let scores: Vec<f32> = ranked_list.iter().map(|(_, s)| *s).collect();
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let std_dev = variance.sqrt();

    // Map std_dev to k: low std_dev (< 0.05) → k=120, high std_dev (> 0.3) → k=20
    let k = if std_dev < 0.05 {
        120
    } else if std_dev > 0.3 {
        20
    } else {
        // Linear interpolation: 120 at 0.05, 20 at 0.3
        let t = (std_dev - 0.05) / 0.25;
        (120.0 - t * 100.0) as usize
    };

    k.clamp(20, 120)
}

/// Compute graph boost factor for re-ranking (8b.2).
///
/// Records with more graph connections get a score boost.
/// Formula: `1 + ln(1 + neighbor_count)` — logarithmic to avoid runaway boost.
pub fn graph_boost(neighbor_count: usize) -> f32 {
    1.0 + (1.0 + neighbor_count as f32).ln()
}

/// Fan out from seed records via graph traversal, deduplicating and resolving
/// full records. Stops early once `result_cap` unique endpoints are collected.
fn fan_out_traversal(
    gi: &dyn GraphIndex,
    storage: &Storage,
    seeds: &[Record],
    steps: &[TraversalStep],
    result_cap: usize,
) -> Result<Vec<Record>> {
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    'outer: for record in seeds {
        let ids = gi.traverse(record.id.clone(), steps)?;
        for id in ids {
            if seen.insert(id.clone()) {
                if let Some(full) = storage.get(&id)? {
                    results.push(full);
                    if results.len() >= result_cap {
                        break 'outer;
                    }
                }
            }
        }
    }
    Ok(results)
}

/// Check if a record satisfies a WHERE clause filter.
pub fn matches_where(record: &Record, clause: &WhereClause) -> bool {
    let field_val = record.data.get(&clause.field);

    match clause.op {
        Op::Eq => field_val == Some(&clause.value),
        Op::Ne => match field_val {
            Some(v) => v != &clause.value,
            None => true, // missing field is not equal to any value
        },
        Op::Gt => compare_op(field_val, &clause.value, |o| {
            o == std::cmp::Ordering::Greater
        }),
        Op::Lt => compare_op(field_val, &clause.value, |o| o == std::cmp::Ordering::Less),
        Op::Gte => compare_op(field_val, &clause.value, |o| o != std::cmp::Ordering::Less),
        Op::Lte => compare_op(field_val, &clause.value, |o| {
            o != std::cmp::Ordering::Greater
        }),
        Op::Contains => match (field_val, &clause.value) {
            (Some(Value::String(haystack)), Value::String(needle)) => {
                haystack.contains(needle.as_str())
            }
            (Some(Value::Array(arr)), val) => arr.contains(val),
            _ => false,
        },
    }
}

fn compare_op(
    field_val: Option<&Value>,
    target: &Value,
    predicate: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    field_val.is_some_and(|val| predicate(compare_json_values(Some(val), Some(target))))
}

/// Compare two optional JSON values, returning an ordering.
/// Mismatched types are ordered by type discriminant (never Equal).
pub fn compare_json_values(a: Option<&Value>, b: Option<&Value>) -> std::cmp::Ordering {
    match (a, b) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(va), Some(vb)) => {
            // Compare numbers.
            if let (Some(na), Some(nb)) = (va.as_f64(), vb.as_f64()) {
                return na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal);
            }
            // Compare strings.
            if let (Some(sa), Some(sb)) = (va.as_str(), vb.as_str()) {
                return sa.cmp(sb);
            }
            // Compare booleans.
            if let (Some(ba), Some(bb)) = (va.as_bool(), vb.as_bool()) {
                return ba.cmp(&bb);
            }
            // Mismatched types: order by type discriminant so they are
            // never reported as Equal (which would make Op::Eq match
            // across types, e.g. "hello" == 42).
            fn type_ord(v: &Value) -> u8 {
                match v {
                    Value::Null => 0,
                    Value::Bool(_) => 1,
                    Value::Number(_) => 2,
                    Value::String(_) => 3,
                    Value::Array(_) => 4,
                    Value::Object(_) => 5,
                }
            }
            type_ord(va).cmp(&type_ord(vb))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;
    use serde_json::json;

    fn setup() -> (Storage, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("query_test.axil");
        let storage = Storage::open(&path).unwrap();

        // Insert test data.
        for i in 0..5 {
            let r = Record::new(
                "items",
                json!({"name": format!("item_{i}"), "score": i * 10, "tags": ["rust", "db"]}),
            );
            storage.insert(&r).unwrap();
        }

        (storage, dir)
    }

    #[test]
    fn filter_eq() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("name", Op::Eq, json!("item_2"))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data["name"], "item_2");
    }

    #[test]
    fn filter_ne() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("name", Op::Ne, json!("item_0"))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn filter_gt() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("score", Op::Gt, json!(20))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 2); // score 30 and 40
    }

    #[test]
    fn filter_lt() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("score", Op::Lt, json!(20))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 2); // score 0 and 10
    }

    #[test]
    fn filter_gte() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("score", Op::Gte, json!(20))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 3); // score 20, 30, 40
    }

    #[test]
    fn filter_lte() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("score", Op::Lte, json!(20))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 3); // score 0, 10, 20
    }

    #[test]
    fn filter_contains_string() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("name", Op::Contains, json!("item_"))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn filter_contains_array() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("tags", Op::Contains, json!("rust"))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn combined_filters() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .where_field("score", Op::Gte, json!(10))
            .where_field("score", Op::Lt, json!(40))
            .exec()
            .unwrap();
        assert_eq!(results.len(), 3); // 10, 20, 30
    }

    #[test]
    fn pagination() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .limit(2)
            .offset(1)
            .exec()
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn order_by_asc() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .order_by("score", SortDirection::Asc)
            .exec()
            .unwrap();
        let scores: Vec<i64> = results
            .iter()
            .map(|r| r.data["score"].as_i64().unwrap())
            .collect();
        assert_eq!(scores, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn order_by_desc() {
        let (storage, _dir) = setup();
        let results = QueryBuilder::new(&storage, None, None)
            .table("items")
            .order_by("score", SortDirection::Desc)
            .exec()
            .unwrap();
        let scores: Vec<i64> = results
            .iter()
            .map(|r| r.data["score"].as_i64().unwrap())
            .collect();
        assert_eq!(scores, vec![40, 30, 20, 10, 0]);
    }

    #[test]
    fn table_required() {
        let (storage, _dir) = setup();
        let res = QueryBuilder::new(&storage, None, None).exec();
        assert!(res.is_err());
    }

    // ── RRF tests ──

    fn rid(s: &str) -> crate::record::RecordId {
        crate::record::RecordId(s.to_string())
    }

    #[test]
    fn rrf_single_list() {
        let list = vec![
            (rid("00000000-0000-0000-0000-000000000001"), 0.95),
            (rid("00000000-0000-0000-0000-000000000002"), 0.80),
            (rid("00000000-0000-0000-0000-000000000003"), 0.50),
        ];
        let result = reciprocal_rank_fusion(&[list], 60);
        assert_eq!(result.len(), 3);
        // First item should have highest score: 1/(60+1) = ~0.0164
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 > result[2].1);
        assert_eq!(result[0].0.as_str(), "00000000-0000-0000-0000-000000000001");
    }

    #[test]
    fn rrf_two_lists_overlap_boosts() {
        // Document A is #1 in both lists — should rank highest.
        // Document B is #2 in list 1 only. Document C is #2 in list 2 only.
        let list1 = vec![
            (rid("00000000-0000-0000-0000-00000000000a"), 0.9),
            (rid("00000000-0000-0000-0000-00000000000b"), 0.7),
        ];
        let list2 = vec![
            (rid("00000000-0000-0000-0000-00000000000a"), 0.8),
            (rid("00000000-0000-0000-0000-00000000000c"), 0.6),
        ];
        let result = reciprocal_rank_fusion(&[list1, list2], 60);
        assert_eq!(result.len(), 3);
        // A appears in both lists: 1/61 + 1/61 = 2/61 ≈ 0.0328
        // B and C each appear once at rank 2: 1/62 ≈ 0.0161
        assert_eq!(result[0].0.as_str(), "00000000-0000-0000-0000-00000000000a");
        let a_score = result[0].1;
        let b_or_c_score = result[1].1;
        // A's score should be roughly double B/C's score.
        assert!(a_score > b_or_c_score * 1.9);
    }

    #[test]
    fn rrf_empty_lists() {
        let result = reciprocal_rank_fusion(&[], 60);
        assert!(result.is_empty());

        let result = reciprocal_rank_fusion(&[vec![]], 60);
        assert!(result.is_empty());
    }

    #[test]
    fn rrf_three_lists_disjoint() {
        // Each list has one unique document — all should get equal scores.
        let list1 = vec![(rid("00000000-0000-0000-0000-000000000001"), 0.9)];
        let list2 = vec![(rid("00000000-0000-0000-0000-000000000002"), 0.8)];
        let list3 = vec![(rid("00000000-0000-0000-0000-000000000003"), 0.7)];
        let result = reciprocal_rank_fusion(&[list1, list2, list3], 60);
        assert_eq!(result.len(), 3);
        // All at rank 1 in their respective list: 1/61 each.
        let eps = 1e-6;
        assert!((result[0].1 - result[1].1).abs() < eps);
        assert!((result[1].1 - result[2].1).abs() < eps);
    }

    #[test]
    fn rrf_preserves_order_with_k() {
        // With k=0, rank differences are exaggerated.
        // With k=60, they're compressed. Both should preserve relative order.
        let list = vec![
            (rid("00000000-0000-0000-0000-000000000001"), 1.0),
            (rid("00000000-0000-0000-0000-000000000002"), 0.5),
            (rid("00000000-0000-0000-0000-000000000003"), 0.1),
        ];
        for k in [0, 1, 10, 60, 1000] {
            let result = reciprocal_rank_fusion(&[list.clone()], k);
            assert_eq!(result[0].0.as_str(), "00000000-0000-0000-0000-000000000001");
            assert_eq!(result[1].0.as_str(), "00000000-0000-0000-0000-000000000002");
            assert_eq!(result[2].0.as_str(), "00000000-0000-0000-0000-000000000003");
        }
    }
}
