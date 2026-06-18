//! Agent-optimized recall — search across index tables and return
//! compact, token-efficient results.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::Axil;

use crate::indexer::{TABLE_DEPS, TABLE_FILES, TABLE_MODULES, TABLE_PROJECT, TABLE_SYMBOLS};
use crate::proxy::{TABLE_CODE_PROXIES, TABLE_CODE_REFS_INDEX};
use crate::token;

/// Context depth level — controls how much detail is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextDepth {
    /// Project + module summaries only (~500 tokens).
    Shallow,
    /// + key files + recent changes (~2000 tokens).
    Medium,
    /// + symbols + full timeline (~5000 tokens).
    Deep,
}

impl ContextDepth {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "shallow" => Self::Shallow,
            "deep" => Self::Deep,
            _ => Self::Medium,
        }
    }

    pub fn default_max_tokens(self) -> usize {
        match self {
            Self::Shallow => 500,
            Self::Medium => 2000,
            Self::Deep => 5000,
        }
    }
}

/// Options for the `context` command.
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Maximum tokens to return.
    pub max_tokens: usize,
    /// Focus areas (module names to prioritize).
    pub focus: Vec<String>,
    /// If true, show what changed since last session.
    pub diff: bool,
    /// Context depth level.
    pub depth: ContextDepth,
    /// Task-focused context: combines vector + graph + rules + timeline.
    pub task: Option<String>,
    /// Project root for freshness checking. If set, includes freshness in output.
    pub project_root: Option<std::path::PathBuf>,
    /// Index config for freshness checking.
    pub index_config: Option<axil_core::IndexConfig>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            focus: Vec::new(),
            diff: false,
            depth: ContextDepth::Medium,
            task: None,
            project_root: None,
            index_config: None,
        }
    }
}

/// A single recall result.
///
/// Code-proxy fields (`proxy_id`, `symbol`, `line_start`/`line_end`,
/// `canonical_id`, `breadcrumb`, `why`, `source_record`) are populated only
/// for `_idx_code_proxies` hits and remain `None` for other index tables.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallResult {
    pub id: String,
    pub score: f32,
    pub tokens: usize,
    pub source: String, // "project", "module", "file", "symbol", "proxy"
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_end: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breadcrumb: Option<String>,
    /// Short, deterministic reason this hit ranked. See `why` constants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_record: Option<String>,
}

/// Recall + pointer-attached related memories in a single call.
///
/// Returns the structural recall output, plus any memories whose
/// `code_refs` point at one of the returned code proxies. Related
/// memories are ranked by their proxy hits' position; ties broken by
/// record `updated_at`.
pub fn recall_with_related(
    db: &Axil,
    query: &str,
    top_k: usize,
    related_limit: usize,
) -> axil_core::Result<RecallWithRelated> {
    let primary = recall(db, query, top_k)?;
    let proxy_hits: Vec<RecallResult> = primary
        .iter()
        .filter(|r| r.source == "proxy")
        .cloned()
        .collect();
    let related = related_memories_for_proxies(db, &proxy_hits, related_limit)?;
    let graph_neighbors =
        graph_neighbors_for_proxies(db, &proxy_hits, related_limit).unwrap_or_default();
    Ok(RecallWithRelated {
        primary,
        related,
        graph_neighbors,
    })
}

/// Output of [`recall_with_related`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallWithRelated {
    /// Primary recall results (mix of code proxies and other index hits).
    pub primary: Vec<RecallResult>,
    /// Memories whose `code_refs` point at any of the proxy hits in
    /// `primary`.
    pub related: Vec<RecallResult>,
    /// Graph neighbors of matched proxies (Phase 13b.7 / 13b.8 P1).
    /// Empty when no graph plugin or SCIP edges are present.
    #[serde(default)]
    pub graph_neighbors: Vec<RecallResult>,
}

/// Reason strings used in `RecallResult::why` for code-proxy hits. Kept
/// short and deterministic — full score breakdowns belong in
/// `explain-code-hit`.
pub const WHY_VECTOR: &str = "matched via vector proxy";
pub const WHY_FTS: &str = "matched via full-text proxy";
pub const WHY_PATH_BOOST: &str = "matched symbol/path/breadcrumb";
pub const WHY_POINTER_ATTACHED: &str = "memory attached to matched code proxy";
pub const WHY_GRAPH_NEIGHBOR: &str = "graph neighbor of matched proxy";

/// Expand a list of proxy hits with up to `limit` graph neighbors via
/// Phase 13 SCIP edges (`calls`/`references`/`implements`/`type_of`).
///
/// Strategy:
/// 1. For each direct proxy hit with a `canonical_id`, look up the
///    `_entities` row whose `canonical_id` matches.
/// 2. Walk outgoing graph edges of those entities and collect their
///    target entity rows.
/// 3. For each target entity, find the proxy whose `canonical_id` matches
///    and emit it as a neighbor hit (deduped against the input set and
///    capped at `limit`).
///
/// No-op when the graph plugin is missing or no SCIP canonical ids exist.
/// Proxy hits without a canonical id pass through (they would otherwise
/// require a per-symbol lookup that is not yet wired).
pub fn graph_neighbors_for_proxies(
    db: &Axil,
    direct: &[RecallResult],
    limit: usize,
) -> axil_core::Result<Vec<RecallResult>> {
    if direct.is_empty() || limit == 0 || !db.has_graph_index() {
        return Ok(Vec::new());
    }

    // Map canonical_id -> entity record id. Both maps are optional —
    // when no SCIP data is present they stay empty and the SCIP-bridge
    // pass below becomes a no-op while the proxy-edge pass still runs.
    let entities = db.list("_entities").unwrap_or_default();
    let mut canonical_to_entity: std::collections::HashMap<String, axil_core::RecordId> =
        std::collections::HashMap::new();
    for e in &entities {
        if let Some(cid) = e.data.get("canonical_id").and_then(|v| v.as_str()) {
            canonical_to_entity.insert(cid.to_string(), e.id.clone());
        }
    }
    let mut canonical_to_proxy: std::collections::HashMap<String, axil_core::Record> =
        std::collections::HashMap::new();
    for r in db.list(TABLE_CODE_PROXIES).unwrap_or_default() {
        if let Some(cid) = r.data.get("canonical_id").and_then(|v| v.as_str()) {
            canonical_to_proxy.insert(cid.to_string(), r);
        }
    }

    let direct_proxy_ids: std::collections::HashSet<String> = direct
        .iter()
        .filter(|r| r.source == "proxy")
        .map(|r| r.id.clone())
        .collect();

    let entity_edge_kinds = ["calls", "references", "implements", "type_of"];
    // Edges that hang directly off proxy records (no entity bridge needed).
    let proxy_edge_kinds = ["same_file", "tests"];

    let mut out: Vec<RecallResult> = Vec::new();
    let mut seen: std::collections::HashSet<String> = direct_proxy_ids.clone();

    'outer: for hit in direct {
        // ── 1. Phase 13 SCIP entity edges (canonical_id bridge) ─────
        if let Some(canonical) = &hit.canonical_id {
            if let Some(entity_id) = canonical_to_entity.get(canonical).cloned() {
                for edge_kind in &entity_edge_kinds {
                    let edges =
                        match db.edges(&entity_id, Some(edge_kind), axil_core::Direction::Out) {
                            Ok(e) => e,
                            Err(_) => continue,
                        };
                    for edge in edges {
                        let target_entity = match db.get(&edge.to)? {
                            Some(r) => r,
                            None => continue,
                        };
                        let target_canonical = match target_entity
                            .data
                            .get("canonical_id")
                            .and_then(|v| v.as_str())
                        {
                            Some(c) => c.to_string(),
                            None => continue,
                        };
                        let neighbor_proxy = match canonical_to_proxy.get(&target_canonical) {
                            Some(p) => p,
                            None => continue,
                        };
                        let id_str = neighbor_proxy.id.to_string();
                        if !seen.insert(id_str.clone()) {
                            continue;
                        }
                        if let Some(mut rr) = record_to_recall_result(neighbor_proxy, 0.0) {
                            rr.why = Some(format!("{} (via {edge_kind})", WHY_GRAPH_NEIGHBOR));
                            out.push(rr);
                            if out.len() >= limit {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        // ── 2. Proxy-to-proxy edges (same_file / tests) ─────────────
        let proxy_id = match axil_core::RecordId::from_string(&hit.id) {
            Ok(rid) => rid,
            Err(_) => continue,
        };
        for edge_kind in &proxy_edge_kinds {
            for direction in [axil_core::Direction::Out, axil_core::Direction::In] {
                let edges = match db.edges(&proxy_id, Some(edge_kind), direction) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                for edge in edges {
                    let target_id = if direction == axil_core::Direction::Out {
                        edge.to
                    } else {
                        edge.from
                    };
                    let neighbor = match db.get(&target_id)? {
                        Some(r) => r,
                        None => continue,
                    };
                    if neighbor.table != TABLE_CODE_PROXIES {
                        continue;
                    }
                    let id_str = neighbor.id.to_string();
                    if !seen.insert(id_str.clone()) {
                        continue;
                    }
                    if let Some(mut rr) = record_to_recall_result(&neighbor, 0.0) {
                        rr.why = Some(format!("{} (via {edge_kind})", WHY_GRAPH_NEIGHBOR));
                        out.push(rr);
                        if out.len() >= limit {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Find memories whose `code_refs` point at any of the given proxy hits.
///
/// When recall returns a code proxy, the agent also wants prior
/// decisions/errors/fixes stored against that same anchor — they survive
/// line movement because matching prefers `proxy_id`/`canonical_id` over
/// path/line.
///
/// Resolves matches via the `_idx_code_refs` reverse index: one
/// `db.list("_idx_code_refs")` plus one `db.get` per matched record_id.
/// Resolves matches via `_idx_code_refs` first; tops up from a
/// full-table walk when the index returns fewer than `limit` hits, so
/// memories from older DBs (no reverse-index rows) and from records
/// where best-effort sync skipped a row still surface.
pub fn related_memories_for_proxies(
    db: &Axil,
    proxy_results: &[RecallResult],
    limit: usize,
) -> axil_core::Result<Vec<RecallResult>> {
    if proxy_results.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let mut wanted_keys: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(proxy_results.len() * 4);
    for p in proxy_results {
        for key in axil_core::code_refs::proxy_match_keys(
            p.proxy_id.as_deref(),
            p.canonical_id.as_deref(),
            p.path.as_deref(),
            p.symbol.as_deref(),
        ) {
            wanted_keys.insert(key);
        }
    }

    let index_rows = db.list(TABLE_CODE_REFS_INDEX).unwrap_or_default();
    let mut hits: Vec<RecallResult> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Walk all matching index rows; keep collecting until `hits` reaches
    // `limit`. Stale/tombstoned rows that fail `db.get` don't consume a
    // slot, so under-return only happens when nothing valid matches.
    for row in &index_rows {
        if hits.len() >= limit {
            break;
        }
        let key = match row.data.get("key").and_then(|v| v.as_str()) {
            Some(k) => k,
            None => continue,
        };
        if !wanted_keys.contains(key) {
            continue;
        }
        let rid_str = match row.data.get("record_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if !seen_ids.insert(rid_str.clone()) {
            continue;
        }
        let parsed = match axil_core::RecordId::from_string(&rid_str) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let record = match db.get(&parsed) {
            Ok(Some(r)) => r,
            _ => continue,
        };
        let src_table = row
            .data
            .get("src_table")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        hits.push(record_to_pointer_hit(rid_str, src_table, &record));
    }

    // Best-effort sync can leave records unindexed (errors, pre-hook DBs,
    // mixed old/new data). Top up from the fallback walk so old memories
    // don't silently disappear once the reverse index gains its first row.
    if hits.len() < limit {
        let remaining = limit - hits.len();
        let extras = related_memories_fallback_scan(db, proxy_results, remaining + seen_ids.len())?;
        for hit in extras {
            if hits.len() >= limit {
                break;
            }
            if seen_ids.insert(hit.id.clone()) {
                hits.push(hit);
            }
        }
    }
    Ok(hits)
}

fn record_to_pointer_hit(
    id: String,
    src_table: String,
    record: &axil_core::Record,
) -> RecallResult {
    let summary = record
        .data
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    RecallResult {
        id,
        score: 0.0,
        tokens: 0,
        source: src_table,
        summary,
        why: Some(WHY_POINTER_ATTACHED.to_string()),
        ..Default::default()
    }
}

/// Walk every non-internal table and inspect `data.code_refs`. Worst
/// case is `O(records * refs)` per recall, bounded by memory volume.
/// Called as a top-up when the reverse-index path returns fewer than
/// `limit` hits — covers mixed old/new DBs where some memories have
/// `code_refs` in their data but no `_idx_code_refs` row, and pure
/// best-effort sync gaps.
fn related_memories_fallback_scan(
    db: &Axil,
    proxy_results: &[RecallResult],
    limit: usize,
) -> axil_core::Result<Vec<RecallResult>> {
    let mut proxy_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut canonical_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut path_symbol: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    let mut path_only: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in proxy_results {
        if let Some(pid) = &p.proxy_id {
            proxy_ids.insert(pid.clone());
        }
        if let Some(cid) = &p.canonical_id {
            canonical_ids.insert(cid.clone());
        }
        match (&p.path, &p.symbol) {
            (Some(pa), Some(sy)) => {
                path_symbol.insert((pa.clone(), sy.clone()));
            }
            (Some(pa), None) => {
                path_only.insert(pa.clone());
            }
            _ => {}
        }
    }

    let mut hits: Vec<RecallResult> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for table in db.tables().unwrap_or_default() {
        if table.starts_with('_') {
            continue;
        }
        let records = match db.list(&table) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for record in &records {
            let refs = match record.data.get("code_refs").and_then(|v| v.as_array()) {
                Some(a) => a,
                None => continue,
            };
            let mut matched = false;
            for r in refs {
                let pid = r.get("proxy_id").and_then(|v| v.as_str());
                let cid = r.get("canonical_id").and_then(|v| v.as_str());
                let pa = r.get("path").and_then(|v| v.as_str());
                let sy = r.get("symbol").and_then(|v| v.as_str());
                if pid.map(|p| proxy_ids.contains(p)).unwrap_or(false)
                    || cid.map(|c| canonical_ids.contains(c)).unwrap_or(false)
                {
                    matched = true;
                    break;
                }
                if let (Some(pa), Some(sy)) = (pa, sy) {
                    if path_symbol.contains(&(pa.to_string(), sy.to_string())) {
                        matched = true;
                        break;
                    }
                }
                if sy.is_none() {
                    if let Some(pa) = pa {
                        if path_only.contains(pa) {
                            matched = true;
                            break;
                        }
                    }
                }
            }
            if !matched {
                continue;
            }
            let id_str = record.id.to_string();
            if !seen_ids.insert(id_str.clone()) {
                continue;
            }
            let summary = record
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            hits.push(RecallResult {
                id: id_str,
                score: 0.0,
                tokens: 0,
                source: table.clone(),
                summary,
                why: Some(WHY_POINTER_ATTACHED.to_string()),
                ..Default::default()
            });
            if hits.len() >= limit {
                return Ok(hits);
            }
        }
    }
    Ok(hits)
}

/// Search across all index tables for the given query.
///
/// When `_idx_code_proxies` records are present, vector and FTS hits are
/// RRF-fused (instead of vector-first early-return) so a strong FTS-only
/// match on `path`/`symbol`/`breadcrumb` still surfaces. Exact
/// path/symbol/breadcrumb matches get a deterministic boost via
/// `boost_proxy_path_match`.
pub fn recall(db: &Axil, query: &str, top_k: usize) -> axil_core::Result<Vec<RecallResult>> {
    let mut fused: std::collections::HashMap<String, RecallResult> =
        std::collections::HashMap::new();
    let k_rrf = 60.0; // standard reciprocal-rank-fusion constant

    // Pool larger than top_k so the right proxy is present in the fused set
    // to be boosted — on a large corpus FTS single-term matches can fill the
    // first dozen slots, pushing the name-matching proxy past top_k*3.
    let pool = (top_k * 5).max(40);
    let mut had_any = false;

    // Query-side expansion: the original query plus a few derived forms
    // (concatenated-identifier + synonym variants). The concatenated form is
    // what makes "url resolver" reach `URLResolver` via vector — the spaced
    // form is dominated by common-term FTS matches. Expansions are fused with
    // a lower weight so they broaden recall without overriding the original.
    let expansions = expand_query(query);
    for (qi, q) in expansions.iter().enumerate() {
        let weight = if qi == 0 { 1.0 } else { 0.6 };

        if let Ok(vector_results) = db.similar_to(q, pool) {
            had_any = true;
            for (rank, item) in vector_results.iter().enumerate() {
                let mut r = match record_to_recall_result(&item.0, item.1) {
                    Some(r) => r,
                    None => continue,
                };
                if r.source == "proxy" {
                    r.why.get_or_insert_with(|| WHY_VECTOR.to_string());
                }
                let rr_score = weight / (k_rrf + rank as f32 + 1.0);
                let entry = fused.entry(r.id.clone()).or_insert_with(|| zero_scored(&r));
                entry.score += rr_score;
            }
        }

        if let Ok(fts_results) = db.search_text(q, pool) {
            had_any = true;
            for (rank, item) in fts_results.iter().enumerate() {
                let mut r = match record_to_recall_result(&item.0, item.1) {
                    Some(r) => r,
                    None => continue,
                };
                if r.source == "proxy" {
                    r.why.get_or_insert_with(|| WHY_FTS.to_string());
                }
                let rr_score = weight / (k_rrf + rank as f32 + 1.0);
                let entry = fused.entry(r.id.clone()).or_insert_with(|| zero_scored(&r));
                entry.score += rr_score;
            }
        }
    }

    if had_any && !fused.is_empty() {
        // Boost path/symbol/breadcrumb matches on proxies already fused in
        // by vector + FTS. FTS indexes those fields at store time, so an
        // exact-equality match always reaches this point as a fused entry
        // — no need to walk the full proxy table.
        //
        // Match rules — conservative to avoid short queries ("rs", "src")
        // inflating every proxy:
        //   * Exact case-insensitive equality on `path`, `symbol`, or
        //     `canonical_id` always boosts.
        //   * Substring/word match only when the trimmed query is at
        //     least MIN_BOOST_LEN chars AND the matched field is a
        //     standalone token, not a coincidental substring inside a
        //     longer identifier.
        const MIN_BOOST_LEN: usize = 4;
        let q_trim = query.trim();
        let q_lower = q_trim.to_lowercase();
        let allow_substr = q_lower.len() >= MIN_BOOST_LEN;

        // Per-term identity boost: reward proxies whose symbol/path/breadcrumb
        // *identifiers* contain the query's content words. CamelCase-aware, so
        // a query "url resolver" matches `URLResolver` and `urls/resolvers.py`.
        // This is what lifts the name-matching proxy above files that merely
        // repeat one common FTS term ("url" / "migration") across the corpus.
        let q_terms: Vec<String> = identifier_word_set(q_trim)
            .into_iter()
            .filter(|t| t.len() >= 3 && !is_stopword(t))
            .collect();

        for entry in fused.values_mut() {
            if entry.source != "proxy" {
                continue;
            }
            let path = entry.path.as_deref().unwrap_or("");
            let symbol = entry.symbol.as_deref().unwrap_or("");
            let breadcrumb = entry.breadcrumb.as_deref().unwrap_or("");
            let canonical = entry.canonical_id.as_deref().unwrap_or("");

            let exact = path.eq_ignore_ascii_case(q_trim)
                || symbol.eq_ignore_ascii_case(q_trim)
                || canonical.eq_ignore_ascii_case(q_trim);
            if exact {
                entry.score += 0.10;
                entry.why.get_or_insert_with(|| WHY_PATH_BOOST.to_string());
            }

            // Fraction of query content-words present in this proxy's identity.
            if !q_terms.is_empty() {
                let mut hay: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                hay.extend(identifier_word_set(symbol));
                hay.extend(identifier_word_set(path));
                hay.extend(identifier_word_set(breadcrumb));
                let matched = q_terms.iter().filter(|t| hay.contains(*t)).count();
                if matched > 0 {
                    // 0.045/term, capped — two name-matched terms (~0.09) clear
                    // the ~0.016/rank RRF spread of single-term FTS hits.
                    entry.score += (0.045 * matched as f32).min(0.18);
                    entry.why.get_or_insert_with(|| WHY_PATH_BOOST.to_string());
                }
            }

            // Legacy whole-query token match (kept for short single-token
            // queries that identifier_word_set filters out, e.g. "rs").
            let token = allow_substr
                && !exact
                && (token_match(path, &q_lower)
                    || token_match(symbol, &q_lower)
                    || token_match(breadcrumb, &q_lower));
            if token {
                entry.score += 0.05;
                entry.why.get_or_insert_with(|| WHY_PATH_BOOST.to_string());
            }
        }

        let mut sorted: Vec<RecallResult> = fused.into_values().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let filtered: Vec<RecallResult> = sorted
            .into_iter()
            .filter(|r| is_index_table(&r.source))
            .take(top_k)
            .collect();
        let deduped = dedupe_proxy_pointers(filtered, top_k);
        if !deduped.is_empty() {
            return Ok(deduped);
        }
    }

    // Fallback text matching across all index tables (also covers proxies).
    let mut text_match = recall_text_match(db, query, top_k)?;
    text_match = dedupe_proxy_pointers(text_match, top_k);
    Ok(text_match)
}

/// Clone a recall result with its score zeroed. The fused score must be
/// **pure reciprocal-rank fusion** (+ the identity boost) — seeding it with
/// the backend's raw similarity (cosine ~0.7 vs BM25 ~0.9–15, different
/// scales) lets the larger-magnitude backend dominate and drowns both the
/// RRF rank signal and the boost. Starting from 0 makes rank + boost decide.
fn zero_scored(r: &RecallResult) -> RecallResult {
    let mut e = r.clone();
    e.score = 0.0;
    e
}

/// Query-side expansion. Returns the original query first (weight 1.0 at the
/// call site) followed by up to a few derived forms that broaden recall for
/// the cases plain fusion misses:
///
/// * **Concatenated identifier** — content words joined (`"url resolver"` →
///   `"urlresolver"`). The vector embedder matches this to the CamelCase
///   symbol (`URLResolver`) far better than the spaced form, whose FTS signal
///   is swamped by the common term ("url") across a large corpus.
/// * **Synonym / abbreviation variant** — one extra query with common code
///   vocabulary bridged (`config`↔`configuration`, `auth`↔`authentication`,
///   `db`↔`database`, …) so a query worded differently from the code still
///   matches.
///
/// Bounded to keep recall latency in check (each variant is one more
/// vector+FTS pair). Single-word / already-identifier queries return just the
/// original.
fn expand_query(query: &str) -> Vec<String> {
    let mut out = vec![query.to_string()];
    let words: Vec<String> = identifier_word_set(query)
        .into_iter()
        .filter(|t| t.len() >= 3 && !is_stopword(t))
        .collect();
    if words.len() < 2 {
        return out;
    }

    // Concatenated-identifier form (cap the word count so we don't build a
    // nonsense token from a long sentence — symbol names are short).
    if words.len() <= 4 {
        let joined: String = words.concat();
        if joined.len() >= 5 && !out.iter().any(|q| q.eq_ignore_ascii_case(&joined)) {
            out.push(joined);
        }
    }

    // Synonym-augmented variant: append synonyms of content words to the
    // original query (one extra sub-query, deduped).
    let mut syns: Vec<&'static str> = Vec::new();
    for w in &words {
        if let Some(list) = concept_synonyms(w) {
            for s in list {
                if !words.iter().any(|w2| w2 == s) && !syns.contains(s) {
                    syns.push(s);
                }
            }
        }
    }
    if !syns.is_empty() {
        let variant = format!("{query} {}", syns.join(" "));
        out.push(variant);
    }

    out.truncate(4);
    out
}

/// Small, hand-curated bridge for common code-vocabulary mismatches. Kept
/// conservative — only well-established abbreviation/expansion pairs, so the
/// synonym variant adds signal without dragging in unrelated files.
fn concept_synonyms(word: &str) -> Option<&'static [&'static str]> {
    Some(match word {
        "config" | "configuration" => &["config", "configuration", "settings"],
        "settings" => &["settings", "config", "configuration"],
        "auth" | "authentication" => &["auth", "authentication"],
        "authorization" | "authorize" => &["authorization", "permission"],
        "db" | "database" => &["db", "database"],
        "url" | "uri" => &["url", "uri", "route"],
        "middleware" => &["middleware", "handler"],
        "init" | "initialize" => &["init", "initialize"],
        "repo" | "repository" => &["repo", "repository"],
        "param" | "parameter" => &["param", "parameter", "argument"],
        "ctx" | "context" => &["ctx", "context"],
        "connection" | "conn" => &["connection", "conn"],
        "async" | "asynchronous" => &["async", "asynchronous"],
        _ => return None,
    })
}

/// Split a string into lowercase identifier words, breaking on
/// non-alphanumeric runs AND CamelCase / acronym boundaries. Mirrors the
/// FTS code tokenizer so the recall boost matches the same way the index
/// does: `"URLResolver"` → `["url","resolver"]`,
/// `"urls/resolvers.py"` → `["urls","resolvers","py"]`,
/// `"get_response"` → `["get","response"]`.
fn identifier_word_set(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !cur.is_empty() {
            let prev = chars[i - 1];
            let boundary = (prev.is_lowercase() && c.is_uppercase())
                || (prev.is_uppercase()
                    && c.is_uppercase()
                    && i + 1 < chars.len()
                    && chars[i + 1].is_lowercase())
                || (prev.is_numeric() != c.is_numeric());
            if boundary {
                out.push(std::mem::take(&mut cur));
            }
        }
        for lc in c.to_lowercase() {
            cur.push(lc);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Common English / query filler words that should not drive the identity
/// boost (they appear in many proxies and carry no locating signal).
fn is_stopword(t: &str) -> bool {
    matches!(
        t,
        "the" | "and" | "for" | "that" | "this" | "with" | "into" | "from"
            | "where" | "what" | "which" | "how" | "does" | "find" | "implement"
            | "implemented" | "implements" | "code" | "class" | "function"
            | "method" | "actual" | "core" | "against" | "run" | "runs" | "use"
            | "used" | "uses" | "set" | "get" | "are" | "was" | "its" | "out"
    )
}

/// Whether `q_lower` (already lowercased) appears in `haystack` as a
/// standalone token rather than a coincidental substring. Tokens are
/// split on ASCII non-alphanumeric runs (`/`, `:`, `_`, `>`, ` `, ...)
/// so `recall` matches `db::recall()` and `axildb > db.rs > recall` but
/// not `recall_with_feedback` unless the full-token form is searched.
fn token_match(haystack: &str, q_lower: &str) -> bool {
    if haystack.is_empty() || q_lower.is_empty() {
        return false;
    }
    let lower = haystack.to_lowercase();
    if lower == q_lower {
        return true;
    }
    lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|tok| tok == q_lower || tok.split('_').any(|piece| piece == q_lower))
}

/// Deduplicate proxy hits that point at the same `(path, symbol)` or the
/// same `proxy_id`. Keeps the highest-scoring entry. Non-proxy results pass
/// through unchanged.
fn dedupe_proxy_pointers(results: Vec<RecallResult>, top_k: usize) -> Vec<RecallResult> {
    let mut seen_proxy_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_path_symbol: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        if r.source == "proxy" {
            if let Some(pid) = &r.proxy_id {
                if !seen_proxy_ids.insert(pid.clone()) {
                    continue;
                }
            }
            let key = (
                r.path.clone().unwrap_or_default(),
                r.symbol.clone().unwrap_or_default(),
            );
            if !key.0.is_empty() || !key.1.is_empty() {
                if !seen_path_symbol.insert(key) {
                    continue;
                }
            }
        }
        out.push(r);
        if out.len() >= top_k {
            break;
        }
    }
    out
}

/// Check if a source tag corresponds to an index table.
fn is_index_table(source: &str) -> bool {
    matches!(
        source,
        "project"
            | "module"
            | "file"
            | "symbol"
            | "dep"
            | "proxy"
            | "_idx_project"
            | "_idx_files"
            | "_idx_modules"
            | "_idx_symbols"
            | "_idx_deps"
            | "_idx_code_proxies"
    )
}

/// Convert a single (Record, score) into a `RecallResult`. Returns `None`
/// when the record is not from an index table we know about.
fn record_to_recall_result(record: &axil_core::Record, score: f32) -> Option<RecallResult> {
    let id = record.id.to_string();
    let tokens = record
        .data
        .get("tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let result = match record.table.as_str() {
        "_idx_project" => RecallResult {
            id,
            score,
            tokens,
            source: "project".to_string(),
            summary: record
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            kind: Some("project".to_string()),
            ..Default::default()
        },
        "_idx_modules" => RecallResult {
            id,
            score,
            tokens,
            source: "module".to_string(),
            summary: record
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            path: record
                .data
                .get("path")
                .and_then(|v| v.as_str())
                .map(String::from),
            kind: Some("module".to_string()),
            ..Default::default()
        },
        "_idx_files" => RecallResult {
            id,
            score,
            tokens,
            source: "file".to_string(),
            summary: record
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            path: record
                .data
                .get("path")
                .and_then(|v| v.as_str())
                .map(String::from),
            kind: record
                .data
                .get("language")
                .and_then(|v| v.as_str())
                .map(String::from),
            ..Default::default()
        },
        "_idx_symbols" => RecallResult {
            id,
            score,
            tokens,
            source: "symbol".to_string(),
            summary: format!(
                "{} {} in {}",
                record
                    .data
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                record
                    .data
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .or_else(|| record.data.get("name").and_then(|v| v.as_str()))
                    .unwrap_or(""),
                record
                    .data
                    .get("file")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
            ),
            path: record
                .data
                .get("file")
                .and_then(|v| v.as_str())
                .map(String::from),
            kind: record
                .data
                .get("kind")
                .and_then(|v| v.as_str())
                .map(String::from),
            symbol: record
                .data
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from),
            line_start: record
                .data
                .get("line")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            ..Default::default()
        },
        "_idx_deps" => RecallResult {
            id,
            score,
            tokens,
            source: "dep".to_string(),
            summary: format!(
                "{} {} — {}",
                record
                    .data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                record
                    .data
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                record
                    .data
                    .get("purpose")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
            ),
            kind: Some("dep".to_string()),
            ..Default::default()
        },
        "_idx_code_proxies" => {
            // Carry pointer fields out so the agent sees
            // `path:line symbol — why` without re-reading the proxy.
            let symbol = record.data.get("symbol").and_then(|v| v.as_str());
            let kind = record
                .data
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let path = record
                .data
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let line = record.data.get("line_start").and_then(|v| v.as_u64());
            let summary = match (symbol, line) {
                (Some(s), Some(l)) => format!("{kind} {s} ({path}:{l})"),
                (Some(s), None) => format!("{kind} {s} ({path})"),
                (None, Some(l)) => format!("{kind} ({path}:{l})"),
                (None, None) => format!("{kind} ({path})"),
            };
            RecallResult {
                id,
                score,
                tokens,
                source: "proxy".to_string(),
                summary,
                path: Some(path.to_string()),
                kind: Some(kind.to_string()),
                proxy_id: record
                    .data
                    .get("proxy_id")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                symbol: symbol.map(String::from),
                line_start: line.map(|v| v as usize),
                line_end: record
                    .data
                    .get("line_end")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize),
                canonical_id: record
                    .data
                    .get("canonical_id")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                breadcrumb: record
                    .data
                    .get("breadcrumb")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                why: None,
                source_record: record
                    .data
                    .get("source_record")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            }
        }
        _ => return None,
    };
    Some(result)
}

/// Legacy helper kept so older code paths can still convert vector/FTS
/// result batches to recall results in one shot.
#[allow(dead_code)]
fn records_to_recall_results(records: &[(axil_core::Record, f32)]) -> Vec<RecallResult> {
    records
        .iter()
        .filter_map(|(r, s)| record_to_recall_result(r, *s))
        .collect()
}

/// Fallback: search by text matching across all index tables.
fn recall_text_match(db: &Axil, query: &str, top_k: usize) -> axil_core::Result<Vec<RecallResult>> {
    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    let mut results = Vec::new();

    let scan = |table: &str, results: &mut Vec<RecallResult>| -> axil_core::Result<()> {
        for record in &db.list(table)? {
            let score = score_record(&record.data, &query_terms);
            if score > 0.0 {
                if let Some(rr) = record_to_recall_result(record, score) {
                    results.push(rr);
                }
            }
        }
        Ok(())
    };

    scan(TABLE_PROJECT, &mut results)?;
    scan(TABLE_MODULES, &mut results)?;
    scan(TABLE_FILES, &mut results)?;
    scan(TABLE_SYMBOLS, &mut results)?;
    scan(TABLE_CODE_PROXIES, &mut results)?;

    // Sort by score (descending)
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    Ok(results)
}

/// Generate project context within a token budget.
///
/// Returns the project overview plus top-ranked module summaries,
/// respecting the token limit.
pub fn context(db: &Axil, opts: &ContextOptions) -> axil_core::Result<Value> {
    // If --task is provided, delegate to task-focused context.
    if let Some(ref task) = opts.task {
        return task_context(db, task, opts);
    }

    let mut tokens_used = 0usize;
    let max = opts.max_tokens;
    let mut sources: serde_json::Map<String, Value> = serde_json::Map::new();

    // Always start with project overview
    let projects = db.list(TABLE_PROJECT)?;
    let project_summary = if let Some(proj) = projects.first() {
        let t = token::estimate_json_tokens(&proj.data);
        tokens_used += t;
        sources.insert(
            "project_overview".to_string(),
            json!({"tokens": t, "type": "key-value"}),
        );
        proj.data.clone()
    } else {
        return Ok(json!({"error": "no index found — run `axil index .` first"}));
    };

    // Add modules, prioritizing focused areas
    let modules = db.list(TABLE_MODULES)?;
    let mut module_summaries: Vec<Value> = Vec::new();
    let module_start = tokens_used;

    let mut sorted_modules: Vec<&axil_core::Record> = modules.iter().collect();
    if !opts.focus.is_empty() {
        sorted_modules.sort_by(|a, b| {
            let a_focused = opts.focus.iter().any(|f| {
                a.data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|n| n.contains(f.as_str()))
                    .unwrap_or(false)
            });
            let b_focused = opts.focus.iter().any(|f| {
                b.data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|n| n.contains(f.as_str()))
                    .unwrap_or(false)
            });
            b_focused.cmp(&a_focused)
        });
    }

    for module in &sorted_modules {
        let summary = module
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let name = module
            .data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let files = module
            .data
            .get("files")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        let entry = format!("{name}: {summary} ({files} files)");
        let entry_tokens = token::estimate_tokens(&entry);

        if tokens_used + entry_tokens > max {
            break;
        }
        tokens_used += entry_tokens;
        module_summaries.push(json!(entry));
    }
    let module_type = if opts.focus.is_empty() {
        "key-value"
    } else {
        "key-value+focus"
    };
    sources.insert(
        "modules".to_string(),
        json!({"tokens": tokens_used - module_start, "type": module_type}),
    );

    // Medium/Deep: include key files
    let mut key_files: Vec<Value> = Vec::new();
    if opts.depth != ContextDepth::Shallow {
        let files_start = tokens_used;
        let files = db.list(TABLE_FILES)?;

        if opts.diff {
            let mut file_entries: Vec<(&axil_core::Record, &str)> = files
                .iter()
                .filter_map(|r| {
                    let modified = r.data.get("last_modified").and_then(|v| v.as_str())?;
                    Some((r, modified))
                })
                .collect();
            file_entries.sort_by(|a, b| b.1.cmp(a.1));

            for (record, _modified) in &file_entries {
                let path = record
                    .data
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let summary = record
                    .data
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let entry = format!("{path}: {summary}");
                let entry_tokens = token::estimate_tokens(&entry);

                if tokens_used + entry_tokens > max {
                    break;
                }
                tokens_used += entry_tokens;
                key_files.push(json!(entry));
            }
        }
        if tokens_used > files_start {
            sources.insert(
                "recent_changes".to_string(),
                json!({"tokens": tokens_used - files_start, "type": "time-series"}),
            );
        }
    }

    // Deep: include symbols
    let mut symbols: Vec<Value> = Vec::new();
    if opts.depth == ContextDepth::Deep {
        let sym_start = tokens_used;
        let all_symbols = db.list(TABLE_SYMBOLS)?;
        for sym in &all_symbols {
            let name = sym.data.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let kind = sym.data.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let file = sym.data.get("file").and_then(|v| v.as_str()).unwrap_or("");
            let entry = format!("{kind} {name} ({file})");
            let entry_tokens = token::estimate_tokens(&entry);

            if tokens_used + entry_tokens > max {
                break;
            }
            tokens_used += entry_tokens;
            symbols.push(json!(entry));
        }
        if tokens_used > sym_start {
            sources.insert(
                "symbols".to_string(),
                json!({"tokens": tokens_used - sym_start, "type": "key-value"}),
            );
        }
    }

    // Build context response
    let mut result = json!({
        "project": project_summary.get("summary").and_then(|v| v.as_str()).unwrap_or(""),
        "type": project_summary.get("type").and_then(|v| v.as_str()).unwrap_or(""),
        "tech_stack": project_summary.get("tech_stack").unwrap_or(&json!([])),
        "modules": module_summaries,
        "conventions": project_summary.get("conventions").unwrap_or(&json!({})),
        "sources": sources,
        "tokens_used": tokens_used,
    });

    if !key_files.is_empty() {
        result["recent_changes"] = json!(key_files);
    }
    if !symbols.is_empty() {
        result["symbols"] = json!(symbols);
    }

    // Include freshness if project root is provided
    if let (Some(root), Some(config)) = (&opts.project_root, &opts.index_config) {
        let report = crate::freshness::check_freshness(db, root, config);
        result["freshness"] = crate::freshness::freshness_to_json(&report);
    }

    Ok(result)
}

/// Task-focused context: combines vector + rules + recent changes for a specific task.
fn task_context(db: &Axil, task: &str, opts: &ContextOptions) -> axil_core::Result<Value> {
    let max = opts.max_tokens;
    let mut tokens_used = 0usize;
    let mut sources: serde_json::Map<String, Value> = serde_json::Map::new();
    let budget_per_section = max / 5; // 5 sections now: code, modules, similar, rules, recent

    // 0. Relevant code (proxy hits) — pointer-first.
    let mut relevant_code: Vec<Value> = Vec::new();
    let mut related_memories: Vec<Value> = Vec::new();
    let mut graph_neighbors_out: Vec<Value> = Vec::new();
    if let Ok(rwr) = recall_with_related(db, task, 8, 5) {
        let section_start = tokens_used;
        for r in rwr.primary.iter().filter(|r| r.source == "proxy") {
            let entry = json!({
                "path": r.path,
                "symbol": r.symbol,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "breadcrumb": r.breadcrumb,
                "canonical_id": r.canonical_id,
                "proxy_id": r.proxy_id,
                "kind": r.kind,
                "why": r.why,
            });
            let t = token::estimate_json_tokens(&entry);
            if tokens_used + t > section_start + budget_per_section {
                break;
            }
            tokens_used += t;
            relevant_code.push(entry);
        }
        sources.insert(
            "relevant_code".to_string(),
            json!({"tokens": tokens_used - section_start, "type": "code-proxy"}),
        );

        let section_start = tokens_used;
        for r in &rwr.related {
            let entry = json!({
                "id": r.id,
                "table": r.source,
                "summary": r.summary,
                "why": r.why,
            });
            let t = token::estimate_json_tokens(&entry);
            if tokens_used + t > section_start + budget_per_section {
                break;
            }
            tokens_used += t;
            related_memories.push(entry);
        }
        if tokens_used > section_start {
            sources.insert(
                "related_memories".to_string(),
                json!({"tokens": tokens_used - section_start, "type": "pointer-attached"}),
            );
        }

        let section_start = tokens_used;
        for r in &rwr.graph_neighbors {
            let entry = json!({
                "path": r.path,
                "symbol": r.symbol,
                "line_start": r.line_start,
                "breadcrumb": r.breadcrumb,
                "canonical_id": r.canonical_id,
                "proxy_id": r.proxy_id,
                "kind": r.kind,
                "why": r.why,
            });
            let t = token::estimate_json_tokens(&entry);
            if tokens_used + t > section_start + budget_per_section {
                break;
            }
            tokens_used += t;
            graph_neighbors_out.push(entry);
        }
        if tokens_used > section_start {
            sources.insert(
                "graph_neighbors".to_string(),
                json!({"tokens": tokens_used - section_start, "type": "scip-graph"}),
            );
        }
    }

    // 1. Relevant modules via keyword matching (reuse prefetch's helper).
    let keywords = crate::prefetch::extract_keywords(task);
    let (module_section, module_tokens) =
        crate::prefetch::build_module_context(db, &keywords, budget_per_section)?;
    let module_entries = module_section.data;
    tokens_used += module_tokens;
    sources.insert(
        "relevant_modules".to_string(),
        json!({"tokens": module_tokens, "type": "keyword"}),
    );

    // 2. Vector-similar records
    let mut similar: Vec<Value> = Vec::new();
    let section_start = tokens_used;
    if let Ok(hits) = db.similar_to(task, 5) {
        for (record, score) in &hits {
            let summary = record
                .data
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let entry = json!({"id": record.id.to_string(), "score": score, "summary": summary});
            let t = token::estimate_json_tokens(&entry);
            if tokens_used + t > section_start + budget_per_section {
                break;
            }
            tokens_used += t;
            similar.push(entry);
        }
    }
    sources.insert(
        "similar_context".to_string(),
        json!({"tokens": tokens_used - section_start, "type": "vector"}),
    );

    // 3. Active rules
    let mut rules: Vec<Value> = Vec::new();
    let section_start = tokens_used;
    let rule_records = db.list(crate::rules::TABLE_RULES).unwrap_or_default();
    for record in &rule_records {
        let t = token::estimate_json_tokens(&record.data);
        if tokens_used + t > section_start + budget_per_section {
            break;
        }
        tokens_used += t;
        rules.push(record.data.clone());
    }
    sources.insert(
        "active_rules".to_string(),
        json!({"tokens": tokens_used - section_start, "type": "key-value"}),
    );

    // 4. Recent changes
    let mut recent: Vec<Value> = Vec::new();
    let section_start = tokens_used;
    let seven_days = 7 * 86400;
    let recent_records = db.since(None, seven_days).unwrap_or_default();
    for record in &recent_records {
        let summary = record
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let path = record
            .data
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let entry = json!({"summary": summary, "path": path});
        let t = token::estimate_json_tokens(&entry);
        if tokens_used + t > section_start + budget_per_section {
            break;
        }
        tokens_used += t;
        recent.push(entry);
    }
    sources.insert(
        "recent_changes".to_string(),
        json!({"tokens": tokens_used - section_start, "type": "time-series"}),
    );

    Ok(json!({
        "task": task,
        "relevant_code": relevant_code,
        "related_memories": related_memories,
        "graph_neighbors": graph_neighbors_out,
        "relevant_modules": module_entries,
        "similar_context": similar,
        "active_rules": rules,
        "recent_changes": recent,
        "sources": sources,
        "tokens_used": tokens_used,
    }))
}

/// Render a `context()` result as a lean, agent-facing text block.
///
/// The JSON form (`context()`) carries every section plus scores, ids, and
/// nulls — useful for programmatic consumers, but heavy when an agent just
/// needs to locate code. This renders only the high-signal pointers as one
/// line per hit (`path:line  symbol — why`), dropping the vector
/// `similar_context`, module keyword echoes, empty rules, and per-section
/// token bookkeeping. It is typically ~10× smaller than the JSON and is the
/// CLI default for `code-context`.
pub fn render_context_compact(value: &Value) -> String {
    let mut out = String::new();
    let line = |e: &Value| -> String {
        let path = e.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let sym = e.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
        let why = e.get("why").and_then(|v| v.as_str()).unwrap_or("");
        let loc = match e.get("line_start").and_then(|v| v.as_u64()) {
            Some(l) => format!("{path}:{l}"),
            None => path.to_string(),
        };
        let mut s = loc;
        if !sym.is_empty() {
            s.push(' ');
            s.push_str(sym);
        }
        if !why.is_empty() {
            s.push_str(" — ");
            s.push_str(why);
        }
        s
    };

    let arr = |k: &str| -> Vec<Value> {
        value
            .get(k)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    };

    let code = arr("relevant_code");
    if !code.is_empty() {
        out.push_str("Relevant code:\n");
        for e in &code {
            out.push_str("  ");
            out.push_str(&line(e));
            out.push('\n');
        }
    }
    let neighbors = arr("graph_neighbors");
    if !neighbors.is_empty() {
        out.push_str("Graph neighbors (callers/callees):\n");
        for e in &neighbors {
            out.push_str("  ");
            out.push_str(&line(e));
            out.push('\n');
        }
    }
    let mems = arr("related_memories");
    if !mems.is_empty() {
        out.push_str("Related memories:\n");
        for e in &mems {
            let summary = e.get("summary").and_then(|v| v.as_str()).unwrap_or("");
            let table = e.get("table").and_then(|v| v.as_str()).unwrap_or("memory");
            out.push_str(&format!("  [{table}] {summary}\n"));
        }
    }
    if out.is_empty() {
        out.push_str("(no relevant code found — try `axil code-search \"<term>\"` or `axil fts`)\n");
    }
    out
}

/// Get index statistics including token efficiency metrics.
/// If `project_root` and `config` are provided, includes freshness info.
pub fn stats(
    db: &Axil,
    project_root: Option<&std::path::Path>,
    config: Option<&axil_core::IndexConfig>,
) -> axil_core::Result<Value> {
    let project_records = db.list(TABLE_PROJECT)?;
    let file_records = db.list(TABLE_FILES)?;
    let module_records = db.list(TABLE_MODULES)?;
    let symbol_records = db.list(TABLE_SYMBOLS)?;
    let dep_records = db.list(TABLE_DEPS)?;

    let count_tokens = |records: &[axil_core::Record]| -> usize {
        records
            .iter()
            .map(|r| {
                r.data
                    .get("tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or_else(|| token::estimate_json_tokens(&r.data) as u64)
                    as usize
            })
            .sum()
    };

    let project_tokens = count_tokens(&project_records);
    let file_tokens = count_tokens(&file_records);
    let module_tokens = count_tokens(&module_records);
    let symbol_tokens = count_tokens(&symbol_records);
    let dep_tokens = count_tokens(&dep_records);
    let total_index_tokens =
        project_tokens + file_tokens + module_tokens + symbol_tokens + dep_tokens;

    let total_source_tokens = project_records
        .first()
        .and_then(|r| r.data.get("total_source_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let ratio = if total_index_tokens > 0 {
        format!("{}:1", total_source_tokens / total_index_tokens)
    } else {
        "N/A".to_string()
    };

    let mut result = json!({
        "index": {
            "total_source_tokens": total_source_tokens,
            "total_index_tokens": total_index_tokens,
            "compression_ratio": ratio,
            "tables": {
                "project": {"records": project_records.len(), "tokens": project_tokens},
                "modules": {"records": module_records.len(), "tokens": module_tokens},
                "files": {"records": file_records.len(), "tokens": file_tokens},
                "symbols": {"records": symbol_records.len(), "tokens": symbol_tokens},
                "deps": {"records": dep_records.len(), "tokens": dep_tokens},
            }
        }
    });

    if let (Some(root), Some(cfg)) = (project_root, config) {
        let report = crate::freshness::check_freshness(db, root, cfg);
        result["freshness"] = crate::freshness::freshness_to_json(&report);
    }

    Ok(result)
}

// ── Scoring ──────────────────────────────────────────────────────────

fn score_record(data: &Value, query_terms: &[&str]) -> f32 {
    // `proxy_text` is intentionally excluded — it carries the full
    // doc/section body and would over-promote markdown sections in the
    // keyword fallback. FTS indexes `proxy_text` separately.
    let searchable_fields = [
        "summary",
        "name",
        "path",
        "doc",
        "signature",
        "purpose",
        "symbol",
        "breadcrumb",
        "canonical_id",
    ];
    let mut text = String::new();

    for field in &searchable_fields {
        if let Some(val) = data.get(*field).and_then(|v| v.as_str()) {
            text.push(' ');
            text.push_str(val);
        }
    }

    // Also search array fields
    for field in &["exports", "key_types", "tech_stack", "public_api"] {
        if let Some(arr) = data.get(*field).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    text.push(' ');
                    text.push_str(s);
                }
            }
        }
    }

    let text_lower = text.to_lowercase();

    // Score: fraction of query terms that match
    let matching = query_terms
        .iter()
        .filter(|term| text_lower.contains(**term))
        .count();
    if matching == 0 {
        return 0.0;
    }

    let base_score = matching as f32 / query_terms.len() as f32;

    // Boost exact matches
    let has_exact = text_lower.contains(&query_terms.join(" "));
    let boost = if has_exact { 0.2 } else { 0.0 };

    (base_score + boost).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_matching() {
        let data = json!({"summary": "JWT auth middleware", "name": "auth"});
        let score = score_record(&data, &["auth", "middleware"]);
        assert!(score > 0.5);
    }

    #[test]
    fn identifier_word_set_splits_camel_and_separators() {
        assert_eq!(identifier_word_set("URLResolver"), vec!["url", "resolver"]);
        assert_eq!(identifier_word_set("SQLCompiler"), vec!["sql", "compiler"]);
        assert_eq!(
            identifier_word_set("urls/resolvers.py"),
            vec!["urls", "resolvers", "py"]
        );
        assert_eq!(identifier_word_set("get_response"), vec!["get", "response"]);
        assert_eq!(
            identifier_word_set("MigrationAutodetector"),
            vec!["migration", "autodetector"]
        );
    }

    #[test]
    fn expand_query_adds_concatenated_and_synonyms() {
        let ex = expand_query("url resolver");
        assert_eq!(ex[0], "url resolver"); // original always first
        assert!(ex.iter().any(|q| q == "urlresolver")); // concatenated identifier
        // synonym variant present (url -> uri/route)
        assert!(ex.iter().any(|q| q.contains("uri") || q.contains("route")));
        // the concatenated form never duplicates the original spaced query
        assert_eq!(ex.iter().filter(|q| q.as_str() == "url resolver").count(), 1);
        // a single content-word query has nothing to concatenate/expand
        assert_eq!(expand_query("resolver"), vec!["resolver".to_string()]);
        // bounded
        assert!(expand_query("a b c d e f g").len() <= 4);
    }

    #[test]
    fn zero_scored_resets_similarity() {
        let mut r = RecallResult {
            score: 0.93,
            ..Default::default()
        };
        r.symbol = Some("X".into());
        let z = zero_scored(&r);
        assert_eq!(z.score, 0.0);
        assert_eq!(z.symbol.as_deref(), Some("X")); // other fields preserved
    }

    #[test]
    fn identity_boost_terms_skip_stopwords() {
        // The discriminative content words survive; filler is dropped.
        let q = "Where is the URL resolver that matches a request";
        let terms: Vec<String> = identifier_word_set(q)
            .into_iter()
            .filter(|t| t.len() >= 3 && !is_stopword(t))
            .collect();
        assert!(terms.contains(&"url".to_string()));
        assert!(terms.contains(&"resolver".to_string()));
        assert!(!terms.contains(&"the".to_string()));
        assert!(!terms.contains(&"where".to_string()));
        assert!(!terms.contains(&"that".to_string()));
    }

    #[test]
    fn compact_context_is_lean_pointer_lines() {
        let v = json!({
            "task": "where is jsonify",
            "relevant_code": [
                {"path": "src/flask/json/__init__.py", "symbol": "jsonify",
                 "line_start": 138, "why": "matched via full-text proxy",
                 "proxy_id": "abc", "canonical_id": null, "kind": "symbol"}
            ],
            "graph_neighbors": [
                {"path": "src/flask/json/__init__.py", "symbol": "dumps",
                 "line_start": 200, "why": "graph neighbor"}
            ],
            "related_memories": [{"table": "decisions", "summary": "use orjson"}],
            // Noise sections that compact must drop:
            "similar_context": [{"id": "x", "score": 0.77, "summary": "noise"}],
            "active_rules": [], "recent_changes": [],
            "sources": {"similar_context": {"tokens": 999}}, "tokens_used": 999
        });
        let out = render_context_compact(&v);
        assert!(out.contains("src/flask/json/__init__.py:138 jsonify"));
        assert!(out.contains("Graph neighbors"));
        assert!(out.contains("[decisions] use orjson"));
        // The vector-noise + bookkeeping must NOT leak into the lean view.
        assert!(!out.contains("similar_context"));
        assert!(!out.contains("0.77"));
        assert!(!out.contains("tokens_used"));
        // Lean output is far smaller than the full JSON serialization.
        assert!(out.len() < serde_json::to_string(&v).unwrap().len());
    }

    #[test]
    fn compact_context_handles_empty() {
        let v = json!({"task": "x", "relevant_code": []});
        let out = render_context_compact(&v);
        assert!(out.contains("no relevant code"));
    }

    #[test]
    fn score_no_match() {
        let data = json!({"summary": "database connection pool"});
        let score = score_record(&data, &["authentication"]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn score_partial_match() {
        let data = json!({"summary": "user authentication service"});
        let score = score_record(&data, &["auth", "database"]);
        // Only "auth" matches (as substring of "authentication")
        assert!(score > 0.0);
        assert!(score < 1.0);
    }
}
