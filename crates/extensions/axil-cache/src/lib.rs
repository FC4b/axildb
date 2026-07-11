//! `axil-cache` — a semantic answer cache with code-aware invalidation.
//!
//! An agent often derives an expensive answer ("how does auth token refresh
//! work in this repo?") that it will need again, phrased a little
//! differently, in a later turn or session. This Extension caches the
//! question → answer pair so a *semantically similar* question returns the
//! stored answer instead of re-deriving it.
//!
//! ## What makes it code-aware
//!
//! A purely semantic cache (matching only on question similarity plus a TTL)
//! will happily serve an answer that the codebase has since invalidated. A
//! cache entry here can pin itself to the code it talks about via
//! `code_refs`; each ref carries a content fingerprint captured at put time.
//! On read, after a similarity hit, every ref is re-fingerprinted — if the
//! referenced code changed (or was removed), the entry is dropped and the
//! read reports a miss with reason `stale_code`. See [`codref`] for the
//! fingerprint mechanics.
//!
//! ## Check-on-read
//!
//! Staleness is evaluated lazily, on `cache get`, so nothing needs to hook
//! into file-change events or the brain hooks. TTL expiry is checked the
//! same way. A read never returns an expired or code-stale answer.
//!
//! ## Storage
//!
//! - `_cache_entries` — one row per cached question/answer pair. The
//!   `question` field is embedded (for vector similarity) and full-text
//!   indexed. Each row carries `answer`, `created_at`, optional
//!   `valid_until`, `hit_count`, `last_hit_at`, and `code_refs[]`.
//! - `_cache_meta` — a single row of cumulative counters (`total_hits`,
//!   `total_misses`, `stale_evictions`, `expired_evictions`) so `cache
//!   stats` can report a hit rate and how much code-aware invalidation has
//!   fired.

pub mod codref;
pub mod extension;

pub use codref::{current_fingerprint, resolve_ref, staleness, CodeFingerprint, StaleReason};
pub use extension::CacheExtension;

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, AxilError, Op, Record, Result};

/// Table holding cached question/answer pairs.
///
/// The `_cache_` prefix namespaces every table this Extension owns, so its
/// rows never collide with core tables or another Extension's. Kept plural so
/// the sibling counters table (`_cache_meta`) slots in under the same prefix.
pub const TABLE_CACHE_ENTRIES: &str = "_cache_entries";

/// Table holding the single cumulative-counters row.
pub const TABLE_CACHE_META: &str = "_cache_meta";

/// Field embedded and full-text indexed for similarity search.
const FIELD_QUESTION: &str = "question";

/// Default cosine-similarity threshold for a cache hit. Matches Axil's
/// existing memory-superseding similarity threshold, so "similar enough to
/// reuse" and "similar enough to supersede" stay calibrated together.
pub const DEFAULT_THRESHOLD: f32 = 0.92;

/// Errors from the cache put/get paths.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("cache payload must be a JSON object")]
    NotAnObject,
    #[error("cache put requires a non-empty `question` and `answer`")]
    MissingFields,
    /// The expiry was unusable — an unparseable `valid_until` string, or a
    /// `ttl` too large to represent as a timestamp. One variant covers both:
    /// `CacheError` shipped without `#[non_exhaustive]` in 2.0.1, so adding a
    /// variant is a semver-major change; reusing this one keeps overflow
    /// handling additive.
    #[error("invalid `valid_until` timestamp: {0}")]
    BadTimestamp(String),
    #[error("axil error: {0}")]
    Axil(#[from] AxilError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// A parsed `cache put` request.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PutRequest {
    /// The question this answer resolves. Embedded for semantic recall.
    pub question: String,
    /// The answer to return on a future similar question.
    pub answer: String,
    /// Optional code-ref specs (proxy_id | canonical_id | path[:line]),
    /// resolved and fingerprinted the same way `axil store --code-ref` is.
    #[serde(default)]
    pub code_refs: Vec<String>,
    /// Optional time-to-live in seconds from now.
    #[serde(default)]
    pub ttl: Option<u64>,
    /// Optional explicit expiry timestamp (RFC 3339). Overrides `ttl` when
    /// both are given.
    #[serde(default)]
    pub valid_until: Option<String>,
}

impl PutRequest {
    /// Parse + validate a put request from raw JSON.
    pub fn from_value(value: Value) -> std::result::Result<Self, CacheError> {
        if !value.is_object() {
            return Err(CacheError::NotAnObject);
        }
        let req: PutRequest = serde_json::from_value(value)?;
        if req.question.trim().is_empty() || req.answer.trim().is_empty() {
            return Err(CacheError::MissingFields);
        }
        Ok(req)
    }

    /// Resolve the request's `ttl` / `valid_until` into an absolute expiry.
    pub fn resolve_valid_until(&self) -> std::result::Result<Option<DateTime<Utc>>, CacheError> {
        if let Some(ts) = &self.valid_until {
            let parsed = DateTime::parse_from_rfc3339(ts)
                .map_err(|e| CacheError::BadTimestamp(e.to_string()))?
                .with_timezone(&Utc);
            return Ok(Some(parsed));
        }
        // A caller-supplied `ttl` is unbounded. `TimeDelta::seconds` panics on
        // out-of-range input and a u64 past `i64::MAX` would wrap negative
        // through `as i64`, so convert and add fallibly: `try_seconds` rejects
        // a delta too large to represent and `checked_add_signed` rejects an
        // expiry past chrono's maximum date. Either way the overflow surfaces
        // as an error instead of a panic that would kill the process.
        let Some(secs) = self.ttl else {
            return Ok(None);
        };
        let overflow =
            || CacheError::BadTimestamp(format!("ttl of {secs} seconds overflows the expiry"));
        let secs_i64 = i64::try_from(secs).map_err(|_| overflow())?;
        let delta = chrono::TimeDelta::try_seconds(secs_i64).ok_or_else(overflow)?;
        let expiry = Utc::now().checked_add_signed(delta).ok_or_else(overflow)?;
        Ok(Some(expiry))
    }
}

/// Store a cache entry, resolving each code-ref spec against the current
/// code proxies + working tree and capturing its fingerprint.
///
/// `base_dir` is the directory a relative code-ref path resolves against —
/// the working directory at put time.
pub fn put(
    db: &Axil,
    req: &PutRequest,
    base_dir: &Path,
) -> std::result::Result<Record, CacheError> {
    let valid_until = req.resolve_valid_until()?;
    let code_refs: Vec<Value> = req
        .code_refs
        .iter()
        .map(|spec| resolve_ref(db, spec, base_dir))
        .collect();

    let now = Utc::now();
    let mut data = json!({
        "question": req.question,
        "answer": req.answer,
        "created_at": now.to_rfc3339(),
        "hit_count": 0,
        "code_refs": code_refs,
    });
    if let Some(obj) = data.as_object_mut() {
        if let Some(vu) = valid_until {
            obj.insert("valid_until".into(), json!(vu.to_rfc3339()));
        }
    }

    let record = db.insert(TABLE_CACHE_ENTRIES, data)?;
    // Best-effort search indexing: a DB opened without a vector/FTS engine
    // simply won't match this row semantically. The put still succeeds.
    let _ = db.embed_field(&record.id, FIELD_QUESTION);
    let _ = db.index_text(&record.id, FIELD_QUESTION, &req.question);
    Ok(record)
}

/// A cache hit surfaced to the caller.
#[derive(Debug, Clone, Serialize)]
pub struct CacheHit {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Similarity score of the matched question (1.0 for an exact-text
    /// fallback match when no vector index is configured).
    pub score: f32,
    pub hit_count: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub code_refs: Vec<Value>,
}

/// Why `cache get` returned no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissReason {
    /// No cached entry surfaced for the question at all.
    NoMatch,
    /// The closest entry scored below the similarity threshold.
    BelowThreshold,
    /// The closest matching entry referenced code that has since changed;
    /// it was evicted.
    StaleCode,
    /// The closest matching entry had passed its TTL; it was evicted.
    Expired,
}

impl MissReason {
    /// The stable snake_case wire token for this reason, as surfaced in the
    /// CLI and MCP miss payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoMatch => "no_match",
            Self::BelowThreshold => "below_threshold",
            Self::StaleCode => "stale_code",
            Self::Expired => "expired",
        }
    }
}

/// The outcome of a `cache get`.
#[derive(Debug, Clone)]
pub enum GetOutcome {
    /// One or more fresh hits at or above the threshold.
    Hit(Vec<CacheHit>),
    /// No usable answer, with the reason and (when relevant) the best score
    /// seen and a human detail string.
    Miss {
        reason: MissReason,
        best_score: Option<f32>,
        detail: Option<String>,
    },
}

/// Look up a cached answer for `question`.
///
/// Ranks cache entries by question similarity, then walks them best-first:
/// the first entry at or above `threshold` that is neither expired nor
/// code-stale becomes a hit (up to `top_k`). Expired and code-stale entries
/// encountered along the way are evicted. When no fresh hit clears the bar,
/// the returned [`MissReason`] explains why — distinguishing "nothing
/// similar", "not similar enough", "the code moved on", and "it expired".
pub fn get(
    db: &Axil,
    question: &str,
    threshold: f32,
    top_k: usize,
    base_dir: &Path,
) -> Result<GetOutcome> {
    let top_k = top_k.max(1);
    let candidates = ranked_candidates(db, question)?;
    if candidates.is_empty() {
        bump_meta(db, MetaEvent::Miss);
        return Ok(GetOutcome::Miss {
            reason: MissReason::NoMatch,
            best_score: None,
            detail: None,
        });
    }

    let best_score = candidates.first().map(|(_, s)| *s);
    // Everything below threshold is out of scope for reuse.
    let eligible: Vec<(Record, f32)> = candidates
        .into_iter()
        .filter(|(_, s)| *s >= threshold)
        .collect();
    if eligible.is_empty() {
        bump_meta(db, MetaEvent::Miss);
        return Ok(GetOutcome::Miss {
            reason: MissReason::BelowThreshold,
            best_score,
            detail: None,
        });
    }

    let now = Utc::now();
    let mut hits: Vec<CacheHit> = Vec::new();
    // Reason the best eligible entry failed, surfaced when no fresh hit is
    // found. The first (highest-scoring) failure wins so the miss reason
    // describes the entry the agent most expected to reuse.
    let mut failure: Option<(MissReason, Option<String>)> = None;

    for (record, score) in eligible {
        if is_expired(&record.data, now) {
            let _ = db.delete(&record.id);
            bump_meta(db, MetaEvent::ExpiredEviction);
            failure.get_or_insert((MissReason::Expired, None));
            continue;
        }
        if let Some(reason) = code_stale_reason(db, &record.data, base_dir) {
            let _ = db.delete(&record.id);
            bump_meta(db, MetaEvent::StaleEviction);
            failure.get_or_insert((MissReason::StaleCode, Some(reason)));
            continue;
        }

        hits.push(record_to_hit(db, record, score));
        if hits.len() >= top_k {
            break;
        }
    }

    if hits.is_empty() {
        bump_meta(db, MetaEvent::Miss);
        let (reason, detail) = failure.unwrap_or((MissReason::NoMatch, None));
        return Ok(GetOutcome::Miss {
            reason,
            best_score,
            detail,
        });
    }
    bump_meta(db, MetaEvent::Hit);
    Ok(GetOutcome::Hit(hits))
}

/// Rank `_cache_entries` by similarity to `question`, best-first.
///
/// Uses vector search when the database has an embedder + vector index;
/// otherwise falls back to an exact question-text match (score 1.0) so the
/// cache remains usable without embeddings.
fn ranked_candidates(db: &Axil, question: &str) -> Result<Vec<(Record, f32)>> {
    // Table-scoped similarity. The ANN index is not partitioned by table, so a
    // global `similar_to` returns a fixed-size top-N in which a memory-heavy
    // database's non-cache rows can fill every slot and crowd cached answers out
    // of the window — the cache then reports a false miss even though a good
    // entry is stored. `_cache_entries` is small and retention-bounded, so we
    // score it directly: embed the question once and rank every live entry by
    // exact cosine against its own stored question text. No non-cache row can
    // displace a cache entry because none is ever scored, and the score stays
    // consistent with the index (same deterministic embedder, same cosine metric
    // the threshold is calibrated against).
    if db.has_vector_index() && db.has_embedder() {
        if let Ok(query_vec) = db.embed_query(question) {
            let mut out: Vec<(Record, f32)> = Vec::new();
            for record in db.list(TABLE_CACHE_ENTRIES)? {
                let Some(entry_question) = record.data.get(FIELD_QUESTION).and_then(|v| v.as_str())
                else {
                    continue;
                };
                match db.embed_query(entry_question) {
                    Ok(entry_vec) => out.push((record, cosine(&query_vec, &entry_vec))),
                    // Skip an entry we cannot embed rather than fail the read.
                    Err(_) => continue,
                }
            }
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(out);
        }
        // Embedding the query itself failed — degrade to the exact-text path.
    }

    // No vector index: exact-text fallback.
    let rows = db
        .query()
        .table(TABLE_CACHE_ENTRIES)
        .where_field(FIELD_QUESTION, Op::Eq, json!(question))
        .exec()?;
    Ok(rows.into_iter().map(|r| (r, 1.0)).collect())
}

/// Cosine similarity between two vectors, `0.0` when either is all-zero (no
/// direction to compare). Shared by the scoped ranking above and its tests so
/// the metric that gates a cache hit is defined in exactly one place.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// `true` when the entry carries a `valid_until` that has passed.
fn is_expired(data: &Value, now: DateTime<Utc>) -> bool {
    data.get("valid_until")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|vu| vu.with_timezone(&Utc) < now)
        .unwrap_or(false)
}

/// Return a human reason if any of the entry's code refs is stale, else
/// `None`.
fn code_stale_reason(db: &Axil, data: &Value, base_dir: &Path) -> Option<String> {
    let refs = data.get("code_refs").and_then(|v| v.as_array())?;
    for code_ref in refs {
        let stored: CodeFingerprint = code_ref
            .get("fingerprint")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        if stored.is_empty() {
            continue;
        }
        let current = current_fingerprint(db, code_ref, base_dir);
        if let Some(reason) = staleness(&stored, &current) {
            let target = code_ref
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| code_ref.get("symbol").and_then(|v| v.as_str()))
                .or_else(|| code_ref.get("proxy_id").and_then(|v| v.as_str()))
                .unwrap_or("<ref>");
            return Some(format!("{target}: {}", reason.as_str()));
        }
    }
    None
}

/// Increment `hit_count` / stamp `last_hit_at` on a served entry, then map
/// it into a [`CacheHit`]. Counter update is best-effort.
fn record_to_hit(db: &Axil, record: Record, score: f32) -> CacheHit {
    let prior = record
        .data
        .get("hit_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let new_count = prior + 1;
    let mut updated = record.data.clone();
    if let Some(obj) = updated.as_object_mut() {
        obj.insert("hit_count".into(), json!(new_count));
        obj.insert("last_hit_at".into(), json!(Utc::now().to_rfc3339()));
    }
    let _ = db.update(&record.id, updated);

    CacheHit {
        id: record.id.to_string(),
        question: str_field(&record.data, "question"),
        answer: str_field(&record.data, "answer"),
        score,
        hit_count: new_count,
        code_refs: record
            .data
            .get("code_refs")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
    }
}

fn str_field(data: &Value, key: &str) -> String {
    data.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

/// Cumulative cache statistics, for `cache stats`.
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    /// Live entries currently in the cache.
    pub entries: usize,
    /// Answers served across the cache's lifetime.
    pub total_hits: u64,
    /// Misses across the cache's lifetime.
    pub total_misses: u64,
    /// Hit rate over `total_hits + total_misses`, or `null` before any read.
    pub hit_rate: Option<f32>,
    /// Entries evicted on read because their referenced code changed.
    pub stale_evictions: u64,
    /// Entries evicted on read because their TTL had passed.
    pub expired_evictions: u64,
}

/// Read cumulative statistics.
pub fn stats(db: &Axil) -> Result<CacheStats> {
    let entries = db.list(TABLE_CACHE_ENTRIES)?.len();
    let meta = meta_row(db);
    let total_hits = meta_u64(&meta, "total_hits");
    let total_misses = meta_u64(&meta, "total_misses");
    let denom = total_hits + total_misses;
    let hit_rate = if denom > 0 {
        Some(total_hits as f32 / denom as f32)
    } else {
        None
    };
    Ok(CacheStats {
        entries,
        total_hits,
        total_misses,
        hit_rate,
        stale_evictions: meta_u64(&meta, "stale_evictions"),
        expired_evictions: meta_u64(&meta, "expired_evictions"),
    })
}

/// What to clear in [`clear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearScope {
    /// Every cache entry.
    All,
    /// Only entries past their TTL.
    Expired,
}

/// Remove cache entries per `scope`. Returns the number removed. Cumulative
/// counters in `_cache_meta` are historical and are left untouched.
pub fn clear(db: &Axil, scope: ClearScope) -> Result<usize> {
    let now = Utc::now();
    let mut removed = 0;
    for row in db.list(TABLE_CACHE_ENTRIES)? {
        let drop = match scope {
            ClearScope::All => true,
            ClearScope::Expired => is_expired(&row.data, now),
        };
        if drop && db.delete(&row.id)? {
            removed += 1;
        }
    }
    Ok(removed)
}

// ── cumulative counters (`_cache_meta`) ─────────────────────────────────────

/// A countable read event.
#[derive(Debug, Clone, Copy)]
enum MetaEvent {
    Hit,
    Miss,
    StaleEviction,
    ExpiredEviction,
}

/// Fetch (or synthesize an empty) `_cache_meta` singleton row.
fn meta_row(db: &Axil) -> Value {
    db.list(TABLE_CACHE_META)
        .ok()
        .and_then(|rows| rows.into_iter().next())
        .map(|r| r.data)
        .unwrap_or_else(|| json!({}))
}

fn meta_u64(data: &Value, key: &str) -> u64 {
    data.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// Increment the appropriate `_cache_meta` counter. Best-effort — a failed
/// counter write never fails the surrounding read.
fn bump_meta(db: &Axil, event: MetaEvent) {
    let rows = match db.list(TABLE_CACHE_META) {
        Ok(r) => r,
        Err(_) => return,
    };
    let existing = rows.into_iter().next();
    let mut data = existing
        .as_ref()
        .map(|r| r.data.clone())
        .unwrap_or_else(|| {
            json!({
                "total_hits": 0,
                "total_misses": 0,
                "stale_evictions": 0,
                "expired_evictions": 0,
            })
        });

    let key = match event {
        MetaEvent::Hit => "total_hits",
        MetaEvent::Miss => "total_misses",
        MetaEvent::StaleEviction => "stale_evictions",
        MetaEvent::ExpiredEviction => "expired_evictions",
    };
    if let Some(obj) = data.as_object_mut() {
        let cur = obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
        obj.insert(key.into(), json!(cur + 1));
    }

    match existing {
        Some(row) => {
            let _ = db.update(&row.id, data);
        }
        None => {
            let _ = db.insert(TABLE_CACHE_META, data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axil_core::plugin::{Capability, Engine, SearchIndex, TextEmbedder, VectorIndex};
    use axil_core::RecordId;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// A deterministic in-memory vector engine that computes real cosine
    /// similarity, so threshold behavior is testable without ONNX. Embeds
    /// text as a bag-of-lowercased-words vector over a fixed vocabulary.
    #[derive(Default)]
    struct CosineVector {
        vectors: Mutex<HashMap<String, Vec<f32>>>,
    }

    impl Engine for CosineVector {
        fn name(&self) -> &str {
            "cosine-vector"
        }
        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability::VectorSearch]
        }
        fn on_record_insert(&self, _record: &Record) -> Result<()> {
            Ok(())
        }
        fn on_record_delete(&self, id: &RecordId) -> Result<()> {
            self.vectors.lock().unwrap().remove(&id.to_string());
            Ok(())
        }
    }

    impl VectorIndex for CosineVector {
        fn add(&self, id: RecordId, vector: &[f32]) -> Result<()> {
            self.vectors
                .lock()
                .unwrap()
                .insert(id.to_string(), vector.to_vec());
            Ok(())
        }
        fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>> {
            let mut scored: Vec<(RecordId, f32)> = self
                .vectors
                .lock()
                .unwrap()
                .iter()
                .filter_map(|(id, v)| {
                    RecordId::from_string(id).ok().map(|rid| (rid, cosine(query, v)))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);
            Ok(scored)
        }
        fn count(&self) -> usize {
            self.vectors.lock().unwrap().len()
        }
        fn dimensions(&self) -> usize {
            VOCAB.len()
        }
    }

    const VOCAB: &[&str] = &[
        "how", "does", "auth", "token", "refresh", "work", "login", "cache", "the", "flow",
    ];

    impl TextEmbedder for CosineVector {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let lowered = text.to_lowercase();
            let words: Vec<&str> = lowered.split_whitespace().collect();
            let mut v = vec![0.0_f32; VOCAB.len()];
            for (i, term) in VOCAB.iter().enumerate() {
                if words.contains(term) {
                    v[i] = 1.0;
                }
            }
            Ok(v)
        }
    }

    #[derive(Default)]
    struct NoopFts;
    impl Engine for NoopFts {
        fn name(&self) -> &str {
            "noop-fts"
        }
        fn capabilities(&self) -> Vec<Capability> {
            vec![Capability::FullTextSearch]
        }
        fn on_record_insert(&self, _record: &Record) -> Result<()> {
            Ok(())
        }
        fn on_record_delete(&self, _id: &RecordId) -> Result<()> {
            Ok(())
        }
    }
    impl SearchIndex for NoopFts {
        fn index_text(&self, _id: &RecordId, _field: &str, _text: &str) -> Result<()> {
            Ok(())
        }
        fn search_text(&self, _query: &str, _limit: usize) -> Result<Vec<(RecordId, f32)>> {
            Ok(Vec::new())
        }
    }

    fn vector_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("cache.axil"))
            .with_vector_and_embedder(CosineVector::default())
            .with_fts_index(Arc::new(NoopFts))
            .build()
            .unwrap();
        (db, dir)
    }

    fn plain_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("cache.axil")).build().unwrap();
        (db, dir)
    }

    fn put_req(question: &str, answer: &str) -> PutRequest {
        PutRequest {
            question: question.into(),
            answer: answer.into(),
            ..Default::default()
        }
    }

    #[test]
    fn put_request_validates_fields() {
        assert!(matches!(
            PutRequest::from_value(json!([1, 2])).unwrap_err(),
            CacheError::NotAnObject
        ));
        assert!(matches!(
            PutRequest::from_value(json!({"question": "", "answer": "x"})).unwrap_err(),
            CacheError::MissingFields
        ));
        let ok = PutRequest::from_value(json!({"question": "q", "answer": "a"})).unwrap();
        assert_eq!(ok.question, "q");
    }

    #[test]
    fn put_then_get_roundtrip_on_similar_question() {
        let (db, dir) = vector_db();
        put(&db, &put_req("how does auth token refresh work", "it rotates the token"), dir.path())
            .unwrap();
        // A differently phrased but semantically overlapping question.
        let out = get(&db, "how does the auth token refresh flow work", 0.7, 1, dir.path()).unwrap();
        match out {
            GetOutcome::Hit(hits) => {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].answer, "it rotates the token");
                assert_eq!(hits[0].hit_count, 1);
            }
            other => panic!("expected hit, got {other:?}"),
        }
    }

    #[test]
    fn cache_entry_survives_many_higher_scoring_non_cache_rows() {
        let (db, dir) = vector_db();
        // A legitimately cached answer, phrased with partial overlap so it
        // scores below a perfect match but comfortably above the get threshold.
        put(
            &db,
            &put_req("how does auth token refresh", "rotate the refresh token"),
            dir.path(),
        )
        .unwrap();

        // Flood the database with non-cache rows that each match the query
        // *perfectly* (cosine 1.0) — far more than the old global top-40 fetch
        // window. Under the previous `similar_to(question, 40)` these filled
        // every slot and crowded the cache entry out, a silent false miss.
        for i in 0..60 {
            let rec = db
                .insert(
                    "memories",
                    json!({"text": "how does auth token refresh work", "n": i}),
                )
                .unwrap();
            db.embed_field(&rec.id, "text").unwrap();
        }

        // Table-scoped ranking never scores the memory rows, so the cache entry
        // is still found.
        let out = get(&db, "how does auth token refresh work", 0.5, 1, dir.path()).unwrap();
        match out {
            GetOutcome::Hit(hits) => {
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].answer, "rotate the refresh token");
            }
            other => panic!("cache entry crowded out by non-cache rows: {other:?}"),
        }
    }

    #[test]
    fn threshold_boundary_gates_a_weaker_match() {
        let (db, dir) = vector_db();
        put(&db, &put_req("how does auth token refresh work", "answer"), dir.path()).unwrap();
        // "how does login work" overlaps on {how,does,work} out of the stored
        // {how,does,auth,token,refresh,work} — a partial match well under 0.92.
        let strict = get(&db, "how does login work", 0.92, 1, dir.path()).unwrap();
        assert!(matches!(
            strict,
            GetOutcome::Miss { reason: MissReason::BelowThreshold, .. }
        ));
        // The same query clears a lower bar.
        let loose = get(&db, "how does login work", 0.3, 1, dir.path()).unwrap();
        assert!(matches!(loose, GetOutcome::Hit(_)));
    }

    #[test]
    fn empty_cache_reports_no_match() {
        let (db, dir) = vector_db();
        let out = get(&db, "anything", 0.5, 1, dir.path()).unwrap();
        assert!(matches!(
            out,
            GetOutcome::Miss { reason: MissReason::NoMatch, .. }
        ));
    }

    #[test]
    fn expired_entry_is_a_miss_and_evicted() {
        let (db, dir) = vector_db();
        let mut req = put_req("how does auth token refresh work", "answer");
        // Already expired.
        req.valid_until = Some((Utc::now() - chrono::Duration::hours(1)).to_rfc3339());
        put(&db, &req, dir.path()).unwrap();
        let out = get(&db, "how does auth token refresh work", 0.5, 1, dir.path()).unwrap();
        assert!(matches!(
            out,
            GetOutcome::Miss { reason: MissReason::Expired, .. }
        ));
        // Evicted on read.
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 0);
    }

    #[test]
    fn code_ref_file_change_invalidates_entry() {
        let (db, dir) = vector_db();
        let file = dir.path().join("auth.rs");
        std::fs::write(&file, "fn refresh() { old }").unwrap();
        let mut req = put_req("how does auth token refresh work", "see auth.rs");
        req.code_refs = vec!["auth.rs".into()];
        put(&db, &req, dir.path()).unwrap();

        // Fresh: hit.
        let hit = get(&db, "how does auth token refresh work", 0.5, 1, dir.path()).unwrap();
        assert!(matches!(hit, GetOutcome::Hit(_)));

        // Edit the referenced file: the next read is a stale-code miss.
        std::fs::write(&file, "fn refresh() { new logic }").unwrap();
        let out = get(&db, "how does auth token refresh work", 0.5, 1, dir.path()).unwrap();
        match out {
            GetOutcome::Miss { reason, detail, .. } => {
                assert_eq!(reason, MissReason::StaleCode);
                assert!(detail.unwrap().contains("auth.rs"));
            }
            other => panic!("expected stale-code miss, got {other:?}"),
        }
        // Evicted.
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 0);
    }

    #[test]
    fn exact_match_fallback_without_vector_index() {
        let (db, dir) = plain_db();
        put(&db, &put_req("exact question", "exact answer"), dir.path()).unwrap();
        let hit = get(&db, "exact question", 0.92, 1, dir.path()).unwrap();
        match hit {
            GetOutcome::Hit(hits) => assert_eq!(hits[0].answer, "exact answer"),
            other => panic!("expected exact-match hit, got {other:?}"),
        }
        // A different question yields no exact match.
        let miss = get(&db, "different question", 0.92, 1, dir.path()).unwrap();
        assert!(matches!(miss, GetOutcome::Miss { .. }));
    }

    #[test]
    fn stats_track_hits_misses_and_evictions() {
        let (db, dir) = vector_db();
        put(&db, &put_req("how does auth token refresh work", "answer"), dir.path()).unwrap();
        // One hit.
        let _ = get(&db, "how does auth token refresh work", 0.5, 1, dir.path()).unwrap();
        // One below-threshold miss.
        let _ = get(&db, "unrelated words here", 0.92, 1, dir.path()).unwrap();
        let s = stats(&db).unwrap();
        assert_eq!(s.total_hits, 1);
        assert_eq!(s.total_misses, 1);
        assert_eq!(s.hit_rate, Some(0.5));
        assert_eq!(s.entries, 1);
    }

    #[test]
    fn clear_all_and_expired() {
        let (db, dir) = vector_db();
        put(&db, &put_req("how does auth token refresh work", "a"), dir.path()).unwrap();
        let mut expired = put_req("how does login cache work", "b");
        expired.valid_until = Some((Utc::now() - chrono::Duration::hours(1)).to_rfc3339());
        put(&db, &expired, dir.path()).unwrap();

        // Clear only expired: drops one.
        assert_eq!(clear(&db, ClearScope::Expired).unwrap(), 1);
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 1);
        // Clear all: drops the remainder.
        assert_eq!(clear(&db, ClearScope::All).unwrap(), 1);
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 0);
    }

    #[test]
    fn ttl_seconds_resolves_to_future_expiry() {
        let req = PutRequest {
            question: "q".into(),
            answer: "a".into(),
            ttl: Some(3600),
            ..Default::default()
        };
        let vu = req.resolve_valid_until().unwrap().unwrap();
        assert!(vu > Utc::now());
    }

    #[test]
    fn huge_ttl_is_an_error_not_a_panic() {
        // Past i64::MAX: `as i64` would wrap negative and the old add panicked.
        let req = PutRequest {
            question: "q".into(),
            answer: "a".into(),
            ttl: Some(u64::MAX),
            ..Default::default()
        };
        assert!(matches!(
            req.resolve_valid_until(),
            Err(CacheError::BadTimestamp(_))
        ));

        // In i64 range but large enough that `now + delta` overflows chrono's
        // maximum date — the second panic path.
        let req = PutRequest {
            question: "q".into(),
            answer: "a".into(),
            ttl: Some(10_000_000_000_000_000),
            ..Default::default()
        };
        assert!(matches!(
            req.resolve_valid_until(),
            Err(CacheError::BadTimestamp(_))
        ));
    }

    #[test]
    fn put_with_huge_ttl_returns_error_not_panic() {
        // The panic was reachable straight through `put` from CLI/MCP input.
        let (db, dir) = plain_db();
        let mut req = put_req("q", "a");
        req.ttl = Some(u64::MAX);
        assert!(matches!(
            put(&db, &req, dir.path()),
            Err(CacheError::BadTimestamp(_))
        ));
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 0);
    }
}
