//! Recall over `_dep_docs` — surface version-pinned dependency docs.
//!
//! Dependency-doc chunks are stored in the internal `_dep_docs` table,
//! which the general `recall` pipeline skips (internal tables are not
//! agent memories). This module provides the scoped query that the
//! `axil dep-docs` command exposes: vector similarity + full-text search
//! over `_dep_docs` only.

use axil_core::Axil;

use crate::ingest::TABLE_DEP_DOCS;
use crate::DocsError;

/// One documentation chunk matched by a query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DepDocHit {
    /// Dependency the chunk documents.
    pub dep_name: String,
    /// Exact version the docs were pinned to.
    pub dep_version: String,
    /// Ecosystem (`cargo` / `npm`).
    pub ecosystem: String,
    /// Heading breadcrumb of the chunk's section.
    pub section_path: String,
    /// The documentation text.
    pub content: String,
    /// `"doc"` for API documentation, `"migration"` for a changelog
    /// entry surfaced after a version bump.
    pub doc_kind: String,
    /// Whether this chunk documents a superseded or removed version —
    /// only ever `true` when the query opted into archived results.
    pub superseded: bool,
    /// Relevance score (vector similarity or FTS score).
    pub score: f32,
}

/// Search `_dep_docs` for documentation chunks matching `query`.
///
/// Combines vector similarity and full-text search, keeps only
/// `_dep_docs` records, optionally narrows to a single dependency by
/// name, and returns the top `top_k` by score. Both search backends are
/// best-effort — a database opened without the vector or FTS plugin
/// simply contributes no hits from that backend.
///
/// Archived chunks — docs for a superseded or removed version — are
/// excluded unless `include_superseded` is set, mirroring how the
/// archived memory tier is hidden from default recall.
pub fn query_dep_docs(
    db: &Axil,
    query: &str,
    top_k: usize,
    dep_filter: Option<&str>,
    include_superseded: bool,
) -> Result<Vec<DepDocHit>, DocsError> {
    use std::collections::HashMap;

    // Over-fetch: the search backends rank across *all* records, so a
    // generous window is needed before filtering down to `_dep_docs`.
    let fetch = top_k.max(10).saturating_mul(8);
    let mut by_id: HashMap<String, (axil_core::Record, f32)> = HashMap::new();

    if let Ok(hits) = db.similar_to(query, fetch) {
        for (rec, score) in hits {
            if rec.table == TABLE_DEP_DOCS {
                by_id.entry(rec.id.to_string()).or_insert((rec, score));
            }
        }
    }
    if let Ok(hits) = db.search_text(query, fetch) {
        for (rec, score) in hits {
            if rec.table == TABLE_DEP_DOCS {
                by_id.entry(rec.id.to_string()).or_insert((rec, score));
            }
        }
    }

    let mut hits: Vec<DepDocHit> = by_id
        .into_values()
        .filter_map(|(rec, score)| {
            let data = &rec.data;
            let archived = crate::ingest::is_archived(data);
            if archived && !include_superseded {
                return None;
            }
            let dep_name = data.get("dep_name").and_then(|v| v.as_str())?.to_string();
            if let Some(filter) = dep_filter {
                if dep_name != filter {
                    return None;
                }
            }
            Some(DepDocHit {
                dep_name,
                dep_version: str_field(data, "dep_version"),
                ecosystem: str_field(data, "ecosystem"),
                section_path: str_field(data, "section_path"),
                content: str_field(data, "content"),
                doc_kind: data
                    .get("doc_kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("doc")
                    .to_string(),
                superseded: archived,
                score,
            })
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    Ok(hits)
}

/// Read a string field from a record's data, defaulting to empty.
fn str_field(data: &serde_json::Value, key: &str) -> String {
    data.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}
