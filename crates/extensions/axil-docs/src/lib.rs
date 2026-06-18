//! Dependency documentation memory for Axil — Phase 16.
//!
//! Pre-loads version-pinned documentation for a project's dependencies
//! into Axil memory, so an agent can recall library docs without
//! re-reading `node_modules`/crate source or round-tripping the web.
//!
//! **P0-lite scope:** Rust (Cargo) and npm. Python, Go and Java land in
//! a later increment.
//!
//! This crate is wired behind the default-off `deps` Cargo feature of
//! `axil-cli` — nothing here runs unless the feature is enabled and the
//! agent explicitly invokes `axil deps …`.
//!
//! ## P0.1 — manifest detection + dependency resolution
//!
//! - [`manifest`] detects and parses every supported manifest in a repo.
//! - [`resolve`] pins each dependency to the exact version in the
//!   lockfile (falling back to "unpinned" when no lockfile exists).

pub mod extension;
pub mod imports;
pub mod ingest;
pub mod local;
pub mod manifest;
pub mod pipeline;
pub mod query;
pub mod refresh;
pub mod resolve;
#[cfg(feature = "web-docs")]
pub mod web;

pub use extension::DocsExtension;
pub use imports::scan_project_imports;
pub use ingest::{
    active_dep_version, diff_dep_docs, ingest_dep_docs, ingest_migration_note, split_doc_sections,
    sweep_removed_deps, DocChunk, DEFAULT_MAX_CHUNKS_PER_DEP, DEFAULT_MAX_MIGRATION_CHUNKS,
    TABLE_DEPS, TABLE_DEP_DOCS, TABLE_DEP_MANIFESTS,
};
pub use local::{
    extract_changelog, extract_local_doc, ExtractedDoc, NotFound, MIN_LOCAL_DOC_CHARS,
};
pub use manifest::{
    detect_manifests, parse_manifest, DepKind, Dependency, DetectedManifest, Ecosystem,
};
pub use pipeline::{
    collect_gated_transitive, collect_unique_deps, ingest_manifests, sweep_removed_for_manifests,
};
pub use query::{query_dep_docs, DepDocHit};
pub use refresh::{manifest_drift, record_manifest_state, Drift};
pub use resolve::{lockfile_packages, resolve_dependencies};
#[cfg(feature = "web-docs")]
pub use web::fetch_web_doc;

/// Errors produced while detecting, parsing or resolving manifests.
#[derive(Debug, thiserror::Error)]
pub enum DocsError {
    /// A manifest or lockfile path, prefixed to a wrapped parse error.
    #[error("{context}: {source}")]
    Parse {
        /// Which file failed (path + what was being parsed).
        context: String,
        /// The underlying parse/IO failure.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A database operation (insert / list / delete) failed during ingest.
    #[error("database error: {0}")]
    Db(String),
}

impl DocsError {
    /// Wrap an error with the file path it occurred on.
    pub(crate) fn at(
        path: &std::path::Path,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        DocsError::Parse {
            context: path.display().to_string(),
            source: Box::new(source),
        }
    }
}

// ── shared row helpers ──────────────────────────────────────────────────────
//
// The dep-docs tables (`_deps`, `_dep_docs`, `_dep_manifests`) are keyed by
// content, not record id, so upserts and drift checks all reduce to "scan a
// table for rows matching a predicate". These two helpers are that scan.

/// Find the first row in `table` whose data satisfies `pred`.
pub(crate) fn find_row(
    db: &axil_core::Axil,
    table: &str,
    pred: impl Fn(&serde_json::Value) -> bool,
) -> Result<Option<axil_core::Record>, DocsError> {
    let rows = db.list(table).map_err(|e| DocsError::Db(e.to_string()))?;
    Ok(rows.into_iter().find(|row| pred(&row.data)))
}

/// Delete every row in `table` whose data satisfies `pred`; returns the count.
pub(crate) fn delete_rows_where(
    db: &axil_core::Axil,
    table: &str,
    pred: impl Fn(&serde_json::Value) -> bool,
) -> Result<usize, DocsError> {
    let rows = db.list(table).map_err(|e| DocsError::Db(e.to_string()))?;
    let mut deleted = 0;
    for row in rows {
        if pred(&row.data) {
            db.delete(&row.id)
                .map_err(|e| DocsError::Db(e.to_string()))?;
            deleted += 1;
        }
    }
    Ok(deleted)
}
