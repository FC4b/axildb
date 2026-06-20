//! Doc ingestion pipeline — chunk raw documentation text into searchable
//! `_dep_docs` records.
//!
//! This module is *source-agnostic*: it takes raw doc text (markdown or
//! plain) and produces stored, embedded, FTS-indexed chunks. Where the
//! text comes from — local extraction or web fallback — is the caller's
//! concern.
//!
//! axil-core tables are dynamic (no DDL), so the three dep-docs tables
//! exist simply by being written to under their reserved names.

use std::collections::BTreeMap;

use axil_core::Axil;
use serde_json::json;

use crate::manifest::{Dependency, Ecosystem};
use crate::DocsError;

/// One row per detected manifest file (drift detection).
pub const TABLE_DEP_MANIFESTS: &str = "_dep_manifests";
/// One row per resolved dependency.
pub const TABLE_DEPS: &str = "_deps";
/// One row per documentation chunk.
pub const TABLE_DEP_DOCS: &str = "_dep_docs";

/// Default per-dependency chunk cap, so one huge doc cannot dominate
/// recall or bloat the database.
pub const DEFAULT_MAX_CHUNKS_PER_DEP: usize = 80;

/// Per-bump cap on changelog ("migration") chunks. A changelog lists
/// its newest entries first, so the cap keeps the most recent.
pub const DEFAULT_MAX_MIGRATION_CHUNKS: usize = 16;

/// Minimum useful chunk length in bytes — shorter fragments (stray
/// headings, separators) are dropped.
const MIN_CHUNK_BYTES: usize = 24;

/// A single documentation chunk: one section of a dependency's docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocChunk {
    /// Heading breadcrumb, e.g. `"Usage > Async"`. Empty for the preamble.
    pub section_path: String,
    /// The section's text content.
    pub content: String,
}

/// Split markdown doc text into section chunks by ATX heading.
///
/// Each `#`..`######` heading starts a new chunk and extends the
/// `>`-joined breadcrumb; content before the first heading becomes a
/// preamble chunk. Heading lines inside fenced code blocks are ignored.
pub fn split_doc_sections(text: &str) -> Vec<DocChunk> {
    let mut chunks: Vec<DocChunk> = Vec::new();
    // Heading stack: (level, title) describing the current breadcrumb.
    let mut trail: Vec<(usize, String)> = Vec::new();
    let mut cur_section = String::new();
    let mut cur_body = String::new();
    let mut in_fence = false;

    let flush = |chunks: &mut Vec<DocChunk>, section: &str, body: &str| {
        let trimmed = body.trim();
        if trimmed.len() >= MIN_CHUNK_BYTES {
            chunks.push(DocChunk {
                section_path: section.to_string(),
                content: trimmed.to_string(),
            });
        }
    };

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            cur_body.push_str(line);
            cur_body.push('\n');
            continue;
        }
        let heading = if in_fence { None } else { atx_heading(trimmed) };
        if let Some((level, title)) = heading {
            // Close the chunk that ended at this heading.
            flush(&mut chunks, &cur_section, &cur_body);
            cur_body.clear();
            // Pop the breadcrumb back to this heading's parent, then push.
            while trail.last().map(|(l, _)| *l >= level).unwrap_or(false) {
                trail.pop();
            }
            trail.push((level, title));
            cur_section = trail
                .iter()
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join(" > ");
        } else {
            cur_body.push_str(line);
            cur_body.push('\n');
        }
    }
    flush(&mut chunks, &cur_section, &cur_body);
    chunks
}

/// Parse an ATX heading line (`## Title`) into `(level, title)`.
fn atx_heading(line: &str) -> Option<(usize, String)> {
    if !line.starts_with('#') {
        return None;
    }
    let level = line.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    // ATX requires a space after the `#` run (a bare `#` is not a heading).
    let rest = &line[level..];
    if !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    if title.is_empty() {
        return None;
    }
    Some((level, title.to_string()))
}

/// Approximate token count for a chunk (bytes / 4, rounded up).
fn approx_tokens(content: &str) -> usize {
    content.len().div_ceil(4)
}

/// `_deps.status` — the dependency is the current, declared version.
const STATUS_ACTIVE: &str = "active";
/// `_deps.status` — an older version, kept after a version bump.
const STATUS_SUPERSEDED: &str = "superseded";
/// `_deps.status` — the dependency is no longer declared in any manifest.
const STATUS_REMOVED: &str = "removed";

/// A `_deps` row's `status`, defaulting to `active` for rows written
/// before the field existed.
fn dep_status(data: &serde_json::Value) -> &str {
    data.get("status")
        .and_then(|v| v.as_str())
        .unwrap_or(STATUS_ACTIVE)
}

/// Whether a `_dep_docs` chunk has been archived — it documents a
/// superseded or removed version, kept for migration questions but
/// excluded from default recall.
pub(crate) fn is_archived(data: &serde_json::Value) -> bool {
    data.get("archived")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// A `_dep_docs` chunk's `doc_kind` — `"doc"` (API documentation) or
/// `"migration"` (a changelog entry). Rows written before the field
/// existed default to `"doc"`.
fn doc_kind(data: &serde_json::Value) -> &str {
    data.get("doc_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("doc")
}

/// The active `_deps` row for a dependency, if one exists. Superseded
/// and removed rows for the same name/ecosystem are skipped.
fn active_dep_row(
    db: &Axil,
    name: &str,
    ecosystem: &str,
) -> Result<Option<axil_core::Record>, DocsError> {
    crate::find_row(db, TABLE_DEPS, |d| {
        d.get("name").and_then(|v| v.as_str()) == Some(name)
            && d.get("ecosystem").and_then(|v| v.as_str()) == Some(ecosystem)
            && dep_status(d) == STATUS_ACTIVE
    })
}

/// Ingest a dependency's documentation into the database.
///
/// Splits `doc_text` into section chunks (capped at `max_chunks`),
/// writes each as a `_dep_docs` record, and upserts the dependency's
/// `_deps` row. When the database has a vector index / FTS index each
/// chunk is embedded and full-text indexed so `recall` can find it;
/// without those plugins the chunk is still stored.
///
/// Re-ingesting an *unchanged* dependency is a near-free no-op: the doc
/// text is content-hashed and, when it matches the stored `_deps` row
/// (same version), the whole pipeline — the per-chunk re-embed most of
/// all — is skipped.
///
/// A **version bump** does not discard the old version: its `_dep_docs`
/// chunks are *archived* (kept, flagged) and its `_deps` row is marked
/// `superseded` and linked to the replacement, so migration questions
/// can still reach the old API. A same-version
/// re-extract simply replaces the live chunks.
///
/// Returns the number of chunks stored (or already present, if skipped).
pub fn ingest_dep_docs(
    db: &Axil,
    dep: &Dependency,
    doc_text: &str,
    source: &str,
    max_chunks: usize,
) -> Result<usize, DocsError> {
    let version = dep
        .version
        .clone()
        .unwrap_or_else(|| "unpinned".to_string());
    let doc_hash = crate::refresh::hash_text(doc_text);

    // One small scan of `_deps` locates this dependency's active row.
    let existing = active_dep_row(db, &dep.name, dep.ecosystem.as_str())?;

    // Unchanged — same version, same doc content. Skip the whole
    // re-ingest (and the expensive per-chunk re-embed with it).
    if let Some(row) = &existing {
        let unchanged = row.data.get("doc_hash").and_then(|v| v.as_str())
            == Some(doc_hash.as_str())
            && row.data.get("version").and_then(|v| v.as_str()) == Some(version.as_str());
        if unchanged {
            let stored = row
                .data
                .get("doc_chunks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            return Ok(stored as usize);
        }
    }

    let version_changed = existing
        .as_ref()
        .and_then(|r| r.data.get("version").and_then(|v| v.as_str()))
        .map(|old| old != version)
        .unwrap_or(false);

    if version_changed {
        // A bump archives the previous version's chunks rather than
        // deleting them — version history must survive.
        archive_dep_chunks(db, &dep.name, dep.ecosystem.as_str())?;
    } else if existing.is_some() {
        // Same version, re-extracted: the live doc chunks are genuinely
        // stale. Replace only those — chunks archived by an earlier
        // bump stay archived, and migration notes are left intact.
        crate::delete_rows_where(db, TABLE_DEP_DOCS, |d| {
            d.get("dep_name").and_then(|v| v.as_str()) == Some(dep.name.as_str())
                && d.get("ecosystem").and_then(|v| v.as_str()) == Some(dep.ecosystem.as_str())
                && !is_archived(d)
                && doc_kind(d) == "doc"
        })?;
    }

    let mut chunks = split_doc_sections(doc_text);
    chunks.truncate(max_chunks);
    let chunk_count = chunks.len();

    for (idx, chunk) in chunks.iter().enumerate() {
        let data = json!({
            "dep_name": dep.name,
            "dep_version": version,
            "ecosystem": dep.ecosystem.as_str(),
            "section_path": chunk.section_path,
            "content": chunk.content,
            "tokens": approx_tokens(&chunk.content),
            "source": source,
            "chunk_index": idx,
            "archived": false,
            "_memory_type": "reference",
        });
        let record = db
            .insert(TABLE_DEP_DOCS, data)
            .map_err(|e| DocsError::Db(e.to_string()))?;
        // `_`-prefixed tables skip the auto insert-hooks, so drive
        // embedding + FTS indexing explicitly. A missing vector/FTS
        // plugin is not an error — the chunk is still stored.
        let _ = db.embed_field(&record.id, "content");
        let _ = db.index_text(&record.id, "content", &chunk.content);
    }

    write_dep_row(
        db,
        dep,
        &version,
        chunk_count,
        source,
        &doc_hash,
        existing.as_ref(),
        version_changed,
    )?;
    Ok(chunk_count)
}

/// Flag every live `_dep_docs` chunk for a dependency as archived.
/// Returns the number of chunks archived.
fn archive_dep_chunks(db: &Axil, name: &str, ecosystem: &str) -> Result<usize, DocsError> {
    let rows = db
        .list(TABLE_DEP_DOCS)
        .map_err(|e| DocsError::Db(e.to_string()))?;
    let mut archived = 0;
    for row in rows {
        let matches = row.data.get("dep_name").and_then(|v| v.as_str()) == Some(name)
            && row.data.get("ecosystem").and_then(|v| v.as_str()) == Some(ecosystem)
            && !is_archived(&row.data);
        if !matches {
            continue;
        }
        let mut data = row.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("archived".to_string(), json!(true));
        }
        db.update(&row.id, data)
            .map_err(|e| DocsError::Db(e.to_string()))?;
        archived += 1;
    }
    Ok(archived)
}

/// Write the dependency's `_deps` row.
///
/// A version bump retains `existing` — marked `superseded` and linked
/// to the new row (a `superseded_by` field plus a best-effort graph
/// edge) — so version history survives. A same-version re-extract
/// simply replaces the row.
#[allow(clippy::too_many_arguments)]
fn write_dep_row(
    db: &Axil,
    dep: &Dependency,
    version: &str,
    chunk_count: usize,
    source: &str,
    doc_hash: &str,
    existing: Option<&axil_core::Record>,
    version_changed: bool,
) -> Result<(), DocsError> {
    let data = json!({
        "name": dep.name,
        "version": version,
        "ecosystem": dep.ecosystem.as_str(),
        "kind": dep.kind.as_str(),
        "declared_range": dep.declared_range,
        "doc_chunks": chunk_count,
        "doc_source": source,
        "doc_hash": doc_hash,
        "status": STATUS_ACTIVE,
    });
    let new_row = db
        .insert(TABLE_DEPS, data)
        .map_err(|e| DocsError::Db(e.to_string()))?;

    match existing {
        // Version bump — keep the old row, mark it superseded and link
        // it to the replacement.
        Some(old) if version_changed => {
            let mut old_data = old.data.clone();
            if let Some(obj) = old_data.as_object_mut() {
                obj.insert("status".to_string(), json!(STATUS_SUPERSEDED));
                obj.insert("superseded_by".to_string(), json!(new_row.id.to_string()));
            }
            db.update(&old.id, old_data)
                .map_err(|e| DocsError::Db(e.to_string()))?;
            // The graph edge is best-effort: a database opened without
            // the graph plugin simply skips it.
            let _ = db.relate(&old.id, "superseded_by", &new_row.id, None);
        }
        // Same version, re-extracted — the old row is genuinely stale.
        Some(old) => {
            db.delete(&old.id)
                .map_err(|e| DocsError::Db(e.to_string()))?;
        }
        None => {}
    }
    Ok(())
}

/// Sweep dependencies that are no longer declared in any manifest.
///
/// Any `active` `_deps` row whose `(name, ecosystem)` is absent from
/// `current` is marked `removed` and its doc chunks archived — kept,
/// not deleted, so the agent can still recall a dropped library's
/// docs. Returns the names swept. This is the *removed* set of the
/// three-way drift diff; `ingest_dep_docs` covers *added* and
/// *version-changed*.
pub fn sweep_removed_deps(db: &Axil, current: &[Dependency]) -> Result<Vec<String>, DocsError> {
    use std::collections::HashSet;
    let current_keys: HashSet<(String, String)> = current
        .iter()
        .map(|d| (d.name.clone(), d.ecosystem.as_str().to_string()))
        .collect();

    let rows = db
        .list(TABLE_DEPS)
        .map_err(|e| DocsError::Db(e.to_string()))?;
    let mut removed = Vec::new();
    for row in rows {
        if dep_status(&row.data) != STATUS_ACTIVE {
            continue;
        }
        let name = row.data.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let ecosystem = row
            .data
            .get("ecosystem")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name.is_empty() || current_keys.contains(&(name.to_string(), ecosystem.to_string())) {
            continue;
        }
        let mut data = row.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("status".to_string(), json!(STATUS_REMOVED));
        }
        db.update(&row.id, data)
            .map_err(|e| DocsError::Db(e.to_string()))?;
        archive_dep_chunks(db, name, ecosystem)?;
        removed.push(name.to_string());
    }
    Ok(removed)
}

/// The active `_deps` row's resolved version for a dependency, if it
/// has already been ingested. Lets a caller detect a version bump
/// *before* re-ingesting.
pub fn active_dep_version(
    db: &Axil,
    name: &str,
    ecosystem: Ecosystem,
) -> Result<Option<String>, DocsError> {
    let row = active_dep_row(db, name, ecosystem.as_str())?;
    Ok(row.and_then(|r| {
        r.data
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    }))
}

/// Ingest a dependency's changelog as `migration`-tagged `_dep_docs`
/// chunks.
///
/// Called on a version bump: `changelog_text` — read from the on-disk
/// dependency copy — is split into sections, capped at
/// [`DEFAULT_MAX_MIGRATION_CHUNKS`], and stored alongside the new
/// version's regular doc chunks. Each chunk carries `doc_kind:
/// "migration"` (so recall can label it a changelog entry) and
/// `from_version` (the version bumped away from). Migration chunks
/// survive a same-version doc re-extract and are archived, like any
/// chunk, by the next version bump. Returns the number stored.
pub fn ingest_migration_note(
    db: &Axil,
    dep: &Dependency,
    from_version: &str,
    changelog_text: &str,
) -> Result<usize, DocsError> {
    let version = dep
        .version
        .clone()
        .unwrap_or_else(|| "unpinned".to_string());
    let mut chunks = split_doc_sections(changelog_text);
    chunks.truncate(DEFAULT_MAX_MIGRATION_CHUNKS);
    let chunk_count = chunks.len();

    for (idx, chunk) in chunks.iter().enumerate() {
        let data = json!({
            "dep_name": dep.name,
            "dep_version": version,
            "ecosystem": dep.ecosystem.as_str(),
            "section_path": chunk.section_path,
            "content": chunk.content,
            "tokens": approx_tokens(&chunk.content),
            "source": "local",
            "chunk_index": idx,
            "archived": false,
            "doc_kind": "migration",
            "from_version": from_version,
            "_memory_type": "reference",
        });
        let record = db
            .insert(TABLE_DEP_DOCS, data)
            .map_err(|e| DocsError::Db(e.to_string()))?;
        let _ = db.embed_field(&record.id, "content");
        let _ = db.index_text(&record.id, "content", &chunk.content);
    }
    Ok(chunk_count)
}

/// Append a `- item` bullet list under `label` to `out`, skipping an
/// empty list. An empty section path renders as `(preamble)`.
fn append_diff_list(out: &mut String, label: &str, items: &[&str]) {
    if items.is_empty() {
        return;
    }
    out.push('\n');
    out.push_str(label);
    out.push_str(":\n");
    for it in items {
        out.push_str("- ");
        out.push_str(if it.is_empty() { "(preamble)" } else { it });
        out.push('\n');
    }
}

/// Diff a dependency's documentation across a version bump and store
/// the section-level delta as a `doc_kind: "doc_diff"` chunk
///.
///
/// Compares the now-archived `from_version` doc chunks against the
/// freshly-ingested current-version ones, keyed by section breadcrumb,
/// and records which sections were added, removed or changed. Unlike
/// the dependency's own changelog ([`ingest_migration_note`]), this is
/// the *observed* doc delta — it catches changes the authors never
/// wrote up. Returns 1 if a diff chunk was stored, or 0 when the docs
/// were unchanged or the old version's chunks are unavailable.
pub fn diff_dep_docs(db: &Axil, dep: &Dependency, from_version: &str) -> Result<usize, DocsError> {
    let to_version = dep
        .version
        .clone()
        .unwrap_or_else(|| "unpinned".to_string());
    if from_version == to_version {
        return Ok(0);
    }

    let rows = db
        .list(TABLE_DEP_DOCS)
        .map_err(|e| DocsError::Db(e.to_string()))?;
    // Section breadcrumb → content, for each side of the bump.
    let mut old: BTreeMap<String, String> = BTreeMap::new();
    let mut new: BTreeMap<String, String> = BTreeMap::new();
    for row in &rows {
        let d = &row.data;
        let same_dep = d.get("dep_name").and_then(|v| v.as_str()) == Some(dep.name.as_str())
            && d.get("ecosystem").and_then(|v| v.as_str()) == Some(dep.ecosystem.as_str());
        if !same_dep || doc_kind(d) != "doc" {
            continue;
        }
        let version = d.get("dep_version").and_then(|v| v.as_str()).unwrap_or("");
        let section = d
            .get("section_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = d
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if version == from_version && is_archived(d) {
            old.insert(section, content);
        } else if version == to_version && !is_archived(d) {
            new.insert(section, content);
        }
    }
    // Without the old version's chunks there is nothing to diff against
    // (e.g. it was ingested before version history was retained).
    if old.is_empty() {
        return Ok(0);
    }

    let mut added: Vec<&str> = Vec::new();
    let mut removed: Vec<&str> = Vec::new();
    let mut changed: Vec<&str> = Vec::new();
    for section in new.keys() {
        if !old.contains_key(section) {
            added.push(section.as_str());
        }
    }
    for section in old.keys() {
        if !new.contains_key(section) {
            removed.push(section.as_str());
        }
    }
    for (section, content) in &new {
        if let Some(old_content) = old.get(section) {
            if old_content != content {
                changed.push(section.as_str());
            }
        }
    }
    if added.is_empty() && removed.is_empty() && changed.is_empty() {
        return Ok(0);
    }

    let mut body = format!(
        "Documentation changes for {} {from_version} \u{2192} {to_version}.\n",
        dep.name
    );
    append_diff_list(&mut body, "Added sections", &added);
    append_diff_list(&mut body, "Removed sections", &removed);
    append_diff_list(&mut body, "Changed sections", &changed);

    let data = json!({
        "dep_name": dep.name,
        "dep_version": to_version,
        "ecosystem": dep.ecosystem.as_str(),
        "section_path": format!("Doc changes {from_version} \u{2192} {to_version}"),
        "content": body,
        "tokens": approx_tokens(&body),
        "source": "local",
        "chunk_index": 0,
        "archived": false,
        "doc_kind": "doc_diff",
        "from_version": from_version,
        "_memory_type": "reference",
    });
    let record = db
        .insert(TABLE_DEP_DOCS, data)
        .map_err(|e| DocsError::Db(e.to_string()))?;
    let _ = db.embed_field(&record.id, "content");
    let _ = db.index_text(&record.id, "content", &body);
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{DepKind, Dependency, Ecosystem};

    fn sample_dep() -> Dependency {
        Dependency {
            name: "serde".to_string(),
            ecosystem: Ecosystem::Cargo,
            kind: DepKind::Direct,
            declared_range: "1".to_string(),
            version: Some("1.0.210".to_string()),
        }
    }

    #[test]
    fn splits_markdown_into_breadcrumbed_sections() {
        let text = "\
Intro paragraph that is long enough to keep.

# Serde

Serde is a serialization framework for Rust.

## Usage

Derive `Serialize` on your type.

### Async

Some deeper detail about async usage here.
";
        let chunks = split_doc_sections(text);
        let sections: Vec<&str> = chunks.iter().map(|c| c.section_path.as_str()).collect();
        assert!(sections.contains(&""), "preamble chunk present");
        assert!(sections.contains(&"Serde"));
        assert!(sections.contains(&"Serde > Usage"));
        assert!(sections.contains(&"Serde > Usage > Async"));
    }

    #[test]
    fn ignores_headings_inside_code_fences() {
        let text = "\
# Real Heading

Body text long enough to survive the minimum-length filter.

```
# not a heading
also fenced content
```
";
        let chunks = split_doc_sections(text);
        assert!(chunks.iter().all(|c| c.section_path != "not a heading"));
    }

    #[test]
    fn ingest_writes_embeds_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let dep = sample_dep();
        let text = "\
# Serde

Serde is a framework for serializing and deserializing Rust data.

## Derive

Use the derive macro to implement the traits automatically.
";
        let n = ingest_dep_docs(&db, &dep, text, "local", 80).unwrap();
        assert!(n >= 2, "expected the heading sections to chunk");
        assert_eq!(db.list(TABLE_DEP_DOCS).unwrap().len(), n);
        assert_eq!(db.list(TABLE_DEPS).unwrap().len(), 1);

        // Re-ingest: same counts, no duplicate rows.
        let n2 = ingest_dep_docs(&db, &dep, text, "local", 80).unwrap();
        assert_eq!(n2, n);
        assert_eq!(db.list(TABLE_DEP_DOCS).unwrap().len(), n);
        assert_eq!(db.list(TABLE_DEPS).unwrap().len(), 1);
    }

    #[test]
    fn ingest_respects_the_chunk_cap() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let mut text = String::new();
        for i in 0..20 {
            text.push_str(&format!(
                "## Section {i}\n\nContent for section {i}, long enough to keep.\n\n"
            ));
        }
        let n = ingest_dep_docs(&db, &sample_dep(), &text, "local", 5).unwrap();
        assert_eq!(n, 5, "chunk cap must be honoured");
        assert_eq!(db.list(TABLE_DEP_DOCS).unwrap().len(), 5);
    }

    #[test]
    fn unchanged_reingest_skips_changed_reingest_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let dep = sample_dep();
        let v1 = "# Alpha\n\nThe first body, comfortably past the minimum length.\n";
        let n1 = ingest_dep_docs(&db, &dep, v1, "local", 80).unwrap();
        assert!(n1 >= 1);

        // Re-ingest identical text — skipped: the very same rows
        // survive (not deleted and recreated).
        let row_ids = |db: &axil_core::Axil| -> Vec<String> {
            let mut ids: Vec<String> = db
                .list(TABLE_DEP_DOCS)
                .unwrap()
                .iter()
                .map(|r| r.id.to_string())
                .collect();
            ids.sort();
            ids
        };
        let ids_before = row_ids(&db);
        assert_eq!(ingest_dep_docs(&db, &dep, v1, "local", 80).unwrap(), n1);
        assert_eq!(
            row_ids(&db),
            ids_before,
            "unchanged re-ingest must not recreate rows"
        );

        // Changed text — old chunks replaced, not appended.
        let v2 = "# Alpha\n\nA rewritten first body, still well past the minimum.\n\n\
                  ## Beta\n\nA brand new second section, also long enough to keep.\n";
        let n2 = ingest_dep_docs(&db, &dep, v2, "local", 80).unwrap();
        assert_eq!(db.list(TABLE_DEP_DOCS).unwrap().len(), n2);
        assert!(n2 > n1, "the changed text added a section");
    }

    #[test]
    fn version_bump_archives_old_chunks_and_supersedes() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let mut dep = sample_dep(); // serde 1.0.210
        let v1 = "# Serde\n\nThe original serde documentation, long enough to keep.\n";
        let n1 = ingest_dep_docs(&db, &dep, v1, "local", 80).unwrap();
        assert!(n1 >= 1);

        // Bump the version — old chunks must survive, archived.
        dep.version = Some("1.0.999".to_string());
        let v2 = "# Serde\n\nRewritten docs for the new release, also comfortably long.\n";
        let n2 = ingest_dep_docs(&db, &dep, v2, "local", 80).unwrap();

        let docs = db.list(TABLE_DEP_DOCS).unwrap();
        let archived = docs.iter().filter(|r| is_archived(&r.data)).count();
        let live = docs.iter().filter(|r| !is_archived(&r.data)).count();
        assert_eq!(archived, n1, "every old-version chunk is kept, archived");
        assert_eq!(live, n2, "the new version's chunks are live");

        // `_deps`: the old row superseded + linked, the new row active.
        let deps = db.list(TABLE_DEPS).unwrap();
        assert_eq!(deps.len(), 2, "the old version's row is retained");
        let old = deps
            .iter()
            .find(|r| r.data.get("version").and_then(|v| v.as_str()) == Some("1.0.210"))
            .unwrap();
        assert_eq!(dep_status(&old.data), STATUS_SUPERSEDED);
        assert!(
            old.data.get("superseded_by").is_some(),
            "superseded row links to its replacement"
        );
        let new = deps
            .iter()
            .find(|r| r.data.get("version").and_then(|v| v.as_str()) == Some("1.0.999"))
            .unwrap();
        assert_eq!(dep_status(&new.data), STATUS_ACTIVE);
    }

    #[test]
    fn sweep_marks_dropped_dependency_removed() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let serde = sample_dep();
        let mut tokio = sample_dep();
        tokio.name = "tokio".to_string();
        let text = "# Lib\n\nSome documentation body, comfortably long enough to keep.\n";
        let serde_chunks = ingest_dep_docs(&db, &serde, text, "local", 80).unwrap();
        ingest_dep_docs(&db, &tokio, text, "local", 80).unwrap();

        // The next sync sees only serde — tokio was dropped from the manifest.
        let removed = sweep_removed_deps(&db, std::slice::from_ref(&serde)).unwrap();
        assert_eq!(removed, vec!["tokio".to_string()]);

        let deps = db.list(TABLE_DEPS).unwrap();
        let status_of = |name: &str| {
            deps.iter()
                .find(|r| r.data.get("name").and_then(|v| v.as_str()) == Some(name))
                .map(|r| dep_status(&r.data).to_string())
                .unwrap()
        };
        assert_eq!(status_of("tokio"), STATUS_REMOVED);
        assert_eq!(status_of("serde"), STATUS_ACTIVE);

        // tokio's chunks are archived; serde's stay live.
        let docs = db.list(TABLE_DEP_DOCS).unwrap();
        let all_archived = |name: &str, want: bool| {
            docs.iter()
                .filter(|r| r.data.get("dep_name").and_then(|v| v.as_str()) == Some(name))
                .all(|r| is_archived(&r.data) == want)
        };
        assert!(
            all_archived("tokio", true),
            "dropped dep's chunks are archived"
        );
        assert!(all_archived("serde", false), "kept dep's chunks stay live");
        assert!(serde_chunks >= 1);
    }

    #[test]
    fn migration_note_stored_and_survives_a_doc_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let dep = sample_dep(); // serde 1.0.210
        let docs = "# Serde\n\nThe serde docs, long enough to comfortably keep.\n";
        ingest_dep_docs(&db, &dep, docs, "local", 80).unwrap();
        assert_eq!(
            active_dep_version(&db, "serde", Ecosystem::Cargo)
                .unwrap()
                .as_deref(),
            Some("1.0.210")
        );

        // Store a migration note for the bump into 1.0.210.
        let changelog = "## 1.0.210\n\nFixed a soundness hole in the derive macro.\n";
        let m = ingest_migration_note(&db, &dep, "1.0.200", changelog).unwrap();
        assert!(m >= 1);
        let migration_rows: Vec<_> = db
            .list(TABLE_DEP_DOCS)
            .unwrap()
            .into_iter()
            .filter(|r| doc_kind(&r.data) == "migration")
            .collect();
        assert_eq!(migration_rows.len(), m);
        assert_eq!(
            migration_rows[0]
                .data
                .get("from_version")
                .and_then(|v| v.as_str()),
            Some("1.0.200")
        );

        // A same-version doc re-extract replaces doc chunks but leaves
        // the migration note untouched.
        let docs2 = "# Serde\n\nRewritten serde docs, also comfortably long enough.\n";
        ingest_dep_docs(&db, &dep, docs2, "local", 80).unwrap();
        let still_there = db
            .list(TABLE_DEP_DOCS)
            .unwrap()
            .into_iter()
            .filter(|r| doc_kind(&r.data) == "migration")
            .count();
        assert_eq!(still_there, m, "migration notes survive a doc refresh");
    }

    #[test]
    fn doc_diff_records_added_removed_and_changed_sections() {
        let dir = tempfile::tempdir().unwrap();
        let db = axil_core::Axil::open(dir.path().join("m.axil"))
            .build()
            .unwrap();
        let mut dep = sample_dep(); // serde 1.0.210
        let v1 = "# Serde\n\nThe serde overview, long enough to keep comfortably.\n\n\
                  ## Legacy API\n\nThe old legacy API section, also long enough to keep.\n";
        ingest_dep_docs(&db, &dep, v1, "local", 80).unwrap();

        // Bump: the overview is rewritten, Legacy API drops, Async is new.
        dep.version = Some("2.0.0".to_string());
        let v2 = "# Serde\n\nThe serde overview REWRITTEN, still long enough to keep.\n\n\
                  ## Async\n\nA brand new async section, comfortably long enough to keep.\n";
        ingest_dep_docs(&db, &dep, v2, "local", 80).unwrap();

        let stored = diff_dep_docs(&db, &dep, "1.0.210").unwrap();
        assert_eq!(stored, 1, "a diff chunk is stored");

        let diff = db
            .list(TABLE_DEP_DOCS)
            .unwrap()
            .into_iter()
            .find(|r| doc_kind(&r.data) == "doc_diff")
            .unwrap();
        let body = diff.data.get("content").and_then(|v| v.as_str()).unwrap();
        assert!(body.contains("Added sections") && body.contains("Serde > Async"));
        assert!(body.contains("Removed sections") && body.contains("Serde > Legacy API"));
        assert!(body.contains("Changed sections"), "the rewritten overview");
        assert_eq!(
            diff.data.get("from_version").and_then(|v| v.as_str()),
            Some("1.0.210")
        );

        // Identical docs across a bump → no diff chunk.
        dep.version = Some("2.0.0".to_string());
        let n = diff_dep_docs(&db, &dep, "2.0.0").unwrap();
        assert_eq!(n, 0, "same version is not a diff");
    }
}
