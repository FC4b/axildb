//! Portable memory export/import — mergeable JSONL for moving one database's
//! memory to another machine or teammate through git or a file share.
//!
//! This is a *record-level* transport, deliberately distinct from
//! [`crate::branch`] / [`crate::snapshot`], which take a binary, whole-file copy
//! of every `.axil*` companion. A branch/snapshot is a point-in-time clone you
//! restore *over* a database (replacing it); an export is a stream of individual
//! records and edges you *merge into* an existing database, so two developers'
//! memories can be combined. That is the pre-team-sync stopgap: commit an export
//! file, a teammate imports it with `--dedup`, and both memories converge.
//!
//! ## What travels, what does not
//!
//! - **Records** and the **graph edges** between them travel as JSONL lines.
//! - **Embeddings do not travel.** Vectors are machine-local ONNX artifacts, so
//!   every imported record is re-embedded through the normal insert path on the
//!   destination. Full-text and `code_refs` indexes are likewise rebuilt on
//!   insert. The wire format therefore stays small and portable.
//! - **Rebuilt index tables** (`_idx_*`) are never exported — they are derived
//!   from the records themselves and regenerated on import.
//!
//! ## Determinism
//!
//! Output is fully deterministic: records and edges are emitted in ULID order
//! and object keys are canonicalized for the dedup hash. No timestamps are
//! invented — the header carries only counts, and every record keeps its own
//! `created_at`. A re-export of an unchanged database yields a byte-identical
//! file, so it diffs cleanly in git.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::db::Axil;
use crate::error::{AxilError, Result};
use crate::plugin::Direction;
use crate::record::{Record, RecordId};

/// Wire-format identifier written in the header line.
pub const FORMAT: &str = "axil-export-jsonl";

/// Wire-format version. Bumped only on a breaking change to the line schema.
pub const FORMAT_VERSION: u32 = 1;

/// True for tables that are rebuilt from the records themselves and must never
/// travel in an export (they are regenerated on import).
///
/// Only the `_idx_*` reverse indexes qualify: `_idx_code_refs` and
/// `_idx_code_proxies` are reconstructed from records / SCIP ingest.
pub fn is_rebuilt_index_table(table: &str) -> bool {
    table.starts_with("_idx_")
}

/// Whether a table should be included in an export given the options.
fn table_selected(table: &str, opts: &ExportOptions) -> bool {
    // Rebuilt indexes never travel — they are derived on import.
    if is_rebuilt_index_table(table) {
        return false;
    }
    // An explicit allowlist wins over the system/user heuristic.
    if let Some(ref allow) = opts.tables {
        return allow.iter().any(|t| t == table);
    }
    // Default: user memory tables only. System tables (prefix `_`) require the
    // explicit opt-in, since re-importing them is only safe for a subset.
    if table.starts_with('_') {
        return opts.include_system;
    }
    true
}

/// Options controlling what an export emits.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Explicit table allowlist. When `Some`, exactly these tables are exported
    /// (still excluding `_idx_*`); when `None`, the user/system heuristic applies.
    pub tables: Option<Vec<String>>,
    /// Only export records whose `created_at` is at or after this instant.
    pub since: Option<DateTime<Utc>>,
    /// Include system tables (prefix `_`, except the always-skipped `_idx_*`).
    pub include_system: bool,
}

/// Summary of what an export wrote.
#[derive(Debug, Clone, Serialize)]
pub struct ExportStats {
    /// Number of record lines written.
    pub records: usize,
    /// Number of edge lines written.
    pub edges: usize,
    /// Number of distinct tables covered.
    pub tables: usize,
}

/// Options controlling an import.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Skip a record whose id already exists, or whose canonical content hash
    /// matches an existing record in the same table.
    pub dedup: bool,
    /// Compute and report the outcome without writing anything.
    pub dry_run: bool,
}

/// Post-import verification of the embedding index.
///
/// Auto-embedding on insert is deliberately best-effort — a mid-import
/// embedder failure must never lose the record — which means an import can
/// finish with records stored but not semantically searchable. This block
/// surfaces that state in the report instead of leaving it silent: any
/// `missing` (or an `engine_unavailable` with `affected > 0`) means the
/// destination needs `axil heal --reindex` once its embedder is healthy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EmbeddingVerification {
    /// A vector index and embedder were attached; every imported record that
    /// should have been embedded was checked against the index by id.
    Verified {
        /// Imported records that met the auto-embed condition.
        expected: usize,
        /// How many of those have a stored vector.
        indexed: usize,
        /// `expected - indexed` — stored but not semantically searchable.
        missing: usize,
    },
    /// No vector index or no embedder was attached at import time, so
    /// `affected` embeddable records were imported without embeddings.
    EngineUnavailable { affected: usize },
    /// Dry run — nothing was written, so there was nothing to verify.
    SkippedDryRun,
}

/// Outcome of an import.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportReport {
    /// Records inserted (or, in a dry run, that would be inserted).
    pub imported: usize,
    /// Records skipped because their id already existed (`--dedup`).
    pub skipped_id: usize,
    /// Records skipped because a same-content record already existed (`--dedup`).
    pub skipped_dup: usize,
    /// Edges created between resolvable endpoints.
    pub edges_created: usize,
    /// Edges skipped: a missing endpoint, or a duplicate under `--dedup`.
    pub edges_skipped: usize,
    /// Records whose id had to be remapped on insert. Always 0 — the importer
    /// preserves original ids so checkpoint `references[]` and `code_refs`
    /// survive the round trip. Surfaced for honesty if that ever changes.
    pub id_remapped: usize,
    /// Post-import embedding verification. `None` only in intermediate states;
    /// [`import_from_reader`] always fills it before returning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<EmbeddingVerification>,
}

// ── Wire format ─────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct Header {
    kind: String,
    format: String,
    format_version: u32,
    axil_version: String,
    record_count: usize,
    edge_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct RecordLine {
    kind: String,
    table: String,
    id: String,
    data: Value,
    created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    updated_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EdgeLine {
    kind: String,
    from: String,
    edge_type: String,
    to: String,
    #[serde(default)]
    props: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
}

// ── Export ──────────────────────────────────────────────────────────

/// Export selected records and their inter-record edges as JSONL to `writer`.
///
/// The first line is a `{"kind":"header",...}` object carrying the format
/// version, the running `axil` crate version, and record/edge counts. Record
/// lines follow in ULID order, then edge lines (only edges whose *both*
/// endpoints are in the exported set) in edge-ULID order. Embeddings are never
/// written; they are rebuilt on import.
pub fn export_to_writer<W: Write>(
    db: &Axil,
    opts: &ExportOptions,
    writer: &mut W,
) -> Result<ExportStats> {
    // Collect selected records, sorted by id (ULIDs sort chronologically) so the
    // output is deterministic and git-diffable.
    let mut tables: Vec<String> = db
        .tables()?
        .into_iter()
        .filter(|t| table_selected(t, opts))
        .collect();
    tables.sort();

    let mut records: Vec<Record> = Vec::new();
    for table in &tables {
        for record in db.list(table)? {
            if let Some(since) = opts.since {
                if record.created_at < since {
                    continue;
                }
            }
            records.push(record);
        }
    }
    records.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));

    let exported_ids: HashSet<&str> = records.iter().map(|r| r.id.as_str()).collect();

    // Gather edges among exported records. Each edge is keyed by its own id, and
    // iterating the outgoing edges of every record visits each edge exactly once
    // (an edge has a single `from`). Keep only edges whose target is also in the
    // exported set so imported edges never dangle.
    let mut edges: Vec<EdgeLine> = Vec::new();
    if db.has_graph_index() {
        for record in &records {
            for e in db.edges(&record.id, None, Direction::Out)? {
                if exported_ids.contains(e.to.as_str()) {
                    edges.push(EdgeLine {
                        kind: "edge".to_string(),
                        from: e.from.to_string(),
                        edge_type: e.edge_type,
                        to: e.to.to_string(),
                        props: e.properties,
                        created_at: Some(e.created_at),
                    });
                }
            }
        }
        // Deterministic edge order: by (from, type, to).
        edges.sort_by(|a, b| {
            (a.from.as_str(), a.edge_type.as_str(), a.to.as_str()).cmp(&(
                b.from.as_str(),
                b.edge_type.as_str(),
                b.to.as_str(),
            ))
        });
    }

    let header = Header {
        kind: "header".to_string(),
        format: FORMAT.to_string(),
        format_version: FORMAT_VERSION,
        axil_version: env!("CARGO_PKG_VERSION").to_string(),
        record_count: records.len(),
        edge_count: edges.len(),
    };
    write_line(writer, &header)?;

    for record in &records {
        let line = RecordLine {
            kind: "record".to_string(),
            table: record.table.clone(),
            id: record.id.to_string(),
            data: record.data.clone(),
            created_at: record.created_at.to_rfc3339(),
            updated_at: Some(record.updated_at.to_rfc3339()),
            metadata: record.metadata.clone(),
        };
        write_line(writer, &line)?;
    }

    for edge in &edges {
        write_line(writer, edge)?;
    }

    Ok(ExportStats {
        records: records.len(),
        edges: edges.len(),
        tables: tables.len(),
    })
}

fn write_line<W: Write, T: Serialize>(writer: &mut W, value: &T) -> Result<()> {
    let s = serde_json::to_string(value)?;
    writer
        .write_all(s.as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(|e| AxilError::plugin(format!("failed to write export line: {e}")))
}

// ── Import ──────────────────────────────────────────────────────────

/// Import records and edges from a JSONL `reader` produced by
/// [`export_to_writer`].
///
/// Records are recreated through the normal insert path so every engine fires
/// (embedding, FTS, graph auto-link, `code_refs`), and their **original ids are
/// preserved** so cross-references survive. Edges are recreated with
/// [`Axil::relate`]; an edge whose endpoint was not imported (and does not
/// already exist) is skipped rather than dangling.
///
/// With `opts.dedup`, a record is skipped when its id already exists or when a
/// same-content record already exists in the same table; a duplicate edge is
/// likewise skipped. With `opts.dry_run`, nothing is written but the report
/// reflects what would happen.
pub fn import_from_reader<R: BufRead>(
    db: &Axil,
    opts: &ImportOptions,
    reader: R,
) -> Result<ImportReport> {
    // Pre-scan existing state so dedup can consult it without a per-record query.
    // `present_ids` also tracks ids added during this import so intra-file
    // duplicates and edge endpoints resolve correctly (even in a dry run).
    let mut present_ids: HashSet<String> = HashSet::new();
    let mut content_hashes: HashMap<String, HashSet<String>> = HashMap::new();
    for table in db.tables()? {
        for record in db.list(&table)? {
            present_ids.insert(record.id.to_string());
            if opts.dedup {
                content_hashes
                    .entry(table.clone())
                    .or_default()
                    .insert(content_hash(&record.data));
            }
        }
    }

    let mut report = ImportReport::default();
    let mut header_seen = false;
    let mut edge_lines: Vec<EdgeLine> = Vec::new();
    // Ids of imported records that met the auto-embed condition, verified
    // against the vector index after the loop.
    let mut embeddable: Vec<RecordId> = Vec::new();

    for (lineno, line) in reader.lines().enumerate() {
        let line = line
            .map_err(|e| AxilError::plugin(format!("failed to read import line {lineno}: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed).map_err(|e| {
            AxilError::InvalidQuery(format!("malformed JSONL at line {}: {e}", lineno + 1))
        })?;
        let kind = value.get("kind").and_then(|k| k.as_str()).unwrap_or("");

        match kind {
            "header" => {
                let header: Header = serde_json::from_value(value).map_err(|e| {
                    AxilError::InvalidQuery(format!("invalid export header: {e}"))
                })?;
                if header.format != FORMAT {
                    return Err(AxilError::InvalidQuery(format!(
                        "unrecognized export format '{}' (expected '{FORMAT}')",
                        header.format
                    )));
                }
                if header.format_version > FORMAT_VERSION {
                    return Err(AxilError::InvalidQuery(format!(
                        "export format version {} is newer than supported version {FORMAT_VERSION} \
                         — upgrade axil to import this file",
                        header.format_version
                    )));
                }
                header_seen = true;
            }
            "record" => {
                let line: RecordLine = serde_json::from_value(value).map_err(|e| {
                    AxilError::InvalidQuery(format!("invalid record line: {e}"))
                })?;
                // Rebuilt indexes are regenerated on insert — never import them.
                if is_rebuilt_index_table(&line.table) {
                    continue;
                }
                import_record(
                    db,
                    opts,
                    &line,
                    &mut present_ids,
                    &mut content_hashes,
                    &mut embeddable,
                    &mut report,
                )?;
            }
            "edge" => {
                // Buffer edges; apply them after every record line is processed so
                // an edge can reference a record defined later in the stream.
                let line: EdgeLine = serde_json::from_value(value)
                    .map_err(|e| AxilError::InvalidQuery(format!("invalid edge line: {e}")))?;
                edge_lines.push(line);
            }
            other => {
                return Err(AxilError::InvalidQuery(format!(
                    "unknown line kind '{other}' at line {}",
                    lineno + 1
                )));
            }
        }
    }

    if !header_seen {
        return Err(AxilError::InvalidQuery(
            "missing export header — is this an axil export file?".to_string(),
        ));
    }

    for edge in &edge_lines {
        import_edge(db, opts, edge, &present_ids, &mut report)?;
    }

    // Self-verification: importing is only half the contract — the records
    // must also be *findable*. Embedding is the one index that can silently
    // fail (best-effort by design), so check it explicitly and put the result
    // in the report rather than leaving the gap for the user to discover as
    // weaker recall.
    report.embeddings = Some(if opts.dry_run {
        EmbeddingVerification::SkippedDryRun
    } else if db.has_vector_index() && db.has_embedder() {
        let indexed = embeddable.iter().filter(|id| db.has_embedding(id)).count();
        EmbeddingVerification::Verified {
            expected: embeddable.len(),
            indexed,
            missing: embeddable.len() - indexed,
        }
    } else {
        EmbeddingVerification::EngineUnavailable {
            affected: embeddable.len(),
        }
    });

    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn import_record(
    db: &Axil,
    opts: &ImportOptions,
    line: &RecordLine,
    present_ids: &mut HashSet<String>,
    content_hashes: &mut HashMap<String, HashSet<String>>,
    embeddable: &mut Vec<RecordId>,
    report: &mut ImportReport,
) -> Result<()> {
    if opts.dedup && present_ids.contains(&line.id) {
        report.skipped_id += 1;
        return Ok(());
    }

    let hash = content_hash(&line.data);
    if opts.dedup {
        if let Some(hashes) = content_hashes.get(&line.table) {
            if hashes.contains(&hash) {
                report.skipped_dup += 1;
                return Ok(());
            }
        }
    }

    if !opts.dry_run {
        let record = build_record(line)?;
        let track = expects_embedding(&line.table, &line.data).then(|| record.id.clone());
        db.insert_preserving(record)?;
        if let Some(id) = track {
            embeddable.push(id);
        }
    }

    // Track the just-imported record so later lines and edges see it, and so a
    // dry run reports the same counts a real run would.
    present_ids.insert(line.id.clone());
    if opts.dedup {
        content_hashes
            .entry(line.table.clone())
            .or_default()
            .insert(hash);
    }
    report.imported += 1;
    Ok(())
}

fn import_edge(
    db: &Axil,
    opts: &ImportOptions,
    line: &EdgeLine,
    present_ids: &HashSet<String>,
    report: &mut ImportReport,
) -> Result<()> {
    // Both endpoints must resolve, else the edge would dangle.
    if !present_ids.contains(&line.from) || !present_ids.contains(&line.to) {
        report.edges_skipped += 1;
        return Ok(());
    }

    let from = RecordId::from_string(line.from.clone())?;
    let to = RecordId::from_string(line.to.clone())?;

    // Under --dedup, skip an edge that already exists with the same
    // (from, type, to) so re-import is idempotent. The lookup is read-only, so
    // it runs in a dry run too and the report predicts the real outcome.
    if opts.dedup && db.has_graph_index() {
        let existing = db.edges(&from, Some(&line.edge_type), Direction::Out)?;
        if existing.iter().any(|e| e.to == to) {
            report.edges_skipped += 1;
            return Ok(());
        }
    }

    if !opts.dry_run {
        let props = if line.props.is_null() {
            None
        } else {
            Some(line.props.clone())
        };
        match db.relate(&from, &line.edge_type, &to, props) {
            Ok(_) => report.edges_created += 1,
            Err(_) => {
                // Endpoint vanished between scan and relate, or no graph engine.
                report.edges_skipped += 1;
                return Ok(());
            }
        }
    } else {
        report.edges_created += 1;
    }
    Ok(())
}

/// Whether an imported record should end up with an embedding — mirrors the
/// auto-embed condition in the insert path: a user (non-`_`) table with enough
/// searchable text to embed. Kept in lockstep with `Axil::insert_record` so
/// verification never flags a record insert would not have embedded anyway.
fn expects_embedding(table: &str, data: &Value) -> bool {
    if table.starts_with('_') {
        return false;
    }
    let text = crate::util::searchable_text(data);
    !text.is_empty() && text.len() > 5
}

/// Reconstruct a [`Record`] from an import line, preserving id, timestamps, and
/// metadata. Timestamps that fail to parse fall back to the current time so a
/// slightly malformed line still imports rather than aborting the whole file.
fn build_record(line: &RecordLine) -> Result<Record> {
    let id = RecordId::from_string(line.id.clone())?;
    let created_at = DateTime::parse_from_rfc3339(&line.created_at)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let updated_at = line
        .updated_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(created_at);
    Ok(Record {
        id,
        table: line.table.clone(),
        data: line.data.clone(),
        created_at,
        updated_at,
        metadata: line.metadata.clone(),
    })
}

// ── Content hashing ─────────────────────────────────────────────────

/// SHA-256 of a record's canonicalized content, used for `--dedup` matching.
///
/// The canonical form sorts all object keys and drops top-level fields whose
/// name starts with `_` — those are Axil-internal and drift on their own
/// (importance decays, tiers change), so two teammates' copies of the *same*
/// memory would otherwise hash differently. Dropping them makes content dedup
/// work across machines.
fn content_hash(data: &Value) -> String {
    let canon = canonical_value(data, true);
    let bytes = serde_json::to_vec(&canon).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

/// Return a canonicalized clone of `value`: object keys sorted recursively, and
/// (only at the top level when `strip_underscore_top`) `_`-prefixed keys removed.
fn canonical_value(value: &Value, strip_underscore_top: bool) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map
                .iter()
                .filter(|(k, _)| !(strip_underscore_top && k.starts_with('_')))
                .map(|(k, v)| (k.clone(), canonical_value(v, false)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = serde_json::Map::new();
            for (k, v) in entries {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(|v| canonical_value(v, false)).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rebuilt_index_tables_are_recognized() {
        assert!(is_rebuilt_index_table("_idx_code_refs"));
        assert!(is_rebuilt_index_table("_idx_code_proxies"));
        assert!(!is_rebuilt_index_table("_entities"));
        assert!(!is_rebuilt_index_table("decisions"));
    }

    #[test]
    fn table_selection_default_excludes_system_and_indexes() {
        let opts = ExportOptions::default();
        assert!(table_selected("decisions", &opts));
        assert!(!table_selected("_entities", &opts));
        assert!(!table_selected("_idx_code_refs", &opts));
    }

    #[test]
    fn table_selection_include_system_keeps_underscore_but_not_index() {
        let opts = ExportOptions {
            include_system: true,
            ..Default::default()
        };
        assert!(table_selected("_entities", &opts));
        assert!(table_selected("decisions", &opts));
        // `_idx_*` is always excluded, even with include_system.
        assert!(!table_selected("_idx_code_refs", &opts));
    }

    #[test]
    fn explicit_allowlist_overrides_heuristic() {
        let opts = ExportOptions {
            tables: Some(vec!["decisions".to_string(), "_entities".to_string()]),
            ..Default::default()
        };
        assert!(table_selected("decisions", &opts));
        assert!(table_selected("_entities", &opts));
        assert!(!table_selected("errors", &opts));
        // Even an explicitly listed `_idx_*` table is refused.
        let opts2 = ExportOptions {
            tables: Some(vec!["_idx_code_refs".to_string()]),
            ..Default::default()
        };
        assert!(!table_selected("_idx_code_refs", &opts2));
    }

    #[test]
    fn content_hash_ignores_internal_fields_and_key_order() {
        let a = json!({"summary": "x", "reason": "y", "_importance": 0.9});
        let b = json!({"reason": "y", "summary": "x", "_importance": 0.2});
        assert_eq!(content_hash(&a), content_hash(&b));

        let c = json!({"summary": "different", "reason": "y"});
        assert_ne!(content_hash(&a), content_hash(&c));
    }

    #[test]
    fn canonical_value_sorts_nested_keys() {
        let v = json!({"b": {"z": 1, "a": 2}, "a": 3});
        let canon = canonical_value(&v, false);
        let s = serde_json::to_string(&canon).unwrap();
        assert_eq!(s, r#"{"a":3,"b":{"a":2,"z":1}}"#);
    }
}
