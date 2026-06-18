//! SCIP (Sourcegraph Code Intelligence Protocol) ingest for Axil.
//!
//! Parses a `*.scip` protobuf file, translates it into Axil records and
//! edges, and upgrades any pre-existing provisional entities (from the
//! regex extractor in `axil-core::entity`) to their grounded SCIP
//! canonical ids.
//!
//! **Intentionally narrow:** we consume SCIP, we do not produce it.
//! Users run an indexer (`scip-rust`, `scip-python`, `scip-typescript`, …)
//! to generate `index.scip`; we ingest it here.
//!
//! Edges emitted:
//!   - `entity-[:defined_in]->file` (direct)
//!   - `entity-[:references]->entity` (direct, from `SymbolInformation.relationships.is_reference`)
//!   - `entity-[:implements]->entity` (direct, from `is_implementation`)
//!   - `entity-[:type_of]->entity` (direct, from `is_type_definition`)
//!   - `entity-[:calls]->entity` (heuristic: enclosing-range match)
//!   - `file-[:imports]->file` (heuristic: `SymbolRole::Import`)

pub mod proto;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use axil_core::{Axil, AxilError, RecordId, Result};
use serde::Serialize;
use serde_json::{json, Value};

pub use proto::{decode_index, symbol_role};

pub const EDGE_DEFINED_IN: &str = "defined_in";
pub const EDGE_REFERENCES: &str = "references";
pub const EDGE_IMPLEMENTS: &str = "implements";
pub const EDGE_TYPE_OF: &str = "type_of";
pub const EDGE_CALLS: &str = "calls";
pub const EDGE_IMPORTS: &str = "imports";

pub const TABLE_ENTITIES: &str = "_entities";
pub const TABLE_IDX_FILES: &str = "_idx_files";

/// Summary of an ingest run. `applied = false` indicates a dry-run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IngestReport {
    pub applied: bool,
    pub indexer_name: String,
    pub indexer_version: String,
    pub symbol_count: usize,
    pub document_count: usize,
    pub defined_in_edges: usize,
    pub references_edges: usize,
    pub implements_edges: usize,
    pub type_of_edges: usize,
    pub calls_edges: usize,
    pub imports_edges: usize,
    /// Entities created for symbols not yet seen.
    pub entities_created: usize,
    /// Provisional entities rewritten to grounded canonical ids.
    pub provisional_upgraded: usize,
    /// Provisional entities left alone because SCIP didn't provide
    /// enough scope to disambiguate them.
    pub provisional_ambiguous: usize,
}

impl IngestReport {
    pub fn total_edges(&self) -> usize {
        self.defined_in_edges
            + self.references_edges
            + self.implements_edges
            + self.type_of_edges
            + self.calls_edges
            + self.imports_edges
    }
}

/// How a ranked relationship was derived. Persisted on edge properties
/// so downstream recall can down-weight heuristic edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Directly stated in SCIP data (no inference).
    Direct,
    /// Derived via enclosing-range or import-role heuristics.
    Heuristic,
}

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::Direct => "direct",
            Confidence::Heuristic => "heuristic",
        }
    }
}

/// Ingest a SCIP index file into the given Axil database.
///
/// Idempotent: re-ingesting the same file produces the same graph — no
/// duplicate edges, no duplicate entity rows.
pub fn ingest_scip(db: &Axil, scip_path: &Path) -> Result<IngestReport> {
    ingest_scip_opts(db, scip_path, IngestOptions::default())
}

#[derive(Debug, Clone, Default)]
pub struct IngestOptions {
    /// When true, don't write anything. Count would-be operations and
    /// return the report.
    pub dry_run: bool,
}

pub fn ingest_scip_opts(db: &Axil, scip_path: &Path, opts: IngestOptions) -> Result<IngestReport> {
    // Library-side guard: SCIP ingest fundamentally writes call/ref/
    // implements/type_of edges. Without a graph plugin attached,
    // `relate_once` silently early-returns and the resulting
    // IngestReport claims thousands of edges while zero are persisted.
    // The CLI auto-attaches via `open_for_scip_ingest`; this catches
    // the same misuse from library callers. `dry_run` is exempt
    // because it doesn't write anything anyway.
    if !opts.dry_run && db.graph_index_ref().is_none() {
        return Err(AxilError::plugin(
            "SCIP ingest requires a graph plugin — open the DB with \
             `with_graph_plugin()` (or use `axil ingest-scip`, which \
             attaches it automatically). Without a graph store, edge \
             counts in IngestReport would be misleading: the rows \
             land but the edges silently drop."
                .to_string(),
        ));
    }

    let bytes = std::fs::read(scip_path)
        .map_err(|e| AxilError::plugin(format!("reading {}: {e}", scip_path.display())))?;
    let index =
        decode_index(&bytes).map_err(|e| AxilError::plugin(format!("SCIP decode failed: {e}")))?;

    let mut report = IngestReport {
        applied: !opts.dry_run,
        ..Default::default()
    };

    if let Some(meta) = &index.metadata {
        if let Some(tool) = &meta.tool_info {
            report.indexer_name = tool.name.clone();
            report.indexer_version = tool.version.clone();
        }
    }
    report.document_count = index.documents.len();

    // Pass 1: index all SymbolInformation rows across documents + externals
    // so we know which symbols have Definition occurrences and where they
    // live. This map keys `symbol_string → (document_path, display_name,
    // kind, language)`.
    let mut symbol_info: HashMap<String, SymbolMeta> = HashMap::new();
    for doc in &index.documents {
        let axil_lang = normalize_scip_language(&doc.language);
        for sym in &doc.symbols {
            symbol_info.insert(
                sym.symbol.clone(),
                SymbolMeta {
                    document: Some(doc.relative_path.clone()),
                    display_name: sym.display_name.clone(),
                    kind: sym.kind,
                    language: axil_lang.clone(),
                },
            );
        }
    }
    for sym in &index.external_symbols {
        symbol_info.entry(sym.symbol.clone()).or_insert(SymbolMeta {
            document: None,
            display_name: sym.display_name.clone(),
            kind: sym.kind,
            language: None,
        });
    }

    report.symbol_count = symbol_info.len();

    // Pass 0 — upgrade provisional entities BEFORE we start creating new
    // rows for SCIP symbols. That way `ensure_entity_record` finds the
    // rewritten row in the cache and doesn't insert a duplicate.
    let provisional_candidates = load_provisional_entities(db)?;
    if !opts.dry_run && !provisional_candidates.is_empty() {
        upgrade_provisional_entities(db, &provisional_candidates, &symbol_info, &mut report)?;
    }

    // We need to create / find entity records lazily during the passes;
    // cache by canonical SCIP symbol string. Load AFTER the provisional
    // upgrade so we pick up rewritten canonical ids.
    let mut entity_cache = load_entity_cache(db)?;
    let mut alias_seen = if opts.dry_run {
        AliasDedupe::new()
    } else {
        load_alias_dedupe(db)?
    };

    // All edge writes funnel through here so we commit them in a
    // single redb txn at the end (Phase 14 perf, friction #8).
    let mut edge_buf = EdgeBuffer::new();
    // Same idea for entity rows + alias rows: queue with synchronous
    // ID allocation, flush at the end. Without this the per-record
    // redb commits dominate ingest wall time on large indexes.
    let mut entity_buf = EntityBuffer::new();
    let mut alias_buf = AliasBuffer::new();

    // Pass 2: emit `defined_in` edges (direct) for every document's
    // Definition occurrences, and materialize file records if missing.
    // Preload existing `_idx_files` rows so per-file lookups are O(1).
    let mut file_cache: HashMap<String, RecordId> = if opts.dry_run {
        HashMap::new()
    } else {
        load_file_cache(db)?
    };

    // Progress line: SCIP ingest on a 14-crate workspace processes
    // hundreds of documents; without periodic output users see nothing
    // for minutes and assume the process is hung. Emit every 100 docs.
    let total_docs = index.documents.len();
    let progress_interval = (total_docs / 20).max(50);
    let progress_enabled = total_docs >= 100 && !opts.dry_run;
    if progress_enabled {
        eprintln!(
            "axil ingest-scip: {total_docs} documents, {} symbols",
            report.symbol_count
        );
    }
    for (doc_idx, doc) in index.documents.iter().enumerate() {
        if progress_enabled && doc_idx > 0 && doc_idx % progress_interval == 0 {
            eprintln!(
                "axil ingest-scip: {}/{} docs, {} entities, {} edges queued",
                doc_idx,
                total_docs,
                report.entities_created,
                edge_buf.len(),
            );
        }
        let file_rid = if opts.dry_run {
            // Use a synthetic id so counts still work without writing.
            RecordId::new()
        } else {
            ensure_file_record(db, &doc.relative_path, &mut file_cache)?
        };

        // Walk occurrences once, grouping by enclosing definition.
        // Definitions on this document give us both an `entity` row and
        // a `defined_in` edge to the file.
        let mut defs_in_doc: Vec<(&proto::Occurrence, RecordId)> = Vec::new();
        for occ in &doc.occurrences {
            let is_def = (occ.symbol_roles & symbol_role::DEFINITION) != 0;
            if !is_def {
                continue;
            }
            let def_location = scip_def_location(&doc.relative_path, occ);
            let entity_rid = if opts.dry_run {
                RecordId::new()
            } else {
                ensure_entity_record(
                    &occ.symbol,
                    symbol_info.get(&occ.symbol),
                    def_location.as_ref(),
                    &mut entity_cache,
                    &mut alias_seen,
                    &mut report,
                    &mut entity_buf,
                    &mut alias_buf,
                )?
            };
            if !opts.dry_run {
                edge_buf.push(
                    entity_rid.clone(),
                    EDGE_DEFINED_IN,
                    file_rid.clone(),
                    Confidence::Direct,
                );
            }
            report.defined_in_edges += 1;
            defs_in_doc.push((occ, entity_rid));
        }

        // Pass 3 (same document): `calls` edges via enclosing-range
        // heuristic. For each non-Definition occurrence X, find the
        // definition D whose range encloses X's range; emit `D calls X`.
        for occ in &doc.occurrences {
            let roles = occ.symbol_roles;
            let is_def = (roles & symbol_role::DEFINITION) != 0;
            if is_def {
                continue;
            }
            // Skip import roles here — they drive file imports separately.
            if (roles & symbol_role::IMPORT) != 0 {
                continue;
            }
            let Some(caller_rid) = find_enclosing_definition(&defs_in_doc, occ) else {
                continue;
            };
            let callee_rid = if opts.dry_run {
                RecordId::new()
            } else {
                ensure_entity_record(
                    &occ.symbol,
                    symbol_info.get(&occ.symbol),
                    None,
                    &mut entity_cache,
                    &mut alias_seen,
                    &mut report,
                    &mut entity_buf,
                    &mut alias_buf,
                )?
            };
            if caller_rid == callee_rid {
                continue; // a definition can't "call" itself in this sense.
            }
            if !opts.dry_run {
                edge_buf.push(
                    caller_rid.clone(),
                    EDGE_CALLS,
                    callee_rid.clone(),
                    Confidence::Heuristic,
                );
            }
            report.calls_edges += 1;
        }

        // Pass 4: file-level `imports` edges (heuristic).
        // For each occurrence with SymbolRole::Import, find the document
        // the imported symbol is defined in and emit `this_file -> that_file`.
        let mut seen_imports: std::collections::HashSet<String> = std::collections::HashSet::new();
        for occ in &doc.occurrences {
            if (occ.symbol_roles & symbol_role::IMPORT) == 0 {
                continue;
            }
            let Some(other_doc) = symbol_info
                .get(&occ.symbol)
                .and_then(|m| m.document.as_deref())
            else {
                continue;
            };
            if other_doc == doc.relative_path {
                continue;
            }
            if !seen_imports.insert(other_doc.to_string()) {
                continue;
            }
            let other_rid = if opts.dry_run {
                RecordId::new()
            } else {
                ensure_file_record(db, other_doc, &mut file_cache)?
            };
            if !opts.dry_run {
                edge_buf.push(
                    file_rid.clone(),
                    EDGE_IMPORTS,
                    other_rid,
                    Confidence::Heuristic,
                );
            }
            report.imports_edges += 1;
        }
    }

    // Pass 5: direct relationships from SymbolInformation (references,
    // implements, type_of). These live on SymbolInformation.relationships
    // and apply to the symbol they're attached to.
    for doc in &index.documents {
        for sym in &doc.symbols {
            let from_rid = if opts.dry_run {
                RecordId::new()
            } else {
                ensure_entity_record(
                    &sym.symbol,
                    symbol_info.get(&sym.symbol),
                    None,
                    &mut entity_cache,
                    &mut alias_seen,
                    &mut report,
                    &mut entity_buf,
                    &mut alias_buf,
                )?
            };
            for rel in &sym.relationships {
                let to_rid = if opts.dry_run {
                    RecordId::new()
                } else {
                    ensure_entity_record(
                        &rel.symbol,
                        symbol_info.get(&rel.symbol),
                        None,
                        &mut entity_cache,
                        &mut alias_seen,
                        &mut report,
                        &mut entity_buf,
                        &mut alias_buf,
                    )?
                };
                if rel.is_reference {
                    if !opts.dry_run {
                        edge_buf.push(
                            from_rid.clone(),
                            EDGE_REFERENCES,
                            to_rid.clone(),
                            Confidence::Direct,
                        );
                    }
                    report.references_edges += 1;
                }
                if rel.is_implementation {
                    if !opts.dry_run {
                        edge_buf.push(
                            from_rid.clone(),
                            EDGE_IMPLEMENTS,
                            to_rid.clone(),
                            Confidence::Direct,
                        );
                    }
                    report.implements_edges += 1;
                }
                if rel.is_type_definition {
                    if !opts.dry_run {
                        edge_buf.push(
                            from_rid.clone(),
                            EDGE_TYPE_OF,
                            to_rid.clone(),
                            Confidence::Direct,
                        );
                    }
                    report.type_of_edges += 1;
                }
            }
        }
    }

    // Flush all buffered writes. Entity + alias rows land in `.axil`
    // (one redb file) and edges land in `.axil.graph` (separate redb
    // file), so the two flushes hit independent stores. Order is
    // independent because RecordIds are pre-allocated at queue-time
    // and edges store opaque IDs — we flush core first so a
    // partial-failure leaves a queryable entity table even if the
    // edge txn errors.
    if !opts.dry_run {
        if progress_enabled {
            eprintln!(
                "axil ingest-scip: flushing {} entities, {} aliases, {} edges",
                entity_buf.pending.len(),
                alias_buf.pending.len(),
                edge_buf.len(),
            );
        }
        flush_entities_and_aliases(db, &mut entity_buf, &mut alias_buf)?;
        let pending = edge_buf.len();
        let written = edge_buf.flush(db)?;
        if progress_enabled {
            eprintln!(
                "axil ingest-scip: wrote {written} new edges ({} dedup'd)",
                pending.saturating_sub(written),
            );
        }
    }

    Ok(report)
}

#[derive(Debug, Clone, Default)]
struct SymbolMeta {
    /// `relative_path` of the Document that contains the definition,
    /// if any. `None` for external symbols.
    document: Option<String>,
    display_name: String,
    #[allow(dead_code)]
    kind: i32,
    language: Option<String>,
}

fn load_entity_cache(db: &Axil) -> Result<HashMap<String, RecordId>> {
    let rows = db.list(TABLE_ENTITIES)?;
    let mut cache = HashMap::with_capacity(rows.len());
    for r in rows {
        if let Some(cid) = r.data.get("canonical_id").and_then(|v| v.as_str()) {
            cache.insert(cid.to_string(), r.id);
        }
    }
    Ok(cache)
}

#[derive(Debug, Clone)]
struct ProvisionalEntity {
    record_id: RecordId,
    display_name: String,
    lang_hint: Option<String>,
    old_canonical_id: String,
}

fn load_provisional_entities(db: &Axil) -> Result<Vec<ProvisionalEntity>> {
    let rows = db.list(TABLE_ENTITIES)?;
    let mut out = Vec::new();
    for r in rows {
        let Some(cid) = r.data.get("canonical_id").and_then(|v| v.as_str()) else {
            continue;
        };
        if !cid.starts_with("provisional:") {
            continue;
        }
        let display = r
            .data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let lang = r
            .data
            .get("entity_type")
            .and_then(|v| v.get("code_symbol"))
            .and_then(|v| v.get("lang_hint"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.push(ProvisionalEntity {
            record_id: r.id,
            display_name: display,
            lang_hint: lang,
            old_canonical_id: cid.to_string(),
        });
    }
    Ok(out)
}

/// Look up an entity record by SCIP canonical id; create it if missing.
/// Local de-dupe set for alias registration during one ingest pass.
/// `(alias, scope, canonical_id)` — avoids the per-call table scan that
/// `Axil::register_entity_alias` does. Pre-seeded from the table so a
/// second ingest over the same SCIP file stays O(1) per alias.
type AliasDedupe = std::collections::HashSet<(String, String, String)>;

fn load_alias_dedupe(db: &Axil) -> Result<AliasDedupe> {
    let rows = db.list(axil_core::SCIP_ALIAS_TABLE).unwrap_or_default();
    let mut set = AliasDedupe::with_capacity(rows.len());
    for r in rows {
        let a = r.data.get("alias").and_then(|v| v.as_str());
        let s = r.data.get("scope").and_then(|v| v.as_str());
        let c = r.data.get("canonical_id").and_then(|v| v.as_str());
        if let (Some(a), Some(s), Some(c)) = (a, s, c) {
            set.insert((a.to_string(), s.to_string(), c.to_string()));
        }
    }
    Ok(set)
}

/// Definition location captured from a SCIP Definition occurrence —
/// (path, line_start, line_end), all 1-indexed lines. Stored on the
/// `_entities` row at insert time so the indexer's proxy builder can
/// fill `line_end` precisely without re-parsing SCIP later.
type DefLocation = (String, usize, usize);

#[allow(clippy::too_many_arguments)]
fn ensure_entity_record(
    symbol: &str,
    meta: Option<&SymbolMeta>,
    def_location: Option<&DefLocation>,
    cache: &mut HashMap<String, RecordId>,
    alias_seen: &mut AliasDedupe,
    report: &mut IngestReport,
    entity_buf: &mut EntityBuffer,
    alias_buf: &mut AliasBuffer,
) -> Result<RecordId> {
    if let Some(id) = cache.get(symbol) {
        return Ok(id.clone());
    }
    let (display_name, language) = match meta {
        Some(m) => (
            if m.display_name.is_empty() {
                display_name_from_symbol(symbol)
            } else {
                m.display_name.clone()
            },
            m.language.clone(),
        ),
        None => (display_name_from_symbol(symbol), None),
    };
    let entity_type = json!({
        "code_symbol": { "lang_hint": language }
    });
    let mut data = json!({
        "name": display_name,
        "canonical_id": symbol,
        "entity_type": entity_type,
        "source_text": "scip",
    });
    if let Some((path, ls, le)) = def_location {
        let obj = data.as_object_mut().expect("entity data is an object");
        obj.insert("def_path".into(), json!(path));
        obj.insert("def_line_start".into(), json!(ls));
        obj.insert("def_line_end".into(), json!(le));
    }
    let id = entity_buf.queue(data);
    cache.insert(symbol.to_string(), id.clone());
    report.entities_created += 1;

    if !display_name.is_empty() {
        let mut scopes: Vec<String> = Vec::with_capacity(3);
        if let Some(lang) = language.as_deref() {
            scopes.push(format!("lang:{lang}"));
        }
        if let Some(file) = meta.and_then(|m| m.document.as_deref()) {
            scopes.push(format!("file:{file}"));
        }
        scopes.push("global".into());
        for scope in scopes {
            let key = (display_name.clone(), scope.clone(), symbol.to_string());
            if alias_seen.insert(key) {
                alias_buf.queue(&display_name, &scope, symbol);
            }
        }
    }

    Ok(id)
}

/// Map SCIP's `Document.language` string (e.g. `"Rust"`, `"TypeScript"`,
/// `"Python"`) onto the short language tag the regex extractor emits
/// (`"rust"`, `"ts"`, `"python"`). Unknown languages fall back to a
/// lowercased version of the original string.
fn normalize_scip_language(scip_lang: &str) -> Option<String> {
    if scip_lang.is_empty() {
        return None;
    }
    let tag = match scip_lang.to_ascii_lowercase().as_str() {
        "rust" => "rust",
        "python" => "python",
        "typescript" | "ts" => "ts",
        "javascript" | "js" => "ts", // JS and TS share the extractor key
        "go" => "go",
        "java" | "kotlin" => "java",
        "ruby" => "ruby",
        "scala" => "scala",
        "c" | "c++" | "cpp" => "cpp",
        other => return Some(other.to_string()),
    };
    Some(tag.to_string())
}

/// Derive a readable display name from a SCIP symbol string when the
/// indexer did not supply `SymbolInformation.display_name`. SCIP symbols
/// look like `rust-analyzer cargo axil-core 0.4.0 db/Axil#open().`; we
/// take the last whitespace-separated token, then trim trailing
/// punctuation and the leading scope path.
fn display_name_from_symbol(symbol: &str) -> String {
    let tail = symbol.rsplit(' ').next().unwrap_or(symbol);
    let trimmed = tail.trim_end_matches(['.', '#', '(', ')']);
    let leaf = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if leaf.is_empty() {
        symbol.to_string()
    } else {
        leaf.to_string()
    }
}

/// Look up an `_idx_files` row by path; create a stub if none exists.
///
/// The cache must be preloaded by `load_file_cache` so the per-call
/// `db.list` scan never reaches here on the hot path. Stub rows carry
/// `_scip_stub: true` so the full indexer's incremental pass can
/// distinguish them from its own writes.
fn ensure_file_record(
    db: &Axil,
    relative_path: &str,
    cache: &mut HashMap<String, RecordId>,
) -> Result<RecordId> {
    if let Some(id) = cache.get(relative_path) {
        return Ok(id.clone());
    }
    let rec = db.insert(
        TABLE_IDX_FILES,
        json!({
            "path": relative_path,
            "summary": "",
            "imports": [],
            "exports": [],
            "_scip_stub": true,
        }),
    )?;
    cache.insert(relative_path.to_string(), rec.id.clone());
    Ok(rec.id)
}

/// Preload the `_idx_files` cache so per-file `ensure_file_record` calls
/// don't each do a full table scan. O(F) one-shot read at ingest start.
fn load_file_cache(db: &Axil) -> Result<HashMap<String, RecordId>> {
    let rows = db.list(TABLE_IDX_FILES)?;
    let mut cache = HashMap::with_capacity(rows.len());
    for r in rows {
        if let Some(path) = r.data.get("path").and_then(|v| v.as_str()) {
            cache.insert(path.to_string(), r.id);
        }
    }
    Ok(cache)
}

/// Buffer of pending entity inserts. IDs are allocated synchronously
/// at `queue` time so callers can immediately use them in graph edges,
/// while the actual `_entities` storage write is deferred to a single
/// batched txn at flush time. Mirrors the pattern in `EdgeBuffer`.
struct EntityBuffer {
    pending: Vec<axil_core::Record>,
}

impl EntityBuffer {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Allocate a `RecordId` and queue the insert. Returns the ID
    /// synchronously so subsequent edge writes can reference it.
    fn queue(&mut self, data: Value) -> RecordId {
        let record = axil_core::Record::new(TABLE_ENTITIES, data);
        let id = record.id.clone();
        self.pending.push(record);
        id
    }
}

/// Buffer of pending `_scip_aliases` rows. The single-call alias
/// registrar (`db.register_entity_alias`) does a full table scan per
/// call to dedupe — fine at one or two aliases, O(N²) at thousands.
/// This buffer dedupes against a preloaded snapshot once, then writes
/// new rows in a single batch.
struct AliasBuffer {
    pending: Vec<axil_core::Record>,
}

impl AliasBuffer {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    fn queue(&mut self, alias_name: &str, scope: &str, canonical_id: &str) {
        self.pending.push(axil_core::Record::new(
            axil_core::SCIP_ALIAS_TABLE,
            json!({
                "alias": alias_name,
                "scope": scope,
                "canonical_id": canonical_id,
            }),
        ));
    }
}

/// Flush entity + alias buffers in a single redb write transaction.
/// Both target the same `.axil` core store, so concatenating them
/// folds two fsyncs into one — `Storage::insert_batch` already groups
/// records by table internally, so the call shape is unchanged.
///
/// Side effects mirror what `Axil::insert` does for these tables in
/// the per-record path:
/// - Per-record `insert` audit entries via `Axil::audit_inserts`.
///   Required so `axil audit-log` doesn't have a SCIP-shaped hole.
/// - Atlas canonical-ID publish for each `_entities` row.
///   `run_insert_hooks` short-circuits on `_`-prefixed tables in
///   `Axil::insert`, so there's nothing to replay there.
///
/// Failure window note: the per-record path called `audit` and
/// `publish` immediately after each `storage.insert` succeeded. The
/// batch path commits all rows first, then loops the post-commit
/// side effects. If the process dies between the commit and the
/// audit/publish loop, the local DB has the new rows while the audit
/// log and any canonical-ID subscriber miss them. This is acceptable
/// because:
///   - the audit log is durable on a best-effort basis already
///     (writes are unsynced unless the global `audit_enabled` flag
///     enforces durability),
///   - the `CanonicalPublisher` seam is fire-and-forget by contract
///     (the default has no subscriber; an external coordinator that
///     implements it may drop events under load), so the guarantee is
///     "publish may miss events" rather than "transactionally
///     consistent". The widened window is a coverage degradation for
///     SCIP ingest only, never a correctness regression for the local DB.
fn flush_entities_and_aliases(
    db: &Axil,
    entity_buf: &mut EntityBuffer,
    alias_buf: &mut AliasBuffer,
) -> Result<()> {
    if entity_buf.pending.is_empty() && alias_buf.pending.is_empty() {
        return Ok(());
    }
    // `_entities` and `_scip_aliases` are both internal — importance
    // scoring, consent defaults, vector embedding, and auto-link
    // hooks all skip naturally inside the storage layer's batch path.
    let mut combined: Vec<axil_core::Record> =
        Vec::with_capacity(entity_buf.pending.len() + alias_buf.pending.len());
    combined.append(&mut entity_buf.pending);
    combined.append(&mut alias_buf.pending);
    db.storage().insert_batch(&combined)?;
    db.audit_inserts(&combined);
    for record in &combined {
        if record.table == TABLE_ENTITIES {
            db.publish_canonical_for_record(record);
        }
    }
    Ok(())
}

/// Buffer of pending edge writes. SCIP ingest produces ~200k edges on
/// a 10 MB workspace index; persisting each in its own redb txn turned
/// ingest into a 15-minute job (Phase 14 dogfood friction #8). The
/// buffer collects intent during the multi-pass walk, then flushes the
/// whole batch through `GraphIndex::relate_batch` (single redb txn) at
/// the end of ingest.
///
/// Idempotence is preserved by deduping against existing edges at
/// flush time — same (from, edge_type, to) triple as the per-call
/// `relate_once` it replaced.
struct EdgeBuffer {
    pending: Vec<(RecordId, &'static str, RecordId, Confidence)>,
}

impl EdgeBuffer {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    fn push(
        &mut self,
        from: RecordId,
        edge_type: &'static str,
        to: RecordId,
        confidence: Confidence,
    ) {
        self.pending.push((from, edge_type, to, confidence));
    }

    fn len(&self) -> usize {
        self.pending.len()
    }

    /// Persist all buffered edges in a single graph-store txn,
    /// deduping against in-buffer dupes and against edges already on
    /// disk. Returns the number actually written. Empties the buffer.
    fn flush(&mut self, db: &Axil) -> Result<usize> {
        if self.pending.is_empty() {
            return Ok(0);
        }
        let Some(gi) = db.graph_index_ref() else {
            self.pending.clear();
            return Ok(0);
        };

        // Drop in-batch duplicates so we don't ask the graph index
        // about the same triple twice. The HashSet pays for itself
        // because SCIP frequently emits the same (defined_in, calls)
        // edge from multiple occurrences in the same document.
        let mut seen_in_batch: HashSet<(RecordId, &'static str, RecordId)> = HashSet::new();
        let mut deduped: Vec<(RecordId, &'static str, RecordId, Confidence)> =
            Vec::with_capacity(self.pending.len());
        for spec in self.pending.drain(..) {
            let key = (spec.0.clone(), spec.1, spec.2.clone());
            if seen_in_batch.insert(key) {
                deduped.push(spec);
            }
        }

        // First-ingest fast path: when the graph store is empty no
        // existing edge can collide, so skip all the per-`from`
        // existence reads. This is the common case (a fresh DB ingest
        // is exactly zero existing edges) and saves O(unique-froms)
        // in-memory walks.
        let skip_disk_dedup = gi.edge_count() == 0;

        // Group by `from` only — one outbound read per source vertex,
        // filter by edge_type in-memory. Grouping by (from, edge_type)
        // would call `gi.edges` once per type per source, paying for
        // the same `outgoing[from]` walk multiple times on heavy
        // sources (a definition typically has `defined_in`, `calls`,
        // and `references` edges all rooted at the same id).
        let mut by_from: HashMap<RecordId, Vec<(&'static str, RecordId, Confidence)>> =
            HashMap::new();
        for (from, etype, to, conf) in deduped {
            by_from.entry(from).or_default().push((etype, to, conf));
        }

        let mut to_write: Vec<(RecordId, String, RecordId, Value)> = Vec::new();
        for (from, targets) in by_from {
            let existing: HashSet<(String, RecordId)> = if skip_disk_dedup {
                HashSet::new()
            } else {
                gi.edges(from.clone(), None, axil_core::Direction::Out)?
                    .into_iter()
                    .map(|e| (e.edge_type, e.to))
                    .collect()
            };
            for (etype, to, conf) in targets {
                if existing.contains(&(etype.to_string(), to.clone())) {
                    continue;
                }
                to_write.push((
                    from.clone(),
                    etype.to_string(),
                    to,
                    json!({ "confidence": conf.as_str(), "scip_grounded": true }),
                ));
            }
        }

        let count = to_write.len();
        gi.relate_batch(to_write)?;
        Ok(count)
    }
}

/// Find the Definition occurrence in `defs_in_doc` whose range encloses
/// `needle`'s range. Returns the entity RecordId of the enclosing def.
/// Ties are broken by tightest enclosure (smallest byte span).
fn find_enclosing_definition(
    defs_in_doc: &[(&proto::Occurrence, RecordId)],
    needle: &proto::Occurrence,
) -> Option<RecordId> {
    let (n_line, n_col) = first_point(&needle.range)?;
    let mut best: Option<(&RecordId, i64)> = None;
    for (def_occ, rid) in defs_in_doc {
        // Prefer `enclosing_range` when the indexer supplies it; it
        // spans the full body of the definition. Otherwise fall back to
        // `range` (just the name), which will miss nested calls — that's
        // an acknowledged heuristic gap.
        let span = if !def_occ.enclosing_range.is_empty() {
            range_span(&def_occ.enclosing_range)
        } else {
            range_span(&def_occ.range)
        };
        let Some(((s_line, s_col), (e_line, e_col), width)) = span else {
            continue;
        };
        // Is (n_line, n_col) inside [(s_line,s_col), (e_line,e_col))?
        let after_start = (n_line, n_col) >= (s_line, s_col);
        let before_end = (n_line, n_col) < (e_line, e_col);
        if after_start && before_end {
            match best {
                None => best = Some((rid, width)),
                Some((_, w)) if width < w => best = Some((rid, width)),
                _ => {}
            }
        }
    }
    best.map(|(rid, _)| rid.clone())
}

fn first_point(range: &[i32]) -> Option<(i32, i32)> {
    match range.len() {
        3 | 4 => Some((range[0], range[1])),
        _ => None,
    }
}

/// Extract a `(path, line_start, line_end)` tuple from a SCIP Definition
/// occurrence. Prefers `enclosing_range` (covers the whole symbol body)
/// over `range` (just the name token). SCIP encodes lines 0-indexed; we
/// convert to 1-indexed to match how the indexer numbers lines.
fn scip_def_location(path: &str, occ: &proto::Occurrence) -> Option<DefLocation> {
    let r = if !occ.enclosing_range.is_empty() {
        occ.enclosing_range.as_slice()
    } else {
        occ.range.as_slice()
    };
    let (start, end, _) = range_span(r)?;
    let ls = (start.0 as i64).max(0) as usize + 1;
    let le = (end.0 as i64).max(start.0 as i64) as usize + 1;
    Some((path.to_string(), ls, le))
}

fn range_span(range: &[i32]) -> Option<((i32, i32), (i32, i32), i64)> {
    let (start, end) = match range.len() {
        3 => ((range[0], range[1]), (range[0], range[2])),
        4 => ((range[0], range[1]), (range[2], range[3])),
        _ => return None,
    };
    // A rough "width" used only for tightness-ranking; lines * 10000 + col.
    let width = ((end.0 - start.0) as i64).max(0) * 10_000 + ((end.1 - start.1) as i64).max(0);
    Some((start, end, width))
}

/// Rewrite provisional canonical ids into their SCIP-grounded form when
/// `display_name + lang_hint` match exactly one SCIP symbol. Never
/// silently merges on ambiguous matches.
fn upgrade_provisional_entities(
    db: &Axil,
    candidates: &[ProvisionalEntity],
    symbol_info: &HashMap<String, SymbolMeta>,
    report: &mut IngestReport,
) -> Result<()> {
    // Pre-group SCIP symbols by (display_name, language) for fast lookup.
    let mut by_name_lang: HashMap<(String, Option<String>), Vec<&String>> = HashMap::new();
    for (sym_str, meta) in symbol_info {
        let disp = if meta.display_name.is_empty() {
            display_name_from_symbol(sym_str)
        } else {
            meta.display_name.clone()
        };
        by_name_lang
            .entry((disp, meta.language.clone()))
            .or_default()
            .push(sym_str);
    }

    // Build a single set of all canonical ids currently in `_entities` so
    // the per-candidate `has_grounded` check is O(1) instead of an
    // O(N²) repeated table scan.
    let grounded_ids: std::collections::HashSet<String> = db
        .list(TABLE_ENTITIES)?
        .into_iter()
        .filter_map(|r| {
            r.data
                .get("canonical_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    for prov in candidates {
        let key = (prov.display_name.clone(), prov.lang_hint.clone());
        let Some(matches) = by_name_lang.get(&key) else {
            continue;
        };
        if matches.len() != 1 {
            report.provisional_ambiguous += 1;
            continue;
        }
        let grounded = matches[0].as_str();
        if grounded_ids.contains(grounded) {
            db.merge_entities(&prov.old_canonical_id, grounded)?;
        } else if let Some(r) = db.get(&prov.record_id)? {
            if let Value::Object(mut map) = r.data {
                map.insert(
                    "canonical_id".to_string(),
                    Value::String(grounded.to_string()),
                );
                map.insert("source_text".to_string(), Value::String("scip".to_string()));
                db.update(&prov.record_id, Value::Object(map))?;
            }
        }
        report.provisional_upgraded += 1;
    }
    Ok(())
}

// ── Discovery: `axil doctor` helper ──────────────────────────────────

/// A SCIP index file discovered on disk.
#[derive(Debug, Clone)]
pub struct ScipFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub modified_secs_ago: u64,
}

/// Scan the given roots for `*.scip` files. Does not recurse into
/// subdirectories beyond the first level — this is a detection helper,
/// not a walker. Accepts the repo root and `.axil/`.
pub fn discover_scip_files(roots: &[&Path]) -> Vec<ScipFile> {
    let mut found = Vec::new();
    let now = SystemTime::now();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(ext) = path.extension() else {
                continue;
            };
            if ext != "scip" {
                continue;
            }
            let (size_bytes, modified_secs_ago) = std::fs::metadata(&path)
                .and_then(|m| {
                    let size = m.len();
                    let mtime = m.modified().unwrap_or(now);
                    let age = now.duration_since(mtime).unwrap_or_default().as_secs();
                    Ok((size, age))
                })
                .unwrap_or((0, 0));
            found.push(ScipFile {
                path,
                size_bytes,
                modified_secs_ago,
            });
        }
    }
    found
}

/// Inspect a SCIP file without ingesting it: return tool name/version
/// and basic counts. Used by `axil doctor` to print diagnostics.
pub fn inspect_scip(path: &Path) -> Result<IngestReport> {
    let bytes = std::fs::read(path)
        .map_err(|e| AxilError::plugin(format!("reading {}: {e}", path.display())))?;
    let index =
        decode_index(&bytes).map_err(|e| AxilError::plugin(format!("SCIP decode failed: {e}")))?;
    let mut report = IngestReport {
        applied: false,
        ..Default::default()
    };
    if let Some(meta) = &index.metadata {
        if let Some(tool) = &meta.tool_info {
            report.indexer_name = tool.name.clone();
            report.indexer_version = tool.version.clone();
        }
    }
    report.document_count = index.documents.len();
    let mut syms = 0usize;
    for doc in &index.documents {
        syms += doc.symbols.len();
    }
    syms += index.external_symbols.len();
    report.symbol_count = syms;
    Ok(report)
}

// ── --watch stabilization gate ──────────────────────────────────────

/// Wait until `path` has been stable (same size and mtime) for the
/// given stabilization window. Returns `Ok(true)` when stable, or
/// `Ok(false)` if the file kept changing across the check window.
///
/// Never relies on temp-file-then-rename atomicity — only on the file
/// being quiescent for long enough that a full protobuf has almost
/// certainly landed.
pub fn wait_for_stable(path: &Path, timeout: std::time::Duration) -> Result<bool> {
    let start = std::time::Instant::now();
    let mut last: Option<(u64, SystemTime)> = None;
    let mut stable_checks = 0usize;
    while start.elapsed() < timeout {
        let md = match std::fs::metadata(path) {
            Ok(md) => md,
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };
        let snap = (md.len(), md.modified().unwrap_or(SystemTime::now()));
        if last == Some(snap) {
            stable_checks += 1;
            // Need two consecutive stable observations across +500ms and
            // +1500ms to call it stable.
            if stable_checks >= 2 {
                return Ok(true);
            }
        } else {
            stable_checks = 0;
            last = Some(snap);
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(false)
}
