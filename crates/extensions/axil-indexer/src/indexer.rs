//! Core indexer — orchestrates scanning, parsing, and storage.
//!
//! The `ProjectIndexer` scans a project directory, parses each source file,
//! and stores compact records in an Axil database. It supports both full
//! and incremental re-indexing.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::SystemTime;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, RecordId};

use crate::config_sections::{split_json_sections, split_toml_sections, split_yaml_sections};
use crate::markdown::{section_canonical_id, split_sections};
use crate::parser::{self, ParsedFile};
use crate::progress::{IndexProgress, NoopProgress};
use crate::proxy::{
    build_proxy, proxy_to_record, CodeProxy, ProxyInput, ProxyKind, TABLE_CODE_PROXIES,
};
use crate::scanner::{self, Language, ProjectType, ScannedFile};
use crate::token;
use axil_core::IndexConfig;

// ── Table names ──────────────────────────────────────────────────────

pub const TABLE_PROJECT: &str = "_idx_project";
pub const TABLE_FILES: &str = "_idx_files";
pub const TABLE_MODULES: &str = "_idx_modules";
pub const TABLE_SYMBOLS: &str = "_idx_symbols";
pub const TABLE_DEPS: &str = "_idx_deps";

/// Result of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub indexed_files: usize,
    pub modules: usize,
    pub symbols: usize,
    pub deps: usize,
    pub tables_created: Vec<String>,
    pub tokens_saved_estimate: String,
    /// For incremental: how many files were changed vs unchanged.
    pub changed: Option<usize>,
    pub unchanged: Option<usize>,
}

/// SCIP definition match for a `(path, symbol)` pair. Carries the
/// canonical id plus precise lines from the SCIP enclosing range when
/// the indexer recorded them on `_entities` (`def_line_start`/`def_line_end`).
#[derive(Clone)]
struct ScipDefHit {
    canonical_id: String,
    line_start: Option<usize>,
    line_end: Option<usize>,
}

/// `(path, symbol_name) -> ScipDefHit` lookup. Built once per index pass
/// from `_entities` so per-symbol proxy creation stays O(1).
type ScipDefMap = HashMap<(String, String), ScipDefHit>;

/// Cached navigation metadata for an existing proxy. Stored alongside
/// `proxy_text` so the dedupe path can detect line moves and changed
/// underlying record IDs even when the proxy_text is byte-identical.
struct ExistingProxy {
    record_id: RecordId,
    proxy_text: String,
    line_start: Option<u64>,
    line_end: Option<u64>,
    source_record: Option<String>,
}

/// Cache of existing `_idx_code_proxies` rows keyed by `proxy_id`, plus a
/// running set of proxy_ids touched during the current index pass.
///
/// `existing` lets the dedupe path skip the (expensive) `embed_text` +
/// `index_text` calls when both `proxy_text` AND navigation metadata
/// are unchanged. Pure line moves (proxy_text identical, line range
/// shifted) refresh navigation fields via `db.update` without re-embed.
/// `touched` records ids emitted during the pass so the post-pass prune
/// loop can drop survivors whose symbol/section was removed.
#[derive(Default)]
struct ProxyDedupCache {
    existing: HashMap<String, ExistingProxy>,
    touched: std::collections::HashSet<String>,
}

/// The main project indexer.
pub struct ProjectIndexer<'a> {
    db: &'a Axil,
    config: IndexConfig,
    progress: Box<dyn IndexProgress>,
}

impl<'a> ProjectIndexer<'a> {
    pub fn new(db: &'a Axil, config: IndexConfig) -> Self {
        Self {
            db,
            config,
            progress: Box::new(NoopProgress),
        }
    }

    /// Attach a progress reporter. The CLI uses this for an indicatif
    /// progress bar; library users can pass their own implementation or
    /// leave the default `NoopProgress`.
    pub fn with_progress(mut self, progress: Box<dyn IndexProgress>) -> Self {
        self.progress = progress;
        self
    }

    /// Run a full index of the project at `root`.
    pub fn index_full(&self, root: &Path) -> axil_core::Result<IndexResult> {
        // Detect project
        let project_type = scanner::detect_project_type(root);
        let project_name = scanner::detect_project_name(root, project_type);

        // Scan files
        let files = scanner::scan_files(root, &self.config);
        self.progress.start(files.len());

        // Clear existing index tables
        self.clear_index_tables()?;

        // Parse and index each file
        let mut file_records = Vec::new();
        let mut all_symbols = Vec::new();
        let mut file_meta: Vec<(ScannedFile, ParsedFile, String, RecordId)> = Vec::new();
        let mut total_source_tokens = 0usize;

        let mut total_lines = 0usize;

        for (i, scanned) in files.iter().enumerate() {
            let source = match std::fs::read_to_string(&scanned.path) {
                Ok(s) => s,
                Err(_) => {
                    self.progress.file_indexed(i + 1, &scanned.rel_path);
                    continue;
                }
            };

            let parsed = parser::parse_file(&source, scanned.language, self.config.index_private);
            let line_count = source.lines().count();
            total_source_tokens += token::estimate_tokens(&source);
            total_lines += line_count;

            let file_data = build_file_record(scanned, &parsed, &source);
            let record = self.db.insert(TABLE_FILES, file_data)?;
            // Auto-embed summary for vector search (silently skip if no embedder)
            let _ = self.db.embed_field(&record.id, "summary");
            file_records.push((scanned.rel_path.clone(), record.id.clone(), parsed.clone()));
            file_meta.push((scanned.clone(), parsed.clone(), source, record.id.clone()));

            if self.config.symbol_depth != "none" {
                for sym in &parsed.symbols {
                    all_symbols.push((scanned.rel_path.clone(), sym.clone()));
                }
            }

            self.progress.file_indexed(i + 1, &scanned.rel_path);
        }

        // Index modules (group by directory)
        self.progress.phase("modules");
        let module_count = self.index_modules(root, &file_records)?;

        // Create graph edges (module→file, module→module deps)
        self.progress.phase("graph edges");
        self.create_graph_edges(&file_records)?;

        // Index symbols (returning record IDs so proxy creation can attach to them)
        self.progress.phase("symbols");
        let inserted_symbols = self.index_symbols_with_records(&all_symbols)?;
        let symbol_count = inserted_symbols.len();

        // Group inserted symbols back by file path so each file's proxy
        // step can link symbol proxies to their own _idx_symbols record.
        let mut symbols_by_file: HashMap<String, Vec<(parser::ParsedSymbol, RecordId)>> =
            HashMap::new();
        for (file_path, sym, sym_record) in inserted_symbols {
            symbols_by_file
                .entry(file_path)
                .or_default()
                .push((sym, sym_record));
        }
        self.progress.phase("code proxies");
        let scip_defs = self.build_scip_def_map();
        for (scanned, parsed, source, file_record) in &file_meta {
            let empty: Vec<(parser::ParsedSymbol, RecordId)> = Vec::new();
            let syms = symbols_by_file.get(&scanned.rel_path).unwrap_or(&empty);
            // Full index already wiped `_idx_code_proxies` via clear_index_tables,
            // so no dedup cache is useful here — every proxy is new.
            let _ = self.create_file_proxies(
                scanned,
                parsed,
                source,
                file_record,
                syms,
                &project_name,
                &scip_defs,
                None,
            );
        }

        // Index dependencies
        self.progress.phase("dependencies");
        let dep_count = self.index_dependencies(root, project_type)?;

        // Generate and store project overview
        self.progress.phase("project overview");
        self.index_project_overview(
            &project_name,
            project_type,
            files.len(),
            total_lines,
            &file_records,
            total_source_tokens,
        )?;

        // Calculate index token total from what we just inserted
        let index_tokens: usize = [
            TABLE_PROJECT,
            TABLE_FILES,
            TABLE_MODULES,
            TABLE_SYMBOLS,
            TABLE_DEPS,
            TABLE_CODE_PROXIES,
        ]
        .iter()
        .flat_map(|t| self.db.list(t).unwrap_or_default())
        .map(|r| r.data.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize)
        .sum();
        let ratio = if index_tokens > 0 {
            total_source_tokens / index_tokens
        } else {
            0
        };

        self.progress.finish();
        Ok(IndexResult {
            indexed_files: file_records.len(),
            modules: module_count,
            symbols: symbol_count,
            deps: dep_count,
            tables_created: vec![
                TABLE_PROJECT.to_string(),
                TABLE_FILES.to_string(),
                TABLE_MODULES.to_string(),
                TABLE_SYMBOLS.to_string(),
                TABLE_DEPS.to_string(),
                TABLE_CODE_PROXIES.to_string(),
            ],
            tokens_saved_estimate: format!("~{ratio}:1 compression ({index_tokens} index tokens vs {total_source_tokens} source tokens)"),
            changed: None,
            unchanged: None,
        })
    }

    /// Run an incremental index — only re-index files whose content has changed.
    ///
    /// Compares stored content hashes against current file hashes.
    /// Only changed files are re-parsed and their records updated.
    /// Parent modules and the project overview are regenerated.
    pub fn index_incremental(&self, root: &Path) -> axil_core::Result<IndexResult> {
        let project_type = scanner::detect_project_type(root);
        let project_name = scanner::detect_project_name(root, project_type);

        // Scan current files on disk
        let current_files = scanner::scan_files(root, &self.config);
        self.progress.start(current_files.len());

        // Build a map of stored file records: path → (record_id, content_hash)
        let stored_files = self.db.list(TABLE_FILES)?;
        let mut stored_map: HashMap<String, (RecordId, String)> = HashMap::new();
        for record in &stored_files {
            if let (Some(path), Some(hash)) = (
                record.data.get("path").and_then(|v| v.as_str()),
                record.data.get("content_hash").and_then(|v| v.as_str()),
            ) {
                stored_map.insert(path.to_string(), (record.id.clone(), hash.to_string()));
            }
        }

        // Determine which files changed
        let mut changed_count = 0usize;
        let mut unchanged_count = 0usize;
        // Cache source strings to avoid double-reads
        let mut changed_sources: Vec<(&ScannedFile, String)> = Vec::new();

        for (i, scanned) in current_files.iter().enumerate() {
            let source = match std::fs::read_to_string(&scanned.path) {
                Ok(s) => s,
                Err(_) => {
                    self.progress.file_indexed(i + 1, &scanned.rel_path);
                    continue;
                }
            };
            let current_hash = hash_content(&source);

            if let Some((_, stored_hash)) = stored_map.get(&scanned.rel_path) {
                if *stored_hash == current_hash {
                    unchanged_count += 1;
                    self.progress.file_indexed(i + 1, &scanned.rel_path);
                    continue;
                }
            }
            changed_count += 1;
            changed_sources.push((scanned, source));
            self.progress.file_indexed(i + 1, &scanned.rel_path);
        }

        // Detect deleted files
        let current_paths: std::collections::HashSet<&str> =
            current_files.iter().map(|f| f.rel_path.as_str()).collect();
        let has_deletes = stored_map
            .keys()
            .any(|p| !current_paths.contains(p.as_str()));

        if changed_count == 0 && !has_deletes {
            self.progress.finish();
            return Ok(IndexResult {
                indexed_files: 0,
                modules: 0,
                symbols: 0,
                deps: 0,
                tables_created: Vec::new(),
                tokens_saved_estimate: "no changes".to_string(),
                changed: Some(0),
                unchanged: Some(unchanged_count),
            });
        }

        // Build a HashSet for O(1) changed-path lookups
        let changed_paths: std::collections::HashSet<&str> = changed_sources
            .iter()
            .map(|(s, _)| s.rel_path.as_str())
            .collect();

        // Delete records for removed/changed files
        for (path, (id, _)) in &stored_map {
            if !current_paths.contains(path.as_str()) || changed_paths.contains(path.as_str()) {
                let _ = self.db.delete(id);
            }
        }

        // Delete old symbols for changed/deleted files
        let symbols = self.db.list(TABLE_SYMBOLS)?;
        for sym in &symbols {
            if let Some(file_path) = sym.data.get("file").and_then(|v| v.as_str()) {
                if !current_paths.contains(file_path) || changed_paths.contains(file_path) {
                    let _ = self.db.delete(&sym.id);
                }
            }
        }

        // Snapshot proxies before regenerating so the dedup cache (below)
        // can skip insert+embed for unchanged proxies. The post-pass prune
        // loop deletes survivors whose path is gone or whose proxy_id was
        // never re-emitted (i.e. the symbol/section was removed).
        let proxies_before_pass: Vec<axil_core::Record> =
            self.db.list(TABLE_CODE_PROXIES).unwrap_or_default();

        // Re-index changed files (using cached source strings — no re-read)
        let mut new_symbols = Vec::new();
        let mut changed_meta: Vec<(ScannedFile, ParsedFile, String, RecordId)> = Vec::new();
        for (scanned, source) in &changed_sources {
            let parsed = parser::parse_file(source, scanned.language, self.config.index_private);
            let file_data = build_file_record(scanned, &parsed, source);
            let record = self.db.insert(TABLE_FILES, file_data)?;
            let _ = self.db.embed_field(&record.id, "summary");
            changed_meta.push((
                (*scanned).clone(),
                parsed.clone(),
                source.clone(),
                record.id.clone(),
            ));

            if self.config.symbol_depth != "none" {
                for sym in &parsed.symbols {
                    new_symbols.push((scanned.rel_path.clone(), sym.clone()));
                }
            }
        }

        let inserted_symbols = self.index_symbols_with_records(&new_symbols)?;
        let symbol_count = inserted_symbols.len();

        // Recreate proxies for changed files only.
        let mut symbols_by_file: HashMap<String, Vec<(parser::ParsedSymbol, RecordId)>> =
            HashMap::new();
        for (file_path, sym, sym_record) in inserted_symbols {
            symbols_by_file
                .entry(file_path)
                .or_default()
                .push((sym, sym_record));
        }
        let scip_defs = self.build_scip_def_map();
        let mut dedup = ProxyDedupCache::default();
        for row in &proxies_before_pass {
            let pid = match row.data.get("proxy_id").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            dedup.existing.insert(
                pid,
                ExistingProxy {
                    record_id: row.id.clone(),
                    proxy_text: row
                        .data
                        .get("proxy_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    line_start: row.data.get("line_start").and_then(|v| v.as_u64()),
                    line_end: row.data.get("line_end").and_then(|v| v.as_u64()),
                    source_record: row
                        .data
                        .get("source_record")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                },
            );
        }
        for (scanned, parsed, source, file_record) in &changed_meta {
            let empty: Vec<(parser::ParsedSymbol, RecordId)> = Vec::new();
            let syms = symbols_by_file.get(&scanned.rel_path).unwrap_or(&empty);
            let _ = self.create_file_proxies(
                scanned,
                parsed,
                source,
                file_record,
                syms,
                &project_name,
                &scip_defs,
                Some(&mut dedup),
            );
        }
        // Prune proxies whose symbol/section was removed from a changed
        // file, plus all proxies owned by deleted files. Untouched proxies
        // belonging to unchanged files survive — that's the win: we
        // didn't re-embed them, and we don't drop them either.
        for row in &proxies_before_pass {
            let path = match row.data.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => continue,
            };
            let pid = row
                .data
                .get("proxy_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let path_dropped = !current_paths.contains(path);
            let path_changed = changed_paths.contains(path);
            let untouched = !dedup.touched.contains(pid);
            if path_dropped || (path_changed && untouched) {
                let _ = self.db.delete(&row.id);
            }
        }

        // Clear old graph edges for index tables
        self.clear_graph_edges()?;

        // Regenerate modules from stored record data (no file re-reads)
        let old_modules = self.db.list(TABLE_MODULES)?;
        for m in &old_modules {
            let _ = self.db.delete(&m.id);
        }

        let all_file_records = self.db.list(TABLE_FILES)?;
        let full_records: Vec<(String, RecordId, ParsedFile)> = all_file_records
            .iter()
            .map(|rec| {
                let path = rec
                    .data
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let parsed = reconstruct_parsed_file(&rec.data);
                (path, rec.id.clone(), parsed)
            })
            .collect();

        let module_count = self.index_modules(root, &full_records)?;

        // Recreate graph edges
        self.create_graph_edges(&full_records)?;

        // Regenerate deps and project overview
        let old_deps = self.db.list(TABLE_DEPS)?;
        for d in &old_deps {
            let _ = self.db.delete(&d.id);
        }
        let dep_count = self.index_dependencies(root, project_type)?;

        let old_projects = self.db.list(TABLE_PROJECT)?;
        for p in &old_projects {
            let _ = self.db.delete(&p.id);
        }

        // Compute totals from stored records (no file re-reads)
        let total_lines: usize = all_file_records
            .iter()
            .filter_map(|r| r.data.get("line_count").and_then(|v| v.as_u64()))
            .sum::<u64>() as usize;
        let total_source_tokens: usize = all_file_records
            .iter()
            .filter_map(|r| r.data.get("size_bytes").and_then(|v| v.as_u64()))
            .map(|bytes| (bytes as usize).div_ceil(4)) // same heuristic as estimate_tokens
            .sum();

        self.index_project_overview(
            &project_name,
            project_type,
            all_file_records.len(),
            total_lines,
            &full_records,
            total_source_tokens,
        )?;

        let index_tokens: usize = [
            TABLE_PROJECT,
            TABLE_FILES,
            TABLE_MODULES,
            TABLE_SYMBOLS,
            TABLE_DEPS,
            TABLE_CODE_PROXIES,
        ]
        .iter()
        .flat_map(|t| self.db.list(t).unwrap_or_default())
        .map(|r| r.data.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize)
        .sum();
        let ratio = if index_tokens > 0 {
            total_source_tokens / index_tokens
        } else {
            0
        };

        self.progress.finish();
        Ok(IndexResult {
            indexed_files: changed_sources.len(),
            modules: module_count,
            symbols: symbol_count,
            deps: dep_count,
            tables_created: Vec::new(),
            // Disclose the basis the same way the full-index path does, so a
            // re-index's ratio reads as the estimate it is. Source tokens here
            // come from cached `size_bytes` (no file re-read), not a fresh
            // count — the one honest difference from index_full.
            tokens_saved_estimate: format!(
                "~{ratio}:1 compression ({index_tokens} index tokens vs {total_source_tokens} source tokens, from cached sizes)"
            ),
            changed: Some(changed_count),
            unchanged: Some(unchanged_count),
        })
    }

    /// Check if an index already exists for this database.
    pub fn has_index(&self) -> bool {
        self.db
            .list(TABLE_PROJECT)
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn clear_index_tables(&self) -> axil_core::Result<()> {
        for table in &[
            TABLE_PROJECT,
            TABLE_FILES,
            TABLE_MODULES,
            TABLE_SYMBOLS,
            TABLE_DEPS,
            TABLE_CODE_PROXIES,
        ] {
            let records = self.db.list(table)?;
            for record in records {
                let _ = self.db.delete(&record.id);
            }
        }
        Ok(())
    }

    /// Insert a `CodeProxy` and explicitly embed + FTS-index it.
    ///
    /// `_idx_code_proxies` is an internal table — normal insert hooks skip
    /// it, so we must hand-roll vector embedding (`embed_text`) and FTS
    /// indexing (`index_text`) for `proxy_text`/`path`/`symbol`/`signature`/
    /// `breadcrumb`.
    ///
    /// Consults a `ProxyDedupCache` when present. If an existing proxy
    /// with the same `proxy_id` already has the same `proxy_text`, the
    /// existing `RecordId` is returned and both insert and embed are
    /// skipped — the dominant cost on incremental re-index. The cache's
    /// `touched` set records the id either way so the caller can prune
    /// removed symbols afterwards. Failures inside embed/index calls are
    /// silently ignored (matches existing `embed_field` style elsewhere
    /// in this file).
    fn store_proxy_with_dedupe(
        &self,
        proxy: &CodeProxy,
        dedup: Option<&mut ProxyDedupCache>,
    ) -> axil_core::Result<RecordId> {
        if let Some(cache) = dedup {
            cache.touched.insert(proxy.proxy_id.clone());
            if let Some(existing) = cache.existing.get(&proxy.proxy_id) {
                let id = existing.record_id.clone();
                let text_matches = existing.proxy_text == proxy.proxy_text;
                let nav_matches = existing.line_start == proxy.line_start.map(|n| n as u64)
                    && existing.line_end == proxy.line_end.map(|n| n as u64)
                    && existing.source_record.as_deref() == proxy.source_record.as_deref();

                if text_matches && nav_matches {
                    return Ok(id);
                }

                // Refresh storage. When proxy_text is unchanged, skip the
                // expensive embed/FTS pass — only navigation metadata
                // (line_start/line_end/source_record) shifted, e.g. on a
                // pure line move or after the underlying _idx_files /
                // _idx_symbols row got a new RecordId.
                let data = proxy_to_record(proxy);
                self.db.update(&id, data)?;
                if !text_matches {
                    let _ = self.db.embed_text(&id, &proxy.proxy_text);
                    self.fts_index_proxy(&id, proxy);
                }
                cache.existing.insert(
                    proxy.proxy_id.clone(),
                    ExistingProxy {
                        record_id: id.clone(),
                        proxy_text: proxy.proxy_text.clone(),
                        line_start: proxy.line_start.map(|n| n as u64),
                        line_end: proxy.line_end.map(|n| n as u64),
                        source_record: proxy.source_record.clone(),
                    },
                );
                return Ok(id);
            }
        }
        let data = proxy_to_record(proxy);
        let record = self.db.insert(TABLE_CODE_PROXIES, data)?;
        let _ = self.db.embed_text(&record.id, &proxy.proxy_text);
        self.fts_index_proxy(&record.id, proxy);
        Ok(record.id)
    }

    fn fts_index_proxy(&self, id: &RecordId, proxy: &CodeProxy) {
        let _ = self.db.index_text(id, "proxy_text", &proxy.proxy_text);
        let _ = self.db.index_text(id, "path", &proxy.path);
        if let Some(sym) = &proxy.symbol {
            let _ = self.db.index_text(id, "symbol", sym);
        }
        if let Some(sig) = &proxy.signature {
            let _ = self.db.index_text(id, "signature", sig);
        }
        let _ = self.db.index_text(id, "breadcrumb", &proxy.breadcrumb);
        if let Some(c) = &proxy.canonical_id {
            let _ = self.db.index_text(id, "canonical_id", c);
        }
    }

    /// Build a `(path, symbol_name) -> ScipDefHit` lookup table from
    /// `_entities` in a single pass. Empty when SCIP data is absent.
    ///
    /// Replaces a per-symbol full table scan with one O(entities) read at
    /// the start of an index pass — keeps `create_file_proxies` O(symbols)
    /// instead of O(symbols × entities).
    fn build_scip_def_map(&self) -> ScipDefMap {
        let rows = match self.db.list("_entities") {
            Ok(r) => r,
            Err(_) => return ScipDefMap::default(),
        };
        let mut map: ScipDefMap = HashMap::with_capacity(rows.len());
        for row in rows {
            let cid = match row.data.get("canonical_id").and_then(|v| v.as_str()) {
                Some(c) if !c.starts_with("provisional:") => c.to_string(),
                _ => continue,
            };
            let name = row
                .data
                .get("name")
                .and_then(|v| v.as_str())
                .or_else(|| row.data.get("symbol").and_then(|v| v.as_str()));
            let path = row
                .data
                .get("def_path")
                .and_then(|v| v.as_str())
                .or_else(|| row.data.get("path").and_then(|v| v.as_str()))
                .or_else(|| row.data.get("file").and_then(|v| v.as_str()));
            let (Some(name), Some(path)) = (name, path) else {
                continue;
            };
            let line_start = row
                .data
                .get("def_line_start")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let line_end = row
                .data
                .get("def_line_end")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            map.insert(
                (path.to_string(), name.to_string()),
                ScipDefHit {
                    canonical_id: cid,
                    line_start,
                    line_end,
                },
            );
        }
        map
    }

    /// Build proxies for a single file: one file proxy + N symbol proxies
    /// + (for markdown) one section proxy per heading. Inserts each through
    /// `store_proxy` so embedding and FTS happen consistently.
    ///
    /// Also wires graph edges between proxies that share a path
    /// (`same_file`) when the graph plugin is loaded — silently skipped
    /// otherwise. `tests` edges are best-effort by symbol-name
    /// convention (`test_<name>` / `it_<name>` / `<name>_test`).
    #[allow(clippy::too_many_arguments)]
    fn create_file_proxies(
        &self,
        scanned: &ScannedFile,
        parsed: &ParsedFile,
        source: &str,
        file_record: &RecordId,
        inserted_symbols: &[(parser::ParsedSymbol, RecordId)],
        project_name: &str,
        scip_defs: &ScipDefMap,
        dedup: Option<&mut ProxyDedupCache>,
    ) -> axil_core::Result<usize> {
        let mut count = 0usize;
        let module = std::path::Path::new(&scanned.rel_path)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("")
            .to_string();
        let language = scanned.language.as_str();
        let line_count = source.lines().count();

        // Inserted proxy ids for this file — used by the same_file/tests
        // edge pass after all per-file proxies are in.
        let mut symbol_proxy_ids: Vec<(String, RecordId)> = Vec::new(); // (symbol_name, proxy_id)

        // ── File proxy ───────────────────────────────────────────────
        let file_proxy = build_proxy(ProxyInput {
            kind: ProxyKind::File,
            project: Some(project_name),
            module: if module.is_empty() {
                None
            } else {
                Some(&module)
            },
            path: &scanned.rel_path,
            language: Some(language),
            symbol: None,
            signature: None,
            line_start: Some(1),
            line_end: Some(line_count.max(1)),
            canonical_id: None,
            summary: Some(parsed.summary.as_str()),
            doc: parsed.module_doc.as_deref(),
            imports: &parsed.imports,
            exports: &parsed.exports,
            key_types: &parsed.key_types,
            heading_path: None,
            source_record: Some(&file_record.to_string()),
            token_budget: 0,
        });
        // Reborrow the optional cache so each store call can take a mutable
        // borrow without giving up the outer Option. Needed because
        // `Option<&mut T>` doesn't implement Copy.
        let mut dedup = dedup;
        let file_proxy_id = self.store_proxy_with_dedupe(&file_proxy, dedup.as_deref_mut())?;
        count += 1;

        // ── Symbol proxies ───────────────────────────────────────────
        for (sym, sym_record) in inserted_symbols {
            let scip_hit = scip_defs.get(&(scanned.rel_path.clone(), sym.name.clone()));
            let canonical = scip_hit.map(|h| h.canonical_id.as_str());
            let line_start = scip_hit.and_then(|h| h.line_start).unwrap_or(sym.line);
            let line_end = scip_hit.and_then(|h| h.line_end);
            let proxy = build_proxy(ProxyInput {
                kind: ProxyKind::Symbol,
                project: Some(project_name),
                module: if module.is_empty() {
                    None
                } else {
                    Some(&module)
                },
                path: &scanned.rel_path,
                language: Some(language),
                symbol: Some(sym.name.as_str()),
                signature: Some(sym.signature.as_str()),
                line_start: Some(line_start),
                line_end,
                canonical_id: canonical,
                summary: None,
                doc: sym.doc.as_deref(),
                imports: &[],
                exports: &[],
                key_types: &[],
                heading_path: None,
                source_record: Some(&sym_record.to_string()),
                token_budget: 0,
            });
            let pid = self.store_proxy_with_dedupe(&proxy, dedup.as_deref_mut())?;
            symbol_proxy_ids.push((sym.name.clone(), pid));
            count += 1;
        }

        // ── Section proxies ────────────────────────────
        // Markdown headings (P0) + TOML/JSON top-level sections (P1).
        let sections: Vec<crate::markdown::ParsedSection> = match scanned.language {
            Language::Markdown => split_sections(source),
            Language::Toml => split_toml_sections(source),
            Language::Json => split_json_sections(source),
            Language::Yaml => split_yaml_sections(source),
            _ => Vec::new(),
        };
        for section in &sections {
            let canonical = section_canonical_id(&scanned.rel_path, &section.heading_path);
            let proxy = build_proxy(ProxyInput {
                kind: ProxyKind::Section,
                project: Some(project_name),
                module: if module.is_empty() {
                    None
                } else {
                    Some(&module)
                },
                path: &scanned.rel_path,
                language: Some(language),
                symbol: Some(section.heading.as_str()),
                signature: None,
                line_start: Some(section.line_start),
                line_end: Some(section.line_end),
                canonical_id: Some(canonical.as_str()),
                summary: None,
                doc: Some(section.body.as_str()),
                imports: &[],
                exports: &[],
                key_types: &[],
                heading_path: Some(&section.heading_path),
                source_record: Some(&file_record.to_string()),
                token_budget: 0,
            });
            let _ = self.store_proxy_with_dedupe(&proxy, dedup.as_deref_mut())?;
            count += 1;
        }

        // The file proxy gets one `same_file` edge per symbol (linear);
        // recall traverses from a symbol hit back to its siblings via the
        // file hub. Symbol-to-symbol pair edges are O(N²) per file, so
        // generated/macro-expanded files (200+ symbols) get capped at
        // MAX_PAIR_SYMBOLS_PER_FILE pair edges; above the cap the
        // file-proxy hub keeps the graph still useful (one extra hop).
        const MAX_PAIR_SYMBOLS_PER_FILE: usize = 32; // ≈ 496 pair edges max
        if self.db.has_graph_index() {
            for (_sym_name, pid) in &symbol_proxy_ids {
                let _ = self.db.relate(&file_proxy_id, "same_file", pid, None);
            }
            if symbol_proxy_ids.len() <= MAX_PAIR_SYMBOLS_PER_FILE {
                for (i, (_, a)) in symbol_proxy_ids.iter().enumerate() {
                    for (_, b) in symbol_proxy_ids.iter().skip(i + 1) {
                        let _ = self.db.relate(a, "same_file", b, None);
                    }
                }
            }
        }

        // Emit `tests` edges from test-symbol proxies to their inferred
        // target symbol proxy. Heuristic: when the path looks like a test
        // file or the symbol name starts with `test_` / `it_`, strip the
        // prefix and look up a sibling symbol proxy by name in this file.
        // Cross-file resolution is left for the SCIP graph layer.
        if self.db.has_graph_index() && looks_like_test_path(&scanned.rel_path) {
            let by_name: HashMap<&str, &RecordId> = symbol_proxy_ids
                .iter()
                .map(|(n, p)| (n.as_str(), p))
                .collect();
            for (sym_name, pid) in &symbol_proxy_ids {
                if let Some(target) = test_target_name(sym_name) {
                    if let Some(target_pid) = by_name.get(target) {
                        if *target_pid != pid {
                            let _ = self.db.relate(pid, "tests", target_pid, None);
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    fn index_modules(
        &self,
        _root: &Path,
        file_records: &[(String, RecordId, ParsedFile)],
    ) -> axil_core::Result<usize> {
        // Group files by parent directory
        let mut dir_files: HashMap<String, Vec<&(String, RecordId, ParsedFile)>> = HashMap::new();

        for entry in file_records {
            let dir = Path::new(&entry.0)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            dir_files.entry(dir).or_default().push(entry);
        }

        let mut count = 0;
        for (dir_path, entries) in &dir_files {
            let module_name = Path::new(dir_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(dir_path)
                .to_string();

            // Aggregate file summaries for module summary
            let file_names: Vec<&str> = entries
                .iter()
                .map(|(path, _, _)| {
                    Path::new(path.as_str())
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(path)
                })
                .collect();

            let summaries: Vec<&str> = entries
                .iter()
                .map(|(_, _, parsed)| parsed.summary.as_str())
                .filter(|s| *s != "no summary available")
                .collect();

            let module_summary = if summaries.is_empty() {
                format!("{} files", entries.len())
            } else {
                // Deduplicate and combine unique summaries
                let mut unique: Vec<&str> = Vec::new();
                for s in &summaries {
                    if !unique.iter().any(|u| u == s) {
                        unique.push(s);
                    }
                }
                let combined = unique.join(". ");
                let max_chars = self.config.max_module_summary_tokens * 4;
                if combined.len() > max_chars {
                    // Truncate at sentence boundary
                    let truncated = &combined[..max_chars];
                    match truncated.rfind(". ") {
                        Some(pos) => format!("{}.", &truncated[..pos]),
                        None => format!("{truncated}..."),
                    }
                } else {
                    combined
                }
            };

            // Collect public API across files
            let public_api: Vec<String> = entries
                .iter()
                .flat_map(|(_, _, parsed)| parsed.exports.iter().cloned())
                .take(20)
                .collect();

            // Internal deps (other module dirs referenced in imports)
            let internal_deps: Vec<String> = entries
                .iter()
                .flat_map(|(_, _, parsed)| parsed.imports.iter().cloned())
                .filter(|imp| {
                    // Check if import matches another module directory
                    dir_files.keys().any(|d| d.contains(imp.as_str()))
                })
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            // External deps
            let external_deps: Vec<String> = entries
                .iter()
                .flat_map(|(_, _, parsed)| parsed.imports.iter().cloned())
                .filter(|imp| !internal_deps.contains(imp))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(10)
                .collect();

            let module_data = json!({
                "path": format!("{dir_path}/"),
                "name": module_name,
                "summary": module_summary,
                "files": file_names,
                "public_api": public_api,
                "internal_deps": internal_deps,
                "external_deps": external_deps,
                "tokens": token::estimate_json_tokens(&json!({"summary": &module_summary})),
            });

            let module_record = self.db.insert(TABLE_MODULES, module_data)?;
            let _ = self.db.embed_field(&module_record.id, "summary");
            count += 1;
        }

        Ok(count)
    }

    /// Insert one `_idx_symbols` record per parsed symbol and return the
    /// inserted record IDs alongside their parsed symbol. structural proxy
    /// creation needs the record IDs so symbol proxies can point at their
    /// `_idx_symbols` source record without a second discovery pass.
    fn index_symbols_with_records(
        &self,
        symbols: &[(String, parser::ParsedSymbol)],
    ) -> axil_core::Result<Vec<(String, parser::ParsedSymbol, RecordId)>> {
        let mut out = Vec::with_capacity(symbols.len());
        for (file_path, sym) in symbols {
            let data = json!({
                "name": sym.name,
                "kind": sym.kind.as_str(),
                "file": file_path,
                "line": sym.line,
                "signature": sym.signature,
                "doc": sym.doc,
                "tokens": token::estimate_tokens(
                    &format!("{} {} {}", sym.name, sym.signature, sym.doc.as_deref().unwrap_or(""))
                ),
            });
            let record = self.db.insert(TABLE_SYMBOLS, data)?;
            out.push((file_path.clone(), sym.clone(), record.id));
        }
        Ok(out)
    }

    /// Create graph edges: module →contains→ file, module →depends_on→ module.
    /// Silently skips if no graph plugin is loaded.
    fn create_graph_edges(
        &self,
        file_records: &[(String, RecordId, ParsedFile)],
    ) -> axil_core::Result<()> {
        if !self.db.has_graph_index() {
            return Ok(());
        }

        // Build path→id map for files
        let file_id_map: HashMap<String, &RecordId> = file_records
            .iter()
            .map(|(path, id, _)| (path.clone(), id))
            .collect();

        let modules = self.db.list(TABLE_MODULES)?;

        // Build module name→id map for exact dependency matching
        let module_name_map: HashMap<String, RecordId> = modules
            .iter()
            .filter_map(|m| {
                let name = m.data.get("name").and_then(|v| v.as_str())?.to_string();
                Some((name, m.id.clone()))
            })
            .collect();

        // module →contains→ file
        for module in &modules {
            let module_path = match module.data.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => continue,
            };

            if let Some(files) = module.data.get("files").and_then(|v| v.as_array()) {
                for file_name in files {
                    if let Some(name) = file_name.as_str() {
                        // For root-level modules (path="/"), files have no directory prefix
                        let file_path = if module_path == "/" {
                            name.to_string()
                        } else {
                            format!("{}{}", module_path, name)
                        };
                        if let Some(file_id) = file_id_map.get(&file_path) {
                            let _ = self.db.relate(&module.id, "contains", file_id, None);
                        }
                    }
                }
            }

            // module →depends_on→ module (via internal_deps, exact name match)
            if let Some(deps) = module.data.get("internal_deps").and_then(|v| v.as_array()) {
                for dep in deps {
                    if let Some(dep_name) = dep.as_str() {
                        if let Some(target_id) = module_name_map.get(dep_name) {
                            if *target_id != module.id {
                                let _ = self.db.relate(&module.id, "depends_on", target_id, None);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove graph edges created by the indexer.
    /// Used during incremental re-indexing before recreating edges.
    fn clear_graph_edges(&self) -> axil_core::Result<()> {
        if !self.db.has_graph_index() {
            return Ok(());
        }

        // Remove edges originating from module records
        let modules = self.db.list(TABLE_MODULES)?;
        for module in &modules {
            if let Ok(edges) = self.db.edges(&module.id, None, axil_core::Direction::Out) {
                for edge in &edges {
                    let _ = self.db.unrelate(&edge.id);
                }
            }
        }

        Ok(())
    }

    fn index_dependencies(
        &self,
        root: &Path,
        project_type: ProjectType,
    ) -> axil_core::Result<usize> {
        let deps = parse_dependencies(root, project_type);
        let mut count = 0;
        for dep in deps {
            self.db.insert(TABLE_DEPS, dep)?;
            count += 1;
        }
        Ok(count)
    }

    fn index_project_overview(
        &self,
        name: &str,
        project_type: ProjectType,
        file_count: usize,
        total_lines: usize,
        file_records: &[(String, RecordId, ParsedFile)],
        total_source_tokens: usize,
    ) -> axil_core::Result<()> {
        // Detect tech stack from imports
        let mut tech_stack: Vec<String> = Vec::new();
        let mut all_imports: HashMap<String, usize> = HashMap::new();
        for (_, _, parsed) in file_records {
            for imp in &parsed.imports {
                *all_imports.entry(imp.clone()).or_default() += 1;
            }
        }
        let mut sorted_imports: Vec<_> = all_imports.into_iter().collect();
        sorted_imports.sort_by(|a, b| b.1.cmp(&a.1));
        for (imp, _) in sorted_imports.iter().take(10) {
            if ![
                "std", "crate", "self", "super", "os", "sys", "path", "io", "fs",
            ]
            .contains(&imp.as_str())
            {
                tech_stack.push(imp.clone());
            }
        }

        // Detect modules
        let mut module_names: Vec<String> = Vec::new();
        let modules = self.db.list(TABLE_MODULES)?;
        for m in &modules {
            if let Some(name) = m.data.get("name").and_then(|v| v.as_str()) {
                module_names.push(name.to_string());
            }
        }

        // Detect entry points
        let entry_points: Vec<String> = file_records
            .iter()
            .filter(|(path, _, _)| {
                path.ends_with("main.rs")
                    || path.ends_with("main.py")
                    || path.ends_with("index.ts")
                    || path.ends_with("index.js")
                    || path.ends_with("app.ts")
                    || path.ends_with("app.py")
                    || path.ends_with("main.go")
                    || path.ends_with("lib.rs")
            })
            .map(|(path, _, _)| path.clone())
            .collect();

        // Detect conventions
        let mut conventions = serde_json::Map::new();

        // Error handling convention
        let has_thiserror = file_records
            .iter()
            .any(|(_, _, p)| p.patterns.contains(&"error_handling".to_string()));
        if has_thiserror {
            conventions.insert("error_handling".to_string(), json!("thiserror"));
        }

        // Testing convention
        let has_tests = file_records
            .iter()
            .any(|(_, _, p)| p.patterns.contains(&"tests".to_string()));
        if has_tests {
            conventions.insert("testing".to_string(), json!("inline tests detected"));
        }

        let project_data = json!({
            "name": name,
            "type": project_type.as_str(),
            "summary": format!(
                "{name} — {type} project with {files} files, {modules} modules",
                name = name,
                type = project_type.as_str(),
                files = file_count,
                modules = module_names.len(),
            ),
            "tech_stack": tech_stack,
            "entry_points": entry_points,
            "modules": module_names,
            "file_count": file_count,
            "line_count": total_lines,
            "total_source_tokens": total_source_tokens,
            "conventions": Value::Object(conventions),
            "indexed_at": Utc::now().to_rfc3339(),
            "tokens": token::estimate_tokens(&format!(
                "{name} {type} {tech}",
                name = name,
                type = project_type.as_str(),
                tech = tech_stack.join(" "),
            )),
        });

        let proj_record = self.db.insert(TABLE_PROJECT, project_data)?;
        let _ = self.db.embed_field(&proj_record.id, "summary");
        Ok(())
    }
}

// ── Dependency parsing ────────────────────────────────────────────────

fn parse_dependencies(root: &Path, project_type: ProjectType) -> Vec<Value> {
    match project_type {
        ProjectType::Rust => parse_cargo_deps(root),
        ProjectType::TypeScript | ProjectType::JavaScript => parse_npm_deps(root),
        ProjectType::Python => parse_python_deps(root),
        _ => Vec::new(),
    }
}

fn parse_cargo_deps(root: &Path) -> Vec<Value> {
    let cargo_path = root.join("Cargo.toml");
    let contents = match std::fs::read_to_string(&cargo_path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let parsed: toml::Table = match contents.parse() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut deps = Vec::new();

    // Check [dependencies], [workspace.dependencies], [dev-dependencies]
    let dep_tables = [("dependencies", "direct"), ("dev-dependencies", "dev")];

    for (table_name, kind) in &dep_tables {
        if let Some(dep_table) = parsed.get(*table_name).and_then(|v| v.as_table()) {
            for (name, value) in dep_table {
                let version = match value {
                    toml::Value::String(v) => v.clone(),
                    toml::Value::Table(t) => t
                        .get("version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("*")
                        .to_string(),
                    _ => "*".to_string(),
                };

                // Skip path dependencies (workspace members)
                if let toml::Value::Table(t) = value {
                    if t.contains_key("path") {
                        continue;
                    }
                }

                deps.push(json!({
                    "name": name,
                    "version": version,
                    "kind": kind,
                    "purpose": infer_dep_purpose(name),
                }));
            }
        }
    }

    // Workspace dependencies
    if let Some(ws) = parsed.get("workspace").and_then(|v| v.as_table()) {
        if let Some(ws_deps) = ws.get("dependencies").and_then(|v| v.as_table()) {
            for (name, value) in ws_deps {
                let version = match value {
                    toml::Value::String(v) => v.clone(),
                    toml::Value::Table(t) => t
                        .get("version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("*")
                        .to_string(),
                    _ => "*".to_string(),
                };
                if let toml::Value::Table(t) = value {
                    if t.contains_key("path") {
                        continue;
                    }
                }
                if !deps
                    .iter()
                    .any(|d| d.get("name").and_then(|v| v.as_str()) == Some(name))
                {
                    deps.push(json!({
                        "name": name,
                        "version": version,
                        "kind": "workspace",
                        "purpose": infer_dep_purpose(name),
                    }));
                }
            }
        }
    }

    deps
}

fn parse_npm_deps(root: &Path) -> Vec<Value> {
    let pkg_path = root.join("package.json");
    let contents = match std::fs::read_to_string(&pkg_path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let parsed: Value = match serde_json::from_str(&contents) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut deps = Vec::new();

    for (section, kind) in &[("dependencies", "direct"), ("devDependencies", "dev")] {
        if let Some(dep_obj) = parsed.get(*section).and_then(|v| v.as_object()) {
            for (name, version) in dep_obj {
                deps.push(json!({
                    "name": name,
                    "version": version.as_str().unwrap_or("*"),
                    "kind": kind,
                    "purpose": infer_dep_purpose(name),
                }));
            }
        }
    }

    deps
}

fn parse_python_deps(root: &Path) -> Vec<Value> {
    let mut deps = Vec::new();

    // Try pyproject.toml first
    if let Ok(contents) = std::fs::read_to_string(root.join("pyproject.toml")) {
        if let Ok(parsed) = contents.parse::<toml::Table>() {
            if let Some(project) = parsed.get("project").and_then(|v| v.as_table()) {
                if let Some(dep_list) = project.get("dependencies").and_then(|v| v.as_array()) {
                    for dep in dep_list {
                        if let Some(dep_str) = dep.as_str() {
                            let name = dep_str
                                .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                                .next()
                                .unwrap_or(dep_str)
                                .to_string();
                            deps.push(json!({
                                "name": name,
                                "version": dep_str,
                                "kind": "direct",
                                "purpose": infer_dep_purpose(&name),
                            }));
                        }
                    }
                }
            }
        }
    }

    // Fallback: requirements.txt
    if deps.is_empty() {
        if let Ok(contents) = std::fs::read_to_string(root.join("requirements.txt")) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let name = line
                    .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                    .next()
                    .unwrap_or(line)
                    .to_string();
                deps.push(json!({
                    "name": name,
                    "version": line,
                    "kind": "direct",
                    "purpose": infer_dep_purpose(&name),
                }));
            }
        }
    }

    deps
}

/// Best-effort purpose inference for common dependencies.
fn infer_dep_purpose(name: &str) -> &'static str {
    match name {
        // Rust
        "serde" | "serde_json" => "serialization",
        "tokio" | "async-std" => "async runtime",
        "actix-web" | "axum" | "warp" | "rocket" => "web framework",
        "sqlx" | "diesel" | "sea-orm" | "rusqlite" => "database",
        "thiserror" | "anyhow" | "eyre" => "error handling",
        "clap" | "structopt" => "CLI framework",
        "tracing" | "log" | "env_logger" => "logging",
        "reqwest" | "hyper" => "HTTP client",
        "chrono" | "time" => "date/time",
        "regex" => "regular expressions",
        "redb" => "embedded database",
        "tantivy" => "full-text search",
        // JS/TS
        "react" | "react-dom" => "UI framework",
        "next" => "fullstack framework",
        "express" | "fastify" | "koa" => "web framework",
        "prisma" | "typeorm" | "drizzle-orm" => "ORM/database",
        "typescript" => "type system",
        "jest" | "vitest" | "mocha" => "testing",
        "tailwindcss" => "CSS framework",
        "zod" | "yup" | "joi" => "validation",
        // Python
        "fastapi" | "flask" | "django" => "web framework",
        "pydantic" => "validation",
        "pytest" => "testing",
        "sqlalchemy" => "ORM/database",
        "numpy" | "pandas" => "data science",
        "requests" | "httpx" => "HTTP client",
        _ => "",
    }
}

/// Build the JSON data for a file record.
fn build_file_record(scanned: &ScannedFile, parsed: &ParsedFile, source: &str) -> Value {
    let line_count = source.lines().count();
    let modified = scanned
        .modified
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| {
            chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .unwrap_or_default()
                .to_rfc3339()
        })
        .unwrap_or_default();

    json!({
        "path": scanned.rel_path,
        "language": scanned.language.as_str(),
        "size_bytes": scanned.size_bytes,
        "line_count": line_count,
        "summary": parsed.summary,
        "exports": parsed.exports,
        "imports": parsed.imports,
        "key_types": parsed.key_types,
        "patterns": parsed.patterns,
        "complexity": complexity_label(line_count, parsed.symbols.len()),
        "last_modified": modified,
        "content_hash": hash_content(source),
        "tokens": token::estimate_json_tokens(&json!({"summary": &parsed.summary})),
    })
}

/// Reconstruct a lightweight ParsedFile from stored record JSON.
/// Used during incremental re-indexing to avoid re-reading source files.
fn reconstruct_parsed_file(data: &Value) -> ParsedFile {
    let get_string_array = |key: &str| -> Vec<String> {
        data.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };

    ParsedFile {
        summary: data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        exports: get_string_array("exports"),
        imports: get_string_array("imports"),
        key_types: get_string_array("key_types"),
        patterns: get_string_array("patterns"),
        symbols: Vec::new(), // Symbols are stored separately in _idx_symbols
        module_doc: None,
    }
}

/// Compute a fast content hash for change detection.
pub fn hash_content(content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Heuristic: a path is a test file when any path component is one of
/// `tests`, `test`, `__tests__`, `spec`, or the basename contains the
/// `test`/`spec` token (e.g. `auth_test.rs`, `auth.spec.ts`).
pub(crate) fn looks_like_test_path(rel_path: &str) -> bool {
    let lower = rel_path.to_lowercase();
    let path = std::path::Path::new(&lower);
    for component in path.components() {
        if let Some(s) = component.as_os_str().to_str() {
            if matches!(s, "tests" | "test" | "__tests__" | "spec" | "specs") {
                return true;
            }
        }
    }
    let basename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if basename.contains("_test")
        || basename.contains("_spec")
        || basename.contains(".test")
        || basename.contains(".spec")
    {
        return true;
    }
    if basename.starts_with("test_")
        || basename.starts_with("spec_")
        || basename.ends_with("_test")
        || basename.ends_with("_spec")
    {
        return true;
    }
    false
}

/// Heuristic: peel a conventional test prefix/suffix off `name` and
/// return the inferred target symbol name. Returns `None` when the
/// symbol does not look like a test.
pub(crate) fn test_target_name(name: &str) -> Option<&str> {
    if let Some(rest) = name.strip_prefix("test_") {
        if !rest.is_empty() {
            return Some(rest);
        }
    }
    if let Some(rest) = name.strip_prefix("it_") {
        if !rest.is_empty() {
            return Some(rest);
        }
    }
    if let Some(rest) = name.strip_suffix("_test") {
        if !rest.is_empty() {
            return Some(rest);
        }
    }
    None
}

fn complexity_label(line_count: usize, symbol_count: usize) -> &'static str {
    if line_count < 50 && symbol_count < 5 {
        "low"
    } else if line_count < 200 && symbol_count < 15 {
        "medium"
    } else {
        "high"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complexity_labels() {
        assert_eq!(complexity_label(10, 2), "low");
        assert_eq!(complexity_label(100, 10), "medium");
        assert_eq!(complexity_label(500, 30), "high");
    }

    #[test]
    fn infer_common_deps() {
        assert_eq!(infer_dep_purpose("serde"), "serialization");
        assert_eq!(infer_dep_purpose("tokio"), "async runtime");
        assert_eq!(infer_dep_purpose("react"), "UI framework");
        assert_eq!(infer_dep_purpose("obscure_crate"), "");
    }

    #[test]
    fn looks_like_test_path_recognizes_common_layouts() {
        assert!(looks_like_test_path("tests/foo.rs"));
        assert!(looks_like_test_path("crate/tests/integration.rs"));
        assert!(looks_like_test_path("src/auth_test.rs"));
        assert!(looks_like_test_path("src/auth.test.ts"));
        assert!(looks_like_test_path("src/auth.spec.ts"));
        assert!(looks_like_test_path("__tests__/login.js"));
        assert!(looks_like_test_path("spec/login_spec.rb"));
        // `_spec` basename suffix outside a `spec/` directory should
        // also count — e.g. RSpec files placed under app/ or src/.
        assert!(looks_like_test_path("src/login_spec.rb"));
        assert!(looks_like_test_path("app/models/user_spec.rb"));

        assert!(!looks_like_test_path("src/lib.rs"));
        assert!(!looks_like_test_path("src/auth/login.ts"));
        assert!(!looks_like_test_path("crates/axil-core/src/db.rs"));
    }

    #[test]
    fn test_target_name_strips_conventional_affixes() {
        assert_eq!(test_target_name("test_login"), Some("login"));
        assert_eq!(test_target_name("it_login"), Some("login"));
        assert_eq!(test_target_name("login_test"), Some("login"));

        // Empty residue isn't a target.
        assert_eq!(test_target_name("test_"), None);
        assert_eq!(test_target_name("_test"), None);
        // Plain names without an affix aren't tests.
        assert_eq!(test_target_name("login"), None);
        assert_eq!(test_target_name("validate"), None);
    }
}
