use std::collections::HashMap;
use std::path::Path;

use parking_lot::Mutex;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, Value, STORED, STRING, TEXT};
use tantivy::snippet::SnippetGenerator;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use axil_core::record::RecordId;
use axil_core::AxilError;

/// Maximum query length in bytes to prevent resource exhaustion from
/// complex wildcard/regex patterns in tantivy's Lucene-like parser.
const MAX_QUERY_LEN: usize = 512;

/// Maximum text length (in bytes) indexed per field. Longer text is truncated
/// at a UTF-8 boundary to prevent excessive tokenizer memory usage.
const MAX_INDEXED_TEXT_LEN: usize = 65_536;

/// Over-fetch factor when deduplicating multi-field results.
const DEDUP_OVER_FETCH: usize = 5;

/// High-value field names that get a score boost in global search.
/// Matches in these fields are 2x more important than matches in other fields.
const BOOSTED_FIELDS: &[&str] = &["title", "summary", "name", "subject", "heading"];
const BOOST_FACTOR: f32 = 2.0;

/// Schema version marker for migration detection.
/// v2: added field_name. v3: body STORED for snippet generation.
const SCHEMA_VERSION: &str = "3";

/// Low-level full-text search index backed by tantivy.
///
/// Each record field is indexed as a separate tantivy document with:
/// - `id` — the record ID (STRING | STORED, for full-record deletion and dedup)
/// - `doc_key` — `"{id}\0{field_name}"` composite (STRING, for per-field deletion)
/// - `field_name` — the JSON field name (STRING | STORED, for field-scoped search)
/// - `body` — the searchable text content (TEXT, for BM25 search)
pub struct FtsIndex {
    id_field: Field,
    doc_key_field: Field,
    field_name_field: Field,
    body_field: Field,
    index: Index,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    /// True if the index was rebuilt during open (schema migration).
    rebuilt: bool,
}

impl FtsIndex {
    /// Whether the index was rebuilt during open (schema migration).
    /// When true, the caller should reindex all records.
    pub fn needs_reindex(&self) -> bool {
        self.rebuilt
    }

    /// Create or open an FTS index at the given directory path.
    pub fn new(path: &Path) -> axil_core::Result<Self> {
        std::fs::create_dir_all(path).map_err(|e| {
            AxilError::plugin(format!(
                "failed to create FTS directory {}: {e}",
                path.display()
            ))
        })?;

        // Check for schema migration need.
        let version_file = path.join(".schema_version");
        let needs_rebuild = if version_file.exists() {
            let v = std::fs::read_to_string(&version_file).unwrap_or_default();
            v.trim() != SCHEMA_VERSION
        } else {
            // No version file — could be v1 or fresh. If tantivy files exist, rebuild.
            path.join("meta.json").exists()
        };

        if needs_rebuild {
            Self::rebuild_index_dir(path)?;
        }

        let (schema, id_field, doc_key_field, field_name_field, body_field) = Self::build_schema();

        let dir = MmapDirectory::open(path)
            .map_err(|e| AxilError::plugin(format!("failed to open FTS directory: {e}")))?;
        let index = Index::open_or_create(dir, schema)
            .map_err(|e| AxilError::plugin(format!("failed to open/create FTS index: {e}")))?;

        let writer = index
            .writer(15_000_000)
            .map_err(|e| AxilError::plugin(format!("failed to create FTS writer: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e: tantivy::TantivyError| {
                AxilError::plugin(format!("failed to create FTS reader: {e}"))
            })?;

        std::fs::write(&version_file, SCHEMA_VERSION).map_err(|e| {
            AxilError::plugin(format!(
                "failed to write FTS schema version file {}: {e}",
                version_file.display()
            ))
        })?;

        Ok(Self {
            id_field,
            doc_key_field,
            field_name_field,
            body_field,
            index,
            writer: Mutex::new(writer),
            reader,
            rebuilt: needs_rebuild,
        })
    }

    fn build_schema() -> (Schema, Field, Field, Field, Field) {
        let mut sb = Schema::builder();
        let id_field = sb.add_text_field("id", STRING | STORED);
        let doc_key_field = sb.add_text_field("doc_key", STRING);
        let field_name_field = sb.add_text_field("field_name", STRING | STORED);
        let body_field = sb.add_text_field("body", TEXT | STORED);
        (
            sb.build(),
            id_field,
            doc_key_field,
            field_name_field,
            body_field,
        )
    }

    fn rebuild_index_dir(path: &Path) -> axil_core::Result<()> {
        // Atomic rebuild: create new empty dir, rename old → .old, rename new → target,
        // then remove .old. If we crash between renames, .old still has data.
        let old_path = path.with_extension("fts.old");
        let new_path = path.with_extension("fts.new");
        // Clean up any leftover temp dirs from prior crashes.
        let _ = std::fs::remove_dir_all(&new_path);
        let _ = std::fs::remove_dir_all(&old_path);
        std::fs::create_dir_all(&new_path).map_err(|e| {
            AxilError::plugin(format!(
                "FTS rebuild: failed to create temp dir {}: {e}",
                new_path.display()
            ))
        })?;
        // Rename existing → .old (preserves data if crash happens next).
        if path.exists() {
            std::fs::rename(path, &old_path).map_err(|e| {
                AxilError::plugin(format!(
                    "FTS rebuild: failed to rename {}: {e}",
                    path.display()
                ))
            })?;
        }
        // Rename new → target.
        std::fs::rename(&new_path, path).map_err(|e| {
            AxilError::plugin(format!("FTS rebuild: failed to rename new dir: {e}"))
        })?;
        // Clean up old dir (best effort).
        let _ = std::fs::remove_dir_all(&old_path);
        Ok(())
    }

    /// Build the composite doc_key: `"{id}\0{field_name}"`.
    fn doc_key(id: &RecordId, field: &str) -> String {
        format!("{}\0{}", id.as_str(), field)
    }

    /// Index multiple fields for a single record (full replacement).
    pub fn add_document(
        &self,
        id: &RecordId,
        fields: &HashMap<String, String>,
    ) -> axil_core::Result<()> {
        self.add_document_impl(id, fields, true)
    }

    /// Index multiple fields without committing (for batch use).
    /// Call `commit()` after the batch to flush changes.
    pub fn add_document_deferred(
        &self,
        id: &RecordId,
        fields: &HashMap<String, String>,
    ) -> axil_core::Result<()> {
        self.add_document_impl(id, fields, false)
    }

    fn add_document_impl(
        &self,
        id: &RecordId,
        fields: &HashMap<String, String>,
        do_commit: bool,
    ) -> axil_core::Result<()> {
        let mut writer = self.writer.lock();
        let term = Term::from_field_text(self.id_field, id.as_str());
        writer.delete_term(term);
        for (field_name, text) in fields {
            let truncated = truncate_utf8(text, MAX_INDEXED_TEXT_LEN);
            let mut doc = TantivyDocument::default();
            doc.add_text(self.id_field, id.as_str());
            doc.add_text(self.doc_key_field, Self::doc_key(id, field_name));
            doc.add_text(self.field_name_field, field_name);
            doc.add_text(self.body_field, truncated);
            writer
                .add_document(doc)
                .map_err(|e| AxilError::plugin(format!("failed to add FTS document: {e}")))?;
        }
        if do_commit {
            Self::commit_writer(&mut writer, &self.reader)?;
        }
        Ok(())
    }

    /// Index a single field for a record (per-field upsert).
    pub fn index_field(&self, id: &RecordId, field: &str, text: &str) -> axil_core::Result<()> {
        self.index_field_impl(id, field, text, true)
    }

    /// Index a single field without committing (for batch use).
    pub fn index_field_deferred(
        &self,
        id: &RecordId,
        field: &str,
        text: &str,
    ) -> axil_core::Result<()> {
        self.index_field_impl(id, field, text, false)
    }

    fn index_field_impl(
        &self,
        id: &RecordId,
        field: &str,
        text: &str,
        do_commit: bool,
    ) -> axil_core::Result<()> {
        let mut writer = self.writer.lock();
        let key = Self::doc_key(id, field);
        let key_term = Term::from_field_text(self.doc_key_field, &key);
        writer.delete_term(key_term);
        let truncated = truncate_utf8(text, MAX_INDEXED_TEXT_LEN);
        let mut doc = TantivyDocument::default();
        doc.add_text(self.id_field, id.as_str());
        doc.add_text(self.doc_key_field, &key);
        doc.add_text(self.field_name_field, field);
        doc.add_text(self.body_field, truncated);
        writer
            .add_document(doc)
            .map_err(|e| AxilError::plugin(format!("failed to add FTS document: {e}")))?;
        if do_commit {
            Self::commit_writer(&mut writer, &self.reader)?;
        }
        Ok(())
    }

    /// Remove all indexed documents for a record ID.
    pub fn remove_document(&self, id: &RecordId) -> axil_core::Result<()> {
        let mut writer = self.writer.lock();
        let term = Term::from_field_text(self.id_field, id.as_str());
        writer.delete_term(term);
        Self::commit_writer(&mut writer, &self.reader)
    }

    /// Search all fields, returning deduplicated ranked results.
    pub fn search(&self, query: &str, limit: usize) -> axil_core::Result<Vec<(RecordId, f32)>> {
        self.search_internal(query, None, limit)
    }

    /// Fuzzy search with Levenshtein distance tolerance.
    pub fn search_fuzzy(
        &self,
        query: &str,
        distance: u8,
        limit: usize,
    ) -> axil_core::Result<Vec<(RecordId, f32)>> {
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        Self::validate_query(query)?;

        let searcher = self.reader.searcher();
        let mut query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        query_parser.set_field_fuzzy(self.body_field, false, distance.min(2), true);
        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| AxilError::InvalidQuery(format!("FTS fuzzy query parse error: {e}")))?;

        let boost_factor = 2;
        let top_docs = self.execute_search(&searcher, &*parsed, limit * boost_factor)?;
        let mut results = self.collect_deduped(&searcher, &top_docs, limit, true)?;
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Search within a specific field only.
    pub fn search_in_field(
        &self,
        query: &str,
        field: &str,
        limit: usize,
    ) -> axil_core::Result<Vec<(RecordId, f32)>> {
        self.search_internal(query, Some(field), limit)
    }

    /// Search and return results with highlighted snippets.
    pub fn search_with_snippets(
        &self,
        query: &str,
        limit: usize,
        max_snippet_chars: usize,
    ) -> axil_core::Result<Vec<(RecordId, f32, String)>> {
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        Self::validate_query(query)?;

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| AxilError::InvalidQuery(format!("FTS query parse error: {e}")))?;

        let mut snippet_gen = SnippetGenerator::create(&searcher, &*parsed, self.body_field)
            .map_err(|e| AxilError::plugin(format!("failed to create snippet generator: {e}")))?;
        snippet_gen.set_max_num_chars(max_snippet_chars);

        let boost_factor = 2;
        let top_docs = self.execute_search(&searcher, &*parsed, limit * boost_factor)?;

        let mut results = Vec::with_capacity(limit);
        let mut seen = std::collections::HashSet::new();
        for (score, doc_address) in &top_docs {
            let doc: TantivyDocument = searcher
                .doc(*doc_address)
                .map_err(|e| AxilError::plugin(format!("failed to retrieve FTS document: {e}")))?;
            if let Some(rid) = self.extract_deduped_id(&doc, &mut seen)? {
                let mut normalized = score / (score + 1.0);
                // Apply field boosting (consistent with search_internal).
                if let Some(field_val) = doc.get_first(self.field_name_field) {
                    if let Some(fn_str) = field_val.as_str() {
                        if BOOSTED_FIELDS.contains(&fn_str) {
                            normalized = (normalized * BOOST_FACTOR).min(0.999);
                        }
                    }
                }
                let snippet = snippet_gen.snippet_from_doc(&doc);
                results.push((rid, normalized, snippet.to_html()));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    fn search_internal(
        &self,
        query: &str,
        field_scope: Option<&str>,
        limit: usize,
    ) -> axil_core::Result<Vec<(RecordId, f32)>> {
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        Self::validate_query(query)?;

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        let body_query = query_parser
            .parse_query(query)
            .map_err(|e| AxilError::InvalidQuery(format!("FTS query parse error: {e}")))?;

        let effective_query: Box<dyn tantivy::query::Query> = if let Some(field) = field_scope {
            let field_term = Term::from_field_text(self.field_name_field, field);
            let field_query = TermQuery::new(field_term, IndexRecordOption::Basic);
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, body_query),
                (Occur::Must, Box::new(field_query)),
            ]))
        } else {
            body_query
        };

        let apply_boost = field_scope.is_none();
        // When boosting, fetch more candidates to ensure boosted-field docs
        // aren't cut off by BM25 ordering before boost is applied.
        let boost_factor = if apply_boost { 2 } else { 1 };
        let top_docs = self.execute_search(&searcher, &*effective_query, limit * boost_factor)?;
        let mut results = self.collect_deduped(&searcher, &top_docs, limit, apply_boost)?;

        if apply_boost {
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);
        }

        Ok(results)
    }

    // ── Shared helpers ─────────────────────────────────────────────

    fn validate_query(query: &str) -> axil_core::Result<()> {
        if query.len() > MAX_QUERY_LEN {
            return Err(AxilError::InvalidQuery(format!(
                "FTS query exceeds maximum length of {MAX_QUERY_LEN} bytes"
            )));
        }
        // Case-insensitive check — Tantivy's QueryParser accepts field names
        // in any case (ID:, Id:, id: all work).
        let lower = query.to_ascii_lowercase();
        if lower.contains("id:") || lower.contains("doc_key:") || lower.contains("field_name:") {
            return Err(AxilError::InvalidQuery(
                "FTS queries must not target internal fields".into(),
            ));
        }
        Ok(())
    }

    fn execute_search(
        &self,
        searcher: &tantivy::Searcher,
        query: &dyn tantivy::query::Query,
        limit: usize,
    ) -> axil_core::Result<Vec<(f32, tantivy::DocAddress)>> {
        let fetch_limit = limit.saturating_mul(DEDUP_OVER_FETCH);
        searcher
            .search(query, &TopDocs::with_limit(fetch_limit))
            .map_err(|e| AxilError::plugin(format!("FTS search failed: {e}")))
    }

    /// Extract a deduplicated RecordId from a tantivy document.
    /// Returns `None` if the record was already seen.
    fn extract_deduped_id(
        &self,
        doc: &TantivyDocument,
        seen: &mut std::collections::HashSet<String>,
    ) -> axil_core::Result<Option<RecordId>> {
        if let Some(id_value) = doc.get_first(self.id_field) {
            if let Some(id_str) = Value::as_str(&id_value) {
                let id_string = id_str.to_string();
                if seen.insert(id_string.clone()) {
                    let rid = RecordId::from_string(&id_string).map_err(|e| {
                        AxilError::plugin(format!("invalid record ID in FTS index: {e}"))
                    })?;
                    return Ok(Some(rid));
                }
            }
        }
        Ok(None)
    }

    /// Collect deduplicated, normalized results from tantivy search output.
    /// When `apply_boost` is true, high-value field names get a score multiplier.
    /// Does NOT early-exit when boosting to ensure correct ordering after re-sort.
    fn collect_deduped(
        &self,
        searcher: &tantivy::Searcher,
        top_docs: &[(f32, tantivy::DocAddress)],
        limit: usize,
        apply_boost: bool,
    ) -> axil_core::Result<Vec<(RecordId, f32)>> {
        let mut results = Vec::with_capacity(limit);
        let mut seen = std::collections::HashSet::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(*doc_address)
                .map_err(|e| AxilError::plugin(format!("failed to retrieve FTS document: {e}")))?;
            if let Some(rid) = self.extract_deduped_id(&doc, &mut seen)? {
                let mut normalized = score / (score + 1.0);
                if apply_boost {
                    if let Some(fn_value) = doc.get_first(self.field_name_field) {
                        if let Some(fn_str) = Value::as_str(&fn_value) {
                            if BOOSTED_FIELDS.contains(&fn_str) {
                                normalized = (normalized * BOOST_FACTOR).min(0.999);
                            }
                        }
                    }
                }
                results.push((rid, normalized));
                // Only early-exit when not boosting (boosting requires full
                // collection before re-sort to avoid missing higher-scored items).
                if !apply_boost && results.len() >= limit {
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Return all unique record IDs currently in the FTS index.
    pub fn all_indexed_ids(&self) -> axil_core::Result<Vec<RecordId>> {
        let searcher = self.reader.searcher();
        let mut ids = std::collections::HashSet::new();
        for segment_reader in searcher.segment_readers() {
            if let Ok(inv_index) = segment_reader.inverted_index(self.id_field) {
                let Ok(mut terms) = inv_index.terms().stream() else {
                    continue;
                };
                while terms.advance() {
                    if let Ok(s) = std::str::from_utf8(terms.key()) {
                        ids.insert(
                            RecordId::from_string(s).unwrap_or_else(|_| RecordId(s.to_string())),
                        );
                    }
                }
            }
        }
        Ok(ids.into_iter().collect())
    }

    /// Commit any pending writes to disk.
    pub fn commit(&self) -> axil_core::Result<()> {
        let mut writer = self.writer.lock();
        Self::commit_writer(&mut writer, &self.reader)
    }

    fn commit_writer(writer: &mut IndexWriter, reader: &IndexReader) -> axil_core::Result<()> {
        writer
            .commit()
            .map_err(|e| AxilError::plugin(format!("failed to commit FTS index: {e}")))?;
        reader
            .reload()
            .map_err(|e| AxilError::plugin(format!("failed to reload FTS reader: {e}")))?;
        Ok(())
    }
}

/// Truncate a string to at most `max_bytes`, respecting UTF-8 boundaries.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_index() -> (FtsIndex, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let idx_path = dir.path().join("fts");
        let index = FtsIndex::new(&idx_path).unwrap();
        (index, dir)
    }

    #[test]
    fn add_and_search() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert(
            "summary".to_string(),
            "Fixed the authentication timeout bug in login flow".to_string(),
        );
        index.add_document(&id, &fields).unwrap();

        let results = index.search("authentication", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 > 0.0);
        assert!(results[0].1 < 1.0);
    }

    #[test]
    fn search_no_results() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("summary".to_string(), "hello world".to_string());
        index.add_document(&id, &fields).unwrap();
        assert!(index.search("nonexistent", 10).unwrap().is_empty());
    }

    #[test]
    fn search_empty_query() {
        let (index, _dir) = temp_index();
        assert!(index.search("", 10).unwrap().is_empty());
    }

    #[test]
    fn search_empty_index() {
        let (index, _dir) = temp_index();
        assert!(index.search("anything", 10).unwrap().is_empty());
    }

    #[test]
    fn remove_document() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("summary".to_string(), "important text to find".to_string());
        index.add_document(&id, &fields).unwrap();
        assert_eq!(index.search("important", 10).unwrap().len(), 1);

        index.remove_document(&id).unwrap();
        assert!(index.search("important", 10).unwrap().is_empty());
    }

    #[test]
    fn multi_field_search() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Rust programming language".to_string());
        fields.insert(
            "body".to_string(),
            "Systems programming with memory safety".to_string(),
        );
        index.add_document(&id, &fields).unwrap();

        assert_eq!(index.search("Rust", 10).unwrap().len(), 1);
        assert_eq!(index.search("memory safety", 10).unwrap().len(), 1);
    }

    #[test]
    fn multiple_documents_ranked() {
        let (index, _dir) = temp_index();

        let id1 = RecordId::new();
        let mut f1 = HashMap::new();
        f1.insert("text".to_string(), "rust rust rust programming".to_string());
        index.add_document(&id1, &f1).unwrap();

        let id2 = RecordId::new();
        let mut f2 = HashMap::new();
        f2.insert("text".to_string(), "rust programming language".to_string());
        index.add_document(&id2, &f2).unwrap();

        let results = index.search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn index_single_field() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        index
            .index_field(&id, "summary", "database indexing performance")
            .unwrap();
        assert_eq!(index.search("indexing", 10).unwrap().len(), 1);
    }

    #[test]
    fn deduplicates_multi_field_hits() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Learn rust today".to_string());
        fields.insert(
            "desc".to_string(),
            "A rust tutorial for beginners".to_string(),
        );
        index.add_document(&id, &fields).unwrap();
        assert_eq!(index.search("rust", 10).unwrap().len(), 1);
    }

    #[test]
    fn index_field_preserves_other_fields() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        index.index_field(&id, "title", "Rust guide").unwrap();
        index.index_field(&id, "body", "memory safety").unwrap();

        assert_eq!(index.search("Rust", 10).unwrap().len(), 1);
        assert_eq!(index.search("memory", 10).unwrap().len(), 1);
    }

    // ── Field-scoped search ────────────────────────────────────────

    #[test]
    fn field_scoped_search_basic() {
        let (index, _dir) = temp_index();
        let id1 = RecordId::new();
        let mut f1 = HashMap::new();
        f1.insert("title".to_string(), "auth timeout fix".to_string());
        f1.insert("notes".to_string(), "deployed to prod".to_string());
        index.add_document(&id1, &f1).unwrap();

        let id2 = RecordId::new();
        let mut f2 = HashMap::new();
        f2.insert("title".to_string(), "login redesign".to_string());
        f2.insert("notes".to_string(), "auth module touched".to_string());
        index.add_document(&id2, &f2).unwrap();

        // "auth" in title → only id1
        let results = index.search_in_field("auth", "title", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);

        // "auth" in notes → only id2
        let results = index.search_in_field("auth", "notes", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id2);

        // "auth" globally → both
        assert_eq!(index.search("auth", 10).unwrap().len(), 2);
    }

    #[test]
    fn field_scoped_no_match_in_target() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "database optimization".to_string());
        fields.insert("body".to_string(), "authentication improved".to_string());
        index.add_document(&id, &fields).unwrap();

        assert!(index
            .search_in_field("authentication", "title", 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn field_scoped_nonexistent_field() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "auth guide".to_string());
        index.add_document(&id, &fields).unwrap();

        assert!(index
            .search_in_field("auth", "nonexistent", 10)
            .unwrap()
            .is_empty());
    }

    // ── Phrase queries ─────────────────────────────────────────────

    #[test]
    fn phrase_query_exact_match() {
        let (index, _dir) = temp_index();
        let id1 = RecordId::new();
        let mut f1 = HashMap::new();
        f1.insert(
            "text".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        );
        index.add_document(&id1, &f1).unwrap();

        let id2 = RecordId::new();
        let mut f2 = HashMap::new();
        f2.insert(
            "text".to_string(),
            "brown and quick are different words".to_string(),
        );
        index.add_document(&id2, &f2).unwrap();

        // Phrase query: exact sequence
        let results = index.search("\"quick brown fox\"", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn phrase_query_slop() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("text".to_string(), "the quick brown fox".to_string());
        index.add_document(&id, &fields).unwrap();

        // Slop of 1: allows one word gap
        let results = index.search("\"quick fox\"~1", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    // ── Fuzzy search ───────────────────────────────────────────────

    #[test]
    fn fuzzy_search_with_typo() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert(
            "text".to_string(),
            "authentication timeout error".to_string(),
        );
        index.add_document(&id, &fields).unwrap();

        // Typo: "authenticaton" (missing 'i')
        let results = index.search_fuzzy("authenticaton", 1, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }

    #[test]
    fn fuzzy_search_exact_still_works() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert(
            "text".to_string(),
            "database indexing performance".to_string(),
        );
        index.add_document(&id, &fields).unwrap();

        let results = index.search_fuzzy("database", 1, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn fuzzy_search_too_far_no_match() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("text".to_string(), "authentication".to_string());
        index.add_document(&id, &fields).unwrap();

        // "xxxxx" is too far from "authentication" even with distance 2
        let results = index.search_fuzzy("xxxxx", 2, 10).unwrap();
        assert!(results.is_empty());
    }

    // ── Snippet generation ─────────────────────────────────────────

    #[test]
    fn snippet_highlights_matched_terms() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert(
            "text".to_string(),
            "Fixed the authentication timeout bug in the login flow".to_string(),
        );
        index.add_document(&id, &fields).unwrap();

        let results = index
            .search_with_snippets("authentication", 10, 150)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        // Snippet should contain <b> tags around matched term
        assert!(
            results[0].2.contains("<b>"),
            "snippet should highlight matched term: {}",
            results[0].2
        );
    }

    #[test]
    fn snippet_empty_for_no_match() {
        let (index, _dir) = temp_index();
        let id = RecordId::new();
        let mut fields = HashMap::new();
        fields.insert("text".to_string(), "hello world".to_string());
        index.add_document(&id, &fields).unwrap();

        let results = index.search_with_snippets("nonexistent", 10, 150).unwrap();
        assert!(results.is_empty());
    }

    // ── Field boosting ─────────────────────────────────────────────

    #[test]
    fn boosted_field_ranks_higher() {
        let (index, _dir) = temp_index();

        // Record 1: "rust" in title (boosted field)
        let id1 = RecordId::new();
        let mut f1 = HashMap::new();
        f1.insert("title".to_string(), "rust programming".to_string());
        index.add_document(&id1, &f1).unwrap();

        // Record 2: "rust" in notes (non-boosted field)
        let id2 = RecordId::new();
        let mut f2 = HashMap::new();
        f2.insert("notes".to_string(), "rust programming".to_string());
        index.add_document(&id2, &f2).unwrap();

        let results = index.search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        // Title match should score higher due to boost
        assert_eq!(results[0].0, id1, "title match should rank first");
        assert!(
            results[0].1 > results[1].1,
            "boosted field should have higher score"
        );
    }
}
