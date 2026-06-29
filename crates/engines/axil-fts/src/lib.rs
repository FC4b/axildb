pub mod code_tokenizer;
pub mod index;

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use axil_core::plugin::{Capability, Engine, SearchIndex};
use axil_core::record::{Record, RecordId};
use axil_core::{companion_path, AxilBuilder, Result};

use crate::index::FtsIndex;

/// Full-text search plugin backed by tantivy.
///
/// Stores a tantivy index directory alongside the `.axil` database file
/// at `<db_path>.fts/`. Supports both explicit indexing via `index_text()`
/// and automatic indexing of string fields on record insert.
pub struct FtsEngine {
    index: FtsIndex,
}

impl FtsEngine {
    /// Open or create the FTS index for the database at `db_path`.
    pub fn open(db_path: &Path) -> Result<Self> {
        let fts_path = companion_path(db_path, ".fts");
        let index = FtsIndex::new(&fts_path)?;
        Ok(Self { index })
    }

    /// True if the index was rebuilt during open (schema migration).
    /// The caller should reindex all records from storage.
    pub fn needs_reindex(&self) -> bool {
        self.index.needs_reindex()
    }

    /// Maximum number of string fields to auto-index per record.
    const MAX_AUTO_INDEX_FIELDS: usize = 32;

    /// Extract string fields from record data for auto-indexing.
    ///
    /// Caps at [`Self::MAX_AUTO_INDEX_FIELDS`] fields to prevent excessive
    /// tantivy document creation for records with many string fields.
    fn extract_text_fields(data: &Value) -> std::collections::HashMap<String, String> {
        let mut fields = std::collections::HashMap::new();
        if let Some(obj) = data.as_object() {
            for (key, value) in obj {
                if fields.len() >= Self::MAX_AUTO_INDEX_FIELDS {
                    break;
                }
                let text = Self::extract_text_from_value(value);
                if !text.is_empty() {
                    fields.insert(key.clone(), text);
                }
            }
        }
        fields
    }

    /// Recursively extract text from a JSON value.
    fn extract_text_from_value(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Array(arr) => {
                let parts: Vec<String> = arr
                    .iter()
                    .filter_map(|v| {
                        let t = Self::extract_text_from_value(v);
                        if t.is_empty() {
                            None
                        } else {
                            Some(t)
                        }
                    })
                    .collect();
                parts.join(" ")
            }
            Value::Object(obj) => {
                let parts: Vec<String> = obj
                    .values()
                    .filter_map(|v| {
                        let t = Self::extract_text_from_value(v);
                        if t.is_empty() {
                            None
                        } else {
                            Some(t)
                        }
                    })
                    .collect();
                parts.join(" ")
            }
            _ => String::new(),
        }
    }
}

impl Engine for FtsEngine {
    fn name(&self) -> &str {
        "fts"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::FullTextSearch]
    }

    /// Auto-index all string fields from the record's JSON data.
    fn on_record_insert(&self, record: &Record) -> Result<()> {
        let fields = Self::extract_text_fields(&record.data);
        if !fields.is_empty() {
            self.index.add_document(&record.id, &fields)?;
        }
        Ok(())
    }

    /// Re-index all string fields after a record update.
    fn on_record_update(&self, record: &Record) -> Result<()> {
        let fields = Self::extract_text_fields(&record.data);
        if fields.is_empty() {
            // No string fields left — remove stale docs.
            self.index.remove_document(&record.id)?;
        } else {
            // add_document does delete-then-add internally.
            self.index.add_document(&record.id, &fields)?;
        }
        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        self.index.remove_document(id)
    }
}

impl SearchIndex for FtsEngine {
    fn index_text(&self, id: &RecordId, field: &str, text: &str) -> Result<()> {
        self.index.index_field(id, field, text)
    }

    fn would_index(&self, record: &Record) -> bool {
        // Mirror on_record_insert: a record with no extractable text fields
        // produces no document, so it must not be flagged as a missing doc.
        !Self::extract_text_fields(&record.data).is_empty()
    }

    /// Buffers every document with `add_document_deferred`, then commits once.
    fn index_records_batch(&self, records: &[Record]) -> Result<()> {
        for record in records {
            if record.table.starts_with('_') {
                continue;
            }
            let fields = Self::extract_text_fields(&record.data);
            if !fields.is_empty() {
                self.index.add_document_deferred(&record.id, &fields)?;
            }
        }
        self.index.commit()
    }

    /// Buffers every field with `index_field_deferred`, then commits once.
    fn index_field_batch(&self, field: &str, entries: &[(&RecordId, &str)]) -> Result<()> {
        for &(id, text) in entries {
            self.index.index_field_deferred(id, field, text)?;
        }
        self.index.commit()
    }

    fn search_text(&self, query: &str, limit: usize) -> Result<Vec<(RecordId, f32)>> {
        self.index.search(query, limit)
    }

    fn search_field(&self, query: &str, field: &str, limit: usize) -> Result<Vec<(RecordId, f32)>> {
        self.index.search_in_field(query, field, limit)
    }

    fn search_fuzzy(
        &self,
        query: &str,
        distance: u8,
        limit: usize,
    ) -> Result<Vec<(RecordId, f32)>> {
        self.index.search_fuzzy(query, distance, limit)
    }

    fn search_with_snippets(
        &self,
        query: &str,
        limit: usize,
        max_chars: usize,
    ) -> Result<Vec<(RecordId, f32, String)>> {
        self.index.search_with_snippets(query, limit, max_chars)
    }

    fn all_indexed_ids(&self) -> Result<Vec<RecordId>> {
        self.index.all_indexed_ids()
    }

    fn optimize(&self) -> Result<()> {
        self.index.commit()
    }
}

// ── Builder Extension ──────────────────────────────────────────────

/// Extension trait for `AxilBuilder` to enable FTS support.
pub trait AxilBuilderFtsExt {
    /// Enable full-text search with a companion `.fts/` directory.
    fn with_fts_engine(self) -> Result<Self>
    where
        Self: Sized;
}

impl AxilBuilderFtsExt for AxilBuilder {
    fn with_fts_engine(mut self) -> Result<Self> {
        let plugin = FtsEngine::open(self.path())?;
        if plugin.needs_reindex() {
            self.needs_fts_reindex = true;
        }
        let arc: Arc<dyn SearchIndex> = Arc::new(plugin);
        Ok(self.with_fts_index(arc))
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Check if an FTS store exists for the given database path.
pub fn has_fts_store(db_path: &Path) -> bool {
    companion_path(db_path, ".fts").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_fts_engine() -> (FtsEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let plugin = FtsEngine::open(&db_path).unwrap();
        (plugin, dir)
    }

    #[test]
    fn auto_index_on_insert() {
        let (plugin, _dir) = temp_fts_engine();
        let record = Record::new(
            "sessions",
            json!({"summary": "Fixed authentication bug", "count": 42}),
        );
        plugin.on_record_insert(&record).unwrap();

        // Should find by string field content.
        let results = plugin.search_text("authentication", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, record.id);

        // Numeric fields should not be indexed.
        let results = plugin.search_text("42", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn delete_removes_from_index() {
        let (plugin, _dir) = temp_fts_engine();
        let record = Record::new("sessions", json!({"summary": "Important data"}));
        plugin.on_record_insert(&record).unwrap();
        assert_eq!(plugin.search_text("important", 10).unwrap().len(), 1);

        plugin.on_record_delete(&record.id).unwrap();
        assert!(plugin.search_text("important", 10).unwrap().is_empty());
    }

    #[test]
    fn explicit_index_text() {
        let (plugin, _dir) = temp_fts_engine();
        let id = RecordId::new();
        plugin
            .index_text(&id, "custom", "explicitly indexed content")
            .unwrap();

        let results = plugin.search_text("explicitly", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }

    #[test]
    fn has_fts_store_detection() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        assert!(!has_fts_store(&db_path));

        let _plugin = FtsEngine::open(&db_path).unwrap();
        assert!(has_fts_store(&db_path));
    }
}
