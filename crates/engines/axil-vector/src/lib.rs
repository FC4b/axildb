pub mod binary;
pub mod download;
pub mod embed;
pub mod hnsw;
pub mod mmap;
pub mod models;
pub mod quantize;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use redb::{Database, ReadOnlyDatabase, ReadableDatabase, ReadableTable, TableDefinition};

use axil_core::error::AxilError;
use axil_core::plugin::{Capability, Plugin, TextEmbedder, VectorIndex};
use axil_core::record::{Record, RecordId};

use axil_core::db::AxilBuilder;

use crate::embed::Embedder;
use crate::hnsw::HnswIndex;
use crate::models::EmbeddingModel;

/// redb table: record_id → raw f32 bytes.
const VECTORS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("vectors");

/// redb table: meta key → value bytes.
const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

/// Wrap any error into AxilError::Plugin, preserving the source chain.
fn plugin_err(e: impl std::error::Error + Send + Sync + 'static) -> AxilError {
    AxilError::Plugin(Box::new(e))
}

/// Configuration for the vector plugin.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Vector dimensions.
    pub dimensions: usize,
    /// Fields to auto-embed on insert (requires embedder).
    pub auto_embed_fields: Vec<String>,
}

/// Vector search plugin for Axil.
///
/// Provides HNSW-based approximate nearest neighbor search with optional
/// ONNX-based text embedding. Vectors are persisted in a separate redb
/// file alongside the main database.
pub struct VectorPlugin {
    config: VectorConfig,
    index: RwLock<HnswIndex>,
    vector_db: Database,
    embedder: Option<Embedder>,
}

impl VectorPlugin {
    /// Open (or create) a vector store alongside the given database path.
    ///
    /// The vector data is stored at `<db_path>.vec`.
    pub fn open(db_path: impl AsRef<Path>, dimensions: usize) -> axil_core::Result<Self> {
        Self::open_with_config(
            db_path,
            VectorConfig {
                dimensions,
                auto_embed_fields: Vec::new(),
            },
        )
    }

    /// Open with full configuration.
    pub fn open_with_config(
        db_path: impl AsRef<Path>,
        config: VectorConfig,
    ) -> axil_core::Result<Self> {
        let vec_path = vector_db_path(db_path.as_ref());
        let vector_db = Database::create(&vec_path).map_err(plugin_err)?;

        // Ensure tables exist + validate/store dimensions in one write txn.
        let txn = vector_db.begin_write().map_err(plugin_err)?;
        {
            let _ = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
            let meta = txn.open_table(META_TABLE).map_err(plugin_err)?;

            // Check stored dimensions for model mismatch.
            if let Some(guard) = meta.get("dimensions").map_err(plugin_err)? {
                let stored_str = std::str::from_utf8(guard.value()).map_err(plugin_err)?;
                let stored: usize = stored_str.parse().map_err(plugin_err)?;
                if stored != config.dimensions {
                    return Err(AxilError::plugin(format!(
                        "dimension mismatch: vector store was created with {stored} dimensions, \
                         but {} requested. Use `axil reembed` to re-index with new dimensions.",
                        config.dimensions
                    )));
                }
            }
            drop(meta);

            let mut meta_w = txn.open_table(META_TABLE).map_err(plugin_err)?;
            meta_w
                .insert("dimensions", config.dimensions.to_string().as_bytes())
                .map_err(plugin_err)?;
        }
        txn.commit().map_err(plugin_err)?;

        // Load existing vectors from storage.
        let vectors = load_all_vectors(&vector_db, config.dimensions)?;
        let index = HnswIndex::from_vectors(config.dimensions, vectors);

        Ok(Self {
            config,
            index: RwLock::new(index),
            vector_db,
            embedder: None,
        })
    }

    /// Attach an embedder for text-to-vector conversion.
    pub fn with_embedder(mut self, embedder: Embedder) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Attach an embedder by model type.
    ///
    /// If the model files are not available locally, they are downloaded
    /// automatically from HuggingFace on first use.
    pub fn with_model(self, model: EmbeddingModel) -> Result<Self, String> {
        if !matches!(model, EmbeddingModel::Custom { .. })
            && !crate::download::is_model_available(&model)
        {
            eprintln!(
                "Model {} not found locally — downloading ({})...",
                model.name(),
                model.approx_size(),
            );
            crate::download::download_model(&model)?;
        }
        let embedder = Embedder::new(model)?;
        Ok(self.with_embedder(embedder))
    }

    /// Number of vectors currently indexed.
    pub fn vector_count(&self) -> usize {
        self.index.read().len()
    }

    /// Configured dimensions.
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }
}

impl Plugin for VectorPlugin {
    fn name(&self) -> &str {
        "vector"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::VectorSearch]
    }

    fn on_record_insert(&self, record: &Record) -> axil_core::Result<()> {
        if self.config.auto_embed_fields.is_empty() || self.embedder.is_none() {
            return Ok(());
        }

        let embedder = self.embedder.as_ref().ok_or_else(|| {
            axil_core::AxilError::plugin("auto-embed requires an embedder but none is configured")
        })?;

        // Concatenate text from all configured fields into one string,
        // then produce a single embedding per record. This avoids
        // overwriting earlier fields when multiple are configured.
        let mut parts = Vec::new();
        for field in &self.config.auto_embed_fields {
            if let Some(text) = record.data.get(field).and_then(|v| v.as_str()) {
                parts.push(text);
            }
        }

        if parts.is_empty() {
            return Ok(());
        }

        let combined = parts.join(" ");
        let vector = embedder
            .embed(&combined)
            .map_err(|e| AxilError::plugin(format!("auto-embed failed: {e}")))?;

        // Persist to disk first so a crash can't leave the in-memory index
        // ahead of storage.
        persist_vector(&self.vector_db, &record.id, &vector)?;
        self.index
            .write()
            .add(record.id.clone(), vector.clone())
            .map_err(AxilError::plugin)?;

        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> axil_core::Result<()> {
        self.index.write().remove(id);
        delete_vector(&self.vector_db, id)
    }
}

impl VectorIndex for VectorPlugin {
    fn add(&self, id: RecordId, vector: &[f32]) -> axil_core::Result<()> {
        // Persist to disk first so a crash can't leave the in-memory
        // index ahead of storage.
        persist_vector(&self.vector_db, &id, vector)?;
        self.index
            .write()
            .add(id, vector.to_vec())
            .map_err(AxilError::plugin)?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> axil_core::Result<Vec<(RecordId, f32)>> {
        // Fast path: read lock when the HNSW graph is current.
        {
            let idx = self.index.read();
            if !idx.needs_rebuild() {
                return idx.search_clean(query, top_k).map_err(AxilError::plugin);
            }
        }
        // Slow path: write lock to rebuild, then search.
        let mut idx = self.index.write();
        idx.rebuild_if_needed();
        idx.search_clean(query, top_k).map_err(AxilError::plugin)
    }

    fn count(&self) -> usize {
        self.vector_count()
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn deleted_count(&self) -> usize {
        self.index.read().deletes_since_rebuild()
    }

    fn all_ids(&self) -> axil_core::Result<Vec<RecordId>> {
        Ok(self.index.read().vectors().keys().cloned().collect())
    }

    fn get_vector(&self, id: &RecordId) -> axil_core::Result<Option<Vec<f32>>> {
        Ok(self.index.read().vectors().get(id).cloned())
    }

    fn rebuild(&self) -> axil_core::Result<usize> {
        let mut idx = self.index.write();
        idx.rebuild_if_needed();
        Ok(idx.len())
    }
}

impl TextEmbedder for VectorPlugin {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        match &self.embedder {
            Some(e) => e.embed(text).map_err(AxilError::plugin),
            None => Err(AxilError::plugin(
                "no embedder configured — use with_embedder() or with_model()",
            )),
        }
    }

    fn embed_batch(&self, texts: &[&str]) -> axil_core::Result<Vec<Vec<f32>>> {
        match &self.embedder {
            #[cfg(feature = "embed")]
            Some(e) => e.embed_batch_impl(texts).map_err(AxilError::plugin),
            #[cfg(not(feature = "embed"))]
            Some(e) => texts
                .iter()
                .map(|t| e.embed(t).map_err(AxilError::plugin))
                .collect(),
            None => Err(AxilError::plugin(
                "no embedder configured — use with_embedder() or with_model()",
            )),
        }
    }
}

// ── Persistence helpers ─────────────────────────────────────────────

/// Derive the vector database path from the main database path.
pub fn vector_db_path(main_path: &Path) -> PathBuf {
    let mut p = main_path.as_os_str().to_owned();
    p.push(".vec");
    PathBuf::from(p)
}

/// Serialize a vector as raw little-endian f32 bytes.
fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Deserialize raw bytes back to a vector.
fn bytes_to_vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Persist a single vector to the vector database.
fn persist_vector(db: &Database, id: &RecordId, vector: &[f32]) -> axil_core::Result<()> {
    let bytes = vector_to_bytes(vector);
    let txn = db.begin_write().map_err(plugin_err)?;
    {
        let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        table
            .insert(id.as_str(), bytes.as_slice())
            .map_err(plugin_err)?;
    }
    txn.commit().map_err(plugin_err)?;
    Ok(())
}

/// Delete a vector from the vector database.
fn delete_vector(db: &Database, id: &RecordId) -> axil_core::Result<()> {
    let txn = db.begin_write().map_err(plugin_err)?;
    {
        let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        table.remove(id.as_str()).map_err(plugin_err)?;
    }
    txn.commit().map_err(plugin_err)?;
    Ok(())
}

/// Read stored dimensions from a vector database without fully opening the plugin.
///
/// Returns `Ok(None)` if the `.vec` file doesn't exist. Returns an error if the
/// file exists but is corrupt or unreadable (so callers can distinguish "missing"
/// from "broken").
pub fn read_stored_dimensions(db_path: impl AsRef<Path>) -> axil_core::Result<Option<usize>> {
    let vec_path = vector_db_path(db_path.as_ref());
    if !vec_path.exists() {
        return Ok(None);
    }
    // Read-only open: never creates files, correct semantic for a probe.
    let db = ReadOnlyDatabase::open(&vec_path).map_err(plugin_err)?;
    let txn = db.begin_read().map_err(plugin_err)?;
    let meta = match txn.open_table(META_TABLE) {
        Ok(t) => t,
        // .vec exists but has no meta table — corrupt, not missing.
        Err(redb::TableError::TableDoesNotExist(_)) => {
            return Err(AxilError::plugin(
                "vector store exists but has no metadata table — file may be corrupt",
            ));
        }
        Err(e) => return Err(plugin_err(e)),
    };
    match meta.get("dimensions").map_err(plugin_err)? {
        // .vec exists with meta table but no dimensions key — corrupt.
        None => Err(AxilError::plugin(
            "vector store exists but has no dimensions metadata — file may be corrupt",
        )),
        Some(guard) => {
            let s = std::str::from_utf8(guard.value()).map_err(plugin_err)?;
            let dims: usize = s
                .parse()
                .map_err(|e: std::num::ParseIntError| plugin_err(e))?;
            Ok(Some(dims))
        }
    }
}

// ── AxilBuilder extension ──────────────────────────────────────────

/// Extension trait that adds vector plugin support to [`AxilBuilder`].
///
/// Lets users write `Axil::open(path).with_vector(384)?.build()?`
/// without manually constructing a `VectorPlugin`.
pub trait AxilBuilderVectorExt {
    /// Enable vector search with the given dimensions.
    ///
    /// Creates a `VectorPlugin` internally using the builder's path.
    fn with_vector(self, dims: usize) -> axil_core::Result<Self>
    where
        Self: Sized;

    /// Enable vector search, auto-detecting dimensions from an existing vector store.
    ///
    /// Returns an error if no vector store exists at the expected path.
    fn with_vector_auto(self) -> axil_core::Result<Self>
    where
        Self: Sized;

    /// Enable vector search with an embedding model.
    ///
    /// Creates a `VectorPlugin` with the model's native dimensions and attaches
    /// the embedder for `embed_field()`, `embed_text()`, and `similar_to()`.
    fn with_embedder_model(self, model: EmbeddingModel) -> axil_core::Result<Self>
    where
        Self: Sized;
}

impl AxilBuilderVectorExt for AxilBuilder {
    fn with_vector(self, dims: usize) -> axil_core::Result<Self> {
        let path = self.path().to_path_buf();
        let plugin = VectorPlugin::open(&path, dims)?;
        Ok(self.with_vector_and_embedder(plugin))
    }

    fn with_vector_auto(self) -> axil_core::Result<Self> {
        let path = self.path().to_path_buf();
        let dims = read_stored_dimensions(&path)?.ok_or_else(|| {
            AxilError::plugin(
                "no vector store found — create one first with with_vector(dims) or \
                 `axil create --vector <dims>`",
            )
        })?;
        self.with_vector(dims)
    }

    fn with_embedder_model(self, model: EmbeddingModel) -> axil_core::Result<Self> {
        let path = self.path().to_path_buf();
        let dims = model.dimensions();
        let plugin = VectorPlugin::open(&path, dims)?;
        let plugin = plugin.with_model(model).map_err(AxilError::plugin)?;
        Ok(self.with_vector_and_embedder(plugin))
    }
}

/// Load all persisted vectors into a HashMap.
///
/// Validates each RecordId and skips vectors with wrong dimensions.
fn load_all_vectors(
    db: &Database,
    expected_dims: usize,
) -> axil_core::Result<HashMap<RecordId, Vec<f32>>> {
    let txn = db.begin_read().map_err(plugin_err)?;
    let table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;

    let mut vectors = HashMap::new();
    let iter = table.iter().map_err(plugin_err)?;

    for entry in iter {
        let entry: (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
            entry.map_err(plugin_err)?;
        let id_str: &str = entry.0.value();
        let bytes: &[u8] = entry.1.value();

        let id = match RecordId::from_string(id_str) {
            Ok(id) => id,
            Err(_) => continue,
        };

        let vector = bytes_to_vector(bytes);
        if vector.len() != expected_dims {
            continue;
        }

        vectors.insert(id, vector);
    }

    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_plugin(dims: usize) -> (VectorPlugin, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let plugin = VectorPlugin::open(&path, dims).unwrap();
        (plugin, dir)
    }

    #[test]
    fn add_and_search() {
        let (plugin, _dir) = temp_plugin(3);
        let id1 = RecordId::new();
        let id2 = RecordId::new();

        plugin.add(id1.clone(), &[1.0, 0.0, 0.0]).unwrap();
        plugin.add(id2.clone(), &[0.0, 1.0, 0.0]).unwrap();

        let results = plugin.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist.axil");
        let id = RecordId::new();

        {
            let plugin = VectorPlugin::open(&path, 3).unwrap();
            plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
            assert_eq!(plugin.vector_count(), 1);
        }

        {
            let plugin = VectorPlugin::open(&path, 3).unwrap();
            assert_eq!(plugin.vector_count(), 1);
            let results = plugin.search(&[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, id);
        }
    }

    #[test]
    fn delete_removes_from_index_and_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("del.axil");
        let id = RecordId::new();

        {
            let plugin = VectorPlugin::open(&path, 3).unwrap();
            plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
            plugin.on_record_delete(&id).unwrap();
            assert_eq!(plugin.vector_count(), 0);
        }

        {
            let plugin = VectorPlugin::open(&path, 3).unwrap();
            assert_eq!(plugin.vector_count(), 0);
        }
    }

    #[test]
    fn on_record_insert_without_embedder_is_noop() {
        let (plugin, _dir) = temp_plugin(3);
        let record = Record::new("test", json!({"summary": "hello"}));
        plugin.on_record_insert(&record).unwrap();
        assert_eq!(plugin.vector_count(), 0);
    }

    #[test]
    fn vector_db_path_derivation() {
        let p = vector_db_path(Path::new("/tmp/my.axil"));
        assert_eq!(p, PathBuf::from("/tmp/my.axil.vec"));
    }

    #[test]
    fn vector_byte_roundtrip() {
        let v = vec![1.0_f32, -2.5, 3.14, 0.0];
        let bytes = vector_to_bytes(&v);
        let v2 = bytes_to_vector(&bytes);
        assert_eq!(v, v2);
    }

    #[test]
    fn dimension_mismatch_on_reopen_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dims.axil");

        {
            let plugin = VectorPlugin::open(&path, 3).unwrap();
            let id = RecordId::new();
            plugin.add(id, &[1.0, 0.0, 0.0]).unwrap();
        }

        let result = VectorPlugin::open(&path, 768);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("dimension mismatch"), "error was: {err}");
    }
}
