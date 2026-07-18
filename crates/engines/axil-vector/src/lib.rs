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
use redb::{
    Database, ReadOnlyDatabase, ReadableDatabase, ReadableTable, ReadableTableMetadata,
    TableDefinition,
};

use axil_core::error::AxilError;
use axil_core::plugin::{Capability, Engine, TextEmbedder, VectorIndex};
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
pub struct VectorEngine {
    config: VectorConfig,
    index: RwLock<HnswIndex>,
    vector_db: Database,
    embedder: Option<Embedder>,
}

impl VectorEngine {
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

    /// Open (or create) a named vector space at exactly `vec_path`.
    ///
    /// Unlike [`VectorEngine::open`], the caller supplies the full companion
    /// path (`<db>.axil.vec.<name>`) rather than the base db path — named
    /// spaces derive their own file names. When the store already exists its
    /// persisted dimension governs; `dim = Some(d)` creates/validates a store
    /// with dimension `d`, while `dim = None` requires an existing store (a
    /// read path never conjures storage) and errors otherwise.
    pub fn open_space_at(vec_path: &Path, dim: Option<usize>) -> axil_core::Result<Self> {
        let existed = vec_path.exists();
        if !existed && dim.is_none() {
            return Err(AxilError::plugin(format!(
                "vector space {} does not exist",
                vec_path.display()
            )));
        }
        let vector_db = Database::create(vec_path).map_err(plugin_err)?;

        // An existing store's persisted dimension governs; probing it takes a
        // read txn only, so pure-read opens (similar/get_vector/listings) pay
        // no durable write commit. Only a fresh store writes: one txn creates
        // the tables and persists the requested dimension.
        let dimensions = if existed {
            let txn = vector_db.begin_read().map_err(plugin_err)?;
            let stored = match txn.open_table(META_TABLE) {
                Ok(meta) => match meta.get("dimensions").map_err(plugin_err)? {
                    Some(guard) => {
                        let s = std::str::from_utf8(guard.value()).map_err(plugin_err)?;
                        Some(s.parse::<usize>().map_err(plugin_err)?)
                    }
                    None => None,
                },
                Err(redb::TableError::TableDoesNotExist(_)) => None,
                Err(e) => return Err(plugin_err(e)),
            };
            match (dim, stored) {
                (Some(requested), Some(stored)) if requested != stored => {
                    return Err(AxilError::plugin(format!(
                        "dimension mismatch: vector space was created with {stored} \
                         dimensions, but {requested} requested"
                    )));
                }
                (_, Some(stored)) => stored,
                // File exists but carries no dimension (e.g. an interrupted
                // create): a write path may (re)initialize it, a read may not.
                (Some(requested), None) => {
                    Self::write_space_meta_txn(&vector_db, requested)?;
                    requested
                }
                (None, None) => {
                    return Err(AxilError::plugin(
                        "vector space has no stored dimension — file may be corrupt",
                    ));
                }
            }
        } else {
            let requested = dim.expect("checked above: fresh store requires a dimension");
            Self::write_space_meta_txn(&vector_db, requested)?;
            requested
        };

        let vectors = load_all_vectors(&vector_db, dimensions)?;
        let index = HnswIndex::from_vectors(dimensions, vectors);
        Ok(Self {
            config: VectorConfig {
                dimensions,
                auto_embed_fields: Vec::new(),
            },
            index: RwLock::new(index),
            vector_db,
            embedder: None,
        })
    }

    /// One write txn that ensures a space's tables exist and persists its
    /// dimension — the only durable write an `open_space_at` ever performs.
    fn write_space_meta_txn(vector_db: &Database, dimensions: usize) -> axil_core::Result<()> {
        let txn = vector_db.begin_write().map_err(plugin_err)?;
        {
            let _ = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
            let mut meta = txn.open_table(META_TABLE).map_err(plugin_err)?;
            meta.insert("dimensions", dimensions.to_string().as_bytes())
                .map_err(plugin_err)?;
        }
        txn.commit().map_err(plugin_err)
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

impl Engine for VectorEngine {
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

impl VectorIndex for VectorEngine {
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

    fn add_batch(&self, items: &[(RecordId, &[f32])]) -> axil_core::Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        // Persist the whole batch to disk first (one fsync) so a crash can't
        // leave the in-memory index ahead of storage, then add to the live index.
        persist_vectors_batch(&self.vector_db, items)?;
        let mut idx = self.index.write();
        for (id, vector) in items {
            idx.add(id.clone(), vector.to_vec())
                .map_err(AxilError::plugin)?;
        }
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> axil_core::Result<Vec<(RecordId, f32)>> {
        // The graph is always live and searchable, including immediately after
        // an incremental `add` and despite accumulated tombstones — so search
        // never needs the write lock. Compaction (tombstone reclaim) is the
        // background worker's job, off this hot path.
        self.index
            .read()
            .search_clean(query, top_k)
            .map_err(AxilError::plugin)
    }

    fn count(&self) -> usize {
        self.vector_count()
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn deleted_count(&self) -> usize {
        // Total reclaimable tombstones (removes AND re-adds), so the background
        // compactor fires on update-heavy workloads, not just deletes.
        self.index.read().tombstones()
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

impl TextEmbedder for VectorEngine {
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

/// Derive a named vector-space companion path: `<main_path>.vec.<space>`.
///
/// The default/unnamed space keeps the plain `<main_path>.vec` file from
/// [`vector_db_path`]; a named space appends `.<space>` so the two never
/// collide.
pub fn vector_space_db_path(main_path: &Path, space: &str) -> PathBuf {
    let mut p = main_path.as_os_str().to_owned();
    p.push(".vec.");
    p.push(space);
    PathBuf::from(p)
}

use axil_core::is_valid_space_name;

/// List persisted named-space names for the database at `main_path`.
///
/// Scans the database's directory for companion files of the form
/// `<file_name>.vec.<name>` where `<name>` matches `[a-z0-9_-]{1,32}`, without
/// opening any of them. The default `<file_name>.vec` store is excluded (no
/// trailing `.<name>`), as are unrelated companions (`.graph`, `.fts`, …).
pub fn list_vector_space_names(main_path: &Path) -> axil_core::Result<Vec<String>> {
    let Some(dir) = main_path.parent() else {
        return Ok(Vec::new());
    };
    let Some(base) = main_path.file_name().and_then(|n| n.to_str()) else {
        return Ok(Vec::new());
    };
    let prefix = format!("{base}.vec.");
    let mut names = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        // A missing directory just means no spaces yet.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(plugin_err(e)),
    };
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if let Some(space) = name.strip_prefix(&prefix) {
            if is_valid_space_name(space) {
                names.push(space.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    Ok(names)
}

/// Opens named vector spaces backed by [`VectorEngine`] companion files.
///
/// Register it on a builder via [`with_vector_spaces`]
/// so `add_vector_in` / `similar_in` / `get_vector_in` / `vector_spaces` on the
/// resulting [`axil_core::Axil`] handle can open per-space storage on demand.
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorSpaceFactory;

impl axil_core::VectorSpaceFactory for VectorSpaceFactory {
    fn open_space(
        &self,
        main_path: &Path,
        space: &str,
        dim: Option<usize>,
    ) -> axil_core::Result<std::sync::Arc<dyn VectorIndex>> {
        let vec_path = vector_space_db_path(main_path, space);
        let engine = VectorEngine::open_space_at(&vec_path, dim)?;
        Ok(std::sync::Arc::new(engine))
    }

    fn space_names(&self, main_path: &Path) -> axil_core::Result<Vec<String>> {
        list_vector_space_names(main_path)
    }

    fn space_meta(&self, main_path: &Path, space: &str) -> axil_core::Result<(usize, usize)> {
        let vec_path = vector_space_db_path(main_path, space);
        if !vec_path.exists() {
            return Err(AxilError::plugin(format!(
                "vector space {} does not exist",
                vec_path.display()
            )));
        }
        // Read-only probe: no writable handle, no vector load, no index build —
        // listings stay proportional to metadata, not store size.
        let db = ReadOnlyDatabase::open(&vec_path).map_err(plugin_err)?;
        let txn = db.begin_read().map_err(plugin_err)?;
        let dimensions = match txn.open_table(META_TABLE) {
            Ok(meta) => match meta.get("dimensions").map_err(plugin_err)? {
                Some(guard) => {
                    let s = std::str::from_utf8(guard.value()).map_err(plugin_err)?;
                    s.parse::<usize>().map_err(plugin_err)?
                }
                None => {
                    return Err(AxilError::plugin(
                        "vector space has no stored dimension — file may be corrupt",
                    ))
                }
            },
            Err(redb::TableError::TableDoesNotExist(_)) => {
                return Err(AxilError::plugin(
                    "vector space has no metadata table — file may be corrupt",
                ))
            }
            Err(e) => return Err(plugin_err(e)),
        };
        let count = match txn.open_table(VECTORS_TABLE) {
            Ok(t) => t.len().map_err(plugin_err)? as usize,
            Err(redb::TableError::TableDoesNotExist(_)) => 0,
            Err(e) => return Err(plugin_err(e)),
        };
        Ok((dimensions, count))
    }
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

/// Persist many vectors to the vector database under a single write transaction.
///
/// One `begin_write`/`commit` for the whole batch amortizes the per-record
/// fsync, which dominates the per-chunk ingest cost on boot/scip/deps-refresh.
fn persist_vectors_batch(
    db: &Database,
    items: &[(RecordId, &[f32])],
) -> axil_core::Result<()> {
    let txn = db.begin_write().map_err(plugin_err)?;
    {
        let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        for (id, vector) in items {
            let bytes = vector_to_bytes(vector);
            table
                .insert(id.as_str(), bytes.as_slice())
                .map_err(plugin_err)?;
        }
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
/// without manually constructing a `VectorEngine`.
pub trait AxilBuilderVectorExt {
    /// Enable vector search with the given dimensions.
    ///
    /// Creates a `VectorEngine` internally using the builder's path.
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
    /// Creates a `VectorEngine` with the model's native dimensions and attaches
    /// the embedder for `embed_field()`, `embed_text()`, and `similar_to()`.
    fn with_embedder_model(self, model: EmbeddingModel) -> axil_core::Result<Self>
    where
        Self: Sized;

}

/// Register the named-vector-space factory on a builder.
///
/// Additive and independent of the default vector index: it only enables
/// `add_vector_in` / `similar_in` / `get_vector_in` / `vector_spaces` on the
/// built handle. Cheap (a unit factory), so it is safe to call on every
/// write-path open.
///
/// A free function rather than an [`AxilBuilderVectorExt`] method so adding
/// it did not grow the published trait (a breaking change for external
/// implementors).
pub fn with_vector_spaces(builder: AxilBuilder) -> AxilBuilder {
    builder.with_vector_space_factory(std::sync::Arc::new(VectorSpaceFactory))
}

impl AxilBuilderVectorExt for AxilBuilder {
    fn with_vector(self, dims: usize) -> axil_core::Result<Self> {
        let path = self.path().to_path_buf();
        let plugin = VectorEngine::open(&path, dims)?;
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
        let plugin = VectorEngine::open(&path, dims)?;
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

    fn temp_engine(dims: usize) -> (VectorEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let plugin = VectorEngine::open(&path, dims).unwrap();
        (plugin, dir)
    }

    #[test]
    fn re_add_counts_as_reclaimable_for_compaction() {
        // A re-add (every update / re-embed) tombstones the old graph node.
        // The background compactor gates on deleted_count(), so re-add
        // tombstones MUST surface there — otherwise an update-heavy workload
        // accumulates dead nodes that never compact (regression: deleted_count
        // previously returned deletes_since_rebuild, bumped only by remove).
        let (plugin, _dir) = temp_engine(4);
        let id = RecordId::new();
        plugin.add(id.clone(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(plugin.deleted_count(), 0);
        for _ in 0..5 {
            plugin.add(id.clone(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        }
        assert!(
            plugin.deleted_count() >= 5,
            "re-add tombstones must count toward compaction, got {}",
            plugin.deleted_count()
        );
    }

    #[test]
    fn add_and_search() {
        let (plugin, _dir) = temp_engine(3);
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
            let plugin = VectorEngine::open(&path, 3).unwrap();
            plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
            assert_eq!(plugin.vector_count(), 1);
        }

        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
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
            let plugin = VectorEngine::open(&path, 3).unwrap();
            plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
            plugin.on_record_delete(&id).unwrap();
            assert_eq!(plugin.vector_count(), 0);
        }

        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
            assert_eq!(plugin.vector_count(), 0);
        }
    }

    #[test]
    fn add_batch_parity_with_add_loop() {
        // The same vectors added via add_batch must be searchable identically
        // to the per-record add path.
        let (plugin, _dir) = temp_engine(3);
        let id1 = RecordId::new();
        let id2 = RecordId::new();
        let id3 = RecordId::new();
        let v1 = [1.0_f32, 0.0, 0.0];
        let v2 = [0.0_f32, 1.0, 0.0];
        let v3 = [0.0_f32, 0.0, 1.0];

        plugin
            .add_batch(&[
                (id1.clone(), v1.as_slice()),
                (id2.clone(), v2.as_slice()),
                (id3.clone(), v3.as_slice()),
            ])
            .unwrap();

        assert_eq!(plugin.vector_count(), 3);
        assert_eq!(plugin.search(&v1, 1).unwrap()[0].0, id1);
        assert_eq!(plugin.search(&v2, 1).unwrap()[0].0, id2);
        assert_eq!(plugin.search(&v3, 1).unwrap()[0].0, id3);
    }

    #[test]
    fn add_batch_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch_persist.axil");
        let id1 = RecordId::new();
        let id2 = RecordId::new();

        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
            plugin
                .add_batch(&[
                    (id1.clone(), [1.0_f32, 0.0, 0.0].as_slice()),
                    (id2.clone(), [0.0_f32, 1.0, 0.0].as_slice()),
                ])
                .unwrap();
            assert_eq!(plugin.vector_count(), 2);
        }

        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
            assert_eq!(plugin.vector_count(), 2);
            assert_eq!(plugin.search(&[1.0, 0.0, 0.0], 1).unwrap()[0].0, id1);
            assert_eq!(plugin.search(&[0.0, 1.0, 0.0], 1).unwrap()[0].0, id2);
        }
    }

    #[test]
    fn add_batch_empty_is_noop() {
        let (plugin, _dir) = temp_engine(3);
        plugin.add_batch(&[]).unwrap();
        assert_eq!(plugin.vector_count(), 0);
    }

    #[test]
    fn on_record_insert_without_embedder_is_noop() {
        let (plugin, _dir) = temp_engine(3);
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
            let plugin = VectorEngine::open(&path, 3).unwrap();
            let id = RecordId::new();
            plugin.add(id, &[1.0, 0.0, 0.0]).unwrap();
        }

        let result = VectorEngine::open(&path, 768);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("dimension mismatch"), "error was: {err}");
    }
}
