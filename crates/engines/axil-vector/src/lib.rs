pub mod binary;
pub mod download;
pub mod embed;
pub mod hnsw;
pub mod mmap;
pub mod models;
pub mod quantize;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

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
use crate::hnsw::{check_vector, HnswIndex, VectorDefect};
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
    /// Entries present in the on-disk store but not loadable into the live
    /// index at open (unparsable id, malformed bytes, wrong dimensions,
    /// non-finite or all-zero values). Cleared by a purge.
    skipped_at_load: AtomicUsize,
}

impl VectorEngine {
    /// Validate a vector against the configured dimensions before any durable
    /// write: the on-disk store has no schema gate of its own, and a rejected
    /// insert must leave any previously stored vector for this id untouched.
    /// Runs the very check [`HnswIndex::add`] applies (and the load path
    /// filters with), so an insert that passes here can never be rejected by
    /// the index after persisting, nor skipped at the next open.
    fn validate_vector(&self, id: &RecordId, vector: &[f32]) -> axil_core::Result<()> {
        check_vector(vector, self.config.dimensions).map_err(|defect| {
            AxilError::plugin(match defect {
                VectorDefect::Dimensions { expected, got } => {
                    format!("dimension mismatch for {id}: expected {expected}, got {got}")
                }
                VectorDefect::NonFinite => {
                    format!("vector for {id} contains non-finite values (NaN or infinity)")
                }
                VectorDefect::Zero => format!(
                    "vector for {id} is all zeros — it has no direction, so cosine \
                     similarity is undefined"
                ),
            })
        })
    }

    /// Entries skipped at the last open because they were unloadable
    /// (unparsable id, malformed bytes, wrong dimensions, non-finite or
    /// all-zero values). `0` after a clean load or a purge.
    pub fn skipped_at_load(&self) -> usize {
        self.skipped_at_load.load(Ordering::Relaxed)
    }

    /// Whether the ANN graph has been built. It is built on the first search
    /// that needs it — never by opening the store, inserting or deleting.
    pub fn is_graph_built(&self) -> bool {
        self.index.read().is_graph_built()
    }

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
        let (vectors, skipped_at_load) = load_all_vectors(&vector_db, config.dimensions)?;
        warn_unloadable(skipped_at_load, &vec_path);
        let index = HnswIndex::from_vectors(config.dimensions, vectors);

        Ok(Self {
            config,
            index: RwLock::new(index),
            vector_db,
            embedder: None,
            skipped_at_load: AtomicUsize::new(skipped_at_load),
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

        let (vectors, skipped_at_load) = load_all_vectors(&vector_db, dimensions)?;
        warn_unloadable(skipped_at_load, vec_path);
        let index = HnswIndex::from_vectors(dimensions, vectors);
        Ok(Self {
            config: VectorConfig {
                dimensions,
                auto_embed_fields: Vec::new(),
            },
            index: RwLock::new(index),
            vector_db,
            embedder: None,
            skipped_at_load: AtomicUsize::new(skipped_at_load),
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
        self.validate_vector(&record.id, &vector)?;

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
    fn skipped_at_load(&self) -> usize {
        VectorEngine::skipped_at_load(self)
    }

    fn purge_unloadable(&self) -> axil_core::Result<usize> {
        let dimensions = self.config.dimensions;
        let txn = self.vector_db.begin_write().map_err(plugin_err)?;
        let purged = {
            let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
            // Judged by the same rule the load path applies, inside this write
            // txn — a row rewritten with a valid vector since open is kept.
            let mut doomed = Vec::new();
            for entry in table.iter().map_err(plugin_err)? {
                let (key, value) = entry.map_err(plugin_err)?;
                if decode_row(key.value(), value.value(), dimensions).is_none() {
                    doomed.push(key.value().to_string());
                }
            }
            for key in &doomed {
                table.remove(key.as_str()).map_err(plugin_err)?;
            }
            doomed.len()
        };
        txn.commit().map_err(plugin_err)?;
        // Unloadable rows were never in the live index, so it needs no change.
        self.skipped_at_load.store(0, Ordering::Relaxed);
        Ok(purged)
    }

    fn add(&self, id: RecordId, vector: &[f32]) -> axil_core::Result<()> {
        // Validate before touching storage: a rejected write must leave any
        // previously stored vector for this id intact.
        self.validate_vector(&id, vector)?;
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
        // Validate the whole batch before any durable write so one bad item
        // can't leave a half-persisted batch behind.
        for (id, vector) in items {
            self.validate_vector(id, vector)?;
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
        // The graph is searchable immediately after an incremental `add` and
        // despite accumulated tombstones, and its one-time lazy build is
        // synchronized inside the index — so search never needs the write
        // lock. Compaction (tombstone reclaim) is the background worker's job,
        // off this hot path.
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
    // Most deletes have nothing here — a record that was never embedded, or a
    // named space that never held the id (the record-delete fan-out visits
    // every space). A read txn answers that without a durable write commit.
    {
        let txn = db.begin_read().map_err(plugin_err)?;
        match txn.open_table(VECTORS_TABLE) {
            Ok(table) => {
                if table.get(id.as_str()).map_err(plugin_err)?.is_none() {
                    return Ok(());
                }
            }
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(plugin_err(e)),
        }
    }
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

/// Decode one stored row, or `None` when it can never enter the index: an
/// unparsable id, a byte length that is not exactly `dims` f32s, or a vector
/// [`check_vector`] rejects. The one rule shared by load and purge.
fn decode_row(id_str: &str, bytes: &[u8], dims: usize) -> Option<(RecordId, Vec<f32>)> {
    let id = RecordId::from_string(id_str).ok()?;
    if bytes.len() != dims.checked_mul(4)? {
        return None;
    }
    let vector = bytes_to_vector(bytes);
    check_vector(&vector, dims).ok()?;
    Some((id, vector))
}

/// Load all persisted vectors into a HashMap.
///
/// Skips (and counts) every row [`decode_row`] rejects, so a bad row left by
/// an older version can never reach — or panic — the index.
fn load_all_vectors(
    db: &Database,
    expected_dims: usize,
) -> axil_core::Result<(HashMap<RecordId, Vec<f32>>, usize)> {
    let txn = db.begin_read().map_err(plugin_err)?;
    let table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;

    let mut vectors = HashMap::new();
    let mut skipped = 0usize;
    let iter = table.iter().map_err(plugin_err)?;

    for entry in iter {
        let entry: (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
            entry.map_err(plugin_err)?;
        match decode_row(entry.0.value(), entry.1.value(), expected_dims) {
            Some((id, vector)) => {
                vectors.insert(id, vector);
            }
            None => skipped += 1,
        }
    }

    Ok((vectors, skipped))
}

/// Tell the operator, once per open, about rows the load had to skip.
fn warn_unloadable(skipped: usize, vec_path: &Path) {
    if skipped > 0 {
        eprintln!(
            "[axil vector] warning: skipped {skipped} unloadable entr{} in {} \
             (unparsable id, malformed bytes, wrong dimensions, non-finite or all-zero \
             values); `axil heal --reindex` clears them",
            if skipped == 1 { "y" } else { "ies" },
            vec_path.display()
        );
    }
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
    fn rejected_add_wrong_dims_leaves_previous_vector_searchable() {
        // A wrong-dimension insert must be rejected BEFORE it touches the
        // durable store — otherwise the error surfaces but the old vector is
        // already gone (and silently skipped at the next open).
        let (plugin, dir) = temp_engine(3);
        let id = RecordId::new();
        plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();

        assert!(plugin.add(id.clone(), &[1.0, 0.0]).is_err());
        let results = plugin.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);

        // Durable state, not just the live index.
        drop(plugin);
        let path = dir.path().join("test.axil");
        let reopened = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(reopened.vector_count(), 1);
        assert_eq!(reopened.search(&[1.0, 0.0, 0.0], 5).unwrap()[0].0, id);
    }

    #[test]
    fn rejected_add_non_finite_leaves_previous_vector_searchable() {
        let (plugin, _dir) = temp_engine(3);
        let id = RecordId::new();
        plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();

        assert!(plugin.add(id.clone(), &[f32::NAN, 0.0, 0.0]).is_err());
        assert!(plugin
            .add(id.clone(), &[f32::INFINITY, 0.0, 0.0])
            .is_err());
        let results = plugin.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }

    #[test]
    fn add_batch_rejects_whole_batch_on_bad_item() {
        // Batch validation happens before any durable write: one bad item
        // must not leave a half-persisted batch behind.
        let (plugin, _dir) = temp_engine(3);
        let good = RecordId::new();
        let bad = RecordId::new();

        let err = plugin
            .add_batch(&[
                (good.clone(), &[1.0, 0.0, 0.0]),
                (bad.clone(), &[1.0, 0.0]), // wrong dims
            ])
            .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));

        assert_eq!(plugin.vector_count(), 0);
        assert!(plugin.search(&[1.0, 0.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn unloadable_entries_are_counted_and_skipped_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("skip.axil");
        let vec_path = vector_db_path(&path);
        let good = RecordId::new();
        let wrong_dims = RecordId::new();

        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
            plugin.add(good.clone(), &[1.0, 0.0, 0.0]).unwrap();
        }

        // Inject two unloadable rows directly: one unparsable id, one valid
        // id whose vector has the wrong dimension count.
        {
            let db = redb::Database::open(&vec_path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(VECTORS_TABLE).unwrap();
                table
                    .insert("not-a-record-id", vector_to_bytes(&[1.0, 0.0, 0.0]).as_slice())
                    .unwrap();
                table
                    .insert(
                        wrong_dims.to_string().as_str(),
                        vector_to_bytes(&[1.0, 0.0]).as_slice(),
                    )
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        let plugin = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(plugin.skipped_at_load(), 2);
        assert_eq!(plugin.vector_count(), 1);
        assert_eq!(plugin.search(&[1.0, 0.0, 0.0], 5).unwrap()[0].0, good);
    }

    /// Write raw rows straight into a store's vectors table, bypassing every
    /// insert-time check (what an older version, or corruption, leaves behind).
    fn inject_rows(vec_path: &Path, rows: &[(&str, Vec<u8>)]) {
        let db = redb::Database::open(vec_path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(VECTORS_TABLE).unwrap();
            for (key, bytes) in rows {
                table.insert(*key, bytes.as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    #[test]
    fn open_insert_delete_never_build_the_graph() {
        // Every CLI invocation attaches the vector engine — `store`, the
        // hook-spawned children — and most never search. None of them may pay
        // for a full graph build.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy.axil");
        let items: Vec<(RecordId, Vec<f32>)> = (0..300)
            .map(|i| {
                let t = i as f32;
                (RecordId::new(), vec![t.sin(), t.cos(), (t * 0.3).sin(), 1.0])
            })
            .collect();
        {
            let plugin = VectorEngine::open(&path, 4).unwrap();
            let refs: Vec<(RecordId, &[f32])> = items
                .iter()
                .map(|(id, v)| (id.clone(), v.as_slice()))
                .collect();
            plugin.add_batch(&refs).unwrap();
            assert!(!plugin.is_graph_built());
        }

        let plugin = VectorEngine::open(&path, 4).unwrap();
        assert!(!plugin.is_graph_built(), "open must not build the graph");
        plugin.add(RecordId::new(), &[0.5, 0.5, 0.5, 0.5]).unwrap();
        plugin.on_record_delete(&items[0].0).unwrap();
        plugin.on_record_delete(&RecordId::new()).unwrap();
        assert!(!plugin.is_graph_built(), "insert/delete must not build the graph");
        assert_eq!(plugin.vector_count(), 300);

        let hits = plugin.search(&items[7].1, 1).unwrap();
        assert_eq!(hits[0].0, items[7].0);
        assert!(plugin.is_graph_built(), "the first graph-path search builds it");
    }

    #[test]
    fn purge_unloadable_removes_bad_rows_for_good() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("purge.axil");
        let good = RecordId::new();
        {
            let plugin = VectorEngine::open(&path, 3).unwrap();
            plugin.add(good.clone(), &[1.0, 0.0, 0.0]).unwrap();
            // Nothing to purge on a clean store.
            assert_eq!(plugin.purge_unloadable().unwrap(), 0);
        }
        let orphan = RecordId::new();
        inject_rows(
            &vector_db_path(&path),
            &[
                ("not-a-record-id", vector_to_bytes(&[1.0, 0.0, 0.0])),
                (orphan.as_str(), vector_to_bytes(&[1.0, 0.0])),
            ],
        );

        let plugin = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(plugin.skipped_at_load(), 2);
        assert_eq!(plugin.purge_unloadable().unwrap(), 2);
        assert_eq!(VectorIndex::skipped_at_load(&plugin), 0);
        assert_eq!(plugin.search(&[1.0, 0.0, 0.0], 5).unwrap()[0].0, good);

        drop(plugin);
        let reopened = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(reopened.skipped_at_load(), 0);
        assert_eq!(reopened.vector_count(), 1);
        assert_eq!(reopened.get_vector(&good).unwrap(), Some(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn huge_magnitude_vectors_survive_insert_and_reopen() {
        // [1e20; 3] squares overflow f32. The graph's cosine distance used to
        // panic on the second such insert — after the vector was already
        // persisted, so the store then panicked on every reopen too.
        let (plugin, dir) = temp_engine(3);
        let a = RecordId::new();
        let b = RecordId::new();
        plugin.add(a.clone(), &[1e20, 1e20, 1e20]).unwrap();
        plugin.add(b.clone(), &[1e20, 1e20, 1e20]).unwrap();
        let hits = plugin.search(&[1.0, 1.0, 1.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|(_, s)| (s - 1.0).abs() < 1e-5), "{hits:?}");

        drop(plugin);
        let reopened = VectorEngine::open(dir.path().join("test.axil"), 3).unwrap();
        assert_eq!(reopened.skipped_at_load(), 0);
        assert_eq!(reopened.vector_count(), 2);
        let hits = reopened.search(&[2.0, 2.0, 2.0], 2).unwrap();
        assert!(hits.iter().all(|(_, s)| (s - 1.0).abs() < 1e-5), "{hits:?}");
    }

    #[test]
    fn zero_vector_is_rejected_before_persisting() {
        let (plugin, dir) = temp_engine(3);
        let id = RecordId::new();
        plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();

        let err = plugin.add(id.clone(), &[0.0, 0.0, 0.0]).unwrap_err();
        assert!(err.to_string().contains("zero"), "unexpected error: {err}");
        let err = plugin
            .add_batch(&[(RecordId::new(), [0.0_f32, 0.0, 0.0].as_slice())])
            .unwrap_err();
        assert!(err.to_string().contains("zero"), "unexpected error: {err}");

        drop(plugin);
        let reopened = VectorEngine::open(dir.path().join("test.axil"), 3).unwrap();
        assert_eq!(reopened.vector_count(), 1);
        assert_eq!(reopened.get_vector(&id).unwrap(), Some(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn load_skips_zero_and_truncated_rows_and_indexes_huge_rows() {
        // Rows written before insert-time validation covered them: the load
        // path must count the unusable ones and never panic on any of them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.axil");
        let vec_path = vector_db_path(&path);
        drop(VectorEngine::open(&path, 3).unwrap());

        let huge = [RecordId::new(), RecordId::new()];
        {
            let db = redb::Database::open(&vec_path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(VECTORS_TABLE).unwrap();
                // Directions spread over the sphere (a degenerate corpus can
                // leave outliers unreachable in any HNSW graph).
                for i in 0..200 {
                    let t = i as f32;
                    let v = [(t * 1.7).sin(), (t * 2.9).cos(), (t * 0.61).sin()];
                    table
                        .insert(RecordId::new().as_str(), vector_to_bytes(&v).as_slice())
                        .unwrap();
                }
                for id in &huge {
                    let v = [0.3e20, -0.5e20, 0.8e20];
                    table
                        .insert(id.as_str(), vector_to_bytes(&v).as_slice())
                        .unwrap();
                }
                table
                    .insert(
                        RecordId::new().as_str(),
                        vector_to_bytes(&[0.0, 0.0, 0.0]).as_slice(),
                    )
                    .unwrap();
                let mut truncated = vector_to_bytes(&[1.0, 0.0, 0.0]);
                truncated.extend_from_slice(&[0, 0]);
                table
                    .insert(RecordId::new().as_str(), truncated.as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        let plugin = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(plugin.skipped_at_load(), 2, "zero row + truncated row");
        assert_eq!(plugin.vector_count(), 202);
        // > 128 live vectors, so this exercises the graph path.
        let hits = plugin.search(&[0.3, -0.5, 0.8], 2).unwrap();
        let ids: std::collections::HashSet<&RecordId> = hits.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, huge.iter().collect());
        assert!(hits.iter().all(|(_, s)| (s - 1.0).abs() < 1e-5), "{hits:?}");
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
