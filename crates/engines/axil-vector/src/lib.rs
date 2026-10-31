pub mod binary;
pub mod download;
pub mod embed;
pub mod hnsw;
pub mod identity;
pub mod mmap;
pub mod models;
pub mod quantize;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::{Mutex, RwLock};
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
use crate::identity::{check_store_model_at, ModelCheck, ModelMismatch, StoredModel, MODEL_KEY};
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
    /// Held by every mutation across its `.vec` commit and its live-index
    /// update, so the two never interleave with another mutation's (a delete
    /// cannot land between an add's disk write and its index insert). Kept
    /// apart from `index` so searches never wait on a disk commit.
    mutation: Mutex<()>,
    vector_db: Database,
    embedder: Option<Embedder>,
    /// Entries present in the on-disk store but not loadable into the live
    /// index at open (unparsable id, malformed bytes, wrong dimensions,
    /// non-finite or all-zero values). Cleared by a purge.
    skipped_at_load: AtomicUsize,
    /// The embedding model whose vectors the store holds
    /// ([`EmbeddingModel::store_identity`]), when it was opened for one by
    /// [`open_store_for_model`]; `None` for a raw open.
    model: Option<String>,
    /// Whether the store's metadata names `model` yet. A store taken over at
    /// open is named by its next vector write, inside that write's
    /// transaction, so an open that only reads writes nothing.
    model_recorded: AtomicBool,
    /// The database whose default store this is, when it is one: a model
    /// attached to it is checked against the model the store records. `None`
    /// for a named space, whose vectors come from its caller.
    default_of: Option<PathBuf>,
    /// Why the store refuses the model attached to it (see
    /// [`VectorEngine::with_model`]): it searches, adds and embeds nothing,
    /// and a database builder records this instead of attaching it.
    refusal: Option<axil_core::DegradedEngine>,
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
    /// The vector data is stored at `<db_path>.vec`. The store is opened raw,
    /// for vectors its caller supplies; a model given to it later
    /// ([`VectorEngine::with_model`], [`VectorEngine::with_embedder`]) is
    /// checked against the model the store records.
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
        let db_path = db_path.as_ref();
        let mut engine = Self::open_raw(db_path, config)?;
        engine.default_of = Some(db_path.to_path_buf());
        Ok(engine)
    }

    /// [`VectorEngine::open_with_config`], short of naming the database the
    /// store belongs to.
    fn open_raw(db_path: &Path, config: VectorConfig) -> axil_core::Result<Self> {
        let vec_path = vector_db_path(db_path);
        let vector_db = Database::create(&vec_path).map_err(open_err)?;

        // An existing store with matching dimensions needs no write: a read
        // txn confirms it, so read-only commands (recall, fts, hook lookups)
        // never hold a write txn a timeout could kill mid-commit.
        let stored = {
            let txn = vector_db.begin_read().map_err(plugin_err)?;
            match txn.open_table(META_TABLE) {
                Ok(meta) => match meta.get("dimensions").map_err(plugin_err)? {
                    Some(guard) => {
                        let s = std::str::from_utf8(guard.value()).map_err(plugin_err)?;
                        Some(s.parse::<usize>().map_err(plugin_err)?)
                    }
                    None => None,
                },
                Err(redb::TableError::TableDoesNotExist(_)) => None,
                Err(e) => return Err(plugin_err(e)),
            }
        };
        if stored == Some(config.dimensions) {
            return Self::load(vector_db, config, &vec_path);
        }

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
        Self::load(vector_db, config, &vec_path)
    }

    /// Load a store's persisted vectors into a fresh in-memory index.
    fn load(vector_db: Database, config: VectorConfig, vec_path: &Path) -> axil_core::Result<Self> {
        let (vectors, skipped_at_load) = load_all_vectors(&vector_db, config.dimensions)?;
        warn_unloadable(skipped_at_load, vec_path);
        let index = HnswIndex::from_vectors(config.dimensions, vectors);

        Ok(Self {
            config,
            index: RwLock::new(index),
            mutation: Mutex::new(()),
            vector_db,
            embedder: None,
            skipped_at_load: AtomicUsize::new(skipped_at_load),
            model: None,
            model_recorded: AtomicBool::new(true),
            default_of: None,
            refusal: None,
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
        let vector_db = Database::create(vec_path).map_err(open_err)?;

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
                    write_meta_txn(&vector_db, requested, None)?;
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
            write_meta_txn(&vector_db, requested, None)?;
            requested
        };

        let config = VectorConfig {
            dimensions,
            auto_embed_fields: Vec::new(),
        };
        Self::load(vector_db, config, vec_path)
    }

    /// Attach an embedder for text-to-vector conversion.
    ///
    /// On a database's default store the embedder's model is checked against
    /// the model the store records, as [`open_store_for_model`] checks it. A
    /// store that holds another model's vectors refuses it: the embedder is
    /// dropped, the refusal is printed once on stderr, every search, add and
    /// embed fails with it, and a database builder records it
    /// ([`axil_core::Axil::degraded_engines`]) instead of attaching the
    /// engine.
    pub fn with_embedder(mut self, embedder: Embedder) -> Self {
        if self.check_model(embedder.model()) {
            self.embedder = Some(embedder);
        }
        self
    }

    /// Attach an embedder by model type.
    ///
    /// If the model files are not available locally (missing, or not at
    /// their pinned sizes), the pinned files are downloaded from HuggingFace
    /// on first use.
    ///
    /// The model is checked against the store first, as in
    /// [`VectorEngine::with_embedder`]; a store that refuses it comes back
    /// refused, without the model loaded or downloaded.
    pub fn with_model(mut self, model: EmbeddingModel) -> Result<Self, String> {
        if !self.check_model(&model) {
            return Ok(self);
        }
        if !matches!(model, EmbeddingModel::Custom { .. })
            && !crate::download::is_model_available(&model)
        {
            axil_core::diagln!(
                "Model {} is not installed or does not match its pinned files — downloading ({})...",
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

    /// The model name a vector write must record in the store's metadata:
    /// `Some` until the store names the model it was opened for. Read under
    /// the mutation lock, with [`VectorEngine::recorded`] after the commit.
    fn model_to_record(&self) -> Option<&str> {
        match &self.model {
            Some(model) if !self.model_recorded.load(Ordering::Acquire) => Some(model.as_str()),
            _ => None,
        }
    }

    /// Note that a committed write recorded `model` (from
    /// [`VectorEngine::model_to_record`]).
    fn recorded(&self, model: Option<&str>) {
        if model.is_some() {
            self.model_recorded.store(true, Ordering::Release);
        }
    }

    /// Check a database's default store against `model` before an embedder
    /// for it attaches: `false` when the store holds another model's vectors,
    /// which leaves the engine refused. Takes the store over for `model` the
    /// way [`open_store_for_model`] does otherwise, so its next write records
    /// the model.
    ///
    /// A store opened for this model already passed the check, and a named
    /// space holds vectors its caller supplies, so neither is checked.
    fn check_model(&mut self, model: &EmbeddingModel) -> bool {
        let name = model.store_identity();
        if self.model.as_deref() == Some(name.as_str()) {
            return true;
        }
        let Some(db_path) = self.default_of.clone() else {
            return true;
        };
        let stored = self
            .vector_db
            .begin_read()
            .map_err(plugin_err)
            .and_then(|txn| stored_model_in(&txn));
        let recorded = match stored {
            Ok(Some(stored)) => match check_store_model_at(&db_path, &stored, model) {
                ModelCheck::Recorded => true,
                ModelCheck::Empty => false,
                ModelCheck::Legacy => {
                    identity::note_adopted(&db_path, model);
                    false
                }
                ModelCheck::Refused(mismatch) => {
                    identity::warn_refused(&db_path, &mismatch);
                    self.refuse(mismatch.degraded(&db_path));
                    return false;
                }
            },
            // The open just wrote its dimensions, so this does not happen.
            Ok(None) => false,
            Err(e) => {
                self.refuse(vector_store_unavailable(&db_path, &e));
                return false;
            }
        };
        self.model = Some(name);
        self.model_recorded = AtomicBool::new(recorded);
        true
    }

    /// Leave the engine refused for `why`, with no embedder.
    fn refuse(&mut self, why: axil_core::DegradedEngine) {
        self.embedder = None;
        self.refusal = Some(why);
    }

    /// The error every search, add and embed fails with on a refused engine.
    fn ensure_usable(&self) -> axil_core::Result<()> {
        match &self.refusal {
            Some(why) => Err(AxilError::plugin(format!(
                "{}. Fix: {}",
                why.summary(),
                why.fix
            ))),
            None => Ok(()),
        }
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
            .embed_document(&combined)
            .map_err(|e| AxilError::plugin(format!("auto-embed failed: {e}")))?;
        self.validate_vector(&record.id, &vector)?;

        // Persist to disk first so a crash can't leave the in-memory index
        // ahead of storage.
        let _mutation = self.mutation.lock();
        let model = self.model_to_record();
        persist_vector(&self.vector_db, &record.id, &vector, model)?;
        self.recorded(model);
        self.index
            .write()
            .add(record.id.clone(), vector.clone())
            .map_err(AxilError::plugin)?;

        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> axil_core::Result<()> {
        let _mutation = self.mutation.lock();
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
        let _mutation = self.mutation.lock();
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
        self.ensure_usable()?;
        // Validate before touching storage: a rejected write must leave any
        // previously stored vector for this id intact.
        self.validate_vector(&id, vector)?;
        // Persist to disk first so a crash can't leave the in-memory
        // index ahead of storage.
        let _mutation = self.mutation.lock();
        let model = self.model_to_record();
        persist_vector(&self.vector_db, &id, vector, model)?;
        self.recorded(model);
        self.index
            .write()
            .add(id, vector.to_vec())
            .map_err(AxilError::plugin)?;
        Ok(())
    }

    fn add_batch(&self, items: &[(RecordId, &[f32])]) -> axil_core::Result<()> {
        self.ensure_usable()?;
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
        let _mutation = self.mutation.lock();
        let model = self.model_to_record();
        persist_vectors_batch(&self.vector_db, items, model)?;
        self.recorded(model);
        let mut idx = self.index.write();
        for (id, vector) in items {
            idx.add(id.clone(), vector.to_vec())
                .map_err(AxilError::plugin)?;
        }
        Ok(())
    }

    fn remove_batch(&self, ids: &[RecordId]) -> axil_core::Result<usize> {
        self.remove_batch_if(ids, &|_| true)
    }

    fn remove_batch_if(
        &self,
        ids: &[RecordId],
        doomed: &dyn Fn(&RecordId) -> bool,
    ) -> axil_core::Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        // The check runs under the mutation lock, so an add for one of these
        // ids either finished before it (and the check sees its record) or
        // waits until the removal is done.
        let _mutation = self.mutation.lock();
        let ids: Vec<RecordId> = ids.iter().filter(|id| doomed(id)).cloned().collect();
        if ids.is_empty() {
            return Ok(0);
        }
        // One commit for the whole batch, before the live index changes, so a
        // failed commit leaves the two in step.
        delete_vectors_batch(&self.vector_db, &ids)?;
        let mut idx = self.index.write();
        Ok(ids.iter().filter(|id| idx.remove(id)).count())
    }

    fn search(&self, query: &[f32], top_k: usize) -> axil_core::Result<Vec<(RecordId, f32)>> {
        self.ensure_usable()?;
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

    fn embedding_model(&self) -> Option<String> {
        self.model.clone()
    }

    fn embedding_model_assumed(&self) -> bool {
        self.model.is_some()
            && !self.model_recorded.load(Ordering::Acquire)
            && self.vector_count() > 0
    }

    fn refusal(&self) -> Option<axil_core::DegradedEngine> {
        self.refusal.clone()
    }
}

impl TextEmbedder for VectorEngine {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.ensure_usable()?;
        match &self.embedder {
            Some(e) => e.embed_document(text).map_err(AxilError::plugin),
            None => Err(AxilError::plugin(
                "no embedder configured — use with_embedder() or with_model()",
            )),
        }
    }

    fn embed_query(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.ensure_usable()?;
        match &self.embedder {
            Some(e) => e.embed_query(text).map_err(AxilError::plugin),
            None => Err(AxilError::plugin(
                "no embedder configured — use with_embedder() or with_model()",
            )),
        }
    }

    fn embed_batch(&self, texts: &[&str]) -> axil_core::Result<Vec<Vec<f32>>> {
        self.ensure_usable()?;
        match &self.embedder {
            #[cfg(feature = "embed")]
            Some(e) => e.embed_documents_batch(texts).map_err(AxilError::plugin),
            #[cfg(not(feature = "embed"))]
            Some(e) => texts
                .iter()
                .map(|t| e.embed_document(t).map_err(AxilError::plugin))
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

    fn remove_from_space(
        &self,
        main_path: &Path,
        space: &str,
        id: &RecordId,
    ) -> axil_core::Result<()> {
        let vec_path = vector_space_db_path(main_path, space);
        if !vec_path.exists() {
            return Ok(());
        }
        // Straight to the durable store: no vector load, no graph build, and
        // no write commit unless the space actually holds the id.
        let db = Database::open(&vec_path).map_err(open_err)?;
        delete_vector(&db, id)
    }

    fn space_meta(&self, main_path: &Path, space: &str) -> axil_core::Result<(usize, usize)> {
        let vec_path = vector_space_db_path(main_path, space);
        if !vec_path.exists() {
            return Err(AxilError::plugin(format!(
                "vector space {} does not exist",
                vec_path.display()
            )));
        }
        // Metadata-only probe (writable only to repair an uncleanly closed
        // store): no vector load, no index build, so listings stay
        // proportional to metadata, not store size.
        let (meta, _repaired) = read_repairing(&vec_path, |txn| read_space_meta(&txn))?;
        Ok(meta)
    }
}

/// Dimensions and vector count of a named space, from its metadata alone.
fn read_space_meta(txn: &redb::ReadTransaction) -> axil_core::Result<(usize, usize)> {
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

/// One write txn that ensures a store's tables exist and persists its
/// dimension, and the model whose vectors it is for when there is one: the
/// only durable write opening a store ever performs.
fn write_meta_txn(
    vector_db: &Database,
    dimensions: usize,
    model: Option<&str>,
) -> axil_core::Result<()> {
    let txn = vector_db.begin_write().map_err(plugin_err)?;
    {
        let _ = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        let mut meta = txn.open_table(META_TABLE).map_err(plugin_err)?;
        meta.insert("dimensions", dimensions.to_string().as_bytes())
            .map_err(plugin_err)?;
        if let Some(model) = model {
            meta.insert(MODEL_KEY, model.as_bytes())
                .map_err(plugin_err)?;
        }
    }
    txn.commit().map_err(plugin_err)
}

/// Record `model` in the store's metadata within the write transaction
/// `txn`, when there is one to record.
fn record_model_in(txn: &redb::WriteTransaction, model: Option<&str>) -> axil_core::Result<()> {
    if let Some(model) = model {
        let mut meta = txn.open_table(META_TABLE).map_err(plugin_err)?;
        meta.insert(MODEL_KEY, model.as_bytes())
            .map_err(plugin_err)?;
    }
    Ok(())
}

/// Persist a single vector to the vector database, recording `model` in the
/// same transaction when it is `Some`.
fn persist_vector(
    db: &Database,
    id: &RecordId,
    vector: &[f32],
    model: Option<&str>,
) -> axil_core::Result<()> {
    let bytes = vector_to_bytes(vector);
    let txn = db.begin_write().map_err(plugin_err)?;
    {
        let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        table
            .insert(id.as_str(), bytes.as_slice())
            .map_err(plugin_err)?;
    }
    record_model_in(&txn, model)?;
    txn.commit().map_err(plugin_err)?;
    Ok(())
}

/// Persist many vectors to the vector database under a single write
/// transaction, recording `model` in it when it is `Some`.
///
/// One `begin_write`/`commit` for the whole batch amortizes the per-record
/// fsync, which dominates the per-chunk ingest cost on boot/scip/deps-refresh.
fn persist_vectors_batch(
    db: &Database,
    items: &[(RecordId, &[f32])],
    model: Option<&str>,
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
    record_model_in(&txn, model)?;
    txn.commit().map_err(plugin_err)?;
    Ok(())
}

/// Delete many vectors from the vector database under a single write
/// transaction. Ids the store does not hold are skipped.
fn delete_vectors_batch(db: &Database, ids: &[RecordId]) -> axil_core::Result<()> {
    let txn = db.begin_write().map_err(plugin_err)?;
    {
        let mut table = txn.open_table(VECTORS_TABLE).map_err(plugin_err)?;
        for id in ids {
            table.remove(id.as_str()).map_err(plugin_err)?;
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

/// Map a redb open error. "Held open by another process" becomes the retryable
/// [`AxilError::Busy`], so callers can wait for the writer instead of treating
/// the store as broken; anything else is a plugin error.
fn open_err(e: redb::DatabaseError) -> AxilError {
    match e {
        redb::DatabaseError::DatabaseAlreadyOpen => AxilError::Busy,
        other => plugin_err(other),
    }
}

/// Run `read` against a store without taking its writer lock, unless the store
/// was not closed cleanly. Returns the result and whether a repair ran.
///
/// A process that dies mid-write (a hook killed by its timeout, a crash) leaves
/// the redb file needing repair, and a read-only open refuses such a file. In
/// that case the store is opened writable once, which runs redb's repair, and
/// read from that handle. Engine opens are writable anyway, so this writes
/// nothing that opening the store would not.
fn read_repairing<T>(
    path: &Path,
    read: impl FnOnce(redb::ReadTransaction) -> axil_core::Result<T>,
) -> axil_core::Result<(T, bool)> {
    match ReadOnlyDatabase::open(path) {
        Ok(db) => Ok((read(db.begin_read().map_err(plugin_err)?)?, false)),
        Err(redb::DatabaseError::RepairAborted) => {
            let db = Database::open(path).map_err(open_err)?;
            Ok((read(db.begin_read().map_err(plugin_err)?)?, true))
        }
        Err(e) => Err(open_err(e)),
    }
}

/// What [`probe_vector_store`] found at a database's default vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorStoreProbe {
    /// No `.vec` file exists.
    Missing,
    /// The store is readable.
    Ready {
        /// Stored vector dimensions.
        dimensions: usize,
        /// True when the store had not been closed cleanly and the probe
        /// repaired it.
        repaired: bool,
    },
}

impl VectorStoreProbe {
    /// Stored dimensions, or `None` when the store is missing.
    pub fn dimensions(self) -> Option<usize> {
        match self {
            Self::Missing => None,
            Self::Ready { dimensions, .. } => Some(dimensions),
        }
    }
}

/// Probe the default vector store without loading it, repairing it first if it
/// was not closed cleanly.
///
/// Errors are never "missing": [`AxilError::Busy`] means another process holds
/// the store open for writing (retry), and a plugin error means the file is
/// corrupt or unreadable.
///
/// It reads the store before the database is opened, so it runs the open guard
/// first ([`axil_core::open_guard`]), whether the store exists or not: a
/// process that has the database open is asked to let go before this reads
/// its store, or before the open that follows creates one.
pub fn probe_vector_store(db_path: impl AsRef<Path>) -> axil_core::Result<VectorStoreProbe> {
    axil_core::open_guard(db_path.as_ref());
    let vec_path = vector_db_path(db_path.as_ref());
    if !vec_path.exists() {
        return Ok(VectorStoreProbe::Missing);
    }
    let (dimensions, repaired) = read_repairing(&vec_path, |txn| stored_dimensions(&txn))?;
    Ok(VectorStoreProbe::Ready {
        dimensions,
        repaired,
    })
}

/// Describe a default vector store that exists but could not be probed, for
/// [`AxilBuilder::with_degraded_engine`]. Shared by every adapter so the
/// reason and the remedy read the same everywhere.
pub fn vector_store_unavailable(
    db_path: impl AsRef<Path>,
    err: &AxilError,
) -> axil_core::DegradedEngine {
    let db_path = db_path.as_ref();
    let vec_path = vector_db_path(db_path);
    axil_core::DegradedEngine {
        engine: "vector".to_string(),
        reason: format!(
            "{} could not be opened ({err}), so recall runs without vectors",
            vec_path.display()
        ),
        // Not "rename it in place": a `<db>.vec.<name>` file next to the
        // database is read as a named vector space. `axil init` sizes the new
        // store for the embedding model in axil.toml, the size every later
        // open expects; `--vector-dims` would have to repeat that by hand.
        fix: format!(
            "move {} to another directory, then run `axil init {}` (it creates \
             an empty store sized for the embedding model set in axil.toml) and \
             `axil heal --reindex` to re-embed your memory records",
            vec_path.display(),
            db_path.display()
        ),
    }
}

/// Read stored dimensions from a vector database without fully opening the plugin.
///
/// Returns `Ok(None)` if the `.vec` file doesn't exist. Returns an error if the
/// file exists but is corrupt or unreadable (so callers can distinguish "missing"
/// from "broken"). A store that was not closed cleanly is repaired first; use
/// [`probe_vector_store`] to learn whether that happened.
pub fn read_stored_dimensions(db_path: impl AsRef<Path>) -> axil_core::Result<Option<usize>> {
    probe_vector_store(db_path).map(VectorStoreProbe::dimensions)
}

fn stored_dimensions(txn: &redb::ReadTransaction) -> axil_core::Result<usize> {
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
            s.parse()
                .map_err(|e: std::num::ParseIntError| plugin_err(e))
        }
    }
}

/// What a store's metadata says about its vectors, read within `txn`: `None`
/// when it records no dimensions (a store whose creation never committed).
fn stored_model_in(txn: &redb::ReadTransaction) -> axil_core::Result<Option<StoredModel>> {
    let meta = match txn.open_table(META_TABLE) {
        Ok(t) => t,
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
        Err(e) => return Err(plugin_err(e)),
    };
    let dimensions = match meta.get("dimensions").map_err(plugin_err)? {
        Some(guard) => std::str::from_utf8(guard.value())
            .map_err(plugin_err)?
            .parse::<usize>()
            .map_err(plugin_err)?,
        None => return Ok(None),
    };
    let model = match meta.get(MODEL_KEY).map_err(plugin_err)? {
        Some(guard) => Some(
            std::str::from_utf8(guard.value())
                .map_err(plugin_err)?
                .to_string(),
        ),
        None => None,
    };
    let vectors = match txn.open_table(VECTORS_TABLE) {
        Ok(t) => t.len().map_err(plugin_err)? as usize,
        Err(redb::TableError::TableDoesNotExist(_)) => 0,
        Err(e) => return Err(plugin_err(e)),
    };
    Ok(Some(StoredModel {
        dimensions,
        model,
        vectors,
    }))
}

/// What the default vector store of the database at `db_path` records about
/// the model whose vectors it holds, read without loading them: `None` when
/// there is no store.
///
/// Like [`probe_vector_store`], it runs the open guard first and repairs a
/// store that was not closed cleanly. A store with no dimensions recorded is
/// an error (corrupt), not missing.
pub fn read_stored_model(db_path: impl AsRef<Path>) -> axil_core::Result<Option<StoredModel>> {
    axil_core::open_guard(db_path.as_ref());
    let vec_path = vector_db_path(db_path.as_ref());
    if !vec_path.exists() {
        return Ok(None);
    }
    let (stored, _repaired) = read_repairing(&vec_path, |txn| stored_model_in(&txn))?;
    stored.map(Some).ok_or_else(|| {
        AxilError::plugin(
            "vector store exists but has no dimensions metadata — file may be corrupt",
        )
    })
}

/// Why `model` must not use the default vector store of the database at
/// `db_path` (see [`identity`]), read without loading it: `None` when it may,
/// or when there is no store.
pub fn store_model_mismatch(
    db_path: impl AsRef<Path>,
    model: &EmbeddingModel,
) -> axil_core::Result<Option<ModelMismatch>> {
    let db_path = db_path.as_ref();
    let Some(stored) = read_stored_model(db_path)? else {
        return Ok(None);
    };
    Ok(match check_store_model_at(db_path, &stored, model) {
        ModelCheck::Refused(mismatch) => Some(mismatch),
        ModelCheck::Recorded | ModelCheck::Empty | ModelCheck::Legacy => None,
    })
}

/// Ids of the records with a vector in the default vector store of the
/// database at `db_path`, read without loading the vectors: empty when there
/// is no store. Keys that are not record ids are skipped.
pub fn stored_vector_ids(db_path: impl AsRef<Path>) -> axil_core::Result<Vec<RecordId>> {
    axil_core::open_guard(db_path.as_ref());
    let vec_path = vector_db_path(db_path.as_ref());
    if !vec_path.exists() {
        return Ok(Vec::new());
    }
    let (ids, _repaired) = read_repairing(&vec_path, |txn| {
        let table = match txn.open_table(VECTORS_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(plugin_err(e)),
        };
        let mut ids = Vec::new();
        for entry in table.iter().map_err(plugin_err)? {
            let (key, _) = entry.map_err(plugin_err)?;
            if let Ok(id) = RecordId::from_string(key.value()) {
                ids.push(id);
            }
        }
        Ok(ids)
    })?;
    Ok(ids)
}

/// What [`open_store_for_model`] found at a database's default vector store.
// A refused store is the rare case, so the size difference is not worth a box.
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum ModelStore {
    /// The store holds the model's vectors, or is new or taken over for them
    /// (see [`identity`]). Open, with no embedder attached yet.
    Open(VectorEngine),
    /// The store holds another model's vectors, so it was not loaded.
    Refused(ModelMismatch),
}

/// Open (or create) the default vector store of the database at `db_path` for
/// `model`'s vectors, without loading the model.
///
/// A new store records the model at once, and one taken over records it with
/// its next vector write, so an open that only reads writes nothing. Taking
/// over a store from before model records for any model but the default says
/// so once on stderr ([`identity::adoption_notice`]). A store holding another
/// model's vectors is left unloaded and comes back as [`ModelStore::Refused`],
/// for [`refuse_vector_store`].
pub fn open_store_for_model(
    db_path: impl AsRef<Path>,
    model: &EmbeddingModel,
) -> axil_core::Result<ModelStore> {
    let db_path = db_path.as_ref();
    let vec_path = vector_db_path(db_path);
    let name = model.store_identity();
    let vector_db = Database::create(&vec_path).map_err(open_err)?;
    let stored = {
        let txn = vector_db.begin_read().map_err(plugin_err)?;
        stored_model_in(&txn)?
    };
    let recorded = match stored {
        // A new store, or one whose creation never committed. One that holds
        // vectors without saying how wide they are or whose they are cannot
        // be taken for this model's.
        None => {
            let rows = {
                let txn = vector_db.begin_read().map_err(plugin_err)?;
                match txn.open_table(VECTORS_TABLE) {
                    Ok(t) => t.len().map_err(plugin_err)?,
                    Err(redb::TableError::TableDoesNotExist(_)) => 0,
                    Err(e) => return Err(plugin_err(e)),
                }
            };
            if rows > 0 {
                return Err(AxilError::plugin(format!(
                    "{} holds {rows} vector{} but records neither their dimensions nor their \
                     model, so the file may be corrupt; run `axil --db {} reembed` to rebuild it",
                    vec_path.display(),
                    if rows == 1 { "" } else { "s" },
                    db_path.display()
                )));
            }
            write_meta_txn(&vector_db, model.dimensions(), Some(name.as_str()))?;
            true
        }
        Some(stored) => match check_store_model_at(db_path, &stored, model) {
            ModelCheck::Recorded => true,
            ModelCheck::Empty => false,
            ModelCheck::Legacy => {
                identity::note_adopted(db_path, model);
                false
            }
            ModelCheck::Refused(mismatch) => return Ok(ModelStore::Refused(mismatch)),
        },
    };
    let config = VectorConfig {
        dimensions: model.dimensions(),
        auto_embed_fields: Vec::new(),
    };
    let mut engine = VectorEngine::load(vector_db, config, &vec_path)?;
    engine.model = Some(name);
    engine.model_recorded = AtomicBool::new(recorded);
    engine.default_of = Some(db_path.to_path_buf());
    Ok(ModelStore::Open(engine))
}

/// Create an empty vector store at exactly `vec_path` for `model`'s vectors,
/// recording the model at once, without loading the model: the replacement a
/// rebuild fills beside a database's live default store and then moves into
/// its place, so the live store serves on, untouched, until the replacement
/// is complete.
///
/// Errors when `vec_path` exists. A path of the form `<db>.vec.<name>` would
/// be read as a named vector space by every open, so a rebuild must not use
/// one.
pub fn create_store_for_model_at(
    vec_path: &Path,
    model: &EmbeddingModel,
) -> axil_core::Result<VectorEngine> {
    if vec_path.exists() {
        return Err(AxilError::plugin(format!(
            "{} already exists",
            vec_path.display()
        )));
    }
    let name = model.store_identity();
    let vector_db = Database::create(vec_path).map_err(open_err)?;
    write_meta_txn(&vector_db, model.dimensions(), Some(name.as_str()))?;
    let config = VectorConfig {
        dimensions: model.dimensions(),
        auto_embed_fields: Vec::new(),
    };
    let mut engine = VectorEngine::load(vector_db, config, vec_path)?;
    engine.model = Some(name);
    Ok(engine)
}

/// Leave the default vector store of `builder`'s database detached because
/// it holds another model's vectors: say so once on stderr, and record it on
/// the handle ([`axil_core::Axil::degraded_engines`]), where `doctor`, `boot`
/// and recall report it and where an operation that needs vectors finds the
/// reason it fails.
pub fn refuse_vector_store(builder: AxilBuilder, mismatch: &ModelMismatch) -> AxilBuilder {
    let path = builder.path().to_path_buf();
    identity::warn_refused(&path, mismatch);
    builder.with_degraded_engine(mismatch.degraded(&path))
}

/// Attach the default vector store for `model`'s vectors without loading the
/// model, creating the store when there is none: for a command that sets the
/// store up (`init`) or writes raw vectors into it, which embeds nothing.
///
/// Checks the store's model as [`AxilBuilderVectorExt::with_embedder_model`]
/// does, and refuses one that holds another model's vectors the same way.
pub fn with_model_store(
    builder: AxilBuilder,
    model: &EmbeddingModel,
) -> axil_core::Result<AxilBuilder> {
    let path = builder.path().to_path_buf();
    Ok(match open_store_for_model(&path, model)? {
        ModelStore::Open(engine) => builder.with_vector_and_embedder(engine),
        ModelStore::Refused(mismatch) => refuse_vector_store(builder, &mismatch),
    })
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
    ///
    /// A store that holds another model's vectors (see [`identity`]) is not
    /// attached, and the model is not loaded: the database opens without
    /// vectors, the refusal is printed once on stderr and recorded on the
    /// handle ([`axil_core::Axil::degraded_engines`]), and nothing is
    /// embedded into the store until it is rebuilt.
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
        match open_store_for_model(&path, &model)? {
            ModelStore::Open(engine) => {
                let plugin = engine.with_model(model).map_err(AxilError::plugin)?;
                Ok(self.with_vector_and_embedder(plugin))
            }
            ModelStore::Refused(mismatch) => Ok(refuse_vector_store(self, &mismatch)),
        }
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
        axil_core::diagln!(
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
        assert!(
            !plugin.is_graph_built(),
            "a store this size is searched by exact scan, which builds no graph"
        );
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

    /// A store as a killed writer leaves it: the file is copied while its
    /// engine still holds it open, so it carries committed data but was never
    /// closed cleanly. Returns the db path whose `.vec` is the dirty copy.
    fn needs_repair_store(dir: &Path, id: &RecordId) -> PathBuf {
        let live = dir.join("live.axil");
        let dirty = dir.join("dirty.axil");
        let plugin = VectorEngine::open(&live, 3).unwrap();
        plugin.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
        std::fs::copy(vector_db_path(&live), vector_db_path(&dirty)).unwrap();
        drop(plugin);
        dirty
    }

    #[test]
    fn probe_repairs_a_store_that_was_not_closed_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let id = RecordId::new();
        let path = needs_repair_store(dir.path(), &id);
        assert!(
            matches!(
                ReadOnlyDatabase::open(vector_db_path(&path)),
                Err(redb::DatabaseError::RepairAborted)
            ),
            "fixture must need repair, or this test proves nothing"
        );

        assert_eq!(
            probe_vector_store(&path).unwrap(),
            VectorStoreProbe::Ready {
                dimensions: 3,
                repaired: true
            }
        );
        // Repaired for good: the next probe is a plain read, and the data survived.
        assert_eq!(
            probe_vector_store(&path).unwrap(),
            VectorStoreProbe::Ready {
                dimensions: 3,
                repaired: false
            }
        );
        let plugin = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(plugin.search(&[1.0, 0.0, 0.0], 1).unwrap()[0].0, id);
    }

    #[test]
    fn probe_reports_a_store_held_by_a_writer_as_busy() {
        let (plugin, dir) = temp_engine(3);
        let err = probe_vector_store(dir.path().join("test.axil")).unwrap_err();
        assert!(err.is_busy(), "expected Busy, got {err}");
        drop(plugin);
        assert_eq!(
            read_stored_dimensions(dir.path().join("test.axil")).unwrap(),
            Some(3)
        );
    }

    /// The probe reads the store before the database is opened, so it asks
    /// the open guard first, whether there is a store to read or not.
    #[test]
    fn probe_runs_the_open_guard_first() {
        use std::cell::RefCell;
        thread_local! {
            static SEEN: RefCell<Option<Vec<PathBuf>>> = const { RefCell::new(None) };
        }
        fn record(path: &Path) {
            SEEN.with(|s| {
                if let Some(seen) = s.borrow_mut().as_mut() {
                    seen.push(path.to_path_buf());
                }
            });
        }
        assert!(axil_core::set_open_guard(record));
        SEEN.with(|s| *s.borrow_mut() = Some(Vec::new()));
        let (engine, dir) = temp_engine(4);
        drop(engine);
        let db = dir.path().join("test.axil");
        let missing = dir.path().join("none.axil");
        probe_vector_store(&db).unwrap();
        read_stored_dimensions(&missing).unwrap();
        let seen = SEEN.with(|s| s.borrow_mut().take()).unwrap();
        assert_eq!(seen, [db, missing]);
    }

    #[test]
    fn probe_tells_missing_from_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.axil");
        assert_eq!(
            probe_vector_store(&path).unwrap(),
            VectorStoreProbe::Missing
        );

        std::fs::write(vector_db_path(&path), b"not a redb file").unwrap();
        let err = probe_vector_store(&path).unwrap_err();
        assert!(!err.is_busy(), "corrupt must not look retryable: {err}");
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
    fn an_add_waits_for_a_removal_in_progress() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("serialized.axil");
        let id = RecordId::new();
        let engine = VectorEngine::open(&path, 3).unwrap();
        engine.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();

        // The removal decides while an add for the same id is under way. The
        // add must wait for it rather than land between its disk delete and
        // its live-index update, which would leave the two disagreeing.
        let (started, start) = std::sync::mpsc::channel::<()>();
        let (shared, add_id) = (&engine, id.clone());
        std::thread::scope(|scope| {
            let adder = scope.spawn(move || {
                start.recv().unwrap();
                shared.add(add_id, &[0.0, 1.0, 0.0]).unwrap();
            });
            let removed = engine
                .remove_batch_if(&[id.clone()], &|_| {
                    started.send(()).unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    true
                })
                .unwrap();
            assert_eq!(removed, 1);
            adder.join().unwrap();
        });

        // The add came after the removal: its vector is live and on disk.
        assert_eq!(engine.get_vector(&id).unwrap(), Some(vec![0.0, 1.0, 0.0]));
        drop(engine);
        let reopened = VectorEngine::open(&path, 3).unwrap();
        assert_eq!(reopened.get_vector(&id).unwrap(), Some(vec![0.0, 1.0, 0.0]));
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

    const BGE: &str = "bge-small-en-v1.5";
    const GRANITE: &str = "granite-embedding-small-english-r2";

    /// A 384-dimension unit vector along axis `i`.
    fn unit384(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; 384];
        v[i] = 1.0;
        v
    }

    /// The model the store's metadata records, read straight from the file.
    fn recorded_model(path: &Path) -> Option<String> {
        read_stored_model(path).unwrap().unwrap().model
    }

    fn opened(store: ModelStore) -> VectorEngine {
        match store {
            ModelStore::Open(engine) => engine,
            ModelStore::Refused(m) => panic!("the store was refused: {m:?}"),
        }
    }

    fn refused(store: ModelStore) -> ModelMismatch {
        match store {
            ModelStore::Open(_) => panic!("the store must be refused"),
            ModelStore::Refused(m) => m,
        }
    }

    /// A store written before model records: dimensions and one vector.
    fn legacy_store(path: &Path) {
        let raw = VectorEngine::open(path, 384).unwrap();
        raw.add(RecordId::new(), &unit384(1)).unwrap();
    }

    #[test]
    fn a_new_store_records_its_model_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.axil");
        let model = EmbeddingModel::GraniteSmallR2;
        let engine = opened(open_store_for_model(&path, &model).unwrap());
        assert_eq!(engine.embedding_model().as_deref(), Some(GRANITE));
        drop(engine);
        let stored = read_stored_model(&path).unwrap().unwrap();
        assert_eq!(stored.model.as_deref(), Some(GRANITE));
        assert_eq!((stored.dimensions, stored.vectors), (384, 0));
    }

    #[test]
    fn a_store_of_another_model_is_refused_without_being_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bge.axil");
        let id = RecordId::new();
        {
            let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
            engine.add(id.clone(), &unit384(0)).unwrap();
        }

        let m = refused(open_store_for_model(&path, &EmbeddingModel::GraniteSmallR2).unwrap());
        assert_eq!(m.stored.as_deref(), Some(BGE));
        assert_eq!((m.model.as_str(), m.vectors), (GRANITE, 1));
        assert_eq!(
            store_model_mismatch(&path, &EmbeddingModel::GraniteSmallR2).unwrap(),
            Some(m)
        );

        // Nothing changed: the store still records bge-small and serves it.
        assert_eq!(recorded_model(&path).as_deref(), Some(BGE));
        assert_eq!(
            store_model_mismatch(&path, &EmbeddingModel::BgeSmall).unwrap(),
            None
        );
        let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
        assert_eq!(engine.get_vector(&id).unwrap(), Some(unit384(0)));
    }

    #[test]
    fn a_store_of_another_width_is_refused_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.axil");
        {
            let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeBase).unwrap());
            let mut v = vec![0.0; 768];
            v[3] = 1.0;
            engine.add(RecordId::new(), &v).unwrap();
        }
        let m = refused(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
        assert_eq!((m.stored_dimensions, m.model_dimensions), (768, 384));
    }

    #[test]
    fn a_legacy_store_is_recorded_by_its_first_write_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.axil");
        legacy_store(&path);
        assert_eq!(recorded_model(&path), None);

        // The default model takes it over, and an open that only reads
        // writes nothing.
        let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
        assert_eq!(engine.vector_count(), 1);
        assert_eq!(engine.embedding_model().as_deref(), Some(BGE));
        drop(engine);
        assert_eq!(recorded_model(&path), None);

        // Its first vector write records the model.
        let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
        engine
            .add_batch(&[(RecordId::new(), unit384(2).as_slice())])
            .unwrap();
        drop(engine);
        assert_eq!(recorded_model(&path).as_deref(), Some(BGE));

        // From then on it is bge-small's.
        refused(open_store_for_model(&path, &EmbeddingModel::GraniteSmallR2).unwrap());
    }

    #[test]
    fn a_legacy_store_is_refused_to_granite_but_not_to_a_custom_model_passed_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.axil");
        legacy_store(&path);
        let m = refused(open_store_for_model(&path, &EmbeddingModel::GraniteSmallR2).unwrap());
        assert_eq!(m.stored, None);
        // No axil.toml names it, so a caller passed it in and embedded with it.
        let custom = EmbeddingModel::Custom {
            path: PathBuf::from("/models/mine/model.onnx"),
            dimensions: 384,
            pooling: models::PoolingStrategy::Cls,
            max_seq_len: 512,
        };
        let engine = opened(open_store_for_model(&path, &custom).unwrap());
        assert!(engine.embedding_model_assumed());
        drop(engine);
        // Opening only reads: still unrecorded.
        assert_eq!(recorded_model(&path), None);
    }

    #[test]
    fn a_legacy_store_is_taken_over_by_a_pre_identity_model_of_its_width() {
        // A bge-base database written before model records.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.axil");
        {
            let raw = VectorEngine::open(&path, 768).unwrap();
            let mut v = vec![0.0; 768];
            v[0] = 1.0;
            raw.add(RecordId::new(), &v).unwrap();
        }
        let engine = opened(open_store_for_model(&path, &EmbeddingModel::BgeBase).unwrap());
        assert_eq!(engine.vector_count(), 1);
        let mut v = vec![0.0; 768];
        v[1] = 1.0;
        engine.add(RecordId::new(), &v).unwrap();
        drop(engine);
        assert_eq!(recorded_model(&path).as_deref(), Some("bge-base-en-v1.5"));
    }

    #[test]
    fn an_empty_store_goes_to_the_model_that_writes_first() {
        // Created under the default model (say by `axil init`), then the
        // config moved to granite before anything was embedded.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.axil");
        drop(opened(
            open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap(),
        ));
        assert_eq!(recorded_model(&path).as_deref(), Some(BGE));

        let engine = opened(open_store_for_model(&path, &EmbeddingModel::GraniteSmallR2).unwrap());
        engine.add(RecordId::new(), &unit384(4)).unwrap();
        drop(engine);
        assert_eq!(recorded_model(&path).as_deref(), Some(GRANITE));
    }

    #[test]
    fn a_raw_open_keeps_the_recorded_model_and_names_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.axil");
        drop(opened(
            open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap(),
        ));
        {
            let raw = VectorEngine::open(&path, 384).unwrap();
            assert_eq!(raw.embedding_model(), None);
            raw.add(RecordId::new(), &unit384(5)).unwrap();
        }
        assert_eq!(recorded_model(&path).as_deref(), Some(BGE));
    }

    #[test]
    fn stored_vector_ids_lists_owners_without_loading_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ids.axil");
        assert!(stored_vector_ids(&path).unwrap().is_empty());
        assert_eq!(read_stored_model(&path).unwrap(), None);

        let (a, b) = (RecordId::new(), RecordId::new());
        {
            let engine = VectorEngine::open(&path, 384).unwrap();
            engine.add(a.clone(), &unit384(0)).unwrap();
            engine.add(b.clone(), &unit384(1)).unwrap();
        }
        inject_rows(
            &vector_db_path(&path),
            &[("not-a-record-id", vector_to_bytes(&unit384(2)))],
        );
        let ids: std::collections::HashSet<RecordId> =
            stored_vector_ids(&path).unwrap().into_iter().collect();
        assert_eq!(ids, [a, b].into_iter().collect());
    }

    #[test]
    fn a_store_with_vectors_but_no_dimensions_is_not_taken_for_any_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("headless.axil");
        {
            let db = redb::Database::create(vector_db_path(&path)).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(VECTORS_TABLE).unwrap();
                let id = RecordId::new();
                table
                    .insert(id.as_str(), vector_to_bytes(&unit384(0)).as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }
        for model in [EmbeddingModel::BgeSmall, EmbeddingModel::GraniteSmallR2] {
            let err = match open_store_for_model(&path, &model) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{model} must not take over vectors of unknown width"),
            };
            assert!(
                err.contains("holds 1 vector but") && err.contains("reembed"),
                "{err}"
            );
        }
        // Nothing was written into it.
        let db = redb::Database::open(vector_db_path(&path)).unwrap();
        let txn = db.begin_read().unwrap();
        assert!(txn.open_table(META_TABLE).is_err(), "no metadata written");
    }

    #[test]
    fn a_replacement_store_records_its_model_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let at = dir.path().join("memory.axil.vec-rebuild");
        let store = create_store_for_model_at(&at, &EmbeddingModel::GraniteSmallR2).unwrap();
        assert_eq!(store.embedding_model().as_deref(), Some(GRANITE));
        store.add(RecordId::new(), &unit384(0)).unwrap();
        drop(store);
        let db = Database::open(&at).unwrap();
        let txn = db.begin_read().unwrap();
        let stored = stored_model_in(&txn).unwrap().unwrap();
        assert_eq!(
            (stored.model.as_deref(), stored.vectors),
            (Some(GRANITE), 1)
        );
        drop(txn);
        drop(db);
        assert!(create_store_for_model_at(&at, &EmbeddingModel::GraniteSmallR2).is_err());
    }

    #[test]
    fn a_raw_open_given_another_model_refuses_it_without_loading_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.axil");
        let id = RecordId::new();
        opened(open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap())
            .add(id.clone(), &unit384(0))
            .unwrap();

        // granite's files are not needed: the check comes first.
        let engine = VectorEngine::open(&path, 384)
            .unwrap()
            .with_model(EmbeddingModel::GraniteSmallR2)
            .unwrap();
        let refusal = engine.refusal().expect("refused");
        assert!(
            refusal.reason.contains(BGE) && refusal.reason.contains(GRANITE),
            "{}",
            refusal.reason
        );
        assert!(refusal.fix.contains("reembed"), "{}", refusal.fix);
        for err in [
            engine.add(RecordId::new(), &unit384(1)).unwrap_err(),
            engine.search(&unit384(0), 1).unwrap_err(),
            engine.embed("redb").unwrap_err(),
        ] {
            assert!(err.to_string().contains(GRANITE), "{err}");
        }
        drop(engine);
        assert_eq!(recorded_model(&path).as_deref(), Some(BGE));

        // Nor is a named space checked: it holds its caller's vectors.
        let space = VectorEngine::open_space_at(&vector_space_db_path(&path, "fp"), Some(384))
            .unwrap()
            .with_model(EmbeddingModel::Custom {
                path: dir.path().join("missing.onnx"),
                dimensions: 384,
                pooling: crate::models::PoolingStrategy::Cls,
                max_seq_len: 512,
            });
        // It goes on to load the model, which with `embed` fails on the
        // missing file and without it attaches a stub: never a refusal.
        if cfg!(feature = "embed") {
            let err = space.err().expect("the model file is missing");
            assert!(
                err.contains("model file not found") && err.contains("missing.onnx"),
                "{err}"
            );
        } else {
            let space = space.expect("a stub embedder attaches");
            assert!(space.refusal().is_none(), "a named space is never refused");
            assert!(space.embedder.is_some());
        }
    }
}
