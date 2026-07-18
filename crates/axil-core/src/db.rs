use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use crate::diagnostics::{
    self, BenchReport, BenchResult, CheckResult, DatabaseMeta, DatabaseStats, DoctorReport,
    RecordStats, Severity,
};
use crate::error::{AxilError, Result};
use crate::extension::Extension;
use crate::metrics::{AuditEntry, Metrics, OpType, SlowQueryEntry};
use crate::plugin::{
    Direction, EdgeInfo, GraphIndex, Engine, SearchIndex, TextEmbedder, TimeSeriesIndex,
    TraversalStep, VectorIndex,
};
use crate::record::{Record, RecordId};
use crate::storage::Storage;

/// Scoped-alias table. Distinct from axil-memory's
/// `_entity_aliases` (which uses the `{entity, alias}` schema for
/// natural-language aliasing) — these two tables have incompatible row
/// shapes and must not be merged.
pub const SCIP_ALIAS_TABLE: &str = "_scip_aliases";
const RECALL_CHUNKS_TABLE: &str = "_recall_chunks";
const RECALL_CHUNK_MAX_BYTES: usize = 1600;
const RECALL_CHUNK_OVERLAP_BYTES: usize = 400;

/// Marker row written once after `migrate_entity_canonical_id` finishes.
/// Presence of the marker lets `Axil::open()` skip the full-table scan
/// on every subsequent open.
const ENTITY_MIGRATION_MARKER: &str = "entity_canonical_id_v1";

/// Known companion file suffixes for file discovery.
const COMPANION_SUFFIXES: &[(&str, &str)] = &[
    (".vec", "vector"),
    (".graph", "graph"),
    (".ts", "timeseries"),
    (".fts", "fts"),
];

/// Build a companion file path by appending a suffix to the base database path.
pub fn companion_path(base: &Path, suffix: &str) -> PathBuf {
    let mut p = base.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

/// Map an Engine role or bare companion suffix to its canonical `(role, suffix)`.
///
/// Accepts the role (`vector`/`graph`/`timeseries`/`fts`) or the suffix with or
/// without a leading dot (`vec`/`.vec`, `graph`, `ts`/`.ts`, `fts`). `None` for
/// anything else.
fn resolve_engine_suffix(engine: &str) -> Option<(&'static str, &'static str)> {
    match engine.trim_start_matches('.') {
        "vec" | "vector" => Some(("vector", ".vec")),
        "graph" => Some(("graph", ".graph")),
        "ts" | "timeseries" => Some(("timeseries", ".ts")),
        "fts" => Some(("fts", ".fts")),
        _ => None,
    }
}

/// Delete the companion file/dir left behind by a removed Engine.
///
/// When an Engine is dropped (e.g. the binary is rebuilt without its feature,
/// or it is listed in `[engines] disabled`), its companion file becomes an
/// orphan the core can no longer interpret. This removes it cleanly. `engine`
/// accepts the role (`vector`/`graph`/`timeseries`/`fts`) or the bare companion
/// suffix (`vec`/`graph`/`ts`/`fts`); the core `.axil` file is never touched.
///
/// Idempotent: a missing companion reports `existed: false` rather than erroring.
/// Operate on a database that is **not currently open with this Engine attached**
/// that is why the CLI never opens the DB on the `--drop-engine` path.
pub fn drop_engine_companion(
    base: &Path,
    engine: &str,
) -> Result<crate::diagnostics::DropEngineReport> {
    let (role, suffix) = resolve_engine_suffix(engine).ok_or_else(|| {
        crate::error::AxilError::InvalidQuery(format!(
            "unknown engine `{engine}` — expected one of: vector, graph, timeseries, fts"
        ))
    })?;
    let path = companion_path(base, suffix);

    let (existed, bytes_freed) = if path.is_dir() {
        let size = dir_size(&path);
        std::fs::remove_dir_all(&path)
            .map_err(|e| crate::error::AxilError::Storage(Box::new(e)))?;
        (true, size)
    } else if path.exists() {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        std::fs::remove_file(&path).map_err(|e| crate::error::AxilError::Storage(Box::new(e)))?;
        (true, size)
    } else {
        (false, 0)
    };

    Ok(crate::diagnostics::DropEngineReport {
        engine: role.to_string(),
        suffix: suffix.to_string(),
        path: path.display().to_string(),
        existed,
        bytes_freed,
    })
}

/// Information about a database and its plugins.
#[derive(Debug)]
pub struct DatabaseInfo {
    /// Base database path.
    pub path: PathBuf,
    /// All files belonging to this database with their roles.
    pub files: Vec<(PathBuf, String, u64)>,
    /// Total size in bytes across all files.
    pub total_size: u64,
    /// Total record count.
    pub total_records: usize,
    /// Tables with their record counts.
    pub tables: Vec<(String, usize)>,
    /// Per-plugin statistics, keyed by plugin name (e.g. "vector", "graph", "fts").
    /// Each value is a JSON object with plugin-specific stats.
    /// Only populated for plugins that were attached when the database was opened.
    pub plugins: BTreeMap<String, Value>,
}

/// Report returned by `Axil::heal()`.
#[derive(Debug, Default)]
pub struct HealReport {
    /// Daily summaries created from old records.
    pub daily_summaries_created: usize,
    /// Original records purged after daily summarization.
    pub records_purged: usize,
    /// Weekly summaries created from old daily summaries.
    pub weekly_summaries_created: usize,
    /// Daily summaries purged after weekly consolidation.
    pub daily_summaries_purged: usize,
}

/// Builder for configuring and opening an Axil database.
pub struct AxilBuilder {
    path: PathBuf,
    plugins: Vec<Box<dyn Engine>>,
    vector_index: Option<Arc<dyn VectorIndex>>,
    embedder: Option<Arc<dyn TextEmbedder>>,
    graph_index: Option<Arc<dyn GraphIndex>>,
    timeseries_index: Option<Arc<dyn TimeSeriesIndex>>,
    fts_index: Option<Arc<dyn SearchIndex>>,
    llm_provider: Option<Arc<dyn crate::llm::LlmProvider>>,
    llm_config: crate::llm::LlmConfig,
    canonical_publisher: Option<Arc<dyn CanonicalPublisher>>,
    /// Factory for opening named vector spaces (additive; see
    /// [`VectorSpaceFactory`]). `None` leaves named-space operations unavailable
    /// while the default vector index behaves exactly as before.
    vector_space_factory: Option<Arc<dyn VectorSpaceFactory>>,
    /// registered Tier-2 Extensions.
    extensions: Vec<Arc<dyn Extension>>,
    /// Set by FTS plugin when schema migration required a rebuild.
    pub needs_fts_reindex: bool,
    /// Open the core store read-only (no single-writer lock), serving committed
    /// records while another process holds the writable handle.
    read_only: bool,
    /// Optional encryption-at-rest cipher for core record bodies, applied to the
    /// `Storage` handle at [`AxilBuilder::build`] time. `None` means cleartext
    /// bodies — the default. Available only under the off-by-default
    /// `encryption` feature; see [`crate::crypto`] for the honest scope.
    #[cfg(feature = "encryption")]
    cipher: Option<crate::crypto::Cipher>,
}

/// Best-effort sink for canonical IDs published to a workspace control
/// plane (Atlas v0.2+). Implementations should never block the calling
/// `insert` thread — fire-and-forget over a channel or background task.
/// Errors are intentionally swallowed: Atlas is advisory, never on the
/// critical path.
pub trait CanonicalPublisher: Send + Sync {
    fn publish(&self, canonical_id: &str);
}

/// Descriptor for one named vector space (see [`Axil::vector_spaces`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VectorSpaceInfo {
    /// Space name (validated `[a-z0-9_-]{1,32}`).
    pub name: String,
    /// Vector dimension bound to this space on first write.
    pub dimensions: usize,
    /// Number of vectors currently stored in the space.
    pub count: usize,
}

/// Opener for named vector spaces backed by companion files
/// (`<db>.axil.vec.<name>`).
///
/// `axil-core` cannot construct a concrete [`VectorIndex`] itself (the HNSW
/// engine lives in a downstream crate), so the adapter that knows how to build
/// one registers a factory via
/// [`AxilBuilder::with_vector_space_factory`]. Named spaces are additive: the
/// default/unnamed vector index ([`Axil::add_vector`], [`Axil::similar_to_vector`])
/// never consults a factory and is unaffected when one is absent.
pub trait VectorSpaceFactory: Send + Sync {
    /// Open — creating when `dim` is `Some` — the named space for the database
    /// at `main_path`.
    ///
    /// `dim = Some(d)` binds/validates dimension `d` (first write path);
    /// `dim = None` opens an existing space for reading (its stored dimension
    /// governs) and errors if the space does not exist. Each opened space owns
    /// an independent companion file, so a caller must not open the same space
    /// twice concurrently — [`Axil`] caches opened spaces to honor this.
    fn open_space(
        &self,
        main_path: &Path,
        space: &str,
        dim: Option<usize>,
    ) -> Result<Arc<dyn VectorIndex>>;

    /// Names of persisted named spaces for the database at `main_path`, without
    /// opening them (a cheap directory scan of `<main_path>.vec.<name>`).
    fn space_names(&self, main_path: &Path) -> Result<Vec<String>>;

    /// Cheap metadata probe for one named space: `(dimensions, vector count)`.
    ///
    /// Must not take a writable handle, load vectors, or build a search
    /// index — listings call this per space, so it has to stay proportional
    /// to metadata, not store size. Errors if the space does not exist.
    fn space_meta(&self, main_path: &Path, space: &str) -> Result<(usize, usize)>;
}

/// True if `name` is a valid vector-space name (`[a-z0-9_-]{1,32}`).
///
/// Names become companion-file suffixes, so the rule is shared by the API
/// validator here and the engine's directory-scan filter — keep them one
/// implementation.
pub fn is_valid_space_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 32
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Validate a vector-space name, with the failure as an [`AxilError`].
fn validate_space_name(space: &str) -> Result<()> {
    if is_valid_space_name(space) {
        Ok(())
    } else {
        Err(AxilError::InvalidQuery(format!(
            "invalid vector space name '{space}': must match [a-z0-9_-]{{1,32}}"
        )))
    }
}

impl AxilBuilder {
    /// Get the database path this builder was configured with.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Register a vector index plugin for vector search.
    pub fn with_vector_index(mut self, index: Box<dyn VectorIndex>) -> Self {
        self.vector_index = Some(Arc::from(index));
        self
    }

    /// Register a text embedder for text-to-vector conversion.
    ///
    /// Required for `embed_field()`, `embed_text()`, and `similar_to()`.
    /// Not needed for raw vector operations (`add_vector`, `similar_to_vector`).
    pub fn with_embedder(mut self, embedder: Box<dyn TextEmbedder>) -> Self {
        self.embedder = Some(Arc::from(embedder));
        self
    }

    /// Register a factory for named vector spaces (`<db>.axil.vec.<name>`).
    ///
    /// Additive: the default/unnamed vector index is unaffected. Once set,
    /// [`Axil::add_vector_in`], [`Axil::similar_in`], [`Axil::get_vector_in`],
    /// and [`Axil::vector_spaces`] can open per-space companion files on demand.
    pub fn with_vector_space_factory(mut self, factory: Arc<dyn VectorSpaceFactory>) -> Self {
        self.vector_space_factory = Some(factory);
        self
    }

    /// Register a plugin that provides both vector indexing and text embedding.
    ///
    /// Convenience for plugins like `VectorEngine` that implement both traits.
    pub fn with_vector_and_embedder<T>(mut self, plugin: T) -> Self
    where
        T: VectorIndex + TextEmbedder + 'static,
    {
        let arc = Arc::new(plugin);
        self.vector_index = Some(arc.clone());
        self.embedder = Some(arc);
        self
    }

    /// Register a graph index plugin for graph traversal.
    pub fn with_graph_index(mut self, index: Arc<dyn GraphIndex>) -> Self {
        self.graph_index = Some(index);
        self
    }

    /// Register a time-series index plugin for time-range queries.
    pub fn with_timeseries_index(mut self, index: Arc<dyn TimeSeriesIndex>) -> Self {
        self.timeseries_index = Some(index);
        self
    }

    /// Register a full-text search index plugin.
    pub fn with_fts_index(mut self, index: Arc<dyn SearchIndex>) -> Self {
        self.fts_index = Some(index);
        self
    }

    /// Deprecated: this is a no-op. Use `AxilBuilderGraphExt::with_graph_engine()`
    /// from the `axil-graph` crate to enable graph support.
    #[deprecated(note = "use axil_graph::AxilBuilderGraphExt::with_graph_engine() instead")]
    pub fn with_graph(self) -> Self {
        self
    }

    /// Register an LLM provider for enhanced intelligence.
    ///
    /// When set, LLM-enhanced code paths (entity extraction, consolidation,
    /// query understanding) will use this provider. Without it, all features
    /// fall back to algorithmic implementations.
    pub fn with_llm(mut self, provider: Arc<dyn crate::llm::LlmProvider>) -> Self {
        self.llm_provider = Some(provider);
        self
    }

    /// Set LLM configuration (limits, pricing, etc.).
    pub fn with_llm_config(mut self, config: crate::llm::LlmConfig) -> Self {
        self.llm_config = config;
        self
    }

    /// Register a custom plugin.
    pub fn with_engine(mut self, plugin: Box<dyn Engine>) -> Self {
        self.plugins.push(plugin);
        self
    }

    /// Register a Tier-2 Extension by value.
    ///
    /// The Extension's [`Extension::id`] and [`Extension::table_prefixes`]
    /// are validated against already-registered Extensions. The call
    /// panics on duplicate ids or overlapping table prefixes — both are
    /// builder-time programmer errors, not runtime conditions.
    pub fn with_extension<E>(self, ext: E) -> Self
    where
        E: Extension + 'static,
    {
        self.with_extension_arc(Arc::new(ext))
    }

    /// Register a Tier-2 Extension that is already wrapped in
    /// `Arc<dyn Extension>`.
    ///
    /// Useful for hosts that load Extensions dynamically (or hold them
    /// behind a trait object for any other reason) — same validation
    /// and panic semantics as [`AxilBuilder::with_extension`].
    pub fn with_extension_arc(mut self, ext: Arc<dyn Extension>) -> Self {
        let id = ext.id().to_string();
        if self.extensions.iter().any(|e| e.id() == id) {
            panic!(
                "axil: extension id `{id}` registered twice — each \
                 Extension::id() must be unique within a database",
            );
        }
        for prefix in ext.table_prefixes() {
            for existing in &self.extensions {
                for existing_prefix in existing.table_prefixes() {
                    if prefix_overlaps(prefix, existing_prefix) {
                        panic!(
                            "axil: extension `{id}` claims table prefix \
                             `{prefix}` which overlaps with extension \
                             `{}` prefix `{existing_prefix}` — table \
                             prefixes must be disjoint",
                            existing.id(),
                        );
                    }
                }
            }
        }
        self.extensions.push(ext);
        self
    }

    /// Register a workspace-control-plane sink for canonical IDs.
    ///
    /// Each `_entities` insert with a non-empty `canonical_id` will fire
    /// `publisher.publish(canonical_id)` after the storage commit. The
    /// publisher contract is best-effort and non-blocking — all errors
    /// are swallowed inside the publisher implementation, so a missing
    /// or unreachable Atlas never affects local writes.
    pub fn with_canonical_publisher(mut self, publisher: Arc<dyn CanonicalPublisher>) -> Self {
        self.canonical_publisher = Some(publisher);
        self
    }

    /// Open the core store read-only, without taking the exclusive
    /// single-writer lock.
    ///
    /// The resulting handle serves committed records (get/list/recall) and
    /// every mutation fails with [`AxilError::Busy`](crate::AxilError::Busy).
    /// redb's read-only open requests a shared lock that cannot coexist with a
    /// live writer's exclusive lock, so this also fails with `Busy` while a
    /// writer is active — it is a fallback for the gap between short-lived
    /// writer sessions, not a way to read through a live writer. The file must
    /// already exist — a read-only open never creates it.
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Attach an encryption-at-rest cipher to the core record store.
    ///
    /// When set, every core record body is sealed with XChaCha20-Poly1305 before
    /// it is written and unsealed on read; `.vec` embeddings and `.fts` tokens
    /// stay cleartext (see [`crate::crypto`] for the honest scope). Available
    /// only under the off-by-default `encryption` feature. Construct the cipher
    /// from a key source via [`Cipher::from_env`](crate::crypto::Cipher::from_env),
    /// [`Cipher::from_key_file`](crate::crypto::Cipher::from_key_file), or
    /// [`Cipher::resolve`](crate::crypto::Cipher::resolve) — keys never touch the
    /// `.axil` file.
    ///
    /// ```no_run
    /// # #[cfg(feature = "encryption")]
    /// # fn demo() -> axil_core::error::Result<()> {
    /// use axil_core::{Axil, crypto::Cipher};
    /// let db = Axil::open("./memory.axil")
    ///     .with_cipher(Cipher::from_env()?) // reads AXIL_ENC_KEY
    ///     .build()?;
    /// # let _ = db;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "encryption")]
    pub fn with_cipher(mut self, cipher: crate::crypto::Cipher) -> Self {
        self.cipher = Some(cipher);
        self
    }

    /// Open the database and return the handle.
    pub fn build(self) -> Result<Axil> {
        let needs_fts_reindex = self.needs_fts_reindex;
        let storage = if self.read_only {
            Storage::open_read_only(&self.path)?
        } else {
            Storage::open(&self.path)?
        };
        // Attach the encryption cipher (if configured) to the freshly opened
        // core store before any read/write goes through it, so bodies seal and
        // unseal transparently. An explicit `with_cipher` wins; otherwise fall
        // back to the process-wide default installed by the adapter — that
        // fallback is what covers raw `Axil::open(path).build()` opens this code
        // can't reach (core-internal `branch_merge`, the workspace fan-out).
        // No-op without the `encryption` feature — the rebinding is cfg'd out
        // and the default build stays byte-identical.
        #[cfg(feature = "encryption")]
        let storage = match self.cipher.or_else(crate::crypto::default_cipher) {
            Some(cipher) => storage.with_cipher(cipher),
            None => storage,
        };
        let llm_usage = Arc::new(crate::llm::LlmUsageTracker::new(
            self.llm_config.cost_per_1m_input,
            self.llm_config.cost_per_1m_output,
        ));
        let llm_rate_limiter = Arc::new(crate::llm::LlmRateLimiter::new(
            self.llm_config.limits.clone(),
        ));
        let db = Axil {
            path: self.path,
            storage,
            plugins: self.plugins,
            vector_index: self.vector_index,
            embedder: self.embedder,
            graph_index: self.graph_index,
            timeseries_index: self.timeseries_index,
            fts_index: self.fts_index,
            llm_provider: self.llm_provider,
            llm_usage,
            llm_rate_limiter,
            metrics: Arc::new(Metrics::new()),
            vector_space_factory: self.vector_space_factory,
            vector_space_cache: std::sync::RwLock::new(std::collections::HashMap::new()),
            slow_query_threshold_ms: 100.0,
            audit_enabled: std::sync::atomic::AtomicBool::new(false),
            log_counter: std::sync::atomic::AtomicU64::new(0),
            feedback_store: crate::feedback::FeedbackStore::new(),
            canonical_publisher: self.canonical_publisher,
            extensions: std::sync::RwLock::new(self.extensions),
            #[cfg(feature = "event-log")]
            event_log_enabled: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "event-log")]
            event_cursor: crate::event_log::EventCursor::new(),
        };

        // Backfill `_entities.canonical_id` for rows written before the
        // schema gained the field. Skipped after the first successful run
        // via a marker row in `_axil_migrations` — without the marker this
        // would full-scan `_entities` on every open. Fresh DBs (no
        // `_entities` table) skip both the scan and the marker write so
        // we don't materialize system tables for nothing. A read-only handle
        // can't write the marker, so skip the migration entirely — the writer
        // process owns it.
        if !db.storage.is_read_only() && db.entities_table_nonempty() && !db.entity_migration_done()
        {
            let _ = db
                .migrate_entity_canonical_id()
                .and_then(|_| db.mark_entity_migration_done());
        }

        // Auto-reindex FTS if schema migration rebuilt the index. Never on a
        // read-only handle (it can't write the index, and the writer owns it).
        if needs_fts_reindex && !db.storage.is_read_only() {
            if let Some(ref fi) = db.fts_index {
                let mut reindex_count = 0usize;
                let mut reindex_errors = 0usize;
                let tables = db.storage.tables().unwrap_or_default();
                for table in &tables {
                    if let Ok(records) = db.storage.list(table, usize::MAX, 0) {
                        for record in &records {
                            match fi.on_record_insert(record) {
                                Ok(()) => reindex_count += 1,
                                Err(_) => reindex_errors += 1,
                            }
                        }
                    }
                }
                if reindex_errors > 0 {
                    eprintln!(
                        "axil: FTS schema migration reindexed {reindex_count} records, {reindex_errors} errors. Run `axil heal --reindex` to retry."
                    );
                }
            }
        }

        Ok(db)
    }
}

/// true if two table prefixes overlap (one is a prefix of
/// the other, or they're equal). Used by [`AxilBuilder::with_extension`]
/// to reject conflicting Extension registrations at build time.
fn prefix_overlaps(a: &str, b: &str) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Maximum audit log entries before auto-rotation.
const MAX_AUDIT_ENTRIES: usize = 10_000;

/// Maximum slow query log entries.
const MAX_SLOW_QUERIES: usize = 1_000;

/// Main database handle. All operations go through here.
pub struct Axil {
    path: PathBuf,
    storage: Storage,
    plugins: Vec<Box<dyn Engine>>,
    vector_index: Option<Arc<dyn VectorIndex>>,
    embedder: Option<Arc<dyn TextEmbedder>>,
    graph_index: Option<Arc<dyn GraphIndex>>,
    timeseries_index: Option<Arc<dyn TimeSeriesIndex>>,
    fts_index: Option<Arc<dyn SearchIndex>>,
    /// Optional LLM provider for enhanced intelligence.
    llm_provider: Option<Arc<dyn crate::llm::LlmProvider>>,
    /// Session-level LLM usage tracker.
    llm_usage: Arc<crate::llm::LlmUsageTracker>,
    /// Rate limiter for LLM calls.
    llm_rate_limiter: Arc<crate::llm::LlmRateLimiter>,
    metrics: Arc<Metrics>,
    /// Factory for opening named vector spaces on demand (additive, see
    /// [`VectorSpaceFactory`]).
    vector_space_factory: Option<Arc<dyn VectorSpaceFactory>>,
    /// Named vector spaces opened this session, keyed by name. Cached so a
    /// space's companion file is opened at most once per handle (redb allows a
    /// single writable handle per file per process) and repeated writes/reads
    /// reuse the live index.
    vector_space_cache: std::sync::RwLock<std::collections::HashMap<String, Arc<dyn VectorIndex>>>,
    slow_query_threshold_ms: f64,
    audit_enabled: std::sync::atomic::AtomicBool,
    /// Monotonic counter for generating unique log keys within a session.
    log_counter: std::sync::atomic::AtomicU64,
    /// Relevance feedback store.
    feedback_store: crate::feedback::FeedbackStore,
    /// Optional Atlas canonical-ID publisher.
    canonical_publisher: Option<Arc<dyn CanonicalPublisher>>,
    /// Registered Tier-2 Extensions. Behind a lock so plugins that need a live
    /// `Axil` handle (WASM plugins) can register *after* the database is open,
    /// not only at builder time.
    extensions: std::sync::RwLock<Vec<Arc<dyn Extension>>>,
    /// Runtime gate for the durable semantic event log. Off by default even when
    /// the `event-log` feature is compiled in — it is a write-amplifier, so the
    /// caller opts in explicitly via [`Axil::set_event_log_enabled`].
    #[cfg(feature = "event-log")]
    event_log_enabled: std::sync::atomic::AtomicBool,
    /// Monotonic ULID cursor source for the event tape. Same-millisecond events
    /// still sort in commit order.
    #[cfg(feature = "event-log")]
    event_cursor: crate::event_log::EventCursor,
}

impl Axil {
    /// Start building a database at the given path.
    pub fn open(path: impl AsRef<Path>) -> AxilBuilder {
        AxilBuilder {
            path: path.as_ref().to_path_buf(),
            plugins: Vec::new(),
            vector_index: None,
            embedder: None,
            graph_index: None,
            timeseries_index: None,
            fts_index: None,
            llm_provider: None,
            llm_config: crate::llm::LlmConfig::default(),
            canonical_publisher: None,
            vector_space_factory: None,
            extensions: Vec::new(),
            needs_fts_reindex: false,
            read_only: false,
            #[cfg(feature = "encryption")]
            cipher: None,
        }
    }

    /// Tier-2 Extensions registered with [`AxilBuilder::with_extension`].
    ///
    /// Returns an empty slice if no Extensions are registered.
    /// Adapters (CLI, MCP, HTTP, …) iterate this to discover the
    /// CLI surfaces, MCP tools, boot blocks, and per-file recall
    /// contributions each Extension provides.
    pub fn extensions(&self) -> Vec<Arc<dyn Extension>> {
        self.extensions
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Register a Tier-2 Extension on an already-open database.
    ///
    /// Native Extensions register at builder time via
    /// [`AxilBuilder::with_extension`]; this is the post-build path for
    /// Extensions that need a live `Axil` handle to construct — notably WASM
    /// plugins loaded from `.axil/plugins/`, whose host imports call back into
    /// *this* database. Enforces the same disjoint-id / disjoint-prefix
    /// invariants as the builder, but returns an error rather than panicking
    /// since it runs at runtime, not at programmer-controlled build time.
    pub fn register_extension(&self, ext: Arc<dyn Extension>) -> Result<()> {
        // Prefixes reserved by Axil core + built-in engines/extensions. A
        // runtime (WASM) Extension may not declare a prefix overlapping these,
        // even when the owning built-in isn't registered in *this* build — so an
        // untrusted plugin can't claim a core/engine table by declaring its
        // prefix while the legitimate owner is absent. (Built-in Extensions
        // register via the builder, not this path, so they're unaffected.)
        const RESERVED_TABLE_PREFIXES: &[&str] = &[
            // core memory tables
            "decisions",
            "errors",
            "context",
            "sessions",
            "rules",
            // core / engine / built-in-extension internal tables
            "_entities",
            "_entity_aliases",
            "_summaries",
            "_beliefs",
            "_idx_",
            "_scip_",
            "_dep_",
            "_checkpoint_",
            "_recall",
        ];

        let mut exts = self
            .extensions
            .write()
            .map_err(|_| AxilError::plugin("extensions registry lock poisoned"))?;
        let id = ext.id();
        if exts.iter().any(|e| e.id() == id) {
            return Err(AxilError::plugin(format!(
                "extension id `{id}` is already registered"
            )));
        }
        for prefix in ext.table_prefixes() {
            if let Some(reserved) = RESERVED_TABLE_PREFIXES
                .iter()
                .find(|r| prefix_overlaps(prefix, r))
            {
                return Err(AxilError::plugin(format!(
                    "extension `{id}` table prefix `{prefix}` is reserved by Axil core/engines \
                     (overlaps `{reserved}`) — choose a distinct, more specific prefix"
                )));
            }
            for existing in exts.iter() {
                for existing_prefix in existing.table_prefixes() {
                    if prefix_overlaps(prefix, existing_prefix) {
                        return Err(AxilError::plugin(format!(
                            "extension `{id}` table prefix `{prefix}` overlaps with existing \
                             extension `{}` prefix `{existing_prefix}`",
                            existing.id(),
                        )));
                    }
                }
            }
        }
        exts.push(ext);
        Ok(())
    }

    fn insert_record(&self, record: Record) -> Result<Record> {
        self.insert_record_inner(record).map(|(record, _superseded)| record)
    }

    /// Insert a record and report how many existing records it superseded.
    ///
    /// Identical to [`Axil::insert_record`], but returns the auto-supersede
    /// count alongside the stored record so callers that need to account for
    /// side effects (notably [`crate::portable`] import, which must surface how
    /// many local records an import demoted) can see it. Normal inserts discard
    /// the count via the [`Axil::insert_record`] wrapper.
    fn insert_record_inner(&self, mut record: Record) -> Result<(Record, usize)> {
        let table = record.table.clone();
        let _span = crate::otel::span("axil.insert", &[("table", table.clone())]);
        let timer = self.metrics.start_timer(OpType::Insert);
        let mut superseded_count = 0usize;
        // Auto-score importance if not already set.
        if record.data.get("_importance").is_none() && !table.starts_with('_') {
            let score = crate::importance::compute_importance(&record.data);
            if let Some(obj) = record.data.as_object_mut() {
                obj.insert("_importance".to_string(), serde_json::json!(score));
            }
        }
        if !table.starts_with('_') {
            let has_consent = record
                .metadata
                .as_ref()
                .and_then(|m| m.get("consent"))
                .is_some();
            if !has_consent {
                let _ = self.apply_consent_defaults(&mut record);
            }
        }
        self.storage.insert(&record)?;
        self.audit("insert", &record.id, &table);
        #[cfg(feature = "event-log")]
        self.capture_semantic_event("insert", &record);
        self.run_insert_hooks(&record)?;

        self.publish_canonical_for_record(&record);

        // Auto-embed and auto-supersede: embed text, then reuse the vector
        // for supersede check to avoid redundant embedding.
        if !table.starts_with('_') && self.has_vector_index() && self.embedder.is_some() {
            let text = crate::util::searchable_text(&record.data);
            if !text.is_empty() && text.len() > 5 {
                if let Ok(vec) = self.require_embedder().and_then(|e| e.embed(&text)) {
                    let indexed = self
                        .require_vector_index()
                        .and_then(|vi| vi.add(record.id.clone(), &vec))
                        .is_ok();
                    // Only supersede if the new record was successfully indexed
                    if indexed {
                        match self.auto_supersede_with_vector(&record, &vec) {
                            Ok(n) => superseded_count = n,
                            Err(e) => {
                                eprintln!("warning: auto-supersede failed for {}: {e}", record.id)
                            }
                        }
                    }
                }
            }
        }

        if !table.starts_with('_') {
            self.sync_recall_chunks_for_record(&record)?;
        }

        // Auto-link: if graph is available, extract entities and create edges.
        if !table.starts_with('_') && self.has_graph_index() {
            if let Err(e) = self.auto_link(&record.id, None) {
                eprintln!("warning: auto-link failed for {}: {e}", record.id);
            }
        }

        // Sync the code_refs reverse index so callers (CLI, MCP, library)
        // automatically maintain `_idx_code_refs` without per-call hooks.
        // Best-effort: a failure here doesn't roll back the storage write.
        if let Err(e) =
            crate::code_refs::sync_for_record(self, &record, crate::code_refs::SyncMode::Insert)
        {
            eprintln!(
                "warning: code_refs index sync failed for {}: {e}",
                record.id
            );
        }

        let elapsed = timer.finish();
        crate::otel::record_operation("insert", &table, elapsed);
        Ok((record, superseded_count))
    }

    /// Whether a record would be auto-embedded into the vector index on insert.
    ///
    /// Mirrors the auto-embed gate in [`Axil::insert_record`] **exactly**
    /// (non-internal table, `searchable_text` non-empty, length > 5). The
    /// reverse-orphan reconciliation in [`Axil::count_missing_embeddings`] /
    /// [`Axil::reembed_missing`] depends on this matching the insert gate, or it
    /// false-positives on records that were never meant to be embedded. Kept next
    /// to the insert path so any drift is immediately visible.
    ///
    /// `pub(crate)` so [`crate::portable`] import can reuse the exact same gate
    /// when deciding which imported records to verify as embedded, rather than
    /// keeping a second copy that could silently drift.
    pub(crate) fn is_embeddable(record: &Record) -> bool {
        if record.table.starts_with('_') {
            return false;
        }
        let text = crate::util::searchable_text(&record.data);
        !text.is_empty() && text.len() > 5
    }

    /// The table-prefix half of the FTS insert gate: non-internal tables only.
    ///
    /// The FTS engine ALSO skips records with no extractable text (see
    /// [`SearchIndex::would_index`]), so reverse-orphan reconciliation must
    /// combine this with `would_index` — otherwise a text-less record (e.g.
    /// `{"n": 5}`) is flagged "missing an FTS document" forever and a re-index
    /// is a phantom no-op.
    fn is_fts_indexable(record: &Record) -> bool {
        !record.table.starts_with('_')
    }

    /// Count records that should have a vector embedding but don't.
    ///
    /// Takes an already-scanned record slice so the caller can share one full
    /// scan across reconciliation checks. Returns 0 unless both a vector index
    /// and an embedder are configured — a user who only ever calls `add_vector`
    /// manually (no embedder) must not be flagged as missing embeddings.
    fn count_missing_embeddings(&self, records: &[Record]) -> usize {
        if !self.has_vector_index() || self.embedder.is_none() {
            return 0;
        }
        let Some(ref vi) = self.vector_index else {
            return 0;
        };
        let indexed: std::collections::HashSet<RecordId> =
            vi.all_ids().unwrap_or_default().into_iter().collect();
        records
            .iter()
            .filter(|r| Self::is_embeddable(r) && !indexed.contains(&r.id))
            .count()
    }

    /// Count records that should have an FTS document but don't.
    ///
    /// Takes an already-scanned record slice so the caller can share one full
    /// scan across reconciliation checks. Returns 0 unless an FTS index is
    /// configured.
    fn count_missing_fts(&self, records: &[Record]) -> usize {
        let Some(ref fi) = self.fts_index else {
            return 0;
        };
        let indexed: std::collections::HashSet<RecordId> =
            fi.all_indexed_ids().unwrap_or_default().into_iter().collect();
        records
            .iter()
            .filter(|r| Self::is_fts_indexable(r) && fi.would_index(r) && !indexed.contains(&r.id))
            .count()
    }

    /// Re-embed and re-index records that committed to storage but are missing
    /// their vector embedding and/or FTS document.
    ///
    /// This closes the reverse-orphan gap: [`Axil::insert_record`] commits the
    /// core record first and then fans out to the secondary indexes, swallowing
    /// embed/index failures. A torn insert can therefore leave a stored memory
    /// permanently invisible to recall. `reembed_missing` reconciles the live
    /// records against the vector and FTS indexes and regenerates the missing
    /// entries.
    ///
    /// Returns `(embeddings_restored, fts_docs_restored)`. Best-effort per
    /// record — a single failure does not abort the sweep. A no-op (returns
    /// `(0, 0)`) when no embedder/vector index/FTS index is configured.
    pub fn reembed_missing(&self) -> Result<(usize, usize)> {
        let records = self.storage.scan_all_records()?;

        let mut embeddings_restored = 0usize;
        if self.has_vector_index() && self.embedder.is_some() {
            if let Some(ref vi) = self.vector_index {
                let indexed: std::collections::HashSet<RecordId> =
                    vi.all_ids().unwrap_or_default().into_iter().collect();
                for record in &records {
                    if !Self::is_embeddable(record) || indexed.contains(&record.id) {
                        continue;
                    }
                    let text = crate::util::searchable_text(&record.data);
                    if self.embed_text(&record.id, &text).is_ok() {
                        embeddings_restored += 1;
                    }
                }
            }
        }

        let mut fts_restored = 0usize;
        if let Some(ref fi) = self.fts_index {
            let indexed: std::collections::HashSet<RecordId> =
                fi.all_indexed_ids().unwrap_or_default().into_iter().collect();
            for record in &records {
                if !Self::is_fts_indexable(record)
                    || !fi.would_index(record)
                    || indexed.contains(&record.id)
                {
                    continue;
                }
                if fi.on_record_insert(record).is_ok() {
                    fts_restored += 1;
                }
            }
        }

        if embeddings_restored > 0 || fts_restored > 0 {
            self.audit_heal_action(
                "reembed_missing",
                &format!(
                    "restored {embeddings_restored} missing embeddings, {fts_restored} missing FTS docs"
                ),
            );
        }

        Ok((embeddings_restored, fts_restored))
    }

    /// Insert a new record into the given table.
    ///
    /// The record is always committed to storage first. Engine hooks run
    /// after the commit. Generic plugin failures are swallowed (storage is
    /// authoritative), but **vector and timeseries index failures are
    /// propagated** so the caller can detect indexing issues (e.g. missing
    /// embeddings or timeseries capacity exceeded).
    ///
    /// On hook failure the record IS persisted — the error signals that the
    /// secondary index was not updated, not that the insert was rolled back.
    pub fn insert(&self, table: &str, data: Value) -> Result<Record> {
        self.insert_record(Record::new(table, data))
    }

    /// Insert a new record with an explicit creation timestamp.
    ///
    /// Useful for imported histories and benchmark corpora where chronological
    /// ranking should follow source timestamps rather than ingest time.
    pub fn insert_at(
        &self,
        table: &str,
        data: Value,
        created_at: chrono::DateTime<chrono::Utc>,
    ) -> Result<Record> {
        let mut record = Record::new(table, data);
        record.created_at = created_at;
        record.updated_at = created_at;
        self.insert_record(record)
    }

    /// Insert a fully-formed record, preserving its id, timestamps, and metadata.
    ///
    /// Routes through the same internal insert path as [`Axil::insert`], so every
    /// engine still fires (embedding, FTS, graph auto-link, `code_refs` sync).
    /// Used by [`crate::portable`] import so record ids survive the round trip:
    /// checkpoint `references[]` and `code_refs` point at ids, and remapping them
    /// would break those pointers. If a record with the same id already exists it
    /// is overwritten (upsert), matching the storage layer's id-keyed semantics.
    pub fn insert_preserving(&self, record: Record) -> Result<Record> {
        self.insert_record(record)
    }

    /// Like [`Axil::insert_preserving`], but also returns how many existing
    /// records the insert auto-superseded.
    ///
    /// Used by [`crate::portable`] import so the `ImportReport` can honestly
    /// surface how many local records an import demoted (auto-supersede fires
    /// on the import path just as it does on a normal insert). The recency
    /// guard in [`Axil::auto_supersede_with_vector`] means an import that
    /// preserves an *older* `created_at` never supersedes a fresher local
    /// record, so a non-zero count here always reflects a genuine, newer
    /// incoming record replacing an older near-duplicate.
    pub(crate) fn insert_preserving_counted(&self, record: Record) -> Result<(Record, usize)> {
        self.insert_record_inner(record)
    }

    /// Insert multiple records in a single transaction.
    ///
    /// Much faster than calling `insert()` in a loop because all records share
    /// one storage transaction. Vector embedding is micro-batched and indexed
    /// through a single `add_batch` per micro-batch (one `.vec` commit), and
    /// FTS indexing is batched too — so the whole batch costs a handful of
    /// fsyncs rather than one per record.
    pub fn insert_batch(&self, table: &str, data_items: Vec<Value>) -> Result<Vec<Record>> {
        let timer = self.metrics.start_timer(OpType::Insert);
        let skip_internal = !table.starts_with('_');
        let records: Vec<Record> = data_items
            .into_iter()
            .map(|mut data| {
                if skip_internal && data.get("_importance").is_none() {
                    let score = crate::importance::compute_importance(&data);
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("_importance".to_string(), serde_json::json!(score));
                    }
                }
                Record::new(table, data)
            })
            .collect();

        self.insert_batch_records(records, skip_internal, timer)
    }

    /// Insert multiple records with explicit creation timestamps.
    ///
    /// Each `(data, created_at)` pair produces one record whose timestamps
    /// are set to the provided value instead of "now".
    pub fn insert_batch_at(
        &self,
        table: &str,
        items: Vec<(Value, chrono::DateTime<chrono::Utc>)>,
    ) -> Result<Vec<Record>> {
        let timer = self.metrics.start_timer(OpType::Insert);
        let skip_internal = !table.starts_with('_');
        let records: Vec<Record> = items
            .into_iter()
            .map(|(mut data, ts)| {
                if skip_internal && data.get("_importance").is_none() {
                    let score = crate::importance::compute_importance(&data);
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("_importance".to_string(), serde_json::json!(score));
                    }
                }
                let mut record = Record::new(table, data);
                record.created_at = ts;
                record.updated_at = ts;
                record
            })
            .collect();

        self.insert_batch_records(records, skip_internal, timer)
    }

    /// Bulk insert optimized for historical/imported data.
    ///
    /// Skips recall-chunk creation and auto-link, keeping only storage +
    /// batch embedding + FTS indexing. Use this when ingesting large
    /// corpora where per-record chunking would dominate runtime.
    /// Callers can run `sync_recall_chunks` on individual records later
    /// if chunked recall is needed.
    pub fn insert_batch_raw(
        &self,
        table: &str,
        items: Vec<(Value, chrono::DateTime<chrono::Utc>)>,
    ) -> Result<Vec<Record>> {
        let timer = self.metrics.start_timer(OpType::Insert);
        let skip_internal = !table.starts_with('_');
        let records: Vec<Record> = items
            .into_iter()
            .map(|(mut data, ts)| {
                if skip_internal && data.get("_importance").is_none() {
                    let score = crate::importance::compute_importance(&data);
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("_importance".to_string(), serde_json::json!(score));
                    }
                }
                let mut record = Record::new(table, data);
                record.created_at = ts;
                record.updated_at = ts;
                record
            })
            .collect();

        self.storage.insert_batch(&records)?;

        let has_embedder = self.has_vector_index() && self.embedder.is_some();

        // Batch embed all records.
        if skip_internal && has_embedder {
            const MICRO_BATCH: usize = 32;
            let texts: Vec<String> = records
                .iter()
                .map(|r| crate::util::searchable_text(&r.data))
                .collect();
            let embeddable: Vec<(usize, &str)> = texts
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.is_empty() && t.len() > 5)
                .map(|(i, t)| (i, t.as_str()))
                .collect();

            if !embeddable.is_empty() {
                if let (Ok(embedder), Ok(vi)) =
                    (self.require_embedder(), self.require_vector_index())
                {
                    for chunk in embeddable.chunks(MICRO_BATCH) {
                        let batch_texts: Vec<&str> = chunk.iter().map(|(_, t)| *t).collect();
                        let vectors = embedder.embed_batch(&batch_texts).or_else(|_| {
                            batch_texts
                                .iter()
                                .map(|t| embedder.embed(t))
                                .collect::<Result<Vec<_>>>()
                        });
                        if let Ok(vecs) = vectors {
                            let batch: Vec<(RecordId, &[f32])> = chunk
                                .iter()
                                .enumerate()
                                .filter(|(vec_idx, _)| *vec_idx < vecs.len())
                                .map(|(vec_idx, &(rec_idx, _))| {
                                    (records[rec_idx].id.clone(), vecs[vec_idx].as_slice())
                                })
                                .collect();
                            let _ = vi.add_batch(&batch);
                        }
                    }
                }
            }
        }

        // Audit only — no recall chunks in raw batch mode.
        // Callers index FTS fields explicitly as needed.
        for record in &records {
            self.audit("insert", &record.id, table);
        }

        timer.finish();
        Ok(records)
    }

    /// Shared implementation for batch insert variants.
    ///
    /// Uses batch embedding (single ONNX call) instead of per-record embed
    /// for 5-10x faster vector indexing on bulk ingestion.
    fn insert_batch_records(
        &self,
        mut records: Vec<Record>,
        skip_internal: bool,
        timer: crate::metrics::Timer,
    ) -> Result<Vec<Record>> {
        // Per-table consent defaults must apply to batch writes too —
        // without this, bulk imports read back as implicit `private`
        // regardless of `axil consent default`, breaking cross-project
        // recall for bulk-ingested records.
        if skip_internal {
            for record in records.iter_mut() {
                let has_consent = record
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("consent"))
                    .is_some();
                if !has_consent {
                    let _ = self.apply_consent_defaults(record);
                }
            }
        }
        self.storage.insert_batch(&records)?;

        let has_embedder = self.has_vector_index() && self.embedder.is_some();
        let has_graph = self.has_graph_index();

        // Batch embed: collect all texts, run micro-batched ONNX inference, then index.
        // Micro-batching (8 texts per call) avoids the quadratic padding cost of
        // one giant batch while still amortising the ONNX session lock overhead.
        if skip_internal && has_embedder {
            const MICRO_BATCH: usize = 32;

            let texts: Vec<String> = records
                .iter()
                .map(|r| crate::util::searchable_text(&r.data))
                .collect();
            let embeddable: Vec<(usize, &str)> = texts
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.is_empty() && t.len() > 5)
                .map(|(i, t)| (i, t.as_str()))
                .collect();

            if !embeddable.is_empty() {
                if let (Ok(embedder), Ok(vi)) =
                    (self.require_embedder(), self.require_vector_index())
                {
                    for chunk in embeddable.chunks(MICRO_BATCH) {
                        let batch_texts: Vec<&str> = chunk.iter().map(|(_, t)| *t).collect();
                        let vectors = embedder.embed_batch(&batch_texts).or_else(|_| {
                            batch_texts
                                .iter()
                                .map(|t| embedder.embed(t))
                                .collect::<Result<Vec<_>>>()
                        });
                        if let Ok(vecs) = vectors {
                            let batch: Vec<(RecordId, &[f32])> = chunk
                                .iter()
                                .enumerate()
                                .filter(|(vec_idx, _)| *vec_idx < vecs.len())
                                .map(|(vec_idx, &(rec_idx, _))| {
                                    (records[rec_idx].id.clone(), vecs[vec_idx].as_slice())
                                })
                                .collect();
                            let _ = vi.add_batch(&batch);
                        }
                    }
                }
            }
        }

        for record in &records {
            self.audit("insert", &record.id, &record.table);
            // Skip run_insert_hooks — batch embedding above already indexed
            // vectors. Timeseries runs in this loop; FTS is batched below.
            // Running hooks here would double-embed every record through
            // VectorEngine::on_record_insert.
            if !record.table.starts_with('_') {
                for plugin in &self.plugins {
                    let _ = plugin.on_record_insert(record);
                }
                if let Some(ref tsi) = self.timeseries_index {
                    let _ = tsi.on_record_insert(record);
                }
            }
        }

        // FTS: index the whole batch through one commit (see
        // `SearchIndex::index_records_batch`) rather than one per record.
        if skip_internal {
            if let Some(ref fi) = self.fts_index {
                let _ = fi.index_records_batch(&records);
            }
        }

        if skip_internal {
            // Sync recall chunks for all records in one pass — flattens chunks
            // across the whole batch, embeds them through a single micro-batched
            // loop, then fans out. Avoids the quadratic cost of the previous
            // per-record path which called `embed_batch` once per record and
            // paid the ONNX session-mutex latency N times.
            let _ = self.batch_sync_recall_chunks(&records);

            if has_graph {
                for record in &records {
                    if let Err(e) = self.auto_link(&record.id, None) {
                        eprintln!("warning: auto-link failed for {}: {e}", record.id);
                    }
                }
            }
        }

        timer.finish();
        Ok(records)
    }

    /// Run all plugin insert hooks for a record after storage commit.
    ///
    /// Generic plugin failures are swallowed (storage is authoritative), but
    /// vector, timeseries, and FTS index failures are propagated so callers
    /// can detect indexing issues.
    fn run_insert_hooks(&self, record: &Record) -> Result<()> {
        let internal = record.table.starts_with('_');

        // Generic plugins, the vector index and the FTS index hold
        // user-facing content only — internal tables must not pollute
        // semantic or full-text search results.
        if !internal {
            for plugin in &self.plugins {
                let _ = plugin.on_record_insert(record);
            }
            if let Some(ref vi) = self.vector_index {
                vi.on_record_insert(record)?;
            }
            if let Some(ref fi) = self.fts_index {
                fi.on_record_insert(record)?;
            }
        }

        // The timeseries index is a time index, not a search index:
        // `_summaries` rows (created by `downsample`) are themselves
        // time-series data and must be reachable via since()/timeline().
        // Other internal tables stay out to avoid time-query noise.
        if !internal || record.table == "_summaries" {
            if let Some(ref tsi) = self.timeseries_index {
                tsi.on_record_insert(record)?;
            }
        }
        Ok(())
    }

    /// Get a record by ID.
    pub fn get(&self, id: &RecordId) -> Result<Option<Record>> {
        let timer = self.metrics.start_timer(OpType::Get);
        let result = self.storage.get(id);
        let elapsed = timer.finish();
        crate::otel::record_operation("get", "", elapsed);
        result
    }

    /// Get a record by ID and bump its activation level.
    ///
    /// This is the preferred method for agent recall paths where access
    /// frequency should influence future ranking. The activation bump is
    /// lazy (applied on read, not via background process).
    pub fn get_and_activate(
        &self,
        id: &RecordId,
        config: &crate::activation::ActivationConfig,
    ) -> Result<Option<Record>> {
        let record = self.get(id)?;
        if let Some(ref r) = record {
            self.bump_activation(r, config)?;
        }
        Ok(record)
    }

    /// Bump a record's activation level and update `_last_accessed`.
    ///
    /// This writes the updated activation back to storage. The decay is
    /// computed lazily at read time, so only the raw activation value is stored.
    pub fn bump_activation(
        &self,
        record: &Record,
        config: &crate::activation::ActivationConfig,
    ) -> Result<()> {
        let now = chrono::Utc::now();
        let (new_activation, timestamp) = crate::activation::compute_bump(record, &now, config);

        let mut data = record.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("_activation".to_string(), serde_json::json!(new_activation));
            obj.insert("_last_accessed".to_string(), serde_json::json!(timestamp));
        }
        self.storage.update(&record.id, data)?;
        Ok(())
    }

    /// Delete a record by ID.
    ///
    /// Engine hooks run after the storage write commits. See `insert` for
    /// the rationale on swallowing plugin errors.
    pub fn delete(&self, id: &RecordId) -> Result<bool> {
        let _span = crate::otel::span("axil.delete", &[]);
        let timer = self.metrics.start_timer(OpType::Delete);

        // Capture the record before deleting for audit trail and companion cleanup.
        let existing_record = self.storage.get(id)?;
        let table_name = existing_record
            .as_ref()
            .map(|r| r.table.clone())
            .unwrap_or_default();

        let deleted = self.storage.delete(id)?;

        if deleted {
            if table_name != RECALL_CHUNKS_TABLE {
                if let Some(ref record) = existing_record {
                    self.delete_recall_chunks_for_source(record)?;
                }
            }
            for plugin in &self.plugins {
                if let Err(_e) = plugin.on_record_delete(id) {
                    // Storage already committed — same rationale as insert.
                }
            }

            if let Some(ref vi) = self.vector_index {
                vi.on_record_delete(id)?;
            }

            // Cascade-delete graph edges referencing this record.
            if let Some(ref gi) = self.graph_index {
                gi.on_record_delete(id)?;
            }

            // Remove from time index.
            if let Some(ref tsi) = self.timeseries_index {
                tsi.on_record_delete(id)?;
            }

            if let Some(ref fi) = self.fts_index {
                fi.on_record_delete(id)?;
            }

            // Drop reverse-index rows pointing at the deleted record so
            // related-memory recall doesn't surface a tombstone.
            if !table_name.starts_with('_') {
                if let Err(e) = crate::code_refs::drop_for_record(self, id) {
                    eprintln!("warning: code_refs index cleanup failed for {id}: {e}");
                }
            }

            // Doubt beliefs sourced from deleted entity facts.
            if table_name == "_entities" {
                if let Ok(beliefs) = self.storage.list("_beliefs", usize::MAX, 0) {
                    let deleted_id_str = id.to_string();
                    for belief in &beliefs {
                        // Beliefs auto-generated from entities reference the entity name.
                        // If the source entity fact is deleted, mark the belief as doubted.
                        let is_consolidated = belief.data.get("source").and_then(|v| v.as_str())
                            == Some("consolidated");
                        let already_doubted = belief
                            .data
                            .get("doubted")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        if is_consolidated && !already_doubted {
                            let mut data = belief.data.clone();
                            if let Some(obj) = data.as_object_mut() {
                                obj.insert("doubted".to_string(), serde_json::json!(true));
                                obj.insert("confidence".to_string(), serde_json::json!(0.3));
                                obj.insert(
                                    "_doubt_reason".to_string(),
                                    serde_json::json!(format!(
                                        "source entity fact {deleted_id_str} deleted"
                                    )),
                                );
                            }
                            let _ = self.storage.update(&belief.id, data);
                        }
                    }
                }
            }

            self.audit("delete", id, &table_name);
        }

        let elapsed = timer.finish();
        crate::otel::record_operation("delete", &table_name, elapsed);
        Ok(deleted)
    }

    /// Update a record's data.
    ///
    /// Engine hooks run after the storage write commits. See `insert` for
    /// the rationale on swallowing generic plugin errors.
    pub fn update(&self, id: &RecordId, data: Value) -> Result<Record> {
        let timer = self.metrics.start_timer(OpType::Update);
        let record = self.storage.update(id, data)?;

        for plugin in &self.plugins {
            if let Err(_e) = plugin.on_record_update(&record) {
                // Storage already committed — same rationale as insert.
            }
        }

        if let Some(ref tsi) = self.timeseries_index {
            tsi.on_record_update(&record)?;
        }

        if let Some(ref fi) = self.fts_index {
            fi.on_record_update(&record)?;
        }

        // Auto re-embed: if text changed and embedder is available, update the vector.
        if !record.table.starts_with('_') && self.has_vector_index() && self.embedder.is_some() {
            let text = crate::util::searchable_text(&record.data);
            if !text.is_empty() && text.len() > 5 {
                let _ = self.embed_text(id, &text);
            }
        }

        if !record.table.starts_with('_') {
            self.sync_recall_chunks_for_record(&record)?;
        }

        if let Err(e) =
            crate::code_refs::sync_for_record(self, &record, crate::code_refs::SyncMode::Update)
        {
            eprintln!(
                "warning: code_refs index sync failed for {}: {e}",
                record.id
            );
        }

        self.audit("update", id, &record.table);
        #[cfg(feature = "event-log")]
        self.capture_semantic_event("update", &record);
        timer.finish();
        Ok(record)
    }

    /// List records in a table.
    pub fn list(&self, table: &str) -> Result<Vec<Record>> {
        self.storage.list(table, usize::MAX, 0)
    }

    /// List records from a table with a limit (avoids loading entire table).
    pub fn list_with_limit(&self, table: &str, limit: usize) -> Result<Vec<Record>> {
        self.storage.list(table, limit, 0)
    }

    /// Start building a query.
    pub fn query(&self) -> crate::query::QueryBuilder<'_> {
        let mut qb = crate::query::QueryBuilder::new(
            &self.storage,
            self.vector_index.as_ref().map(|a| a.as_ref()),
            self.embedder.as_ref().map(|a| a.as_ref()),
        );
        if let Some(ref gi) = self.graph_index {
            qb = qb.with_graph(gi.as_ref());
        }
        if let Some(ref tsi) = self.timeseries_index {
            qb = qb.with_timeseries(tsi.as_ref());
        }
        if let Some(ref fi) = self.fts_index {
            qb = qb.with_fts(fi.as_ref());
        }
        qb
    }

    /// List all table names.
    pub fn tables(&self) -> Result<Vec<String>> {
        self.storage.tables()
    }

    /// Total number of records.
    pub fn total_records(&self) -> Result<usize> {
        self.storage.total_records()
    }

    /// Count records in a specific table.
    pub fn count(&self, table: &str) -> Result<usize> {
        self.storage.count(table)
    }

    /// List all table names with their record counts (single transaction).
    pub fn tables_with_counts(&self) -> Result<Vec<(String, usize)>> {
        self.storage.tables_with_counts()
    }

    // ── Vector API ──────────────────────────────────────────────────

    fn require_vector_index(&self) -> Result<&dyn VectorIndex> {
        self.vector_index
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| AxilError::plugin("no vector index configured"))
    }

    fn require_embedder(&self) -> Result<&dyn TextEmbedder> {
        self.embedder
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| AxilError::plugin("no embedder configured — use with_embedder()"))
    }

    /// Embed a record field and store the resulting vector.
    ///
    /// Extracts the text value of the given field from the record's JSON data,
    /// embeds it using the configured model, and stores the vector in the index.
    pub fn embed_field(&self, id: &RecordId, field: &str) -> Result<()> {
        let _span = crate::otel::span("axil.embed", &[("field", field.to_string())]);
        let vi = self.require_vector_index()?;
        let embedder = self.require_embedder()?;

        let record = self
            .storage
            .get(id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {id}")))?;

        let text = record
            .data
            .get(field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AxilError::InvalidQuery(format!("field '{field}' not found or not a string"))
            })?;

        let vector = embedder.embed(text)?;
        vi.add(id.clone(), &vector)?;
        Ok(())
    }

    /// Embed many `(record, text)` pairs and store the vectors in one pass.
    ///
    /// Micro-batches the ONNX inference and indexes every vector through a
    /// single `add_batch` per micro-batch (one `.vec` commit), so ingesting N
    /// chunks costs a handful of fsyncs instead of one per chunk. Callers that
    /// already hold the text (e.g. dep-docs chunk ingest) should prefer this
    /// over a loop of [`Axil::embed_text`]. Records are assumed to exist —
    /// pass ids from records you just inserted. A no-op when no vector index or
    /// embedder is configured.
    pub fn embed_fields_batch(&self, entries: &[(&RecordId, &str)]) -> Result<()> {
        if entries.is_empty() || !self.has_vector_index() || self.embedder.is_none() {
            return Ok(());
        }
        let embedder = self.require_embedder()?;
        let vi = self.require_vector_index()?;

        const MICRO_BATCH: usize = 32;
        for chunk in entries.chunks(MICRO_BATCH) {
            let texts: Vec<&str> = chunk.iter().map(|(_, t)| *t).collect();
            let vectors = embedder.embed_batch(&texts).or_else(|_| {
                texts
                    .iter()
                    .map(|t| embedder.embed(t))
                    .collect::<Result<Vec<_>>>()
            })?;
            if vectors.len() != chunk.len() {
                return Err(AxilError::plugin(format!(
                    "embedder returned {} vectors for {} texts — refusing to \
                     silently drop the remainder",
                    vectors.len(),
                    chunk.len()
                )));
            }
            let batch: Vec<(RecordId, &[f32])> = chunk
                .iter()
                .zip(&vectors)
                .map(|((id, _), v)| ((*id).clone(), v.as_slice()))
                .collect();
            vi.add_batch(&batch)?;
        }
        Ok(())
    }

    /// FTS-index the same named field across many records in one commit.
    ///
    /// Thin wrapper over the buffered `index_field_batch` plugin hook — turns N
    /// per-chunk commits into one. A no-op when no FTS index is configured.
    pub fn index_text_batch(&self, field: &str, entries: &[(&RecordId, &str)]) -> Result<()> {
        if entries.is_empty() || !self.has_fts_index() {
            return Ok(());
        }
        self.require_fts_index()?.index_field_batch(field, entries)
    }

    /// Embed arbitrary text and associate the vector with a record.
    ///
    /// Verifies the record exists before persisting the vector.
    pub fn embed_text(&self, id: &RecordId, text: &str) -> Result<()> {
        if self.storage.get(id)?.is_none() {
            return Err(AxilError::NotFound(format!("record {id}")));
        }
        let vi = self.require_vector_index()?;
        let embedder = self.require_embedder()?;
        let vector = embedder.embed(text)?;
        vi.add(id.clone(), &vector)?;
        Ok(())
    }

    /// Embed arbitrary text into a vector using the configured embedder.
    ///
    /// Unlike [`Axil::embed_text`] (which embeds *and stores* a record's field
    /// in the vector index), this returns the raw embedding for `text` without
    /// touching any record — used by host callers that need an embedding on its
    /// own (e.g. a WASM plugin's `embed-text` import). Errors if no embedder is
    /// configured.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.require_embedder()?.embed(text)
    }

    /// Semantic search: embed text and find similar records.
    pub fn similar_to(&self, text: &str, top_k: usize) -> Result<Vec<(Record, f32)>> {
        let embedder = self.require_embedder()?;
        let vector = embedder.embed(text)?;
        self.similar_to_vector(&vector, top_k)
    }

    /// Raw vector search: find records with similar vectors.
    pub fn similar_to_vector(&self, vector: &[f32], top_k: usize) -> Result<Vec<(Record, f32)>> {
        let _span = crate::otel::span(
            "axil.vector_search",
            &[
                ("top_k", top_k.to_string()),
                ("dimensions", vector.len().to_string()),
            ],
        );
        let timer = self.metrics.start_timer(OpType::VectorSearch);
        let vi = self.require_vector_index()?;
        let results = vi.search(vector, top_k)?;
        let records = self.resolve_scored_public(results)?;
        let elapsed = timer.finish();
        crate::otel::record_operation("vector_search", "", elapsed);
        Ok(records)
    }

    /// Add a pre-computed vector for a record.
    ///
    /// Verifies the record exists before persisting the vector to avoid
    /// orphaned index entries.
    pub fn add_vector(&self, id: &RecordId, vector: &[f32]) -> Result<()> {
        if self.storage.get(id)?.is_none() {
            return Err(AxilError::NotFound(format!("record {id}")));
        }
        self.require_vector_index()?.add(id.clone(), vector)
    }

    /// Check whether a vector index is configured.
    pub fn has_vector_index(&self) -> bool {
        self.vector_index.is_some()
    }

    /// Fetch a record's stored vector from the default vector index, if present.
    ///
    /// Returns `Ok(None)` when no vector index is configured or the record has
    /// no stored vector. Used by `similar --id` to search with a record's own
    /// fingerprint.
    pub fn get_vector(&self, id: &RecordId) -> Result<Option<Vec<f32>>> {
        match self.vector_index.as_ref() {
            Some(vi) => vi.get_vector(id),
            None => Ok(None),
        }
    }

    // ── Named vector spaces (additive) ──────────────────────────────
    //
    // A named space is an independent HNSW index in a companion file
    // (`<db>.axil.vec.<name>`) with its own stored dimension, so a raw
    // strategy fingerprint never collides with the 384-dim text-embedding
    // index in the default space. These methods are entirely additive: they
    // are the *only* callers of the space factory + cache, and the default
    // vector path ([`Axil::add_vector`] / [`Axil::similar_to_vector`]) is
    // untouched.

    /// Open (or create when `dim` is `Some`) a named vector space, caching the
    /// live index so the companion file is opened at most once per handle.
    fn open_or_create_space(&self, space: &str, dim: Option<usize>) -> Result<Arc<dyn VectorIndex>> {
        validate_space_name(space)?;
        {
            let cache = self
                .vector_space_cache
                .read()
                .map_err(|_| AxilError::plugin("vector space cache lock poisoned"))?;
            if let Some(vi) = cache.get(space) {
                return Ok(vi.clone());
            }
        }
        let factory = self.vector_space_factory.as_ref().ok_or_else(|| {
            AxilError::plugin(
                "no vector-space factory registered — named spaces require the vector engine",
            )
        })?;
        // Hold the write lock across the open so two callers can't both create
        // the same companion file (redb allows one writable handle per file).
        let mut cache = self
            .vector_space_cache
            .write()
            .map_err(|_| AxilError::plugin("vector space cache lock poisoned"))?;
        if let Some(vi) = cache.get(space) {
            return Ok(vi.clone());
        }
        let vi = factory.open_space(&self.path, space, dim)?;
        cache.insert(space.to_string(), vi.clone());
        Ok(vi)
    }

    /// Add a pre-computed vector for a record into a named vector space.
    ///
    /// The space is created lazily on first write, binding the supplied
    /// vector's length as its dimension; later writes with a different length
    /// error with the store's dimension-mismatch message. Verifies the record
    /// exists before persisting to avoid orphaned index entries. Space names
    /// are validated `[a-z0-9_-]{1,32}`.
    pub fn add_vector_in(&self, space: &str, id: &RecordId, vector: &[f32]) -> Result<()> {
        if self.storage.get(id)?.is_none() {
            return Err(AxilError::NotFound(format!("record {id}")));
        }
        let vi = self.open_or_create_space(space, Some(vector.len()))?;
        // Guard the cached-engine path: `open_or_create_space` only validates
        // dimension when it opens the file, so a space already open this session
        // is checked here before we persist a wrong-length vector.
        if vi.dimensions() != vector.len() {
            return Err(AxilError::plugin(format!(
                "dimension mismatch: vector space '{space}' was created with {} dimensions, \
                 but {} supplied",
                vi.dimensions(),
                vector.len()
            )));
        }
        vi.add(id.clone(), vector)
    }

    /// Search a named vector space for records with similar vectors.
    ///
    /// Errors if the space does not exist. Results are resolved to records
    /// directly by id (no recall-core chunk-proxy rewriting), so raw
    /// fingerprints round-trip predictably.
    pub fn similar_in(
        &self,
        space: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<(Record, f32)>> {
        let vi = self.open_or_create_space(space, None)?;
        // Name the space in the error instead of letting the engine's
        // lower-level dimension check fire with less context.
        if vi.dimensions() != vector.len() {
            return Err(AxilError::plugin(format!(
                "dimension mismatch: vector space '{space}' holds {}-dimensional \
                 vectors, but the query has {}",
                vi.dimensions(),
                vector.len()
            )));
        }
        let results = vi.search(vector, top_k)?;
        let mut records = Vec::with_capacity(results.len());
        for (id, score) in results {
            if let Some(record) = self.storage.get(&id)? {
                records.push((record, score));
            }
        }
        Ok(records)
    }

    /// Fetch a record's stored vector from a named vector space, if present.
    pub fn get_vector_in(&self, space: &str, id: &RecordId) -> Result<Option<Vec<f32>>> {
        let vi = self.open_or_create_space(space, None)?;
        vi.get_vector(id)
    }

    /// List the named vector spaces on disk with each space's dimension and
    /// vector count, sorted by name.
    ///
    /// Returns an empty list when no space factory is registered. Spaces opened
    /// this session answer from the live engine; the rest are probed read-only
    /// via [`VectorSpaceFactory::space_meta`] — no writable handle is taken, so
    /// a listing never contends with concurrent vector writes.
    pub fn vector_spaces(&self) -> Result<Vec<VectorSpaceInfo>> {
        let Some(factory) = self.vector_space_factory.as_ref() else {
            return Ok(Vec::new());
        };
        let names = factory.space_names(&self.path)?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let cached = {
                let cache = self
                    .vector_space_cache
                    .read()
                    .map_err(|_| AxilError::plugin("vector space cache lock poisoned"))?;
                cache.get(&name).cloned()
            };
            let (dimensions, count) = match cached {
                Some(vi) => (vi.dimensions(), vi.count()),
                None => factory.space_meta(&self.path, &name)?,
            };
            out.push(VectorSpaceInfo {
                name,
                dimensions,
                count,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Check whether an embedder (text-to-vector) is configured.
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Whether a stored vector exists for `id` in the vector index.
    ///
    /// Relies on the backend's [`VectorIndex::get_vector`]; a backend without
    /// direct retrieval reports `false`. Auto-embedding on insert is
    /// best-effort (an embedder failure must not lose the record), so a record
    /// can exist without a vector — this is how callers (e.g. import
    /// verification) detect that state instead of discovering it as silently
    /// weaker recall.
    pub fn has_embedding(&self, id: &RecordId) -> bool {
        self.vector_index
            .as_ref()
            .and_then(|vi| vi.get_vector(id).ok().flatten())
            .is_some()
    }

    // ── Graph API ───────────────────────────────────────────────────

    fn require_graph_index(&self) -> Result<&dyn GraphIndex> {
        self.graph_index
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| AxilError::plugin("no graph index configured"))
    }

    /// Create a directed edge between two records.
    ///
    /// Both records must exist. Returns the edge ID.
    pub fn relate(
        &self,
        from: &RecordId,
        edge_type: &str,
        to: &RecordId,
        props: Option<Value>,
    ) -> Result<RecordId> {
        // Verify both endpoints exist.
        if self.storage.get(from)?.is_none() {
            return Err(AxilError::NotFound(format!("source record {from}")));
        }
        if self.storage.get(to)?.is_none() {
            return Err(AxilError::NotFound(format!("target record {to}")));
        }

        let gi = self.require_graph_index()?;
        gi.relate(
            from.clone(),
            edge_type,
            to.clone(),
            props.unwrap_or(Value::Object(Default::default())),
        )
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    pub fn unrelate(&self, edge_id: &RecordId) -> Result<bool> {
        self.require_graph_index()?.unrelate(edge_id)
    }

    /// Get neighbor records reachable from a record.
    pub fn neighbors(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<Record>> {
        let ids = self
            .require_graph_index()?
            .neighbors(id.clone(), edge_type, direction)?;
        self.resolve_ids(&ids)
    }

    /// Traverse a path from a starting record using path syntax.
    ///
    /// Path syntax: `->edge_type` (outgoing), `<-edge_type` (incoming),
    /// `<->edge_type` (both). Example: `->modified->file`.
    pub fn traverse(&self, start: &RecordId, path: &str) -> Result<Vec<Record>> {
        let steps = crate::plugin::parse_path(path)?;
        self.traverse_steps(start, &steps)
    }

    /// Traverse using pre-parsed steps.
    pub fn traverse_steps(&self, start: &RecordId, steps: &[TraversalStep]) -> Result<Vec<Record>> {
        let _span = crate::otel::span("axil.traverse", &[("depth", steps.len().to_string())]);
        let timer = self.metrics.start_timer(OpType::Traversal);
        let ids = self.require_graph_index()?.traverse(start.clone(), steps)?;
        let result = self.resolve_ids(&ids);
        let elapsed = timer.finish();
        crate::otel::record_operation("traverse", "", elapsed);
        result
    }

    /// Resolve a list of record IDs into full records, skipping missing ones.
    fn resolve_ids(&self, ids: &[RecordId]) -> Result<Vec<Record>> {
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(record) = self.storage.get(id)? {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Check whether a graph index is configured.
    pub fn has_graph_index(&self) -> bool {
        self.graph_index.is_some()
    }

    /// List edges attached to a record in the given direction.
    pub fn edges(
        &self,
        id: &RecordId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<EdgeInfo>> {
        self.require_graph_index()?
            .edges(id.clone(), edge_type, direction)
    }

    // ── Time-Series API ─────────────────────────────────────────────

    fn require_timeseries_index(&self) -> Result<&dyn TimeSeriesIndex> {
        self.timeseries_index
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| AxilError::plugin("no timeseries index configured"))
    }

    /// Get records created within the last `duration_secs` seconds.
    pub fn since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<Record>> {
        let ids = self
            .require_timeseries_index()?
            .since(table, duration_secs)?;
        self.resolve_ids(&ids)
    }

    /// Get the most recent `limit` records, ordered newest first.
    pub fn timeline(&self, table: Option<&str>, limit: usize) -> Result<Vec<Record>> {
        let ids = self.require_timeseries_index()?.latest(table, limit)?;
        self.resolve_ids(&ids)
    }

    /// Get records modified within the last `duration_secs` seconds.
    pub fn changed_since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<Record>> {
        let ids = self
            .require_timeseries_index()?
            .changed_since(table, duration_secs)?;
        self.resolve_ids(&ids)
    }

    /// Get records in a time range (microseconds since epoch).
    pub fn time_range(
        &self,
        table: Option<&str>,
        start_us: i64,
        end_us: i64,
    ) -> Result<Vec<Record>> {
        let ids = self
            .require_timeseries_index()?
            .range(table, start_us, end_us)?;
        self.resolve_ids(&ids)
    }

    /// Count records grouped by time bucket within a range.
    ///
    /// Returns `(bucket_start_us, count)` pairs sorted chronologically.
    pub fn count_by_bucket(
        &self,
        table: Option<&str>,
        bucket: crate::plugin::TimeBucket,
        start_us: i64,
        end_us: i64,
    ) -> Result<Vec<(i64, usize)>> {
        self.require_timeseries_index()?
            .count_by_bucket(table, bucket, start_us, end_us)
    }

    /// Downsample old records by creating daily summary records and
    /// optionally purging the originals.
    ///
    /// Records older than `retain_days` are grouped by (table, day).
    /// For each group a summary record is inserted into the `_summaries`
    /// table (if it doesn't already exist for that day+table). When
    /// `purge` is true the original records are deleted after summarising.
    ///
    /// Returns `(summaries_created, records_purged)`.
    pub fn downsample(&self, retain_days: u64, purge: bool) -> Result<(usize, usize)> {
        use chrono::Utc;

        const SUMMARIES_TABLE: &str = "_summaries";

        let tsi = self.require_timeseries_index()?;

        let now_us = Utc::now().timestamp_micros();
        let delta_us = i64::try_from(retain_days)
            .ok()
            .and_then(|d| d.checked_mul(86_400_000_000))
            .unwrap_or(i64::MAX);
        let cutoff_us = now_us.saturating_sub(delta_us);

        let old_ids = tsi.range(None, 0, cutoff_us)?;
        if old_ids.is_empty() {
            return Ok((0, 0));
        }

        // Group by (table, day).
        let mut groups: std::collections::BTreeMap<(String, String), Vec<RecordId>> =
            std::collections::BTreeMap::new();
        for id in &old_ids {
            if let Some(record) = self.storage.get(id)? {
                if record.table == SUMMARIES_TABLE {
                    continue;
                }
                let day = record.created_at.format("%Y-%m-%d").to_string();
                groups
                    .entry((record.table.clone(), day))
                    .or_default()
                    .push(id.clone());
            }
        }

        // Load existing summaries once — keyed by (table, day) → (record ID, count).
        // Tracking the count lets us skip records already accounted for.
        let mut existing_summaries: std::collections::HashMap<(String, String), (RecordId, u64)> =
            self.storage
                .list(SUMMARIES_TABLE, usize::MAX, 0)?
                .iter()
                .filter_map(|r| {
                    let t = r.data.get("table")?.as_str()?.to_string();
                    let d = r.data.get("date")?.as_str()?.to_string();
                    let c = r.data.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
                    Some(((t, d), (r.id.clone(), c)))
                })
                .collect();

        let mut summaries_created = 0usize;
        let mut records_purged = 0usize;

        for ((table, day), ids) in &groups {
            let key = (table.clone(), day.clone());
            let new_count = ids.len() as u64;

            if let Some((existing_id, old_count)) = existing_summaries.get(&key) {
                // Summary already exists. Compute the correct total:
                // - purge=true: old records were deleted, so ids only contains
                //   NEW records. Total = old_count + new arrivals.
                // - purge=false: old records still exist, so ids contains ALL
                //   records in this bucket. Total = ids.len() (the current truth).
                let correct_count = if purge {
                    old_count + new_count
                } else {
                    new_count
                };
                if correct_count != *old_count {
                    if let Some(existing) = self.storage.get(existing_id)? {
                        let mut new_data = existing.data.clone();
                        new_data["count"] = serde_json::json!(correct_count);
                        self.update(existing_id, new_data)?;
                    }
                    // Update our local cache so subsequent groups see the new count.
                    if let Some(entry) = existing_summaries.get_mut(&key) {
                        entry.1 = correct_count;
                    }
                }
            } else {
                let summary = serde_json::json!({
                    "table": table,
                    "date": day,
                    "count": new_count,
                    "type": "daily_summary",
                });
                let summary_record = self.insert(SUMMARIES_TABLE, summary)?;
                existing_summaries.insert(key, (summary_record.id, new_count));
                summaries_created += 1;
            }

            if purge {
                for id in ids {
                    if self.delete(id)? {
                        records_purged += 1;
                    }
                }
            }
        }

        Ok((summaries_created, records_purged))
    }

    /// Consolidate daily summaries older than `daily_retention_days` into
    /// weekly summaries and optionally purge the consumed daily summaries.
    ///
    /// Weekly summaries are stored in `_summaries` with
    /// `"type": "weekly_summary"` and `"week"` set to the ISO week start date.
    ///
    /// Returns `(weeklies_created, dailies_purged)`.
    pub fn downsample_weekly(
        &self,
        daily_retention_days: u64,
        purge: bool,
    ) -> Result<(usize, usize)> {
        use chrono::{Datelike, Utc};

        const SUMMARIES_TABLE: &str = "_summaries";

        let now = Utc::now();
        // Cap at ~100 years to avoid chrono::Duration::days overflow.
        let days = i64::try_from(daily_retention_days)
            .unwrap_or(i64::MAX)
            .min(36500);
        let cutoff = now - chrono::Duration::days(days);

        // Load all daily summaries.
        let all_summaries = self.storage.list(SUMMARIES_TABLE, usize::MAX, 0)?;

        // Filter to old daily summaries.
        let old_dailies: Vec<_> = all_summaries
            .iter()
            .filter(|r| {
                r.data.get("type").and_then(|v| v.as_str()) == Some("daily_summary")
                    && r.data
                        .get("date")
                        .and_then(|v| v.as_str())
                        .and_then(|d| chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
                        .is_some_and(|date| date <= cutoff.date_naive())
            })
            .collect();

        if old_dailies.is_empty() {
            return Ok((0, 0));
        }

        // Group by (table, week_start_date).
        let mut weekly_groups: std::collections::BTreeMap<(String, String), (u64, Vec<RecordId>)> =
            std::collections::BTreeMap::new();
        for r in &old_dailies {
            let table = match r.data.get("table").and_then(|v| v.as_str()) {
                Some(t) => t.to_string(),
                None => continue,
            };
            let date_str = match r.data.get("date").and_then(|v| v.as_str()) {
                Some(d) => d,
                None => continue,
            };
            let date = match chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                Ok(d) => d,
                Err(_) => continue,
            };
            let count = r.data.get("count").and_then(|v| v.as_u64()).unwrap_or(0);

            // Compute Monday of the week.
            let weekday = date.weekday().num_days_from_monday();
            let monday = date - chrono::Duration::days(weekday as i64);
            let week_key = monday.format("%Y-%m-%d").to_string();

            let entry = weekly_groups
                .entry((table, week_key))
                .or_insert((0, Vec::new()));
            entry.0 += count;
            entry.1.push(r.id.clone());
        }

        // Check for existing weekly summaries.
        // Map existing weekly summaries: (table, week) → (record_id, count).
        let mut existing_weeklies: std::collections::HashMap<(String, String), (RecordId, u64)> =
            all_summaries
                .iter()
                .filter(|r| r.data.get("type").and_then(|v| v.as_str()) == Some("weekly_summary"))
                .filter_map(|r| {
                    let t = r.data.get("table")?.as_str()?.to_string();
                    let w = r.data.get("week")?.as_str()?.to_string();
                    let c = r.data.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
                    Some(((t, w), (r.id.clone(), c)))
                })
                .collect();

        let mut weeklies_created = 0usize;
        let mut dailies_purged = 0usize;

        for ((table, week), (count, daily_ids)) in &weekly_groups {
            let key = (table.clone(), week.clone());
            if let Some((existing_id, old_count)) = existing_weeklies.get(&key) {
                // Weekly already exists. Same logic as downsample():
                // - purge=true: old dailies were deleted, so `count` is from
                //   genuinely new dailies. Total = old + new.
                // - purge=false: old dailies still exist, so `count` is the
                //   sum of ALL dailies in this week. Use it as-is.
                let new_total = if purge { old_count + count } else { *count };
                if new_total != *old_count {
                    if let Some(existing) = self.storage.get(existing_id)? {
                        let mut new_data = existing.data.clone();
                        new_data["count"] = serde_json::json!(new_total);
                        self.update(existing_id, new_data)?;
                    }
                    if let Some(entry) = existing_weeklies.get_mut(&key) {
                        entry.1 = new_total;
                    }
                }
            } else {
                let rec = self.insert(
                    SUMMARIES_TABLE,
                    serde_json::json!({
                        "table": table,
                        "week": week,
                        "count": count,
                        "type": "weekly_summary",
                    }),
                )?;
                existing_weeklies.insert(key, (rec.id, *count));
                weeklies_created += 1;
            }

            if purge {
                for id in daily_ids {
                    if self.delete(id)? {
                        dailies_purged += 1;
                    }
                }
            }
        }

        Ok((weeklies_created, dailies_purged))
    }

    /// Run a full heal cycle using the provided config.
    ///
    /// 1. Downsample old records into daily summaries (purging originals).
    /// 2. Consolidate old daily summaries into weekly summaries (purging dailies).
    ///
    /// Returns a summary of what was done.
    pub fn heal(&self, config: &crate::config::TimeseriesConfig) -> Result<HealReport> {
        let (daily_summaries, records_purged) =
            self.downsample(config.full_retention_days, true)?;
        let (weekly_summaries, dailies_purged) =
            self.downsample_weekly(config.daily_summary_days, true)?;
        Ok(HealReport {
            daily_summaries_created: daily_summaries,
            records_purged,
            weekly_summaries_created: weekly_summaries,
            daily_summaries_purged: dailies_purged,
        })
    }

    /// Check whether a timeseries index is configured.
    pub fn has_timeseries_index(&self) -> bool {
        self.timeseries_index.is_some()
    }

    // ── FTS API ────────────────────────────────────────────────────────

    fn require_fts_index(&self) -> Result<&dyn SearchIndex> {
        self.fts_index
            .as_ref()
            .map(|arc| arc.as_ref())
            .ok_or_else(|| AxilError::plugin("no FTS index configured"))
    }

    /// Manually index a text field for full-text search.
    ///
    /// The record must exist. This is in addition to any auto-indexing
    /// that happens on insert.
    pub fn index_text(&self, id: &RecordId, field: &str, text: &str) -> Result<()> {
        if self.storage.get(id)?.is_none() {
            return Err(AxilError::NotFound(format!("record {id}")));
        }
        self.require_fts_index()?.index_text(id, field, text)
    }

    /// Full-text search across all indexed fields.
    ///
    /// Returns records with their BM25 relevance scores (normalized to `[0, 1)`).
    pub fn search_text(&self, query: &str, limit: usize) -> Result<Vec<(Record, f32)>> {
        let fi = self.require_fts_index()?;
        self.resolve_scored_public(fi.search_text(query, limit)?)
    }

    /// Full-text search scoped to a specific field.
    pub fn search_field(
        &self,
        query: &str,
        field: &str,
        limit: usize,
    ) -> Result<Vec<(Record, f32)>> {
        let fi = self.require_fts_index()?;
        self.resolve_scored_public(fi.search_field(query, field, limit)?)
    }

    /// Full-text search with snippet/highlight generation.
    ///
    /// Returns `(Record, score, snippet_html)` with matched terms in `<b>` tags.
    pub fn search_with_snippets(
        &self,
        query: &str,
        limit: usize,
        max_chars: usize,
    ) -> Result<Vec<(Record, f32, String)>> {
        let fi = self.require_fts_index()?;
        let results = fi.search_with_snippets(query, limit, max_chars)?;
        self.resolve_scored_snippets_public(results)
    }

    /// Fuzzy full-text search with typo tolerance (Levenshtein distance 1-2).
    pub fn search_fuzzy(
        &self,
        query: &str,
        distance: u8,
        limit: usize,
    ) -> Result<Vec<(Record, f32)>> {
        let fi = self.require_fts_index()?;
        self.resolve_scored_public(fi.search_fuzzy(query, distance, limit)?)
    }

    fn resolve_scored_public(
        &self,
        results: Vec<(crate::record::RecordId, f32)>,
    ) -> Result<Vec<(Record, f32)>> {
        let mut best: std::collections::HashMap<RecordId, (Record, f32, usize)> =
            std::collections::HashMap::new();
        for (rank, (rid, score)) in results.into_iter().enumerate() {
            let Some((source_id, record)) = self.resolve_recall_candidate(&rid)? else {
                continue;
            };
            let entry = best
                .entry(source_id.clone())
                .or_insert_with(|| (record, score, rank));
            if score > entry.1 {
                entry.0 = self.storage.get(&source_id)?.unwrap_or(entry.0.clone());
                entry.1 = score;
            }
            entry.2 = entry.2.min(rank);
        }

        let mut values: Vec<(Record, f32, usize)> = best.into_values().collect();
        values.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });
        Ok(values
            .into_iter()
            .map(|(record, score, _)| (record, score))
            .collect())
    }

    fn resolve_scored_snippets_public(
        &self,
        results: Vec<(crate::record::RecordId, f32, String)>,
    ) -> Result<Vec<(Record, f32, String)>> {
        let mut best: std::collections::HashMap<RecordId, (Record, f32, String, usize)> =
            std::collections::HashMap::new();
        for (rank, (rid, score, snippet)) in results.into_iter().enumerate() {
            let Some((source_id, record)) = self.resolve_recall_candidate(&rid)? else {
                continue;
            };
            match best.entry(source_id.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((record, score, snippet, rank));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let entry = entry.get_mut();
                    if score > entry.1 {
                        entry.0 = self.storage.get(&source_id)?.unwrap_or(entry.0.clone());
                        entry.1 = score;
                        entry.2 = snippet;
                    }
                    entry.3 = entry.3.min(rank);
                }
            }
        }

        let mut values: Vec<(Record, f32, String, usize)> = best.into_values().collect();
        values.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.3.cmp(&b.3))
        });
        Ok(values
            .into_iter()
            .map(|(record, score, snippet, _)| (record, score, snippet))
            .collect())
    }

    /// Check whether a FTS index is configured.
    pub fn has_fts_index(&self) -> bool {
        self.fts_index.is_some()
    }

    /// Commit pending FTS writes and optimize the index.
    pub fn fts_optimize(&self) -> Result<()> {
        let fi = self.require_fts_index()?;
        fi.optimize()
    }

    // ── LLM API ─────────────────────────────────────────────

    /// Check whether an LLM provider is configured and available.
    pub fn has_llm(&self) -> bool {
        self.llm_provider.as_ref().is_some_and(|p| p.is_available())
    }

    /// Get the LLM model name, or "none" if no provider is configured.
    pub fn llm_model_name(&self) -> &str {
        self.llm_provider
            .as_ref()
            .map(|p| p.model_name())
            .unwrap_or("none")
    }

    /// Get current session LLM usage statistics.
    pub fn llm_usage(&self) -> crate::llm::LlmUsage {
        self.llm_usage.usage()
    }

    /// Rate-limited, usage-tracked LLM call. All public LLM methods delegate here. All public LLM methods go through here.
    fn llm_call_guarded<F>(&self, f: F) -> Result<crate::llm::LlmResponse>
    where
        F: FnOnce(&dyn crate::llm::LlmProvider) -> Result<crate::llm::LlmResponse>,
    {
        let provider = self
            .llm_provider
            .as_ref()
            .ok_or_else(|| AxilError::plugin("no LLM configured"))?;

        if !provider.is_available() {
            self.llm_usage.record_fallback();
            return Err(AxilError::plugin("LLM provider is not available"));
        }

        let usage = self.llm_usage.usage();
        if !self.llm_rate_limiter.check_and_record(&usage) {
            self.llm_usage.record_fallback();
            return Err(AxilError::plugin(
                "LLM budget/rate limit reached — falling back to algorithmic",
            ));
        }

        let response = f(provider.as_ref())?;
        self.llm_usage
            .record_call(response.input_tokens, response.output_tokens);
        Ok(response)
    }

    /// Call the LLM for text completion, with rate limiting and usage tracking.
    ///
    /// Returns `Err` if no LLM is configured or budget is exceeded —
    /// callers should fall back to algorithmic.
    pub fn llm_complete(&self, prompt: &str) -> Result<crate::llm::LlmResponse> {
        self.llm_call_guarded(|p| p.complete(prompt))
    }

    /// Call the LLM for structured JSON extraction, with rate limiting and usage tracking.
    ///
    /// Returns `Err` if no LLM is configured or budget is exceeded —
    /// callers should fall back to algorithmic.
    pub fn llm_extract_json(
        &self,
        prompt: &str,
        schema_hint: &str,
    ) -> Result<crate::llm::LlmResponse> {
        self.llm_call_guarded(|p| p.extract_json(prompt, schema_hint))
    }

    /// Extract entities using LLM if available, falling back to regex-based extraction.
    ///
    /// LLM extraction produces richer results (person names, concepts) while
    /// regex-based extraction catches code identifiers and file paths.
    /// When both are available, results are merged (LLM wins on conflicts).
    pub fn extract_entities_enhanced(&self, text: &str) -> Vec<crate::entity::Entity> {
        // Always get algorithmic entities as baseline.
        let mut entities = crate::entity::extract_entities(text);

        // Try LLM enhancement.
        if self.has_llm() {
            let prompt = format!(
                "Extract named entities from this text. Return a JSON array.\n\
                 Text: \"{text}\"\n\
                 Schema: [{{\"name\": \"string\", \"type\": \"person|code|concept|file\"}}]\n\
                 Return ONLY the JSON array, no other text."
            );
            let schema_hint = r#"[{"name": "string", "type": "person|code|concept|file"}]"#;

            if let Ok(response) = self.llm_extract_json(&prompt, schema_hint) {
                if let Ok(llm_entities) =
                    serde_json::from_str::<Vec<serde_json::Value>>(&response.text)
                {
                    let mut seen: std::collections::HashSet<String> =
                        entities.iter().map(|e| e.name.to_lowercase()).collect();

                    for val in llm_entities {
                        if let (Some(name), Some(etype)) = (
                            val.get("name").and_then(|v| v.as_str()),
                            val.get("type").and_then(|v| v.as_str()),
                        ) {
                            let key = name.to_lowercase();
                            if seen.insert(key) {
                                let entity_type = match etype {
                                    "person" => crate::entity::EntityType::Reference,
                                    "file" => crate::entity::EntityType::File,
                                    "code" => crate::entity::EntityType::Code,
                                    _ => crate::entity::EntityType::Reference,
                                };
                                entities.push(crate::entity::Entity {
                                    name: name.to_string(),
                                    entity_type,
                                    source_text: name.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        entities
    }

    /// Consolidate facts about an entity using LLM if available,
    /// falling back to template-based merging.
    pub fn consolidate_entity_enhanced(
        &self,
        entity_name: &str,
        facts: &[(String, chrono::DateTime<chrono::Utc>)],
    ) -> String {
        if facts.is_empty() {
            return format!("No known facts about '{entity_name}'.");
        }

        // Try LLM consolidation.
        if self.has_llm() && facts.len() > 1 {
            let facts_text: String = facts
                .iter()
                .enumerate()
                .map(|(i, (text, dt))| format!("{}. '{}' ({})", i + 1, text, dt.format("%Y-%m-%d")))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Consolidate these facts about '{entity_name}' into a single brief summary.\n\
                 Facts (newest first):\n{facts_text}\n\n\
                 Write a brief, factual summary noting what changed and when. \
                 Return ONLY the summary text, no other text."
            );

            if let Ok(response) = self.llm_complete(&prompt) {
                let trimmed = response.text.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }

        // Algorithmic fallback: template-based merge.
        if facts.len() == 1 {
            return facts[0].0.clone();
        }

        let latest = &facts[0];
        let mut summary = format!("{} (latest: {})", latest.0, latest.1.format("%Y-%m-%d"));
        if facts.len() > 1 {
            summary.push_str(&format!(
                ". {} prior fact{}.",
                facts.len() - 1,
                if facts.len() > 2 { "s" } else { "" }
            ));
        }
        summary
    }

    /// Auto-categorize a record's table using LLM if available.
    ///
    /// Given the record data and existing table names, the LLM suggests
    /// the best table. Returns `None` if no LLM is available or if
    /// the suggestion can't be determined.
    pub fn auto_categorize(&self, data: &Value) -> Option<String> {
        if !self.has_llm() {
            return None;
        }

        let tables = self.tables().ok()?;
        if tables.is_empty() {
            return None;
        }

        let data_str = serde_json::to_string(data).ok()?;
        // Truncate for safety.
        let data_preview = if data_str.len() > 500 {
            &data_str[..500]
        } else {
            &data_str
        };

        let tables_str = tables.join(", ");
        let prompt = format!(
            "Given this data, which table does it belong in? \
             Available tables: [{tables_str}]\n\
             Data: {data_preview}\n\n\
             Return ONLY the table name, nothing else."
        );

        let response = self.llm_complete(&prompt).ok()?;
        let suggestion = response.text.trim().to_lowercase();

        // Only return if it matches an existing table.
        tables
            .iter()
            .find(|t| t.to_lowercase() == suggestion)
            .cloned()
    }

    /// Backfill the timeseries index with all existing records.
    ///
    /// Call this after opening a database with a new timeseries plugin
    /// on a pre-existing database that already has records. Returns the
    /// number of records processed (including re-inserts of already-indexed
    /// records, since `add()` is idempotent).
    ///
    /// Records are processed in batches to avoid loading the entire
    /// database into memory at once.
    pub fn backfill_timeseries(&self) -> Result<usize> {
        let tsi = self.require_timeseries_index()?;
        let tables = self.storage.tables()?;
        let mut count = 0;
        const BATCH_SIZE: usize = 1000;
        for table in &tables {
            let mut offset = 0;
            loop {
                let records = self.storage.list(table, BATCH_SIZE, offset)?;
                if records.is_empty() {
                    break;
                }
                let batch_len = records.len();
                for record in &records {
                    match tsi.on_record_insert(record) {
                        Ok(()) => count += 1,
                        Err(e) => {
                            // Stop on limit errors rather than silently swallowing.
                            return Err(e);
                        }
                    }
                }
                offset += batch_len;
                if batch_len < BATCH_SIZE {
                    break;
                }
            }
        }
        Ok(count)
    }

    // ── Self-Healing API ──────────────────────────────────

    /// Compact the database: purge expired records, superseded records,
    /// and clean orphaned edges/vectors/FTS entries.
    pub fn compact(&self) -> Result<crate::diagnostics::CompactReport> {
        let start = std::time::Instant::now();
        let size_before = self.database_size();

        let mut purged_expired = 0usize;
        let mut purged_superseded = 0usize;

        // 1. Purge expired and superseded records.
        // Uses self.delete() which handles all plugin cascade cleanup.
        // Pinned or high-importance records are protected from compaction.
        let now = chrono::Utc::now();
        let all_ids = self.storage.all_record_ids()?;
        for id in &all_ids {
            if let Some(record) = self.storage.get(id)? {
                // Protect high-importance records from compaction
                if crate::importance::is_pinned(&record.data)
                    || crate::importance::get_importance(&record.data) >= 0.8
                {
                    continue;
                }
                if is_expired_record(&record, &now) {
                    self.delete(id)?;
                    purged_expired += 1;
                } else if is_superseded_record(&record) {
                    self.delete(id)?;
                    purged_superseded += 1;
                }
            }
        }

        // 2. Clean orphaned edges, vectors, and FTS entries
        let cleaned_orphaned_edges = self.clean_orphaned_edges();
        let cleaned_orphaned_vectors = self.clean_orphaned_vectors();
        let cleaned_orphaned_fts = self.clean_orphaned_fts();

        let size_after = self.database_size();
        let freed = size_before.saturating_sub(size_after);

        self.audit_heal_action(
            "compact",
            &format!(
                "purged {purged_expired} expired, {purged_superseded} superseded, \
             {cleaned_orphaned_edges} orphaned edges, {cleaned_orphaned_vectors} orphaned vectors, \
             {cleaned_orphaned_fts} orphaned FTS entries"
            ),
        );

        Ok(crate::diagnostics::CompactReport {
            compacted: purged_expired
                + purged_superseded
                + cleaned_orphaned_edges
                + cleaned_orphaned_vectors
                + cleaned_orphaned_fts
                > 0,
            purged_expired,
            purged_superseded,
            cleaned_orphaned_edges,
            cleaned_orphaned_vectors,
            cleaned_orphaned_fts,
            freed_estimate_bytes: freed,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Count orphaned edges (edges pointing to deleted records).
    fn count_orphaned_edges(&self) -> usize {
        if let Some(ref gi) = self.graph_index {
            gi.all_edge_ids()
                .ok()
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|(_, from, to)| {
                            self.storage.get(from).ok().flatten().is_none()
                                || self.storage.get(to).ok().flatten().is_none()
                        })
                        .count()
                })
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Remove orphaned edges. Returns count removed.
    pub fn clean_orphaned_edges(&self) -> usize {
        let mut cleaned = 0;
        if let Some(ref gi) = self.graph_index {
            if let Ok(edges) = gi.all_edge_ids() {
                for (edge_id, from, to) in &edges {
                    let from_exists = self.storage.get(from).ok().flatten().is_some();
                    let to_exists = self.storage.get(to).ok().flatten().is_some();
                    if (!from_exists || !to_exists) && gi.unrelate(edge_id).unwrap_or(false) {
                        cleaned += 1;
                    }
                }
            }
        }
        cleaned
    }

    /// Remove orphaned vectors. Returns count removed.
    pub fn clean_orphaned_vectors(&self) -> usize {
        let mut cleaned = 0;
        if let Some(ref vi) = self.vector_index {
            if let Ok(vec_ids) = vi.all_ids() {
                for vid in &vec_ids {
                    if self.storage.get(vid).ok().flatten().is_none() {
                        let _ = vi.on_record_delete(vid);
                        cleaned += 1;
                    }
                }
            }
        }
        cleaned
    }

    /// Remove orphaned FTS entries (indexed docs whose records no longer exist).
    pub fn clean_orphaned_fts(&self) -> usize {
        let mut cleaned = 0;
        if let Some(ref fi) = self.fts_index {
            if let Ok(fts_ids) = fi.all_indexed_ids() {
                for fid in &fts_ids {
                    if self.storage.get(fid).ok().flatten().is_none() {
                        let _ = fi.on_record_delete(fid);
                        cleaned += 1;
                    }
                }
            }
        }
        cleaned
    }

    /// Count expired and superseded records in a single pass over storage.
    fn count_dead_records(&self) -> (usize, usize) {
        let now = chrono::Utc::now();
        let mut expired = 0usize;
        let mut superseded = 0usize;
        // Scan all records directly (single pass) instead of IDs + N gets.
        if let Ok(records) = self.storage.scan_all_records() {
            for record in &records {
                if is_expired_record(record, &now) {
                    expired += 1;
                } else if is_superseded_record(record) {
                    superseded += 1;
                }
            }
        }
        (expired, superseded)
    }

    /// Compact the vector index only when the tombstone ratio exceeds
    /// `threshold` (e.g. `HealingConfig::vector_rebuild_threshold`).
    ///
    /// Incremental inserts keep the live graph searchable, so deletes only
    /// accumulate tombstoned nodes; this reclaims them in one rebuild once they
    /// outweigh the live set enough to matter. Returns the number of tombstones
    /// reclaimed (`0` when the ratio was below threshold or there is no vector
    /// index). Intended for the background worker — off the write path.
    pub fn compact_vector_index_if_needed(&self, threshold: f64) -> Result<usize> {
        let Some(ref vi) = self.vector_index else {
            return Ok(0);
        };
        let deleted = vi.deleted_count();
        let live = vi.count();
        let total = live + deleted;
        if total == 0 {
            return Ok(0);
        }
        let ratio = deleted as f64 / total as f64;
        if ratio <= threshold {
            return Ok(0);
        }
        self.vector_rebuild()?;
        Ok(deleted)
    }

    /// Rebuild the vector index to compact tombstones.
    pub fn vector_rebuild(&self) -> Result<crate::diagnostics::VectorRebuildReport> {
        let start = std::time::Instant::now();
        let vi = self.require_vector_index()?;
        let old_size = vi.count();
        let deleted = vi.deleted_count();
        let new_size = vi.rebuild()?;

        self.audit_heal_action(
            "vector_rebuild",
            &format!(
                "rebuilt vector index: {old_size} -> {new_size}, removed {deleted} tombstones"
            ),
        );

        Ok(crate::diagnostics::VectorRebuildReport {
            rebuilt: true,
            reason: if deleted > 0 {
                "deletion_ratio_exceeded".to_string()
            } else {
                "manual".to_string()
            },
            old_size,
            new_size,
            deleted_removed: deleted,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Detect problems in the database.
    pub fn detect_problems(&self) -> Vec<crate::diagnostics::ProblemDetection> {
        // `count_orphaned_edges` and `count_dead_records` are each a full-DB
        // scan; compute them once and reuse via the shared inner impl.
        self.detect_problems_with(self.count_orphaned_edges(), self.count_dead_records())
    }

    /// Inner impl of [`detect_problems`] that reuses already-computed
    /// orphaned-edge and dead-record counts. [`report`] computes both
    /// counts for its own use and passes them here, so a single `report()`
    /// scans the DB once instead of three times.
    fn detect_problems_with(
        &self,
        orphaned: usize,
        dead_records: (usize, usize),
    ) -> Vec<crate::diagnostics::ProblemDetection> {
        let mut problems = Vec::new();

        // Hot table imbalance
        if let Ok(tables) = self.tables_with_counts() {
            let total: usize = tables.iter().map(|(_, c)| c).sum();
            if total > 100 {
                for (name, count) in &tables {
                    let ratio = *count as f64 / total as f64;
                    if ratio > 0.9 {
                        problems.push(crate::diagnostics::ProblemDetection {
                            detector: "hot_table_imbalance".to_string(),
                            severity: Severity::Warning,
                            message: format!(
                                "Table '{}' contains {:.0}% of all records ({}/{})",
                                name,
                                ratio * 100.0,
                                count,
                                total
                            ),
                            recommendation: format!(
                                "Consider TTL on '{}' table or separate high-volume data",
                                name
                            ),
                            auto_fixable: false,
                        });
                    }
                }
            }
        }

        // Index size mismatch (vectors vs records)
        if let Some(ref vi) = self.vector_index {
            let vec_count = vi.count();
            let total = self.total_records().unwrap_or(0);
            // Only flag if vectors exist but count is very different
            if vec_count > 0 && total > 0 {
                let ratio = vec_count as f64 / total as f64;
                if !(0.5..=2.0).contains(&ratio) {
                    problems.push(crate::diagnostics::ProblemDetection {
                        detector: "index_size_mismatch".to_string(),
                        severity: Severity::Warning,
                        message: format!(
                            "Vector count ({}) doesn't match record count ({})",
                            vec_count, total
                        ),
                        recommendation: "Consider reindexing: axil heal --reindex".to_string(),
                        auto_fixable: true,
                    });
                }
            }

            // Vector deletion ratio
            let deleted = vi.deleted_count();
            let total_vec = vec_count + deleted;
            if total_vec > 0 {
                let del_ratio = deleted as f64 / total_vec as f64;
                if del_ratio > 0.2 {
                    problems.push(crate::diagnostics::ProblemDetection {
                        detector: "vector_deletion_ratio".to_string(),
                        severity: Severity::Warning,
                        message: format!(
                            "Vector index deletion ratio is {:.1}% ({} deleted / {} total)",
                            del_ratio * 100.0,
                            deleted,
                            total_vec
                        ),
                        recommendation: "Rebuild vector index: axil heal --reindex".to_string(),
                        auto_fixable: true,
                    });
                }
            }
        }

        // Reverse-orphan reconciliation: records that committed to core storage
        // but never got their vector embedding and/or FTS document. A torn
        // insert fan-out (insert commits first, then swallows embed/index
        // failures) can otherwise leave a stored memory invisible to recall.
        // One scan feeds both the embedding and FTS reconciliation.
        if self.vector_index.is_some() || self.fts_index.is_some() {
            if let Ok(records) = self.storage.scan_all_records() {
                let missing_embeddings = self.count_missing_embeddings(&records);
                if missing_embeddings > 0 {
                    // Only auto-fixable when an embedder can regenerate them; a
                    // manual `add_vector` user with no embedder is already gated
                    // out by `count_missing_embeddings`, but guard the flag too.
                    problems.push(crate::diagnostics::ProblemDetection {
                        detector: "missing_embeddings".to_string(),
                        severity: Severity::Warning,
                        message: format!(
                            "{} record(s) committed without a vector embedding",
                            missing_embeddings
                        ),
                        recommendation: "Re-embed missing records: axil heal --reindex".to_string(),
                        auto_fixable: self.embedder.is_some(),
                    });
                }

                let missing_fts = self.count_missing_fts(&records);
                if missing_fts > 0 {
                    problems.push(crate::diagnostics::ProblemDetection {
                        detector: "missing_fts".to_string(),
                        severity: Severity::Warning,
                        message: format!(
                            "{} record(s) committed without an FTS document",
                            missing_fts
                        ),
                        recommendation: "Re-index missing records: axil heal --reindex".to_string(),
                        auto_fixable: true,
                    });
                }
            }
        }

        // Orphaned edges (count passed in by the caller)
        if orphaned > 0 {
            problems.push(crate::diagnostics::ProblemDetection {
                detector: "orphaned_edges".to_string(),
                severity: Severity::Warning,
                message: format!("{} orphaned edges found", orphaned),
                recommendation: "Clean orphans: axil heal --orphans".to_string(),
                auto_fixable: true,
            });
        }

        // Expired and superseded records pending cleanup (counts passed in)
        let (expired_count, superseded_count) = dead_records;

        if expired_count > 0 {
            problems.push(crate::diagnostics::ProblemDetection {
                detector: "expired_records".to_string(),
                severity: if expired_count > 1000 {
                    Severity::Warning
                } else {
                    Severity::Ok
                },
                message: format!("{} expired records pending cleanup", expired_count),
                recommendation: "Run 'axil heal --compact' to purge expired records".to_string(),
                auto_fixable: true,
            });
        }

        if superseded_count > 0 {
            problems.push(crate::diagnostics::ProblemDetection {
                detector: "superseded_records".to_string(),
                severity: if superseded_count > 500 {
                    Severity::Warning
                } else {
                    Severity::Ok
                },
                message: format!("{} superseded records pending cleanup", superseded_count),
                recommendation: "Run 'axil heal --compact' to purge superseded records".to_string(),
                auto_fixable: true,
            });
        }

        // Storage efficiency
        let file_size = self.database_size();
        let total = self.total_records().unwrap_or(0);
        if file_size > 10 * 1024 * 1024 && total < 100 {
            problems.push(crate::diagnostics::ProblemDetection {
                detector: "storage_bloat".to_string(),
                severity: Severity::Warning,
                message: format!(
                    "Large file ({}) with few records ({}) — may benefit from compaction",
                    diagnostics::human_bytes(file_size),
                    total
                ),
                recommendation: "Run 'axil compact' to reclaim space".to_string(),
                auto_fixable: true,
            });
        }

        problems
    }

    /// Generate a comprehensive health report with recommendations.
    pub fn report(&self) -> Result<crate::diagnostics::HealthReport> {
        let now = chrono::Utc::now();
        let info = self.info()?;
        let total_records = info.total_records;
        // Both are full-DB scans; compute once and share with detect_problems
        // (which would otherwise recompute them) and the report body below.
        let orphaned_edges = self.count_orphaned_edges();
        let dead_records = self.count_dead_records();
        let problems = self.detect_problems_with(orphaned_edges, dead_records);
        let slow_queries = self.slow_queries(Some(100), None);

        // Compute score (start at 100, subtract for problems)
        let mut score: i32 = 100;
        let mut warnings = 0usize;
        let mut errors = 0usize;
        for p in &problems {
            match p.severity {
                Severity::Error => {
                    score -= 15;
                    errors += 1;
                }
                Severity::Warning => {
                    score -= 5;
                    warnings += 1;
                }
                Severity::Ok => {}
            }
        }
        let score = score.max(0) as u32;

        let overall_health = if score >= 85 {
            "good"
        } else if score >= 60 {
            "warning"
        } else {
            "critical"
        };

        // Expired/superseded counts (computed once above, reused here)
        let (expired_count, superseded_count) = dead_records;
        let live_count = total_records.saturating_sub(expired_count + superseded_count);
        let live_ratio = if total_records > 0 {
            live_count as f64 / total_records as f64
        } else {
            1.0
        };

        // Index stats
        let vec_count = self.vector_index.as_ref().map(|vi| vi.count()).unwrap_or(0);
        let edge_count = self
            .graph_index
            .as_ref()
            .map(|gi| gi.edge_count())
            .unwrap_or(0);
        // orphaned_edges computed once at the top of report()
        let vector_deletion_ratio = self.vector_index.as_ref().map(|vi| {
            let deleted = vi.deleted_count();
            let total = vi.count() + deleted;
            if total > 0 {
                deleted as f64 / total as f64
            } else {
                0.0
            }
        });

        // Build recommendations
        let mut recommendations = Vec::new();
        for p in &problems {
            if p.auto_fixable {
                let priority = match p.severity {
                    Severity::Error => "high",
                    Severity::Warning => "medium",
                    Severity::Ok => "low",
                };
                let command = match p.detector.as_str() {
                    "vector_deletion_ratio" | "index_size_mismatch" => "axil heal --reindex",
                    "orphaned_edges" => "axil heal --orphans",
                    _ => "axil heal --compact",
                };
                recommendations.push(crate::diagnostics::Recommendation {
                    priority: priority.to_string(),
                    action: p.recommendation.clone(),
                    auto_fixable: true,
                    command: command.to_string(),
                });
            }
        }

        let index_status = if orphaned_edges > 0 || vector_deletion_ratio.unwrap_or(0.0) > 0.15 {
            Severity::Warning
        } else {
            Severity::Ok
        };

        let data_status = if live_ratio < 0.7 {
            Severity::Warning
        } else {
            Severity::Ok
        };

        Ok(crate::diagnostics::HealthReport {
            generated_at: now.to_rfc3339(),
            overall_health: overall_health.to_string(),
            score,
            summary: format!(
                "{} warnings, {} errors. {}",
                warnings,
                errors,
                if score >= 85 {
                    "Database is healthy.".to_string()
                } else {
                    format!("Score: {}/100. Optimization recommended.", score)
                }
            ),
            sections: crate::diagnostics::HealthSections {
                storage: crate::diagnostics::StorageSection {
                    status: if info.total_size > 100 * 1024 * 1024 {
                        Severity::Warning
                    } else {
                        Severity::Ok
                    },
                    size: diagnostics::human_bytes(info.total_size),
                    record_count: total_records,
                    table_count: info.tables.len(),
                },
                performance: crate::diagnostics::PerformanceSection {
                    status: if slow_queries.len() > 10 {
                        Severity::Warning
                    } else {
                        Severity::Ok
                    },
                    slow_queries_count: slow_queries.len(),
                },
                indexes: crate::diagnostics::IndexSection {
                    status: index_status,
                    vectors: vec_count,
                    edges: edge_count,
                    orphaned_edges,
                    vector_deletion_ratio,
                },
                data_quality: crate::diagnostics::DataQualitySection {
                    status: data_status,
                    expired_records: expired_count,
                    superseded_records: superseded_count,
                    live_ratio,
                },
            },
            recommendations,
        })
    }

    /// Run all auto-fixable healing actions.
    ///
    /// If `dry_run` is true, returns what would be fixed without doing it.
    pub fn heal_all(
        &self,
        config: &crate::config::HealingConfig,
        dry_run: bool,
    ) -> Result<crate::diagnostics::SelfHealReport> {
        let start = std::time::Instant::now();
        let mut actions = Vec::new();

        if dry_run {
            // Detect problems and report what would be done
            let problems = self.detect_problems();
            for p in &problems {
                if p.auto_fixable {
                    actions.push(crate::diagnostics::HealAction {
                        action: p.detector.clone(),
                        result: format!("[dry-run] would fix: {}", p.message),
                    });
                }
            }
            return Ok(crate::diagnostics::SelfHealReport {
                healed: false,
                actions,
                duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }

        // 1. Compact (expired + superseded + orphans)
        let compact_report = self.compact()?;
        if compact_report.compacted {
            actions.push(crate::diagnostics::HealAction {
                action: "compact".to_string(),
                result: format!(
                    "purged {} expired, {} superseded, {} orphaned edges, {} orphaned vectors",
                    compact_report.purged_expired,
                    compact_report.purged_superseded,
                    compact_report.cleaned_orphaned_edges,
                    compact_report.cleaned_orphaned_vectors,
                ),
            });
        }

        // 2. Rebuild vector index if needed
        if let Some(ref vi) = self.vector_index {
            let deleted = vi.deleted_count();
            let total = vi.count() + deleted;
            if total > 0 {
                let ratio = deleted as f64 / total as f64;
                if ratio > config.vector_rebuild_threshold {
                    let rebuild_report = self.vector_rebuild()?;
                    actions.push(crate::diagnostics::HealAction {
                        action: "vector_rebuild".to_string(),
                        result: format!(
                            "rebuilt: {} -> {}, removed {} tombstones",
                            rebuild_report.old_size,
                            rebuild_report.new_size,
                            rebuild_report.deleted_removed,
                        ),
                    });
                }
            }
        }

        // 3. Re-embed / re-index records lost by a torn insert fan-out (records
        // that committed to core storage but never got their vector embedding
        // and/or FTS document).
        let (reembedded, refts) = self.reembed_missing()?;
        if reembedded > 0 || refts > 0 {
            actions.push(crate::diagnostics::HealAction {
                action: "reembed_missing".to_string(),
                result: format!(
                    "restored {reembedded} missing embeddings, {refts} missing FTS docs"
                ),
            });
        }

        Ok(crate::diagnostics::SelfHealReport {
            healed: !actions.is_empty(),
            actions,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Check if compaction is needed based on config thresholds.
    pub fn needs_compaction(&self, config: &crate::config::HealingConfig) -> bool {
        let total = self.total_records().unwrap_or(0);
        if total == 0 {
            return false;
        }

        let (expired, superseded) = self.count_dead_records();

        if expired >= config.compact_expired_threshold {
            return true;
        }
        if superseded >= config.compact_superseded_threshold {
            return true;
        }

        // Check live ratio
        let dead = expired + superseded;
        let live_ratio = if total > 0 {
            (total - dead) as f64 / total as f64
        } else {
            1.0
        };
        if live_ratio < config.compact_live_ratio_threshold {
            return true;
        }

        false
    }

    /// Take a metrics snapshot for trend tracking.
    pub fn snapshot_metrics(&self) -> Result<crate::diagnostics::MetricsHistoryEntry> {
        let now = chrono::Utc::now();
        let info = self.info()?;
        let total = info.total_records;

        let vec_count = self.vector_index.as_ref().map(|vi| vi.count()).unwrap_or(0);
        let edge_count = self
            .graph_index
            .as_ref()
            .map(|gi| gi.edge_count())
            .unwrap_or(0);

        // Compute live ratio via shared scan
        let (expired, superseded) = self.count_dead_records();
        let dead = expired + superseded;
        let live_ratio = if total > 0 {
            (total - dead) as f64 / total as f64
        } else {
            1.0
        };

        let entry = crate::diagnostics::MetricsHistoryEntry {
            timestamp: now.to_rfc3339(),
            record_count: total,
            file_size_bytes: info.total_size,
            vector_count: vec_count,
            edge_count,
            live_ratio,
        };

        // Persist to storage
        let key = now.to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let _ = self.storage.append_metrics_snapshot(&key, &bytes);
            // Keep at most 365 daily snapshots
            let _ = self.storage.trim_metrics_history(365);
        }

        Ok(entry)
    }

    /// Get trend data over a period.
    pub fn trends(&self, days: u64) -> Result<crate::diagnostics::TrendReport> {
        // Load a generous number of entries, then filter by actual timestamp
        let raw = self.storage.list_metrics_history(10_000)?;
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days.min(36500) as i64);
        let cutoff_str = cutoff.to_rfc3339();

        let entries: Vec<crate::diagnostics::MetricsHistoryEntry> = raw
            .into_iter()
            .filter_map(|(_, bytes)| serde_json::from_slice(&bytes).ok())
            .filter(|e: &crate::diagnostics::MetricsHistoryEntry| {
                e.timestamp.as_str() >= cutoff_str.as_str()
            })
            .collect();

        let mut trends = std::collections::BTreeMap::new();

        if entries.len() >= 2 {
            let newest = &entries[0]; // newest first from storage
            let oldest = entries.last().unwrap();

            // Record count trend
            trends.insert(
                "record_count".to_string(),
                make_trend(oldest.record_count as f64, newest.record_count as f64),
            );

            // File size trend
            trends.insert(
                "file_size_bytes".to_string(),
                make_trend(oldest.file_size_bytes as f64, newest.file_size_bytes as f64),
            );

            // Vector count trend
            trends.insert(
                "vector_count".to_string(),
                make_trend(oldest.vector_count as f64, newest.vector_count as f64),
            );

            // Live ratio trend
            let lr_trend = make_trend(oldest.live_ratio, newest.live_ratio);
            let alert = if newest.live_ratio < 0.7 {
                Some("compact recommended".to_string())
            } else {
                None
            };
            trends.insert(
                "live_ratio".to_string(),
                crate::diagnostics::MetricTrend { alert, ..lr_trend },
            );
        }

        Ok(crate::diagnostics::TrendReport {
            period: format!("{}d", days),
            snapshots: entries.len(),
            trends,
        })
    }

    /// Log a heal action to the audit trail.
    fn audit_heal_action(&self, action: &str, detail: &str) {
        if !self
            .audit_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let entry = crate::metrics::AuditEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            operation: format!("heal:{}", action),
            record_id: String::new(),
            table: detail.to_string(),
        };
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let key = self.log_key();
            let _ = self.storage.append_audit(&key, &bytes);
        }
    }

    // ── File Management ─────────────────────────────────────────────

    /// Get the base database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush every engine's buffered state to its backing file.
    ///
    /// Most engines commit each write synchronously, so this is a no-op for
    /// them; the Tantivy FTS engine buffers documents and overrides
    /// [`Engine::flush`] to commit. Called before [`Axil::branch_create`] copies
    /// the companion files so each one reflects every committed record.
    fn flush_engines(&self) -> Result<()> {
        for plugin in &self.plugins {
            plugin.flush()?;
        }
        if let Some(ref v) = self.vector_index {
            v.flush()?;
        }
        if let Some(ref g) = self.graph_index {
            g.flush()?;
        }
        if let Some(ref ts) = self.timeseries_index {
            ts.flush()?;
        }
        if let Some(ref f) = self.fts_index {
            f.flush()?;
        }
        Ok(())
    }

    /// Create a point-in-time-consistent branch from this live handle.
    ///
    /// A branch is a file-level copy of the core `.axil` plus every companion
    /// file ([`branch_create`](crate::branch::branch_create) documents the
    /// layout). Driving it from the live handle makes the copy internally
    /// consistent under Axil's single-writer model:
    ///
    /// 1. While this handle is open it holds redb's **exclusive** OS write lock,
    ///    so no other process can mutate the database.
    /// 2. [`flush_engines`](Self::flush_engines) commits any buffered engine
    ///    state (notably the FTS index) so every companion file is current.
    /// 3. The handle is then **dropped**, releasing the OS lock, and only then
    ///    are the files copied. On Windows, redb holds a byte-range lock on the
    ///    core file for the lifetime of the [`Database`], so an open file cannot
    ///    be copied (`fs::copy` fails with a sharing-violation); the
    ///    flush-then-close-then-copy sequence is what makes this work
    ///    cross-platform. Single-writer means a brief in-process quiesce is
    ///    sufficient — there is no need to hold the OS lock during the byte copy.
    ///
    /// Taking `self` by value enforces at the type level that no mutation method
    /// can run on this handle once the branch operation has begun.
    pub fn branch_create(self, name: &str) -> Result<PathBuf> {
        // Validate before flushing/closing so a bad name is a cheap rejection.
        crate::branch::validate_branch_name(name)?;
        self.flush_engines()?;
        let db_path = self.path.clone();
        // Drop every redb handle (core + companions) so the files are unlocked
        // for copying. This is the quiesce point: no writer is active and all
        // committed state is on disk.
        drop(self);
        crate::branch::copy_branch_files(&db_path, name)
    }

    /// List all files belonging to this database (core + companions).
    pub fn files(&self) -> Vec<PathBuf> {
        self.files_with_roles()
            .into_iter()
            .map(|(p, _, _)| p)
            .collect()
    }

    /// Total bytes across all files belonging to this database.
    pub fn database_size(&self) -> u64 {
        self.files_with_roles()
            .into_iter()
            .map(|(_, _, size)| size)
            .sum()
    }

    /// Unified database info including plugin stats.
    pub fn info(&self) -> Result<DatabaseInfo> {
        let files = self.files_with_roles();
        let total_size = files.iter().map(|(_, _, s)| s).sum();
        let total_records = self.total_records()?;
        let tables = self.tables_with_counts()?;

        let mut plugins = BTreeMap::new();
        if let Some(ref vi) = self.vector_index {
            plugins.insert(
                "vector".to_string(),
                serde_json::json!({
                    "count": vi.count(),
                    "dimensions": vi.dimensions(),
                }),
            );
        }
        if self.graph_index.is_some() {
            plugins.insert(
                "graph".to_string(),
                serde_json::json!({
                    "enabled": true,
                }),
            );
        }
        if let Some(ref tsi) = self.timeseries_index {
            plugins.insert(
                "timeseries".to_string(),
                serde_json::json!({
                    "entries": tsi.entry_count(),
                }),
            );
        }
        if self.fts_index.is_some() {
            plugins.insert(
                "fts".to_string(),
                serde_json::json!({
                    "enabled": true,
                }),
            );
        }

        Ok(DatabaseInfo {
            path: self.path.clone(),
            files,
            total_size,
            total_records,
            tables,
            plugins,
        })
    }

    /// Internal: collect all files with their roles and sizes in one pass.
    fn files_with_roles(&self) -> Vec<(PathBuf, String, u64)> {
        let mut files = Vec::new();
        let core_size = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        files.push((self.path.clone(), "core".to_string(), core_size));

        for &(suffix, role) in COMPANION_SUFFIXES {
            let p = companion_path(&self.path, suffix);
            if p.exists() {
                let size = if p.is_dir() {
                    // For directory-based companions (e.g. .fts/), sum file sizes.
                    dir_size(&p)
                } else {
                    std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
                };
                files.push((p, role.to_string(), size));
            }
        }
        files
    }

    // ── Diagnostics API ────────────────────────────────────────────────

    /// Get the metrics collector.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Enable or disable the audit trail.
    pub fn set_audit_enabled(&self, enabled: bool) {
        self.audit_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the slow query threshold in milliseconds.
    pub fn set_slow_query_threshold(&mut self, ms: f64) {
        self.slow_query_threshold_ms = ms;
    }

    /// Generate a unique, sortable key for log entries.
    fn log_key(&self) -> String {
        let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
        let seq = self
            .log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{ts}:{seq:06}")
    }

    /// Record a slow query if the duration exceeds the threshold.
    /// Persisted to the database so entries survive across CLI invocations.
    pub fn record_slow_query(&self, command: &str, duration_ms: f64, result_count: usize) {
        if duration_ms < self.slow_query_threshold_ms {
            return;
        }
        let entry = SlowQueryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            command: command.to_string(),
            duration_ms,
            result_count,
        };
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let key = self.log_key();
            let _ = self.storage.append_slow_query(&key, &bytes);
            let _ = self.storage.trim_slow_queries(MAX_SLOW_QUERIES);
        }
    }

    /// Get the slow query log. Entries are returned newest first.
    pub fn slow_queries(&self, limit: Option<usize>, after: Option<&str>) -> Vec<SlowQueryEntry> {
        let max = limit.unwrap_or(MAX_SLOW_QUERIES);
        let raw = self.storage.list_slow_queries(max).unwrap_or_default();
        raw.into_iter()
            .filter_map(|(_, bytes)| serde_json::from_slice::<SlowQueryEntry>(&bytes).ok())
            .filter(|e| after.is_none_or(|a| e.timestamp.as_str() > a))
            .collect()
    }

    /// Clear the slow query log.
    pub fn clear_slow_queries(&self) {
        let _ = self.storage.clear_slow_queries();
    }

    /// Record an audit log entry. Persisted to the database.
    /// Storage-only update bypass for internal index maintainers that
    /// need to write back metadata onto a record without re-triggering
    /// `Axil::update`'s hook chain (which would recurse through
    /// `code_refs::sync_for_record`).
    pub(crate) fn storage_update_raw(&self, id: &RecordId, data: Value) -> Result<Record> {
        self.storage.update(id, data)
    }

    fn audit(&self, operation: &str, record_id: &RecordId, table: &str) {
        if !self
            .audit_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        // Skip internal index-maintenance writes (`_idx_*`, `_entities`,
        // `_scip_aliases`, `_recall_chunks`, …). The per-record audit log
        // records user-visible operations; index churn is noise. Bulk
        // imports that DO need their internal rows audited go through
        // `audit_inserts`, which bypasses this skip on purpose.
        if table.starts_with('_') {
            return;
        }
        self.write_audit_entry(operation, record_id, table);
    }

    /// Append a single audit entry to the log.
    ///
    /// Unconditional: callers own the `audit_enabled` gate and any
    /// table-prefix filtering. Shared by `audit` (per-record, user
    /// tables) and `audit_inserts` (bulk imports, internal tables).
    fn write_audit_entry(&self, operation: &str, record_id: &RecordId, table: &str) {
        let entry = AuditEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            operation: operation.to_string(),
            record_id: record_id.to_string(),
            table: table.to_string(),
        };
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let key = self.log_key();
            let _ = self.storage.append_audit(&key, &bytes);
            let _ = self.storage.trim_audit(MAX_AUDIT_ENTRIES);
        }
    }

    /// Get the audit log. Entries are returned newest first.
    pub fn audit_log(
        &self,
        limit: Option<usize>,
        after: Option<&str>,
        table: Option<&str>,
        op: Option<&str>,
    ) -> Vec<AuditEntry> {
        let max = limit.unwrap_or(MAX_AUDIT_ENTRIES);
        let raw = self.storage.list_audit(max).unwrap_or_default();
        raw.into_iter()
            .filter_map(|(_, bytes)| serde_json::from_slice::<AuditEntry>(&bytes).ok())
            .filter(|e| {
                if let Some(after) = after {
                    if e.timestamp.as_str() <= after {
                        return false;
                    }
                }
                if let Some(table) = table {
                    if e.table != table {
                        return false;
                    }
                }
                if let Some(op) = op {
                    if e.operation != op {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&self) {
        let _ = self.storage.clear_audit();
    }

    /// Run comprehensive health checks on the database.
    pub fn doctor(&self) -> Result<DoctorReport> {
        let mut checks = Vec::new();

        // 1. Database file readable and valid
        let total = match self.storage.total_records() {
            Ok(n) => {
                checks.push(CheckResult {
                    name: "database_readable".to_string(),
                    status: Severity::Ok,
                    detail: format!("database file valid, {n} records"),
                    fix: None,
                });
                n
            }
            Err(e) => {
                checks.push(CheckResult {
                    name: "database_readable".to_string(),
                    status: Severity::Error,
                    detail: format!("database read error: {e}"),
                    fix: None,
                });
                0
            }
        };

        // 2. Record integrity — count all records
        let tables = self.storage.tables_with_counts().unwrap_or_default();
        let table_sum: usize = tables.iter().map(|(_, c)| c).sum();
        if total == table_sum {
            checks.push(CheckResult {
                name: "record_integrity".to_string(),
                status: Severity::Ok,
                detail: format!("{total} records verified across {} tables", tables.len()),
                fix: None,
            });
        } else {
            checks.push(CheckResult {
                name: "record_integrity".to_string(),
                status: Severity::Warning,
                detail: format!("record count mismatch: total={total} but table sum={table_sum}"),
                fix: Some("axil compact".to_string()),
            });
        }

        // Reconcile the vector and FTS indexes against live records once: a
        // torn insert can commit a record but skip its embedding/FTS document,
        // leaving the memory invisible to recall. Scan a single time and share
        // the slice between both index checks below.
        let index_scan = if self.vector_index.is_some() || self.fts_index.is_some() {
            self.storage.scan_all_records().ok()
        } else {
            None
        };

        // 3. Vector index sync
        if let Some(ref vi) = self.vector_index {
            let vec_count = vi.count();
            let missing = index_scan
                .as_ref()
                .map(|recs| self.count_missing_embeddings(recs))
                .unwrap_or(0);
            if missing == 0 {
                checks.push(CheckResult {
                    name: "vector_index".to_string(),
                    status: Severity::Ok,
                    detail: format!(
                        "{vec_count} vectors indexed, dimensions={}",
                        vi.dimensions()
                    ),
                    fix: None,
                });
            } else {
                checks.push(CheckResult {
                    name: "vector_index".to_string(),
                    status: Severity::Warning,
                    detail: format!(
                        "{vec_count} vectors indexed, dimensions={}; {missing} record(s) missing an embedding",
                        vi.dimensions()
                    ),
                    fix: Some("axil heal --reindex".to_string()),
                });
            }
        }

        // 4. Graph: orphaned edges
        if let Some(ref gi) = self.graph_index {
            let edge_count = gi.edge_count();
            let mut orphaned = 0;
            if let Ok(edges) = gi.all_edge_ids() {
                for (_, from, to) in &edges {
                    let from_exists = self.storage.get(from).ok().flatten().is_some();
                    let to_exists = self.storage.get(to).ok().flatten().is_some();
                    if !from_exists || !to_exists {
                        orphaned += 1;
                    }
                }
            }
            if orphaned == 0 {
                checks.push(CheckResult {
                    name: "orphaned_edges".to_string(),
                    status: Severity::Ok,
                    detail: format!("{edge_count} edges, no orphans"),
                    fix: None,
                });
            } else {
                checks.push(CheckResult {
                    name: "orphaned_edges".to_string(),
                    status: Severity::Warning,
                    detail: format!("{orphaned} orphaned edges found (of {edge_count} total)"),
                    fix: Some("axil heal".to_string()),
                });
            }
        }

        // 5. Timeseries index
        if let Some(ref tsi) = self.timeseries_index {
            let entry_count = tsi.entry_count();
            checks.push(CheckResult {
                name: "timeseries_index".to_string(),
                status: Severity::Ok,
                detail: format!("{entry_count} time entries indexed"),
                fix: None,
            });
        }

        // 6. FTS index
        if self.fts_index.is_some() {
            let missing = index_scan
                .as_ref()
                .map(|recs| self.count_missing_fts(recs))
                .unwrap_or(0);
            if missing == 0 {
                checks.push(CheckResult {
                    name: "fts_index".to_string(),
                    status: Severity::Ok,
                    detail: "FTS index present".to_string(),
                    fix: None,
                });
            } else {
                checks.push(CheckResult {
                    name: "fts_index".to_string(),
                    status: Severity::Warning,
                    detail: format!("FTS index present; {missing} record(s) missing an FTS document"),
                    fix: Some("axil heal --reindex".to_string()),
                });
            }
        }

        // 7. Storage fragmentation estimate
        let info = self.info()?;
        let file_size = info.total_size;
        if file_size > 10 * 1024 * 1024 && total < 100 {
            checks.push(CheckResult {
                name: "storage_efficiency".to_string(),
                status: Severity::Warning,
                detail: format!(
                    "large file ({}) with few records ({total}) — may benefit from compaction",
                    diagnostics::human_bytes(file_size)
                ),
                fix: Some("axil compact".to_string()),
            });
        } else {
            checks.push(CheckResult {
                name: "storage_efficiency".to_string(),
                status: Severity::Ok,
                detail: format!(
                    "{} total, {total} records",
                    diagnostics::human_bytes(file_size)
                ),
                fix: None,
            });
        }

        Ok(DoctorReport::from_checks(checks))
    }

    /// Collect comprehensive database statistics.
    /// Compute activation-level statistics for records in a table (or all tables).
    pub fn activation_stats(
        &self,
        table_filter: Option<&str>,
        config: &crate::activation::ActivationConfig,
    ) -> Result<crate::activation::ActivationStats> {
        let now = chrono::Utc::now();
        let records = if let Some(table) = table_filter {
            self.storage.list(table, usize::MAX, 0)?
        } else {
            let mut all = Vec::new();
            for table in self.storage.tables()? {
                all.extend(self.storage.list(&table, usize::MAX, 0)?);
            }
            all
        };
        Ok(crate::activation::compute_stats(
            &records,
            &now,
            config.half_life_days,
        ))
    }

    pub fn stats(&self, table_filter: Option<&str>) -> Result<DatabaseStats> {
        let info = self.info()?;
        let snap = self.metrics.snapshot();

        let tables_json = if let Some(filter) = table_filter {
            let count = self.storage.count(filter).unwrap_or(0);
            serde_json::json!({ filter: count })
        } else {
            let mut map = serde_json::Map::new();
            for (name, count) in &info.tables {
                map.insert(name.clone(), serde_json::json!(count));
            }
            Value::Object(map)
        };

        let vec_count = self.vector_index.as_ref().map(|vi| vi.count()).unwrap_or(0);
        let vec_dims = self
            .vector_index
            .as_ref()
            .map(|vi| vi.dimensions())
            .unwrap_or(0);
        let edge_count = self
            .graph_index
            .as_ref()
            .map(|gi| gi.edge_count())
            .unwrap_or(0);
        let ts_entries = self
            .timeseries_index
            .as_ref()
            .map(|tsi| tsi.entry_count())
            .unwrap_or(0);

        // Build performance JSON from latency percentiles.
        let perf =
            serde_json::to_value(&snap.latencies).unwrap_or(Value::Object(Default::default()));

        Ok(DatabaseStats {
            database: DatabaseMeta {
                path: info.path.display().to_string(),
                size_bytes: info.total_size,
                size_human: diagnostics::human_bytes(info.total_size),
                created: snap.db_created_at,
                last_write: snap.last_write_at,
                last_read: snap.last_read_at,
            },
            records: RecordStats {
                total: info.total_records,
                tables: tables_json,
            },
            indexes: crate::diagnostics::IndexStats {
                vectors: vec_count,
                vector_dimensions: vec_dims,
                fts_enabled: self.fts_index.is_some(),
                edges: edge_count,
                timeseries_entries: ts_entries,
            },
            performance: perf,
        })
    }

    /// Run built-in micro-benchmarks.
    pub fn bench(&self) -> Result<BenchReport> {
        let mut results = Vec::new();

        // Insert benchmark: 1000 records
        let bench_table = "_bench_tmp";
        let insert_count = 1000usize;
        let start = std::time::Instant::now();
        let mut ids = Vec::with_capacity(insert_count);
        for i in 0..insert_count {
            let r = self.insert(
                bench_table,
                serde_json::json!({"i": i, "text": "benchmark record"}),
            )?;
            ids.push(r.id);
        }
        let insert_elapsed = start.elapsed();
        let insert_ops = insert_count as f64 / insert_elapsed.as_secs_f64();
        let insert_per = insert_elapsed.as_secs_f64() * 1000.0 / insert_count as f64;
        results.push(BenchResult {
            name: format!("insert_{insert_count}"),
            ops_per_sec: insert_ops,
            avg_ms: insert_per,
            iterations: insert_count,
        });

        // Get benchmark
        let get_count = 1000usize.min(ids.len());
        let start = std::time::Instant::now();
        for id in ids.iter().take(get_count) {
            let _ = self.get(id)?;
        }
        let get_elapsed = start.elapsed();
        let get_ops = get_count as f64 / get_elapsed.as_secs_f64();
        let get_per = get_elapsed.as_secs_f64() * 1000.0 / get_count as f64;
        results.push(BenchResult {
            name: format!("get_{get_count}"),
            ops_per_sec: get_ops,
            avg_ms: get_per,
            iterations: get_count,
        });

        // Vector search benchmark (if index configured)
        if let Some(ref vi) = self.vector_index {
            let dims = vi.dimensions();
            let query: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.01).sin()).collect();
            let search_count = 100;
            let start = std::time::Instant::now();
            for _ in 0..search_count {
                let _ = vi.search(&query, 5)?;
            }
            let search_elapsed = start.elapsed();
            let search_ops = search_count as f64 / search_elapsed.as_secs_f64();
            let search_per = search_elapsed.as_secs_f64() * 1000.0 / search_count as f64;
            results.push(BenchResult {
                name: format!("vector_search_{}_top5", vi.count()),
                ops_per_sec: search_ops,
                avg_ms: search_per,
                iterations: search_count,
            });
        }

        // FTS benchmark (if index configured)
        if let Some(ref fi) = self.fts_index {
            // Index some text first.
            for (i, id) in ids.iter().enumerate().take(100) {
                let _ = fi.index_text(
                    id,
                    "text",
                    &format!("benchmark record number {i} with some text"),
                );
            }
            let fts_count = 100;
            let start = std::time::Instant::now();
            for _ in 0..fts_count {
                let _ = fi.search_text("benchmark record", 10)?;
            }
            let fts_elapsed = start.elapsed();
            let fts_ops = fts_count as f64 / fts_elapsed.as_secs_f64();
            let fts_per = fts_elapsed.as_secs_f64() * 1000.0 / fts_count as f64;
            results.push(BenchResult {
                name: "fts_search_100".to_string(),
                ops_per_sec: fts_ops,
                avg_ms: fts_per,
                iterations: fts_count,
            });
        }

        // Graph traversal benchmark (if index configured)
        if let Some(ref gi) = self.graph_index {
            // Create a chain of edges: id[0] -> id[1] -> id[2] -> ...
            let chain_len = 50.min(ids.len());
            for i in 0..chain_len.saturating_sub(1) {
                let _ = gi.relate(
                    ids[i].clone(),
                    "bench_next",
                    ids[i + 1].clone(),
                    Value::Object(Default::default()),
                );
            }
            if chain_len > 2 {
                // Depth-1 traversal
                let traverse_count = 100;
                let steps_d1 = crate::plugin::parse_path("->bench_next").unwrap_or_default();
                let start = std::time::Instant::now();
                for _ in 0..traverse_count {
                    let _ = gi.traverse(ids[0].clone(), &steps_d1)?;
                }
                let trav_elapsed = start.elapsed();
                let trav_ops = traverse_count as f64 / trav_elapsed.as_secs_f64();
                let trav_per = trav_elapsed.as_secs_f64() * 1000.0 / traverse_count as f64;
                results.push(BenchResult {
                    name: "graph_traverse_depth1".to_string(),
                    ops_per_sec: trav_ops,
                    avg_ms: trav_per,
                    iterations: traverse_count,
                });

                // Depth-3 traversal
                let steps_d3 = crate::plugin::parse_path("->bench_next->bench_next->bench_next")
                    .unwrap_or_default();
                let start = std::time::Instant::now();
                for _ in 0..traverse_count {
                    let _ = gi.traverse(ids[0].clone(), &steps_d3)?;
                }
                let trav_elapsed = start.elapsed();
                let trav_ops = traverse_count as f64 / trav_elapsed.as_secs_f64();
                let trav_per = trav_elapsed.as_secs_f64() * 1000.0 / traverse_count as f64;
                results.push(BenchResult {
                    name: "graph_traverse_depth3".to_string(),
                    ops_per_sec: trav_ops,
                    avg_ms: trav_per,
                    iterations: traverse_count,
                });
            }
        }

        // Clean up benchmark records (also cascades edges).
        for id in &ids {
            let _ = self.delete(id);
        }

        Ok(BenchReport {
            benchmarks: results,
            system: diagnostics::SystemInfo::current(),
        })
    }

    // ── Intelligent Database ──────────────────────────────

    /// Recall with multi-signal scoring and explanation.
    ///
    /// Combines vector similarity, recency decay, graph proximity, keyword overlap,
    /// temporal proximity, preference matching, and relevance feedback into a single
    /// ranked result set. Each result includes a score breakdown.
    ///
    /// This is the primary intelligent recall API — replaces simple `similar_to()`
    /// for agent memory use cases.
    pub fn recall(
        &self,
        query: &str,
        top_k: usize,
        config: Option<crate::scoring::RecallConfig>,
    ) -> Result<Vec<crate::scoring::RecallResult>> {
        self.recall_with_feedback(query, top_k, config, Some(&self.feedback_store))
    }

    /// Recall with multi-signal scoring, explanation, and optional feedback store.
    pub fn recall_with_feedback(
        &self,
        query: &str,
        top_k: usize,
        config: Option<crate::scoring::RecallConfig>,
        feedback_store: Option<&crate::feedback::FeedbackStore>,
    ) -> Result<Vec<crate::scoring::RecallResult>> {
        let timer = self.metrics.start_timer(OpType::Recall);

        // Respect caller-supplied `now` for deterministic testing; default already uses Utc::now()
        let cfg = config.unwrap_or_default();
        let now = cfg.now;
        let mut cfg = cfg;

        // Parse temporal expressions from the query
        let effective_query = if cfg.temporal_target.is_none() {
            if let Some((target, cleaned)) = crate::temporal::parse_temporal(query, &now) {
                cfg.temporal_target = Some(target);
                cleaned
            } else {
                query.to_string()
            }
        } else {
            query.to_string()
        };

        // Extract keywords if not already set
        if cfg.query_keywords.is_empty() {
            cfg.query_keywords = crate::scoring::extract_keywords(&effective_query);
        }

        // B: widen candidate pool so downstream fusion / rerank
        // has enough raw material; minor CPU cost, meaningful recall lift.
        let fetch_k = top_k.saturating_mul(8).max(40);
        // Step 1: Vector search when an embedder + vector index are available.
        // Fall back to other retrieval signals instead of erroring out.
        let query_vec = if let (Some(embedder), Some(vi)) =
            (self.embedder.as_ref(), self.vector_index.as_ref())
        {
            match embedder.embed(&effective_query) {
                Ok(query_vec) => {
                    let vector_results = vi.search(&query_vec, fetch_k.saturating_mul(2))?;
                    (Some(query_vec), vector_results)
                }
                Err(_) => (None, Vec::new()),
            }
        } else {
            (None, Vec::new())
        };
        let (query_vec, vector_results) = query_vec;

        // Step 2: Compute FTS scores and merge FTS-only candidates into results
        // (runs even if vector is empty — FTS can provide standalone results)
        let fts_results = if let Some(ref fi) = self.fts_index {
            fi.search_text(&effective_query, fetch_k.saturating_mul(2))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Early return only if both vector AND FTS found nothing
        if vector_results.is_empty() && fts_results.is_empty() {
            timer.finish();
            return Ok(Vec::new());
        }

        // RRF constant — standard value from Cormack et al. (2009).
        const RRF_K: f32 = 60.0;

        // Identifier-aware query classification. When the query is an exact
        // identifier (UUID / path / code symbol / hostname / long numeric id),
        // the FTS list gets a tilt below so an exact lexical match isn't diluted
        // by semantic neighbors. Natural-language queries classify as
        // NaturalLanguage and the tilt is exactly zero — fusion stays pure RRF.
        let query_class = crate::query_class::classify_query(&effective_query);

        let mut candidates: std::collections::HashMap<RecordId, RecallCandidate> =
            std::collections::HashMap::new();
        for (rank, (rid, vec_score)) in vector_results.iter().enumerate() {
            let Some((source_id, source_record)) = self.resolve_recall_candidate(rid)? else {
                continue;
            };
            let rrf = 1.0 / (RRF_K + rank as f32);
            let entry = candidates
                .entry(source_id.clone())
                .or_insert_with(|| RecallCandidate {
                    record: source_record,
                    vector_score: *vec_score,
                    fts_score: 0.0,
                    first_rank: rank,
                    rrf_score: 0.0,
                    fts_rank: None,
                });
            if *vec_score > entry.vector_score {
                entry.vector_score = *vec_score;
            }
            entry.first_rank = entry.first_rank.min(rank);
            entry.rrf_score += rrf;
        }
        for (rank, (rid, fts_score)) in fts_results.iter().enumerate() {
            let Some((source_id, source_record)) = self.resolve_recall_candidate(rid)? else {
                continue;
            };
            let rrf = 1.0 / (RRF_K + rank as f32);
            let entry = candidates
                .entry(source_id.clone())
                .or_insert_with(|| RecallCandidate {
                    record: source_record,
                    vector_score: 0.0,
                    fts_score: *fts_score,
                    first_rank: fetch_k + rank,
                    rrf_score: 0.0,
                    fts_rank: Some(rank),
                });
            if *fts_score > entry.fts_score {
                entry.fts_score = *fts_score;
            }
            entry.first_rank = entry.first_rank.min(fetch_k + rank);
            entry.rrf_score += rrf;
            // Keep the best (smallest) FTS rank when a record appears multiple
            // times after canonical resolution.
            entry.fts_rank = Some(match entry.fts_rank {
                Some(existing) => existing.min(rank),
                None => rank,
            });
        }

        if candidates.is_empty() {
            timer.finish();
            return Ok(Vec::new());
        }

        let candidate_vector_results: Vec<(RecordId, f32)> = candidates
            .iter()
            .map(|(rid, candidate)| (rid.clone(), candidate.vector_score))
            .collect();

        // Step 3: Compute graph proximity for candidates
        let graph_scores: std::collections::HashMap<String, f32> =
            if let Some(ref gi) = self.graph_index {
                compute_graph_proximity(gi.as_ref(), &candidate_vector_results)
            } else {
                std::collections::HashMap::new()
            };

        // Step 4: Compute feedback boosts from provided store
        let candidate_ids: Vec<RecordId> = candidates.keys().cloned().collect();
        let feedback_boosts: std::collections::HashMap<RecordId, f32> =
            if let (Some(fb_store), Some(query_vec)) = (feedback_store, query_vec.as_ref()) {
                fb_store.compute_boosts(&query_vec, &candidate_ids, &cfg.now)
            } else {
                std::collections::HashMap::new()
            };

        // Step 5: Score and rank all candidates
        // Pre-compute scope/confidence/importance filters.
        let has_scope_filter = !cfg.scope_filter.is_empty();
        let mut scored_results = Vec::new();
        let mut aggregated_candidates: Vec<(RecordId, RecallCandidate)> =
            candidates.into_iter().collect();
        aggregated_candidates.sort_by(|a, b| a.1.first_rank.cmp(&b.1.first_rank));
        for (rid, candidate) in aggregated_candidates {
            let record = candidate.record;
            // Scope filter — skip records outside requested scope(s).
            if has_scope_filter {
                let record_scope = record
                    .data
                    .get("_scope")
                    .and_then(|v| v.as_str())
                    .unwrap_or("project");
                if !cfg.scope_filter.iter().any(|s| s == record_scope) {
                    continue;
                }
            }

            // Confidence filter.
            if let Some(min_conf) = cfg.min_confidence {
                let conf = record
                    .data
                    .get("_confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
                if conf < min_conf {
                    continue;
                }
            }

            // Importance filter.
            if let Some(min_imp) = cfg.min_importance {
                let imp = crate::importance::get_importance(&record.data);
                if imp < min_imp {
                    continue;
                }
            }

            let fts_score = candidate.fts_score;
            let graph_score = graph_scores.get(rid.as_str()).copied().unwrap_or(0.0);
            let feedback_score = feedback_boosts.get(&rid).copied().unwrap_or(0.0);

            let signals = crate::scoring::SignalValues {
                vector_similarity: candidate.vector_score,
                keyword_match: fts_score,
                graph_proximity: graph_score,
                feedback_boost: feedback_score,
                preference_match: 0.0,
                activation: 0.0,
                rrf: candidate.rrf_score,
            };

            let (mut score, mut explanation) =
                crate::scoring::fuse_signals(&record, &signals, &cfg);

            // Identifier tilt: for exact-identifier queries only, add an extra
            // RRF term for the candidate's FTS rank computed with a *shrunk*
            // constant (RRF_K_FTS_IDENTIFIER ≪ 60). 1/(k+rank) grows fast as k
            // shrinks, so the top FTS hit gets a large bump while lower FTS ranks
            // fall off quickly — exactly the "exact match first" behavior the
            // task wants, without touching the vector/graph/recency signals. For
            // natural-language queries this branch never runs, so fusion is
            // byte-identical to pure RRF.
            if query_class.is_identifier() {
                if let Some(fr) = candidate.fts_rank {
                    // Shrunk FTS RRF constant — top FTS hit ≈ 0.10 here vs
                    // ≈0.017 at the default k=60, an ~6× tilt at rank 0.
                    const RRF_K_FTS_IDENTIFIER: f32 = 10.0;
                    let tilt = 1.0 / (RRF_K_FTS_IDENTIFIER + fr as f32);
                    score += tilt;
                    explanation
                        .signals
                        .push(("fts_identifier_tilt".to_string(), tilt));
                }
            }

            explanation.query_class = Some(query_class.tag());

            scored_results.push(crate::scoring::RecallResult {
                record,
                score,
                explanation,
            });
        }

        // Sort by fused score descending
        scored_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Query-Time Chunk reranking (optional). Rescores the top-K candidates
        // using chunk-level embeddings of their text, then resorts. Applied
        // BEFORE truncation so a chunk-match can promote a session that was
        // barely outside top_k under fused-score-only ranking.
        if let (Some(qtc), Some(q_vec)) = (cfg.qtc.as_ref(), query_vec.as_ref()) {
            if let Some(embedder) = self.embedder.as_ref() {
                self.apply_query_time_chunks(&mut scored_results, q_vec, embedder.as_ref(), qtc);
                scored_results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Collapse near-duplicate results so the scarce top_k
        // slots aren't spent on restatements of the same memory. Runs on the
        // full score-sorted candidate list (before truncation) so top_k
        // *distinct* results survive. Lexical SimHash — needs no vector index,
        // and catches cross-table near-dups the per-table insert-supersede
        // misses. Off by default (see DedupConfig); agent-facing paths opt in.
        if cfg.dedup.enabled {
            collapse_near_duplicates(&mut scored_results, &cfg.dedup);
        }

        // Completeness k-widening: widen when the kept set is far more
        // compressible than the post-dedup pool — a diverse cluster was cut.
        // Runs after dedup so it operates on distinct candidates, and before
        // truncation so it can recover a dropped cluster. One bounded
        // compression pass, single re-trim — cheap on the hot path.
        let mut effective_top_k = top_k;
        if cfg.dedup.completeness_widen {
            widen_k_for_completeness(&scored_results, &mut effective_top_k, &cfg.dedup);
        }

        // Truncate to the (possibly widened) top_k
        scored_results.truncate(effective_top_k);

        // Bump activation for returned results (lazy decay on read).
        if cfg.weights.activation > 0.0 {
            for result in &scored_results {
                let _ = self.bump_activation(&result.record, &cfg.activation_config);
            }
        }

        timer.finish();
        Ok(scored_results)
    }

    /// Mark a recall result as relevant (positive feedback).
    ///
    /// The feedback is stored and used to boost similar results in future queries.
    /// Requires that the query was embedded (pass the query embedding).
    /// Mark a recall result as relevant (positive feedback).
    ///
    /// Stores feedback in the internal `FeedbackStore` and uses it to
    /// boost similar results in future `recall()` queries.
    pub fn mark_relevant(&self, query_embedding: &[f32], record_id: &RecordId) -> Result<()> {
        if self.storage.get(record_id)?.is_none() {
            return Err(AxilError::NotFound(format!("record {record_id}")));
        }
        self.feedback_store
            .mark_relevant(query_embedding, record_id);
        Ok(())
    }

    /// Get the internal feedback store (for inspection/persistence).
    pub fn feedback_store(&self) -> &crate::feedback::FeedbackStore {
        &self.feedback_store
    }

    /// Auto-supersede using a pre-computed embedding vector.
    ///
    /// Returns the number of existing records marked superseded. A normal
    /// insert discards this; [`crate::portable`] import uses it to report how
    /// many local records an import demoted.
    fn auto_supersede_with_vector(&self, new_record: &Record, vector: &[f32]) -> Result<usize> {
        const SUPERSEDE_THRESHOLD: f32 = 0.92;

        let mut superseded = 0usize;
        let candidates = self.similar_to_vector(vector, 10)?;
        for (candidate, score) in &candidates {
            if candidate.id == new_record.id {
                continue;
            }
            if candidate.table != new_record.table {
                continue;
            }
            if *score < SUPERSEDE_THRESHOLD {
                continue;
            }
            if crate::importance::is_pinned(&candidate.data) {
                continue;
            }

            // Recency guard: never let an older record supersede a newer one.
            // A normal insert always stamps `created_at = now()`, so this is a
            // no-op for it (the incoming record is at least as new as anything
            // already stored). It only bites the import path, which preserves
            // the source `created_at`: an imported record must not demote a
            // fresher local near-duplicate just because they are similar.
            if new_record.created_at < candidate.created_at {
                continue;
            }

            // Check if already superseded
            if candidate
                .data
                .get("_superseded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }

            // Mark as superseded
            let mut data = candidate.data.clone();
            if let Some(obj) = data.as_object_mut() {
                obj.insert("_superseded".to_string(), serde_json::json!(true));
                obj.insert(
                    "_superseded_by".to_string(),
                    serde_json::json!(new_record.id.to_string()),
                );
            }
            if self.storage.update(&candidate.id, data).is_ok() {
                superseded += 1;
            }

            // Create graph edge if available
            if self.has_graph_index() {
                let _ = self.relate(&new_record.id, "supersedes", &candidate.id, None);
            }
        }
        Ok(superseded)
    }

    /// Auto-link a record by extracting entities and creating graph edges.
    ///
    /// Extracts entities from the record's text, creates entity nodes in the
    /// `_entities` table, and links via `→mentions→` edges. Also checks
    /// vector similarity against recent records for `→related_to→` edges.
    pub fn auto_link(
        &self,
        record_id: &RecordId,
        similarity_threshold: Option<f32>,
    ) -> Result<AutoLinkReport> {
        let gi = self.require_graph_index()?;
        let record = self
            .storage
            .get(record_id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {record_id}")))?;

        let threshold = similarity_threshold.unwrap_or(0.85);
        let mut report = AutoLinkReport::default();

        // Step 1: Entity extraction (capped to prevent unbounded work)
        const MAX_AUTO_LINK_ENTITIES: usize = 50;
        let text = record_text_for_entity(&record);
        let mut entities = crate::entity::extract_entities(&text);
        entities.truncate(MAX_AUTO_LINK_ENTITIES);
        report.entities_found = entities.len();

        use crate::util::edge_types;

        // Load entity table for all lookups to avoid creating duplicates.
        // Cache keys are canonical_ids; pre-migration rows synthesize one
        // from `name` so lookups still hit.
        let mut known_entities: std::collections::HashMap<String, RecordId> = self
            .storage
            .list("_entities", usize::MAX, 0)?
            .into_iter()
            .filter_map(|r| {
                let canonical = r
                    .data
                    .get("canonical_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        r.data
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })?;
                Some((canonical, r.id))
            })
            .collect();

        // File hint for provisional code-symbol canonical ids: pull a
        // single path-like value off the record so two records mentioning
        // the same symbol in different files don't collapse.
        let file_hint = pick_file_hint(&record.data);

        // Step 2: Create/find entity nodes and link via →mentions→
        for entity in &entities {
            let entity_id = self.find_or_create_entity_cached(
                entity,
                file_hint.as_deref(),
                &mut known_entities,
            )?;
            let edge_exists = gi
                .edges(
                    record_id.clone(),
                    Some(edge_types::MENTIONS),
                    Direction::Out,
                )
                .unwrap_or_default()
                .iter()
                .any(|e| e.to == entity_id);

            if !edge_exists {
                gi.relate(
                    record_id.clone(),
                    edge_types::MENTIONS,
                    entity_id,
                    serde_json::json!({"entity_type": entity.entity_type, "strength": 1.0}),
                )?;
                report.edges_created += 1;
            }
        }

        // Step 3: Similarity-based linking against recent records
        if let (Some(ref vi), Some(ref embedder)) = (&self.vector_index, &self.embedder) {
            if let Ok(vec) = embedder.embed(&text) {
                let similar_records = vi.search(&vec, 20)?;
                for (sim_rid, sim_score) in &similar_records {
                    if sim_rid == record_id || *sim_score < threshold {
                        continue;
                    }
                    if let Some(sim_record) = self.storage.get(sim_rid)? {
                        // Only link records in different sessions/tables
                        if sim_record.table != record.table
                            || (record.created_at - sim_record.created_at)
                                .num_hours()
                                .abs()
                                > 1
                        {
                            let edge_exists = gi
                                .edges(
                                    record_id.clone(),
                                    Some(edge_types::RELATED_TO),
                                    Direction::Out,
                                )
                                .unwrap_or_default()
                                .iter()
                                .any(|e| e.to == *sim_rid);

                            if !edge_exists {
                                gi.relate(
                                    record_id.clone(),
                                    edge_types::RELATED_TO,
                                    sim_rid.clone(),
                                    serde_json::json!({"strength": sim_score}),
                                )?;
                                report.similarity_links += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(report)
    }

    /// Public accessor for the attached graph index, if any.
    /// `axil-scip` uses this to emit SCIP-grounded edges without going
    /// through the Rust query builder — ingest is high-volume and needs
    /// direct edge writes.
    pub fn graph_index_ref(&self) -> Option<&dyn GraphIndex> {
        self.graph_index.as_deref()
    }

    /// Register an alias mapping `alias_name → canonical_id` in the given
    /// scope. Idempotent: a second call with the same tuple is a no-op.
    ///
    /// `scope` examples: `"global"`, `"lang:rust"`, `"file:src/auth.rs"`,
    /// `"crate:axil-core"`. The resolver walks scopes narrowest-first.
    ///
    /// Stored in `_scip_aliases` (separate from axil-memory's
    /// `_entity_aliases`, which uses a different `{entity, alias}` schema).
    pub fn register_entity_alias(
        &self,
        alias_name: &str,
        scope: &str,
        canonical_id: &str,
    ) -> Result<RecordId> {
        let existing = self.storage.list(SCIP_ALIAS_TABLE, usize::MAX, 0)?;
        for r in &existing {
            let a = r.data.get("alias").and_then(|v| v.as_str());
            let s = r.data.get("scope").and_then(|v| v.as_str());
            let c = r.data.get("canonical_id").and_then(|v| v.as_str());
            if a == Some(alias_name) && s == Some(scope) && c == Some(canonical_id) {
                return Ok(r.id.clone());
            }
        }
        let rec = self.insert(
            SCIP_ALIAS_TABLE,
            serde_json::json!({
                "alias": alias_name,
                "scope": scope,
                "canonical_id": canonical_id,
            }),
        )?;
        Ok(rec.id)
    }

    /// Resolve a display name to a canonical entity id, walking the provided
    /// scopes narrowest-first. Returns `None` if no alias matches in any
    /// scope. The caller is responsible for any `natural_canonical_id`
    /// fallback — this function consults the alias table only.
    pub fn resolve_entity_alias(
        &self,
        alias_name: &str,
        scopes: &[&str],
    ) -> Result<Option<String>> {
        let rows = self.storage.list(SCIP_ALIAS_TABLE, usize::MAX, 0)?;
        for scope in scopes {
            for r in &rows {
                let a = r.data.get("alias").and_then(|v| v.as_str());
                let s = r.data.get("scope").and_then(|v| v.as_str());
                let c = r.data.get("canonical_id").and_then(|v| v.as_str());
                if a == Some(alias_name) && s == Some(*scope) {
                    if let Some(canonical) = c {
                        return Ok(Some(canonical.to_string()));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Explicit entity merge: move all aliases and inbound edges from
    /// `from_canonical_id` to `to_canonical_id`, then mark the source
    /// entity row as merged. Never called silently — only via
    /// `axil entity merge`.
    pub fn merge_entities(&self, from_canonical_id: &str, to_canonical_id: &str) -> Result<usize> {
        if from_canonical_id == to_canonical_id {
            return Ok(0);
        }
        let rows = self.storage.list("_entities", usize::MAX, 0)?;
        let from_row = rows.iter().find(|r| {
            r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(from_canonical_id)
        });
        let to_row = rows
            .iter()
            .find(|r| r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(to_canonical_id));
        let (Some(from_row), Some(to_row)) = (from_row, to_row) else {
            return Err(AxilError::NotFound(format!(
                "entity merge: canonical_id not found: from={from_canonical_id}, to={to_canonical_id}"
            )));
        };

        let mut moved = 0usize;

        let aliases = self.storage.list(SCIP_ALIAS_TABLE, usize::MAX, 0)?;
        for r in &aliases {
            if r.data.get("canonical_id").and_then(|v| v.as_str()) == Some(from_canonical_id) {
                let mut new_data = r.data.clone();
                if let Value::Object(map) = &mut new_data {
                    map.insert(
                        "canonical_id".to_string(),
                        Value::String(to_canonical_id.to_string()),
                    );
                }
                self.storage.update(&r.id, new_data)?;
                moved += 1;
            }
        }

        // Re-relate edges from the source onto the target, then delete the
        // originals so the from-row leaves no graph footprint. Skipping
        // `unrelate` is what "move" is — if it fails, the source stays
        // connected and traversals can still reach the merged-away node.
        if let Some(ref gi) = self.graph_index {
            let incoming = gi.edges(from_row.id.clone(), None, Direction::In)?;
            for edge in incoming {
                // Don't carry self-edges (from == from_row, to == from_row) forward.
                if edge.from == from_row.id {
                    gi.unrelate(&edge.id)?;
                    continue;
                }
                gi.relate(
                    edge.from.clone(),
                    &edge.edge_type,
                    to_row.id.clone(),
                    edge.properties.clone(),
                )?;
                gi.unrelate(&edge.id)?;
                moved += 1;
            }
            let outgoing = gi.edges(from_row.id.clone(), None, Direction::Out)?;
            for edge in outgoing {
                gi.relate(
                    to_row.id.clone(),
                    &edge.edge_type,
                    edge.to.clone(),
                    edge.properties.clone(),
                )?;
                gi.unrelate(&edge.id)?;
                moved += 1;
            }
        }

        // Tombstone the source entity row.
        let mut tombstone = from_row.data.clone();
        if let Value::Object(map) = &mut tombstone {
            map.insert(
                "_merged_into".to_string(),
                Value::String(to_canonical_id.to_string()),
            );
        }
        self.storage.update(&from_row.id, tombstone)?;

        Ok(moved)
    }

    fn entities_table_nonempty(&self) -> bool {
        self.storage
            .count("_entities")
            .map(|c| c > 0)
            .unwrap_or(false)
    }

    fn entity_migration_done(&self) -> bool {
        self.storage
            .list("_axil_migrations", usize::MAX, 0)
            .map(|rows| {
                rows.iter().any(|r| {
                    r.data.get("name").and_then(|v| v.as_str()) == Some(ENTITY_MIGRATION_MARKER)
                })
            })
            .unwrap_or(false)
    }

    fn mark_entity_migration_done(&self) -> Result<()> {
        self.insert(
            "_axil_migrations",
            serde_json::json!({
                "name": ENTITY_MIGRATION_MARKER,
                "completed_at": chrono::Utc::now().to_rfc3339(),
            }),
        )?;
        Ok(())
    }

    /// Backfill `_entities.canonical_id` for pre-Phase-13 rows.
    ///
    /// Idempotent — only touches rows missing the field. Returns `Err` if
    /// any row write fails so the caller can avoid marking the migration
    /// complete on a partial backfill.
    pub(crate) fn migrate_entity_canonical_id(&self) -> Result<usize> {
        let mut updated = 0usize;
        let rows = match self.storage.list("_entities", usize::MAX, 0) {
            Ok(r) => r,
            Err(_) => return Ok(0),
        };
        for row in rows {
            if row.data.get("canonical_id").is_some() {
                continue;
            }
            let Some(name) = row.data.get("name").and_then(|v| v.as_str()) else {
                continue;
            };
            let mut new_data = row.data.clone();
            if let Value::Object(map) = &mut new_data {
                map.insert(
                    "canonical_id".to_string(),
                    Value::String(crate::entity::natural_canonical_id(name)),
                );
            }
            self.storage.update(&row.id, new_data)?;
            updated += 1;
        }
        Ok(updated)
    }

    /// Find an existing entity node or create a new one, using an in-memory cache.
    ///
    /// Entities are keyed by `canonical_id`, not by `name` — code symbols
    /// with identical display names (e.g. `login` in Rust and Python) get
    /// distinct ids so they cannot silently merge. Natural-language
    /// entities use `canonical_id = name`.
    ///
    /// `file_hint` scopes provisional code-symbol ids so two files
    /// mentioning `login` produce distinct provisional rows until SCIP
    /// ingest grounds them.
    fn find_or_create_entity_cached(
        &self,
        entity: &crate::entity::Entity,
        file_hint: Option<&str>,
        cache: &mut std::collections::HashMap<String, RecordId>,
    ) -> Result<RecordId> {
        let canonical_id = entity_canonical_id(entity, file_hint);

        // Check cache first (keyed by canonical_id, not display name).
        if let Some(id) = cache.get(&canonical_id) {
            return Ok(id.clone());
        }

        let record = self.insert(
            "_entities",
            serde_json::json!({
                "name": entity.name,
                "canonical_id": canonical_id,
                "entity_type": entity.entity_type,
                "source_text": entity.source_text,
            }),
        )?;
        cache.insert(canonical_id, record.id.clone());
        Ok(record.id)
    }

    /// Detect contradictions and superseding for a newly inserted record.
    ///
    /// Checks vector similarity against existing records in the same table.
    /// Returns conflicts found and creates graph edges for them.
    pub fn detect_conflicts(
        &self,
        record_id: &RecordId,
    ) -> Result<Vec<crate::consolidation::ConflictResult>> {
        let record = self
            .storage
            .get(record_id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {record_id}")))?;

        let vi = self.require_vector_index()?;
        let embedder = self.require_embedder()?;

        let text = record_text_for_entity(&record);
        let query_vec = embedder.embed(&text)?;

        // Find similar records
        let similar = vi.search(&query_vec, 20)?;
        let mut conflicts = Vec::new();

        for (sim_rid, sim_score) in &similar {
            if sim_rid == record_id {
                continue;
            }
            if let Some(existing) = self.storage.get(sim_rid)? {
                let result = crate::consolidation::check_conflict(&record, &existing, *sim_score);
                match &result {
                    crate::consolidation::ConflictResult::Supersedes { .. } => {
                        if let Some(ref gi) = self.graph_index {
                            gi.relate(
                                record_id.clone(),
                                crate::util::edge_types::SUPERSEDES,
                                sim_rid.clone(),
                                serde_json::json!({"similarity": sim_score}),
                            )?;
                        }
                        // Mark old record as superseded via data
                        if let Some(old_record) = self.storage.get(sim_rid)? {
                            let mut new_data = old_record.data.clone();
                            if let serde_json::Value::Object(ref mut map) = new_data {
                                map.insert("_superseded".to_string(), serde_json::json!(true));
                                map.insert(
                                    "_superseded_by".to_string(),
                                    serde_json::json!(record_id.as_str()),
                                );
                            }
                            self.update(sim_rid, new_data)?;
                        }
                        conflicts.push(result);
                    }
                    crate::consolidation::ConflictResult::Contradicts { .. } => {
                        if let Some(ref gi) = self.graph_index {
                            gi.relate(
                                record_id.clone(),
                                crate::util::edge_types::CONTRADICTS,
                                sim_rid.clone(),
                                serde_json::json!({"similarity": sim_score}),
                            )?;
                        }
                        conflicts.push(result);
                    }
                    crate::consolidation::ConflictResult::Novel => {}
                }
            }
        }

        Ok(conflicts)
    }

    /// Consolidate facts about an entity.
    ///
    /// Finds all records mentioning the given entity, groups them by conflict
    /// relationships, and produces a consolidated summary. The summary is
    /// stored as a new record in the `_consolidated` table.
    pub fn consolidate_entity(
        &self,
        entity_name: &str,
    ) -> Result<Option<crate::consolidation::ConsolidatedFact>> {
        // Find entity node
        let entities = self.storage.list("_entities", usize::MAX, 0)?;
        let entity_record = entities
            .iter()
            .find(|r| r.data.get("name").and_then(|v| v.as_str()) == Some(entity_name));

        let entity_id = match entity_record {
            Some(r) => r.id.clone(),
            None => return Ok(None),
        };

        // Find records that mention this entity
        let gi = self.require_graph_index()?;
        use crate::util::edge_types;
        let mentioning_ids = gi.neighbors(entity_id, Some(edge_types::MENTIONS), Direction::In)?;

        if mentioning_ids.is_empty() {
            return Ok(None);
        }

        // Resolve records and check for conflicts
        let mut facts = Vec::new();
        for rid in &mentioning_ids {
            if let Some(record) = self.storage.get(rid)? {
                let supersede_edges = gi
                    .edges(rid.clone(), Some(edge_types::SUPERSEDES), Direction::Out)
                    .unwrap_or_default();
                let contradict_edges = gi
                    .edges(rid.clone(), Some(edge_types::CONTRADICTS), Direction::Out)
                    .unwrap_or_default();

                let conflict = if !supersede_edges.is_empty() {
                    let sim = supersede_edges[0]
                        .properties
                        .get("similarity")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.95) as f32;
                    crate::consolidation::ConflictResult::Supersedes {
                        old_record_id: supersede_edges[0].to.clone(),
                        similarity: sim,
                    }
                } else if !contradict_edges.is_empty() {
                    let sim = contradict_edges[0]
                        .properties
                        .get("similarity")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.93) as f32;
                    crate::consolidation::ConflictResult::Contradicts {
                        existing_record_id: contradict_edges[0].to.clone(),
                        similarity: sim,
                    }
                } else {
                    crate::consolidation::ConflictResult::Novel
                };

                facts.push((record, conflict));
            }
        }

        let consolidated = crate::consolidation::consolidate_facts(entity_name, &facts);

        // Store consolidated record and create edges back to sources.
        // Remove any previous consolidation for this entity first to avoid duplicates.
        if let Some(ref cf) = consolidated {
            // Delete previous consolidated records for this entity.
            if let Ok(existing) = self.storage.list("_consolidated", usize::MAX, 0) {
                for old in &existing {
                    if old.data.get("entity").and_then(|v| v.as_str()) == Some(entity_name) {
                        // Remove old consolidated_into edges pointing to this record.
                        if self.has_graph_index() {
                            if let Ok(edges) = self.edges(
                                &old.id,
                                Some(crate::util::edge_types::CONSOLIDATED_INTO),
                                Direction::In,
                            ) {
                                for edge in &edges {
                                    let _ = self.unrelate(&edge.id);
                                }
                            }
                        }
                        let _ = self.delete(&old.id);
                    }
                }
            }

            if let Ok(summary_record) = self.insert(
                "_consolidated",
                serde_json::json!({
                    "entity": cf.entity,
                    "summary": cf.summary,
                    "source_count": cf.source_ids.len(),
                    "latest_at": cf.latest_at.to_rfc3339(),
                }),
            ) {
                if self.has_graph_index() {
                    for source_id in &cf.source_ids {
                        if let Err(e) = self.relate(
                            source_id,
                            crate::util::edge_types::CONSOLIDATED_INTO,
                            &summary_record.id,
                            None,
                        ) {
                            eprintln!("warning: failed to link source to consolidated: {e}");
                        }
                    }
                }
            }
        }

        Ok(consolidated)
    }

    /// Get entity history — show how facts about an entity evolved over time.
    pub fn entity_history(&self, entity_name: &str) -> Result<Vec<(Record, String)>> {
        // Find entity node
        let entities = self.storage.list("_entities", usize::MAX, 0)?;
        let entity_record = entities
            .iter()
            .find(|r| r.data.get("name").and_then(|v| v.as_str()) == Some(entity_name));

        let entity_id = match entity_record {
            Some(r) => r.id.clone(),
            None => return Ok(Vec::new()),
        };

        // Find mentioning records
        let gi = self.require_graph_index()?;
        let mentioning_ids = gi.neighbors(
            entity_id,
            Some(crate::util::edge_types::MENTIONS),
            Direction::In,
        )?;

        let mut history = Vec::new();
        for rid in &mentioning_ids {
            if let Some(record) = self.storage.get(rid)? {
                // Determine status from data (where detect_conflicts persists it)
                let status =
                    if record.data.get("_superseded").and_then(|v| v.as_bool()) == Some(true) {
                        "superseded".to_string()
                    } else {
                        "current".to_string()
                    };
                history.push((record, status));
            }
        }

        // Sort by creation time
        history.sort_by(|a, b| a.0.created_at.cmp(&b.0.created_at));
        Ok(history)
    }

    /// Warm up the database: rebuild indexes and pre-compute popular recalls.
    pub fn warm_up(&self) -> Result<WarmUpReport> {
        let mut report = WarmUpReport::default();

        // Rebuild vector index if needed
        if let Some(ref vi) = self.vector_index {
            let deleted = vi.deleted_count();
            if deleted > 0 {
                report.vector_rebuilt = vi.rebuild().is_ok();
            }
        }

        report.warmed_up = true;
        Ok(report)
    }

    /// Get the embedder reference (for external use like feedback).
    pub fn embedder(&self) -> Option<&dyn crate::plugin::TextEmbedder> {
        self.embedder.as_ref().map(|a| a.as_ref())
    }

    /// Get the storage reference (for advanced queries).
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Read change-data-capture events from the durable `_changelog` tape, in
    /// commit order, for changes strictly after `cursor` (a ULID `change_id`).
    ///
    /// Pass `None` to read from the oldest retained entry. The `change_id` of
    /// the last returned entry is the cursor for the next pull. Only available
    /// with the off-by-default `cdc` feature; without it there is no tape.
    #[cfg(feature = "cdc")]
    pub fn changes_since(
        &self,
        cursor: Option<&str>,
        limit: usize,
    ) -> Result<Vec<crate::storage::ChangeEntry>> {
        self.storage.changes_since(cursor, limit)
    }

    /// Enable or disable full-body (`before`/`after`) capture on the CDC tape.
    /// Id-only capture is the default; value capture roughly doubles write cost.
    #[cfg(feature = "cdc")]
    pub fn set_cdc_capture_values(&self, enabled: bool) {
        self.storage.set_cdc_capture_values(enabled);
    }

    /// Enable or disable the durable semantic event log.
    ///
    /// Off by default even with the `event-log` feature compiled in: the tape is
    /// a write-amplifier (an extra committed write per allowlisted event), so the
    /// caller opts in. When off, the capture hook is a single relaxed atomic load
    /// and never touches storage.
    #[cfg(feature = "event-log")]
    pub fn set_event_log_enabled(&self, enabled: bool) {
        self.event_log_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the semantic event log is currently capturing.
    #[cfg(feature = "event-log")]
    pub fn event_log_enabled(&self) -> bool {
        self.event_log_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Capture a committed write as a semantic event when it matches the curated
    /// allowlist. No-op unless the event log is enabled.
    ///
    /// Called from `insert` / `update` / `delete` *after* the storage write
    /// commits, so it only ever records facts that are already durable. Captures
    /// allowlisted `_`-prefixed events (belief revisions, checkpoint writes) that
    /// the per-record audit log deliberately skips. Best-effort: a failed append
    /// never fails the originating write.
    #[cfg(feature = "event-log")]
    fn capture_semantic_event(&self, op: &str, record: &Record) {
        if !self.event_log_enabled() {
            return;
        }
        let Some(kind) = crate::event_log::classify(op, record) else {
            return;
        };
        let cursor = self.event_cursor.next();
        let event = crate::event_log::SemanticEvent {
            cursor: cursor.clone(),
            kind: kind.to_string(),
            op: op.to_string(),
            table: record.table.clone(),
            record_id: record.id.to_string(),
            agent_id: crate::event_log::agent_id_of(record),
        };
        if let Ok(bytes) = serde_json::to_vec(&event) {
            let _ = self.storage.append_event(&cursor, &bytes);
        }
    }

    /// Pull committed semantic events strictly after `cursor`, oldest first.
    ///
    /// This is the engine behind the `recall_delta` MCP tool: a second agent
    /// passes the last cursor it saw and gets back what changed since — belief
    /// revisions, decision supersessions, error fixes, checkpoint writes — under
    /// a monotonic ULID order it can resume from. Pass `None` to read from the
    /// oldest retained event.
    ///
    /// `exclude_agent` drops events whose `agent_id` matches (e.g. "skip my own
    /// writes"). This is an ergonomic filter over **committed facts only** — it
    /// does not read record bodies and does not relax cross-agent session
    /// isolation; an agent still resolves a returned `record_id` through the
    /// normal access path. Only available with the `event-log` feature.
    #[cfg(feature = "event-log")]
    pub fn recall_delta(
        &self,
        cursor: Option<&str>,
        exclude_agent: Option<&str>,
        limit: usize,
    ) -> Result<Vec<crate::event_log::SemanticEvent>> {
        // Over-read when filtering so an excluded run can't starve the page; the
        // tape is a high-signal feed so the slop is small.
        let scan_limit = if exclude_agent.is_some() {
            limit.saturating_mul(4).max(limit)
        } else {
            limit
        };
        let raw = self.storage.events_since(cursor, scan_limit)?;
        let mut out = Vec::with_capacity(raw.len().min(limit));
        for bytes in raw {
            let Ok(event) = serde_json::from_slice::<crate::event_log::SemanticEvent>(&bytes) else {
                continue;
            };
            if let Some(excl) = exclude_agent {
                if event.agent_id.as_deref() == Some(excl) {
                    continue;
                }
            }
            out.push(event);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    /// Trim the semantic event log to at most `max` retained entries (oldest
    /// evicted). Driven off the write path by the maintenance thread so an
    /// always-on event log can't grow the core file unboundedly.
    #[cfg(feature = "event-log")]
    pub fn trim_event_log(&self, max: usize) -> Result<()> {
        self.storage.trim_event_log(max)
    }

    /// Number of retained entries on the semantic event log tape.
    #[cfg(feature = "event-log")]
    pub fn event_log_len(&self) -> Result<usize> {
        self.storage.event_log_len()
    }

    /// Append one `insert` audit-log entry per record. Mirrors what
    /// `Axil::insert` does inside the per-record path so bulk-import
    /// flows (e.g. SCIP ingest) keep the audit trail complete without
    /// reaching into the private `audit` helper.
    pub fn audit_inserts(&self, records: &[Record]) {
        if !self
            .audit_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        // Unlike the per-record `audit()`, this deliberately records the
        // `_`-prefixed rows a bulk import created (`_entities`,
        // `_scip_aliases`) — surfacing bulk imports in `axil audit-log`
        // so it has no "SCIP-shaped hole" is exactly this method's job.
        for r in records {
            self.write_audit_entry("insert", &r.id, &r.table);
        }
    }

    /// Atlas publish: every `_entities` row with a
    /// non-empty, non-provisional canonical_id flows to the workspace
    /// control plane so cross-project recall can resolve "who has this
    /// entity" without fan-out. Publisher contract is best-effort +
    /// non-blocking; failure never affects the caller.
    ///
    /// Public so bulk-import paths (e.g. SCIP ingest) that bypass the
    /// per-record `insert` flow keep Atlas in the loop without
    /// reimplementing the prefix filter — single source of truth.
    pub fn publish_canonical_for_record(&self, record: &Record) {
        let Some(publisher) = self.canonical_publisher.as_ref() else {
            return;
        };
        if record.table != "_entities" {
            return;
        }
        let Some(canonical) = record.data.get("canonical_id").and_then(|v| v.as_str()) else {
            return;
        };
        if canonical.is_empty() || canonical.starts_with("provisional:") {
            return;
        }
        publisher.publish(canonical);
    }

    /// Apply `read` and/or `write` consent scopes to a single record.
    ///
    /// Passing `None` for an axis leaves it untouched. Returns the updated
    /// record. A row is appended to `_consent_log` regardless of the
    /// general `audit_log` toggle so compliance trails survive even when
    /// regular auditing is off.
    pub fn set_record_consent(
        &self,
        id: &RecordId,
        read: Option<Value>,
        write: Option<Value>,
    ) -> Result<Record> {
        let mut existing = self
            .storage
            .get(id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {id}")))?;
        existing.set_consent(read, write);
        let updated = self.storage.set_metadata(id, existing.metadata.clone())?;
        self.audit("consent_set", id, &updated.table);
        self.storage.insert(&Record::new(
            "_consent_log",
            serde_json::json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "record_id": id.to_string(),
                "table": updated.table,
                "read_consent": updated.read_consent_raw(),
                "write_consent": updated.write_consent_raw(),
            }),
        ))?;
        Ok(updated)
    }

    /// Read back consent scopes as JSON (always returns concrete values,
    /// substituting the default scope when unset).
    pub fn get_record_consent(&self, id: &RecordId) -> Result<(Value, Value)> {
        let record = self
            .storage
            .get(id)?
            .ok_or_else(|| AxilError::NotFound(format!("record {id}")))?;
        Ok((record.read_consent_raw(), record.write_consent_raw()))
    }

    /// Set a table-level default consent scope. Stored in the internal
    /// `_axil_consent_defaults` table and keyed by table name. Records
    /// inserted with no explicit consent inherit these defaults.
    pub fn set_consent_default(
        &self,
        table: &str,
        read: Option<Value>,
        write: Option<Value>,
    ) -> Result<()> {
        // Load any prior row first: an update that supplies only one axis
        // (`--read` or `--write`) must preserve the other rather than
        // silently dropping it when the row is rebuilt.
        let existing = self
            .storage
            .list("_axil_consent_defaults", usize::MAX, 0)
            .unwrap_or_default();
        let prior = existing
            .iter()
            .find(|row| row.data.get("table").and_then(|v| v.as_str()) == Some(table));

        let mut entry = serde_json::json!({
            "table": table,
            "updated_at": chrono::Utc::now().to_rfc3339(),
        });
        if let Some(r) = read.or_else(|| prior.and_then(|p| p.data.get("read").cloned())) {
            entry["read"] = r;
        }
        if let Some(w) = write.or_else(|| prior.and_then(|p| p.data.get("write").cloned())) {
            entry["write"] = w;
        }

        // Replace the prior row(s) so the defaults table stays flat
        // (one row per table, no supersession games).
        for row in &existing {
            if row.data.get("table").and_then(|v| v.as_str()) == Some(table) {
                let _ = self.storage.delete(&row.id);
            }
        }
        self.storage
            .insert(&Record::new("_axil_consent_defaults", entry))?;
        Ok(())
    }

    /// Read the default consent scope for a table.
    pub fn consent_default(&self, table: &str) -> Result<(Option<Value>, Option<Value>)> {
        let existing = self
            .storage
            .list("_axil_consent_defaults", usize::MAX, 0)
            .unwrap_or_default();
        for row in existing {
            if row.data.get("table").and_then(|v| v.as_str()) == Some(table) {
                let read = row.data.get("read").cloned();
                let write = row.data.get("write").cloned();
                return Ok((read, write));
            }
        }
        Ok((None, None))
    }

    /// Apply this DB's per-table defaults to the given record in place.
    /// Callers typically invoke this right after building a new record.
    pub fn apply_consent_defaults(&self, record: &mut Record) -> Result<()> {
        let (read, write) = self.consent_default(&record.table)?;
        if read.is_some() || write.is_some() {
            record.set_consent(read, write);
        }
        Ok(())
    }

    /// Insert an entity bridge into the `_entity_bridges` table. If an
    /// existing row has the same (local, remote_ws, remote_member,
    /// remote_canonical) tuple, it is replaced — bridges are idempotent
    /// by identity, not by insertion.
    pub fn upsert_bridge(&self, bridge: &Value) -> Result<Record> {
        let key = bridge_key(bridge).ok_or_else(|| {
            AxilError::InvalidQuery("bridge must include local_canonical + remote_* fields".into())
        })?;
        let existing = self
            .storage
            .list("_entity_bridges", usize::MAX, 0)
            .unwrap_or_default();
        for row in existing {
            if bridge_key(&row.data).as_deref() == Some(key.as_str()) {
                let _ = self.storage.delete(&row.id);
            }
        }
        let record = Record::new("_entity_bridges", bridge.clone());
        self.storage.insert(&record)?;
        Ok(record)
    }

    /// List bridges, optionally filtered by local canonical id and/or remote member label.
    pub fn list_bridges(
        &self,
        local_canonical: Option<&str>,
        remote_member_id: Option<&str>,
    ) -> Result<Vec<Record>> {
        let rows = self
            .storage
            .list("_entity_bridges", usize::MAX, 0)
            .unwrap_or_default();
        Ok(rows
            .into_iter()
            .filter(|r| {
                if let Some(lc) = local_canonical {
                    if r.data.get("local_canonical").and_then(|v| v.as_str()) != Some(lc) {
                        return false;
                    }
                }
                if let Some(m) = remote_member_id {
                    if r.data.get("remote_member_id").and_then(|v| v.as_str()) != Some(m) {
                        return false;
                    }
                }
                true
            })
            .collect())
    }

    /// Mark bridges as `dangling` when their remote canonical id no
    /// longer appears locally (best-effort check: can only verify
    /// bridges whose "local" side references an entity this DB knows
    /// about). Returns (verified, dangling_marked).
    pub fn verify_bridges(&self) -> Result<(usize, usize)> {
        let bridges = self
            .storage
            .list("_entity_bridges", usize::MAX, 0)
            .unwrap_or_default();
        let entities = self
            .storage
            .list("_entities", usize::MAX, 0)
            .unwrap_or_default();
        let canonical_ids: std::collections::HashSet<String> = entities
            .iter()
            .filter_map(|r| {
                r.data
                    .get("canonical_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        let mut verified = 0;
        let mut dangling = 0;
        for bridge in bridges {
            let local = bridge.data.get("local_canonical").and_then(|v| v.as_str());
            let already_dangling = bridge
                .data
                .get("dangling")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let still_present = local.map(|l| canonical_ids.contains(l)).unwrap_or(false);

            if still_present {
                verified += 1;
                if already_dangling {
                    let mut new_data = bridge.data.clone();
                    if let Value::Object(map) = &mut new_data {
                        map.insert("dangling".to_string(), Value::Bool(false));
                    }
                    self.storage.update(&bridge.id, new_data)?;
                }
            } else {
                dangling += 1;
                if !already_dangling {
                    let mut new_data = bridge.data.clone();
                    if let Value::Object(map) = &mut new_data {
                        map.insert("dangling".to_string(), Value::Bool(true));
                    }
                    self.storage.update(&bridge.id, new_data)?;
                }
            }
        }
        Ok((verified, dangling))
    }
}

fn bridge_key(bridge: &Value) -> Option<String> {
    let lc = bridge.get("local_canonical")?.as_str()?;
    let rws = bridge.get("remote_workspace_id")?.as_str()?;
    let rm = bridge.get("remote_member_id")?.as_str()?;
    let rc = bridge.get("remote_canonical")?.as_str()?;
    Some(format!("{lc}|{rws}|{rm}|{rc}"))
}

/// Recursively compute the total size of files in a directory.
fn dir_size(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let meta = e.metadata().ok();
                    let is_dir = meta.as_ref().is_some_and(|m| m.is_dir());
                    if is_dir {
                        dir_size(&e.path())
                    } else {
                        meta.map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

/// Check if a record is marked as superseded.
fn is_superseded_record(record: &Record) -> bool {
    // Check data field (where detect_conflicts persists it)
    if record.data.get("_superseded").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    // Also check metadata for backward compatibility
    record
        .metadata
        .as_ref()
        .is_some_and(|meta| meta.get("superseded").and_then(|v| v.as_bool()) == Some(true))
}

/// Check if a record is expired (has a valid_until in the past).
fn is_expired_record(record: &Record, now: &chrono::DateTime<chrono::Utc>) -> bool {
    // Check metadata.valid_until
    if let Some(ref meta) = record.metadata {
        if let Some(vu) = meta.get("valid_until").and_then(|v| v.as_str()) {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(vu) {
                if dt < *now {
                    return true;
                }
            }
        }
    }
    // Check data.valid_until
    if let Some(vu) = record.data.get("valid_until").and_then(|v| v.as_str()) {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(vu) {
            if dt < *now {
                return true;
            }
        }
    }
    false
}

/// Build a trend metric from start and end values.
fn make_trend(start: f64, end: f64) -> crate::diagnostics::MetricTrend {
    let change = if start > 0.0 {
        let pct = ((end - start) / start) * 100.0;
        if pct >= 0.0 {
            format!("+{:.0}%", pct)
        } else {
            format!("{:.0}%", pct)
        }
    } else if end > 0.0 {
        "+inf".to_string()
    } else {
        "0%".to_string()
    };
    crate::diagnostics::MetricTrend {
        start,
        end,
        change,
        alert: None,
    }
}

/// Report from auto-linking a record.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AutoLinkReport {
    /// Number of entities extracted from the record text.
    pub entities_found: usize,
    /// Number of graph edges created for entity mentions.
    pub edges_created: usize,
    /// Number of similarity-based links created.
    pub similarity_links: usize,
}

/// Report from warming up the database.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WarmUpReport {
    /// Whether the warm-up completed.
    pub warmed_up: bool,
    /// Whether the vector index was rebuilt.
    pub vector_rebuilt: bool,
}

/// Compute graph proximity scores for a set of candidate records.
///
/// For each candidate, checks how many other candidates are reachable
/// within 2 hops. More connections = higher graph proximity.
fn compute_graph_proximity(
    gi: &dyn GraphIndex,
    candidates: &[(RecordId, f32)],
) -> std::collections::HashMap<String, f32> {
    let mut scores = std::collections::HashMap::new();
    let ids: std::collections::HashSet<String> = candidates
        .iter()
        .map(|(rid, _)| rid.as_str().to_string())
        .collect();

    if ids.len() <= 1 {
        return scores;
    }

    for (rid, _) in candidates {
        // Check 1-hop neighbors
        let neighbors = gi
            .neighbors(rid.clone(), None, Direction::Both)
            .unwrap_or_default();

        let connected = neighbors
            .iter()
            .filter(|nid| ids.contains(nid.as_str()))
            .count();

        if connected > 0 {
            let proximity = (connected as f32 / ids.len() as f32).min(1.0);
            scores.insert(rid.as_str().to_string(), proximity);
        }
    }

    scores
}

/// Extract text from a record for entity extraction.
fn record_text_for_entity(record: &Record) -> String {
    crate::util::record_text(record)
}

/// Collapse near-duplicate recall results in place.
///
/// Walks the score-sorted list keeping the first (highest-scored) member of
/// each near-duplicate cluster and dropping the rest, so the scarce `top_k`
/// slots aren't spent on near-identical restatements. Membership is decided by
/// lexical SimHash within `cfg.hamming_threshold` bits (greedy single-link
/// clustering), and is **scoped to the same `table`**: two records only
/// collapse if they live in the same table. This keeps a downstream
/// `--table`/table-scoped filter correct — recall runs before that filter, so
/// collapsing a wanted-table record into a higher-scored record in another
/// table (which the filter then drops) would silently lose it.
///
/// The collapse is intentionally *silent and lossless-in-spirit*: at the
/// conservative default threshold only near-exact duplicates (case/whitespace/
/// punctuation/tiny-edit variants of the same text) merge, so the kept
/// representative carries essentially the same content as the ones dropped.
/// Results whose normalized text is shorter than `cfg.min_text_len` are passed
/// through untouched (SimHash is unreliable on very short strings). The result
/// records are not mutated, so no recall-ephemeral state can leak into storage.
fn collapse_near_duplicates(
    results: &mut Vec<crate::scoring::RecallResult>,
    cfg: &crate::scoring::DedupConfig,
) {
    if results.len() < 2 {
        return;
    }
    // (fingerprint, index of the kept representative in `out`).
    let mut kept: Vec<(u64, usize)> = Vec::with_capacity(results.len());
    let mut out: Vec<crate::scoring::RecallResult> = Vec::with_capacity(results.len());
    for r in std::mem::take(results) {
        let norm = crate::simhash::normalize(&crate::util::record_text(&r.record));
        if norm.chars().count() < cfg.min_text_len {
            out.push(r);
            continue;
        }
        let fp = crate::simhash::simhash(&norm);
        let is_dup = kept.iter().any(|(kfp, idx)| {
            crate::simhash::hamming(*kfp, fp) <= cfg.hamming_threshold
                && out[*idx].record.table == r.record.table
        });
        if is_dup {
            // Near-exact, same-table duplicate of a higher-scored representative
            // already kept — drop it so it doesn't consume a top_k slot.
            continue;
        }
        kept.push((fp, out.len()));
        out.push(r);
    }
    *results = out;
}

/// Minimum kept-text length (chars) below which completeness widening is a
/// no-op. DEFLATE ratios on a handful of bytes are dominated by header
/// overhead and say nothing about content diversity.
const WIDEN_MIN_KEPT_CHARS: usize = 64;

/// DEFLATE (level 1) compression ratio of `text`: `compressed_len /
/// original_len`. A *lower* ratio means more internal redundancy (more
/// compressible). Returns `1.0` for empty input (nothing to compress).
///
/// Level 1 is the spec's "zlib L1" — cheapest pass, and the ratio *gap*
/// between two texts is what matters, not the absolute ratio, so a fast level
/// is sufficient.
fn deflate_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }
    use std::io::Write;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(1));
    if encoder.write_all(bytes).is_err() {
        return 1.0;
    }
    match encoder.finish() {
        Ok(compressed) => compressed.len() as f64 / bytes.len() as f64,
        Err(_) => 1.0,
    }
}

/// Decide whether the top-`top_k` cut over `pool_texts` dropped a diverse
/// cluster, and if so return the widened k to re-trim to.
///
/// `pool_texts` is the per-result text of the full (post-dedup) candidate pool,
/// already sorted best-first. Returns `Some(new_k)` only when widening should
/// happen, where `new_k > top_k`; `None` is a no-op.
///
/// Heuristic (spec 20.4): if the kept top-k subset compresses *materially
/// better* than the whole pool — `pool_ratio - kept_ratio > threshold` — then
/// the kept set is far more redundant than the larger pool, so a distinct
/// diverse cluster was likely cut. Widen k conservatively (toward the pool, by
/// a bounded amount) and re-trim once.
fn completeness_widen_target(pool_texts: &[String], top_k: usize, threshold: f32) -> Option<usize> {
    let candidate_count = pool_texts.len();
    // Nothing past the cut to recover, or a degenerate cut.
    if top_k == 0 || candidate_count <= top_k {
        return None;
    }

    let kept: String = pool_texts[..top_k].join("\n");
    // Too little kept text for a DEFLATE ratio to mean anything.
    if kept.chars().count() < WIDEN_MIN_KEPT_CHARS {
        return None;
    }
    let cand: String = pool_texts.join("\n");

    let kept_ratio = deflate_ratio(&kept);
    let pool_ratio = deflate_ratio(&cand);

    // Widen when the kept set is far more compressible (redundant) than the
    // pool it was cut from — a diverse cluster sits just past the cut.
    if (pool_ratio - kept_ratio) as f32 > threshold {
        // Conservative, bounded, single-shot: bump toward the pool by up to
        // 1.5x top_k, capped by what's actually available.
        let widened = ((top_k as f64) * 1.5).ceil() as usize;
        let new_k = widened.min(candidate_count).max(top_k + 1);
        Some(new_k)
    } else {
        None
    }
}

/// Widen the top-k cut over `results` (sorted best-first) when the kept subset
/// is suspiciously more compressible than the full candidate pool — a sign a
/// distinct diverse cluster would be dropped by a naive `truncate(top_k)`.
///
/// Mutates `top_k` in place (the caller then truncates to it). Bounded to a
/// single re-trim and one extra compression pass over the pool/kept text, so
/// it stays cheap on the hot recall path.
fn widen_k_for_completeness(
    results: &[crate::scoring::RecallResult],
    top_k: &mut usize,
    cfg: &crate::scoring::DedupConfig,
) {
    if results.len() <= *top_k {
        return;
    }
    let texts: Vec<String> = results
        .iter()
        .map(|r| crate::util::record_text(&r.record))
        .collect();
    if let Some(new_k) = completeness_widen_target(&texts, *top_k, cfg.widen_threshold) {
        *top_k = new_k;
    }
}

/// Split text into overlapping char windows. Respects UTF-8 boundaries.
/// Returns a single chunk when text is shorter than `max_chars`.
fn chunk_by_chars(text: &str, max_chars: usize, stride: usize) -> Vec<String> {
    if max_chars == 0 || text.is_empty() {
        return Vec::new();
    }
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }
    // Guard against zero-stride infinite loops.
    let stride = stride.max(1).min(max_chars);
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let start = text.floor_char_boundary(pos);
        let end = text.floor_char_boundary(pos.saturating_add(max_chars).min(text.len()));
        if start >= end {
            break;
        }
        out.push(text[start..end].to_string());
        if end >= text.len() {
            break;
        }
        pos = pos.saturating_add(stride);
    }
    out
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(q: &[f32], q_norm: f32, v: &[f32]) -> f32 {
    if q.len() != v.len() || q_norm == 0.0 {
        return 0.0;
    }
    let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let v_norm = l2_norm(v);
    if v_norm == 0.0 {
        return 0.0;
    }
    dot / (q_norm * v_norm)
}

fn parse_source_record_id(record: &Record) -> Option<RecordId> {
    record
        .data
        .get("source_record")
        .and_then(|v| v.as_str())
        .and_then(|s| RecordId::from_string(s).ok())
}

fn recall_chunk_ids(record: &Record) -> Vec<RecordId> {
    record
        .metadata
        .as_ref()
        .and_then(|meta| meta.get("_recall_chunk_ids"))
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str())
        .filter_map(|s| RecordId::from_string(s).ok())
        .collect()
}

#[derive(Clone)]
struct RecallCandidate {
    record: Record,
    vector_score: f32,
    fts_score: f32,
    first_rank: usize,
    /// Reciprocal Rank Fusion score combining vector + FTS rank positions.
    /// rrf_score = sum(1/(k + rank)) across each list the candidate appears in.
    rrf_score: f32,
    /// 0-based rank of this candidate in the FTS list, if it appeared there.
    /// Used to apply the identifier tilt — an extra RRF term computed with a
    /// shrunk constant so exact lexical matches dominate identifier lookups.
    fts_rank: Option<usize>,
}

impl Axil {
    /// Sync recall chunks for an entire batch of records in one pass.
    ///
    /// Flattens every chunk across every record, issues one micro-batched
    /// embed call chain, then fans vectors and chunk records back out. This
    /// replaces the per-record `sync_recall_chunks_for_record` loop in
    /// `insert_batch_records`, which paid the ONNX session-mutex cost once
    /// per record. For the LongMemEval bench (50 sessions × ~7 chunks each)
    /// the flat path cuts insert wall-clock roughly 5-8×.
    fn batch_sync_recall_chunks(&self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        if !self.has_vector_index() && !self.has_fts_index() {
            return Ok(());
        }

        let has_vector = self.has_vector_index() && self.has_embedder();
        let has_fts = self.has_fts_index();

        // For each source record, compute its chunks and assemble chunk
        // Records. Records whose text produces ≤1 chunk get a cleared
        // `_recall_chunk_ids` metadata write and are otherwise skipped.
        struct RecordPlan {
            source_idx: usize,
            chunk_records: Vec<Record>,
            chunk_texts: Vec<String>,
        }
        let mut plans: Vec<RecordPlan> = Vec::with_capacity(records.len());

        for (source_idx, record) in records.iter().enumerate() {
            if record.table.starts_with('_') {
                continue;
            }
            // Drop any prior chunks up front so reindexing stays consistent.
            self.delete_recall_chunks_for_source(record)?;

            let text = crate::util::searchable_text(&record.data);
            let chunks = crate::util::overlapping_chunks(
                &text,
                RECALL_CHUNK_MAX_BYTES,
                RECALL_CHUNK_OVERLAP_BYTES,
            );
            if chunks.len() <= 1 {
                self.persist_recall_chunk_ids(record, &[])?;
                continue;
            }

            let chunk_count = chunks.len();
            let chunk_records: Vec<Record> = chunks
                .iter()
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut r = Record::new(
                        RECALL_CHUNKS_TABLE,
                        serde_json::json!({
                            "source_record": record.id.to_string(),
                            "source_table": record.table,
                            "chunk_idx": chunk_idx,
                            "chunk_count": chunk_count,
                            "content": chunk,
                        }),
                    );
                    r.created_at = record.created_at;
                    r.updated_at = record.created_at;
                    r
                })
                .collect();

            plans.push(RecordPlan {
                source_idx,
                chunk_records,
                chunk_texts: chunks,
            });
        }

        if plans.is_empty() {
            return Ok(());
        }

        // Single batch-insert for every chunk record across the whole input.
        let all_chunk_records: Vec<Record> = plans
            .iter()
            .flat_map(|p| p.chunk_records.iter().cloned())
            .collect();
        self.storage.insert_batch(&all_chunk_records)?;

        // Batch-embed all chunks from all records in one micro-batched loop.
        if has_vector {
            const MICRO_BATCH: usize = 32;
            let flat_texts: Vec<&str> = plans
                .iter()
                .flat_map(|p| p.chunk_texts.iter().map(String::as_str))
                .collect();
            let flat_ids: Vec<RecordId> = plans
                .iter()
                .flat_map(|p| p.chunk_records.iter().map(|r| r.id.clone()))
                .collect();
            debug_assert_eq!(flat_texts.len(), flat_ids.len());

            if let (Ok(embedder), Ok(vi)) = (self.require_embedder(), self.require_vector_index()) {
                let mut cursor = 0usize;
                while cursor < flat_texts.len() {
                    let end = (cursor + MICRO_BATCH).min(flat_texts.len());
                    let slice = &flat_texts[cursor..end];
                    let vectors = embedder.embed_batch(slice).or_else(|_| {
                        slice
                            .iter()
                            .map(|t| embedder.embed(t))
                            .collect::<Result<Vec<_>>>()
                    });
                    if let Ok(vecs) = vectors {
                        let batch: Vec<(RecordId, &[f32])> = vecs
                            .iter()
                            .enumerate()
                            .filter_map(|(i, vec)| {
                                flat_ids
                                    .get(cursor + i)
                                    .map(|id| (id.clone(), vec.as_slice()))
                            })
                            .collect();
                        let _ = vi.add_batch(&batch);
                    }
                    cursor = end;
                }
            }
        }

        // FTS-index every chunk's `content` through one commit (see
        // `SearchIndex::index_field_batch`).
        if has_fts {
            if let Ok(fi) = self.require_fts_index() {
                let entries: Vec<(&RecordId, &str)> = plans
                    .iter()
                    .flat_map(|p| p.chunk_records.iter())
                    .map(|cr| {
                        (
                            &cr.id,
                            cr.data
                                .get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or(""),
                        )
                    })
                    .collect();
                let _ = fi.index_field_batch("content", &entries);
            }
        }

        // Persist per-source-record `_recall_chunk_ids` metadata.
        for plan in &plans {
            let source = &records[plan.source_idx];
            let ids: Vec<RecordId> = plan.chunk_records.iter().map(|r| r.id.clone()).collect();
            self.persist_recall_chunk_ids(source, &ids)?;
        }

        Ok(())
    }

    fn sync_recall_chunks_for_record(&self, record: &Record) -> Result<()> {
        if record.table.starts_with('_') || (!self.has_vector_index() && !self.has_fts_index()) {
            return Ok(());
        }

        self.delete_recall_chunks_for_source(record)?;

        let text = crate::util::searchable_text(&record.data);
        let chunks = crate::util::overlapping_chunks(
            &text,
            RECALL_CHUNK_MAX_BYTES,
            RECALL_CHUNK_OVERLAP_BYTES,
        );
        if chunks.len() <= 1 {
            self.persist_recall_chunk_ids(record, &[])?;
            return Ok(());
        }

        let chunk_count = chunks.len();
        let has_vector = self.has_vector_index() && self.has_embedder();
        let has_fts = self.has_fts_index();

        // Build all chunk records and batch-insert into storage (single transaction).
        let chunk_records: Vec<Record> = chunks
            .iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut r = Record::new(
                    RECALL_CHUNKS_TABLE,
                    serde_json::json!({
                        "source_record": record.id.to_string(),
                        "source_table": record.table,
                        "chunk_idx": chunk_idx,
                        "chunk_count": chunk_count,
                        "content": chunk,
                    }),
                );
                r.created_at = record.created_at;
                r.updated_at = record.created_at;
                r
            })
            .collect();
        self.storage.insert_batch(&chunk_records)?;

        // Batch embed all chunks in one ONNX call.
        if has_vector {
            let chunk_texts: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            if let (Ok(embedder), Ok(vi)) = (self.require_embedder(), self.require_vector_index()) {
                let vectors = embedder.embed_batch(&chunk_texts).or_else(|_| {
                    chunk_texts
                        .iter()
                        .map(|t| embedder.embed(t))
                        .collect::<Result<Vec<_>>>()
                });
                if let Ok(vecs) = vectors {
                    let batch: Vec<(RecordId, &[f32])> = vecs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i < chunk_records.len())
                        .map(|(i, vec)| (chunk_records[i].id.clone(), vec.as_slice()))
                        .collect();
                    let _ = vi.add_batch(&batch);
                }
            }
        }

        // FTS-index all chunks through one commit (see `index_field_batch`).
        if has_fts {
            if let Ok(fi) = self.require_fts_index() {
                let entries: Vec<(&RecordId, &str)> = chunk_records
                    .iter()
                    .map(|cr| {
                        (
                            &cr.id,
                            cr.data
                                .get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or(""),
                        )
                    })
                    .collect();
                let _ = fi.index_field_batch("content", &entries);
            }
        }

        let chunk_ids: Vec<RecordId> = chunk_records.iter().map(|r| r.id.clone()).collect();
        self.persist_recall_chunk_ids(record, &chunk_ids)?;
        Ok(())
    }

    fn delete_recall_chunks_for_source(&self, source_record: &Record) -> Result<()> {
        for chunk_id in recall_chunk_ids(source_record) {
            let _ = self.delete(&chunk_id);
        }
        Ok(())
    }

    fn persist_recall_chunk_ids(
        &self,
        source_record: &Record,
        chunk_ids: &[RecordId],
    ) -> Result<()> {
        let mut record = source_record.clone();
        let meta = record.metadata.get_or_insert_with(|| serde_json::json!({}));
        if let Some(obj) = meta.as_object_mut() {
            if chunk_ids.is_empty() {
                obj.remove("_recall_chunk_ids");
                if obj.is_empty() {
                    record.metadata = None;
                }
            } else {
                let ids: Vec<String> = chunk_ids.iter().map(ToString::to_string).collect();
                obj.insert("_recall_chunk_ids".to_string(), serde_json::json!(ids));
            }
        }
        self.storage.insert(&record)?;
        Ok(())
    }

    /// Query-Time Chunk reranking: rescores the top-K recall candidates by
    /// finding the chunk-level embedding that best matches the query and
    /// blending its cosine into the fused score.
    ///
    /// Fast path: if the candidate already has index-time chunks registered
    /// in `_recall_chunk_ids`, we look up their pre-computed vectors via
    /// `VectorIndex::get_vector` and never touch the embedder. This is the
    /// hot path for any dataset that goes through `insert_batch_raw` /
    /// `insert`, since `sync_recall_chunks_for_record` already populates
    /// both the `_recall_chunks` table and the vector index.
    ///
    /// Slow path: when no index-time chunks exist (short records, or
    /// databases that were built without vector + FTS at insert time),
    /// fall back to chunking the record's text at query time and embedding
    /// each chunk on the fly.
    ///
    /// Writes back `alpha * best_chunk_cosine + (1-alpha) * fused_score` into
    /// each scored result. Caller is responsible for the subsequent re-sort.
    ///
    /// Why not index-time chunking alone: on corpora where each session has
    /// one timestamp, promoting chunks to first-class result candidates makes
    /// every chunk of a recent session share that timestamp — recency-max-pool
    /// then pins the top-K to the latest sessions regardless of content. QTC
    /// sidesteps this by keeping session-level records as the ranked unit and
    /// only using chunk embeddings for scoring.
    fn apply_query_time_chunks(
        &self,
        results: &mut [crate::scoring::RecallResult],
        query_vec: &[f32],
        embedder: &dyn crate::plugin::TextEmbedder,
        qtc: &crate::scoring::QtcConfig,
    ) {
        if results.is_empty() || qtc.chunk_chars == 0 {
            return;
        }

        let q_norm = l2_norm(query_vec);
        if q_norm == 0.0 {
            return;
        }

        let alpha = qtc.alpha.clamp(0.0, 1.0);
        let rerank_n = results.len().min(qtc.top_k);

        for result in results.iter_mut().take(rerank_n) {
            // Fast path: read pre-computed chunk vectors stored at insert.
            if let Some(best) =
                self.qtc_best_cosine_from_stored_chunks(&result.record, query_vec, q_norm)
            {
                result.score = alpha * best + (1.0 - alpha) * result.score;
                continue;
            }

            // Slow path: chunk + embed at query time.
            let text = crate::util::searchable_text(&result.record.data);
            if text.trim().is_empty() {
                continue;
            }
            let chunks = chunk_by_chars(&text, qtc.chunk_chars, qtc.stride_chars);
            if chunks.is_empty() {
                continue;
            }
            let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let Ok(vectors) = embedder.embed_batch(&refs) else {
                continue;
            };
            let mut best: f32 = 0.0;
            for vec in &vectors {
                let sim = cosine(query_vec, q_norm, vec);
                if sim > best {
                    best = sim;
                }
            }
            result.score = alpha * best + (1.0 - alpha) * result.score;
        }
    }

    /// If the source record has index-time chunks registered, pull each
    /// chunk's stored vector from the vector index and return the max
    /// cosine against the query. Returns `None` when:
    /// - the record has no registered chunks (short content → no chunking)
    /// - the vector index is missing or doesn't implement `get_vector`
    /// - every chunk vector lookup fails
    ///
    /// The `None` signal tells the caller to fall back to query-time
    /// embedding.
    fn qtc_best_cosine_from_stored_chunks(
        &self,
        source_record: &Record,
        query_vec: &[f32],
        query_norm: f32,
    ) -> Option<f32> {
        let chunk_ids = recall_chunk_ids(source_record);
        if chunk_ids.is_empty() {
            return None;
        }
        let vi = self.vector_index.as_ref()?;

        let mut best: f32 = f32::NEG_INFINITY;
        let mut any_hit = false;
        for chunk_id in &chunk_ids {
            let Ok(Some(chunk_vec)) = vi.get_vector(chunk_id) else {
                continue;
            };
            any_hit = true;
            let sim = cosine(query_vec, query_norm, &chunk_vec);
            if sim > best {
                best = sim;
            }
        }

        if any_hit && best.is_finite() {
            Some(best)
        } else {
            None
        }
    }

    fn resolve_recall_candidate(
        &self,
        candidate_id: &RecordId,
    ) -> Result<Option<(RecordId, Record)>> {
        let Some(record) = self.storage.get(candidate_id)? else {
            return Ok(None);
        };
        if record.table != RECALL_CHUNKS_TABLE {
            return Ok(Some((record.id.clone(), record)));
        }

        let Some(source_id) = parse_source_record_id(&record) else {
            return Ok(None);
        };
        let Some(source_record) = self.storage.get(&source_id)? else {
            return Ok(None);
        };
        Ok(Some((source_id, source_record)))
    }
}

/// Pull the first file-path value out of a record's data, if any.
/// Walks well-known path fields in priority order. Used by `auto_link`
/// to scope provisional code-symbol ids per file.
pub(crate) fn pick_file_hint(data: &Value) -> Option<String> {
    let obj = data.as_object()?;
    for key in ["file", "path", "file_path"] {
        if let Some(s) = obj.get(key).and_then(|v| v.as_str()) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    for key in ["files", "files_changed", "files_modified"] {
        if let Some(arr) = obj.get(key).and_then(|v| v.as_array()) {
            if let Some(s) = arr.iter().find_map(|v| v.as_str()) {
                if !s.is_empty() {
                    return Some(s.to_string());
                }
            }
        }
    }
    None
}

/// Compute the canonical id for an entity.
///
/// Contract:
/// - Code symbols → `provisional:<hash>` keyed by (name, lang_hint, file_hint)
///   so the same display name in different languages or files does not
///   silently merge. SCIP ingest later rewrites these to grounded SCIP
///   symbol strings.
/// - Natural-language entities → normalized name.
pub(crate) fn entity_canonical_id(
    entity: &crate::entity::Entity,
    file_hint: Option<&str>,
) -> String {
    use crate::entity::EntityType;
    match &entity.entity_type {
        EntityType::CodeSymbol { lang_hint } => {
            crate::entity::provisional_canonical_id(&entity.name, lang_hint.as_deref(), file_hint)
        }
        _ => crate::entity::natural_canonical_id(&entity.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    // Build a RecallResult with the given summary text + score.
    fn rr(summary: &str, score: f32) -> crate::scoring::RecallResult {
        crate::scoring::RecallResult {
            record: Record::new("decisions", json!({ "summary": summary })),
            score,
            explanation: crate::scoring::ScoreExplanation {
                signals: vec![],
                summary: String::new(),
                query_class: None,
            },
        }
    }

    #[test]
    fn collapse_near_duplicates_keeps_top_scored_representative() {
        let cfg = crate::scoring::DedupConfig {
            enabled: true,
            ..Default::default()
        };
        // 3 near-identical (case/whitespace/punctuation only) + 1 distinct,
        // already sorted by score descending as the recall path delivers them.
        let mut results = vec![
            rr("Use RRF fusion for recall because it is rank-based and robust", 0.9),
            rr("use rrf fusion for recall because it is rank-based and robust", 0.7),
            rr("Use RRF  fusion for recall because it is RANK-based and robust.", 0.6),
            rr("Switched the storage backend to redb for ACID single-file durability", 0.5),
        ];
        collapse_near_duplicates(&mut results, &cfg);

        // The two near-dups collapse into the highest-scored representative.
        assert_eq!(results.len(), 2, "expected near-dups collapsed to 2");
        // The surviving representative is the highest-scored cluster member...
        assert_eq!(results[0].score, 0.9);
        // ...and the distinct record survives. The records are not mutated —
        // no recall-ephemeral annotation is written into record.data (so nothing
        // can leak to storage via a later activation bump).
        assert_eq!(results[1].score, 0.5);
        assert!(results[0].record.data.get("_near_duplicates").is_none());
        assert!(results[1].record.data.get("_near_duplicates").is_none());
    }

    #[test]
    fn collapse_near_duplicates_does_not_cross_tables() {
        // Two byte-identical texts in DIFFERENT tables must NOT collapse: recall
        // runs before any `--table` filter, so a cross-table collapse could let
        // the filter silently drop a record the user explicitly asked for.
        let cfg = crate::scoring::DedupConfig {
            enabled: true,
            ..Default::default()
        };
        let text = "Use RRF fusion for recall because it is rank-based and robust";
        let mut a = rr(text, 0.9); // table "decisions" (from rr helper)
        let mut b = rr(text, 0.7);
        b.record.table = "context".to_string();
        a.record.table = "decisions".to_string();
        let mut results = vec![a, b];
        collapse_near_duplicates(&mut results, &cfg);
        assert_eq!(results.len(), 2, "cross-table near-dups must not collapse");
    }

    #[test]
    fn collapse_near_duplicates_preserves_distinct_records() {
        let cfg = crate::scoring::DedupConfig {
            enabled: true,
            ..Default::default()
        };
        let mut results = vec![
            rr("Implemented SCIP code-graph ingest via prost protobuf parsing", 0.9),
            rr("Dependency doc memory pins versions from the lockfile closure", 0.8),
            rr("Entity extraction writes canonical_id and scoped aliases", 0.7),
        ];
        collapse_near_duplicates(&mut results, &cfg);
        assert_eq!(results.len(), 3, "distinct records must not collapse");
    }

    #[test]
    fn collapse_near_duplicates_skips_short_text() {
        // "same text" (9 chars) is below the default min_text_len (24), so even
        // identical short strings pass through untouched — SimHash is unreliable
        // on short text. With a low min_text_len they collapse.
        let strict = crate::scoring::DedupConfig {
            enabled: true,
            ..Default::default()
        };
        let mut a = vec![rr("same text", 0.9), rr("same text", 0.8)];
        collapse_near_duplicates(&mut a, &strict);
        assert_eq!(a.len(), 2, "short text must not be collapsed at default min_len");

        let lenient = crate::scoring::DedupConfig {
            enabled: true,
            min_text_len: 4,
            ..Default::default()
        };
        let mut b = vec![rr("same text", 0.9), rr("same text", 0.8)];
        collapse_near_duplicates(&mut b, &lenient);
        assert_eq!(b.len(), 1, "identical text collapses once long enough to fingerprint");
    }

    // ── Completeness k-widening (20.4) ──────────────────────────

    // A near-duplicate filler line long enough to clear WIDEN_MIN_KEPT_CHARS.
    const DUP_LINE: &str = "Use RRF fusion for recall because it is rank-based and robust to outliers";
    // High-entropy, distinct items (ids / codes / signatures) that share little
    // with the dup cluster — DEFLATE can't fold them into the kept redundancy,
    // so the pool ratio rises sharply when they sit past the cut.
    const DIVERSE_A: &str = "df9a1c3e-7b20-4f51-a8e2-0c6d9b1f4e77 E_TIMEOUT_4096 commit b3f7e2a quux::Zephyr<'x>";
    const DIVERSE_B: &str = "zx81 BBC micro 6502 assembly LDA #$FF STA $D020 raster interrupt vblank NMI handler";

    #[test]
    fn widen_target_fires_when_diverse_cluster_dropped() {
        // top_k=3, but the top 3 are near-identical restatements (highly
        // compressible) while a distinct, high-entropy cluster sits just past
        // the cut. The kept subset compresses far better than the full pool, so
        // k widens to recover the dropped cluster.
        let texts = vec![
            DUP_LINE.to_string(),
            DUP_LINE.to_string(),
            DUP_LINE.to_string(),
            DIVERSE_A.to_string(),
            DIVERSE_B.to_string(),
        ];
        let new_k = completeness_widen_target(&texts, 3, 0.15);
        assert!(
            new_k.is_some(),
            "kept set far more compressible than pool → expected widen"
        );
        let new_k = new_k.unwrap();
        assert!(new_k > 3, "widened k must exceed top_k (got {new_k})");
        // Bounded: ceil(3 * 1.5) = 5, capped at the 5 available candidates.
        assert!(new_k <= 5, "widen stays bounded by ceil(top_k*1.5) and pool size");
    }

    #[test]
    fn widen_glue_retains_diverse_item_via_record_text() {
        // End-to-end through the recall glue: build RecallResults the way the
        // recall path delivers them (sorted, post-dedup), run the widener, and
        // confirm the kept slice after truncation includes the diverse cluster
        // that a plain truncate(top_k) would have dropped.
        let mut results = vec![
            rr(DUP_LINE, 0.9),
            rr(DUP_LINE, 0.8),
            rr(DUP_LINE, 0.7),
            rr(DIVERSE_A, 0.6),
            rr(DIVERSE_B, 0.5),
        ];
        let cfg = crate::scoring::DedupConfig {
            completeness_widen: true,
            ..Default::default()
        };
        let mut k = 3usize;
        widen_k_for_completeness(&results, &mut k, &cfg);
        assert!(k > 3, "expected k to widen past the diverse-cluster cut");
        results.truncate(k);
        let kept_texts: Vec<String> = results
            .iter()
            .map(|r| crate::util::record_text(&r.record))
            .collect();
        assert!(
            kept_texts.iter().any(|t| t.contains("E_TIMEOUT_4096")),
            "the distinct high-entropy item must survive the widened cut"
        );
    }

    #[test]
    fn widen_target_noop_on_already_diverse_results() {
        // All four candidates are distinct prose — the kept top-k is no more
        // compressible than the pool, so no cluster was dropped: no-op.
        let texts = vec![
            "Implemented SCIP code-graph ingest via prost protobuf parsing and edge emission"
                .to_string(),
            "Dependency doc memory pins versions from the resolved lockfile closure on disk"
                .to_string(),
            "Entity extraction writes a canonical id alongside scoped per-language aliases"
                .to_string(),
            "Session checkpoint records replace the free-text session summary pattern entirely"
                .to_string(),
        ];
        assert_eq!(
            completeness_widen_target(&texts, 3, 0.15),
            None,
            "already-diverse top-k must not widen"
        );
    }

    #[test]
    fn widen_target_noop_when_no_extra_candidates() {
        // candidate_count <= top_k: nothing past the cut to recover.
        let texts = vec![DUP_LINE.to_string(), DUP_LINE.to_string()];
        assert_eq!(completeness_widen_target(&texts, 3, 0.15), None);
        assert_eq!(completeness_widen_target(&texts, 2, 0.15), None);
    }

    #[test]
    fn widen_target_noop_on_short_kept_text() {
        // Even a perfect compress-gap is ignored when the kept text is too
        // short for a DEFLATE ratio to mean anything (header overhead).
        let texts = vec![
            "a a a".to_string(),
            "a a a".to_string(),
            "a a a".to_string(),
            "completely distinct longer diverse content sitting just past the cut".to_string(),
        ];
        assert_eq!(
            completeness_widen_target(&texts, 3, 0.15),
            None,
            "short kept text must short-circuit before compressing"
        );
    }

    #[test]
    fn widen_glue_is_noop_when_config_disabled() {
        // Library default: completeness_widen=false → k is never touched, even
        // on a textbook diverse-cluster-dropped pool. Raw API result set
        // unchanged.
        let results = vec![
            rr(DUP_LINE, 0.9),
            rr(DUP_LINE, 0.8),
            rr(DUP_LINE, 0.7),
            rr(
                "Switched the storage backend to redb for ACID single-file durability \
                 with crash-safe write transactions and a custom edge-record layout",
                0.6,
            ),
        ];
        // Mirror recall: the widener only runs when the flag is on.
        let cfg = crate::scoring::DedupConfig::default();
        assert!(!cfg.completeness_widen, "default must be off");
        let mut k = 3usize;
        if cfg.completeness_widen {
            widen_k_for_completeness(&results, &mut k, &cfg);
        }
        assert_eq!(k, 3, "disabled config must never widen k");
    }

    #[test]
    fn insert_and_get() {
        let (db, _dir) = temp_db();
        let record = db.insert("sessions", json!({"summary": "test"})).unwrap();
        let fetched = db.get(&record.id).unwrap().unwrap();
        assert_eq!(fetched.data["summary"], "test");
    }

    // ---- — Extension registration ----

    struct StubExt {
        id: &'static str,
        prefixes: &'static [&'static str],
    }

    impl crate::extension::Extension for StubExt {
        fn id(&self) -> &str {
            self.id
        }
        fn table_prefixes(&self) -> &[&str] {
            self.prefixes
        }
    }

    #[test]
    fn extensions_empty_by_default() {
        let (db, _dir) = temp_db();
        assert!(db.extensions().is_empty());
    }

    #[test]
    fn register_extension_post_build() {
        let (db, _dir) = temp_db();
        assert!(db.extensions().is_empty());

        db.register_extension(Arc::new(StubExt {
            id: "rt",
            prefixes: &["_rt_"],
        }))
        .unwrap();
        assert_eq!(db.extensions().len(), 1);
        assert_eq!(db.extensions()[0].id(), "rt");

        // Duplicate id is rejected at runtime (error, not panic).
        let dup = db.register_extension(Arc::new(StubExt {
            id: "rt",
            prefixes: &["_other_"],
        }));
        assert!(dup.is_err());

        // Overlapping prefix is rejected.
        let overlap = db.register_extension(Arc::new(StubExt {
            id: "rt2",
            prefixes: &["_rt_sub"],
        }));
        assert!(overlap.is_err());
        assert_eq!(db.extensions().len(), 1);
    }

    #[test]
    fn drop_engine_companion_removes_orphan_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("m.axil");
        let companion = companion_path(&base, ".graph");
        std::fs::write(&companion, b"orphan").unwrap();

        // Role and bare suffix both resolve; the core file is never touched.
        let report = drop_engine_companion(&base, "graph").unwrap();
        assert_eq!(report.engine, "graph");
        assert_eq!(report.suffix, ".graph");
        assert!(report.existed);
        assert_eq!(report.bytes_freed, 6);
        assert!(!companion.exists());

        // Idempotent: re-dropping a missing companion is a no-op, not an error.
        let again = drop_engine_companion(&base, ".graph").unwrap();
        assert!(!again.existed);
        assert_eq!(again.bytes_freed, 0);

        // The "vec" suffix and "vector" role are aliases.
        assert_eq!(drop_engine_companion(&base, "vec").unwrap().engine, "vector");
        assert_eq!(drop_engine_companion(&base, "vector").unwrap().engine, "vector");

        // Unknown engines are rejected.
        assert!(drop_engine_companion(&base, "bogus").is_err());
    }

    #[test]
    fn with_extension_registers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path)
            .with_extension(StubExt {
                id: "stub",
                prefixes: &["_stub_"],
            })
            .build()
            .unwrap();
        assert_eq!(db.extensions().len(), 1);
        assert_eq!(db.extensions()[0].id(), "stub");
        assert_eq!(db.extensions()[0].table_prefixes(), &["_stub_"]);
    }

    #[test]
    fn with_extension_two_disjoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path)
            .with_extension(StubExt {
                id: "alpha",
                prefixes: &["_alpha_"],
            })
            .with_extension(StubExt {
                id: "beta",
                prefixes: &["_beta_"],
            })
            .build()
            .unwrap();
        assert_eq!(db.extensions().len(), 2);
    }

    #[test]
    #[should_panic(expected = "registered twice")]
    fn with_extension_rejects_duplicate_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let _ = Axil::open(&path)
            .with_extension(StubExt {
                id: "dup",
                prefixes: &["_a_"],
            })
            .with_extension(StubExt {
                id: "dup",
                prefixes: &["_b_"],
            });
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn with_extension_rejects_overlapping_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let _ = Axil::open(&path)
            .with_extension(StubExt {
                id: "first",
                prefixes: &["_dep_"],
            })
            .with_extension(StubExt {
                id: "second",
                // `_dep_docs` starts with `_dep_`, so this overlaps.
                prefixes: &["_dep_docs"],
            });
    }

    #[test]
    fn prefix_overlaps_helper() {
        // Equal — overlap.
        assert!(prefix_overlaps("_dep_", "_dep_"));
        // One prefixes the other (both directions) — overlap.
        assert!(prefix_overlaps("_dep_", "_dep_docs"));
        assert!(prefix_overlaps("_dep_docs", "_dep_"));
        assert!(prefix_overlaps("_dep_", "_d"));
        assert!(prefix_overlaps("_d", "_dep_"));
        // Disjoint — no overlap.
        assert!(!prefix_overlaps("_dep_", "_scip_"));
        assert!(!prefix_overlaps("_dep_", "_idx_"));
    }

    #[test]
    fn delete_record() {
        let (db, _dir) = temp_db();
        let record = db.insert("sessions", json!({})).unwrap();
        assert!(db.delete(&record.id).unwrap());
        assert!(db.get(&record.id).unwrap().is_none());
    }

    #[test]
    fn update_record() {
        let (db, _dir) = temp_db();
        let record = db.insert("sessions", json!({"v": 1})).unwrap();
        let updated = db.update(&record.id, json!({"v": 2})).unwrap();
        assert_eq!(updated.data["v"], 2);
    }

    #[test]
    fn list_records() {
        let (db, _dir) = temp_db();
        db.insert("items", json!({"a": 1})).unwrap();
        db.insert("items", json!({"a": 2})).unwrap();
        db.insert("other", json!({"b": 1})).unwrap();
        let items = db.list("items").unwrap();
        assert_eq!(items.len(), 2);
    }

    #[test]
    #[allow(deprecated)]
    fn builder_methods_chainable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).with_graph().build().unwrap();
        let _ = db.insert("test", json!({})).unwrap();
    }

    #[test]
    fn no_vector_index_returns_error() {
        let (db, _dir) = temp_db();
        let record = db.insert("test", json!({"text": "hello"})).unwrap();
        assert!(db.embed_field(&record.id, "text").is_err());
        assert!(db.similar_to_vector(&[1.0, 0.0], 5).is_err());
    }

    // ── QTC helpers ────────────────────────────────────────────────────

    #[test]
    fn chunk_by_chars_short_text_single_chunk() {
        let chunks = chunk_by_chars("short", 100, 50);
        assert_eq!(chunks, vec!["short".to_string()]);
    }

    #[test]
    fn chunk_by_chars_long_text_overlapping() {
        let text = "a".repeat(250);
        let chunks = chunk_by_chars(&text, 100, 75);
        // 250-char text with max=100, stride=75 → pos steps 0, 75, 150.
        // Windows: [0..100], [75..175], [150..250]. Last one hits EOF so
        // the loop exits before stepping to 225.
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 100);
    }

    #[test]
    fn chunk_by_chars_respects_utf8_boundaries() {
        // Emoji is 4 bytes; splitting mid-codepoint would panic in from_utf8.
        let text = "ab🦀cd🦀ef🦀gh".repeat(10);
        let chunks = chunk_by_chars(&text, 15, 10);
        // If boundaries are respected, every chunk is valid UTF-8 (implicit
        // in the &str return; panic would fire in the allocator otherwise).
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() > 0);
        }
    }

    #[test]
    fn chunk_by_chars_empty_or_zero_max() {
        assert!(chunk_by_chars("", 100, 50).is_empty());
        assert!(chunk_by_chars("hello", 0, 10).is_empty());
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let na = l2_norm(&a);
        assert!(cosine(&a, na, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![0.6, 0.8, 0.0];
        let na = l2_norm(&a);
        assert!((cosine(&a, na, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_zero_norm_safely() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        // Zero-norm query → 0.0 (not NaN).
        assert_eq!(cosine(&a, 0.0, &b), 0.0);
        // Zero-norm record → 0.0 (not NaN).
        assert_eq!(cosine(&b, l2_norm(&b), &a), 0.0);
    }

    // ── Identifier-aware RRF tilt toward FTS (20.5) ─────────────

    /// A minimal, query-aware in-test FTS index, so the fusion path can be
    /// exercised deterministically without pulling in the tantivy-backed engine.
    /// Records are registered with their text; `search_text` scores each by the
    /// count of query terms it contains and returns the term-matched records in
    /// descending score order — a coarse but realistic BM25 stand-in. This lets
    /// a single fixture serve both the identifier query (exact token → only the
    /// exact record matches) and a natural-language query (its words match the
    /// other record), the way a real FTS would.
    struct StubFts {
        docs: std::sync::Mutex<Vec<(RecordId, String)>>,
    }

    impl StubFts {
        fn new(docs: Vec<(RecordId, String)>) -> Self {
            Self {
                docs: std::sync::Mutex::new(docs),
            }
        }
    }

    impl crate::plugin::Engine for StubFts {
        fn name(&self) -> &str {
            "stub-fts"
        }
        fn capabilities(&self) -> Vec<crate::plugin::Capability> {
            vec![crate::plugin::Capability::FullTextSearch]
        }
        fn on_record_insert(&self, _record: &Record) -> Result<()> {
            Ok(())
        }
        fn on_record_delete(&self, _id: &RecordId) -> Result<()> {
            Ok(())
        }
    }

    impl crate::plugin::SearchIndex for StubFts {
        fn index_text(&self, _id: &RecordId, _field: &str, _text: &str) -> Result<()> {
            Ok(())
        }
        fn search_text(&self, query: &str, limit: usize) -> Result<Vec<(RecordId, f32)>> {
            let terms: Vec<String> = query.to_lowercase().split_whitespace().map(String::from).collect();
            let docs = self.docs.lock().unwrap();
            let mut scored: Vec<(RecordId, f32)> = docs
                .iter()
                .filter_map(|(id, text)| {
                    let lower = text.to_lowercase();
                    let hits = terms.iter().filter(|t| lower.contains(t.as_str())).count();
                    if hits == 0 {
                        None
                    } else {
                        Some((id.clone(), hits as f32))
                    }
                })
                .collect();
            // Stable sort by score desc; ties keep registration order, so the
            // first-registered doc holds FTS rank 0 on equal term counts.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);
            Ok(scored)
        }
    }

    /// Build a DB with two records and a query-aware stub FTS. `id_exact` carries
    /// a UUID token; `id_recent` is created *more recently* (recency-favored) and
    /// carries distinctive natural-language words. The UUID also appears verbatim
    /// in `id_recent`'s text so that, for the identifier query, BOTH records are
    /// FTS candidates and `id_recent` would otherwise win on recency under pure
    /// RRF — the tilt is what promotes the exact rank-0 FTS hit (`id_exact`).
    fn fts_tilt_fixture() -> (Axil, RecordId, RecordId) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tilt.axil");
        const UUID: &str = "550e8400-e29b-41d4-a716-446655440000";

        let id_exact;
        let id_recent;
        let exact_text;
        let recent_text;
        {
            let db = Axil::open(&path).build().unwrap();
            // Exact record: leads with the UUID, inserted FIRST (older).
            exact_text = format!("Auth bug {UUID}");
            let exact = db
                .insert("decisions", json!({ "summary": exact_text }))
                .unwrap();
            id_exact = exact.id.clone();
            std::thread::sleep(std::time::Duration::from_millis(5));
            // Recent record: ALSO mentions the UUID (so the bare-UUID query has
            // the SAME term count for both → equal keyword_match), but is newer
            // so recency favors it. Under pure RRF (rrf weight 0, no vector
            // index) recency breaks the tie and this newer record wins; the
            // identifier tilt — which keys off FTS *rank*, where the exact record
            // sits at rank 0 — must override that. It also carries distinctive NL
            // words for the natural-language query.
            recent_text = format!("Newer deployment cadence note, see {UUID}");
            let recent = db
                .insert("decisions", json!({ "summary": recent_text }))
                .unwrap();
            id_recent = recent.id.clone();
        }

        let fts = std::sync::Arc::new(StubFts::new(vec![
            (id_exact.clone(), exact_text),
            (id_recent.clone(), recent_text),
        ]));
        let db = Axil::open(&path).with_fts_index(fts).build().unwrap();
        // Keep the dir alive for the db's lifetime by leaking it — the test
        // process tears down immediately after, so the bounded leak is fine.
        std::mem::forget(dir);
        (db, id_exact, id_recent)
    }

    #[test]
    fn identifier_query_ranks_exact_fts_hit_first() {
        let (db, id_exact, id_recent) = fts_tilt_fixture();

        // Identifier query (the UUID itself) → tilt fires, exact FTS hit wins.
        let results = db
            .recall(
                "550e8400-e29b-41d4-a716-446655440000",
                5,
                Some(crate::scoring::RecallConfig::default()),
            )
            .unwrap();
        assert!(!results.is_empty(), "identifier recall returned nothing");
        assert_eq!(
            results[0].record.id, id_exact,
            "identifier tilt did not rank the exact FTS hit first; \
             got {:?}",
            results[0].record.id
        );
        // The tilt is recorded on the winning result's explanation.
        assert!(
            results[0]
                .explanation
                .signals
                .iter()
                .any(|(name, v)| name == "fts_identifier_tilt" && *v > 0.0),
            "fts_identifier_tilt signal missing from explanation"
        );
        assert_eq!(
            results[0].explanation.query_class.as_deref(),
            Some("identifier:uuid"),
            "query_class not surfaced in explanation"
        );
        let _ = id_recent;
    }

    #[test]
    fn natural_language_query_is_unaffected_by_tilt() {
        let (db, id_exact, id_recent) = fts_tilt_fixture();

        // Same corpus, but a natural-language query → NO tilt. With rrf weight 0
        // and no vector index, recency dominates, so the newer record wins and
        // the tilt signal is absent.
        let results = db
            .recall(
                "deployment cadence note",
                5,
                Some(crate::scoring::RecallConfig::default()),
            )
            .unwrap();
        assert!(!results.is_empty(), "NL recall returned nothing");
        assert_eq!(
            results[0].record.id, id_recent,
            "natural-language ranking should be recency-led (unchanged), \
             but got {:?}",
            results[0].record.id
        );
        // No identifier tilt anywhere in the NL result set.
        for r in &results {
            assert!(
                !r.explanation
                    .signals
                    .iter()
                    .any(|(name, _)| name == "fts_identifier_tilt"),
                "fts_identifier_tilt must not appear for a natural-language query"
            );
            assert_eq!(
                r.explanation.query_class.as_deref(),
                Some("natural-language"),
                "NL query_class not surfaced"
            );
        }
        let _ = id_exact;
    }

    #[cfg(feature = "event-log")]
    mod event_log {
        use super::*;

        #[test]
        fn off_by_default_captures_nothing() {
            let (db, _dir) = temp_db();
            // Even an allowlisted write is a no-op while the log is disabled.
            db.insert("_checkpoint_records", json!({"goal": "x"}))
                .unwrap();
            assert_eq!(db.event_log_len().unwrap(), 0);
            assert!(db.recall_delta(None, None, 50).unwrap().is_empty());
        }

        #[test]
        fn captures_allowlisted_underscore_table_events() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);

            // Checkpoint write — an allowlisted `_`-prefixed table the plain
            // audit log would skip.
            db.insert("_checkpoint_records", json!({"goal": "ship T14"}))
                .unwrap();
            // Belief revision — also `_`-prefixed.
            db.insert("_beliefs", json!({"statement": "x", "doubted": true}))
                .unwrap();
            // An ordinary write is NOT captured.
            db.insert("notes", json!({"text": "noise"})).unwrap();

            let events = db.recall_delta(None, None, 50).unwrap();
            let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
            assert!(kinds.contains(&crate::event_log::kind::CHECKPOINT_WRITTEN));
            assert!(kinds.contains(&crate::event_log::kind::BELIEF_REVISED));
            assert_eq!(events.len(), 2, "ordinary `notes` write must not be captured");
        }

        #[test]
        fn captures_decision_superseded_and_error_fixed() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);

            let dec = db
                .insert("decisions", json!({"summary": "old"}))
                .unwrap();
            // Mark it superseded via update → decision-superseded.
            db.update(&dec.id, json!({"summary": "old", "_superseded": true}))
                .unwrap();
            // An error with a fix → error-fixed.
            db.insert("errors", json!({"error": "boom", "fix": "patch"}))
                .unwrap();

            let events = db.recall_delta(None, None, 50).unwrap();
            let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
            assert!(kinds.contains(&crate::event_log::kind::DECISION_SUPERSEDED));
            assert!(kinds.contains(&crate::event_log::kind::ERROR_FIXED));
        }

        #[test]
        fn recall_delta_returns_only_post_cursor_events() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);

            for i in 0..5 {
                db.insert("errors", json!({"error": format!("e{i}"), "fix": "f"}))
                    .unwrap();
            }
            let all = db.recall_delta(None, None, 50).unwrap();
            assert_eq!(all.len(), 5);

            // Resume strictly after the 2nd event → events 3,4,5.
            let cursor = all[1].cursor.clone();
            let rest = db.recall_delta(Some(&cursor), None, 50).unwrap();
            assert_eq!(rest.len(), 3);
            assert_eq!(rest[0].cursor, all[2].cursor);
            // Every returned cursor is strictly greater than the resume cursor.
            for e in &rest {
                assert!(e.cursor > cursor);
            }
        }

        #[test]
        fn recall_delta_respects_exclude_agent() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);

            db.insert("errors", json!({"error": "a", "fix": "f", "_agent_id": "agent-a"}))
                .unwrap();
            db.insert("errors", json!({"error": "b", "fix": "f", "_agent_id": "agent-b"}))
                .unwrap();
            db.insert("errors", json!({"error": "c", "fix": "f", "_agent_id": "agent-a"}))
                .unwrap();

            // Excluding agent-a leaves only agent-b's single event.
            let filtered = db.recall_delta(None, Some("agent-a"), 50).unwrap();
            assert_eq!(filtered.len(), 1);
            assert_eq!(filtered[0].agent_id.as_deref(), Some("agent-b"));

            // No exclusion → all three.
            assert_eq!(db.recall_delta(None, None, 50).unwrap().len(), 3);
        }

        #[test]
        fn cursor_is_monotonic_across_same_millisecond_writes() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);

            // A tight insert loop forces many same-millisecond commits; the
            // cursors must still be strictly increasing in commit order.
            for i in 0..200 {
                db.insert("errors", json!({"error": format!("e{i}"), "fix": "f"}))
                    .unwrap();
            }
            let events = db.recall_delta(None, None, 1000).unwrap();
            assert_eq!(events.len(), 200);
            for w in events.windows(2) {
                assert!(
                    w[0].cursor < w[1].cursor,
                    "cursor not monotonic: {} should be < {}",
                    w[0].cursor,
                    w[1].cursor
                );
            }
        }

        #[test]
        fn trim_keeps_newest_within_bound() {
            let (db, _dir) = temp_db();
            db.set_event_log_enabled(true);
            for i in 0..10 {
                db.insert("errors", json!({"error": format!("e{i}"), "fix": "f"}))
                    .unwrap();
            }
            assert_eq!(db.event_log_len().unwrap(), 10);
            db.trim_event_log(4).unwrap();
            assert_eq!(db.event_log_len().unwrap(), 4);
            // Trimming evicts the oldest, so the surviving events are the newest 4.
            let remaining = db.recall_delta(None, None, 50).unwrap();
            assert_eq!(remaining.len(), 4);
        }
    }

    /// Builder-level encryption wiring (`encryption` feature). Storage-level
    /// sealing is covered in `storage.rs`; these prove `AxilBuilder::with_cipher`
    /// actually threads the cipher into the core store, so a full `Axil` opened
    /// through the builder round-trips — the integration the CLI/MCP rely on.
    #[cfg(feature = "encryption")]
    mod encryption {
        use super::*;
        use crate::crypto::Cipher;

        fn key_a() -> Cipher {
            Cipher::from_key_bytes(&[7u8; 32]).unwrap()
        }
        fn key_b() -> Cipher {
            Cipher::from_key_bytes(&[9u8; 32]).unwrap()
        }

        /// insert through a cipher-attached builder → reopen with the same key →
        /// the record reads back as plaintext.
        #[test]
        fn builder_round_trips_through_cipher() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");

            // Scoped so the writer handle drops (releasing the single-writer
            // lock) before we reopen the same file below.
            let id = {
                let db = Axil::open(&path).with_cipher(key_a()).build().unwrap();
                db.insert("secrets", json!({"summary": "classified"}))
                    .unwrap()
                    .id
            };

            let db = Axil::open(&path).with_cipher(key_a()).build().unwrap();
            let fetched = db.get(&id).unwrap().unwrap();
            assert_eq!(fetched.data["summary"], "classified");
            assert_eq!(fetched.table, "secrets");
        }

        /// A builder with no cipher cannot read a body that a cipher-attached
        /// builder wrote — proves the builder genuinely encrypted on disk
        /// (the wiring is not a silent no-op).
        #[test]
        fn cleartext_builder_cannot_read_encrypted_db() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let id = {
                let db = Axil::open(&path).with_cipher(key_a()).build().unwrap();
                db.insert("secrets", json!({"summary": "needle-xyz"}))
                    .unwrap()
                    .id
            };

            // Reopen with NO cipher: the sealed body fails to deserialize as a
            // cleartext record, so `get` surfaces an error rather than plaintext.
            let db = Axil::open(&path).build().unwrap();
            assert!(db.get(&id).is_err());
        }

        /// Wrong key fails loud rather than returning corrupt or partial data.
        #[test]
        fn wrong_key_builder_fails_to_read() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let id = {
                let db = Axil::open(&path).with_cipher(key_a()).build().unwrap();
                db.insert("secrets", json!({"summary": "classified"}))
                    .unwrap()
                    .id
            };

            let db = Axil::open(&path).with_cipher(key_b()).build().unwrap();
            assert!(db.get(&id).is_err());
        }
    }
}
