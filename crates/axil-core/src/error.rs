use thiserror::Error;

/// Core error type for all Axil operations.
///
/// `#[non_exhaustive]`: new variants may be added in minor releases without a
/// breaking change, so downstream `match` on an `AxilError` must include a
/// wildcard arm. (The `Busy` variant added in 2.0 is exactly the kind of
/// additive growth this guards against repeating.)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AxilError {
    /// Storage-layer error (redb).
    #[error("storage error: {0}")]
    Storage(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// Record not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Invalid query.
    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// Serialization / deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// Plugin error — preserves the original error source chain.
    #[error("plugin error: {0}")]
    Plugin(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// The database file is already opened for writing by another process.
    ///
    /// Axil is single-writer: only one process may hold a writable handle to a
    /// given `.axil` (or companion) file at a time. A second writer that tries
    /// to open the same file gets this variant rather than a generic storage
    /// error, so callers can distinguish a transient contention failure (retry
    /// or fall back to a read-only open) from a corrupt or missing file.
    #[error("database busy: already opened for writing by another process")]
    Busy,

    /// An import stopped partway through, after the export header was accepted
    /// and one or more records or edges had already been committed.
    ///
    /// Portable import is fail-fast with *partial state*: each imported record
    /// is written in its own storage transaction, so a mid-stream failure (a
    /// malformed line, an insert error) leaves everything before it committed.
    /// This variant carries the partial [`ImportReport`] so the caller can see
    /// exactly what was written instead of the accounting being discarded with
    /// the error. Failures raised *before* the header is accepted mutate
    /// nothing and surface as their own plain error rather than this variant.
    ///
    /// [`ImportReport`]: crate::portable::ImportReport
    #[error("import interrupted after partial write: {source}")]
    ImportInterrupted {
        /// What was committed before the failure stopped the import.
        report: Box<crate::portable::ImportReport>,
        /// The underlying failure that stopped the import.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl AxilError {
    /// Create a plugin error from a message string.
    pub fn plugin(msg: impl Into<String>) -> Self {
        Self::Plugin(Box::new(PluginMessage(msg.into())))
    }

    /// True if this error means the database is held open for writing by
    /// another process (the single-writer lock is contended).
    ///
    /// Hot read commands use this to decide between a bounded retry on the
    /// writer and a read-only fallback open.
    pub fn is_busy(&self) -> bool {
        matches!(self, AxilError::Busy)
    }
}

/// Simple string-based error for plugin messages without an underlying cause.
#[derive(Debug)]
struct PluginMessage(String);

impl std::fmt::Display for PluginMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PluginMessage {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AxilError>;

// ── From impls ──────────────────────────────────────────────────────

impl From<redb::Error> for AxilError {
    fn from(e: redb::Error) -> Self {
        AxilError::Storage(Box::new(e))
    }
}

impl From<redb::DatabaseError> for AxilError {
    fn from(e: redb::DatabaseError) -> Self {
        // The single-writer lock is contended: another process holds a writable
        // handle to this file. Surface it as the typed `Busy` variant so callers
        // can retry or fall back to a read-only open. Everything else is opaque
        // storage failure.
        match e {
            redb::DatabaseError::DatabaseAlreadyOpen => AxilError::Busy,
            other => AxilError::Storage(Box::new(other)),
        }
    }
}

impl From<redb::TableError> for AxilError {
    fn from(e: redb::TableError) -> Self {
        AxilError::Storage(Box::new(e))
    }
}

impl From<redb::TransactionError> for AxilError {
    fn from(e: redb::TransactionError) -> Self {
        AxilError::Storage(Box::new(e))
    }
}

impl From<redb::StorageError> for AxilError {
    fn from(e: redb::StorageError) -> Self {
        AxilError::Storage(Box::new(e))
    }
}

impl From<redb::CommitError> for AxilError {
    fn from(e: redb::CommitError) -> Self {
        AxilError::Storage(Box::new(e))
    }
}

impl From<serde_json::Error> for AxilError {
    fn from(e: serde_json::Error) -> Self {
        AxilError::Serialization(Box::new(e))
    }
}
