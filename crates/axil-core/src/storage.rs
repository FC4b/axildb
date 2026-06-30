use std::path::Path;

use chrono::Utc;
use redb::{
    Database, ReadOnlyDatabase, ReadTransaction, ReadableDatabase, ReadableTable,
    ReadableTableMetadata, TableDefinition, WriteTransaction,
};

use crate::error::{AxilError, Result};
use crate::record::{Record, RecordId};

/// redb table: record_id (string) → serialized Record (bytes).
const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records");

/// redb table: table_name (string) → JSON array of record IDs.
const TABLE_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("table_index");

/// redb table: key (string) → serialized JSON (bytes) for slow query log.
const SLOW_QUERIES: TableDefinition<&str, &[u8]> = TableDefinition::new("_slow_queries");

/// redb table: key (string) → serialized JSON (bytes) for audit log.
const AUDIT_LOG: TableDefinition<&str, &[u8]> = TableDefinition::new("_audit_log");

/// redb table: key (timestamp) → serialized JSON (bytes) for metrics history snapshots.
const METRICS_HISTORY: TableDefinition<&str, &[u8]> = TableDefinition::new("_metrics_history");

/// redb table: key (ULID `change_id`) → serialized [`ChangeEntry`] (bytes).
///
/// Off-by-default change-data-capture tape. Written inside the same write
/// transaction as the record it describes, so a crash can never desync the two.
/// The ULID key is itself the replay cursor (ULIDs are monotonic).
#[cfg(feature = "cdc")]
const CHANGELOG: TableDefinition<&str, &[u8]> = TableDefinition::new("_changelog");

/// redb table: key (monotonic ULID cursor) → serialized [`SemanticEvent`] bytes.
///
/// Off-by-default semantic event log. Unlike the per-record audit log, this
/// captures only a curated allowlist of agent-meaningful events and is keyed by
/// a monotonic ULID cursor (same-millisecond writes still sort in commit order),
/// so a second agent can pull "what changed since I last looked" deterministically.
#[cfg(feature = "event-log")]
const EVENT_LOG: TableDefinition<&str, &[u8]> = TableDefinition::new("_event_log");

/// Maximum `_changelog` entries retained before the oldest are pruned in-txn.
///
/// Bounds the tape on the existing write path (no separate worker) so an
/// always-on CDC build can't grow the core file unboundedly. A consumer that
/// falls further behind than this loses the ability to replay from an old
/// cursor and must do a full resync.
#[cfg(feature = "cdc")]
const MAX_CHANGELOG_ENTRIES: usize = 100_000;

/// A single change-data-capture event on the durable `_changelog` tape.
///
/// Default capture is id-only (`before`/`after` are `None`); full-body capture
/// is opt-in via [`Storage::set_cdc_capture_values`] because serializing both
/// sides roughly doubles per-write cost.
#[cfg(feature = "cdc")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChangeEntry {
    /// ULID change identifier — also the monotonic replay cursor.
    pub change_id: String,
    /// Mutation kind: `"insert"`, `"update"`, or `"delete"`.
    pub op: String,
    /// Table the record belongs to.
    pub table: String,
    /// Affected record ID.
    pub record_id: String,
    /// Pre-image record body — `Some` only when value capture is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<serde_json::Value>,
    /// Post-image record body — `Some` only when value capture is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<serde_json::Value>,
}

/// Reserved per-client sync bookkeeping for the future Atlas control plane.
///
/// This is a **shape reservation only** — no sync, replication, or push/pull
/// machinery is built here. It exists so Atlas can adopt a stable, versioned
/// `_sync_meta` record layout without a later on-disk migration. One row per
/// `client_id`, keyed in a future `_sync_meta` table; nothing in-tree writes it
/// yet.
#[cfg(feature = "cdc")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SyncMeta {
    /// Schema version of this record's shape (start at 1).
    pub version: u32,
    /// Opaque identifier of the syncing client/replica.
    pub client_id: String,
    /// The last `_changelog` cursor (ULID `change_id`) this client has applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synced_revision: Option<String>,
    /// RFC3339 timestamp of the last successful pull, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pull: Option<String>,
    /// RFC3339 timestamp of the last successful push, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_push: Option<String>,
}

/// Backing redb handle — either a writable database (the normal single-writer
/// process) or a read-only view of a committed-but-unheld file.
///
/// Axil is single-writer: redb takes an **exclusive** file lock on
/// `Database::create`, so a second writer fails with
/// [`AxilError::Busy`](crate::AxilError::Busy). A [`ReadOnlyDatabase`] requests
/// a *shared* lock, which cannot coexist with that exclusive lock — so a
/// read-only open also fails with `Busy` while a writer is live, and succeeds
/// only once the writer has closed. Hot read commands use it as a fallback for
/// the gap between short-lived writer sessions, after a bounded busy-retry.
enum StorageDb {
    Writable(Database),
    ReadOnly(ReadOnlyDatabase),
}

/// Low-level storage backend wrapping a `redb` database handle.
pub struct Storage {
    db: StorageDb,
    /// When `true` (and the `cdc` feature is on), `_changelog` entries carry the
    /// full pre/post record body. Off by default — id-only capture.
    #[cfg(feature = "cdc")]
    cdc_capture_values: std::sync::atomic::AtomicBool,
    /// Monotonic ULID source for `_changelog` cursors. `ulid::Generator`
    /// guarantees each id is strictly greater than the last even within one
    /// millisecond — the property a resumable CDC cursor needs. Plain
    /// `Ulid::new()` would let two same-millisecond changes sort out of order,
    /// so a consumer could skip one past an exclusive cursor and merge-replay
    /// could reorder two same-ms updates to the same record.
    #[cfg(feature = "cdc")]
    changelog_cursor: std::sync::Mutex<ulid::Generator>,
    /// Optional encryption-at-rest cipher for core record bodies. When `Some`
    /// (and the `encryption` feature is on), each record body is sealed with
    /// XChaCha20-Poly1305 before it is written to the `records` table and
    /// unsealed on read. `None` means cleartext bodies — the default. See
    /// [`crate::crypto`] for the wire format and honest scope.
    #[cfg(feature = "encryption")]
    cipher: Option<crate::crypto::Cipher>,
}

impl Storage {
    /// Open (or create) a writable database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path.as_ref())?;

        // Ensure tables exist.
        let txn = db.begin_write()?;
        {
            let _ = txn.open_table(RECORDS)?;
            let _ = txn.open_table(TABLE_INDEX)?;
            let _ = txn.open_table(SLOW_QUERIES)?;
            let _ = txn.open_table(AUDIT_LOG)?;
            let _ = txn.open_table(METRICS_HISTORY)?;
            #[cfg(feature = "cdc")]
            let _ = txn.open_table(CHANGELOG)?;
            #[cfg(feature = "event-log")]
            let _ = txn.open_table(EVENT_LOG)?;
        }
        txn.commit()?;

        Ok(Self {
            db: StorageDb::Writable(db),
            #[cfg(feature = "cdc")]
            cdc_capture_values: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "cdc")]
            changelog_cursor: std::sync::Mutex::new(ulid::Generator::new()),
            #[cfg(feature = "encryption")]
            cipher: None,
        })
    }

    /// Open an existing database read-only, without taking the exclusive
    /// single-writer lock.
    ///
    /// This never creates or modifies the file. It serves committed records to
    /// hot read commands in the gap between writer sessions; it requests a
    /// *shared* lock, so it fails with [`AxilError::Busy`](crate::AxilError::Busy)
    /// while a writer holds the exclusive lock (no read-through of a live
    /// writer). Any mutation method on the returned `Storage` also fails with
    /// `Busy` — a read-only handle cannot open a write transaction. The file
    /// must already exist (it is created only on the writable
    /// [`Storage::open`] path).
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let db = ReadOnlyDatabase::open(path.as_ref())?;
        Ok(Self {
            db: StorageDb::ReadOnly(db),
            #[cfg(feature = "cdc")]
            cdc_capture_values: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "cdc")]
            changelog_cursor: std::sync::Mutex::new(ulid::Generator::new()),
            #[cfg(feature = "encryption")]
            cipher: None,
        })
    }

    /// Attach an encryption-at-rest cipher to this storage handle.
    ///
    /// When set, every core record body is sealed with XChaCha20-Poly1305
    /// before it is written and unsealed on read. This is a builder-style
    /// consuming setter so it can be chained after [`Storage::open`]. Available
    /// only under the off-by-default `encryption` feature — see [`crate::crypto`]
    /// for the wire format, key sources, and honest scope (record bodies only).
    #[cfg(feature = "encryption")]
    pub fn with_cipher(mut self, cipher: crate::crypto::Cipher) -> Self {
        self.cipher = Some(cipher);
        self
    }

    /// Encode a record body for storage in the `records` table.
    ///
    /// Without the `encryption` feature (or with no cipher attached) this is the
    /// plain serde body, byte-identical to a default build. With a cipher
    /// attached the body is sealed and AAD-bound to the record ID.
    #[cfg(feature = "encryption")]
    fn encode_body(&self, record: &Record) -> Result<Vec<u8>> {
        let plaintext = record.to_bytes()?;
        match &self.cipher {
            Some(cipher) => Ok(cipher.encrypt(&plaintext, record.id.as_str())?),
            None => Ok(plaintext),
        }
    }

    /// Decode a stored record body for the given record ID (the redb key).
    ///
    /// With a cipher attached, an AAD/key mismatch fails cleanly rather than
    /// returning corrupt data.
    #[cfg(feature = "encryption")]
    fn decode_body(&self, id: &str, bytes: &[u8]) -> Result<Record> {
        match &self.cipher {
            Some(cipher) => {
                let plaintext = cipher.decrypt(bytes, id)?;
                Record::from_bytes(&plaintext)
            }
            None => Record::from_bytes(bytes),
        }
    }

    /// Passthrough body encode for the default (no-`encryption`) build —
    /// plain serde, byte-identical to calling `record.to_bytes()` directly.
    #[cfg(not(feature = "encryption"))]
    #[inline]
    fn encode_body(&self, record: &Record) -> Result<Vec<u8>> {
        record.to_bytes()
    }

    /// Passthrough body decode for the default (no-`encryption`) build.
    #[cfg(not(feature = "encryption"))]
    #[inline]
    fn decode_body(&self, _id: &str, bytes: &[u8]) -> Result<Record> {
        Record::from_bytes(bytes)
    }

    /// Serialize a [`ChangeEntry`] for the `_changelog` table, sealing it with
    /// the cipher (AAD-bound to its `change_id`) when encryption is on. CDC
    /// value-capture stores full before/after record bodies, so without this the
    /// change tape would hold cleartext copies of bodies the `records` table
    /// seals — defeating encryption-at-rest. With a cipher attached the whole
    /// entry (metadata + bodies) is sealed; without one it is plain serde,
    /// byte-identical to the no-`encryption` build.
    #[cfg(all(feature = "cdc", feature = "encryption"))]
    fn encode_changelog(&self, change_id: &str, entry: &ChangeEntry) -> Result<Vec<u8>> {
        let plaintext = serde_json::to_vec(entry)?;
        match &self.cipher {
            Some(cipher) => Ok(cipher.encrypt(&plaintext, change_id)?),
            None => Ok(plaintext),
        }
    }

    /// Decode a `_changelog` entry for `change_id` (its redb key, the AAD).
    #[cfg(all(feature = "cdc", feature = "encryption"))]
    fn decode_changelog(&self, change_id: &str, bytes: &[u8]) -> Result<ChangeEntry> {
        match &self.cipher {
            Some(cipher) => {
                let plaintext = cipher.decrypt(bytes, change_id)?;
                Ok(serde_json::from_slice(&plaintext)?)
            }
            None => Ok(serde_json::from_slice(bytes)?),
        }
    }

    /// Plain changelog encode/decode for the default (no-`encryption`) build —
    /// byte-identical to calling serde directly.
    #[cfg(all(feature = "cdc", not(feature = "encryption")))]
    #[inline]
    fn encode_changelog(&self, _change_id: &str, entry: &ChangeEntry) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(entry)?)
    }

    #[cfg(all(feature = "cdc", not(feature = "encryption")))]
    #[inline]
    fn decode_changelog(&self, _change_id: &str, bytes: &[u8]) -> Result<ChangeEntry> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// True if this handle is read-only (cannot accept write transactions).
    pub fn is_read_only(&self) -> bool {
        matches!(self.db, StorageDb::ReadOnly(_))
    }

    /// Begin a read transaction. Works on both writable and read-only handles.
    fn begin_read(&self) -> Result<ReadTransaction> {
        match &self.db {
            StorageDb::Writable(db) => Ok(db.begin_read()?),
            StorageDb::ReadOnly(db) => Ok(db.begin_read()?),
        }
    }

    /// Begin a write transaction. Fails with [`AxilError::Busy`] on a read-only
    /// handle — those are opened precisely because a writer is already active.
    fn begin_write(&self) -> Result<WriteTransaction> {
        match &self.db {
            StorageDb::Writable(db) => Ok(db.begin_write()?),
            StorageDb::ReadOnly(_) => Err(AxilError::Busy),
        }
    }

    /// Insert a record. Returns its ID.
    pub fn insert(&self, record: &Record) -> Result<RecordId> {
        let bytes = self.encode_body(record)?;
        let id = record.id.as_str();

        let txn = self.begin_write()?;
        {
            let mut records = txn.open_table(RECORDS)?;

            // Pre-image for CDC value capture (read before the overwrite below).
            #[cfg(feature = "cdc")]
            let cdc_before: Option<serde_json::Value> = if self.cdc_capture_values() {
                records
                    .get(id)?
                    .and_then(|g| self.decode_body(id, g.value()).ok())
                    .map(|r| r.data)
            } else {
                None
            };

            // If this ID already exists under a different table, clean up the old index.
            if let Some(guard) = records.get(id)? {
                let old_bytes: &[u8] = guard.value();
                if let Ok(old_record) = self.decode_body(id, old_bytes) {
                    if old_record.table != record.table {
                        let mut idx = txn.open_table(TABLE_INDEX)?;
                        let mut old_ids = Self::read_index(&idx, &old_record.table)?;
                        old_ids.retain(|rid| rid != &record.id);
                        if old_ids.is_empty() {
                            idx.remove(old_record.table.as_str())?;
                        } else {
                            let old_idx_bytes = serde_json::to_vec(&old_ids)?;
                            idx.insert(old_record.table.as_str(), old_idx_bytes.as_slice())?;
                        }
                    }
                }
            }

            records.insert(id, bytes.as_slice())?;

            // Update table index (with dedup check using HashSet for O(1) lookup).
            let mut idx = txn.open_table(TABLE_INDEX)?;
            let mut ids = Self::read_index(&idx, &record.table)?;
            let id_set: std::collections::HashSet<&RecordId> = ids.iter().collect();
            if !id_set.contains(&record.id) {
                ids.push(record.id.clone());
            }
            let idx_bytes = serde_json::to_vec(&ids)?;
            idx.insert(record.table.as_str(), idx_bytes.as_slice())?;

            #[cfg(feature = "cdc")]
            self.append_changelog(
                &txn,
                "insert",
                &record.table,
                id,
                cdc_before,
                self.cdc_capture_values().then(|| record.data.clone()),
            )?;
        }
        txn.commit()?;

        Ok(record.id.clone())
    }

    /// Insert multiple records in a single transaction for better throughput.
    ///
    /// Assumes all records have fresh IDs (not re-using existing IDs across tables).
    /// For upsert semantics, use `insert()` per-record instead.
    pub fn insert_batch(&self, records: &[Record]) -> Result<Vec<RecordId>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let txn = self.begin_write()?;
        {
            let mut tbl = txn.open_table(RECORDS)?;
            let mut idx = txn.open_table(TABLE_INDEX)?;

            // Group records by table to minimize index reads.
            let mut table_ids: std::collections::HashMap<&str, Vec<RecordId>> =
                std::collections::HashMap::new();

            for record in records {
                let bytes = self.encode_body(record)?;
                tbl.insert(record.id.as_str(), bytes.as_slice())?;
                table_ids
                    .entry(&record.table)
                    .or_default()
                    .push(record.id.clone());

                #[cfg(feature = "cdc")]
                self.append_changelog(
                    &txn,
                    "insert",
                    &record.table,
                    record.id.as_str(),
                    None,
                    self.cdc_capture_values().then(|| record.data.clone()),
                )?;
            }

            // Append new IDs to each table's index with dedup.
            for (table_name, new_ids) in &table_ids {
                let ids = Self::read_index(&idx, table_name)?;
                let existing: std::collections::HashSet<&RecordId> = ids.iter().collect();
                let to_add: Vec<RecordId> = new_ids
                    .iter()
                    .filter(|id| !existing.contains(id))
                    .cloned()
                    .collect();
                drop(existing);
                let mut ids = ids;
                ids.extend(to_add);
                let idx_bytes = serde_json::to_vec(&ids)?;
                idx.insert(*table_name, idx_bytes.as_slice())?;
            }
        }
        txn.commit()?;

        Ok(records.iter().map(|r| r.id.clone()).collect())
    }

    /// Get a record by ID.
    pub fn get(&self, id: &RecordId) -> Result<Option<Record>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(RECORDS)?;

        match table.get(id.as_str())? {
            Some(guard) => {
                let bytes: &[u8] = guard.value();
                let record = self.decode_body(id.as_str(), bytes)?;
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// Delete a record by ID. Returns `true` if the record existed.
    ///
    /// Uses a single write transaction to ensure atomicity between
    /// reading the record, removing it, and updating the table index.
    pub fn delete(&self, id: &RecordId) -> Result<bool> {
        let txn = self.begin_write()?;
        {
            let mut records = txn.open_table(RECORDS)?;

            // Read the record within the write transaction to get table name
            // (and, for CDC value capture, its pre-image body).
            let (table_name, _cdc_before) = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    let record = self.decode_body(id.as_str(), bytes)?;
                    #[cfg(feature = "cdc")]
                    let before = self.cdc_capture_values().then(|| record.data.clone());
                    #[cfg(not(feature = "cdc"))]
                    let before: Option<serde_json::Value> = None;
                    (record.table, before)
                }
                None => return Ok(false),
            };

            records.remove(id.as_str())?;

            // Remove from table index; drop the key if empty.
            let mut idx = txn.open_table(TABLE_INDEX)?;
            let mut ids = Self::read_index(&idx, &table_name)?;
            ids.retain(|rid| rid != id);
            if ids.is_empty() {
                idx.remove(table_name.as_str())?;
            } else {
                let idx_bytes = serde_json::to_vec(&ids)?;
                idx.insert(table_name.as_str(), idx_bytes.as_slice())?;
            }

            #[cfg(feature = "cdc")]
            self.append_changelog(&txn, "delete", &table_name, id.as_str(), _cdc_before, None)?;
        }
        txn.commit()?;

        Ok(true)
    }

    /// List records in a table with optional limit and offset.
    pub fn list(&self, table: &str, limit: usize, offset: usize) -> Result<Vec<Record>> {
        let txn = self.begin_read()?;
        let idx_table = txn.open_table(TABLE_INDEX)?;
        let ids = Self::read_index(&idx_table, table)?;

        let records_table = txn.open_table(RECORDS)?;
        let mut results = Vec::new();

        for rid in ids.into_iter().skip(offset).take(limit) {
            if let Some(guard) = records_table.get(rid.as_str())? {
                let bytes: &[u8] = guard.value();
                let record = self.decode_body(rid.as_str(), bytes)?;
                results.push(record);
            }
        }

        Ok(results)
    }

    /// Update a record's data. Returns the updated record.
    ///
    /// Uses a single write transaction to ensure atomicity.
    pub fn update(&self, id: &RecordId, data: serde_json::Value) -> Result<Record> {
        let txn = self.begin_write()?;
        let record = {
            let mut records = txn.open_table(RECORDS)?;

            // Read the current record within the write transaction.
            let mut record = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    self.decode_body(id.as_str(), bytes)?
                }
                None => return Err(AxilError::NotFound(format!("record {id}"))),
            };

            #[cfg(feature = "cdc")]
            let cdc_before: Option<serde_json::Value> =
                self.cdc_capture_values().then(|| record.data.clone());

            record.data = data;
            record.updated_at = Utc::now();

            let bytes = self.encode_body(&record)?;
            records.insert(id.as_str(), bytes.as_slice())?;

            #[cfg(feature = "cdc")]
            self.append_changelog(
                &txn,
                "update",
                &record.table,
                id.as_str(),
                cdc_before,
                self.cdc_capture_values().then(|| record.data.clone()),
            )?;
            record
        };
        txn.commit()?;

        Ok(record)
    }

    /// Overwrite a record's metadata JSON. Returns the updated record.
    ///
    /// Introduced for so consent scopes can be toggled without
    /// disturbing `data` or `updated_at`. Touches neither the table index
    /// nor plugin hooks — metadata is a side-channel.
    pub fn set_metadata(
        &self,
        id: &RecordId,
        metadata: Option<serde_json::Value>,
    ) -> Result<Record> {
        let txn = self.begin_write()?;
        let record = {
            let mut records = txn.open_table(RECORDS)?;
            let mut record = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    self.decode_body(id.as_str(), bytes)?
                }
                None => return Err(AxilError::NotFound(format!("record {id}"))),
            };
            record.metadata = metadata;
            let bytes = self.encode_body(&record)?;
            records.insert(id.as_str(), bytes.as_slice())?;
            record
        };
        txn.commit()?;
        Ok(record)
    }

    /// Count all record IDs in a given table.
    pub fn count(&self, table: &str) -> Result<usize> {
        let txn = self.begin_read()?;
        let idx = txn.open_table(TABLE_INDEX)?;
        let ids = Self::read_index(&idx, table)?;
        Ok(ids.len())
    }

    /// List all table names that have records.
    pub fn tables(&self) -> Result<Vec<String>> {
        let txn = self.begin_read()?;
        let idx = txn.open_table(TABLE_INDEX)?;
        let mut names = Vec::new();
        let iter = idx.iter()?;
        for entry in iter {
            let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            names.push(key.value().to_string());
        }
        Ok(names)
    }

    /// List all table names with their record counts in a single transaction.
    pub fn tables_with_counts(&self) -> Result<Vec<(String, usize)>> {
        let txn = self.begin_read()?;
        let idx = txn.open_table(TABLE_INDEX)?;
        let mut result = Vec::new();
        let iter = idx.iter()?;
        for entry in iter {
            let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            let bytes: &[u8] = val.value();
            let ids: Vec<RecordId> = serde_json::from_slice(bytes)?;
            result.push((key.value().to_string(), ids.len()));
        }
        Ok(result)
    }

    /// Total number of records across all tables.
    pub fn total_records(&self) -> Result<usize> {
        let txn = self.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        Ok(table.len()? as usize)
    }

    // ── diagnostic log operations ──────────────────────────────────────

    /// Append a slow query entry. Key format: timestamp + counter for ordering.
    pub fn append_slow_query(&self, key: &str, entry: &[u8]) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(SLOW_QUERIES)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read all slow query entries, ordered by key (timestamp).
    pub fn list_slow_queries(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(SLOW_QUERIES)?;
        let mut results = Vec::new();
        // Iterate in reverse (newest first) using rev().
        for entry in table.iter()?.rev() {
            let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            results.push((key.value().to_string(), val.value().to_vec()));
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    /// Clear all slow query entries.
    pub fn clear_slow_queries(&self) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(SLOW_QUERIES)?;
            // Drain all entries.
            let keys: Vec<String> = {
                let mut ks = Vec::new();
                for entry in table.iter()? {
                    let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    ks.push(key.value().to_string());
                }
                ks
            };
            for key in &keys {
                table.remove(key.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Trim slow query log to keep at most `max` entries (removes oldest).
    pub fn trim_slow_queries(&self, max: usize) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(SLOW_QUERIES)?;
            let count = table.len()? as usize;
            if count > max {
                let to_remove = count - max;
                let keys: Vec<String> = {
                    let mut ks = Vec::new();
                    for entry in table.iter()? {
                        let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                            entry?;
                        ks.push(key.value().to_string());
                        if ks.len() >= to_remove {
                            break;
                        }
                    }
                    ks
                };
                for key in &keys {
                    table.remove(key.as_str())?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Append an audit log entry.
    pub fn append_audit(&self, key: &str, entry: &[u8]) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(AUDIT_LOG)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read audit log entries, ordered newest first.
    pub fn list_audit(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(AUDIT_LOG)?;
        let mut results = Vec::new();
        for entry in table.iter()?.rev() {
            let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            results.push((key.value().to_string(), val.value().to_vec()));
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    /// Clear all audit log entries.
    pub fn clear_audit(&self) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(AUDIT_LOG)?;
            let keys: Vec<String> = {
                let mut ks = Vec::new();
                for entry in table.iter()? {
                    let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    ks.push(key.value().to_string());
                }
                ks
            };
            for key in &keys {
                table.remove(key.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Trim audit log to keep at most `max` entries (removes oldest).
    pub fn trim_audit(&self, max: usize) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(AUDIT_LOG)?;
            let count = table.len()? as usize;
            if count > max {
                let to_remove = count - max;
                let keys: Vec<String> = {
                    let mut ks = Vec::new();
                    for entry in table.iter()? {
                        let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                            entry?;
                        ks.push(key.value().to_string());
                        if ks.len() >= to_remove {
                            break;
                        }
                    }
                    ks
                };
                for key in &keys {
                    table.remove(key.as_str())?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    // ── metrics history operations ─────────────────────────────────────

    /// Append a metrics history snapshot.
    pub fn append_metrics_snapshot(&self, key: &str, entry: &[u8]) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(METRICS_HISTORY)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read metrics history entries, ordered newest first.
    pub fn list_metrics_history(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(METRICS_HISTORY)?;
        let mut results = Vec::new();
        for entry in table.iter()?.rev() {
            let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            results.push((key.value().to_string(), val.value().to_vec()));
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    /// Trim metrics history to keep at most `max` entries (removes oldest).
    pub fn trim_metrics_history(&self, max: usize) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(METRICS_HISTORY)?;
            let count = table.len()? as usize;
            if count > max {
                let to_remove = count - max;
                let keys: Vec<String> = {
                    let mut ks = Vec::new();
                    for entry in table.iter()? {
                        let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                            entry?;
                        ks.push(key.value().to_string());
                        if ks.len() >= to_remove {
                            break;
                        }
                    }
                    ks
                };
                for key in &keys {
                    table.remove(key.as_str())?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Get all record IDs across all tables (for bulk operations).
    pub fn all_record_ids(&self) -> Result<Vec<RecordId>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        let mut ids = Vec::new();
        for entry in table.iter()? {
            let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            ids.push(RecordId(key.value().to_string()));
        }
        Ok(ids)
    }

    /// Scan all records in a single pass (deserializes values directly).
    /// More efficient than all_record_ids() + get() for each.
    pub fn scan_all_records(&self) -> Result<Vec<Record>> {
        let txn = self.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (key, value): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            // With encryption on, a decode failure means the wrong key (or a
            // tampered body) — surface it rather than silently dropping rows,
            // which would otherwise turn a key mismatch into an empty scan.
            #[cfg(feature = "encryption")]
            if self.cipher.is_some() {
                records.push(self.decode_body(key.value(), value.value())?);
                continue;
            }
            if let Ok(record) = self.decode_body(key.value(), value.value()) {
                records.push(record);
            }
        }
        Ok(records)
    }

    // ── change-data-capture (cdc feature) ──────────────────────────────

    /// Whether full pre/post record bodies are captured on the `_changelog`
    /// tape (vs. the default id-only entries).
    #[cfg(feature = "cdc")]
    fn cdc_capture_values(&self) -> bool {
        self.cdc_capture_values
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Enable or disable full-body (`before`/`after`) capture on the
    /// `_changelog` tape. Id-only capture is the default; enabling value
    /// capture roughly doubles per-write cost, so it is opt-in.
    #[cfg(feature = "cdc")]
    pub fn set_cdc_capture_values(&self, enabled: bool) {
        self.cdc_capture_values
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Next monotonic `_changelog` cursor — strictly increasing even within one
    /// millisecond. On the astronomically rare per-millisecond random overflow,
    /// falls back to a fresh ULID (a single out-of-order id beats dropping the
    /// change; the next call re-establishes order).
    #[cfg(feature = "cdc")]
    fn next_change_id(&self) -> String {
        let mut generator = match self.changelog_cursor.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match generator.generate() {
            Ok(ulid) => ulid.to_string(),
            Err(_) => ulid::Ulid::new().to_string(),
        }
    }

    /// Append one `_changelog` entry inside the caller's open write
    /// transaction, then prune the oldest entries past the retention bound.
    ///
    /// Because this runs in the same `txn` that mutates `records`, the change
    /// event commits atomically with the record — a crash cannot leave one
    /// without the other.
    #[cfg(feature = "cdc")]
    fn append_changelog(
        &self,
        txn: &WriteTransaction,
        op: &str,
        table: &str,
        record_id: &str,
        before: Option<serde_json::Value>,
        after: Option<serde_json::Value>,
    ) -> Result<()> {
        let change_id = self.next_change_id();
        let entry = ChangeEntry {
            change_id: change_id.clone(),
            op: op.to_string(),
            table: table.to_string(),
            record_id: record_id.to_string(),
            before,
            after,
        };
        let bytes = self.encode_changelog(change_id.as_str(), &entry)?;
        let mut log = txn.open_table(CHANGELOG)?;
        log.insert(change_id.as_str(), bytes.as_slice())?;

        // Self-prune the oldest entries to keep the tape bounded on the write
        // path. ULID keys sort oldest-first, so removing from the front evicts
        // the oldest changes.
        let count = log.len()? as usize;
        if count > MAX_CHANGELOG_ENTRIES {
            let to_remove = count - MAX_CHANGELOG_ENTRIES;
            let mut keys = Vec::with_capacity(to_remove);
            for entry in log.iter()? {
                let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
                keys.push(key.value().to_string());
                if keys.len() >= to_remove {
                    break;
                }
            }
            for key in &keys {
                log.remove(key.as_str())?;
            }
        }
        Ok(())
    }

    /// Ordered range scan over the `_changelog` tape for entries strictly after
    /// `cursor` (a ULID `change_id`), oldest first. Pass `None` to read from the
    /// beginning of the retained tape.
    ///
    /// The returned `change_id` of the last entry is the cursor to pass on the
    /// next pull. If the requested cursor has already been pruned past the
    /// retention bound, the scan resumes from the oldest retained entry — the
    /// consumer is responsible for detecting the gap (e.g. via `_sync_meta`).
    #[cfg(feature = "cdc")]
    pub fn changes_since(&self, cursor: Option<&str>, limit: usize) -> Result<Vec<ChangeEntry>> {
        let txn = self.begin_read()?;
        let log = txn.open_table(CHANGELOG)?;
        let mut out = Vec::new();
        match cursor {
            // Exclusive lower bound: skip the cursor key itself.
            Some(c) => {
                let bounds: (std::ops::Bound<&str>, std::ops::Bound<&str>) =
                    (std::ops::Bound::Excluded(c), std::ops::Bound::Unbounded);
                let range = log.range::<&str>(bounds)?;
                for entry in range {
                    let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    out.push(self.decode_changelog(key.value(), val.value())?);
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            None => {
                for entry in log.iter()? {
                    let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    out.push(self.decode_changelog(key.value(), val.value())?);
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Total number of retained entries on the `_changelog` tape.
    #[cfg(feature = "cdc")]
    pub fn changelog_len(&self) -> Result<usize> {
        let txn = self.begin_read()?;
        let log = txn.open_table(CHANGELOG)?;
        Ok(log.len()? as usize)
    }

    /// Append one serialized [`SemanticEvent`](crate::event_log::SemanticEvent)
    /// to the `_event_log` tape under a monotonic ULID `cursor` key.
    ///
    /// The caller owns cursor generation (via [`Axil`](crate::Axil)'s shared
    /// monotonic generator) so same-millisecond writes stay strictly ordered.
    /// The entry commits in its own write transaction — it is durable independent
    /// of the record write it describes, which is acceptable for a pull-based
    /// "what changed" feed (a torn write at most drops the trailing event, never
    /// corrupts the cursor ordering).
    #[cfg(feature = "event-log")]
    pub fn append_event(&self, cursor: &str, entry: &[u8]) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut log = txn.open_table(EVENT_LOG)?;
            log.insert(cursor, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Ordered range scan over the `_event_log` tape for entries strictly after
    /// `cursor` (a monotonic ULID), oldest first. Pass `None` to read from the
    /// oldest retained entry.
    ///
    /// The cursor of the last returned entry is what the consumer passes on its
    /// next pull. If the requested cursor has already been trimmed past the
    /// retention bound the scan resumes from the oldest retained entry.
    #[cfg(feature = "event-log")]
    pub fn events_since(&self, cursor: Option<&str>, limit: usize) -> Result<Vec<Vec<u8>>> {
        let txn = self.begin_read()?;
        let log = txn.open_table(EVENT_LOG)?;
        let mut out = Vec::new();
        match cursor {
            // Exclusive lower bound: skip the cursor key itself.
            Some(c) => {
                let bounds: (std::ops::Bound<&str>, std::ops::Bound<&str>) =
                    (std::ops::Bound::Excluded(c), std::ops::Bound::Unbounded);
                for entry in log.range::<&str>(bounds)? {
                    let (_, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    out.push(val.value().to_vec());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            None => {
                for entry in log.iter()? {
                    let (_, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    out.push(val.value().to_vec());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Trim the `_event_log` tape to keep at most `max` entries (removes the
    /// oldest). ULID keys sort oldest-first, so the front of the table is evicted.
    #[cfg(feature = "event-log")]
    pub fn trim_event_log(&self, max: usize) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut log = txn.open_table(EVENT_LOG)?;
            let count = log.len()? as usize;
            if count > max {
                let to_remove = count - max;
                let mut keys = Vec::with_capacity(to_remove);
                for entry in log.iter()? {
                    let (key, _): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) =
                        entry?;
                    keys.push(key.value().to_string());
                    if keys.len() >= to_remove {
                        break;
                    }
                }
                for key in &keys {
                    log.remove(key.as_str())?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Total number of retained entries on the `_event_log` tape.
    #[cfg(feature = "event-log")]
    pub fn event_log_len(&self) -> Result<usize> {
        let txn = self.begin_read()?;
        let log = txn.open_table(EVENT_LOG)?;
        Ok(log.len()? as usize)
    }

    // ── helpers ──────────────────────────────────────────────────────

    fn read_index<T: ReadableTable<&'static str, &'static [u8]>>(
        table: &T,
        name: &str,
    ) -> Result<Vec<RecordId>> {
        match table.get(name)? {
            Some(guard) => {
                let bytes: &[u8] = guard.value();
                let ids: Vec<RecordId> = serde_json::from_slice(bytes)?;
                Ok(ids)
            }
            None => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_storage() -> (Storage, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let storage = Storage::open(&path).unwrap();
        (storage, dir)
    }

    #[test]
    fn insert_and_get() {
        let (storage, _dir) = temp_storage();
        let record = Record::new("sessions", json!({"summary": "test"}));
        let id = storage.insert(&record).unwrap();
        let fetched = storage.get(&id).unwrap().unwrap();
        assert_eq!(fetched.id, record.id);
        assert_eq!(fetched.data["summary"], "test");
    }

    #[test]
    fn get_not_found() {
        let (storage, _dir) = temp_storage();
        let id = RecordId::new();
        assert!(storage.get(&id).unwrap().is_none());
    }

    #[test]
    fn delete_existing() {
        let (storage, _dir) = temp_storage();
        let record = Record::new("sessions", json!({"x": 1}));
        let id = storage.insert(&record).unwrap();
        assert!(storage.delete(&id).unwrap());
        assert!(storage.get(&id).unwrap().is_none());
    }

    #[test]
    fn delete_not_found() {
        let (storage, _dir) = temp_storage();
        let id = RecordId::new();
        assert!(!storage.delete(&id).unwrap());
    }

    #[test]
    fn delete_removes_empty_table_from_index() {
        let (storage, _dir) = temp_storage();
        let r = Record::new("ephemeral", json!({}));
        let id = storage.insert(&r).unwrap();
        assert!(storage.tables().unwrap().contains(&"ephemeral".to_string()));
        storage.delete(&id).unwrap();
        assert!(!storage.tables().unwrap().contains(&"ephemeral".to_string()));
    }

    #[test]
    fn list_with_pagination() {
        let (storage, _dir) = temp_storage();
        for i in 0..5 {
            let r = Record::new("items", json!({"i": i}));
            storage.insert(&r).unwrap();
        }
        let all = storage.list("items", 100, 0).unwrap();
        assert_eq!(all.len(), 5);

        let page = storage.list("items", 2, 1).unwrap();
        assert_eq!(page.len(), 2);
    }

    #[test]
    fn update_record() {
        let (storage, _dir) = temp_storage();
        let record = Record::new("sessions", json!({"v": 1}));
        let id = storage.insert(&record).unwrap();
        let updated = storage.update(&id, json!({"v": 2})).unwrap();
        assert_eq!(updated.data["v"], 2);
        assert!(updated.updated_at >= record.created_at);
    }

    #[test]
    fn update_not_found() {
        let (storage, _dir) = temp_storage();
        let id = RecordId::new();
        let res = storage.update(&id, json!({}));
        assert!(res.is_err());
    }

    #[test]
    fn duplicate_id_insert_no_index_corruption() {
        let (storage, _dir) = temp_storage();
        let mut record = Record::new("items", json!({"v": 1}));
        let id = storage.insert(&record).unwrap();
        // Insert again with same ID (simulating upsert).
        record.data = json!({"v": 2});
        storage.insert(&record).unwrap();
        // Index should have only one entry, not two.
        assert_eq!(storage.count("items").unwrap(), 1);
        let fetched = storage.get(&id).unwrap().unwrap();
        assert_eq!(fetched.data["v"], 2);
    }

    #[test]
    fn cross_table_upsert_cleans_old_index() {
        let (storage, _dir) = temp_storage();
        let mut record = Record::new("table_a", json!({"v": 1}));
        storage.insert(&record).unwrap();
        assert_eq!(storage.count("table_a").unwrap(), 1);

        // Re-insert same ID under a different table.
        record.table = "table_b".to_string();
        record.data = json!({"v": 2});
        storage.insert(&record).unwrap();

        // Old table should no longer list the record.
        assert_eq!(storage.count("table_b").unwrap(), 1);
        assert!(!storage.tables().unwrap().contains(&"table_a".to_string()));
        assert_eq!(storage.total_records().unwrap(), 1);
    }

    #[test]
    fn tables_and_count() {
        let (storage, _dir) = temp_storage();
        storage.insert(&Record::new("a", json!({}))).unwrap();
        storage.insert(&Record::new("b", json!({}))).unwrap();
        storage.insert(&Record::new("a", json!({}))).unwrap();

        let mut tables = storage.tables().unwrap();
        tables.sort();
        assert_eq!(tables, vec!["a", "b"]);
        assert_eq!(storage.count("a").unwrap(), 2);
        assert_eq!(storage.count("b").unwrap(), 1);
        assert_eq!(storage.total_records().unwrap(), 3);
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist.axil");

        let id = {
            let storage = Storage::open(&path).unwrap();
            let r = Record::new("data", json!({"persisted": true}));
            storage.insert(&r).unwrap()
        };

        // Reopen
        let storage = Storage::open(&path).unwrap();
        let fetched = storage.get(&id).unwrap().unwrap();
        assert_eq!(fetched.data["persisted"], true);
    }

    #[cfg(feature = "cdc")]
    mod cdc {
        use super::*;

        #[test]
        fn changelog_cursor_is_strictly_monotonic() {
            // `ulid::Ulid::new()` is NOT monotonic within a millisecond — two
            // same-ms ids can sort out of order, which would let `changes_since`
            // skip a change past an exclusive cursor and let merge-replay reorder
            // two same-ms updates to one record. The monotonic generator forbids
            // that: every id is strictly greater than the last across rapid calls.
            let (storage, _dir) = temp_storage();
            let mut prev = String::new();
            for _ in 0..2000 {
                let id = storage.next_change_id();
                assert!(
                    id > prev,
                    "change ids must be strictly increasing: {prev:?} >= {id:?}"
                );
                prev = id;
            }
        }

        #[test]
        fn insert_appends_exactly_one_entry() {
            let (storage, _dir) = temp_storage();
            assert_eq!(storage.changelog_len().unwrap(), 0);
            let r = Record::new("notes", json!({"v": 1}));
            storage.insert(&r).unwrap();
            assert_eq!(storage.changelog_len().unwrap(), 1);
            let changes = storage.changes_since(None, 100).unwrap();
            assert_eq!(changes.len(), 1);
            assert_eq!(changes[0].op, "insert");
            assert_eq!(changes[0].table, "notes");
            assert_eq!(changes[0].record_id, r.id.to_string());
            // Id-only capture by default — no bodies.
            assert!(changes[0].before.is_none());
            assert!(changes[0].after.is_none());
        }

        #[test]
        fn update_appends_exactly_one_entry() {
            let (storage, _dir) = temp_storage();
            let r = Record::new("notes", json!({"v": 1}));
            let id = storage.insert(&r).unwrap();
            storage.update(&id, json!({"v": 2})).unwrap();
            assert_eq!(storage.changelog_len().unwrap(), 2);
            let changes = storage.changes_since(None, 100).unwrap();
            assert_eq!(changes[1].op, "update");
            assert_eq!(changes[1].record_id, id.to_string());
        }

        #[test]
        fn delete_appends_exactly_one_entry() {
            let (storage, _dir) = temp_storage();
            let r = Record::new("notes", json!({"v": 1}));
            let id = storage.insert(&r).unwrap();
            storage.delete(&id).unwrap();
            assert_eq!(storage.changelog_len().unwrap(), 2);
            let changes = storage.changes_since(None, 100).unwrap();
            assert_eq!(changes[1].op, "delete");
            assert_eq!(changes[1].record_id, id.to_string());
        }

        #[test]
        fn changes_are_in_commit_order() {
            let (storage, _dir) = temp_storage();
            let a = Record::new("t", json!({"n": "a"}));
            let b = Record::new("t", json!({"n": "b"}));
            let ida = storage.insert(&a).unwrap();
            let idb = storage.insert(&b).unwrap();
            storage.update(&ida, json!({"n": "a2"})).unwrap();
            storage.delete(&idb).unwrap();

            let changes = storage.changes_since(None, 100).unwrap();
            let ops: Vec<&str> = changes.iter().map(|c| c.op.as_str()).collect();
            assert_eq!(ops, vec!["insert", "insert", "update", "delete"]);
            // ULID cursors are strictly increasing.
            for w in changes.windows(2) {
                assert!(w[0].change_id < w[1].change_id);
            }
        }

        #[test]
        fn changes_since_cursor_is_exclusive() {
            let (storage, _dir) = temp_storage();
            for i in 0..5 {
                storage.insert(&Record::new("t", json!({ "i": i }))).unwrap();
            }
            let all = storage.changes_since(None, 100).unwrap();
            assert_eq!(all.len(), 5);
            let cursor = &all[1].change_id;
            let rest = storage.changes_since(Some(cursor), 100).unwrap();
            // Strictly after index 1 → indices 2,3,4.
            assert_eq!(rest.len(), 3);
            assert_eq!(rest[0].change_id, all[2].change_id);
        }

        #[test]
        fn value_capture_is_opt_in() {
            let (storage, _dir) = temp_storage();
            storage.set_cdc_capture_values(true);
            let r = Record::new("t", json!({"v": 1}));
            let id = storage.insert(&r).unwrap();
            storage.update(&id, json!({"v": 2})).unwrap();
            let changes = storage.changes_since(None, 100).unwrap();
            // insert: after = {v:1}
            assert_eq!(changes[0].after, Some(json!({"v": 1})));
            // update: before = {v:1}, after = {v:2}
            assert_eq!(changes[1].before, Some(json!({"v": 1})));
            assert_eq!(changes[1].after, Some(json!({"v": 2})));
        }

        #[test]
        fn record_and_changelog_share_one_txn() {
            // The changelog entry must commit atomically with the record: after a
            // reopen, the persisted record and its changelog entry are both present
            // (or both absent). We assert co-presence across a close/reopen cycle.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("cdc.axil");
            let id = {
                let storage = Storage::open(&path).unwrap();
                let r = Record::new("t", json!({"v": 1}));
                storage.insert(&r).unwrap()
            };
            let storage = Storage::open(&path).unwrap();
            assert!(storage.get(&id).unwrap().is_some());
            let changes = storage.changes_since(None, 100).unwrap();
            assert_eq!(changes.len(), 1);
            assert_eq!(changes[0].record_id, id.to_string());
        }

        #[test]
        fn batch_insert_appends_one_entry_per_record() {
            let (storage, _dir) = temp_storage();
            let recs = vec![
                Record::new("t", json!({"i": 0})),
                Record::new("t", json!({"i": 1})),
                Record::new("t", json!({"i": 2})),
            ];
            storage.insert_batch(&recs).unwrap();
            assert_eq!(storage.changelog_len().unwrap(), 3);
            let changes = storage.changes_since(None, 100).unwrap();
            assert!(changes.iter().all(|c| c.op == "insert"));
        }
    }

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

        /// CDC value-capture bodies are sealed in the `_changelog` tape (not
        /// stored in cleartext), and `changes_since` round-trips them.
        #[cfg(feature = "cdc")]
        #[test]
        fn changelog_value_capture_is_encrypted_at_rest() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let storage = Storage::open(&path).unwrap().with_cipher(key_a());
            storage.set_cdc_capture_values(true);

            let r = Record::new("secrets", json!({"summary": "changelog-needle-zzz"}));
            storage.insert(&r).unwrap();

            // changes_since decrypts and round-trips the captured after-image.
            let changes = storage.changes_since(None, 10).unwrap();
            assert!(
                changes.iter().any(|c| c
                    .after
                    .as_ref()
                    .and_then(|v| v.get("summary"))
                    .and_then(|s| s.as_str())
                    == Some("changelog-needle-zzz")),
                "changes_since should round-trip the captured body"
            );

            // The raw `_changelog` bytes on disk must NOT contain the plaintext.
            let txn = storage.begin_read().unwrap();
            let log = txn.open_table(CHANGELOG).unwrap();
            for e in log.iter().unwrap() {
                let (_, val) = e.unwrap();
                assert!(
                    !String::from_utf8_lossy(val.value()).contains("changelog-needle-zzz"),
                    "changelog body leaked in cleartext"
                );
            }
        }

        /// insert-with-key → reopen-with-key → get returns plaintext.
        #[test]
        fn round_trip_with_correct_key() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");

            let id = {
                let storage = Storage::open(&path).unwrap().with_cipher(key_a());
                let r = Record::new("secrets", json!({"summary": "classified"}));
                storage.insert(&r).unwrap()
            };

            // Reopen with the same key.
            let storage = Storage::open(&path).unwrap().with_cipher(key_a());
            let fetched = storage.get(&id).unwrap().unwrap();
            assert_eq!(fetched.data["summary"], "classified");
            assert_eq!(fetched.table, "secrets");
        }

        /// The stored bytes on disk must not contain the plaintext.
        #[test]
        fn ciphertext_is_not_plaintext_on_disk() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let storage = Storage::open(&path).unwrap().with_cipher(key_a());
            let r = Record::new("secrets", json!({"summary": "needle-marker-xyz"}));
            let id = storage.insert(&r).unwrap();

            // Reach into the raw redb body for this record and confirm the
            // marker text is absent (it is sealed).
            let txn = storage.begin_read().unwrap();
            let table = txn.open_table(RECORDS).unwrap();
            let guard = table.get(id.as_str()).unwrap().unwrap();
            let raw: &[u8] = guard.value();
            let haystack = String::from_utf8_lossy(raw);
            assert!(!haystack.contains("needle-marker-xyz"));
        }

        /// Reopen with the WRONG key → get fails cleanly (no garbage, no panic).
        #[test]
        fn wrong_key_fails_cleanly() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let id = {
                let storage = Storage::open(&path).unwrap().with_cipher(key_a());
                storage
                    .insert(&Record::new("t", json!({"v": 1})))
                    .unwrap()
            };

            let storage = Storage::open(&path).unwrap().with_cipher(key_b());
            let err = storage.get(&id).unwrap_err();
            assert!(matches!(err, AxilError::Storage(_)));
        }

        /// Reopen with NO key (cleartext handle) → get of an encrypted body
        /// fails cleanly rather than returning garbage.
        #[test]
        fn no_key_on_encrypted_db_fails_cleanly() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let id = {
                let storage = Storage::open(&path).unwrap().with_cipher(key_a());
                storage
                    .insert(&Record::new("t", json!({"v": 1})))
                    .unwrap()
            };

            // Opened without a cipher: the body is a nonce+ciphertext blob, not
            // valid JSON, so from_bytes fails cleanly.
            let storage = Storage::open(&path).unwrap();
            assert!(storage.get(&id).is_err());
        }

        /// A ciphertext moved into a different record's slot fails to decrypt
        /// (AAD is bound to the record ID).
        #[test]
        fn moved_ciphertext_fails_aad() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let storage = Storage::open(&path).unwrap().with_cipher(key_a());

            let a = Record::new("t", json!({"v": "a"}));
            let b = Record::new("t", json!({"v": "b"}));
            let ida = storage.insert(&a).unwrap();
            let idb = storage.insert(&b).unwrap();

            // Pull A's raw sealed body and try to decrypt it under B's id.
            let raw_a = {
                let txn = storage.begin_read().unwrap();
                let table = txn.open_table(RECORDS).unwrap();
                table.get(ida.as_str()).unwrap().unwrap().value().to_vec()
            };
            // Decoding A's body under B's id must fail the AAD check.
            assert!(storage.decode_body(idb.as_str(), &raw_a).is_err());
            // Sanity: under A's own id it still decodes.
            assert!(storage.decode_body(ida.as_str(), &raw_a).is_ok());
        }

        /// update and list round-trip through the cipher too.
        #[test]
        fn update_and_list_round_trip() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let storage = Storage::open(&path).unwrap().with_cipher(key_a());

            let id = storage
                .insert(&Record::new("t", json!({"v": 1})))
                .unwrap();
            storage.update(&id, json!({"v": 2})).unwrap();
            let got = storage.get(&id).unwrap().unwrap();
            assert_eq!(got.data["v"], 2);

            let listed = storage.list("t", 10, 0).unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].data["v"], 2);
        }
    }
}
