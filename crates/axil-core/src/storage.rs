use std::path::Path;

use chrono::Utc;
use redb::{
    Database, MultimapTableDefinition, ReadOnlyDatabase, ReadTransaction, ReadableDatabase,
    ReadableMultimapTable, ReadableTable, ReadableTableMetadata, TableDefinition, WriteTransaction,
};

use crate::error::{AxilError, Result};
use crate::record::{Record, RecordId};

/// redb table: record_id (string) → serialized Record (bytes).
const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records");

/// redb table: table name → its [`TableCounts`].
///
/// It keeps the name the JSON id lists lived under (see
/// [`LEGACY_TABLE_INDEX`]) with a value type older binaries do not know, and
/// that is what stops them: their open checks `table_index` against
/// `&str → &[u8]` and fails with an error that prints
/// [`TABLE_COUNTS_TYPE_NAME`], where they would otherwise read lists that no
/// longer change and miss every record written since.
const TABLE_COUNTS: TableDefinition<&str, TableCounts> = TableDefinition::new("table_index");

/// redb table: (table, position) → id of the record listed there.
///
/// Positions grow in the order ids joined the table, so a range scan over one
/// table's keys is its list order: what `list` returns, and reversed, what
/// `list_newest` returns. Record ids are not that order (ULIDs made in the
/// same millisecond are random, and imports keep foreign ids).
const TABLE_MEMBERS: TableDefinition<(&str, u64), &str> = TableDefinition::new("_table_members");

/// redb table: (table, id) → the id's position in [`TABLE_MEMBERS`], so a
/// write finds whether and where an id is listed without reading the list.
const TABLE_MEMBER_POSITIONS: TableDefinition<(&str, &str), u64> =
    TableDefinition::new("_table_member_positions");

/// The layout binaries before the keyed one wrote: table name → JSON array
/// of the ids listed under it, oldest first, rewritten whole on every insert
/// and delete. A writable open migrates it (see [`Storage::open`]); a
/// read-only handle on a file no writable handle of this version has opened
/// yet reads it as it is.
const LEGACY_TABLE_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("table_index");

/// Redb type name of [`TableCounts`], which an older binary prints when it
/// refuses the file: "table_index is of type Table<&str, …>". It is part of
/// the file format: a different name is a different type to redb, so
/// changing it makes every existing file fail to open.
const TABLE_COUNTS_TYPE_NAME: &str =
    "axil_core::TableCounts (written by a newer axil; upgrade axil to open this database)";

/// Membership counters of one table, the value of [`TABLE_COUNTS`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TableCounts {
    /// Ids listed under the table.
    listed: u64,
    /// Position the next id to join the table gets. Never reused, even after
    /// the table empties (its entry stays, with `listed` 0), so the pair
    /// changes with every insert or delete and never returns to an earlier
    /// value.
    next_position: u64,
}

impl redb::Value for TableCounts {
    type SelfType<'a> = TableCounts;
    type AsBytes<'a> = [u8; 16];

    fn fixed_width() -> Option<usize> {
        Some(16)
    }

    fn from_bytes<'a>(data: &'a [u8]) -> TableCounts
    where
        Self: 'a,
    {
        let half = |range: std::ops::Range<usize>| {
            u64::from_le_bytes(data[range].try_into().expect("fixed width of 16"))
        };
        TableCounts {
            listed: half(0..8),
            next_position: half(8..16),
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a TableCounts) -> [u8; 16]
    where
        Self: 'b,
    {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&value.listed.to_le_bytes());
        bytes[8..].copy_from_slice(&value.next_position.to_le_bytes());
        bytes
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new(TABLE_COUNTS_TYPE_NAME)
    }
}

/// Where a file keeps table membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexLayout {
    /// [`TABLE_COUNTS`], [`TABLE_MEMBERS`] and [`TABLE_MEMBER_POSITIONS`].
    Keyed,
    /// [`LEGACY_TABLE_INDEX`]. Only a read-only handle has this layout: a
    /// writable open migrates the file before anything else touches it.
    Legacy,
}

/// The layout a file's table index is in, or `None` when it has none yet (a
/// new file). A `table_index` that is neither layout (a later format) is an
/// error, never mistaken for the legacy one.
fn detect_index_layout(txn: &ReadTransaction) -> Result<Option<IndexLayout>> {
    match txn.open_table(TABLE_COUNTS) {
        Ok(_) => Ok(Some(IndexLayout::Keyed)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(keyed @ redb::TableError::TableTypeMismatch { .. }) => {
            match txn.open_table(LEGACY_TABLE_INDEX) {
                Ok(_) => Ok(Some(IndexLayout::Legacy)),
                Err(_) => Err(keyed.into()),
            }
        }
        Err(e) => Err(e.into()),
    }
}

/// The three tables that hold table membership, open in one write
/// transaction. Joining or leaving a table is a few point reads and writes,
/// whatever the table's size.
struct Membership<'txn> {
    counts: redb::Table<'txn, &'static str, TableCounts>,
    members: redb::Table<'txn, (&'static str, u64), &'static str>,
    positions: redb::Table<'txn, (&'static str, &'static str), u64>,
}

impl<'txn> Membership<'txn> {
    fn open(txn: &'txn WriteTransaction) -> Result<Self> {
        Ok(Self {
            counts: txn.open_table(TABLE_COUNTS)?,
            members: txn.open_table(TABLE_MEMBERS)?,
            positions: txn.open_table(TABLE_MEMBER_POSITIONS)?,
        })
    }

    fn counts_of(&self, table: &str) -> Result<TableCounts> {
        Ok(self
            .counts
            .get(table)?
            .map(|guard| guard.value())
            .unwrap_or_default())
    }

    /// List `id` last under `table`, unless it is listed there already, in
    /// which case it keeps its place. Returns whether it joined.
    fn join(&mut self, table: &str, id: &str) -> Result<bool> {
        // A single insert is often an upsert of a listed id: look before
        // writing, so that case dirties no page.
        if self.positions.get((table, id))?.is_some() {
            return Ok(false);
        }
        let mut counts = self.counts_of(table)?;
        let joined = self.join_counted(&mut counts, table, id)?;
        if joined {
            self.counts.insert(table, counts)?;
        }
        Ok(joined)
    }

    /// [`Membership::join`] for each `(table, id)` in order, writing each
    /// table's counters once rather than once per id. Meant for batches of
    /// mostly new ids.
    fn join_all<'a>(&mut self, rows: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<()> {
        let mut changed: std::collections::HashMap<&str, TableCounts> =
            std::collections::HashMap::new();
        for (table, id) in rows {
            let counts = match changed.entry(table) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(self.counts_of(table)?)
                }
            };
            self.join_counted(counts, table, id)?;
        }
        for (table, counts) in changed {
            self.counts.insert(table, counts)?;
        }
        Ok(())
    }

    /// List `id` under `table` against `counts`, the table's counters, which
    /// the caller writes back.
    fn join_counted(&mut self, counts: &mut TableCounts, table: &str, id: &str) -> Result<bool> {
        let position = counts.next_position;
        // One B-tree walk for a new id: the insert itself reports an id
        // already listed, whose place is then put back.
        let listed = self
            .positions
            .insert((table, id), position)?
            .map(|guard| guard.value());
        if let Some(listed) = listed {
            self.positions.insert((table, id), listed)?;
            return Ok(false);
        }
        self.members.insert((table, position), id)?;
        counts.listed += 1;
        counts.next_position += 1;
        Ok(true)
    }

    /// Drop `id` from `table`'s list. Returns whether it was listed.
    fn leave(&mut self, table: &str, id: &str) -> Result<bool> {
        let Some(position) = self.positions.remove((table, id))?.map(|g| g.value()) else {
            return Ok(false);
        };
        self.members.remove((table, position))?;
        let mut counts = self.counts_of(table)?;
        counts.listed = counts.listed.saturating_sub(1);
        self.counts.insert(table, counts)?;
        Ok(true)
    }

    /// Fingerprint of the `_entities` list as it stands in this transaction.
    fn entity_fingerprint(&self) -> Result<String> {
        Ok(keyed_entity_fingerprint(
            self.counts.get(ENTITIES_TABLE)?.map(|g| g.value()),
        ))
    }
}

/// Ids listed under `table` in a keyed index, in list order, the first
/// `offset` skipped and at most `limit` returned.
fn keyed_listed_ids<M: ReadableTable<(&'static str, u64), &'static str>>(
    members: &M,
    table: &str,
    offset: usize,
    limit: usize,
) -> Result<Vec<RecordId>> {
    let mut ids = Vec::new();
    for entry in members
        .range((table, 0u64)..=(table, u64::MAX))?
        .skip(offset)
        .take(limit)
    {
        let (_, id) = entry?;
        ids.push(RecordId(id.value().to_string()));
    }
    Ok(ids)
}

/// Decode one table's JSON id list from a legacy index.
fn read_legacy_index<T: ReadableTable<&'static str, &'static [u8]>>(
    index: &T,
    name: &str,
) -> Result<Vec<RecordId>> {
    match index.get(name)? {
        Some(guard) => Ok(serde_json::from_slice(guard.value())?),
        None => Ok(Vec::new()),
    }
}

/// Positions of ids in the `_entities` list, for breaking a tie between rows
/// that share an entity key: a point read each in the keyed layout; in the
/// legacy one, the JSON list decoded once, on the first tie.
enum EntityPositions {
    Unread,
    Legacy(std::collections::HashMap<RecordId, u64>),
}

impl EntityPositions {
    /// `rid`'s position in the `_entities` list (its last, for a legacy list
    /// that repeats it), or `None` when it is not listed.
    fn of(
        &mut self,
        storage: &Storage,
        txn: &ReadTransaction,
        rid: &RecordId,
    ) -> Result<Option<u64>> {
        if storage.layout == IndexLayout::Keyed {
            return Ok(txn
                .open_table(TABLE_MEMBER_POSITIONS)?
                .get((ENTITIES_TABLE, rid.as_str()))?
                .map(|guard| guard.value()));
        }
        if matches!(self, EntityPositions::Unread) {
            let listed = read_legacy_index(&txn.open_table(LEGACY_TABLE_INDEX)?, ENTITIES_TABLE)?;
            *self = EntityPositions::Legacy(
                listed
                    .into_iter()
                    .enumerate()
                    .map(|(pos, rid)| (rid, pos as u64))
                    .collect(),
            );
        }
        Ok(match self {
            EntityPositions::Legacy(positions) => positions.get(rid).copied(),
            EntityPositions::Unread => None,
        })
    }
}

/// Logical table whose rows the entity key index covers.
const ENTITIES_TABLE: &str = "_entities";

/// redb multimap: entity lookup key → id of every live row carrying that key.
///
/// The lookup key is what entity resolution has always keyed `_entities` by:
/// `data.canonical_id` when it is a string, else `data.name` when that is
/// (rows with neither have no key). Several rows can share a key; the one a
/// full `list("_entities")` scan would have kept — the last in list order —
/// is picked at lookup time, see [`Storage::lookup_entity_keys`].
const ENTITY_KEY_INDEX: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("_entity_key_index");

/// redb table: record id → its current entity lookup key, for every id
/// listed under `_entities`.
///
/// This mirrors membership in that id list rather than `record.table`: the two
/// can disagree (a batch insert reusing an id from another table leaves the
/// old list entry behind), and a full scan follows the list. The value is
/// `None` while the id has no key — its body has neither field, or the id is
/// listed but its body is gone.
const ENTITY_KEY_MEMBERS: TableDefinition<&str, Option<&str>> =
    TableDefinition::new("_entity_key_members");

/// redb table: storage-level marker name → value.
const STORAGE_MARKERS: TableDefinition<&str, &str> = TableDefinition::new("_storage_markers");

/// redb table: id of every record deleted while the default vector store was
/// on disk but not attached, so its vector (if it had one) outlived it.
///
/// Written in the delete's own transaction, so the note cannot be lost to a
/// crash between the two; cleared by the next handle that attaches the store
/// and removes those vectors.
const PENDING_VECTOR_DELETES: TableDefinition<&str, ()> =
    TableDefinition::new("_pending_vector_deletes");

/// Marker naming the state of the entity key index.
///
/// `ready:<fingerprint>` once the index has been built, where the fingerprint
/// is of the `_entities` id list as it stood when the index last matched it
/// (see [`keyed_entity_fingerprint`]). Every write that maintains the index
/// and changes that list re-stamps the fingerprint in the same transaction.
/// A writer that predates the index changes the list without re-stamping it,
/// so the mismatch shows at the next lookup and the index is treated as stale
/// rather than trusted.
///
/// Such writers are binaries that keep the legacy JSON id lists, which cannot
/// open a file in the keyed layout, so their writes reach a file only before
/// this version first opens it writable. Migrating the file carries the
/// marker over only when it matches the list it is migrating (see
/// [`Storage::migrate_legacy_index`]); a marker an older binary had already
/// made stale stays stale.
///
/// `failed:<fingerprint>:<id>` after a build stopped on row `<id>`, whose body
/// does not decode, so later calls do not repeat the same doomed build until
/// the list changes. Absent: never built, or dropped.
const ENTITY_KEY_INDEX_MARKER: &str = "entity_key_index_v1";

/// Marker value prefix for a built index; see [`ENTITY_KEY_INDEX_MARKER`].
const ENTITY_KEY_READY: &str = "ready:";

/// Marker value prefix for a failed build; see [`ENTITY_KEY_INDEX_MARKER`].
const ENTITY_KEY_FAILED: &str = "failed:";

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

/// How a write changes an id's membership in the `_entities` id list.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EntityMembership {
    /// The id is (or stays) listed under `_entities`.
    Join,
    /// The id is dropped from the `_entities` list.
    Leave,
    /// The write does not touch the `_entities` list.
    Keep,
}

/// The key entity resolution files an `_entities` body under:
/// `canonical_id` when it is a string, else `name` when that is.
fn entity_lookup_key(data: &serde_json::Value) -> Option<&str> {
    data.get("canonical_id")
        .and_then(|v| v.as_str())
        .or_else(|| data.get("name").and_then(|v| v.as_str()))
}

/// Result of [`Storage::lookup_entity_keys`].
#[derive(Debug, Clone, PartialEq)]
pub enum EntityKeyLookup {
    /// The index answered: each key found, mapped to the row a full scan
    /// would pick. Keys with no row are absent.
    Found(std::collections::HashMap<String, RecordId>),
    /// The index is not built, its last build failed, or this handle does not
    /// use it. Scan instead; [`Storage::ensure_entity_key_index`] builds it.
    Unavailable,
    /// The index disagrees with the rows: a writer that does not maintain it
    /// (an older binary on the same file) changed `_entities` since it was
    /// built. Scan instead, and rebuild it with
    /// [`Storage::rebuild_entity_key_index`].
    Stale,
}

impl EntityKeyLookup {
    /// The resolved map when the index answered, else `None`.
    pub fn into_found(self) -> Option<std::collections::HashMap<String, RecordId>> {
        match self {
            EntityKeyLookup::Found(found) => Some(found),
            EntityKeyLookup::Unavailable | EntityKeyLookup::Stale => None,
        }
    }
}

/// Health of the entity key index, as [`Storage::entity_key_index_status`]
/// reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityKeyIndexStatus {
    /// Not built yet (it is built by the first auto-link that needs it), or
    /// this handle does not use it.
    Absent,
    /// Built and in step with the `_entities` list.
    Current,
    /// Built, but the `_entities` list changed without it: a writer that does
    /// not maintain it wrote to the file. Lookups fall back to scanning until
    /// it is rebuilt.
    Stale,
    /// The last build stopped on an `_entities` row whose body does not
    /// decode (corrupt, or sealed under an encryption key this handle does not
    /// have), and the list has not changed since. Entity resolution scans
    /// instead, as it did before the index existed.
    Failed {
        /// Id of the row that stopped the build.
        row: String,
    },
}

/// What [`Storage::rebuild_entity_key_index`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EntityKeyIndexRebuild {
    /// Distinct ids in the `_entities` list, all of which the index now
    /// covers.
    pub rows: usize,
    /// `_entities` ids whose index entries had to be written or removed to
    /// bring the index in step with the rows; 0 when it already was.
    pub changed: usize,
    /// Whether anything was committed: changed entries, or a fresh ready
    /// stamp on an index that was stale or not built. `false` means the index
    /// was already current and the call wrote nothing.
    pub written: bool,
}

/// Fingerprint of the `_entities` id list in the keyed layout: its
/// [`TableCounts`], so comparing it costs one point read. Every insert into
/// the list advances `next_position` and every delete lowers `listed`, and
/// `next_position` never goes back, so no sequence of changes returns it to
/// an earlier value. The `m` prefix keeps it apart from every
/// [`legacy_entity_fingerprint`], so a marker stamped against the legacy
/// list never matches a keyed one.
fn keyed_entity_fingerprint(counts: Option<TableCounts>) -> String {
    let counts = counts.unwrap_or_default();
    format!("m{}.{}", counts.listed, counts.next_position)
}

/// Fingerprint of the raw `_entities` entry of a legacy `table_index`
/// (`None` when the table has no rows): its byte length and CRC-32, with no
/// JSON decode. The length alone changes with every insert or delete of an
/// id of a different length; the checksum covers the rest, such as one id
/// deleted and another of the same length appended.
fn legacy_entity_fingerprint(list: Option<&[u8]>) -> String {
    match list {
        Some(bytes) => format!("{:x}.{:08x}", bytes.len(), crc32fast::hash(bytes)),
        None => "empty".to_string(),
    }
}

/// Re-stamp the entity key index marker from fingerprint `from` to `to`,
/// when the marker was stamped against `from`: the list it describes moved
/// to another layout unchanged. A marker stamped against anything else (an
/// older binary changed the list since) is left as it is, so it stays stale.
fn restamp_entity_marker(txn: &WriteTransaction, from: &str, to: &str) -> Result<()> {
    let mut markers = txn.open_table(STORAGE_MARKERS)?;
    let restamped = match markers.get(ENTITY_KEY_INDEX_MARKER)? {
        Some(guard) => {
            let value = guard.value();
            if value.strip_prefix(ENTITY_KEY_READY) == Some(from) {
                Some(format!("{ENTITY_KEY_READY}{to}"))
            } else {
                value
                    .strip_prefix(ENTITY_KEY_FAILED)
                    .and_then(|rest| rest.split_once(':'))
                    .filter(|(stamped, _)| *stamped == from)
                    .map(|(_, row)| format!("{ENTITY_KEY_FAILED}{to}:{row}"))
            }
        }
        None => None,
    };
    if let Some(value) = restamped {
        markers.insert(ENTITY_KEY_INDEX_MARKER, value.as_str())?;
    }
    Ok(())
}

/// An `_entities` row whose body does not decode, which stops an index build.
struct UndecodableEntityRow {
    id: String,
    error: AxilError,
}

/// Where the entity key index and the `_entities` rows disagree, as the
/// entries that would bring the index in step.
#[derive(Default)]
struct EntityKeyDiff {
    /// Distinct ids the `_entities` list holds.
    rows: usize,
    /// Member entries to write: id and the key its body carries (`None`: the
    /// id is listed but has no key).
    member_sets: Vec<(String, Option<String>)>,
    /// Member entries of ids the list no longer holds.
    member_removals: Vec<String>,
    /// `(key, id)` pairs the key multimap lacks.
    pair_inserts: Vec<(String, String)>,
    /// `(key, id)` pairs no listed row carries any more.
    pair_removals: Vec<(String, String)>,
}

impl EntityKeyDiff {
    fn is_empty(&self) -> bool {
        self.member_sets.is_empty()
            && self.member_removals.is_empty()
            && self.pair_inserts.is_empty()
            && self.pair_removals.is_empty()
    }

    /// Distinct ids with at least one entry to write or remove.
    fn rows_affected(&self) -> usize {
        let mut ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
        ids.extend(self.member_sets.iter().map(|(id, _)| id.as_str()));
        ids.extend(self.member_removals.iter().map(String::as_str));
        ids.extend(self.pair_inserts.iter().map(|(_, id)| id.as_str()));
        ids.extend(self.pair_removals.iter().map(|(_, id)| id.as_str()));
        ids.len()
    }
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
    /// Set by [`Storage::set_entity_key_index_enabled`]: this handle neither
    /// uses nor maintains the entity key index.
    entity_key_index_off: std::sync::atomic::AtomicBool,
    /// Where this file keeps table membership. Fixed for the handle's life:
    /// a writable handle migrated the file to [`IndexLayout::Keyed`] on open,
    /// and nothing else can write the file while either kind of handle has
    /// it open.
    layout: IndexLayout,
}

impl Storage {
    /// Open (or create) a writable database at the given path. The installed
    /// open guard runs first (see [`crate::open_guard`]).
    ///
    /// A file whose table index is still in the legacy layout (one JSON id
    /// list per table, rewritten whole on every insert and delete) is
    /// migrated to the keyed layout first, in one write transaction: a crash
    /// or error leaves it exactly as it was, and the next open tries again.
    /// Once migrated it stays migrated, so later opens write nothing for it.
    ///
    /// Binaries that only know the legacy layout cannot open a migrated file:
    /// their open fails with an error that says to upgrade axil (`table_index
    /// is of type Table<&str, axil_core::TableCounts (written by a newer
    /// axil; …)>`), rather than reading id lists that no longer change. To
    /// hand a file back to one, see [`Storage::downgrade_table_index`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        crate::open_guard(path.as_ref());
        let db = Database::create(path.as_ref())?;

        #[allow(unused_mut)]
        let mut tables = vec![RECORDS, SLOW_QUERIES, AUDIT_LOG, METRICS_HISTORY];
        #[cfg(feature = "cdc")]
        tables.push(CHANGELOG);
        #[cfg(feature = "event-log")]
        tables.push(EVENT_LOG);
        // Only a store missing a table, or still in the legacy layout, needs
        // a write. Opening an existing one commits nothing, so a read-only
        // command leaves the file untouched.
        let (complete, layout) = {
            let txn = db.begin_read()?;
            let mut complete = true;
            for table in &tables {
                match txn.open_table(*table) {
                    Ok(_) => {}
                    Err(redb::TableError::TableDoesNotExist(_)) => {
                        complete = false;
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            let layout = detect_index_layout(&txn)?;
            if layout == Some(IndexLayout::Keyed) {
                for exists in [
                    txn.open_table(TABLE_MEMBERS).map(|_| ()),
                    txn.open_table(TABLE_MEMBER_POSITIONS).map(|_| ()),
                ] {
                    match exists {
                        Ok(()) => {}
                        Err(redb::TableError::TableDoesNotExist(_)) => complete = false,
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            (complete, layout)
        };
        if !complete || layout != Some(IndexLayout::Keyed) {
            let txn = db.begin_write()?;
            let created = (|| -> Result<()> {
                for table in &tables {
                    let _ = txn.open_table(*table)?;
                }
                if layout == Some(IndexLayout::Legacy) {
                    Self::migrate_legacy_index(&txn)?;
                }
                drop(Membership::open(&txn)?);
                Ok(())
            })();
            match created {
                Ok(()) => txn.commit()?,
                Err(e) => {
                    txn.abort()?;
                    return Err(e);
                }
            }
        }

        Ok(Self {
            db: StorageDb::Writable(db),
            #[cfg(feature = "cdc")]
            cdc_capture_values: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "cdc")]
            changelog_cursor: std::sync::Mutex::new(ulid::Generator::new()),
            #[cfg(feature = "encryption")]
            cipher: None,
            entity_key_index_off: std::sync::atomic::AtomicBool::new(false),
            layout: IndexLayout::Keyed,
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
    /// [`Storage::open`] path). The installed open guard runs first.
    ///
    /// A file still in the legacy table-index layout is read as it is, not
    /// migrated: lists, counts and the entity key index answer from the JSON
    /// id lists until a writable open migrates them.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        crate::open_guard(path.as_ref());
        let db = ReadOnlyDatabase::open(path.as_ref())?;
        let layout = detect_index_layout(&db.begin_read()?)?.unwrap_or(IndexLayout::Keyed);
        Ok(Self {
            db: StorageDb::ReadOnly(db),
            #[cfg(feature = "cdc")]
            cdc_capture_values: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "cdc")]
            changelog_cursor: std::sync::Mutex::new(ulid::Generator::new()),
            #[cfg(feature = "encryption")]
            cipher: None,
            entity_key_index_off: std::sync::atomic::AtomicBool::new(false),
            layout,
        })
    }

    /// Whether the database at `path` still keeps table membership in the
    /// legacy layout, which the next writable [`Storage::open`] converts.
    ///
    /// Asked through a read-only open, so it writes nothing, and with the
    /// open guard skipped, so it never asks a process holding the database
    /// to let go of it. It fails where that open fails: a missing file, one
    /// a writer holds ([`AxilError::Busy`](crate::AxilError::Busy)), one not
    /// closed cleanly.
    pub fn table_index_conversion_pending(path: impl AsRef<Path>) -> Result<bool> {
        let storage = crate::without_open_guard(|| Self::open_read_only(path))?;
        Ok(storage.layout == IndexLayout::Legacy)
    }

    /// Move a legacy table index (one JSON id list per table) to the keyed
    /// layout inside `txn`, which the caller commits only if this succeeds.
    ///
    /// Each list keeps its order. An id a list holds more than once (a batch
    /// insert that repeated an id could do that) is kept at its first place,
    /// which is where a later insert of it would have left it. The legacy
    /// `table_index` is deleted and the name taken by [`TABLE_COUNTS`], whose
    /// value type older binaries refuse.
    ///
    /// A ready or failed entity key index marker stamped against the
    /// `_entities` list being migrated is re-stamped against the keyed list,
    /// so the index stays usable. One stamped against anything else means a
    /// binary older than the index changed the list after it was built: it is
    /// left as it is, and the index stays stale until it is rebuilt.
    fn migrate_legacy_index(txn: &WriteTransaction) -> Result<()> {
        let (lists, legacy_fingerprint) = {
            let legacy = txn.open_table(LEGACY_TABLE_INDEX)?;
            let mut lists = Vec::new();
            for entry in legacy.iter()? {
                let (name, bytes) = entry?;
                lists.push((name.value().to_string(), bytes.value().to_vec()));
            }
            let fingerprint =
                legacy_entity_fingerprint(legacy.get(ENTITIES_TABLE)?.as_ref().map(|g| g.value()));
            (lists, fingerprint)
        };
        txn.delete_table(LEGACY_TABLE_INDEX)?;

        let mut membership = Membership::open(txn)?;
        for (table, bytes) in &lists {
            // Decoded one table at a time, after earlier tables are written:
            // a list that does not decode fails the whole transaction.
            let ids: Vec<RecordId> = serde_json::from_slice(bytes).map_err(|e| {
                AxilError::Storage(
                    format!(
                        "the table index entry of `{table}` is not a JSON id list ({e}); \
                         the file was left in its legacy layout"
                    )
                    .into(),
                )
            })?;
            let mut first_place: std::collections::HashMap<&str, u64> =
                std::collections::HashMap::with_capacity(ids.len());
            for rid in &ids {
                let next = first_place.len() as u64;
                first_place.entry(rid.as_str()).or_insert(next);
            }
            let mut in_order: Vec<(&str, u64)> =
                first_place.iter().map(|(id, pos)| (*id, *pos)).collect();
            in_order.sort_unstable_by_key(|(_, pos)| *pos);
            for (id, position) in &in_order {
                membership
                    .members
                    .insert((table.as_str(), *position), *id)?;
            }
            // Sorted by id, so the writes walk this B-tree in key order.
            in_order.sort_unstable();
            for (id, position) in &in_order {
                membership
                    .positions
                    .insert((table.as_str(), *id), *position)?;
            }
            let listed = in_order.len() as u64;
            membership.counts.insert(
                table.as_str(),
                TableCounts {
                    listed,
                    next_position: listed,
                },
            )?;
        }
        let keyed_fingerprint = membership.entity_fingerprint()?;
        drop(membership);
        restamp_entity_marker(txn, &legacy_fingerprint, &keyed_fingerprint)
    }

    /// Convert this file's table index back to the legacy layout (one JSON
    /// id list per table), the only one binaries before the keyed layout can
    /// open, and close the handle.
    ///
    /// Lists keep their order. A current entity key index stays current for
    /// those binaries; a stale one stays stale. This is the way back to an
    /// older binary: the next writable [`Storage::open`] by this version
    /// migrates the file again, so run it last. One write transaction, so a
    /// crash leaves the file as it was. Fails with [`AxilError::Busy`] on a
    /// read-only handle.
    pub fn downgrade_table_index(self) -> Result<()> {
        let txn = self.begin_write()?;
        let converted = (|| -> Result<()> {
            let (lists, keyed_fingerprint) = {
                let membership = Membership::open(&txn)?;
                let mut lists = Vec::new();
                for entry in membership.counts.iter()? {
                    let (table, counts) = entry?;
                    if counts.value().listed == 0 {
                        continue;
                    }
                    let table = table.value().to_string();
                    let ids = keyed_listed_ids(&membership.members, &table, 0, usize::MAX)?;
                    lists.push((table, ids));
                }
                (lists, membership.entity_fingerprint()?)
            };
            txn.delete_table(TABLE_COUNTS)?;
            txn.delete_table(TABLE_MEMBERS)?;
            txn.delete_table(TABLE_MEMBER_POSITIONS)?;
            let legacy_fingerprint = {
                let mut legacy = txn.open_table(LEGACY_TABLE_INDEX)?;
                for (table, ids) in &lists {
                    let bytes = serde_json::to_vec(ids)?;
                    legacy.insert(table.as_str(), bytes.as_slice())?;
                }
                let fingerprint = legacy_entity_fingerprint(
                    legacy.get(ENTITIES_TABLE)?.as_ref().map(|g| g.value()),
                );
                fingerprint
            };
            restamp_entity_marker(&txn, &keyed_fingerprint, &legacy_fingerprint)
        })();
        match converted {
            Ok(()) => txn.commit()?,
            Err(e) => {
                txn.abort()?;
                return Err(e);
            }
        }
        Ok(())
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

            let old_table: Option<String> = match records.get(id)? {
                Some(guard) => self.decode_body(id, guard.value()).ok().map(|r| r.table),
                None => None,
            };
            let left_entities =
                old_table.as_deref() == Some(ENTITIES_TABLE) && record.table != ENTITIES_TABLE;
            let entities_listed = record.table == ENTITIES_TABLE || left_entities;
            let mut membership = Membership::open(&txn)?;
            // Checked before the `_entities` list changes below: the marker's
            // fingerprint is of the list as the index last saw it.
            let listed_before = entities_listed
                .then(|| membership.entity_fingerprint())
                .transpose()?;
            let maintain = self.entity_index_write_begin(&txn, listed_before.as_deref())?;

            // An id rewritten under another table leaves that table's list.
            if let Some(old_table) = old_table.filter(|t| *t != record.table) {
                membership.leave(&old_table, id)?;
            }

            records.insert(id, bytes.as_slice())?;
            membership.join(&record.table, id)?;

            let entity_membership = if record.table == ENTITIES_TABLE {
                EntityMembership::Join
            } else if left_entities {
                EntityMembership::Leave
            } else {
                EntityMembership::Keep
            };
            self.sync_entity_key(&txn, maintain, id, entity_membership, Some(&record.data))?;
            if maintain && entities_listed {
                Self::entity_index_write_finish(&txn, &membership.entity_fingerprint()?)?;
            }

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
            let mut membership = Membership::open(&txn)?;
            let entities_listed = records.iter().any(|r| r.table == ENTITIES_TABLE);
            let listed_before = entities_listed
                .then(|| membership.entity_fingerprint())
                .transpose()?;
            let maintain = self.entity_index_write_begin(&txn, listed_before.as_deref())?;

            for record in records {
                let bytes = self.encode_body(record)?;
                tbl.insert(record.id.as_str(), bytes.as_slice())?;

                // A batch never removes an id from another table's list, so
                // an id only ever joins `_entities` here. Syncing per record,
                // in order, leaves the key of the last body written for a
                // repeated id — the body `records` ends up holding.
                let entity_membership = if record.table == ENTITIES_TABLE {
                    EntityMembership::Join
                } else {
                    EntityMembership::Keep
                };
                self.sync_entity_key(
                    &txn,
                    maintain,
                    record.id.as_str(),
                    entity_membership,
                    Some(&record.data),
                )?;

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

            // Listed in batch order; an id the batch repeats, or that the
            // table already lists, keeps its first place.
            membership.join_all(records.iter().map(|r| (r.table.as_str(), r.id.as_str())))?;
            if maintain && entities_listed {
                Self::entity_index_write_finish(&txn, &membership.entity_fingerprint()?)?;
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
        self.delete_noting_vector(id, false)
    }

    /// Delete a record by ID and, when `note_vector` is set, record in the same
    /// transaction that its vector still has to go (see
    /// [`Storage::pending_vector_deletes`]). The note is written only when the
    /// record existed. Returns `true` if the record existed.
    pub(crate) fn delete_noting_vector(&self, id: &RecordId, note_vector: bool) -> Result<bool> {
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
            if note_vector {
                txn.open_table(PENDING_VECTOR_DELETES)?
                    .insert(id.as_str(), ())?;
            }

            let mut membership = Membership::open(&txn)?;
            let entities_listed = table_name == ENTITIES_TABLE;
            let listed_before = entities_listed
                .then(|| membership.entity_fingerprint())
                .transpose()?;
            let maintain = self.entity_index_write_begin(&txn, listed_before.as_deref())?;
            membership.leave(&table_name, id.as_str())?;

            let entity_membership = if table_name == ENTITIES_TABLE {
                EntityMembership::Leave
            } else {
                EntityMembership::Keep
            };
            self.sync_entity_key(&txn, maintain, id.as_str(), entity_membership, None)?;
            if maintain && entities_listed {
                Self::entity_index_write_finish(&txn, &membership.entity_fingerprint()?)?;
            }

            #[cfg(feature = "cdc")]
            self.append_changelog(&txn, "delete", &table_name, id.as_str(), _cdc_before, None)?;
        }
        txn.commit()?;

        Ok(true)
    }

    /// Ids of records deleted with a vector note (see
    /// [`Storage::delete_noting_vector`]) that no handle has cleared yet.
    pub(crate) fn pending_vector_deletes(&self) -> Result<Vec<RecordId>> {
        let txn = self.begin_read()?;
        let table = match txn.open_table(PENDING_VECTOR_DELETES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut ids = Vec::new();
        for entry in table.iter()? {
            let (key, _) = entry?;
            ids.push(RecordId(key.value().to_string()));
        }
        Ok(ids)
    }

    /// Clear these ids' vector notes, once their vectors are gone.
    pub(crate) fn clear_pending_vector_deletes(&self, ids: &[RecordId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let txn = self.begin_write()?;
        {
            let mut table = txn.open_table(PENDING_VECTOR_DELETES)?;
            for id in ids {
                table.remove(id.as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Whether a keyed index lists anything under `table`. One point read,
    /// which spares the range scan for the many lookups of tables that are
    /// empty or were never written (per-table settings read on every insert).
    fn keyed_lists_any(&self, txn: &ReadTransaction, table: &str) -> Result<bool> {
        Ok(txn
            .open_table(TABLE_COUNTS)?
            .get(table)?
            .is_some_and(|guard| guard.value().listed > 0))
    }

    /// Ids listed under `table`, in list order, the first `offset` skipped
    /// and at most `limit` returned.
    fn listed_ids(
        &self,
        txn: &ReadTransaction,
        table: &str,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<RecordId>> {
        match self.layout {
            IndexLayout::Keyed => {
                if !self.keyed_lists_any(txn, table)? {
                    return Ok(Vec::new());
                }
                keyed_listed_ids(&txn.open_table(TABLE_MEMBERS)?, table, offset, limit)
            }
            IndexLayout::Legacy => {
                let ids = read_legacy_index(&txn.open_table(LEGACY_TABLE_INDEX)?, table)?;
                Ok(ids.into_iter().skip(offset).take(limit).collect())
            }
        }
    }

    /// List records in a table with optional limit and offset.
    ///
    /// Records come in list order: the order their ids first joined the
    /// table (a re-insert keeps an id's place; an id moved in from another
    /// table joins last). `offset` and `limit` count listed ids, so a listed
    /// id whose record is gone takes a slot and yields nothing.
    pub fn list(&self, table: &str, limit: usize, offset: usize) -> Result<Vec<Record>> {
        let txn = self.begin_read()?;
        let ids = self.listed_ids(&txn, table, offset, limit)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let records_table = txn.open_table(RECORDS)?;
        let mut results = Vec::new();

        for rid in ids {
            if let Some(guard) = records_table.get(rid.as_str())? {
                let bytes: &[u8] = guard.value();
                let record = self.decode_body(rid.as_str(), bytes)?;
                results.push(record);
            }
        }

        Ok(results)
    }

    /// The newest `limit` records of a table, newest first: the end of
    /// [`Storage::list`]'s order, reversed, skipping listed ids whose record
    /// is gone. Reads only that tail.
    pub fn list_newest(&self, table: &str, limit: usize) -> Result<Vec<Record>> {
        let txn = self.begin_read()?;
        if self.layout == IndexLayout::Keyed && !self.keyed_lists_any(&txn, table)? {
            return Ok(Vec::new());
        }
        let records_table = txn.open_table(RECORDS)?;
        let mut results = Vec::new();
        let mut push = |rid: &str| -> Result<bool> {
            if results.len() == limit {
                return Ok(false);
            }
            if let Some(guard) = records_table.get(rid)? {
                results.push(self.decode_body(rid, guard.value())?);
            }
            Ok(true)
        };
        match self.layout {
            IndexLayout::Keyed => {
                let members = txn.open_table(TABLE_MEMBERS)?;
                for entry in members.range((table, 0u64)..=(table, u64::MAX))?.rev() {
                    let (_, id) = entry?;
                    if !push(id.value())? {
                        break;
                    }
                }
            }
            IndexLayout::Legacy => {
                let ids = read_legacy_index(&txn.open_table(LEGACY_TABLE_INDEX)?, table)?;
                for rid in ids.iter().rev() {
                    if !push(rid.as_str())? {
                        break;
                    }
                }
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

            // The table (and so the id's list membership) is unchanged; only
            // the key can move, e.g. a provisional entity upgraded to its
            // canonical id.
            let maintain = self.entity_index_write_begin(&txn, None)?;
            self.sync_entity_key(
                &txn,
                maintain,
                id.as_str(),
                EntityMembership::Keep,
                Some(&record.data),
            )?;

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

    /// Count all record IDs in a given table (the ids it lists, including any
    /// whose record is gone).
    pub fn count(&self, table: &str) -> Result<usize> {
        let txn = self.begin_read()?;
        match self.layout {
            IndexLayout::Keyed => Ok(txn
                .open_table(TABLE_COUNTS)?
                .get(table)?
                .map_or(0, |guard| guard.value().listed as usize)),
            IndexLayout::Legacy => {
                Ok(read_legacy_index(&txn.open_table(LEGACY_TABLE_INDEX)?, table)?.len())
            }
        }
    }

    /// List all table names that have records, in name order.
    pub fn tables(&self) -> Result<Vec<String>> {
        if self.layout == IndexLayout::Keyed {
            return Ok(self
                .tables_with_counts()?
                .into_iter()
                .map(|(name, _)| name)
                .collect());
        }
        // A legacy index drops a table's entry when its list empties, so
        // the names alone answer, with no list decoded.
        let txn = self.begin_read()?;
        let mut names = Vec::new();
        for entry in txn.open_table(LEGACY_TABLE_INDEX)?.iter()? {
            let (name, _) = entry?;
            names.push(name.value().to_string());
        }
        Ok(names)
    }

    /// List all table names with their record counts in a single
    /// transaction, in name order.
    pub fn tables_with_counts(&self) -> Result<Vec<(String, usize)>> {
        let txn = self.begin_read()?;
        let mut result = Vec::new();
        match self.layout {
            IndexLayout::Keyed => {
                for entry in txn.open_table(TABLE_COUNTS)?.iter()? {
                    let (name, counts) = entry?;
                    let listed = counts.value().listed;
                    // A table that emptied keeps its counters, unlisted.
                    if listed > 0 {
                        result.push((name.value().to_string(), listed as usize));
                    }
                }
            }
            IndexLayout::Legacy => {
                for entry in txn.open_table(LEGACY_TABLE_INDEX)?.iter()? {
                    let (name, bytes) = entry?;
                    let ids: Vec<RecordId> = serde_json::from_slice(bytes.value())?;
                    result.push((name.value().to_string(), ids.len()));
                }
            }
        }
        Ok(result)
    }

    /// Total number of records across all tables.
    pub fn total_records(&self) -> Result<usize> {
        let txn = self.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        Ok(table.len()? as usize)
    }

    // ── entity key index ───────────────────────────────────────────────

    /// Whether this handle may read and maintain the entity key index.
    ///
    /// The index holds entity names and canonical ids in cleartext, so a handle
    /// with an encryption cipher neither uses nor maintains it: its first write
    /// discards the index (see [`Storage::entity_index_write_begin`]), and its
    /// lookups fall back to decrypting and scanning the rows. The same holds
    /// for a handle switched off with [`Storage::set_entity_key_index_enabled`].
    pub fn entity_key_index_enabled(&self) -> bool {
        #[cfg(feature = "encryption")]
        if self.cipher.is_some() {
            return false;
        }
        !self
            .entity_key_index_off
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Stop (or resume) using the entity key index on this handle.
    ///
    /// While off, lookups report [`EntityKeyLookup::Unavailable`], nothing is
    /// built, and the next write discards the index so no later handle trusts
    /// an index that missed this handle's writes. Entity resolution then scans
    /// `_entities` the way it did before the index existed, which is what a
    /// comparison against that behaviour needs.
    pub fn set_entity_key_index_enabled(&self, enabled: bool) {
        self.entity_key_index_off
            .store(!enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Decide, at the start of a write transaction, whether it maintains the
    /// entity key index. `listed_before` is the fingerprint of the
    /// `_entities` list as it stands before this write changes it (see
    /// [`Membership::entity_fingerprint`]), or `None` when the write does not
    /// change that list.
    ///
    /// The index is maintained only while the marker says it is ready. A write
    /// that changes the `_entities` list first compares the marker's
    /// fingerprint with the list as it is now: a mismatch means a writer that
    /// does not maintain the index changed the list, so the marker is removed
    /// here and stays removed until a rebuild, even if later writes happen to
    /// restore the list's bytes. The caller re-stamps the marker with
    /// [`Storage::entity_index_write_finish`] after its own list change.
    ///
    /// A handle that does not use the index discards it: the marker and both
    /// tables go in this transaction, so a later cleartext handle cannot trust
    /// an index that missed this write, and entity names and canonical ids a
    /// cleartext store indexed do not outlive a re-seal made under a cipher.
    fn entity_index_write_begin(
        &self,
        txn: &WriteTransaction,
        listed_before: Option<&str>,
    ) -> Result<bool> {
        if !self.entity_key_index_enabled() {
            {
                let mut markers = txn.open_table(STORAGE_MARKERS)?;
                markers.remove(ENTITY_KEY_INDEX_MARKER)?;
            }
            txn.delete_table(ENTITY_KEY_MEMBERS)?;
            txn.delete_multimap_table(ENTITY_KEY_INDEX)?;
            return Ok(false);
        }
        let mut markers = txn.open_table(STORAGE_MARKERS)?;
        let stamped = match markers.get(ENTITY_KEY_INDEX_MARKER)? {
            Some(guard) => match guard.value().strip_prefix(ENTITY_KEY_READY) {
                Some(fingerprint) => fingerprint.to_string(),
                None => return Ok(false),
            },
            None => return Ok(false),
        };
        let Some(current) = listed_before else {
            return Ok(true);
        };
        if current == stamped {
            return Ok(true);
        }
        markers.remove(ENTITY_KEY_INDEX_MARKER)?;
        Ok(false)
    }

    /// Re-stamp the ready marker with `fingerprint`, that of the `_entities`
    /// list a maintained write has just produced.
    fn entity_index_write_finish(txn: &WriteTransaction, fingerprint: &str) -> Result<()> {
        let value = format!("{ENTITY_KEY_READY}{fingerprint}");
        let mut markers = txn.open_table(STORAGE_MARKERS)?;
        markers.insert(ENTITY_KEY_INDEX_MARKER, value.as_str())?;
        Ok(())
    }

    /// Fingerprint of the `_entities` list as a read transaction sees it, in
    /// this file's layout.
    fn entity_fingerprint_read(&self, txn: &ReadTransaction) -> Result<String> {
        Ok(match self.layout {
            IndexLayout::Keyed => keyed_entity_fingerprint(
                txn.open_table(TABLE_COUNTS)?
                    .get(ENTITIES_TABLE)?
                    .map(|guard| guard.value()),
            ),
            IndexLayout::Legacy => legacy_entity_fingerprint(
                txn.open_table(LEGACY_TABLE_INDEX)?
                    .get(ENTITIES_TABLE)?
                    .as_ref()
                    .map(|guard| guard.value()),
            ),
        })
    }

    /// Bring the entity key index up to date for one id, inside the write
    /// transaction that just changed its body or its `_entities` list
    /// membership. `maintain` is what [`Storage::entity_index_write_begin`]
    /// decided for this transaction; `data` is the body `records` now holds
    /// for the id (`None` once it is deleted).
    ///
    /// For an id that is not and does not become an `_entities` member this is
    /// a single point read.
    fn sync_entity_key(
        &self,
        txn: &WriteTransaction,
        maintain: bool,
        id: &str,
        membership: EntityMembership,
        data: Option<&serde_json::Value>,
    ) -> Result<()> {
        if !maintain {
            return Ok(());
        }
        let mut members = txn.open_table(ENTITY_KEY_MEMBERS)?;
        let old: Option<Option<String>> = members
            .get(id)?
            .map(|guard| guard.value().map(str::to_string));
        let is_member = match membership {
            EntityMembership::Join => true,
            EntityMembership::Leave => false,
            EntityMembership::Keep => old.is_some(),
        };
        if !is_member && old.is_none() {
            return Ok(());
        }
        let new_key = if is_member {
            data.and_then(entity_lookup_key)
        } else {
            None
        };
        if is_member && old.is_some() && old.as_ref().and_then(|k| k.as_deref()) == new_key {
            return Ok(());
        }
        let mut keys = txn.open_multimap_table(ENTITY_KEY_INDEX)?;
        if let Some(old_key) = old.flatten() {
            keys.remove(old_key.as_str(), id)?;
        }
        if is_member {
            members.insert(id, new_key)?;
            if let Some(key) = new_key {
                keys.insert(key, id)?;
            }
        } else {
            members.remove(id)?;
        }
        Ok(())
    }

    /// The marker's value, if any.
    fn entity_key_marker(txn: &ReadTransaction) -> Result<Option<String>> {
        let markers = match txn.open_table(STORAGE_MARKERS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        Ok(markers
            .get(ENTITY_KEY_INDEX_MARKER)?
            .map(|guard| guard.value().to_string()))
    }

    fn entity_key_status_in(&self, txn: &ReadTransaction) -> Result<EntityKeyIndexStatus> {
        let Some(marker) = Self::entity_key_marker(txn)? else {
            return Ok(EntityKeyIndexStatus::Absent);
        };
        let current = self.entity_fingerprint_read(txn)?;
        if let Some(stamped) = marker.strip_prefix(ENTITY_KEY_READY) {
            return Ok(if stamped == current {
                EntityKeyIndexStatus::Current
            } else {
                EntityKeyIndexStatus::Stale
            });
        }
        if let Some((stamped, row)) = marker
            .strip_prefix(ENTITY_KEY_FAILED)
            .and_then(|rest| rest.split_once(':'))
        {
            if stamped == current {
                return Ok(EntityKeyIndexStatus::Failed {
                    row: row.to_string(),
                });
            }
        }
        // A failed build whose list has since changed, or a marker this
        // version does not recognise: worth building again.
        Ok(EntityKeyIndexStatus::Absent)
    }

    /// Where the entity key index stands; see [`EntityKeyIndexStatus`].
    /// Always [`EntityKeyIndexStatus::Absent`] on a handle that does not use
    /// the index (an encryption cipher, or switched off).
    pub fn entity_key_index_status(&self) -> Result<EntityKeyIndexStatus> {
        if !self.entity_key_index_enabled() {
            return Ok(EntityKeyIndexStatus::Absent);
        }
        let txn = self.begin_read()?;
        self.entity_key_status_in(&txn)
    }

    /// True while the entity key index is built and in step with the
    /// `_entities` list, i.e. [`Storage::entity_key_index_status`] is
    /// [`EntityKeyIndexStatus::Current`].
    pub fn entity_key_index_ready(&self) -> Result<bool> {
        Ok(self.entity_key_index_status()? == EntityKeyIndexStatus::Current)
    }

    /// Build the entity key index unless it is already current or its last
    /// build failed on the `_entities` list as it stands. Returns `true` when
    /// this call built it.
    ///
    /// Callers build it where they would otherwise pay a full `_entities`
    /// scan (auto-linking a new record), not on open, so commands that never
    /// resolve entities never pay for, or commit, the build. A no-op on
    /// read-only handles and on handles that do not use the index. See
    /// [`Storage::rebuild_entity_key_index`] for the build itself.
    pub fn ensure_entity_key_index(&self) -> Result<bool> {
        if self.is_read_only() || !self.entity_key_index_enabled() {
            return Ok(false);
        }
        match self.entity_key_index_status()? {
            EntityKeyIndexStatus::Current | EntityKeyIndexStatus::Failed { .. } => Ok(false),
            EntityKeyIndexStatus::Absent | EntityKeyIndexStatus::Stale => {
                self.rebuild_entity_key_index().map(|_| true)
            }
        }
    }

    /// Bring the entity key index in step with the `_entities` rows, whatever
    /// state it is in, and stamp it ready.
    ///
    /// Every listed row is decoded, as the scan the index replaces would, but
    /// only the index entries that differ are written: a first build writes
    /// them all, while repairing what an older binary changed writes about as
    /// much as it changed. Deleting and refilling both tables instead would
    /// allocate the whole index again on every repair, in a transaction that
    /// cannot reuse the pages it frees, so each repair would grow the file.
    /// When nothing differs and the marker is already current, the
    /// transaction is aborted and nothing is written.
    ///
    /// The repair commits as one write transaction, so a crash leaves either
    /// the old state or a complete index. A row whose body does not decode
    /// stops it with that error; a separate transaction then records the row
    /// in the marker ([`EntityKeyIndexStatus::Failed`]), so
    /// [`Storage::ensure_entity_key_index`] does not repeat the build until
    /// the list changes, and lookups keep scanning (which meets the same
    /// row). Other errors are returned without being recorded, so a transient
    /// one is retried. A no-op on a handle that does not use the index; fails
    /// with [`AxilError::Busy`] on a read-only handle.
    ///
    /// This is also the repair for the one change the index cannot notice on
    /// its own: an older binary rewriting an existing `_entities` row's
    /// `canonical_id` or `name` in place leaves the `_entities` list's bytes
    /// unchanged. Lookups catch it when a result points at the rewritten row;
    /// [`Storage::verify_entity_key_index`] catches it everywhere.
    pub fn rebuild_entity_key_index(&self) -> Result<EntityKeyIndexRebuild> {
        if !self.entity_key_index_enabled() {
            return Ok(EntityKeyIndexRebuild::default());
        }
        let txn = self.begin_write()?;
        let synced = match self.sync_entity_key_index_in(&txn) {
            Ok(synced) => synced,
            Err(e) => {
                txn.abort()?;
                return Err(e);
            }
        };
        match synced {
            Ok(rebuild) => {
                if rebuild.written {
                    txn.commit()?;
                } else {
                    txn.abort()?;
                }
                Ok(rebuild)
            }
            Err(row) => {
                txn.abort()?;
                self.record_entity_key_failure(&row.id)?;
                Err(row.error)
            }
        }
    }

    /// Write the index entries that differ from the rows inside `txn` and
    /// stamp the ready marker. Returns what changed; the inner error names a
    /// row whose body does not decode.
    fn sync_entity_key_index_in(
        &self,
        txn: &WriteTransaction,
    ) -> Result<std::result::Result<EntityKeyIndexRebuild, UndecodableEntityRow>> {
        let membership = Membership::open(txn)?;
        let listed = keyed_listed_ids(&membership.members, ENTITIES_TABLE, 0, usize::MAX)?;
        let records = txn.open_table(RECORDS)?;
        let keyed = match self.entity_keys_of_rows(listed, &records)? {
            Ok(keyed) => keyed,
            Err(row) => return Ok(Err(row)),
        };
        let mut members = txn.open_table(ENTITY_KEY_MEMBERS)?;
        let mut keys = txn.open_multimap_table(ENTITY_KEY_INDEX)?;
        let diff = Self::entity_key_diff(&keyed, Some(&members), Some(&keys))?;
        for id in &diff.member_removals {
            members.remove(id.as_str())?;
        }
        for (id, key) in &diff.member_sets {
            members.insert(id.as_str(), key.as_deref())?;
        }
        for (key, id) in &diff.pair_removals {
            keys.remove(key.as_str(), id.as_str())?;
        }
        for (key, id) in &diff.pair_inserts {
            keys.insert(key.as_str(), id.as_str())?;
        }
        let ready = format!("{ENTITY_KEY_READY}{}", membership.entity_fingerprint()?);
        let mut markers = txn.open_table(STORAGE_MARKERS)?;
        let restamp = match markers.get(ENTITY_KEY_INDEX_MARKER)? {
            Some(guard) => guard.value() != ready,
            None => true,
        };
        if restamp {
            markers.insert(ENTITY_KEY_INDEX_MARKER, ready.as_str())?;
        }
        let rebuild = EntityKeyIndexRebuild {
            rows: diff.rows,
            changed: diff.rows_affected(),
            written: restamp || !diff.is_empty(),
        };
        Ok(Ok(rebuild))
    }

    /// Record in the marker that a build stopped on `row`, against the
    /// `_entities` list as it stands.
    fn record_entity_key_failure(&self, row: &str) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let fingerprint = Membership::open(&txn)?.entity_fingerprint()?;
            let value = format!("{ENTITY_KEY_FAILED}{fingerprint}:{row}");
            let mut markers = txn.open_table(STORAGE_MARKERS)?;
            markers.insert(ENTITY_KEY_INDEX_MARKER, value.as_str())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Each distinct id of `listed` (the `_entities` list), with the key its
    /// body carries (`None`: the body has no key, or is gone). The inner
    /// error names the first row whose body does not decode.
    #[allow(clippy::type_complexity)]
    fn entity_keys_of_rows<R>(
        &self,
        listed: Vec<RecordId>,
        records: &R,
    ) -> Result<
        std::result::Result<
            std::collections::HashMap<String, Option<String>>,
            UndecodableEntityRow,
        >,
    >
    where
        R: ReadableTable<&'static str, &'static [u8]>,
    {
        let mut keyed = std::collections::HashMap::with_capacity(listed.len());
        for rid in listed {
            if keyed.contains_key(rid.as_str()) {
                continue;
            }
            let key = match records.get(rid.as_str())? {
                Some(guard) => match self.decode_body(rid.as_str(), guard.value()) {
                    Ok(record) => entity_lookup_key(&record.data).map(str::to_string),
                    Err(error) => return Ok(Err(UndecodableEntityRow { id: rid.0, error })),
                },
                None => None,
            };
            keyed.insert(rid.0, key);
        }
        Ok(Ok(keyed))
    }

    /// Compare the index tables (a missing one counts as empty) with the keys
    /// the listed rows carry.
    fn entity_key_diff<M, K>(
        keyed: &std::collections::HashMap<String, Option<String>>,
        members: Option<&M>,
        keys: Option<&K>,
    ) -> Result<EntityKeyDiff>
    where
        M: ReadableTable<&'static str, Option<&'static str>>,
        K: ReadableMultimapTable<&'static str, &'static str>,
    {
        let mut indexed: std::collections::HashMap<String, Option<String>> =
            std::collections::HashMap::with_capacity(keyed.len());
        if let Some(members) = members {
            for entry in members.iter()? {
                let (id, key) = entry?;
                indexed.insert(id.value().to_string(), key.value().map(str::to_string));
            }
        }
        let mut pairs: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::with_capacity(keyed.len());
        if let Some(keys) = keys {
            for entry in keys.iter()? {
                let (key, ids) = entry?;
                let key = key.value().to_string();
                for id in ids {
                    pairs.insert((key.clone(), id?.value().to_string()));
                }
            }
        }

        let mut diff = EntityKeyDiff {
            rows: keyed.len(),
            ..EntityKeyDiff::default()
        };
        let mut wanted: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::with_capacity(keyed.len());
        for (id, key) in keyed {
            if indexed.get(id) != Some(key) {
                diff.member_sets.push((id.clone(), key.clone()));
            }
            if let Some(key) = key {
                let pair = (key.clone(), id.clone());
                if !pairs.contains(&pair) {
                    diff.pair_inserts.push(pair.clone());
                }
                wanted.insert(pair);
            }
        }
        diff.member_removals = indexed
            .into_keys()
            .filter(|id| !keyed.contains_key(id))
            .collect();
        diff.pair_removals = pairs
            .into_iter()
            .filter(|pair| !wanted.contains(pair))
            .collect();
        // Sorted, so the writes walk each B-tree in key order instead of
        // scattering across it.
        diff.member_sets.sort_unstable();
        diff.member_removals.sort_unstable();
        diff.pair_inserts.sort_unstable();
        diff.pair_removals.sort_unstable();
        Ok(diff)
    }

    /// Discard the entity key index and its marker in one write transaction.
    ///
    /// Until it is built again ([`Storage::ensure_entity_key_index`], which
    /// the next auto-link calls), [`Storage::lookup_entity_keys`] reports it
    /// unavailable and callers scan.
    pub fn drop_entity_key_index(&self) -> Result<()> {
        let txn = self.begin_write()?;
        {
            let mut markers = txn.open_table(STORAGE_MARKERS)?;
            markers.remove(ENTITY_KEY_INDEX_MARKER)?;
        }
        txn.delete_table(ENTITY_KEY_MEMBERS)?;
        txn.delete_multimap_table(ENTITY_KEY_INDEX)?;
        txn.commit()?;
        Ok(())
    }

    /// Check a current entity key index against the `_entities` rows it
    /// covers. Returns `None` when the index is not current (nothing to
    /// verify), else how many `_entities` ids have index entries that
    /// disagree with their rows: a key that differs from the body's, or an
    /// entry for an id the list no longer holds.
    ///
    /// Decodes every listed row, so it costs what the scan the index replaces
    /// costs, and writes nothing. Diagnostics use it to catch in-place key
    /// rewrites by a writer that does not maintain the index, which lookups
    /// notice only when a result points at a rewritten row.
    pub fn verify_entity_key_index(&self) -> Result<Option<usize>> {
        if !self.entity_key_index_enabled() {
            return Ok(None);
        }
        let txn = self.begin_read()?;
        if self.entity_key_status_in(&txn)? != EntityKeyIndexStatus::Current {
            return Ok(None);
        }
        let listed = self.listed_ids(&txn, ENTITIES_TABLE, 0, usize::MAX)?;
        let records = txn.open_table(RECORDS)?;
        let keyed = match self.entity_keys_of_rows(listed, &records)? {
            Ok(keyed) => keyed,
            Err(row) => return Err(row.error),
        };
        let members = match txn.open_table(ENTITY_KEY_MEMBERS) {
            Ok(table) => Some(table),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(e.into()),
        };
        let keys = match txn.open_multimap_table(ENTITY_KEY_INDEX) {
            Ok(table) => Some(table),
            Err(redb::TableError::TableDoesNotExist(_)) => None,
            Err(e) => return Err(e.into()),
        };
        let diff = Self::entity_key_diff(&keyed, members.as_ref(), keys.as_ref())?;
        Ok(Some(diff.rows_affected()))
    }

    /// Resolve entity lookup keys to `_entities` record ids through the index.
    ///
    /// The answer is the one a full scan gives when it walks
    /// `list("_entities")` and files each row under its key (`canonical_id`,
    /// else `name`), later rows overwriting earlier ones: keys with no row are
    /// absent, and a key several rows share maps to the row listed last. The
    /// list positions of those rows are read only when a key is shared.
    ///
    /// Before answering, the index is checked against the rows, so a writer
    /// that does not maintain it (an older binary on the same file) cannot
    /// make it answer wrongly:
    ///
    /// - The `_entities` list must still have the fingerprint the index was
    ///   stamped with. Any insert or delete of an `_entities` row changes it.
    ///   This costs one point read (a read-only handle on a file still in the
    ///   legacy layout reads that list's JSON entry and takes a CRC-32 of it,
    ///   with no JSON decode).
    /// - Every row returned is read back and must still carry the key it was
    ///   found under. This catches an in-place rewrite of that row.
    ///
    /// Either failure returns [`EntityKeyLookup::Stale`]. One change stays
    /// invisible here: an older binary rewriting some *other* row's key in
    /// place to a requested one. The lookup then misses that row, or returns
    /// the row a scan would have ranked below it, until a rebuild;
    /// [`Storage::verify_entity_key_index`] is what finds it.
    ///
    /// Returns [`EntityKeyLookup::Unavailable`] when the index is not built,
    /// its last build failed, or this handle does not use it.
    pub fn lookup_entity_keys(&self, keys: &[&str]) -> Result<EntityKeyLookup> {
        if !self.entity_key_index_enabled() {
            return Ok(EntityKeyLookup::Unavailable);
        }
        let txn = self.begin_read()?;
        match self.entity_key_status_in(&txn)? {
            EntityKeyIndexStatus::Current => {}
            EntityKeyIndexStatus::Stale => return Ok(EntityKeyLookup::Stale),
            EntityKeyIndexStatus::Absent | EntityKeyIndexStatus::Failed { .. } => {
                return Ok(EntityKeyLookup::Unavailable)
            }
        }
        let index = match txn.open_multimap_table(ENTITY_KEY_INDEX) {
            Ok(index) => index,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(EntityKeyLookup::Stale),
            Err(e) => return Err(e.into()),
        };
        let records = txn.open_table(RECORDS)?;
        let mut found = std::collections::HashMap::with_capacity(keys.len());
        let mut positions = EntityPositions::Unread;
        for &key in keys {
            let mut ids = Vec::new();
            for value in index.get(key)? {
                ids.push(RecordId(value?.value().to_string()));
            }
            let winner = match ids.len() {
                0 => continue,
                1 => ids.pop(),
                _ => {
                    let mut ranked = Vec::with_capacity(ids.len());
                    for rid in ids {
                        ranked.push((positions.of(self, &txn, &rid)?, rid));
                    }
                    ranked
                        .into_iter()
                        .max_by_key(|(pos, _)| *pos)
                        .map(|(_, rid)| rid)
                }
            };
            let Some(id) = winner else { continue };
            let still_keyed = match records.get(id.as_str())? {
                Some(guard) => {
                    let record = self.decode_body(id.as_str(), guard.value())?;
                    entity_lookup_key(&record.data) == Some(key)
                }
                None => false,
            };
            if !still_keyed {
                return Ok(EntityKeyLookup::Stale);
            }
            found.insert(key.to_string(), id);
        }
        Ok(EntityKeyLookup::Found(found))
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

    /// [`Storage::events_since`], returning each entry's cursor key alongside
    /// its body — a reader that filters entries still needs the key of the
    /// last one it *scanned* to resume past them.
    #[cfg(feature = "event-log")]
    pub fn events_since_keyed(
        &self,
        cursor: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.begin_read()?;
        let log = txn.open_table(EVENT_LOG)?;
        let lower = match cursor {
            // Exclusive lower bound: skip the cursor key itself.
            Some(c) => std::ops::Bound::Excluded(c),
            None => std::ops::Bound::Unbounded,
        };
        let mut out = Vec::new();
        for entry in log.range::<&str>((lower, std::ops::Bound::Unbounded))? {
            let (key, val): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            out.push((key.value().to_string(), val.value().to_vec()));
            if out.len() >= limit {
                break;
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
}

/// Writes that bypass the entity key index the way a binary that predates it
/// does, so tests can put a store in the state such a binary leaves.
///
/// Such binaries keep the legacy id lists and cannot open a keyed file, so
/// for real their writes land only before a migration (the `legacy_layout`
/// tests cover that). These helpers make the same writes on the keyed layout,
/// mid-session, which exercises the staleness checks against any writer that
/// changes the `_entities` list without maintaining the index.
#[cfg(test)]
impl Storage {
    /// Insert the way a binary that predates the entity key index does:
    /// `records` and table membership only, leaving the index and its marker
    /// untouched.
    pub(crate) fn older_writer_insert(&self, record: &Record) {
        let txn = self.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            let mut membership = Membership::open(&txn).unwrap();
            let old_table = records
                .get(record.id.as_str())
                .unwrap()
                .map(|g| Record::from_bytes(g.value()).unwrap().table);
            if let Some(old_table) = old_table.filter(|t| *t != record.table) {
                membership.leave(&old_table, record.id.as_str()).unwrap();
            }
            let bytes = record.to_bytes().unwrap();
            records
                .insert(record.id.as_str(), bytes.as_slice())
                .unwrap();
            membership.join(&record.table, record.id.as_str()).unwrap();
        }
        txn.commit().unwrap();
    }

    /// Delete the way a binary that predates the entity key index does.
    pub(crate) fn older_writer_delete(&self, id: &RecordId) {
        let txn = self.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            let table = match records.get(id.as_str()).unwrap() {
                Some(g) => Record::from_bytes(g.value()).unwrap().table,
                None => return,
            };
            records.remove(id.as_str()).unwrap();
            Membership::open(&txn)
                .unwrap()
                .leave(&table, id.as_str())
                .unwrap();
        }
        txn.commit().unwrap();
    }

    /// Update in place the way a binary that predates the entity key index
    /// does (the SCIP canonical_id rewrite, `merge_entities`).
    pub(crate) fn older_writer_update(&self, id: &RecordId, data: serde_json::Value) {
        let txn = self.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            let mut record =
                Record::from_bytes(records.get(id.as_str()).unwrap().unwrap().value()).unwrap();
            record.data = data;
            let bytes = record.to_bytes().unwrap();
            records.insert(id.as_str(), bytes.as_slice()).unwrap();
        }
        txn.commit().unwrap();
    }

    /// Overwrite a row's stored body with raw bytes, as corruption or a
    /// foreign writer might, leaving every list and index alone.
    pub(crate) fn older_writer_put_raw(&self, id: &RecordId, bytes: &[u8]) {
        let txn = self.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            records.insert(id.as_str(), bytes).unwrap();
        }
        txn.commit().unwrap();
    }

    /// Drop an id from the `_entities` list and `records` without decoding
    /// its body (which may not decode).
    pub(crate) fn older_writer_delete_raw(&self, id: &RecordId) {
        let txn = self.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            records.remove(id.as_str()).unwrap();
            Membership::open(&txn)
                .unwrap()
                .leave(ENTITIES_TABLE, id.as_str())
                .unwrap();
        }
        txn.commit().unwrap();
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
    fn a_vector_note_rides_with_its_delete_until_cleared() {
        let (storage, _dir) = temp_storage();
        assert!(storage.pending_vector_deletes().unwrap().is_empty());

        let noted = storage
            .insert(&Record::new("sessions", json!({"x": 1})))
            .unwrap();
        let plain = storage
            .insert(&Record::new("sessions", json!({"x": 2})))
            .unwrap();
        assert!(storage.delete_noting_vector(&noted, true).unwrap());
        assert!(storage.delete_noting_vector(&plain, false).unwrap());
        // No record, no note: there was nothing to delete.
        assert!(!storage
            .delete_noting_vector(&RecordId::new(), true)
            .unwrap());
        assert_eq!(
            storage.pending_vector_deletes().unwrap(),
            vec![noted.clone()]
        );

        storage.clear_pending_vector_deletes(&[noted]).unwrap();
        assert!(storage.pending_vector_deletes().unwrap().is_empty());
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
    fn list_newest_returns_the_tail_newest_first() {
        let (storage, _dir) = temp_storage();
        for i in 0..5 {
            storage
                .insert(&Record::new("items", json!({"i": i})))
                .unwrap();
        }
        let newest: Vec<i64> = storage
            .list_newest("items", 3)
            .unwrap()
            .iter()
            .map(|r| r.data["i"].as_i64().unwrap())
            .collect();
        assert_eq!(newest, vec![4, 3, 2]);
        assert_eq!(storage.list_newest("items", 10).unwrap().len(), 5);
        assert!(storage.list_newest("nothing", 3).unwrap().is_empty());
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

        /// The entity key index would hold entity names in cleartext, so an
        /// encrypted handle neither uses nor maintains it. Its first write
        /// discards the whole index, so names a cleartext store indexed do
        /// not outlive a re-seal made under the cipher, and a later cleartext
        /// handle cannot trust an index that missed that write.
        #[test]
        fn entity_key_index_is_discarded_under_a_cipher() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("enc.axil");
            let secret = {
                let storage = Storage::open(&path).unwrap();
                storage.ensure_entity_key_index().unwrap();
                storage
                    .insert(&Record::new("_entities", json!({"canonical_id": "redis"})))
                    .unwrap();
                let secret = storage
                    .insert(&Record::new(
                        "_entities",
                        json!({"canonical_id": "acme-merger-codename"}),
                    ))
                    .unwrap();
                assert!(storage.entity_key_index_ready().unwrap());
                secret
            };
            {
                let storage = Storage::open(&path).unwrap().with_cipher(key_a());
                assert!(!storage.ensure_entity_key_index().unwrap());
                assert_eq!(
                    storage.lookup_entity_keys(&["redis"]).unwrap(),
                    EntityKeyLookup::Unavailable
                );
                // Re-seal the sensitive row: an upsert under the cipher (a
                // cleartext body cannot be decrypted, so update or delete
                // would refuse it).
                let mut resealed = Record::new("_entities", json!({"canonical_id": "redacted"}));
                resealed.id = secret.clone();
                storage.insert(&resealed).unwrap();
                let txn = storage.begin_read().unwrap();
                assert!(matches!(
                    txn.open_table(ENTITY_KEY_MEMBERS),
                    Err(redb::TableError::TableDoesNotExist(_))
                ));
                assert!(matches!(
                    txn.open_multimap_table(ENTITY_KEY_INDEX),
                    Err(redb::TableError::TableDoesNotExist(_))
                ));
                assert!(Storage::entity_key_marker(&txn).unwrap().is_none());
                storage
                    .insert(&Record::new("_entities", json!({"canonical_id": "auth"})))
                    .unwrap();
            }
            let storage = Storage::open(&path).unwrap();
            assert_eq!(
                storage.entity_key_index_status().unwrap(),
                EntityKeyIndexStatus::Absent
            );
            // The sealed body can't be indexed without the key: the build
            // fails, lookups stay on the scan path, and the failure is
            // remembered so the next call does not decode the rows again.
            assert!(storage.ensure_entity_key_index().is_err());
            assert_eq!(
                storage.entity_key_index_status().unwrap(),
                EntityKeyIndexStatus::Failed {
                    row: secret.as_str().to_string()
                },
                "the first sealed row in list order stopped the build"
            );
            assert!(!storage.ensure_entity_key_index().unwrap());
            assert_eq!(
                storage.lookup_entity_keys(&["redis"]).unwrap(),
                EntityKeyLookup::Unavailable
            );
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

    // ── entity key index ───────────────────────────────────────────────

    /// The map auto-linking has always built: every row of
    /// `list("_entities")` filed under `canonical_id`, else `name`, later rows
    /// overwriting earlier ones. Written out independently of the index code
    /// so the parity checks compare against the original semantics.
    fn scan_oracle(storage: &Storage) -> std::collections::HashMap<String, RecordId> {
        storage
            .list("_entities", usize::MAX, 0)
            .unwrap()
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
            .collect()
    }

    /// Canonical ids and names share one small pool so rows collide on keys
    /// both through `canonical_id` and through the `name` fallback.
    const KEY_POOL: [&str; 7] = ["redis", "auth", "login", "pool", "onnx", "Redis", ""];

    fn assert_index_matches_scan(storage: &Storage, context: &str) {
        let oracle = scan_oracle(storage);
        let mut keys: Vec<String> = KEY_POOL.iter().map(|k| k.to_string()).collect();
        keys.extend(oracle.keys().cloned());
        keys.push("never-stored".to_string());
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let indexed = match storage.lookup_entity_keys(&key_refs).unwrap() {
            EntityKeyLookup::Found(found) => found,
            other => panic!("{context}: index not usable: {other:?}"),
        };
        assert_eq!(
            indexed, oracle,
            "{context}: index lookups differ from a full scan"
        );
        assert_eq!(
            storage.verify_entity_key_index().unwrap(),
            Some(0),
            "{context}: verification disagrees with a matching index"
        );
    }

    /// Tiny deterministic PRNG so a failing seed replays exactly.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
            &items[self.below(items.len())]
        }
    }

    /// A random `_entities`-style body: canonical_id + name, name only,
    /// canonical_id only, a non-string canonical_id (falls back to name), or
    /// neither field.
    fn random_entity_body(rng: &mut XorShift) -> serde_json::Value {
        let key = *rng.pick(&KEY_POOL);
        let name = *rng.pick(&KEY_POOL);
        match rng.below(6) {
            0 | 1 => json!({"canonical_id": key, "name": name, "salt": rng.next()}),
            2 => json!({"name": name, "salt": rng.next()}),
            3 => json!({ "canonical_id": key }),
            4 => json!({"canonical_id": 7, "name": name}),
            _ => json!({"entity": name, "fact": "no key fields"}),
        }
    }

    /// A fresh ULID most of the time, otherwise a foreign id whose sort
    /// order has nothing to do with insertion order (as a restore or import
    /// that preserves ids produces).
    fn random_new_id(rng: &mut XorShift, counter: &mut u64) -> RecordId {
        *counter += 1;
        match rng.below(4) {
            0 => RecordId(format!("0000-foreign-{:06}", rng.below(1_000_000))),
            1 => RecordId(format!("zzzz-foreign-{counter}")),
            _ => RecordId::new(),
        }
    }

    fn record_with(table: &str, id: RecordId, data: serde_json::Value) -> Record {
        let mut record = Record::new(table, data);
        record.id = id;
        record
    }

    fn open_indexed(path: &Path) -> Storage {
        let storage = Storage::open(path).unwrap();
        storage.ensure_entity_key_index().unwrap();
        storage
    }

    /// The property that matters while the index may be stale: whenever it
    /// answers at all, it answers what a full scan would.
    fn assert_index_never_wrong(storage: &Storage, context: &str) {
        let oracle = scan_oracle(storage);
        let mut keys: Vec<String> = KEY_POOL.iter().map(|k| k.to_string()).collect();
        keys.extend(oracle.keys().cloned());
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        if let EntityKeyLookup::Found(found) = storage.lookup_entity_keys(&key_refs).unwrap() {
            assert_eq!(found, oracle, "{context}: a stale index answered wrongly");
        }
    }

    /// Random writes, some of them by a writer that does not maintain the
    /// index. While the index is trusted every lookup must match a full scan;
    /// after an older writer's insert or delete it must either still match or
    /// decline to answer, until it is rebuilt.
    #[test]
    fn entity_key_index_matches_full_scan_under_random_writes() {
        const TABLES: [&str; 3] = ["_entities", "_entities", "other"];
        for seed in 1..=8u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("parity.axil");
            let mut storage = open_indexed(&path);
            let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut counter = 0u64;
            // Every id ever written, deleted or not, so later steps also hit
            // missing and cross-table ids.
            let mut ids: Vec<RecordId> = Vec::new();
            // False from an older writer's write until the next rebuild.
            let mut trusted = true;

            for step in 0..160 {
                let what = match rng.below(13) {
                    // New `_entities` row (possibly a foreign id, possibly a
                    // key another row already has).
                    0..=2 => {
                        let id = random_new_id(&mut rng, &mut counter);
                        let body = random_entity_body(&mut rng);
                        storage
                            .insert(&record_with("_entities", id.clone(), body))
                            .unwrap();
                        ids.push(id);
                        "insert"
                    }
                    // Upsert an existing id, possibly moving it between tables.
                    3 if !ids.is_empty() => {
                        let id = rng.pick(&ids).clone();
                        let table = *rng.pick(&TABLES);
                        let body = random_entity_body(&mut rng);
                        storage.insert(&record_with(table, id, body)).unwrap();
                        "upsert"
                    }
                    // Batch insert: fresh ids, reused ids (even from another
                    // table), and the same id twice in one batch.
                    4 => {
                        let mut batch: Vec<Record> = Vec::new();
                        for _ in 0..(1 + rng.below(4)) {
                            let id = if !ids.is_empty() && rng.below(3) == 0 {
                                rng.pick(&ids).clone()
                            } else if !batch.is_empty() && rng.below(4) == 0 {
                                rng.pick(&batch).id.clone()
                            } else {
                                random_new_id(&mut rng, &mut counter)
                            };
                            let table = *rng.pick(&TABLES);
                            let body = random_entity_body(&mut rng);
                            batch.push(record_with(table, id.clone(), body));
                            ids.push(id);
                        }
                        storage.insert_batch(&batch).unwrap();
                        "insert_batch"
                    }
                    // Update, including rewriting or dropping canonical_id.
                    5 | 6 if !ids.is_empty() => {
                        let id = rng.pick(&ids).clone();
                        let body = random_entity_body(&mut rng);
                        match storage.update(&id, body) {
                            Ok(_) | Err(AxilError::NotFound(_)) => {}
                            Err(e) => panic!("update: {e}"),
                        }
                        "update"
                    }
                    7 if !ids.is_empty() => {
                        let id = rng.pick(&ids).clone();
                        storage.delete(&id).unwrap();
                        "delete"
                    }
                    8 if rng.below(2) == 0 => {
                        // Plain reopen, via a read-only handle in between: the
                        // index must survive as is.
                        drop(storage);
                        let ro = Storage::open_read_only(&path).unwrap();
                        let context = format!("seed {seed} step {step} read-only");
                        if trusted {
                            assert_index_matches_scan(&ro, &context);
                        } else {
                            assert_index_never_wrong(&ro, &context);
                        }
                        drop(ro);
                        storage = open_indexed(&path);
                        // `ensure` rebuilds only what the marker shows to be
                        // stale; an older writer's delete of a row listed
                        // under two tables leaves the list's bytes alone.
                        if !trusted {
                            storage.rebuild_entity_key_index().unwrap();
                        }
                        trusted = true;
                        "reopen"
                    }
                    8 => {
                        // Lose the index, keep writing, then rebuild it on the
                        // next open.
                        storage.drop_entity_key_index().unwrap();
                        assert_eq!(
                            storage.lookup_entity_keys(&["redis"]).unwrap(),
                            EntityKeyLookup::Unavailable
                        );
                        let id = random_new_id(&mut rng, &mut counter);
                        let body = random_entity_body(&mut rng);
                        storage
                            .insert(&record_with("_entities", id.clone(), body))
                            .unwrap();
                        ids.push(id);
                        drop(storage);
                        storage = open_indexed(&path);
                        trusted = true;
                        "rebuild"
                    }
                    // An older binary inserts a fresh row or deletes any row.
                    10 => {
                        let id = RecordId::new();
                        let table = *rng.pick(&TABLES);
                        let body = random_entity_body(&mut rng);
                        storage.older_writer_insert(&record_with(table, id.clone(), body));
                        ids.push(id);
                        trusted = false;
                        "older insert"
                    }
                    11 if !ids.is_empty() => {
                        let id = rng.pick(&ids).clone();
                        storage.older_writer_delete(&id);
                        trusted = false;
                        "older delete"
                    }
                    // Repair the way auto-linking does once it sees the index
                    // decline: rebuild it.
                    12 if !trusted => {
                        storage.rebuild_entity_key_index().unwrap();
                        trusted = true;
                        "repair"
                    }
                    _ => {
                        let id = random_new_id(&mut rng, &mut counter);
                        storage
                            .insert(&record_with("other", id.clone(), json!({"name": "redis"})))
                            .unwrap();
                        ids.push(id);
                        "insert other"
                    }
                };
                let context = format!("seed {seed} step {step} ({what})");
                if trusted {
                    assert_index_matches_scan(&storage, &context);
                } else {
                    assert_index_never_wrong(&storage, &context);
                }
            }
        }
    }

    #[test]
    fn entity_key_index_is_built_for_a_store_without_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.axil");
        {
            let storage = Storage::open(&path).unwrap();
            let first = record_with(
                "_entities",
                RecordId("zzzz-imported".into()),
                json!({"canonical_id": "redis", "tag": 1}),
            );
            storage.insert(&first).unwrap();
            storage
                .insert(&Record::new(
                    "_entities",
                    json!({"canonical_id": "redis", "tag": 2}),
                ))
                .unwrap();
            storage
                .insert(&Record::new("_entities", json!({"name": "auth"})))
                .unwrap();
            storage
                .insert(&Record::new("_entities", json!({"entity": "no key"})))
                .unwrap();
            // A store written before the index existed has no index tables
            // and no marker.
            storage.drop_entity_key_index().unwrap();
            assert!(!storage.entity_key_index_ready().unwrap());
            assert_eq!(
                storage.lookup_entity_keys(&["redis"]).unwrap(),
                EntityKeyLookup::Unavailable
            );
        }

        // Read-only handles never build it; they keep reporting "scan".
        {
            let ro = Storage::open_read_only(&path).unwrap();
            assert!(!ro.ensure_entity_key_index().unwrap());
            assert_eq!(
                ro.lookup_entity_keys(&["redis"]).unwrap(),
                EntityKeyLookup::Unavailable
            );
        }

        let storage = Storage::open(&path).unwrap();
        assert!(
            !storage.entity_key_index_ready().unwrap(),
            "opening does not build"
        );
        assert!(
            storage.ensure_entity_key_index().unwrap(),
            "the first ensure builds"
        );
        assert!(
            !storage.ensure_entity_key_index().unwrap(),
            "second call is a no-op"
        );
        assert!(storage.entity_key_index_ready().unwrap());
        assert_index_matches_scan(&storage, "after migration");
        // The later of the two `redis` rows wins even though the earlier one
        // has the larger id.
        let found = storage
            .lookup_entity_keys(&["redis"])
            .unwrap()
            .into_found()
            .unwrap();
        let winner = storage.get(&found["redis"]).unwrap().unwrap();
        assert_eq!(winner.data["tag"], 2);
    }

    #[test]
    fn entity_key_index_ignores_rows_of_other_tables() {
        let (storage, _dir) = temp_storage();
        storage.ensure_entity_key_index().unwrap();
        let id = storage
            .insert(&Record::new("sessions", json!({"name": "redis"})))
            .unwrap();
        storage
            .update(&id, json!({"canonical_id": "redis"}))
            .unwrap();
        assert_eq!(
            storage.lookup_entity_keys(&["redis"]).unwrap(),
            EntityKeyLookup::Found(Default::default())
        );
        let txn = storage.begin_read().unwrap();
        if let Ok(members) = txn.open_table(ENTITY_KEY_MEMBERS) {
            assert!(members.get(id.as_str()).unwrap().is_none());
        }
    }

    /// Two entity rows under different keys plus a built index, for the
    /// older-writer scenarios below.
    fn indexed_pair() -> (Storage, tempfile::TempDir, RecordId, RecordId) {
        let (storage, dir) = temp_storage();
        let redis = storage
            .insert(&Record::new("_entities", json!({"canonical_id": "redis"})))
            .unwrap();
        let kafka = storage
            .insert(&Record::new("_entities", json!({"canonical_id": "kafka"})))
            .unwrap();
        assert!(storage.ensure_entity_key_index().unwrap());
        assert_index_matches_scan(&storage, "before the older writer");
        (storage, dir, redis, kafka)
    }

    /// An older binary adds an entity the index has never seen. Trusting the
    /// index would report the key absent, and auto-linking would create a
    /// duplicate entity.
    #[test]
    fn entity_key_index_declines_after_an_older_writer_inserts() {
        let (storage, _dir, _redis, _kafka) = indexed_pair();
        let broker = Record::new("_entities", json!({"canonical_id": "kafka_broker"}));
        storage.older_writer_insert(&broker);

        assert_eq!(
            storage.lookup_entity_keys(&["kafka_broker"]).unwrap(),
            EntityKeyLookup::Stale
        );
        assert_eq!(
            storage.entity_key_index_status().unwrap(),
            EntityKeyIndexStatus::Stale
        );
        assert!(storage.ensure_entity_key_index().unwrap(), "stale rebuilds");
        let found = storage
            .lookup_entity_keys(&["kafka_broker"])
            .unwrap()
            .into_found()
            .unwrap();
        assert_eq!(found["kafka_broker"], broker.id);
        assert_index_matches_scan(&storage, "after the rebuild");
    }

    /// An older binary deletes an entity. Trusting the index would link new
    /// mentions to the deleted id.
    #[test]
    fn entity_key_index_declines_after_an_older_writer_deletes() {
        let (storage, _dir, _redis, kafka) = indexed_pair();
        storage.older_writer_delete(&kafka);
        assert_eq!(
            storage.lookup_entity_keys(&["kafka"]).unwrap(),
            EntityKeyLookup::Stale
        );
        storage.rebuild_entity_key_index().unwrap();
        assert_index_matches_scan(&storage, "after the rebuild");
    }

    /// A maintained write after an older writer's must not re-stamp the
    /// marker over the gap: the index stays untrusted until it is rebuilt,
    /// even when the list later returns to bytes it once had.
    #[test]
    fn entity_key_index_stays_untrusted_after_later_maintained_writes() {
        let (storage, _dir, redis, _kafka) = indexed_pair();
        let broker = Record::new("_entities", json!({"canonical_id": "kafka_broker"}));
        storage.older_writer_insert(&broker);
        // Leaves the list exactly as the marker last saw it, but the index
        // never learned about `redis` being rewritten below.
        storage.delete(&broker.id).unwrap();
        storage.older_writer_update(&redis, json!({"canonical_id": "valkey"}));
        storage
            .insert(&Record::new("_entities", json!({"canonical_id": "pool"})))
            .unwrap();
        assert_eq!(
            storage.lookup_entity_keys(&["valkey"]).unwrap(),
            EntityKeyLookup::Unavailable
        );
        assert!(storage.ensure_entity_key_index().unwrap());
        assert_index_matches_scan(&storage, "after the rebuild");
    }

    /// An older binary rewrites the key of a row the index would return
    /// (SCIP's provisional -> canonical upgrade, `merge_entities`). The list
    /// is unchanged, so only reading the row back catches it.
    #[test]
    fn entity_key_index_declines_when_a_returned_row_was_rewritten() {
        let (storage, _dir, redis, _kafka) = indexed_pair();
        storage.older_writer_update(&redis, json!({"canonical_id": "valkey"}));
        assert_eq!(
            storage.lookup_entity_keys(&["redis"]).unwrap(),
            EntityKeyLookup::Stale
        );
        assert_eq!(storage.verify_entity_key_index().unwrap(), Some(1));
        storage.rebuild_entity_key_index().unwrap();
        assert_index_matches_scan(&storage, "after the rebuild");
    }

    /// The one older-writer change a lookup cannot see: a row rewritten in
    /// place *to* a requested key. Verification finds it, and a rebuild
    /// repairs it.
    #[test]
    fn entity_key_index_verification_finds_a_key_rewritten_onto_a_row() {
        let (storage, _dir, _redis, kafka) = indexed_pair();
        storage.older_writer_update(&kafka, json!({"canonical_id": "rabbitmq"}));
        assert_eq!(
            storage.lookup_entity_keys(&["rabbitmq"]).unwrap(),
            EntityKeyLookup::Found(Default::default()),
            "documented blind spot: the index has not seen the new key"
        );
        assert_eq!(storage.verify_entity_key_index().unwrap(), Some(1));
        storage.rebuild_entity_key_index().unwrap();
        assert_index_matches_scan(&storage, "after the rebuild");
    }

    /// A repair writes only the entries that disagree with the rows, and an
    /// index already in step commits nothing: refilling both tables on every
    /// repair is what grew the file each time an older binary wrote.
    #[test]
    fn entity_key_index_repair_writes_only_what_changed() {
        let (storage, _dir) = temp_storage();
        for i in 0..40 {
            storage
                .insert(&Record::new(
                    "_entities",
                    json!({"canonical_id": format!("entity_{i}")}),
                ))
                .unwrap();
        }
        let first = storage.rebuild_entity_key_index().unwrap();
        assert_eq!(
            first,
            EntityKeyIndexRebuild {
                rows: 40,
                changed: 40,
                written: true
            }
        );
        let again = storage.rebuild_entity_key_index().unwrap();
        assert_eq!(
            again,
            EntityKeyIndexRebuild {
                rows: 40,
                changed: 0,
                written: false
            },
            "an index in step is left alone"
        );

        let added = Record::new("_entities", json!({"canonical_id": "kafka_broker"}));
        storage.older_writer_insert(&added);
        let rewritten = storage.list("_entities", 1, 3).unwrap().remove(0);
        storage.older_writer_update(&rewritten.id, json!({"canonical_id": "valkey"}));
        let deleted = storage.list("_entities", 1, 7).unwrap().remove(0);
        storage.older_writer_delete(&deleted.id);
        assert_eq!(
            storage.entity_key_index_status().unwrap(),
            EntityKeyIndexStatus::Stale
        );
        let repair = storage.rebuild_entity_key_index().unwrap();
        assert_eq!(
            repair,
            EntityKeyIndexRebuild {
                rows: 40,
                changed: 3,
                written: true
            },
            "one added, one rewritten, one deleted"
        );
        assert_index_matches_scan(&storage, "after the repair");

        // An id an older binary added and dropped again costs the repair
        // nothing; only the row it kept is indexed.
        let transient = Record::new("_entities", json!({"entity": "no key"}));
        storage.older_writer_insert(&transient);
        storage.older_writer_insert(&Record::new("_entities", json!({"name": "pool"})));
        storage.older_writer_delete(&transient.id);
        let stamped = storage.rebuild_entity_key_index().unwrap();
        assert!(stamped.written && stamped.changed == 1);
        assert!(storage.entity_key_index_ready().unwrap());
    }

    /// A build that fails is remembered against the list it failed on: the
    /// next `ensure` skips it, and a change to the list makes it try again.
    #[test]
    fn entity_key_index_remembers_a_failed_build_until_the_list_changes() {
        let (storage, _dir) = temp_storage();
        let good = storage
            .insert(&Record::new("_entities", json!({"canonical_id": "redis"})))
            .unwrap();
        let bad = Record::new("_entities", json!({"canonical_id": "broken"}));
        storage.older_writer_insert(&bad);
        storage.older_writer_put_raw(&bad.id, b"not a record");

        assert!(storage.ensure_entity_key_index().is_err());
        assert_eq!(
            storage.entity_key_index_status().unwrap(),
            EntityKeyIndexStatus::Failed {
                row: bad.id.as_str().to_string()
            }
        );
        assert!(!storage.ensure_entity_key_index().unwrap(), "not retried");
        assert_eq!(
            storage.lookup_entity_keys(&["redis"]).unwrap(),
            EntityKeyLookup::Unavailable
        );

        storage.older_writer_delete_raw(&bad.id);
        assert_eq!(
            storage.entity_key_index_status().unwrap(),
            EntityKeyIndexStatus::Absent,
            "the list changed, so the failure no longer applies"
        );
        assert!(storage.ensure_entity_key_index().unwrap());
        let found = storage
            .lookup_entity_keys(&["redis"])
            .unwrap()
            .into_found()
            .unwrap();
        assert_eq!(found["redis"], good);
    }

    /// A handle switched off neither answers nor maintains, and its writes
    /// retire the index so a later handle cannot trust what it missed.
    #[test]
    fn entity_key_index_switched_off_discards_on_write() {
        let (storage, _dir, _redis, _kafka) = indexed_pair();
        storage.set_entity_key_index_enabled(false);
        assert_eq!(
            storage.lookup_entity_keys(&["redis"]).unwrap(),
            EntityKeyLookup::Unavailable
        );
        assert!(!storage.ensure_entity_key_index().unwrap());
        storage
            .insert(&Record::new("_entities", json!({"canonical_id": "pool"})))
            .unwrap();
        storage.set_entity_key_index_enabled(true);
        assert_eq!(
            storage.entity_key_index_status().unwrap(),
            EntityKeyIndexStatus::Absent
        );
        assert!(storage.ensure_entity_key_index().unwrap());
        assert_index_matches_scan(&storage, "after switching back on");
    }

    // ── table index layouts ────────────────────────────────────────────

    /// Files in the legacy table-index layout, as binaries before the keyed
    /// one leave them, and what this version does with them.
    mod legacy_layout {
        use super::*;
        use std::collections::{BTreeMap, HashMap};

        /// Write a file the way a binary with the legacy table index leaves
        /// it: `records` plus one JSON id list per table. The lists are raw,
        /// so one can repeat an id or name an id whose record is gone.
        fn write_legacy(path: &Path, rows: &[Record], lists: &[(&str, Vec<&str>)]) {
            let db = Database::create(path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut records = txn.open_table(RECORDS).unwrap();
                for row in rows {
                    let bytes = row.to_bytes().unwrap();
                    records.insert(row.id.as_str(), bytes.as_slice()).unwrap();
                }
                let mut index = txn.open_table(LEGACY_TABLE_INDEX).unwrap();
                for (table, ids) in lists {
                    let bytes = serde_json::to_vec(ids).unwrap();
                    index.insert(*table, bytes.as_slice()).unwrap();
                }
            }
            txn.commit().unwrap();
        }

        /// Overwrite one table's legacy entry with raw bytes.
        fn put_legacy_entry(path: &Path, table: &str, bytes: &[u8]) {
            let db = Database::create(path).unwrap();
            let txn = db.begin_write().unwrap();
            txn.open_table(LEGACY_TABLE_INDEX)
                .unwrap()
                .insert(table, bytes)
                .unwrap();
            txn.commit().unwrap();
        }

        /// The legacy id lists in a file, or `None` once it is keyed.
        fn legacy_lists(path: &Path) -> Option<BTreeMap<String, Vec<String>>> {
            let db = Database::create(path).unwrap();
            let txn = db.begin_read().unwrap();
            let index = txn.open_table(LEGACY_TABLE_INDEX).ok()?;
            let mut lists = BTreeMap::new();
            for entry in index.iter().unwrap() {
                let (table, bytes) = entry.unwrap();
                lists.insert(
                    table.value().to_string(),
                    serde_json::from_slice(bytes.value()).unwrap(),
                );
            }
            Some(lists)
        }

        /// The first thing an older binary's open does with the table index:
        /// open `table_index` as `&str → &[u8]`.
        fn older_binary_open(path: &Path) -> std::result::Result<(), redb::TableError> {
            let db = Database::create(path).unwrap();
            let txn = db.begin_read().unwrap();
            txn.open_table(LEGACY_TABLE_INDEX).map(|_| ())
        }

        /// An older binary's insert of a fresh record: the record, and its id
        /// appended to its table's JSON list.
        fn older_binary_insert(path: &Path, record: &Record) {
            let db = Database::create(path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let bytes = record.to_bytes().unwrap();
                txn.open_table(RECORDS)
                    .unwrap()
                    .insert(record.id.as_str(), bytes.as_slice())
                    .unwrap();
                let mut index = txn.open_table(LEGACY_TABLE_INDEX).unwrap();
                let mut ids = read_legacy_index(&index, &record.table).unwrap();
                ids.push(record.id.clone());
                let bytes = serde_json::to_vec(&ids).unwrap();
                index
                    .insert(record.table.as_str(), bytes.as_slice())
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        /// An older binary's delete: the record, and its id dropped from its
        /// table's JSON list (the entry goes when the list empties).
        fn older_binary_delete(path: &Path, id: &RecordId) {
            let db = Database::create(path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut records = txn.open_table(RECORDS).unwrap();
                let table = Record::from_bytes(records.get(id.as_str()).unwrap().unwrap().value())
                    .unwrap()
                    .table;
                records.remove(id.as_str()).unwrap();
                let mut index = txn.open_table(LEGACY_TABLE_INDEX).unwrap();
                let mut ids = read_legacy_index(&index, &table).unwrap();
                ids.retain(|rid| rid != id);
                if ids.is_empty() {
                    index.remove(table.as_str()).unwrap();
                } else {
                    let bytes = serde_json::to_vec(&ids).unwrap();
                    index.insert(table.as_str(), bytes.as_slice()).unwrap();
                }
            }
            txn.commit().unwrap();
        }

        fn ids_of(records: &[Record]) -> Vec<String> {
            records.iter().map(|r| r.id.as_str().to_string()).collect()
        }

        fn row(table: &str, id: &str) -> Record {
            record_with(table, RecordId(id.to_string()), json!({ "id": id }))
        }

        /// Ids sorted against their list order, a repeated id, and an id
        /// whose record is gone: the shapes a legacy list can hold.
        fn sample_legacy(path: &Path) {
            let rows = [
                row("notes", "zzzz-first"),
                row("notes", "0000-second"),
                row("notes", "mmmm-third"),
                row("notes", "aaaa-fourth"),
                row("decisions", "only-decision"),
            ];
            write_legacy(
                path,
                &rows,
                &[
                    (
                        "notes",
                        vec![
                            "zzzz-first",
                            "0000-second",
                            "mmmm-third",
                            "0000-second",
                            "gone-record",
                            "aaaa-fourth",
                        ],
                    ),
                    ("decisions", vec!["only-decision"]),
                ],
            );
        }

        /// A read-only handle reads a file nothing of this version has opened
        /// writable from its JSON lists, exactly as the legacy code did, and
        /// leaves the file byte for byte as it was.
        #[test]
        fn a_read_only_open_reads_a_legacy_file_and_writes_nothing() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("legacy.axil");
            sample_legacy(&path);
            let before = std::fs::read(&path).unwrap();

            let ro = Storage::open_read_only(&path).unwrap();
            assert_eq!(
                ids_of(&ro.list("notes", usize::MAX, 0).unwrap()),
                [
                    "zzzz-first",
                    "0000-second",
                    "mmmm-third",
                    "0000-second",
                    "aaaa-fourth"
                ],
                "the legacy list as it stands, repeat included"
            );
            assert_eq!(
                ids_of(&ro.list("notes", 2, 3).unwrap()),
                ["0000-second"],
                "offset and limit count listed ids, the gone one included"
            );
            assert_eq!(
                ids_of(&ro.list_newest("notes", 3).unwrap()),
                ["aaaa-fourth", "0000-second", "mmmm-third"]
            );
            assert_eq!(ro.count("notes").unwrap(), 6);
            assert_eq!(ro.tables().unwrap(), ["decisions", "notes"]);
            assert_eq!(
                ro.tables_with_counts().unwrap(),
                [("decisions".to_string(), 1), ("notes".to_string(), 6)]
            );
            assert!(matches!(
                ro.insert(&row("notes", "new")),
                Err(AxilError::Busy)
            ));
            drop(ro);
            assert!(Storage::table_index_conversion_pending(&path).unwrap());

            assert_eq!(std::fs::read(&path).unwrap(), before, "nothing written");
            assert!(legacy_lists(&path).is_some(), "still legacy");
        }

        /// A writable open migrates the lists in order, each id at its first
        /// place, and later writes keep the legacy semantics: a new id joins
        /// last, a re-inserted id keeps its place, an id moved from another
        /// table joins last.
        #[test]
        fn migration_keeps_list_order_and_later_writes_keep_its_rules() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("legacy.axil");
            sample_legacy(&path);

            drop(Storage::open(&path).unwrap());
            assert!(legacy_lists(&path).is_none(), "migrated");
            assert!(!Storage::table_index_conversion_pending(&path).unwrap());
            let storage = Storage::open(&path).unwrap();
            assert!(
                matches!(
                    Storage::table_index_conversion_pending(&path),
                    Err(AxilError::Busy)
                ),
                "a file a writer holds is not read"
            );
            assert_eq!(
                ids_of(&storage.list("notes", usize::MAX, 0).unwrap()),
                ["zzzz-first", "0000-second", "mmmm-third", "aaaa-fourth"]
            );
            assert_eq!(
                ids_of(&storage.list("notes", 2, 3).unwrap()),
                ["aaaa-fourth"],
                "the gone id keeps its slot"
            );
            assert_eq!(storage.count("notes").unwrap(), 5);
            assert_eq!(
                ids_of(&storage.list_newest("notes", 2).unwrap()),
                ["aaaa-fourth", "mmmm-third"]
            );

            storage.insert(&row("notes", "0001-fifth")).unwrap();
            storage.insert(&row("notes", "zzzz-first")).unwrap();
            storage.insert(&row("notes", "only-decision")).unwrap();
            storage.delete(&RecordId("mmmm-third".to_string())).unwrap();
            assert_eq!(
                ids_of(&storage.list("notes", usize::MAX, 0).unwrap()),
                [
                    "zzzz-first",
                    "0000-second",
                    "aaaa-fourth",
                    "0001-fifth",
                    "only-decision"
                ]
            );
            assert_eq!(
                ids_of(&storage.list_newest("notes", 2).unwrap()),
                ["only-decision", "0001-fifth"]
            );
            assert_eq!(storage.tables().unwrap(), ["notes"], "decisions emptied");
            assert_eq!(storage.count("decisions").unwrap(), 0);
        }

        /// The migration is one write transaction: a list that does not
        /// decode, met after earlier tables were already written, aborts it
        /// and leaves the legacy file exactly as it was. Once the entry is
        /// fixed, the next open migrates, and opening again changes nothing.
        #[test]
        fn a_failed_migration_leaves_the_file_as_it_was_and_the_next_open_finishes() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("legacy.axil");
            sample_legacy(&path);
            put_legacy_entry(&path, "zz_last", b"[\"not closed\"");
            let lists_before = {
                let db = Database::create(&path).unwrap();
                let txn = db.begin_read().unwrap();
                let index = txn.open_table(LEGACY_TABLE_INDEX).unwrap();
                let mut raw = BTreeMap::new();
                for entry in index.iter().unwrap() {
                    let (table, bytes) = entry.unwrap();
                    raw.insert(table.value().to_string(), bytes.value().to_vec());
                }
                raw
            };

            let err = Storage::open(&path).err().expect("migration must fail");
            assert!(err.to_string().contains("zz_last"), "{err}");
            let lists_after = {
                let db = Database::create(&path).unwrap();
                let txn = db.begin_read().unwrap();
                assert!(matches!(
                    txn.open_table(TABLE_MEMBERS),
                    Err(redb::TableError::TableDoesNotExist(_))
                ));
                let index = txn.open_table(LEGACY_TABLE_INDEX).unwrap();
                let mut raw = BTreeMap::new();
                for entry in index.iter().unwrap() {
                    let (table, bytes) = entry.unwrap();
                    raw.insert(table.value().to_string(), bytes.value().to_vec());
                }
                raw
            };
            assert_eq!(lists_after, lists_before, "the legacy lists are untouched");
            let ro = Storage::open_read_only(&path).unwrap();
            assert_eq!(ro.count("notes").unwrap(), 6, "still readable as legacy");
            drop(ro);

            put_legacy_entry(&path, "zz_last", b"[\"only-decision\"]");
            let storage = Storage::open(&path).unwrap();
            assert_eq!(storage.count("zz_last").unwrap(), 1);
            assert_eq!(storage.count("notes").unwrap(), 5);
            // Positions now run ahead of the counts, as they do in use.
            storage.insert(&row("notes", "0001-new")).unwrap();
            storage.delete(&RecordId("zzzz-first".to_string())).unwrap();
            drop(storage);
            let migrated = keyed_dump(&path);
            drop(Storage::open(&path).unwrap());
            assert_eq!(
                keyed_dump(&path),
                migrated,
                "opening a migrated file again leaves its index alone"
            );
        }

        /// Every entry of the three keyed tables, raw.
        #[allow(clippy::type_complexity)]
        fn keyed_dump(
            path: &Path,
        ) -> (
            Vec<(String, u64, String)>,
            Vec<(String, String, u64)>,
            Vec<(String, TableCounts)>,
        ) {
            let db = Database::create(path).unwrap();
            let txn = db.begin_read().unwrap();
            let mut members = Vec::new();
            for entry in txn.open_table(TABLE_MEMBERS).unwrap().iter().unwrap() {
                let (key, id) = entry.unwrap();
                let (table, position) = key.value();
                members.push((table.to_string(), position, id.value().to_string()));
            }
            let mut positions = Vec::new();
            for entry in txn
                .open_table(TABLE_MEMBER_POSITIONS)
                .unwrap()
                .iter()
                .unwrap()
            {
                let (key, position) = entry.unwrap();
                let (table, id) = key.value();
                positions.push((table.to_string(), id.to_string(), position.value()));
            }
            let mut counts = Vec::new();
            for entry in txn.open_table(TABLE_COUNTS).unwrap().iter().unwrap() {
                let (table, value) = entry.unwrap();
                counts.push((table.value().to_string(), value.value()));
            }
            (members, positions, counts)
        }

        /// A file this version has opened writable, new or migrated, is
        /// refused by a binary that only knows the legacy layout, with an
        /// error that says what to do; it never reads stale lists.
        #[test]
        fn an_older_binary_cannot_open_a_keyed_file() {
            let dir = tempfile::tempdir().unwrap();
            let fresh = dir.path().join("fresh.axil");
            drop(Storage::open(&fresh).unwrap());
            let migrated = dir.path().join("migrated.axil");
            sample_legacy(&migrated);
            assert!(older_binary_open(&migrated).is_ok());
            drop(Storage::open(&migrated).unwrap());

            for path in [&fresh, &migrated] {
                let err = older_binary_open(path).unwrap_err();
                assert!(
                    matches!(err, redb::TableError::TableTypeMismatch { .. }),
                    "{err}"
                );
                let message = AxilError::from(err).to_string();
                assert!(message.contains("table_index"), "{message}");
                assert!(message.contains("upgrade axil"), "{message}");
            }
        }

        /// Downgrading writes the lists back in order (an emptied table has
        /// none), and opening the result migrates it to the same lists.
        #[test]
        fn downgrade_writes_the_lists_back_and_a_reopen_migrates_them_again() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("round.axil");
            let lists = {
                let storage = Storage::open(&path).unwrap();
                for id in ["zzzz-1", "0000-2", "mmmm-3"] {
                    storage.insert(&row("notes", id)).unwrap();
                }
                let gone = storage.insert(&row("gone", "gone-1")).unwrap();
                storage.delete(&gone).unwrap();
                storage.insert(&row("decisions", "d-1")).unwrap();
                storage.delete(&RecordId("0000-2".to_string())).unwrap();
                storage.insert(&row("notes", "0000-2")).unwrap();
                storage.downgrade_table_index().unwrap();
                legacy_lists(&path).expect("legacy after a downgrade")
            };
            let expected: BTreeMap<String, Vec<String>> = [
                ("decisions", vec!["d-1"]),
                ("notes", vec!["zzzz-1", "mmmm-3", "0000-2"]),
            ]
            .into_iter()
            .map(|(t, ids)| (t.to_string(), ids.into_iter().map(String::from).collect()))
            .collect();
            assert_eq!(lists, expected);
            assert!(older_binary_open(&path).is_ok());

            let storage = Storage::open(&path).unwrap();
            for (table, ids) in &expected {
                assert_eq!(&ids_of(&storage.list(table, usize::MAX, 0).unwrap()), ids);
                assert_eq!(storage.count(table).unwrap(), ids.len());
            }
            assert_eq!(storage.tables().unwrap(), ["decisions", "notes"]);
        }

        /// Two entity rows, a built index, then the file handed back to the
        /// legacy layout: the state a binary that had the index but not the
        /// keyed layout leaves.
        fn legacy_with_index(path: &Path) -> (RecordId, RecordId) {
            let storage = Storage::open(path).unwrap();
            let redis = storage
                .insert(&Record::new("_entities", json!({"canonical_id": "redis"})))
                .unwrap();
            let kafka = storage
                .insert(&Record::new("_entities", json!({"canonical_id": "kafka"})))
                .unwrap();
            assert!(storage.ensure_entity_key_index().unwrap());
            storage.downgrade_table_index().unwrap();
            (redis, kafka)
        }

        /// The marker moves with the list through a downgrade and a
        /// migration, so an index nobody made stale stays usable: on a
        /// read-only handle over the legacy file, and after migrating it.
        #[test]
        fn a_current_entity_index_stays_current_across_layouts() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("mixed.axil");
            let (redis, _kafka) = legacy_with_index(&path);

            let ro = Storage::open_read_only(&path).unwrap();
            assert_eq!(
                ro.entity_key_index_status().unwrap(),
                EntityKeyIndexStatus::Current
            );
            assert_index_matches_scan(&ro, "read-only over the legacy file");
            drop(ro);

            let storage = Storage::open(&path).unwrap();
            assert_eq!(
                storage.entity_key_index_status().unwrap(),
                EntityKeyIndexStatus::Current
            );
            assert_index_matches_scan(&storage, "after migrating");
            // A shared key is broken by list position, read per id now.
            let copy = storage
                .insert(&Record::new("_entities", json!({"name": "redis"})))
                .unwrap();
            let found = storage
                .lookup_entity_keys(&["redis"])
                .unwrap()
                .into_found()
                .unwrap();
            assert_eq!(found["redis"], copy);
            assert_ne!(found["redis"], redis);
            assert_index_matches_scan(&storage, "after a shared key");
        }

        /// An older binary that predates the index adds or deletes an entity
        /// before this version first opens the file: the migration must not
        /// carry the index's ready stamp over, or lookups would miss the new
        /// row (a duplicate entity) or return the deleted one.
        #[test]
        fn an_older_binarys_entity_writes_before_migration_leave_the_index_stale() {
            let dir = tempfile::tempdir().unwrap();
            let added = dir.path().join("added.axil");
            legacy_with_index(&added);
            let broker = Record::new("_entities", json!({"canonical_id": "kafka_broker"}));
            older_binary_insert(&added, &broker);

            let deleted = dir.path().join("deleted.axil");
            let (_redis, kafka) = legacy_with_index(&deleted);
            older_binary_delete(&deleted, &kafka);

            for (path, key) in [(&added, "kafka_broker"), (&deleted, "kafka")] {
                let ro = Storage::open_read_only(path).unwrap();
                assert_eq!(
                    ro.entity_key_index_status().unwrap(),
                    EntityKeyIndexStatus::Stale,
                    "read-only over the legacy file"
                );
                drop(ro);

                let storage = Storage::open(path).unwrap();
                assert_eq!(
                    storage.entity_key_index_status().unwrap(),
                    EntityKeyIndexStatus::Stale,
                    "after migrating"
                );
                assert_eq!(
                    storage.lookup_entity_keys(&[key]).unwrap(),
                    EntityKeyLookup::Stale
                );
                // A maintained write must not re-stamp over the gap.
                storage
                    .insert(&Record::new("_entities", json!({"canonical_id": "pool"})))
                    .unwrap();
                assert_ne!(
                    storage.entity_key_index_status().unwrap(),
                    EntityKeyIndexStatus::Current
                );
                assert!(storage.ensure_entity_key_index().unwrap());
                assert_index_matches_scan(&storage, "after the rebuild");
            }
            let storage = Storage::open(&added).unwrap();
            let found = storage
                .lookup_entity_keys(&["kafka_broker"])
                .unwrap()
                .into_found()
                .unwrap();
            assert_eq!(found["kafka_broker"], broker.id);
        }

        /// The legacy table index's rules on plain vectors: what `list`
        /// returned before the keyed layout, written out independently of
        /// it. One rule changed on purpose: a batch that repeats an id lists
        /// it once, at its first place (the legacy batch listed it twice).
        #[derive(Default)]
        struct LegacyModel {
            lists: BTreeMap<String, Vec<String>>,
            /// Table each live record's body names.
            body_table: HashMap<String, String>,
        }

        impl LegacyModel {
            fn drop_from(&mut self, table: &str, id: &str) {
                if let Some(ids) = self.lists.get_mut(table) {
                    ids.retain(|listed| listed != id);
                    if ids.is_empty() {
                        self.lists.remove(table);
                    }
                }
            }

            fn append(&mut self, table: &str, id: &str) {
                let ids = self.lists.entry(table.to_string()).or_default();
                if !ids.iter().any(|listed| listed == id) {
                    ids.push(id.to_string());
                }
            }

            fn insert(&mut self, table: &str, id: &str) {
                if let Some(old) = self.body_table.get(id).cloned() {
                    if old != table {
                        self.drop_from(&old, id);
                    }
                }
                self.append(table, id);
                self.body_table.insert(id.to_string(), table.to_string());
            }

            fn insert_batch(&mut self, rows: &[(String, String)]) {
                for (table, id) in rows {
                    self.append(table, id);
                    self.body_table.insert(id.clone(), table.clone());
                }
            }

            fn delete(&mut self, id: &str) {
                if let Some(table) = self.body_table.remove(id) {
                    self.drop_from(&table, id);
                }
            }

            fn assert_matches(&self, storage: &Storage, context: &str) {
                assert_eq!(
                    storage.tables().unwrap(),
                    self.lists.keys().cloned().collect::<Vec<_>>(),
                    "{context}: tables"
                );
                for (table, ids) in &self.lists {
                    let live: Vec<String> = ids
                        .iter()
                        .filter(|id| self.body_table.contains_key(*id))
                        .cloned()
                        .collect();
                    assert_eq!(
                        ids_of(&storage.list(table, usize::MAX, 0).unwrap()),
                        live,
                        "{context}: list of {table}"
                    );
                    assert_eq!(storage.count(table).unwrap(), ids.len(), "{context}");
                    let newest: Vec<String> = live.iter().rev().take(3).cloned().collect();
                    assert_eq!(
                        ids_of(&storage.list_newest(table, 3).unwrap()),
                        newest,
                        "{context}: list_newest of {table}"
                    );
                    let page: Vec<String> = ids
                        .iter()
                        .skip(1)
                        .take(2)
                        .filter(|id| self.body_table.contains_key(*id))
                        .cloned()
                        .collect();
                    assert_eq!(
                        ids_of(&storage.list(table, 2, 1).unwrap()),
                        page,
                        "{context}: page of {table}"
                    );
                }
            }
        }

        /// Random inserts, upserts across tables, batches, deletes, reopens
        /// and downgrade-then-migrate round trips, with ids whose sort order
        /// has nothing to do with insertion order: every list, page,
        /// `list_newest`, count and table name matches the legacy rules
        /// after every step.
        #[test]
        fn list_order_matches_the_legacy_rules_under_random_writes() {
            const TABLES: [&str; 3] = ["notes", "decisions", "_entities"];
            for seed in 1..=6u64 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("order.axil");
                let mut storage = Storage::open(&path).unwrap();
                let mut model = LegacyModel::default();
                let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
                let mut counter = 0u64;
                let mut ids: Vec<RecordId> = Vec::new();
                for step in 0..200 {
                    let what = match rng.below(10) {
                        0..=3 => {
                            let id = random_new_id(&mut rng, &mut counter);
                            let table = *rng.pick(&TABLES);
                            storage
                                .insert(&record_with(table, id.clone(), json!({})))
                                .unwrap();
                            model.insert(table, id.as_str());
                            ids.push(id);
                            "insert"
                        }
                        4 if !ids.is_empty() => {
                            let id = rng.pick(&ids).clone();
                            let table = *rng.pick(&TABLES);
                            storage
                                .insert(&record_with(table, id.clone(), json!({})))
                                .unwrap();
                            model.insert(table, id.as_str());
                            "upsert"
                        }
                        5 => {
                            let mut batch: Vec<(String, RecordId)> = Vec::new();
                            for _ in 0..(1 + rng.below(4)) {
                                let id = if !ids.is_empty() && rng.below(3) == 0 {
                                    rng.pick(&ids).clone()
                                } else if !batch.is_empty() && rng.below(4) == 0 {
                                    rng.pick(&batch).1.clone()
                                } else {
                                    random_new_id(&mut rng, &mut counter)
                                };
                                batch.push((rng.pick(&TABLES).to_string(), id.clone()));
                                ids.push(id);
                            }
                            let records: Vec<Record> = batch
                                .iter()
                                .map(|(t, id)| record_with(t, id.clone(), json!({})))
                                .collect();
                            storage.insert_batch(&records).unwrap();
                            let rows: Vec<(String, String)> = batch
                                .iter()
                                .map(|(t, id)| (t.clone(), id.as_str().to_string()))
                                .collect();
                            model.insert_batch(&rows);
                            "insert_batch"
                        }
                        6 | 7 if !ids.is_empty() => {
                            let id = rng.pick(&ids).clone();
                            storage.delete(&id).unwrap();
                            model.delete(id.as_str());
                            "delete"
                        }
                        8 => {
                            drop(storage);
                            storage = Storage::open(&path).unwrap();
                            "reopen"
                        }
                        _ => {
                            storage.downgrade_table_index().unwrap();
                            let ro = Storage::open_read_only(&path).unwrap();
                            model.assert_matches(
                                &ro,
                                &format!("seed {seed} step {step} legacy read-only"),
                            );
                            drop(ro);
                            storage = Storage::open(&path).unwrap();
                            "downgrade, migrate"
                        }
                    };
                    model.assert_matches(&storage, &format!("seed {seed} step {step} ({what})"));
                }
            }
        }
    }
}
