pub mod index;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use axil_core::plugin::{Capability, Plugin, TimeBucket, TimeSeriesIndex};
use axil_core::record::{Record, RecordId};
use axil_core::{companion_path, AxilBuilder, AxilError, Result};

use crate::index::TimeEntry;

// ── redb table definitions ──────────────────────────────────────────

/// Time entries table: record_id -> serialized TimeEntry.
const TIME_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("time_entries");

// ── Limits ──────────────────────────────────────────────────────────

/// Maximum number of time entries allowed.
const MAX_ENTRIES: usize = 10_000_000;

type TimeBounds = (
    std::ops::Bound<(i64, RecordId)>,
    std::ops::Bound<(i64, RecordId)>,
);

// ── In-memory time index ────────────────────────────────────────────

/// In-memory sorted index for fast time-range queries.
///
/// Two BTreeMaps provide sorted access by created_at and updated_at.
/// A HashMap provides O(1) lookup by record ID for deletions.
struct TimeIndex {
    /// All entries by record ID.
    entries: HashMap<RecordId, TimeEntry>,
    /// Sorted by (created_at_us, record_id) for range queries.
    by_created: BTreeMap<(i64, RecordId), ()>,
    /// Sorted by (updated_at_us, record_id) for change tracking.
    by_updated: BTreeMap<(i64, RecordId), ()>,
}

impl TimeIndex {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            by_created: BTreeMap::new(),
            by_updated: BTreeMap::new(),
        }
    }

    fn add(&mut self, entry: TimeEntry) {
        // Remove stale BTreeMap keys if this record was already indexed
        // (e.g. during backfill or retry after partial failure).
        if let Some(old) = self.entries.get(&entry.record_id) {
            self.by_created
                .remove(&(old.created_at_us, entry.record_id.clone()));
            self.by_updated
                .remove(&(old.updated_at_us, entry.record_id.clone()));
        }
        self.by_created
            .insert((entry.created_at_us, entry.record_id.clone()), ());
        self.by_updated
            .insert((entry.updated_at_us, entry.record_id.clone()), ());
        self.entries.insert(entry.record_id.clone(), entry);
    }

    fn remove(&mut self, record_id: &RecordId) -> Option<TimeEntry> {
        if let Some(entry) = self.entries.remove(record_id) {
            self.by_created
                .remove(&(entry.created_at_us, record_id.clone()));
            self.by_updated
                .remove(&(entry.updated_at_us, record_id.clone()));
            Some(entry)
        } else {
            None
        }
    }

    /// Update the updated_at timestamp for a record.
    fn update_timestamp(&mut self, record_id: &RecordId, new_updated_us: i64) {
        if let Some(entry) = self.entries.get_mut(record_id) {
            // Remove old updated_at key.
            self.by_updated
                .remove(&(entry.updated_at_us, record_id.clone()));
            // Insert new one.
            entry.updated_at_us = new_updated_us;
            self.by_updated
                .insert((new_updated_us, record_id.clone()), ());
        }
    }

    /// Build BTreeMap range bounds for `[start_us, end_us]`.
    fn created_bounds(start_us: i64, end_us: i64) -> TimeBounds {
        use std::ops::Bound;
        let lo = Bound::Included((start_us, RecordId(String::new())));
        let hi = if end_us == i64::MAX {
            Bound::Unbounded
        } else {
            Bound::Excluded((end_us + 1, RecordId(String::new())))
        };
        (lo, hi)
    }

    /// Check if a record passes the optional table filter.
    fn matches_table(&self, rid: &RecordId, table: Option<&str>) -> bool {
        match table {
            None => true,
            Some(t) => self.entries.get(rid).is_some_and(|entry| entry.table == t),
        }
    }

    /// Get record IDs with created_at in [start_us, end_us], optionally filtered by table.
    fn range_created(&self, table: Option<&str>, start_us: i64, end_us: i64) -> Vec<RecordId> {
        if start_us > end_us {
            return Vec::new();
        }
        let bounds = Self::created_bounds(start_us, end_us);
        self.by_created
            .range(bounds)
            .filter(|((_, rid), _)| self.matches_table(rid, table))
            .map(|((_, rid), _)| rid.clone())
            .collect()
    }

    /// Get record IDs with updated_at >= threshold_us, optionally filtered by table.
    fn changed_since(&self, table: Option<&str>, threshold_us: i64) -> Vec<RecordId> {
        use std::ops::Bound;
        let lo = Bound::Included((threshold_us, RecordId(String::new())));
        let hi: Bound<(i64, RecordId)> = Bound::Unbounded;
        self.by_updated
            .range((lo, hi))
            .filter(|((_, rid), _)| self.matches_table(rid, table))
            .map(|((_, rid), _)| rid.clone())
            .collect()
    }

    /// Get the most recent N records by created_at, newest first.
    fn latest(&self, table: Option<&str>, limit: usize) -> Vec<RecordId> {
        if limit == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(limit.min(self.entries.len()));
        for ((_, rid), _) in self.by_created.iter().rev() {
            if result.len() >= limit {
                break;
            }
            if self.matches_table(rid, table) {
                result.push(rid.clone());
            }
        }
        result
    }

    /// Count records grouped by time bucket within [start_us, end_us].
    fn count_by_bucket(
        &self,
        table: Option<&str>,
        bucket: TimeBucket,
        start_us: i64,
        end_us: i64,
    ) -> Vec<(i64, usize)> {
        if start_us > end_us {
            return Vec::new();
        }
        let bounds = Self::created_bounds(start_us, end_us);
        let mut counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for ((ts_us, rid), _) in self.by_created.range(bounds) {
            if self.matches_table(rid, table) {
                *counts.entry(bucket.truncate_us(*ts_us)).or_insert(0) += 1;
            }
        }
        counts.into_iter().collect()
    }

    fn count(&self) -> usize {
        self.entries.len()
    }
}

// ── TimeSeriesPlugin ────────────────────────────────────────────────

/// Time-series plugin for Axil — indexes records by creation and update
/// timestamps for fast range queries.
pub struct TimeSeriesPlugin {
    ts_db: Database,
    index: RwLock<TimeIndex>,
}

impl TimeSeriesPlugin {
    /// Open or create a time-series store at the companion path for the given database.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let ts_path = companion_path(db_path.as_ref(), ".ts");
        let ts_db = Database::create(&ts_path).map_err(|e| {
            AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "failed to open timeseries store at {}: {e}",
                ts_path.display()
            ))))
        })?;

        // Ensure table exists.
        {
            let txn = ts_db.begin_write()?;
            let _ = txn.open_table(TIME_TABLE)?;
            txn.commit()?;
        }

        // Load existing entries into memory, cleaning corrupt entries.
        let mut idx = TimeIndex::new();
        let mut corrupt_keys: Vec<String> = Vec::new();
        {
            let txn = ts_db.begin_read()?;
            let table = txn.open_table(TIME_TABLE)?;
            let iter = table.iter()?;
            for entry in iter {
                let entry = entry?;
                let key = entry.0.value().to_string();
                let bytes = entry.1.value();
                match TimeEntry::from_bytes(bytes) {
                    Ok(te) => idx.add(te),
                    Err(e) => {
                        eprintln!("warning: removing corrupt time entry {key}: {e}");
                        corrupt_keys.push(key);
                    }
                }
            }
        }

        // Remove corrupt entries from disk.
        if !corrupt_keys.is_empty() {
            let txn = ts_db.begin_write()?;
            {
                let mut table = txn.open_table(TIME_TABLE)?;
                for key in &corrupt_keys {
                    table.remove(key.as_str())?;
                }
            }
            txn.commit()?;
        }

        if idx.count() > MAX_ENTRIES {
            return Err(AxilError::Plugin(Box::new(std::io::Error::other(format!(
                "timeseries store has {} entries, exceeding limit of {MAX_ENTRIES}",
                idx.count()
            )))));
        }

        Ok(Self {
            ts_db,
            index: RwLock::new(idx),
        })
    }

    /// Total entries in the index.
    pub fn count(&self) -> usize {
        self.index.read().count()
    }

    // ── Persistence helpers ─────────────────────────────────────────

    fn persist_entry(&self, entry: &TimeEntry) -> Result<()> {
        let bytes = entry
            .to_bytes()
            .map_err(|e| AxilError::Serialization(Box::new(e)))?;
        let txn = self.ts_db.begin_write()?;
        {
            let mut table = txn.open_table(TIME_TABLE)?;
            table.insert(entry.record_id.as_str(), bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    fn remove_entry_from_disk(&self, record_id: &RecordId) -> Result<()> {
        let txn = self.ts_db.begin_write()?;
        {
            let mut table = txn.open_table(TIME_TABLE)?;
            table.remove(record_id.as_str())?;
        }
        txn.commit()?;
        Ok(())
    }
}

// ── Plugin trait ────────────────────────────────────────────────────

impl Plugin for TimeSeriesPlugin {
    fn name(&self) -> &str {
        "timeseries"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::TimeSeries]
    }

    fn on_record_insert(&self, record: &Record) -> Result<()> {
        let entry = TimeEntry {
            record_id: record.id.clone(),
            table: record.table.clone(),
            created_at_us: record.created_at.timestamp_micros(),
            updated_at_us: record.updated_at.timestamp_micros(),
        };

        // Hold write lock across check + disk + memory to prevent TOCTOU
        // races on the MAX_ENTRIES limit and concurrent insert/delete.
        let mut idx = self.index.write();
        if idx.count() >= MAX_ENTRIES {
            return Err(AxilError::InvalidQuery(format!(
                "timeseries entry limit reached ({MAX_ENTRIES})"
            )));
        }
        self.persist_entry(&entry)?;
        idx.add(entry);

        Ok(())
    }

    fn on_record_update(&self, record: &Record) -> Result<()> {
        let new_updated_us = record.updated_at.timestamp_micros();

        // Hold write lock across the entire operation to prevent a
        // concurrent delete from creating orphaned disk entries.
        let mut idx = self.index.write();
        if !idx.entries.contains_key(&record.id) {
            return Ok(());
        }
        let mut entry = match idx.entries.get(&record.id) {
            Some(e) => e.clone(),
            None => return Ok(()), // removed between contains_key and get (shouldn't happen under write lock)
        };
        entry.updated_at_us = new_updated_us;
        self.persist_entry(&entry)?;
        idx.update_timestamp(&record.id, new_updated_us);

        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        // Hold write lock across disk + memory to prevent a concurrent
        // insert/update from resurrecting the entry after deletion.
        let mut idx = self.index.write();
        if idx.entries.contains_key(id) {
            self.remove_entry_from_disk(id)?;
            idx.remove(id);
        }
        Ok(())
    }
}

// ── TimeSeriesIndex trait ───────────────────────────────────────────

impl TimeSeriesIndex for TimeSeriesPlugin {
    fn range(&self, table: Option<&str>, start_us: i64, end_us: i64) -> Result<Vec<RecordId>> {
        Ok(self.index.read().range_created(table, start_us, end_us))
    }

    fn since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<RecordId>> {
        let now_us = Utc::now().timestamp_micros();
        let delta_us = i64::try_from(duration_secs)
            .ok()
            .and_then(|s| s.checked_mul(1_000_000))
            .unwrap_or(i64::MAX);
        let start_us = now_us.saturating_sub(delta_us);
        Ok(self.index.read().range_created(table, start_us, now_us))
    }

    fn latest(&self, table: Option<&str>, limit: usize) -> Result<Vec<RecordId>> {
        Ok(self.index.read().latest(table, limit))
    }

    fn changed_since(&self, table: Option<&str>, duration_secs: u64) -> Result<Vec<RecordId>> {
        let now_us = Utc::now().timestamp_micros();
        let delta_us = i64::try_from(duration_secs)
            .ok()
            .and_then(|s| s.checked_mul(1_000_000))
            .unwrap_or(i64::MAX);
        let threshold_us = now_us.saturating_sub(delta_us);
        Ok(self.index.read().changed_since(table, threshold_us))
    }

    fn changed_since_absolute(
        &self,
        table: Option<&str>,
        threshold_us: i64,
    ) -> Result<Vec<RecordId>> {
        Ok(self.index.read().changed_since(table, threshold_us))
    }

    fn count_by_bucket(
        &self,
        table: Option<&str>,
        bucket: TimeBucket,
        start_us: i64,
        end_us: i64,
    ) -> Result<Vec<(i64, usize)>> {
        Ok(self
            .index
            .read()
            .count_by_bucket(table, bucket, start_us, end_us))
    }

    fn entry_count(&self) -> usize {
        self.index.read().count()
    }
}

// ── Builder extension ───────────────────────────────────────────────

/// Extension trait for adding time-series support to `AxilBuilder`.
pub trait AxilBuilderTimeSeriesExt {
    /// Enable time-series indexing with a companion `.ts` file.
    fn with_timeseries_plugin(self) -> Result<Self>
    where
        Self: Sized;
}

impl AxilBuilderTimeSeriesExt for AxilBuilder {
    fn with_timeseries_plugin(self) -> Result<Self> {
        let plugin = TimeSeriesPlugin::open(self.path())?;
        let arc: Arc<dyn TimeSeriesIndex> = Arc::new(plugin);
        Ok(self.with_timeseries_index(arc))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Check if a timeseries store exists for the given database.
pub fn has_timeseries_store(db_path: &Path) -> bool {
    companion_path(db_path, ".ts").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_ts() -> (TimeSeriesPlugin, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let plugin = TimeSeriesPlugin::open(&path).unwrap();
        (plugin, dir)
    }

    fn make_record(table: &str) -> Record {
        Record::new(table, json!({"data": "test"}))
    }

    #[test]
    fn insert_and_count() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();
        assert_eq!(ts.count(), 1);
    }

    #[test]
    fn insert_and_delete() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();
        assert_eq!(ts.count(), 1);
        ts.on_record_delete(&r.id).unwrap();
        assert_eq!(ts.count(), 0);
    }

    #[test]
    fn since_returns_recent() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();

        let ids = ts.since(None, 60).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], r.id);
    }

    #[test]
    fn latest_returns_newest_first() {
        let (ts, _dir) = temp_ts();
        let r1 = make_record("sessions");
        ts.on_record_insert(&r1).unwrap();
        let r2 = make_record("sessions");
        ts.on_record_insert(&r2).unwrap();
        let r3 = make_record("sessions");
        ts.on_record_insert(&r3).unwrap();

        let ids = ts.latest(None, 2).unwrap();
        assert_eq!(ids.len(), 2);
        // Newest first.
        assert_eq!(ids[0], r3.id);
        assert_eq!(ids[1], r2.id);
    }

    #[test]
    fn range_query() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        let created_us = r.created_at.timestamp_micros();
        ts.on_record_insert(&r).unwrap();

        // Range that includes the record.
        let ids = ts.range(None, created_us - 1, created_us + 1).unwrap();
        assert_eq!(ids.len(), 1);

        // Range that excludes the record.
        let ids = ts.range(None, created_us + 1, created_us + 2).unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn table_filter() {
        let (ts, _dir) = temp_ts();
        let r1 = make_record("sessions");
        let r2 = make_record("decisions");
        ts.on_record_insert(&r1).unwrap();
        ts.on_record_insert(&r2).unwrap();

        let ids = ts.since(Some("sessions"), 60).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], r1.id);

        let ids = ts.since(Some("decisions"), 60).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], r2.id);
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let r = make_record("sessions");

        {
            let ts = TimeSeriesPlugin::open(&path).unwrap();
            ts.on_record_insert(&r).unwrap();
            assert_eq!(ts.count(), 1);
        }

        {
            let ts = TimeSeriesPlugin::open(&path).unwrap();
            assert_eq!(ts.count(), 1);
            let ids = ts.since(None, 60).unwrap();
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], r.id);
        }
    }

    #[test]
    fn on_record_update_refreshes_updated_at() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();

        // Simulate an update with a newer updated_at.
        let mut updated_r = r.clone();
        updated_r.updated_at = chrono::Utc::now() + chrono::Duration::seconds(10);
        ts.on_record_update(&updated_r).unwrap();

        // The entry should reflect the new updated_at.
        let idx = ts.index.read();
        let entry = idx.entries.get(&r.id).unwrap();
        assert_eq!(entry.updated_at_us, updated_r.updated_at.timestamp_micros());
    }

    #[test]
    fn changed_since_returns_recent() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();

        let ids = ts.changed_since(None, 60).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], r.id);
    }

    #[test]
    fn update_nonexistent_is_noop() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        // Don't insert — call on_record_update directly.
        ts.on_record_update(&r).unwrap();
        assert_eq!(ts.count(), 0);
    }

    #[test]
    fn delete_nonexistent_is_ok() {
        let (ts, _dir) = temp_ts();
        let fake_id = RecordId::new();
        ts.on_record_delete(&fake_id).unwrap();
    }

    #[test]
    fn count_by_bucket_day() {
        let (ts, _dir) = temp_ts();
        let r1 = make_record("sessions");
        ts.on_record_insert(&r1).unwrap();
        let r2 = make_record("sessions");
        ts.on_record_insert(&r2).unwrap();

        let now_us = chrono::Utc::now().timestamp_micros();
        let buckets = ts
            .count_by_bucket(None, TimeBucket::Day, now_us - 86_400_000_000, now_us + 1)
            .unwrap();
        // Both records should be in the same day bucket.
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].1, 2);
    }

    #[test]
    fn count_by_bucket_empty_range() {
        let (ts, _dir) = temp_ts();
        let r = make_record("sessions");
        ts.on_record_insert(&r).unwrap();

        // Future range with no records.
        let now_us = chrono::Utc::now().timestamp_micros();
        let buckets = ts
            .count_by_bucket(
                None,
                TimeBucket::Day,
                now_us + 1_000_000_000,
                now_us + 2_000_000_000,
            )
            .unwrap();
        assert!(buckets.is_empty());
    }

    #[test]
    fn count_by_bucket_table_filter() {
        let (ts, _dir) = temp_ts();
        let r1 = make_record("sessions");
        let r2 = make_record("decisions");
        ts.on_record_insert(&r1).unwrap();
        ts.on_record_insert(&r2).unwrap();

        let now_us = chrono::Utc::now().timestamp_micros();
        let buckets = ts
            .count_by_bucket(
                Some("sessions"),
                TimeBucket::Day,
                now_us - 86_400_000_000,
                now_us + 1,
            )
            .unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].1, 1);
    }

    #[test]
    fn has_timeseries_store_false_before_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        assert!(!has_timeseries_store(&path));
    }

    #[test]
    fn has_timeseries_store_true_after_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let _ts = TimeSeriesPlugin::open(&path).unwrap();
        assert!(has_timeseries_store(&path));
    }
}
