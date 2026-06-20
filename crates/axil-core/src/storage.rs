use std::path::Path;

use chrono::Utc;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};

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

/// Low-level storage backend wrapping a `redb::Database`.
pub struct Storage {
    db: Database,
}

impl Storage {
    /// Open (or create) a database at the given path.
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
        }
        txn.commit()?;

        Ok(Self { db })
    }

    /// Insert a record. Returns its ID.
    pub fn insert(&self, record: &Record) -> Result<RecordId> {
        let bytes = record.to_bytes()?;
        let id = record.id.as_str();

        let txn = self.db.begin_write()?;
        {
            let mut records = txn.open_table(RECORDS)?;

            // If this ID already exists under a different table, clean up the old index.
            if let Some(guard) = records.get(id)? {
                let old_bytes: &[u8] = guard.value();
                if let Ok(old_record) = Record::from_bytes(old_bytes) {
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

        let txn = self.db.begin_write()?;
        {
            let mut tbl = txn.open_table(RECORDS)?;
            let mut idx = txn.open_table(TABLE_INDEX)?;

            // Group records by table to minimize index reads.
            let mut table_ids: std::collections::HashMap<&str, Vec<RecordId>> =
                std::collections::HashMap::new();

            for record in records {
                let bytes = record.to_bytes()?;
                tbl.insert(record.id.as_str(), bytes.as_slice())?;
                table_ids
                    .entry(&record.table)
                    .or_default()
                    .push(record.id.clone());
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
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS)?;

        match table.get(id.as_str())? {
            Some(guard) => {
                let bytes: &[u8] = guard.value();
                let record = Record::from_bytes(bytes)?;
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
        let txn = self.db.begin_write()?;
        {
            let mut records = txn.open_table(RECORDS)?;

            // Read the record within the write transaction to get table name.
            let table_name = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    let record = Record::from_bytes(bytes)?;
                    record.table
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
        }
        txn.commit()?;

        Ok(true)
    }

    /// List records in a table with optional limit and offset.
    pub fn list(&self, table: &str, limit: usize, offset: usize) -> Result<Vec<Record>> {
        let txn = self.db.begin_read()?;
        let idx_table = txn.open_table(TABLE_INDEX)?;
        let ids = Self::read_index(&idx_table, table)?;

        let records_table = txn.open_table(RECORDS)?;
        let mut results = Vec::new();

        for rid in ids.into_iter().skip(offset).take(limit) {
            if let Some(guard) = records_table.get(rid.as_str())? {
                let bytes: &[u8] = guard.value();
                let record = Record::from_bytes(bytes)?;
                results.push(record);
            }
        }

        Ok(results)
    }

    /// Update a record's data. Returns the updated record.
    ///
    /// Uses a single write transaction to ensure atomicity.
    pub fn update(&self, id: &RecordId, data: serde_json::Value) -> Result<Record> {
        let txn = self.db.begin_write()?;
        let record = {
            let mut records = txn.open_table(RECORDS)?;

            // Read the current record within the write transaction.
            let mut record = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    Record::from_bytes(bytes)?
                }
                None => return Err(AxilError::NotFound(format!("record {id}"))),
            };

            record.data = data;
            record.updated_at = Utc::now();

            let bytes = record.to_bytes()?;
            records.insert(id.as_str(), bytes.as_slice())?;
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
        let txn = self.db.begin_write()?;
        let record = {
            let mut records = txn.open_table(RECORDS)?;
            let mut record = match records.get(id.as_str())? {
                Some(guard) => {
                    let bytes: &[u8] = guard.value();
                    Record::from_bytes(bytes)?
                }
                None => return Err(AxilError::NotFound(format!("record {id}"))),
            };
            record.metadata = metadata;
            let bytes = record.to_bytes()?;
            records.insert(id.as_str(), bytes.as_slice())?;
            record
        };
        txn.commit()?;
        Ok(record)
    }

    /// Count all record IDs in a given table.
    pub fn count(&self, table: &str) -> Result<usize> {
        let txn = self.db.begin_read()?;
        let idx = txn.open_table(TABLE_INDEX)?;
        let ids = Self::read_index(&idx, table)?;
        Ok(ids.len())
    }

    /// List all table names that have records.
    pub fn tables(&self) -> Result<Vec<String>> {
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        Ok(table.len()? as usize)
    }

    // ── diagnostic log operations ──────────────────────────────────────

    /// Append a slow query entry. Key format: timestamp + counter for ordering.
    pub fn append_slow_query(&self, key: &str, entry: &[u8]) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(SLOW_QUERIES)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read all slow query entries, ordered by key (timestamp).
    pub fn list_slow_queries(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_write()?;
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
        let txn = self.db.begin_write()?;
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
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(AUDIT_LOG)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read audit log entries, ordered newest first.
    pub fn list_audit(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_write()?;
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
        let txn = self.db.begin_write()?;
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
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(METRICS_HISTORY)?;
            table.insert(key, entry)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Read metrics history entries, ordered newest first.
    pub fn list_metrics_history(&self, limit: usize) -> Result<Vec<(String, Vec<u8>)>> {
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_write()?;
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
        let txn = self.db.begin_read()?;
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
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (_, value): (redb::AccessGuard<'_, &str>, redb::AccessGuard<'_, &[u8]>) = entry?;
            if let Ok(record) = Record::from_bytes(value.value()) {
                records.push(record);
            }
        }
        Ok(records)
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
}
