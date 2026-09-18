//! TTL / expiry system for agent memory records.
//!
//! Records can have a `valid_until` timestamp in their metadata.
//! Expired records are excluded from default queries.

use chrono::{DateTime, Duration, Utc};
use serde_json::{json, Value};

use axil_core::{Axil, Record, RecordId, Result};

use crate::types::META_VALID_UNTIL;

/// Engine for managing record TTL and expiry.
pub struct TtlEngine<'a> {
    db: &'a Axil,
}

impl<'a> TtlEngine<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Set a TTL on a record. The record will expire after `duration`.
    pub fn set_ttl(&self, id: &RecordId, duration: Duration) -> Result<Record> {
        let valid_until = Utc::now() + duration;
        self.set_expiry(id, valid_until)
    }

    /// Set an absolute expiry time on a record.
    pub fn set_expiry(&self, id: &RecordId, valid_until: DateTime<Utc>) -> Result<Record> {
        let record = self
            .db
            .get(id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("record {id}")))?;

        let mut data = record.data.clone();
        // Canonical field is top-level `valid_until` — the same place core's
        // cleanup reads. Clear any legacy `_meta.valid_until` so a record can
        // never carry two disagreeing expiry stamps.
        data["valid_until"] = json!(valid_until.to_rfc3339());
        remove_meta_field(&mut data, META_VALID_UNTIL);
        self.db.update(id, data)
    }

    /// Remove TTL from a record (it will no longer expire).
    pub fn clear_ttl(&self, id: &RecordId) -> Result<Record> {
        let record = self
            .db
            .get(id)?
            .ok_or_else(|| axil_core::AxilError::NotFound(format!("record {id}")))?;

        let mut data = record.data.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.remove("valid_until");
        }
        remove_meta_field(&mut data, META_VALID_UNTIL);
        self.db.update(id, data)
    }

    /// Check if a record has expired.
    pub fn is_expired(&self, record: &Record) -> bool {
        is_record_expired(record)
    }

    /// Filter a list of records to exclude expired ones.
    pub fn filter_active(&self, records: Vec<Record>) -> Vec<Record> {
        filter_expired(records)
    }
}

/// Check if a record is expired based on its `_meta.valid_until` field.
pub fn is_record_expired(record: &Record) -> bool {
    if let Some(valid_until) = get_valid_until(record) {
        return Utc::now() > valid_until;
    }
    false
}

/// Check if a record is superseded.
///
/// Reads the canonical top-level `_superseded` flag (what core consolidation
/// writes) plus the legacy `_meta.superseded` sub-object written by older
/// versions of this extension.
pub fn is_record_superseded(record: &Record) -> bool {
    if record.data.get("_superseded").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    record
        .data
        .get("_meta")
        .and_then(|m| m.get(crate::types::META_SUPERSEDED))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Filter out expired and superseded records.
pub fn filter_expired(records: Vec<Record>) -> Vec<Record> {
    records
        .into_iter()
        .filter(|r| !is_record_expired(r) && !is_record_superseded(r))
        .collect()
}

/// Extract the `valid_until` timestamp from a record.
///
/// Reads the canonical top-level `data.valid_until`, then the two legacy
/// spellings (`metadata.valid_until`, the `_meta` sub-object) so records
/// stamped by older writers still expire on time.
pub fn get_valid_until(record: &Record) -> Option<DateTime<Utc>> {
    let raw = record
        .data
        .get("valid_until")
        .and_then(|v| v.as_str())
        .or_else(|| record.metadata.as_ref()?.get("valid_until")?.as_str())
        .or_else(|| {
            record
                .data
                .get("_meta")
                .and_then(|m| m.get(META_VALID_UNTIL))
                .and_then(|v| v.as_str())
        })?;
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

// ── Metadata helpers ─────────────────────────────────────────────────────

/// Set a field in the `_meta` sub-object of a record's data.
pub fn set_meta_field(data: &mut Value, key: &str, value: Value) {
    if !data.get("_meta").is_some_and(|m| m.is_object()) {
        data["_meta"] = json!({});
    }
    data["_meta"][key] = value;
}

/// Remove a field from the `_meta` sub-object.
pub fn remove_meta_field(data: &mut Value, key: &str) {
    if let Some(meta) = data.get_mut("_meta").and_then(|m| m.as_object_mut()) {
        meta.remove(key);
    }
}

/// Get a string field from the `_meta` sub-object.
pub fn get_meta_str<'a>(data: &'a Value, key: &str) -> Option<&'a str> {
    data.get("_meta")
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expired_record_detected() {
        let mut r = axil_core::Record::new("test", json!({}));
        // Not expired — no valid_until
        assert!(!is_record_expired(&r));

        // Set valid_until to the past
        let past = Utc::now() - Duration::hours(1);
        set_meta_field(&mut r.data, META_VALID_UNTIL, json!(past.to_rfc3339()));
        assert!(is_record_expired(&r));

        // Set valid_until to the future
        let future = Utc::now() + Duration::hours(1);
        set_meta_field(&mut r.data, META_VALID_UNTIL, json!(future.to_rfc3339()));
        assert!(!is_record_expired(&r));
    }

    #[test]
    fn filter_removes_expired() {
        let mut r1 = axil_core::Record::new("test", json!({"val": 1}));
        let r2 = axil_core::Record::new("test", json!({"val": 2}));
        let past = Utc::now() - Duration::hours(1);
        set_meta_field(&mut r1.data, META_VALID_UNTIL, json!(past.to_rfc3339()));

        let filtered = filter_expired(vec![r1, r2]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].data["val"], 2);
    }

    #[test]
    fn filter_removes_superseded() {
        let mut r1 = axil_core::Record::new("test", json!({"val": 1}));
        let r2 = axil_core::Record::new("test", json!({"val": 2}));
        set_meta_field(&mut r1.data, crate::types::META_SUPERSEDED, json!(true));

        let filtered = filter_expired(vec![r1, r2]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].data["val"], 2);
    }

    #[test]
    fn canonical_top_level_markers_are_honored() {
        // Core writes `_superseded` / top-level `valid_until`; the extension's
        // filters must treat those spellings as authoritative.
        let superseded = axil_core::Record::new("test", json!({"_superseded": true}));
        assert!(is_record_superseded(&superseded));

        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        let expired = axil_core::Record::new("test", json!({"valid_until": past}));
        assert!(is_record_expired(&expired));

        let filtered = filter_expired(vec![superseded, expired]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn valid_until_read_across_all_spellings() {
        let past = (Utc::now() - Duration::hours(1)).to_rfc3339();
        let future = (Utc::now() + Duration::hours(1)).to_rfc3339();

        // Canonical top-level wins even when a legacy stamp disagrees.
        let mut r = axil_core::Record::new("test", json!({"valid_until": future}));
        set_meta_field(&mut r.data, META_VALID_UNTIL, json!(past));
        assert!(!is_record_expired(&r));

        // metadata.valid_until (core's other legacy location) is honored.
        let mut r2 = axil_core::Record::new("test", json!({}));
        r2.metadata = Some(json!({"valid_until": past}));
        assert!(is_record_expired(&r2));
    }

    #[test]
    fn meta_field_operations() {
        let mut data = json!({"content": "hello"});
        set_meta_field(&mut data, "foo", json!("bar"));
        assert_eq!(get_meta_str(&data, "foo"), Some("bar"));

        remove_meta_field(&mut data, "foo");
        assert_eq!(get_meta_str(&data, "foo"), None);
    }
}
