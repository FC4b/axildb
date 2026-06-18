use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// Unique, time-sortable record identifier (ULID).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RecordId(pub String);

impl RecordId {
    /// Generate a new ULID-based record ID.
    pub fn new() -> Self {
        Self(ulid::Ulid::new().to_string())
    }

    /// Wrap an existing string as a `RecordId`.
    ///
    /// Validates that the string is a valid ULID. Returns an error if not.
    pub fn from_string(s: impl Into<String>) -> crate::error::Result<Self> {
        let s = s.into();
        ulid::Ulid::from_string(&s).map_err(|e| {
            crate::error::AxilError::InvalidQuery(format!("invalid record ID '{s}': {e}"))
        })?;
        Ok(Self(s))
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RecordId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single document stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Unique identifier.
    pub id: RecordId,
    /// Logical table this record belongs to.
    pub table: String,
    /// Arbitrary JSON payload.
    pub data: Value,
    /// When the record was created.
    pub created_at: DateTime<Utc>,
    /// When the record was last updated.
    pub updated_at: DateTime<Utc>,
    /// Optional metadata (tags, TTL, etc.).
    pub metadata: Option<Value>,
}

impl fmt::Display for Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}", self.table, self.id, self.data)
    }
}

impl Record {
    /// Create a new record in the given table with the given data.
    pub fn new(table: impl Into<String>, data: Value) -> Self {
        let now = Utc::now();
        Self {
            id: RecordId::new(),
            table: table.into(),
            data,
            created_at: now,
            updated_at: now,
            metadata: None,
        }
    }

    /// Serialize the record to bytes for storage.
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(Into::into)
    }

    /// Deserialize a record from bytes.
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        serde_json::from_slice(bytes).map_err(Into::into)
    }

    /// Read-consent scope, falling back to `private` when unset.
    pub fn read_consent_raw(&self) -> Value {
        self.metadata
            .as_ref()
            .and_then(|m| m.get("consent"))
            .and_then(|c| c.get("read"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"kind": "private"}))
    }

    /// Write-consent scope, falling back to `source_only` when unset.
    pub fn write_consent_raw(&self) -> Value {
        self.metadata
            .as_ref()
            .and_then(|m| m.get("consent"))
            .and_then(|c| c.get("write"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"kind": "source_only"}))
    }

    /// Update the consent scopes on this record. `None` for an axis
    /// leaves that axis unchanged; pass an explicit scope to set it.
    pub fn set_consent(&mut self, read: Option<Value>, write: Option<Value>) {
        if read.is_none() && write.is_none() {
            return;
        }
        let meta = self.metadata.get_or_insert_with(|| serde_json::json!({}));
        let Some(obj) = meta.as_object_mut() else {
            return;
        };
        let consent = obj
            .entry("consent")
            .or_insert_with(|| serde_json::json!({}));
        let Some(consent_obj) = consent.as_object_mut() else {
            return;
        };
        if let Some(v) = read {
            consent_obj.insert("read".to_string(), v);
        }
        if let Some(v) = write {
            consent_obj.insert("write".to_string(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_id_is_unique() {
        let a = RecordId::new();
        let b = RecordId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn record_creation() {
        let r = Record::new("sessions", json!({"summary": "hello"}));
        assert_eq!(r.table, "sessions");
        assert_eq!(r.data["summary"], "hello");
        assert!(r.metadata.is_none());
        assert!(r.created_at <= Utc::now());
    }

    #[test]
    fn serialization_round_trip() {
        let r = Record::new("test", json!({"key": "value", "num": 42}));
        let bytes = r.to_bytes().unwrap();
        let r2 = Record::from_bytes(&bytes).unwrap();
        assert_eq!(r.id, r2.id);
        assert_eq!(r.table, r2.table);
        assert_eq!(r.data, r2.data);
    }

    #[test]
    fn consent_defaults_to_private_source_only() {
        let r = Record::new("decisions", json!({"summary": "x"}));
        assert_eq!(r.read_consent_raw(), json!({"kind": "private"}));
        assert_eq!(r.write_consent_raw(), json!({"kind": "source_only"}));
    }

    #[test]
    fn set_consent_round_trip() {
        let mut r = Record::new("decisions", json!({"summary": "x"}));
        r.set_consent(
            Some(json!({"kind": "workspace"})),
            Some(json!({"kind": "workspace"})),
        );
        assert_eq!(r.read_consent_raw(), json!({"kind": "workspace"}));
        assert_eq!(r.write_consent_raw(), json!({"kind": "workspace"}));

        // Round trip through bytes.
        let bytes = r.to_bytes().unwrap();
        let r2 = Record::from_bytes(&bytes).unwrap();
        assert_eq!(r2.read_consent_raw(), json!({"kind": "workspace"}));
    }
}
