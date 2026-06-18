use serde::{Deserialize, Serialize};

use axil_core::RecordId;

/// A time-indexed entry linking a record to its timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeEntry {
    /// The record this entry indexes.
    pub record_id: RecordId,
    /// Table the record belongs to.
    pub table: String,
    /// Record creation time in microseconds since epoch.
    pub created_at_us: i64,
    /// Record last-update time in microseconds since epoch.
    pub updated_at_us: i64,
}

impl TimeEntry {
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let entry = TimeEntry {
            record_id: RecordId::new(),
            table: "sessions".to_string(),
            created_at_us: 1_000_000,
            updated_at_us: 2_000_000,
        };
        let bytes = entry.to_bytes().unwrap();
        let restored = TimeEntry::from_bytes(&bytes).unwrap();
        assert_eq!(entry.record_id, restored.record_id);
        assert_eq!(entry.table, restored.table);
        assert_eq!(entry.created_at_us, restored.created_at_us);
        assert_eq!(entry.updated_at_us, restored.updated_at_us);
    }
}
