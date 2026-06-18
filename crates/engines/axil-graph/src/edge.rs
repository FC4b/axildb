use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use axil_core::RecordId;

/// A directed edge connecting two records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique edge identifier (edges are records too).
    pub id: RecordId,
    /// Source record.
    pub from: RecordId,
    /// Target record.
    pub to: RecordId,
    /// Edge type label (e.g. "modified", "mentions", "depends_on").
    pub edge_type: String,
    /// Arbitrary JSON properties on the edge.
    pub properties: Value,
    /// When the edge was created.
    pub created_at: DateTime<Utc>,
    /// When this edge becomes valid (None = always valid). (8b.8)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    /// When this edge expires (None = never expires). (8b.8)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<DateTime<Utc>>,
}

impl Edge {
    /// Create a new edge.
    pub fn new(
        from: RecordId,
        edge_type: impl Into<String>,
        to: RecordId,
        properties: Value,
    ) -> Self {
        Self {
            id: RecordId::new(),
            from,
            to,
            edge_type: edge_type.into(),
            properties,
            created_at: Utc::now(),
            valid_from: None,
            valid_until: None,
        }
    }

    /// Check if this edge is temporally valid at a given point in time.
    pub fn is_valid_at(&self, at: &DateTime<Utc>) -> bool {
        if let Some(ref vf) = self.valid_from {
            if at < vf {
                return false;
            }
        }
        if let Some(ref vu) = self.valid_until {
            if at >= vu {
                return false;
            }
        }
        true
    }

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
    use serde_json::json;

    #[test]
    fn edge_creation() {
        let from = RecordId::new();
        let to = RecordId::new();
        let edge = Edge::new(from.clone(), "modified", to.clone(), json!({"weight": 1.0}));

        assert_eq!(edge.from, from);
        assert_eq!(edge.to, to);
        assert_eq!(edge.edge_type, "modified");
        assert_eq!(edge.properties["weight"], 1.0);
    }

    #[test]
    fn edge_serialization_round_trip() {
        let edge = Edge::new(
            RecordId::new(),
            "relates_to",
            RecordId::new(),
            json!({"weight": 0.9, "note": "test"}),
        );
        let bytes = edge.to_bytes().unwrap();
        let restored = Edge::from_bytes(&bytes).unwrap();
        assert_eq!(edge.id, restored.id);
        assert_eq!(edge.from, restored.from);
        assert_eq!(edge.to, restored.to);
        assert_eq!(edge.edge_type, restored.edge_type);
        assert_eq!(edge.created_at, restored.created_at);
        assert_eq!(edge.properties, restored.properties);
    }
}
