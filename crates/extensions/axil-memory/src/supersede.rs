//! Memory superseding — detect when new facts replace old ones.
//!
//! When a new fact is stored, check vector similarity against same-table
//! records. If similarity > threshold, mark the old record as superseded
//! and create a graph edge `new ->supersedes-> old`.

use chrono::Utc;
use serde_json::json;

use axil_core::{Axil, Record, RecordId, Result};

use crate::ttl::set_meta_field;
use crate::types::{
    EDGE_SUPERSEDES, META_RECORDED_AT, META_SUPERSEDED, META_SUPERSEDED_BY, META_VALID_FROM,
};

/// Default similarity threshold for auto-superseding.
pub const DEFAULT_SUPERSEDE_THRESHOLD: f32 = 0.92;

/// Default number of candidates to fetch when checking for superseding.
pub const DEFAULT_SUPERSEDE_CANDIDATES: usize = 50;

/// Engine for detecting and managing memory superseding.
pub struct SupersedeEngine<'a> {
    db: &'a Axil,
    threshold: f32,
}

impl<'a> SupersedeEngine<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self {
            db,
            threshold: DEFAULT_SUPERSEDE_THRESHOLD,
        }
    }

    /// Set a custom similarity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Check for and apply superseding after inserting a record.
    ///
    /// Searches for similar records in the same table. If any exceed the
    /// threshold, marks them as superseded and creates graph edges.
    ///
    /// Returns the IDs of superseded records.
    pub fn check_and_supersede(&self, new_record: &Record) -> Result<Vec<RecordId>> {
        if !self.db.has_vector_index() {
            return Ok(Vec::new());
        }

        // Search for similar records — fetch extra to account for
        // same-record match and cross-table filtering.
        let candidates = self.db.similar_to(
            &extract_text_content(new_record),
            DEFAULT_SUPERSEDE_CANDIDATES,
        );

        let candidates = match candidates {
            Ok(c) => c,
            Err(_) => return Ok(Vec::new()), // No embedder = no superseding
        };

        let mut superseded = Vec::new();

        for (candidate, similarity) in &candidates {
            // Skip self.
            if candidate.id == new_record.id {
                continue;
            }

            // Only supersede within the same table.
            if candidate.table != new_record.table {
                continue;
            }

            // Skip already superseded records.
            if candidate
                .data
                .get("_meta")
                .and_then(|m| m.get(META_SUPERSEDED))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }

            if *similarity >= self.threshold {
                // Mark old record as superseded.
                let mut old_data = candidate.data.clone();
                set_meta_field(&mut old_data, META_SUPERSEDED, json!(true));
                set_meta_field(
                    &mut old_data,
                    META_SUPERSEDED_BY,
                    json!(new_record.id.to_string()),
                );
                self.db.update(&candidate.id, old_data)?;

                // Create graph edge if graph is available.
                if self.db.has_graph_index() {
                    let _ = self
                        .db
                        .relate(&new_record.id, EDGE_SUPERSEDES, &candidate.id, None);
                }

                superseded.push(candidate.id.clone());
            }
        }

        Ok(superseded)
    }

    /// Get the history of an entity — all versions including superseded ones.
    pub fn history(&self, entity_name: &str, table: &str) -> Result<Vec<Record>> {
        // Get all records in the table matching this entity name.
        let records = self.db.list(table)?;
        let mut matches: Vec<Record> = records
            .into_iter()
            .filter(|r| {
                r.data
                    .get("entity")
                    .and_then(|v| v.as_str())
                    .map(|e| e == entity_name)
                    .unwrap_or(false)
            })
            .collect();

        // Sort by created_at ascending (oldest first) for timeline view.
        matches.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        Ok(matches)
    }

    /// Get records that this record superseded (forward traversal).
    pub fn supersede_chain(&self, id: &RecordId) -> Result<Vec<Record>> {
        if !self.db.has_graph_index() {
            return Ok(Vec::new());
        }
        // Follow supersedes edges forward to find all predecessors.
        self.db.traverse(id, &format!("->{EDGE_SUPERSEDES}"))
    }
}

/// Set bi-temporal metadata on a record's data.
///
/// - `valid_from`: when the fact became true in reality
/// - `recorded_at`: when it was stored in Axil (always now)
pub fn set_bitemporal(data: &mut serde_json::Value, valid_from: Option<chrono::DateTime<Utc>>) {
    let now = Utc::now();
    set_meta_field(data, META_RECORDED_AT, json!(now.to_rfc3339()));
    if let Some(vf) = valid_from {
        set_meta_field(data, META_VALID_FROM, json!(vf.to_rfc3339()));
    } else {
        set_meta_field(data, META_VALID_FROM, json!(now.to_rfc3339()));
    }
}

/// Extract searchable text from a record for similarity comparison.
fn extract_text_content(record: &Record) -> String {
    let mut parts = Vec::new();

    // Try common text fields.
    for field in &[
        "summary",
        "fact",
        "description",
        "content",
        "value",
        "entity",
        "name",
        "pattern_name",
    ] {
        if let Some(s) = record.data.get(field).and_then(|v| v.as_str()) {
            parts.push(s.to_string());
        }
    }

    if parts.is_empty() {
        // Fall back to the full JSON.
        serde_json::to_string(&record.data).unwrap_or_default()
    } else {
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn set_bitemporal_adds_timestamps() {
        use crate::ttl::get_meta_str;
        let mut data = json!({"fact": "test"});
        set_bitemporal(&mut data, None);
        assert!(get_meta_str(&data, META_RECORDED_AT).is_some());
        assert!(get_meta_str(&data, META_VALID_FROM).is_some());
    }

    #[test]
    fn extract_text_from_summary() {
        let r = Record::new("test", json!({"summary": "Fixed auth bug", "other": 42}));
        let text = extract_text_content(&r);
        assert!(text.contains("Fixed auth bug"));
    }

    #[test]
    fn extract_text_fallback() {
        let r = Record::new("test", json!({"x": 1, "y": 2}));
        let text = extract_text_content(&r);
        assert!(!text.is_empty());
    }
}
