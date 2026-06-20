//! Relevance feedback loop for learning from recall usage.
//!
//! Tracks which recall results are actually useful and adjusts future ranking.
//! No LLM required — pure algorithmic feedback using vector similarity to
//! match new queries against past queries that received feedback.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::record::RecordId;

/// Maximum number of feedback entries to retain.
const MAX_FEEDBACK_ENTRIES: usize = 10_000;

/// Feedback half-life in days (stale preferences fade).
const FEEDBACK_HALF_LIFE_DAYS: f64 = 30.0;

/// A single feedback entry recording that a result was useful for a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    /// The query embedding at the time of feedback.
    pub query_embedding: Vec<f32>,
    /// The record that was marked relevant.
    pub record_id: RecordId,
    /// Number of times this record was marked relevant for similar queries.
    pub count: u32,
    /// When this feedback was last reinforced.
    pub last_used: DateTime<Utc>,
    /// When this feedback was first created.
    pub created_at: DateTime<Utc>,
}

/// In-memory feedback store with persistence hooks.
pub struct FeedbackStore {
    /// Feedback entries indexed by a synthetic key.
    entries: RwLock<Vec<FeedbackEntry>>,
}

impl FeedbackStore {
    /// Create a new empty feedback store.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    /// Load from serialized bytes.
    pub fn from_bytes(data: &[u8]) -> crate::error::Result<Self> {
        let entries: Vec<FeedbackEntry> = serde_json::from_slice(data)
            .map_err(|e| crate::error::AxilError::Serialization(Box::new(e)))?;
        Ok(Self {
            entries: RwLock::new(entries),
        })
    }

    /// Serialize to bytes for persistence.
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        let entries = self.entries.read();
        serde_json::to_vec(&*entries)
            .map_err(|e| crate::error::AxilError::Serialization(Box::new(e)))
    }

    /// Record that a result was relevant for a query.
    ///
    /// If a similar query+record pair already exists (cosine similarity > 0.95),
    /// increments the count. Otherwise creates a new entry.
    pub fn mark_relevant(&self, query_embedding: &[f32], record_id: &RecordId) {
        let now = Utc::now();

        // Read-lock scan to find matching entry index
        let match_idx = {
            let entries = self.entries.read();
            entries.iter().position(|entry| {
                entry.record_id == *record_id
                    && cosine_similarity(query_embedding, &entry.query_embedding) > 0.95
            })
        };
        // Read lock released here

        // Write-lock only for the targeted mutation
        let mut entries = self.entries.write();
        if let Some(idx) = match_idx {
            if idx < entries.len() && entries[idx].record_id == *record_id {
                entries[idx].count += 1;
                entries[idx].last_used = now;
                return;
            }
        }

        // New feedback entry
        entries.push(FeedbackEntry {
            query_embedding: query_embedding.to_vec(),
            record_id: record_id.clone(),
            count: 1,
            last_used: now,
            created_at: now,
        });

        // Enforce max size — remove oldest
        if entries.len() > MAX_FEEDBACK_ENTRIES {
            entries.sort_by(|a, b| b.last_used.cmp(&a.last_used));
            entries.truncate(MAX_FEEDBACK_ENTRIES);
        }
    }

    /// Compute feedback boost for a set of candidate records given a query.
    ///
    /// For each candidate, checks if similar past queries had positive feedback
    /// for that record. Returns a map of record_id → boost score (0.0–1.0).
    pub fn compute_boosts(
        &self,
        query_embedding: &[f32],
        candidate_ids: &[RecordId],
        now: &DateTime<Utc>,
    ) -> HashMap<RecordId, f32> {
        let entries = self.entries.read();
        let mut boosts: HashMap<RecordId, f32> = HashMap::new();

        if entries.is_empty() {
            return boosts;
        }

        // Find feedback entries where the past query is similar to current query
        let similar_feedback: Vec<&FeedbackEntry> = entries
            .iter()
            .filter(|e| {
                let sim = cosine_similarity(query_embedding, &e.query_embedding);
                sim > 0.80
            })
            .collect();

        if similar_feedback.is_empty() {
            return boosts;
        }

        for candidate_id in candidate_ids {
            let mut total_boost = 0.0f32;

            for fb in &similar_feedback {
                if fb.record_id == *candidate_id {
                    let query_sim = cosine_similarity(query_embedding, &fb.query_embedding);

                    // Time decay: feedback loses strength over time
                    let age_days = (*now - fb.last_used).num_seconds().max(0) as f64 / 86400.0;
                    let time_decay = (-age_days / FEEDBACK_HALF_LIFE_DAYS).exp() as f32;

                    // Count factor: more feedback = stronger signal (diminishing returns)
                    let count_factor = (fb.count as f32).ln().max(1.0) / 5.0;

                    total_boost += query_sim * time_decay * count_factor.min(1.0);
                }
            }

            if total_boost > 0.0 {
                boosts.insert(candidate_id.clone(), total_boost.min(1.0));
            }
        }

        boosts
    }

    /// Decay old feedback entries and remove those that have fully decayed.
    pub fn decay(&self, now: &DateTime<Utc>) {
        let mut entries = self.entries.write();
        let cutoff_days = FEEDBACK_HALF_LIFE_DAYS * 5.0; // 5 half-lives ≈ 3% remaining
        entries.retain(|e| {
            let age_days = (*now - e.last_used).num_seconds().max(0) as f64 / 86400.0;
            age_days < cutoff_days
        });
    }

    /// Get the number of feedback entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the feedback store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Check if a record has any prior feedback.
    pub fn has_feedback(&self, record_id: &RecordId) -> bool {
        self.entries
            .read()
            .iter()
            .any(|e| e.record_id == *record_id)
    }

    /// Get all feedback entries for inspection/debugging.
    pub fn entries(&self) -> Vec<FeedbackEntry> {
        self.entries.read().clone()
    }
}

impl Default for FeedbackStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors — delegates to shared utility.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::util::cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        (0..8).map(|i| (seed + i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn mark_relevant_creates_entry() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid = RecordId::new();
        store.mark_relevant(&emb, &rid);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn mark_relevant_increments_count() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid = RecordId::new();
        store.mark_relevant(&emb, &rid);
        store.mark_relevant(&emb, &rid);
        let entries = store.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].count, 2);
    }

    #[test]
    fn compute_boosts_for_known_record() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid = RecordId::new();
        store.mark_relevant(&emb, &rid);

        let now = Utc::now();
        let boosts = store.compute_boosts(&emb, &[rid.clone()], &now);
        assert!(boosts.contains_key(&rid));
        assert!(*boosts.get(&rid).unwrap() > 0.0);
    }

    #[test]
    fn compute_boosts_empty_for_unknown() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid1 = RecordId::new();
        let rid2 = RecordId::new();
        store.mark_relevant(&emb, &rid1);

        let now = Utc::now();
        let boosts = store.compute_boosts(&emb, &[rid2.clone()], &now);
        assert!(boosts.is_empty());
    }

    #[test]
    fn decay_removes_old_entries() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid = RecordId::new();
        store.mark_relevant(&emb, &rid);

        // Simulate far future (200 days out)
        let future = Utc::now() + chrono::Duration::days(200);
        store.decay(&future);
        assert!(store.is_empty());
    }

    #[test]
    fn serialization_round_trip() {
        let store = FeedbackStore::new();
        let emb = make_embedding(1.0);
        let rid = RecordId::new();
        store.mark_relevant(&emb, &rid);

        let bytes = store.to_bytes().unwrap();
        let store2 = FeedbackStore::from_bytes(&bytes).unwrap();
        assert_eq!(store2.len(), 1);
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }
}
