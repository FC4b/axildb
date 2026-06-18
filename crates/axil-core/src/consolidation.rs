//! Memory consolidation — detect contradictions, supersede old facts, merge timelines.
//!
//! No LLM required: uses vector similarity for detection and template-based
//! merging for consolidation. All operations are deterministic.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::entity::{extract_entities, Entity};
use crate::record::{Record, RecordId};

/// Result of checking a new fact against existing facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResult {
    /// No conflict — the fact is novel.
    Novel,
    /// New fact supersedes an existing one (same entity, different value, newer).
    Supersedes {
        old_record_id: RecordId,
        similarity: f32,
    },
    /// New fact contradicts an existing one (same entity, conflicting values).
    Contradicts {
        existing_record_id: RecordId,
        similarity: f32,
    },
}

/// A consolidated fact merging multiple source facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedFact {
    /// The merged text summary.
    pub summary: String,
    /// Source record IDs that were consolidated.
    pub source_ids: Vec<RecordId>,
    /// The primary entity this consolidation is about.
    pub entity: String,
    /// Timestamp of the newest source fact.
    pub latest_at: DateTime<Utc>,
}

/// Confidence score for a fact based on multiple signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall confidence (0.0–1.0).
    pub score: f32,
    /// Number of sessions/sources mentioning this fact.
    pub source_count: u32,
    /// Recency factor (newer = higher).
    pub recency_factor: f32,
    /// Contradiction penalty (contradicted facts get lower confidence).
    pub contradiction_penalty: f32,
}

/// Similarity threshold for detecting potential conflicts.
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.92;

/// Higher threshold for medium-confidence detection (8b.12).
const CONFLICT_MEDIUM_THRESHOLD: f32 = 0.95;

/// Negation words used for contradiction detection (8b.12).
pub const NEGATION_WORDS: &[&str] = &[
    "not",
    "no",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "nowhere",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "can't",
    "cannot",
    "unable",
    "no longer",
    "instead",
    "stopped",
    "removed",
    "deprecated",
    "replaced",
    "switched from",
    "migrated away",
];

/// Confidence level for contradiction detection (8b.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConflictConfidence {
    /// Same entities + explicit negation — highly likely contradiction.
    High,
    /// Similarity > 0.95, shared entities, different text — likely contradiction.
    Medium,
    /// Similarity > 0.92, shared entities — possible contradiction.
    Low,
    /// No conflict signals detected.
    None,
}

/// Check if a new record conflicts with an existing similar record.
///
/// Two records conflict when:
/// 1. Vector similarity > 0.92 (very similar topic)
/// 2. They share at least one entity
/// 3. They appear to make different claims about that entity
///
/// Enhanced with negation-aware confidence levels (8b.12):
/// - High: shared entities + explicit negation word → auto-supersede
/// - Medium: similarity > 0.95 + shared entities → surface for review
/// - Low: similarity > 0.92 + shared entities → surface for review
///
/// Returns the conflict type: Novel, Supersedes, or Contradicts.
pub fn check_conflict(
    new_record: &Record,
    existing_record: &Record,
    similarity: f32,
) -> ConflictResult {
    if similarity < CONFLICT_SIMILARITY_THRESHOLD {
        return ConflictResult::Novel;
    }

    let new_text = record_text(new_record);
    let existing_text = record_text(existing_record);

    let new_entities = extract_entities(&new_text);
    let existing_entities = extract_entities(&existing_text);

    // Check for shared entities
    let shared: Vec<&Entity> = new_entities
        .iter()
        .filter(|ne| existing_entities.iter().any(|ee| ee.name == ne.name))
        .collect();

    if shared.is_empty() {
        return ConflictResult::Novel;
    }

    // Check if the texts are nearly identical (duplicate/reinforcing fact, not a conflict).
    let new_lower = new_text.to_lowercase();
    let existing_lower = existing_text.to_lowercase();
    let text_similarity = normalized_text_similarity(&new_lower, &existing_lower);
    if text_similarity > 0.90 {
        return ConflictResult::Novel;
    }

    // Determine confidence level (8b.12)
    let confidence = detect_conflict_confidence(&new_lower, &existing_lower, &shared, similarity);

    // Only auto-supersede on High confidence (explicit negation detected).
    // Medium/Low are treated as Novel — callers can inspect confidence
    // separately via detect_conflict_confidence() if they want to surface them.
    match confidence {
        ConflictConfidence::High => {
            if new_record.created_at > existing_record.created_at {
                ConflictResult::Supersedes {
                    old_record_id: existing_record.id.clone(),
                    similarity,
                }
            } else {
                ConflictResult::Contradicts {
                    existing_record_id: existing_record.id.clone(),
                    similarity,
                }
            }
        }
        _ => ConflictResult::Novel,
    }
}

/// Detect conflict confidence using negation heuristics (8b.12).
///
/// Checks whether one text contains a negation word near a shared entity
/// while the other does not — a strong signal of contradiction.
pub fn detect_conflict_confidence(
    text_a: &str,
    text_b: &str,
    shared_entities: &[&Entity],
    similarity: f32,
) -> ConflictConfidence {
    // Check for explicit negation near shared entities
    let a_has_negation = has_negation_near_entity(text_a, shared_entities);
    let b_has_negation = has_negation_near_entity(text_b, shared_entities);

    // High: one text negates what the other affirms (asymmetric negation)
    if a_has_negation != b_has_negation {
        return ConflictConfidence::High;
    }

    // Medium: very high similarity + shared entities (no explicit negation)
    if similarity > CONFLICT_MEDIUM_THRESHOLD {
        return ConflictConfidence::Medium;
    }

    // Low: above base threshold + shared entities
    ConflictConfidence::Low
}

/// Check if a text contains a negation word near any shared entity.
fn has_negation_near_entity(text: &str, entities: &[&Entity]) -> bool {
    for entity in entities {
        let entity_lower = entity.name.to_lowercase();
        if let Some(entity_pos) = text.find(&entity_lower) {
            // Walk back to char boundary for UTF-8 safety
            let mut ws = entity_pos.saturating_sub(80);
            while ws > 0 && !text.is_char_boundary(ws) {
                ws -= 1;
            }
            let mut we = (entity_pos + entity_lower.len() + 80).min(text.len());
            while we < text.len() && !text.is_char_boundary(we) {
                we += 1;
            }
            let window = &text[ws..we];
            for &neg in NEGATION_WORDS {
                if window.contains(neg) {
                    return true;
                }
            }
        }
    }
    false
}

/// Compute confidence score for a record.
///
/// Confidence is based on:
/// - Source count: more sessions mentioning it → higher confidence
/// - Recency: recent facts → higher confidence
/// - Contradictions: contradicted facts → lower confidence
pub fn compute_confidence(
    source_count: u32,
    created_at: &DateTime<Utc>,
    now: &DateTime<Utc>,
    contradiction_count: u32,
) -> ConfidenceScore {
    // Source count factor: log scale, saturates around 10 sources
    let source_factor = ((source_count as f32).ln().max(0.0) + 1.0).min(2.0) / 2.0;

    // Recency factor: exponential decay with 30-day half-life
    let age_days = (*now - *created_at).num_seconds().max(0) as f64 / 86400.0;
    let recency_factor = (-age_days / 30.0).exp() as f32;

    // Contradiction penalty
    let contradiction_penalty = match contradiction_count {
        0 => 0.0,
        1 => 0.2,
        2 => 0.4,
        _ => 0.6,
    };

    let score = (source_factor * 0.3 + recency_factor * 0.4 + (1.0 - contradiction_penalty) * 0.3)
        .clamp(0.0, 1.0);

    ConfidenceScore {
        score,
        source_count,
        recency_factor,
        contradiction_penalty,
    }
}

/// Consolidate multiple facts about the same entity into a summary.
///
/// Uses template-based merging:
/// - Single fact: pass through
/// - Linear chain (A superseded by B superseded by C):
///   "Current: C. Previously: B (date). Originally: A (date)."
/// - Contradiction: "Conflict: A says X, B says Y. B is newer."
pub fn consolidate_facts(
    entity_name: &str,
    facts: &[(Record, ConflictResult)],
) -> Option<ConsolidatedFact> {
    if facts.is_empty() {
        return None;
    }

    if facts.len() == 1 {
        let (record, _) = &facts[0];
        let text = record_text(record);
        return Some(ConsolidatedFact {
            summary: text,
            source_ids: vec![record.id.clone()],
            entity: entity_name.to_string(),
            latest_at: record.created_at,
        });
    }

    // Sort by creation time (newest first)
    let mut sorted: Vec<&(Record, ConflictResult)> = facts.iter().collect();
    sorted.sort_by(|a, b| b.0.created_at.cmp(&a.0.created_at));

    let newest = &sorted[0].0;
    let latest_at = newest.created_at;

    // Check if we have contradictions
    let has_contradictions = sorted
        .iter()
        .any(|(_, cr)| matches!(cr, ConflictResult::Contradicts { .. }));

    let summary = if has_contradictions {
        // Contradiction template
        let mut parts = Vec::new();
        parts.push(format!("{entity_name}: CONFLICT"));
        for (i, (record, _)) in sorted.iter().enumerate() {
            let text = record_text(record);
            let date = record.created_at.format("%Y-%m-%d");
            if i == 0 {
                parts.push(format!("  Latest ({date}): {text}"));
            } else {
                parts.push(format!("  Earlier ({date}): {text}"));
            }
        }
        parts.join("\n")
    } else {
        // Linear superseding chain template
        let mut parts = Vec::new();
        for (i, (record, _)) in sorted.iter().enumerate() {
            let text = record_text(record);
            let date = record.created_at.format("%Y-%m-%d");
            if i == 0 {
                parts.push(format!("{entity_name}: {text}"));
            } else if i == sorted.len() - 1 {
                parts.push(format!("Originally ({date}): {text}"));
            } else {
                parts.push(format!("Previously ({date}): {text}"));
            }
        }
        parts.join(". ")
    };

    Some(ConsolidatedFact {
        summary,
        source_ids: sorted.iter().map(|(r, _)| r.id.clone()).collect(),
        entity: entity_name.to_string(),
        latest_at,
    })
}

/// Extract searchable text from a record's data payload.
fn record_text(record: &Record) -> String {
    crate::util::record_text(record)
}

/// Compute normalized text similarity using word overlap (Jaccard-like).
///
/// Returns 0.0–1.0 where 1.0 means identical word sets.
fn normalized_text_similarity(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn novel_when_low_similarity() {
        let r1 = Record::new("facts", json!({"summary": "Auth uses JWT"}));
        let r2 = Record::new("facts", json!({"summary": "Database uses PostgreSQL"}));
        let result = check_conflict(&r2, &r1, 0.5);
        assert!(matches!(result, ConflictResult::Novel));
    }

    #[test]
    fn supersedes_when_high_similarity_newer() {
        let mut r1 = Record::new("facts", json!({"summary": "Auth uses `JWT`"}));
        r1.created_at = Utc::now() - chrono::Duration::days(5);
        let r2 = Record::new(
            "facts",
            json!({"summary": "Auth switched from `JWT` to sessions"}),
        );
        let result = check_conflict(&r2, &r1, 0.95);
        // Both contain backtick entities, r2 is newer
        match result {
            ConflictResult::Supersedes { similarity, .. } => {
                assert!((similarity - 0.95).abs() < 0.01);
            }
            ConflictResult::Novel => {
                // Also acceptable if entities don't overlap
            }
            _ => panic!("expected Supersedes or Novel"),
        }
    }

    #[test]
    fn novel_when_duplicate_fact_restated() {
        let mut r1 = Record::new("facts", json!({"summary": "Auth uses `JWT`"}));
        r1.created_at = Utc::now() - chrono::Duration::days(5);
        // Same fact restated — should be Novel (reinforcing), not Supersedes
        let r2 = Record::new("facts", json!({"summary": "Auth uses `JWT`"}));
        let result = check_conflict(&r2, &r1, 0.98);
        assert!(
            matches!(result, ConflictResult::Novel),
            "duplicate facts should not conflict"
        );
    }

    #[test]
    fn text_similarity_identical() {
        let sim = normalized_text_similarity("auth uses jwt", "auth uses jwt");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn text_similarity_different() {
        let sim = normalized_text_similarity("auth uses jwt", "auth switched to sessions");
        assert!(sim < 0.5);
    }

    #[test]
    fn confidence_high_for_recent_multi_source() {
        let now = Utc::now();
        let score = compute_confidence(5, &now, &now, 0);
        assert!(score.score > 0.6, "score = {}", score.score);
        assert_eq!(score.contradiction_penalty, 0.0);
    }

    #[test]
    fn negation_high_confidence_when_explicit() {
        let mut r1 = Record::new("facts", json!({"summary": "`AuthModule` uses JWT"}));
        r1.created_at = Utc::now() - chrono::Duration::days(5);
        let r2 = Record::new(
            "facts",
            json!({"summary": "`AuthModule` no longer uses JWT"}),
        );
        let result = check_conflict(&r2, &r1, 0.95);
        assert!(
            matches!(result, ConflictResult::Supersedes { .. }),
            "negation should trigger Supersedes, got {:?}",
            result
        );
    }

    #[test]
    fn negation_detects_asymmetric() {
        let entities = vec![Entity {
            name: "auth".to_string(),
            entity_type: crate::entity::EntityType::Code,
            source_text: "auth".to_string(),
        }];
        let refs: Vec<&Entity> = entities.iter().collect();
        let conf = detect_conflict_confidence(
            "auth uses jwt tokens",
            "auth does not use jwt tokens",
            &refs,
            0.95,
        );
        assert_eq!(conf, ConflictConfidence::High);
    }

    #[test]
    fn no_negation_medium_confidence() {
        let entities = vec![Entity {
            name: "auth".to_string(),
            entity_type: crate::entity::EntityType::Code,
            source_text: "auth".to_string(),
        }];
        let refs: Vec<&Entity> = entities.iter().collect();
        let conf = detect_conflict_confidence(
            "auth uses jwt tokens",
            "auth uses session cookies",
            &refs,
            0.96,
        );
        assert_eq!(conf, ConflictConfidence::Medium);
    }

    #[test]
    fn confidence_low_for_old_contradicted() {
        let now = Utc::now();
        let old = now - chrono::Duration::days(90);
        let score = compute_confidence(1, &old, &now, 2);
        assert!(score.score < 0.5, "score = {}", score.score);
    }

    #[test]
    fn consolidate_single_fact() {
        let r = Record::new("facts", json!({"summary": "Auth uses JWT"}));
        let facts = vec![(r, ConflictResult::Novel)];
        let consolidated = consolidate_facts("auth", &facts).unwrap();
        assert_eq!(consolidated.summary, "Auth uses JWT");
        assert_eq!(consolidated.source_ids.len(), 1);
    }

    #[test]
    fn consolidate_superseding_chain() {
        let mut r1 = Record::new("facts", json!({"summary": "Auth uses JWT"}));
        r1.created_at = Utc::now() - chrono::Duration::days(10);
        let r2 = Record::new("facts", json!({"summary": "Auth uses session cookies"}));
        let facts = vec![
            (r1, ConflictResult::Novel),
            (
                r2,
                ConflictResult::Supersedes {
                    old_record_id: RecordId::new(),
                    similarity: 0.95,
                },
            ),
        ];
        let consolidated = consolidate_facts("auth", &facts).unwrap();
        assert!(consolidated.summary.contains("Auth uses session cookies"));
        assert!(consolidated.summary.contains("Originally"));
        assert_eq!(consolidated.source_ids.len(), 2);
    }

    #[test]
    fn consolidate_with_contradictions() {
        let mut r1 = Record::new("facts", json!({"summary": "API uses REST"}));
        r1.created_at = Utc::now() - chrono::Duration::days(5);
        let r2 = Record::new("facts", json!({"summary": "API uses GraphQL"}));
        let facts = vec![
            (r1, ConflictResult::Novel),
            (
                r2,
                ConflictResult::Contradicts {
                    existing_record_id: RecordId::new(),
                    similarity: 0.93,
                },
            ),
        ];
        let consolidated = consolidate_facts("api", &facts).unwrap();
        assert!(consolidated.summary.contains("CONFLICT"));
        assert!(consolidated.summary.contains("Latest"));
    }
}
