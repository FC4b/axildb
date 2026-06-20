//! Auto-importance scoring —//!
//! Every record gets an `_importance` score (0.0–1.0) computed at insert time.
//! Signals: entity density, structural markers, text length/complexity.
//! Novelty scoring (vector distance) requires embeddings and runs post-insert.

use serde_json::Value;

use crate::entity::extract_entities;

/// Default importance for records that don't have enough signal.
const DEFAULT_IMPORTANCE: f32 = 0.5;

/// Pinned importance value (never decays).
pub const PINNED_IMPORTANCE: f32 = 1.0;

/// Structural keywords that indicate high-value content.
const HIGH_VALUE_MARKERS: &[&str] = &[
    "decision",
    "decided",
    "chose",
    "approach",
    "architecture",
    "breaking change",
    "migration",
    "security",
    "vulnerability",
    "root cause",
    "root_cause",
    "fix",
    "resolved",
];

/// Structural keywords that indicate lower-value content.
const LOW_VALUE_MARKERS: &[&str] = &["debug", "tmp", "temporary", "wip", "todo", "test output"];

/// Compute importance score for a record's data.
///
/// Returns a score from 0.0 to 1.0 based on:
/// - Entity density (more entities = more connected = more important)
/// - Structural markers (decisions, errors, architecture > ephemeral logs)
/// - Text complexity (longer, more detailed records score higher)
/// - Field richness (more fields = more structured = more important)
pub fn compute_importance(data: &Value) -> f32 {
    let text = crate::util::value_text(data);
    if text.is_empty() {
        return DEFAULT_IMPORTANCE;
    }

    let mut score = DEFAULT_IMPORTANCE;

    // 1. Entity density (0.0–0.2 bonus)
    let entities = extract_entities(&text);
    let entity_bonus = (entities.len() as f32 * 0.05).min(0.2);
    score += entity_bonus;

    // 2. Structural markers (±0.15)
    let text_lower = text.to_lowercase();
    let has_high_value = HIGH_VALUE_MARKERS.iter().any(|m| text_lower.contains(m));
    let has_low_value = LOW_VALUE_MARKERS.iter().any(|m| text_lower.contains(m));
    if has_high_value {
        score += 0.15;
    }
    if has_low_value {
        score -= 0.15;
    }

    // 3. Text complexity (0.0–0.1 bonus)
    let word_count = text.split_whitespace().count();
    let complexity_bonus = if word_count >= 20 {
        0.1
    } else if word_count >= 10 {
        0.05
    } else {
        0.0
    };
    score += complexity_bonus;

    // 4. Field richness (0.0–0.1 bonus)
    if let Value::Object(map) = data {
        let user_fields = map.keys().filter(|k| !k.starts_with('_')).count();
        if user_fields >= 4 {
            score += 0.1;
        } else if user_fields >= 2 {
            score += 0.05;
        }
    }

    // 5. Table-based boost: errors and decisions are inherently important
    if let Some(t) = data.get("type").and_then(|v| v.as_str()) {
        match t {
            "error" | "decision" | "architecture" => score += 0.1,
            _ => {}
        }
    }

    score.clamp(0.05, 1.0)
}

/// Get importance from a record's data, or compute default.
pub fn get_importance(data: &Value) -> f32 {
    data.get("_importance")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_IMPORTANCE)
}

/// Default half-life for importance decay (in days).
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 90.0;

/// Archive threshold — records below this effective importance are candidates for archiving.
pub const ARCHIVE_THRESHOLD: f32 = 0.1;

/// Compute effective importance after time-based decay.
///
/// `effective = base_importance * decay_factor * access_boost`
///
/// - `decay_factor = exp(-age_days * ln(2) / half_life)`
/// - `access_boost = 1.0 + ln(access_count + 1) * 0.1`
///
/// Pinned records skip decay entirely.
pub fn effective_importance(data: &Value, age_days: f64, half_life: f64) -> f32 {
    if is_pinned(data) {
        return PINNED_IMPORTANCE;
    }

    let base = get_importance(data);
    let decay = (-age_days * std::f64::consts::LN_2 / half_life).exp() as f32;

    let access_count = data
        .get("_access_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let access_boost = 1.0 + (access_count as f32 + 1.0).ln() * 0.1;

    (base * decay * access_boost).clamp(0.0, 1.0)
}

/// Apply decay to a record's data, returning the new effective importance.
///
/// Updates `_effective_importance` in the data. Does not modify `_importance` (base score).
/// Returns `None` if the record is pinned or has no importance set.
pub fn apply_decay(data: &mut Value, age_days: f64, half_life: f64) -> Option<f32> {
    if is_pinned(data) {
        return None;
    }
    let effective = effective_importance(data, age_days, half_life);
    if let Some(obj) = data.as_object_mut() {
        obj.insert(
            "_effective_importance".to_string(),
            serde_json::json!(effective),
        );
    }
    Some(effective)
}

/// Check if a record's importance is pinned (manually set to 1.0).
pub fn is_pinned(data: &Value) -> bool {
    data.get("_importance_pinned")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Breakdown of how importance was scored (for transparency).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportanceBreakdown {
    pub score: f32,
    pub entity_count: usize,
    pub entity_bonus: f32,
    pub structural_bonus: f32,
    pub complexity_bonus: f32,
    pub field_bonus: f32,
    pub type_bonus: f32,
}

/// Compute importance with a detailed breakdown.
pub fn compute_importance_breakdown(data: &Value) -> ImportanceBreakdown {
    let text = crate::util::value_text(data);
    let entities = extract_entities(&text);
    let entity_bonus = (entities.len() as f32 * 0.05).min(0.2);

    let text_lower = text.to_lowercase();
    let has_high = HIGH_VALUE_MARKERS.iter().any(|m| text_lower.contains(m));
    let has_low = LOW_VALUE_MARKERS.iter().any(|m| text_lower.contains(m));
    let structural_bonus = if has_high { 0.15 } else { 0.0 } + if has_low { -0.15 } else { 0.0 };

    let word_count = text.split_whitespace().count();
    let complexity_bonus = if word_count >= 20 {
        0.1
    } else if word_count >= 10 {
        0.05
    } else {
        0.0
    };

    let field_bonus = if let Value::Object(map) = data {
        let user_fields = map.keys().filter(|k| !k.starts_with('_')).count();
        if user_fields >= 4 {
            0.1
        } else if user_fields >= 2 {
            0.05
        } else {
            0.0
        }
    } else {
        0.0
    };

    let type_bonus = data
        .get("type")
        .and_then(|v| v.as_str())
        .map(|t| match t {
            "error" | "decision" | "architecture" => 0.1,
            _ => 0.0,
        })
        .unwrap_or(0.0);

    let score = (DEFAULT_IMPORTANCE
        + entity_bonus
        + structural_bonus
        + complexity_bonus
        + field_bonus
        + type_bonus)
        .clamp(0.05, 1.0);

    ImportanceBreakdown {
        score,
        entity_count: entities.len(),
        entity_bonus,
        structural_bonus,
        complexity_bonus,
        field_bonus,
        type_bonus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_data_gets_default() {
        let score = compute_importance(&json!({}));
        assert!((score - DEFAULT_IMPORTANCE).abs() < 0.15);
    }

    #[test]
    fn decision_scores_higher() {
        let decision = json!({
            "type": "decision",
            "summary": "Use JWT for auth tokens instead of sessions",
            "reason": "Stateless, scales horizontally, easy to revoke via short TTL",
        });
        let plain = json!({
            "summary": "Updated readme file",
        });
        assert!(compute_importance(&decision) > compute_importance(&plain));
    }

    #[test]
    fn entity_rich_text_scores_higher() {
        let rich = json!({
            "summary": "Fixed `AuthModule` timeout in `LoginService` affecting `UserController`",
        });
        let plain = json!({
            "summary": "Fixed a bug",
        });
        assert!(compute_importance(&rich) > compute_importance(&plain));
    }

    #[test]
    fn debug_text_scores_lower() {
        let debug = json!({"summary": "debug output from tmp test"});
        let normal = json!({"summary": "Implemented caching layer"});
        assert!(compute_importance(&debug) < compute_importance(&normal));
    }

    #[test]
    fn pinned_check() {
        assert!(!is_pinned(&json!({})));
        assert!(!is_pinned(&json!({"_importance_pinned": false})));
        assert!(is_pinned(&json!({"_importance_pinned": true})));
    }

    #[test]
    fn breakdown_matches_score() {
        let data = json!({
            "type": "error",
            "summary": "Connection pool exhausted under load in `AuthModule`",
            "root_cause": "Default pool size too small",
            "fix": "Increased pool to 50",
        });
        let breakdown = compute_importance_breakdown(&data);
        let score = compute_importance(&data);
        assert!((breakdown.score - score).abs() < 0.01);
    }

    #[test]
    fn score_clamped_to_range() {
        // Even with all bonuses, should not exceed 1.0
        let max = json!({
            "type": "decision",
            "summary": "Critical architecture decision about `AuthModule` security vulnerability in `LoginService` migration with breaking change",
            "reason": "Root cause was a security fix",
            "impact": "High",
            "files": ["a.rs", "b.rs"],
        });
        let score = compute_importance(&max);
        assert!(score <= 1.0);
        assert!(score >= 0.05);
    }
}
