//! Activation-level scoring for memory records.
//!
//! Implements usage-based priority: frequently accessed memories rank higher
//! than stale ones. Activation decays over time following a half-life formula:
//!
//! ```text
//! activation *= 0.5^(days_since_access / half_life)
//! ```
//!
//! Source: A-MEM (2025), MemPalace memory decay patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::record::Record;

/// Default activation level for new records.
pub const DEFAULT_ACTIVATION: f32 = 1.0;

/// Default boost applied on each read/query hit.
pub const DEFAULT_BOOST: f32 = 0.1;

/// Default half-life in days for activation decay.
pub const DEFAULT_HALF_LIFE_DAYS: f64 = 30.0;

/// Default weight for activation in the recall scoring formula.
pub const DEFAULT_ACTIVATION_WEIGHT: f32 = 0.1;

/// Configuration for activation-level scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    /// Boost applied on each read/query hit.
    pub boost: f32,
    /// Half-life in days for activation decay.
    pub half_life_days: f64,
    /// Weight in the recall scoring formula.
    pub activation_weight: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            boost: DEFAULT_BOOST,
            half_life_days: DEFAULT_HALF_LIFE_DAYS,
            activation_weight: DEFAULT_ACTIVATION_WEIGHT,
        }
    }
}

/// Read the current activation level from a record's data.
///
/// Returns `DEFAULT_ACTIVATION` if no activation field is set (backward compat).
pub fn get_activation(record: &Record) -> f32 {
    record
        .data
        .get("_activation")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_ACTIVATION)
}

/// Read the last accessed timestamp from a record's data.
pub fn get_last_accessed(record: &Record) -> Option<DateTime<Utc>> {
    record
        .data
        .get("_last_accessed")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<DateTime<Utc>>().ok())
}

/// Compute the decayed activation level at a given point in time.
///
/// Formula: `activation * 0.5^(days_since_access / half_life)`
///
/// If the record has never been accessed, uses `created_at` as the reference time.
pub fn decayed_activation(record: &Record, now: &DateTime<Utc>, half_life_days: f64) -> f32 {
    let raw_activation = get_activation(record);
    let reference_time = get_last_accessed(record).unwrap_or(record.created_at);

    let days_since = (*now - reference_time).num_seconds().max(0) as f64 / 86400.0;
    let decay_factor = (0.5_f64).powf(days_since / half_life_days);

    (raw_activation as f64 * decay_factor) as f32
}

/// Compute the activation boost for recall scoring.
///
/// Returns: `activation * weight` (additive signal)
pub fn activation_boost(record: &Record, now: &DateTime<Utc>, config: &ActivationConfig) -> f32 {
    let activation = decayed_activation(record, now, config.half_life_days);
    activation * config.activation_weight
}

/// Build the updated data fields for an activation bump.
///
/// Returns `(new_activation, timestamp)` values to merge into record.data.
/// The caller is responsible for actually writing these to the record.
pub fn compute_bump(
    record: &Record,
    now: &DateTime<Utc>,
    config: &ActivationConfig,
) -> (f32, String) {
    let current = decayed_activation(record, now, config.half_life_days);
    let new_activation = (current + config.boost).min(10.0); // cap at 10.0
    let timestamp = now.to_rfc3339();
    (new_activation, timestamp)
}

/// Statistics about activation distribution in a set of records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    /// Number of records with activation data.
    pub records_with_activation: usize,
    /// Number of records without activation data (default activation).
    pub records_without_activation: usize,
    /// Minimum decayed activation.
    pub min: f32,
    /// Maximum decayed activation.
    pub max: f32,
    /// Mean decayed activation.
    pub mean: f32,
    /// Distribution buckets: [0-0.2, 0.2-0.5, 0.5-1.0, 1.0-2.0, 2.0+]
    pub distribution: ActivationDistribution,
}

/// Activation level distribution buckets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationDistribution {
    pub very_low: usize,  // 0.0 - 0.2
    pub low: usize,       // 0.2 - 0.5
    pub medium: usize,    // 0.5 - 1.0
    pub high: usize,      // 1.0 - 2.0
    pub very_high: usize, // 2.0+
}

/// Compute activation statistics for a set of records.
pub fn compute_stats(
    records: &[Record],
    now: &DateTime<Utc>,
    half_life_days: f64,
) -> ActivationStats {
    let mut with_activation = 0usize;
    let mut without_activation = 0usize;
    let mut activations = Vec::with_capacity(records.len());
    let mut dist = ActivationDistribution {
        very_low: 0,
        low: 0,
        medium: 0,
        high: 0,
        very_high: 0,
    };

    for record in records {
        let has_field = record.data.get("_activation").is_some();
        if has_field {
            with_activation += 1;
        } else {
            without_activation += 1;
        }

        let activation = decayed_activation(record, now, half_life_days);
        activations.push(activation);

        match activation {
            a if a < 0.2 => dist.very_low += 1,
            a if a < 0.5 => dist.low += 1,
            a if a < 1.0 => dist.medium += 1,
            a if a < 2.0 => dist.high += 1,
            _ => dist.very_high += 1,
        }
    }

    let (min, max, mean) = if activations.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let min = activations.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = activations
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = activations.iter().sum();
        let mean = sum / activations.len() as f32;
        (min, max, mean)
    };

    ActivationStats {
        records_with_activation: with_activation,
        records_without_activation: without_activation,
        min,
        max,
        mean,
        distribution: dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    #[test]
    fn default_activation_for_new_record() {
        let record = Record::new("test", json!({"summary": "hello"}));
        assert!((get_activation(&record) - DEFAULT_ACTIVATION).abs() < 0.001);
    }

    #[test]
    fn explicit_activation_is_read() {
        let record = Record::new("test", json!({"_activation": 2.5}));
        assert!((get_activation(&record) - 2.5).abs() < 0.001);
    }

    #[test]
    fn decay_at_half_life() {
        let now = Utc::now();
        let mut record = Record::new("test", json!({"_activation": 1.0}));
        let thirty_days_ago = now - Duration::days(30);
        record.data.as_object_mut().unwrap().insert(
            "_last_accessed".to_string(),
            json!(thirty_days_ago.to_rfc3339()),
        );

        let decayed = decayed_activation(&record, &now, DEFAULT_HALF_LIFE_DAYS);
        assert!((decayed - 0.5).abs() < 0.05, "decayed = {decayed}");
    }

    #[test]
    fn decay_at_zero_time() {
        let now = Utc::now();
        let record = Record::new("test", json!({"_activation": 1.0}));
        let decayed = decayed_activation(&record, &now, DEFAULT_HALF_LIFE_DAYS);
        assert!((decayed - 1.0).abs() < 0.05, "decayed = {decayed}");
    }

    #[test]
    fn bump_increases_activation() {
        let now = Utc::now();
        let record = Record::new("test", json!({"_activation": 1.0}));
        let config = ActivationConfig::default();
        let (new_act, _ts) = compute_bump(&record, &now, &config);
        assert!(new_act > 1.0);
        assert!((new_act - 1.1).abs() < 0.05);
    }

    #[test]
    fn bump_capped_at_max() {
        let now = Utc::now();
        let record = Record::new("test", json!({"_activation": 9.95}));
        let config = ActivationConfig::default();
        let (new_act, _ts) = compute_bump(&record, &now, &config);
        assert!(new_act <= 10.0);
    }

    #[test]
    fn stats_empty_records() {
        let stats = compute_stats(&[], &Utc::now(), DEFAULT_HALF_LIFE_DAYS);
        assert_eq!(stats.records_with_activation, 0);
        assert_eq!(stats.records_without_activation, 0);
    }

    #[test]
    fn stats_mixed_records() {
        let now = Utc::now();
        let r1 = Record::new("test", json!({"_activation": 0.1}));
        let r2 = Record::new("test", json!({"summary": "no activation"}));
        let r3 = Record::new("test", json!({"_activation": 3.0}));

        let stats = compute_stats(&[r1, r2, r3], &now, DEFAULT_HALF_LIFE_DAYS);
        assert_eq!(stats.records_with_activation, 2);
        assert_eq!(stats.records_without_activation, 1);
        assert!(stats.min < stats.max);
    }

    #[test]
    fn activation_boost_formula() {
        let now = Utc::now();
        let record = Record::new("test", json!({"_activation": 2.0}));
        let config = ActivationConfig::default();
        let boost = activation_boost(&record, &now, &config);
        // boost = activation(~2.0) * weight(0.1) = ~0.2
        assert!((boost - 0.2).abs() < 0.05, "boost = {boost}");
    }
}
