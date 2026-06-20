//! Tiered memory model.
//!
//! Records are classified into 3 tiers based on age and activation level:
//! - Hot: active session, recent, high activation
//! - Warm: recent, moderate activation
//! - Cold: old, low activation (excluded from default recall)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::record::Record;

/// Memory tier for a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MemoryTier {
    /// Active session, recent, high activation.
    Hot = 0,
    /// Recent, moderate activation (default recall includes this).
    Warm = 1,
    /// Old, low activation (excluded from default recall).
    Cold = 2,
    /// Very old, low importance after decay (only returned with --include-archived).
    Archived = 3,
}

impl MemoryTier {
    /// Decode from a stored u8.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Hot,
            1 => Self::Warm,
            2 => Self::Cold,
            _ => Self::Archived,
        }
    }

    /// Whether this tier is included in default (non-exhaustive) recall.
    pub fn included_in_default_recall(&self) -> bool {
        matches!(self, Self::Hot | Self::Warm)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Archived => "archived",
        }
    }
}

/// Configuration for tier classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Maximum age in days for Hot tier (default: 7).
    pub hot_max_age_days: f64,
    /// Maximum age in days for Warm tier (default: 30).
    pub warm_max_age_days: f64,
    /// Minimum activation level for Hot tier (default: 0.5).
    pub hot_min_activation: f32,
    /// Minimum activation level for Warm tier (default: 0.1).
    pub warm_min_activation: f32,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_max_age_days: 7.0,
            warm_max_age_days: 30.0,
            hot_min_activation: 0.5,
            warm_min_activation: 0.1,
        }
    }
}

/// Classify a record into a memory tier based on age, activation, and importance.
///
/// `decay` is consulted for the per-table half-life used in the Archived
/// check. Pass `None` to use the system default.
pub fn classify_tier(
    record: &Record,
    config: &TierConfig,
    now: &DateTime<Utc>,
    decay: Option<&crate::config::DecayConfig>,
) -> MemoryTier {
    let age_days = (*now - record.created_at).num_seconds().max(0) as f64 / 86400.0;
    let activation = crate::activation::get_activation(record);

    // Pinned records are always Hot.
    if crate::importance::is_pinned(&record.data) {
        return MemoryTier::Hot;
    }

    // Check effective importance for Archived tier. Half-life is resolved
    // per-table when `decay` is provided — errors decay faster, preferences
    // slower — so tiering matches the project's actual memory semantics.
    let half_life = decay
        .map(|d| d.half_life_for(&record.table))
        .unwrap_or(crate::importance::DEFAULT_HALF_LIFE_DAYS);
    let effective_importance =
        crate::importance::effective_importance(&record.data, age_days, half_life);
    if effective_importance < crate::importance::ARCHIVE_THRESHOLD {
        return MemoryTier::Archived;
    }

    if age_days <= config.hot_max_age_days && activation >= config.hot_min_activation {
        MemoryTier::Hot
    } else if age_days <= config.warm_max_age_days && activation >= config.warm_min_activation {
        MemoryTier::Warm
    } else {
        MemoryTier::Cold
    }
}

/// Compute tier distribution for a set of records.
pub fn tier_distribution(
    records: &[Record],
    config: &TierConfig,
    now: &DateTime<Utc>,
    decay: Option<&crate::config::DecayConfig>,
) -> TierStats {
    let mut hot = 0usize;
    let mut warm = 0usize;
    let mut cold = 0usize;
    let mut archived = 0usize;

    for r in records {
        match classify_tier(r, config, now, decay) {
            MemoryTier::Hot => hot += 1,
            MemoryTier::Warm => warm += 1,
            MemoryTier::Cold => cold += 1,
            MemoryTier::Archived => archived += 1,
        }
    }

    TierStats {
        hot,
        warm,
        cold,
        archived,
        total: records.len(),
    }
}

/// Summary of tier distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierStats {
    pub hot: usize,
    pub warm: usize,
    pub cold: usize,
    pub archived: usize,
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_record_is_hot() {
        let r = Record::new("test", json!({"summary": "recent"}));
        let tier = classify_tier(&r, &TierConfig::default(), &Utc::now(), None);
        assert_eq!(tier, MemoryTier::Hot);
    }

    #[test]
    fn old_low_activation_is_cold() {
        let mut r = Record::new("test", json!({"_activation": 0.05}));
        r.created_at = Utc::now() - chrono::Duration::days(60);
        let tier = classify_tier(&r, &TierConfig::default(), &Utc::now(), None);
        assert_eq!(tier, MemoryTier::Cold);
    }

    #[test]
    fn medium_age_is_warm() {
        let mut r = Record::new("test", json!({"_activation": 0.3}));
        r.created_at = Utc::now() - chrono::Duration::days(15);
        let tier = classify_tier(&r, &TierConfig::default(), &Utc::now(), None);
        assert_eq!(tier, MemoryTier::Warm);
    }

    #[test]
    fn tier_included_in_recall() {
        assert!(MemoryTier::Hot.included_in_default_recall());
        assert!(MemoryTier::Warm.included_in_default_recall());
        assert!(!MemoryTier::Cold.included_in_default_recall());
    }

    #[test]
    fn distribution_counts() {
        let now = Utc::now();
        let config = TierConfig::default();
        let records = vec![
            Record::new("t", json!({"_activation": 1.0})),
            {
                let mut r = Record::new("t", json!({"_activation": 0.3}));
                r.created_at = now - chrono::Duration::days(15);
                r
            },
            {
                let mut r = Record::new("t", json!({"_activation": 0.01}));
                r.created_at = now - chrono::Duration::days(60);
                r
            },
        ];
        let stats = tier_distribution(&records, &config, &now, None);
        assert_eq!(stats.hot, 1);
        assert_eq!(stats.warm, 1);
        assert_eq!(stats.cold, 1);
    }
}
