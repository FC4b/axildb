//! Deferred detectors — embedding drift, stale sessions, slow queries, storage growth.
//!
//! Each detector runs a lightweight check and returns a `DetectorResult`.
//! Designed to be called from the worker or maintenance thread.

use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::Axil;

/// Result from a single detector run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResult {
    /// Detector name.
    pub name: String,
    /// Whether the detector found an issue.
    pub triggered: bool,
    /// Human-readable detail.
    pub detail: String,
    /// Severity: "info", "warning", "error".
    pub severity: String,
}

/// Run all detectors and return their results.
pub fn run_all_detectors(db: &Axil) -> Vec<DetectorResult> {
    let mut results = Vec::new();
    results.push(detect_stale_sessions(db));
    results.push(detect_slow_query_patterns(db));
    results.push(detect_storage_growth(db));
    if let Some(r) = detect_embedding_drift(db) {
        results.push(r);
    }
    results
}

/// Detect sessions that have been active for too long (likely abandoned).
///
/// Sessions active for more than 24 hours are flagged as potentially stale.
pub fn detect_stale_sessions(db: &Axil) -> DetectorResult {
    let cutoff = Utc::now() - Duration::hours(24);
    let sessions = match db.list("_sessions") {
        Ok(s) => s,
        Err(_) => {
            return DetectorResult {
                name: "stale_sessions".into(),
                triggered: false,
                detail: "no sessions table".into(),
                severity: "info".into(),
            };
        }
    };

    let stale: Vec<_> = sessions
        .iter()
        .filter(|r| {
            r.data.get("status").and_then(|v| v.as_str()) == Some("active") && r.created_at < cutoff
        })
        .collect();

    let triggered = !stale.is_empty();
    DetectorResult {
        name: "stale_sessions".into(),
        triggered,
        detail: if triggered {
            format!(
                "{} session(s) active for >24h (oldest: {})",
                stale.len(),
                stale
                    .iter()
                    .map(|r| r.created_at)
                    .min()
                    .unwrap_or_else(Utc::now)
                    .format("%Y-%m-%d %H:%M")
            )
        } else {
            "no stale sessions".into()
        },
        severity: if triggered { "warning" } else { "info" }.into(),
    }
}

/// Detect slow query patterns from metrics.
///
/// Checks the metrics slow query log for queries taking >100ms.
pub fn detect_slow_query_patterns(db: &Axil) -> DetectorResult {
    let slow_queries = db.slow_queries(Some(50), None);
    let count = slow_queries.len();

    DetectorResult {
        name: "slow_queries".into(),
        triggered: count > 0,
        detail: if count > 0 {
            let worst = slow_queries
                .iter()
                .map(|q| q.duration_ms)
                .reduce(f64::max)
                .unwrap_or(0.0);
            format!("{count} slow queries recorded (worst: {worst:.0}ms)")
        } else {
            "no slow queries".into()
        },
        severity: if count >= 10 { "warning" } else { "info" }.into(),
    }
}

/// Detect storage growth rate by comparing current size to stored baselines.
///
/// Stores the current size in `_detector_baselines` and compares to the
/// last stored value.
pub fn detect_storage_growth(db: &Axil) -> DetectorResult {
    let current_size = db.info().map(|i| i.total_size).unwrap_or(0);

    // Get last baseline.
    let baselines = db.list("_detector_baselines").unwrap_or_default();
    let last_size_entry = baselines
        .iter()
        .filter(|r| r.data.get("detector").and_then(|v| v.as_str()) == Some("storage_growth"))
        .max_by_key(|r| r.created_at);

    let (triggered, detail) = if let Some(baseline) = last_size_entry {
        let prev_size = baseline
            .data
            .get("size_bytes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let hours_elapsed = (Utc::now() - baseline.created_at).num_hours().max(1) as f64;
        let growth = current_size.saturating_sub(prev_size);
        let growth_rate_mb_day = (growth as f64 / 1_048_576.0) / (hours_elapsed / 24.0);

        if growth_rate_mb_day > 10.0 {
            (
                true,
                format!(
                    "growing at {:.1} MB/day ({} → {} bytes in {:.0}h)",
                    growth_rate_mb_day, prev_size, current_size, hours_elapsed
                ),
            )
        } else {
            (
                false,
                format!(
                    "growth rate: {:.2} MB/day ({} bytes total)",
                    growth_rate_mb_day, current_size
                ),
            )
        }
    } else {
        (false, format!("first measurement: {} bytes", current_size))
    };

    // Store current baseline.
    let _ = db.insert(
        "_detector_baselines",
        serde_json::json!({
            "detector": "storage_growth",
            "size_bytes": current_size,
            "measured_at": Utc::now().to_rfc3339(),
        }),
    );

    DetectorResult {
        name: "storage_growth".into(),
        triggered,
        detail,
        severity: if triggered { "warning" } else { "info" }.into(),
    }
}

/// Detect embedding drift — vectors indexed with a different model than current.
///
/// Returns `None` if no vector index is present.
pub fn detect_embedding_drift(db: &Axil) -> Option<DetectorResult> {
    if !db.has_vector_index() {
        return None;
    }

    // Check the model metadata stored in _config table.
    let config_records = db.list("_config").unwrap_or_default();
    let stored_model = config_records
        .iter()
        .find(|r| r.data.get("key").and_then(|v| v.as_str()) == Some("embedding_model"))
        .and_then(|r| r.data.get("value").and_then(|v| v.as_str()))
        .map(String::from);

    let current_model = config_records
        .iter()
        .find(|r| r.data.get("key").and_then(|v| v.as_str()) == Some("current_embedding_model"))
        .and_then(|r| r.data.get("value").and_then(|v| v.as_str()))
        .map(String::from);

    let (triggered, detail) = match (&stored_model, &current_model) {
        (Some(stored), Some(current)) if stored != current => (
            true,
            format!(
                "vectors indexed with '{}' but current model is '{}'",
                stored, current
            ),
        ),
        (Some(_stored), Some(current)) => (false, format!("model consistent: '{}'", current)),
        (None, _) | (_, None) => (
            false,
            "no model metadata stored (run `axil doctor` to check)".into(),
        ),
    };

    Some(DetectorResult {
        name: "embedding_drift".into(),
        triggered,
        detail,
        severity: if triggered { "error" } else { "info" }.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db() -> (tempfile::TempDir, Axil) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        (dir, db)
    }

    #[test]
    fn stale_sessions_detects_old_active() {
        let (_dir, db) = temp_db();

        // Insert an old active session.
        let mut old_session = crate::Record::new(
            "_sessions",
            json!({"status": "active", "started_at": "2026-01-01T00:00:00Z"}),
        );
        old_session.created_at = Utc::now() - Duration::hours(48);
        db.storage().insert(&old_session).unwrap();

        let result = detect_stale_sessions(&db);
        assert!(result.triggered);
        assert_eq!(result.severity, "warning");
    }

    #[test]
    fn stale_sessions_ignores_recent() {
        let (_dir, db) = temp_db();

        db.insert(
            "_sessions",
            json!({"status": "active", "started_at": Utc::now().to_rfc3339()}),
        )
        .unwrap();

        let result = detect_stale_sessions(&db);
        assert!(!result.triggered);
    }

    #[test]
    fn storage_growth_first_measurement() {
        let (_dir, db) = temp_db();
        db.insert("notes", json!({"text": "test"})).unwrap();

        let result = detect_storage_growth(&db);
        assert!(!result.triggered);
        assert!(result.detail.contains("first measurement"));
    }

    #[test]
    fn storage_growth_stable() {
        let (_dir, db) = temp_db();
        db.insert("notes", json!({"text": "test"})).unwrap();

        // First measurement.
        detect_storage_growth(&db);
        // Second measurement (should show growth rate).
        let result = detect_storage_growth(&db);
        assert!(!result.triggered);
        assert!(result.detail.contains("growth rate"));
    }

    #[test]
    fn slow_query_detection_empty() {
        let (_dir, db) = temp_db();
        let result = detect_slow_query_patterns(&db);
        assert!(!result.triggered);
    }

    #[test]
    fn run_all_returns_results() {
        let (_dir, db) = temp_db();
        let results = run_all_detectors(&db);
        // Should have at least stale_sessions, slow_queries, storage_growth
        assert!(results.len() >= 3);
    }
}
