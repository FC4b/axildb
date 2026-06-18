//! Integration tests for autonomous pattern recognition.
//!
//! Tests detection of repeated failures, hot spots, knowledge gaps,
//! and pattern dismissal.

use axil_core::Axil;
use axil_memory::patterns::{PatternEngine, PatternType};
use axil_memory::types::{TABLE_ENTITIES, TABLE_EPISODES};
use serde_json::json;
use tempfile::TempDir;

fn temp_db() -> (Axil, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

// ── Empty database ───────────────────────────────────────────────────────

#[test]
fn detect_on_empty_db() {
    let (db, _dir) = temp_db();
    let engine = PatternEngine::new(&db);

    let patterns = engine.detect().unwrap();
    assert!(patterns.is_empty());
}

// ── Repeated failure detection ───────────────────────────────────────────

#[test]
fn detect_repeated_failures() {
    let (db, _dir) = temp_db();

    // Insert failure episodes with a recurring keyword.
    for i in 0..5 {
        db.insert(
            TABLE_EPISODES,
            json!({
                "outcome": "failure",
                "summary": format!("timeout error in authentication module attempt {i}")
            }),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let patterns = engine.detect().unwrap();

    let failures: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::RepeatedFailure)
        .collect();
    assert!(
        !failures.is_empty(),
        "expected repeated failure pattern, got: {patterns:?}"
    );
    assert!(failures[0].frequency >= 3);
}

#[test]
fn no_repeated_failure_below_threshold() {
    let (db, _dir) = temp_db();

    // Only 2 failures — below min_frequency of 3.
    for i in 0..2 {
        db.insert(
            TABLE_EPISODES,
            json!({
                "outcome": "failure",
                "summary": format!("timeout error attempt {i}")
            }),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let patterns = engine.detect().unwrap();

    let failures: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::RepeatedFailure)
        .collect();
    assert!(
        failures.is_empty(),
        "2 failures should not trigger pattern with min_frequency=3"
    );
}

// ── Hot spot detection ───────────────────────────────────────────────────

#[test]
fn detect_hot_spots() {
    let (db, _dir) = temp_db();

    // Create many facts about the same entity.
    for i in 0..5 {
        db.insert(
            TABLE_ENTITIES,
            json!({
                "entity": "auth-module",
                "fact": format!("fact number {i} about auth-module")
            }),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let patterns = engine.detect().unwrap();

    let hot_spots: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::HotSpot)
        .collect();
    assert!(!hot_spots.is_empty(), "expected hot spot for auth-module");
    assert!(
        hot_spots[0].name.contains("auth-module"),
        "hot spot should reference auth-module: {}",
        hot_spots[0].name
    );
}

// ── Store and list patterns ──────────────────────────────────────────────

#[test]
fn store_and_list_patterns() {
    let (db, _dir) = temp_db();

    // Create enough data for detection.
    for i in 0..4 {
        db.insert(
            TABLE_EPISODES,
            json!({
                "outcome": "failure",
                "summary": format!("database connection timeout attempt {i}")
            }),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let detected = engine.detect().unwrap();
    assert!(
        !detected.is_empty(),
        "4 failure episodes with shared keyword should produce patterns at threshold 3"
    );

    let stored = engine.store_patterns(&detected).unwrap();
    assert!(stored > 0, "should store at least one pattern");

    let listed = engine.list(None).unwrap();
    assert!(!listed.is_empty(), "list() should return stored patterns");
}

#[test]
fn list_filters_by_type() {
    let (db, _dir) = temp_db();

    // Create hot spot data.
    for i in 0..4 {
        db.insert(
            TABLE_ENTITIES,
            json!({"entity": "cache-layer", "fact": format!("fact {i}")}),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let detected = engine.detect().unwrap();
    engine.store_patterns(&detected).unwrap();

    // List only hot spots.
    let hot_spots = engine.list(Some(PatternType::HotSpot)).unwrap();
    for p in &hot_spots {
        assert_eq!(p.pattern_type, PatternType::HotSpot);
    }

    // List only repeated failures (should be empty since we only created entities).
    let failures = engine.list(Some(PatternType::RepeatedFailure)).unwrap();
    assert!(failures.is_empty());
}

// ── Dismiss patterns ─────────────────────────────────────────────────────

#[test]
fn dismiss_pattern() {
    let (db, _dir) = temp_db();

    for i in 0..4 {
        db.insert(
            TABLE_ENTITIES,
            json!({"entity": "noisy-module", "fact": format!("fact {i}")}),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let detected = engine.detect().unwrap();
    engine.store_patterns(&detected).unwrap();

    let before = engine.list(None).unwrap();
    assert!(!before.is_empty(), "should have patterns before dismiss");

    // Dismiss the first pattern.
    let name = &before[0].name;
    let dismissed = engine.dismiss(name).unwrap();
    assert!(dismissed, "dismiss should return true");

    // Listed patterns should no longer include dismissed one.
    let after = engine.list(None).unwrap();
    assert!(
        !after.iter().any(|p| &p.name == name),
        "dismissed pattern should not appear in list"
    );
}

#[test]
fn dismiss_nonexistent_returns_false() {
    let (db, _dir) = temp_db();
    let engine = PatternEngine::new(&db);

    let result = engine.dismiss("does-not-exist").unwrap();
    assert!(
        !result,
        "dismissing nonexistent pattern should return false"
    );
}

// ── Pattern struct fields ────────────────────────────────────────────────

#[test]
fn pattern_has_required_fields() {
    let (db, _dir) = temp_db();

    for i in 0..5 {
        db.insert(
            TABLE_EPISODES,
            json!({
                "outcome": "failure",
                "summary": format!("authentication timeout error number {i}")
            }),
        )
        .unwrap();
    }

    let engine = PatternEngine::new(&db).with_min_frequency(3);
    let patterns = engine.detect().unwrap();

    let p = patterns
        .first()
        .expect("5 failure episodes with shared keyword should produce a pattern");
    assert!(!p.name.is_empty(), "name should not be empty");
    assert!(!p.description.is_empty(), "description should not be empty");
    assert!(p.frequency >= 3, "frequency should meet threshold");
    assert!(!p.first_seen.is_empty(), "first_seen should be set");
    assert!(!p.last_seen.is_empty(), "last_seen should be set");
    assert!(
        p.confidence > 0.0 && p.confidence <= 1.0,
        "confidence should be 0-1"
    );
    assert!(
        !p.dismissed,
        "newly detected patterns should not be dismissed"
    );
}
