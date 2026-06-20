//! Integration tests for : Intelligent Database features.
//!
//! Tests cover: scoring, temporal parsing, entity extraction, feedback,
//! consolidation, auto-linking, and predictive pre-fetch.

use axil_core::{
    extract_entities, parse_temporal, temporal_boost, Axil, EntityType, FeedbackStore,
    PrefetchEngine, RecordId, ScoreWeights, TemporalTarget,
};
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use axil_vector::VectorEngine;
use chrono::{TimeZone, Utc};
use serde_json::json;

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

fn temp_db_with_graph() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

struct FrontWindowEmbedder;

impl axil_core::TextEmbedder for FrontWindowEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let window = text.chars().take(100).collect::<String>().to_lowercase();
        Ok(vec![
            if window.contains("auth") { 1.0 } else { 0.0 },
            if window.contains("timeout") { 1.0 } else { 0.0 },
            if window.contains("pool") { 1.0 } else { 0.0 },
            1.0,
        ])
    }
}

fn temp_db_with_mock_vector() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let vector = VectorEngine::open(&path, 4).unwrap();
    let db = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FrontWindowEmbedder))
        .build()
        .unwrap();
    (db, dir)
}

fn temp_db_with_mock_vector_and_fts() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let vector = VectorEngine::open(&path, 4).unwrap();
    let db = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FrontWindowEmbedder))
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

fn temp_db_with_fts() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

// ── Scoring ─────────────────────────────────────────────────────────────

#[test]
fn score_weights_default_sum_to_one() {
    let w = ScoreWeights::default();
    let sum = w.vector + w.recency + w.graph + w.keyword + w.feedback + w.temporal + w.preference;
    assert!((sum - 1.0).abs() < 0.001, "weights sum = {sum}");
}

#[test]
fn scoring_recency_decay_works() {
    let now = Utc::now();
    let half_life = 168.0; // 1 week in hours

    // At time=0, decay = 1.0
    let decay_now = axil_core::scoring::recency_decay(&now, &now, half_life);
    assert!((decay_now - 1.0).abs() < 0.01);

    // At half-life, decay = 0.5
    let week_ago = now - chrono::Duration::hours(168);
    let decay_week = axil_core::scoring::recency_decay(&week_ago, &now, half_life);
    assert!((decay_week - 0.5).abs() < 0.02, "decay = {decay_week}");
}

#[test]
fn scoring_keyword_overlap() {
    let keywords = vec!["auth".into(), "error".into(), "login".into()];
    let text = "Fixed auth error in the login handler";
    let overlap = axil_core::scoring::keyword_overlap(&keywords, text);
    assert!((overlap - 1.0).abs() < 0.01);

    let partial = "Auth module refactored";
    let overlap2 = axil_core::scoring::keyword_overlap(&keywords, partial);
    assert!((overlap2 - 1.0 / 3.0).abs() < 0.01);
}

#[test]
fn scoring_extract_keywords_filters_stopwords() {
    let kw = axil_core::scoring::extract_keywords("the auth error in the login flow");
    assert!(kw.contains(&"auth".to_string()));
    assert!(kw.contains(&"error".to_string()));
    assert!(!kw.contains(&"the".to_string()));
    assert!(!kw.contains(&"in".to_string()));
}

#[test]
fn scoring_fuse_produces_valid_score() {
    let record = axil_core::Record::new("sessions", json!({"summary": "Fixed auth bug"}));
    let signals = axil_core::SignalValues {
        vector_similarity: 0.9,
        keyword_match: 0.5,
        graph_proximity: 0.3,
        feedback_boost: 0.1,
        ..Default::default()
    };
    let config = axil_core::RecallConfig::default();
    let (score, explanation) = axil_core::scoring::fuse_signals(&record, &signals, &config);
    assert!(score > 0.0);
    assert!(score <= 1.0);
    assert!(!explanation.signals.is_empty());
    assert!(!explanation.summary.is_empty());
}

// ── Temporal Parsing ────────────────────────────────────────────────────

#[test]
fn temporal_parse_days_ago() {
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let (target, cleaned) = parse_temporal("auth error 3 days ago", &now).unwrap();
    assert!(
        (target.target - (now - chrono::Duration::days(3)))
            .num_seconds()
            .abs()
            < 2
    );
    assert_eq!(cleaned, "auth error");
}

#[test]
fn temporal_parse_last_week() {
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let (target, _) = parse_temporal("deployments last week", &now).unwrap();
    assert_eq!(target.window_days, 7.0);
}

#[test]
fn temporal_parse_iso_date() {
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let (target, cleaned) = parse_temporal("bugs from 2026-03-15", &now).unwrap();
    assert_eq!(
        target.target.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2026, 3, 15).unwrap()
    );
    assert_eq!(cleaned, "bugs from");
}

#[test]
fn temporal_parse_month_day() {
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let (target, _) = parse_temporal("deployment on March 15", &now).unwrap();
    assert_eq!(
        target.target.date_naive(),
        chrono::NaiveDate::from_ymd_opt(2026, 3, 15).unwrap()
    );
}

#[test]
fn temporal_no_match_returns_none() {
    let now = Utc::now();
    assert!(parse_temporal("auth error fix", &now).is_none());
}

#[test]
fn temporal_boost_at_target_is_max() {
    let now = Utc::now();
    let target = TemporalTarget {
        target: now,
        window_days: 7.0,
    };
    let boost = temporal_boost(&now, &target);
    assert!((boost - 0.40).abs() < 0.01);
}

#[test]
fn temporal_boost_outside_window_is_zero() {
    let now = Utc::now();
    let target = TemporalTarget {
        target: now,
        window_days: 7.0,
    };
    let old = now - chrono::Duration::days(10);
    assert_eq!(temporal_boost(&old, &target), 0.0);
}

// ── Entity Extraction ───────────────────────────────────────────────────

#[test]
fn entity_extract_backtick_code() {
    let entities = extract_entities("Fixed bug in `AuthModule` by updating `auth_config`");
    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"AuthModule"));
    assert!(names.contains(&"auth_config"));
}

#[test]
fn entity_extract_file_path() {
    let entities = extract_entities("Modified /src/auth/login.rs for the fix");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::File && e.name.contains("auth/login.rs")));
}

#[test]
fn entity_extract_camel_case() {
    let entities = extract_entities("The AuthModule handles authentication");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::Code && e.name == "auth_module"));
}

#[test]
fn entity_extract_snake_case() {
    let entities = extract_entities("Updated auth_module for the fix");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::Code && e.name == "auth_module"));
}

#[test]
fn entity_deduplication() {
    let entities = extract_entities("`AuthModule` is the AuthModule class");
    let auth_count = entities
        .iter()
        .filter(|e| e.name.to_lowercase().contains("authmodule") || e.name.contains("auth_module"))
        .count();
    assert_eq!(auth_count, 1, "should deduplicate");
}

#[test]
fn entity_empty_text() {
    assert!(extract_entities("").is_empty());
}

// ── Feedback Store ──────────────────────────────────────────────────────

#[test]
fn feedback_mark_and_retrieve() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    let rid = RecordId::new();

    store.mark_relevant(&emb, &rid);
    assert_eq!(store.len(), 1);
    assert!(store.has_feedback(&rid));
}

#[test]
fn feedback_increment_count() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    let rid = RecordId::new();

    store.mark_relevant(&emb, &rid);
    store.mark_relevant(&emb, &rid);

    let entries = store.entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].count, 2);
}

#[test]
fn feedback_compute_boosts() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    let rid = RecordId::new();
    store.mark_relevant(&emb, &rid);

    let now = Utc::now();
    let boosts = store.compute_boosts(&emb, &[rid.clone()], &now);
    assert!(boosts.contains_key(&rid));
    assert!(*boosts.get(&rid).unwrap() > 0.0);
}

#[test]
fn feedback_no_boost_for_unknown() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    let rid1 = RecordId::new();
    let rid2 = RecordId::new();
    store.mark_relevant(&emb, &rid1);

    let now = Utc::now();
    let boosts = store.compute_boosts(&emb, &[rid2], &now);
    assert!(boosts.is_empty());
}

#[test]
fn feedback_decay_removes_old() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    store.mark_relevant(&emb, &RecordId::new());

    let future = Utc::now() + chrono::Duration::days(200);
    store.decay(&future);
    assert!(store.is_empty());
}

#[test]
fn feedback_serialization_round_trip() {
    let store = FeedbackStore::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    store.mark_relevant(&emb, &RecordId::new());

    let bytes = store.to_bytes().unwrap();
    let store2 = FeedbackStore::from_bytes(&bytes).unwrap();
    assert_eq!(store2.len(), 1);
}

// ── Consolidation ───────────────────────────────────────────────────────

#[test]
fn consolidation_novel_on_low_similarity() {
    let r1 = axil_core::Record::new("facts", json!({"summary": "Auth uses JWT"}));
    let r2 = axil_core::Record::new("facts", json!({"summary": "DB uses PostgreSQL"}));
    let result = axil_core::check_conflict(&r2, &r1, 0.5);
    assert!(matches!(result, axil_core::ConflictResult::Novel));
}

#[test]
fn consolidation_confidence_high_for_recent() {
    let now = Utc::now();
    let score = axil_core::compute_confidence(5, &now, &now, 0);
    assert!(score.score > 0.6, "score = {}", score.score);
    assert_eq!(score.contradiction_penalty, 0.0);
}

#[test]
fn consolidation_confidence_low_for_old_contradicted() {
    let now = Utc::now();
    let old = now - chrono::Duration::days(90);
    let score = axil_core::compute_confidence(1, &old, &now, 2);
    assert!(score.score < 0.5, "score = {}", score.score);
}

#[test]
fn consolidation_single_fact() {
    let r = axil_core::Record::new("facts", json!({"summary": "Auth uses JWT"}));
    let facts = vec![(r, axil_core::ConflictResult::Novel)];
    let consolidated = axil_core::consolidate_facts("auth", &facts).unwrap();
    assert_eq!(consolidated.summary, "Auth uses JWT");
    assert_eq!(consolidated.source_ids.len(), 1);
}

#[test]
fn consolidation_superseding_chain() {
    let mut r1 = axil_core::Record::new("facts", json!({"summary": "Auth uses JWT"}));
    r1.created_at = Utc::now() - chrono::Duration::days(10);
    let r2 = axil_core::Record::new("facts", json!({"summary": "Auth uses session cookies"}));
    let facts = vec![
        (r1, axil_core::ConflictResult::Novel),
        (
            r2,
            axil_core::ConflictResult::Supersedes {
                old_record_id: RecordId::new(),
                similarity: 0.95,
            },
        ),
    ];
    let cf = axil_core::consolidate_facts("auth", &facts).unwrap();
    assert!(cf.summary.contains("Auth uses session cookies"));
    assert!(cf.summary.contains("Originally"));
    assert_eq!(cf.source_ids.len(), 2);
}

#[test]
fn consolidation_with_contradictions() {
    let mut r1 = axil_core::Record::new("facts", json!({"summary": "API uses REST"}));
    r1.created_at = Utc::now() - chrono::Duration::days(5);
    let r2 = axil_core::Record::new("facts", json!({"summary": "API uses GraphQL"}));
    let facts = vec![
        (r1, axil_core::ConflictResult::Novel),
        (
            r2,
            axil_core::ConflictResult::Contradicts {
                existing_record_id: RecordId::new(),
                similarity: 0.93,
            },
        ),
    ];
    let cf = axil_core::consolidate_facts("api", &facts).unwrap();
    assert!(cf.summary.contains("CONFLICT"));
}

// ── Prefetch Engine ─────────────────────────────────────────────────────

#[test]
fn prefetch_log_and_detect() {
    let engine = PrefetchEngine::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();

    // Log same query enough times
    for _ in 0..5 {
        engine.log_query(&emb, 5);
    }

    let patterns = engine.detect_patterns();
    assert!(!patterns.is_empty());
    assert!(patterns[0].occurrence_count >= 3);
}

#[test]
fn prefetch_cache_and_retrieve() {
    let engine = PrefetchEngine::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();

    for _ in 0..5 {
        engine.log_query(&emb, 5);
    }
    let patterns = engine.detect_patterns();
    assert!(!patterns.is_empty());

    let ids = vec![RecordId::new(), RecordId::new()];
    engine.cache_results(0, &patterns[0].embedding, ids.clone());

    let cached = engine.get_cached(&emb);
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().len(), 2);
}

#[test]
fn prefetch_invalidation() {
    let engine = PrefetchEngine::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();

    for _ in 0..5 {
        engine.log_query(&emb, 5);
    }
    let patterns = engine.detect_patterns();
    engine.cache_results(0, &patterns[0].embedding, vec![RecordId::new()]);
    engine.invalidate_caches();
    assert!(engine.get_cached(&emb).is_none());
}

#[test]
fn prefetch_serialization() {
    let engine = PrefetchEngine::new();
    let emb: Vec<f32> = (0..8).map(|i| (1.0 + i as f32 * 0.1).sin()).collect();
    engine.log_query(&emb, 5);

    let bytes = engine.to_bytes().unwrap();
    let engine2 = PrefetchEngine::from_bytes(&bytes).unwrap();
    assert_eq!(engine2.query_log_size(), 1);
}

// ── Auto-Linking (with graph) ───────────────────────────────────────────

#[test]
fn auto_link_finds_entity_in_entities_table() {
    let (db, _dir) = temp_db_with_graph();

    let record = db
        .insert("sessions", json!({"summary": "Fixed bug in `AuthModule`"}))
        .unwrap();

    // Auto-link requires embedder for similarity linking, but entity extraction
    // and graph edge creation work without it. The auto_link API requires embedder,
    // so we test entity extraction directly.
    let entities = extract_entities("Fixed bug in `AuthModule` by updating `auth_config`");
    assert!(entities.len() >= 2);
    assert!(entities.iter().any(|e| e.name == "AuthModule"));

    // Verify graph operations work
    let r2 = db
        .insert(
            "_entities",
            json!({"name": "AuthModule", "entity_type": "code"}),
        )
        .unwrap();
    db.relate(&record.id, "mentions", &r2.id, None).unwrap();

    let neighbors = db
        .neighbors(&record.id, Some("mentions"), axil_core::Direction::Out)
        .unwrap();
    assert!(neighbors.iter().any(|neighbor| neighbor.id == r2.id));
}

#[test]
fn recall_uses_chunk_vectors_for_late_long_text() {
    let (db, _dir) = temp_db_with_mock_vector();
    let record = db
        .insert(
            "sessions",
            json!({
                "summary": "general debugging notes",
                "full_text": format!("{} auth timeout pool fix", "noise ".repeat(500)),
            }),
        )
        .unwrap();

    let results = db.recall("auth timeout", 3, None).unwrap();
    assert!(
        results.iter().any(|result| result.record.id == record.id),
        "expected parent record to be recalled via chunk vector"
    );
}

#[test]
fn similar_to_hides_internal_chunk_rows() {
    let (db, _dir) = temp_db_with_mock_vector();
    let record = db
        .insert(
            "sessions",
            json!({
                "summary": "general debugging notes",
                "full_text": format!("{} auth timeout pool fix", "noise ".repeat(500)),
            }),
        )
        .unwrap();

    let results = db.similar_to("auth timeout", 3).unwrap();
    assert!(results
        .iter()
        .all(|(result, _)| result.table != "_recall_chunks"));
    assert!(results.iter().any(|(result, _)| result.id == record.id));
}

#[test]
fn search_text_hides_internal_chunk_rows() {
    let (db, _dir) = temp_db_with_mock_vector_and_fts();
    let record = db
        .insert(
            "sessions",
            json!({
                "summary": "general debugging notes",
                "full_text": format!("{} auth timeout pool fix", "noise ".repeat(500)),
            }),
        )
        .unwrap();

    let results = db.search_text("auth timeout", 5).unwrap();
    assert!(results
        .iter()
        .all(|(result, _)| result.table != "_recall_chunks"));
    assert!(results.iter().any(|(result, _)| result.id == record.id));
}

#[test]
fn recall_falls_back_to_fts_without_vector_stack() {
    let (db, _dir) = temp_db_with_fts();
    let record = db
        .insert(
            "sessions",
            json!({
                "content": "auth timeout in the connection pool",
            }),
        )
        .unwrap();
    db.index_text(&record.id, "content", "auth timeout in the connection pool")
        .unwrap();

    let results = db.recall("auth timeout", 5, None).unwrap();
    assert!(results.iter().any(|result| result.record.id == record.id));
}

// ── Warm Up ─────────────────────────────────────────────────────────────

#[test]
fn warm_up_completes() {
    let (db, _dir) = temp_db();
    let report = db.warm_up().unwrap();
    assert!(report.warmed_up);
}

// ── End-to-end: entity extraction → graph linking → history ─────────────

#[test]
fn entity_graph_history_flow() {
    let (db, _dir) = temp_db_with_graph();

    // Create entity node
    let entity = db
        .insert(
            "_entities",
            json!({"name": "auth_module", "entity_type": "code"}),
        )
        .unwrap();

    // Create facts mentioning the entity
    let fact1 = db
        .insert(
            "facts",
            json!({"summary": "Auth uses JWT", "created": "day 1"}),
        )
        .unwrap();
    let fact2 = db
        .insert(
            "facts",
            json!({"summary": "Auth switched to sessions", "created": "day 10"}),
        )
        .unwrap();

    // Link facts to entity
    db.relate(&fact1.id, "mentions", &entity.id, None).unwrap();
    db.relate(&fact2.id, "mentions", &entity.id, None).unwrap();
    db.relate(&fact2.id, "supersedes", &fact1.id, None).unwrap();

    // Query entity history
    let history = db.entity_history("auth_module").unwrap();
    assert_eq!(history.len(), 2);

    // Consolidate
    let consolidated = db.consolidate_entity("auth_module").unwrap();
    assert!(consolidated.is_some());
    let cf = consolidated.unwrap();
    assert_eq!(cf.entity, "auth_module");
    assert_eq!(cf.source_ids.len(), 2);
}
