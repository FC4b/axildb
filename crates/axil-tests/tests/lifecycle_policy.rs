//! Per-table lifecycle policy (`[lifecycle.tables.<table>]`): append-only /
//! audit-log semantics for tables whose similar records are distinct events.
//!
//! Field-report scenario this pins down: an experiment log stores hundreds of
//! similar-sounding autopsies ("RSI mean-revert, period 7/14/21…"); default
//! auto-supersede demoted them as revisions and auto-heal compaction then
//! hard-deleted the history.

use axil_core::config::{CompactMode, LifecycleConfig, TableLifecycle};
use axil_core::{Axil, HealingConfig};
use serde_json::json;

/// Deterministic 4-dim mock embedder (no ONNX model needed). Same pattern as
/// `self_healing.rs` / `intelligent_db.rs`: identical text → identical vector
/// → cosine similarity 1.0, comfortably above the 0.92 supersede threshold.
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

fn append_only_policy(table: &str) -> LifecycleConfig {
    let mut cfg = LifecycleConfig::default();
    cfg.tables.insert(
        table.to_string(),
        TableLifecycle {
            supersede: false,
            decay: false,
            compact: CompactMode::Never,
        },
    );
    cfg
}

fn open_with_mock_vector(
    dir: &tempfile::TempDir,
    lifecycle: Option<LifecycleConfig>,
    threshold: Option<f32>,
) -> Axil {
    let path = dir.path().join("test.axil");
    let vector = axil_vector::VectorEngine::open(&path, 4).unwrap();
    let mut builder = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FrontWindowEmbedder));
    if let Some(l) = lifecycle {
        builder = builder.with_lifecycle(l);
    }
    if let Some(t) = threshold {
        builder = builder.with_supersede_threshold(t);
    }
    builder.build().unwrap()
}

fn superseded_flag(db: &Axil, id: &axil_core::RecordId) -> bool {
    db.get(id)
        .unwrap()
        .unwrap()
        .data
        .get("_superseded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn mark_superseded(db: &Axil, id: &axil_core::RecordId) {
    let mut data = db.get(id).unwrap().unwrap().data;
    data.as_object_mut()
        .unwrap()
        .insert("_superseded".into(), json!(true));
    db.update(id, data).unwrap();
}

// ── insert-path auto-supersede ──────────────────────────────────────────

#[test]
fn default_policy_supersedes_near_duplicates_on_insert() {
    // Control: pins the existing behavior the opt-out diverges from.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, None, None);

    let v1 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v1"}))
        .unwrap();
    let _v2 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v2"}))
        .unwrap();

    assert!(
        superseded_flag(&db, &v1.id),
        "identical-vector insert into an unconfigured table must supersede (current default)"
    );
}

#[test]
fn supersede_false_keeps_similar_records_live() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, Some(append_only_policy("autopsies")), None);

    let v1 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v1"}))
        .unwrap();
    let v2 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v2"}))
        .unwrap();

    assert!(
        !superseded_flag(&db, &v1.id),
        "supersede = false table must never demote existing records"
    );
    assert!(!superseded_flag(&db, &v2.id));

    // Other tables keep the default behavior under the same handle.
    let f1 = db
        .insert("facts", json!({"summary": "auth timeout fact one"}))
        .unwrap();
    let _f2 = db
        .insert("facts", json!({"summary": "auth timeout fact two"}))
        .unwrap();
    assert!(
        superseded_flag(&db, &f1.id),
        "unconfigured table must still auto-supersede"
    );
}

#[test]
fn supersede_threshold_above_one_disables_demotion() {
    // Wires healing.supersede_similarity_threshold for real: cosine
    // similarity never exceeds 1.0, so a threshold above it is a global off
    // switch.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, None, Some(1.5));
    assert_eq!(db.supersede_threshold(), 1.5);

    let v1 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v1"}))
        .unwrap();
    let _v2 = db
        .insert("autopsies", json!({"summary": "auth timeout experiment v2"}))
        .unwrap();

    assert!(!superseded_flag(&db, &v1.id));
}

// ── compaction ─────────────────────────────────────────────────────────

#[test]
fn compact_never_preserves_marked_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_lifecycle(append_only_policy("autopsies"))
        .build()
        .unwrap();

    let protected = db
        .insert("autopsies", json!({"summary": "old experiment"}))
        .unwrap();
    let purgeable = db.insert("facts", json!({"summary": "old fact"})).unwrap();
    mark_superseded(&db, &protected.id);
    mark_superseded(&db, &purgeable.id);

    let report = db.compact().unwrap();
    assert_eq!(report.purged_superseded, 1, "only the unprotected record");
    assert!(
        db.get(&protected.id).unwrap().is_some(),
        "compact = \"never\" table must keep superseded records"
    );
    assert!(db.get(&purgeable.id).unwrap().is_none());
}

#[test]
fn compact_never_records_do_not_count_as_dead() {
    // If protected records counted as "dead", doctor/session-heal would nag
    // (and auto-heal) forever about records that are kept by design.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_lifecycle(append_only_policy("autopsies"))
        .build()
        .unwrap();

    let r = db
        .insert("autopsies", json!({"summary": "old experiment"}))
        .unwrap();
    mark_superseded(&db, &r.id);

    let problems = db.detect_problems();
    assert!(
        !problems.iter().any(|p| p.detector == "superseded_records"),
        "protected records must not surface as pending cleanup"
    );
}

// ── heal_all honors healing.auto_compact ───────────────────────────────

#[test]
fn heal_all_with_auto_compact_false_skips_purge() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();

    let r = db.insert("facts", json!({"summary": "old fact"})).unwrap();
    mark_superseded(&db, &r.id);

    let config = HealingConfig {
        auto_compact: false,
        ..HealingConfig::default()
    };
    let report = db.heal_all(&config, false).unwrap();
    assert!(
        db.get(&r.id).unwrap().is_some(),
        "auto_compact = false must make heal_all leave superseded records alone"
    );
    assert!(
        report.actions.iter().any(|a| a.action == "compact_skipped"),
        "the skip must be surfaced, not silent"
    );
    assert!(
        !report.healed,
        "an informational skip is not a heal — session-heal logs must not \
         claim a repair while the same records stay pending"
    );

    // Flipping the knob back purges as before.
    let report = db.heal_all(&HealingConfig::default(), false).unwrap();
    assert!(report.actions.iter().any(|a| a.action == "compact"));
    assert!(db.get(&r.id).unwrap().is_none());
}

#[test]
fn brain_threshold_above_one_disables_supersede_resolution() {
    use axil_core::{remember, MemorySource, Observation, PipelineAction};

    fn observe(db: &Axil, text: &str) -> PipelineAction {
        let mut obs = Observation::from_text(text).with_source(MemorySource::Agent);
        obs.table = Some("notes".into());
        remember(db, obs).unwrap().action
    }

    // Control: with the default threshold, an asymmetric-negation conflict on
    // a shared entity is high-confidence and supersedes the original.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, None, None);
    assert_eq!(
        observe(&db, "auth timeout is enabled in `login_flow`"),
        PipelineAction::Stored
    );
    let second = observe(&db, "auth timeout is not enabled in `login_flow`");
    assert!(
        matches!(second, PipelineAction::Superseded { .. }),
        "control: negated conflict must supersede at the default threshold, got {second:?}"
    );

    // Threshold above 1.0 is the documented global off switch — it must cover
    // the brain pipeline too, not just core inserts.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, None, Some(1.5));
    assert_eq!(
        observe(&db, "auth timeout is enabled in `login_flow`"),
        PipelineAction::Stored
    );
    assert_eq!(
        observe(&db, "auth timeout is not enabled in `login_flow`"),
        PipelineAction::Stored,
        "threshold > 1.0 must disable brain-path superseding"
    );
}

// ── config auto-load from axil.toml next to the database ───────────────

#[test]
fn lifecycle_loads_from_axil_toml_next_to_db() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("axil.toml"),
        r#"
[healing]
supersede_similarity_threshold = 0.5

[lifecycle.tables.autopsies]
supersede = false
compact = "never"
"#,
    )
    .unwrap();

    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();

    assert_eq!(db.supersede_threshold(), 0.5);
    let policy = db.lifecycle_policy("autopsies");
    assert!(!policy.supersede);
    assert_eq!(policy.compact, CompactMode::Never);

    let r = db
        .insert("autopsies", json!({"summary": "old experiment"}))
        .unwrap();
    mark_superseded(&db, &r.id);
    let report = db.compact().unwrap();
    assert_eq!(report.purged_superseded, 0);
    assert!(db.get(&r.id).unwrap().is_some());
}

// ── time-series downsampling ───────────────────────────────────────────

#[test]
fn downsample_never_purges_protected_tables() {
    use axil_timeseries::AxilBuilderTimeSeriesExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_timeseries_engine()
        .unwrap()
        .with_lifecycle(append_only_policy("autopsies"))
        .build()
        .unwrap();

    db.insert("autopsies", json!({"summary": "experiment 1"}))
        .unwrap();
    db.insert("autopsies", json!({"summary": "experiment 2"}))
        .unwrap();
    db.insert("sessions", json!({"summary": "old session"}))
        .unwrap();

    // retain_days = 0 makes everything "old". The unprotected table is
    // summarized and purged; the append-only table is untouched — neither
    // summarized nor deleted (this is the bare-`axil heal` deletion path).
    let (summaries, purged) = db.downsample(0, true).unwrap();
    assert_eq!(summaries, 1, "only the sessions group gets a summary");
    assert_eq!(purged, 1, "only the sessions record is purged");
    assert_eq!(
        db.list("autopsies").unwrap().len(),
        2,
        "compact = \"never\" table must survive downsampling"
    );
    assert_eq!(db.list("sessions").unwrap().len(), 0);
}

// ── brain pipeline (axil remember) ─────────────────────────────────────

#[test]
fn brain_remember_stores_every_event_in_append_only_table() {
    use axil_core::{remember, MemorySource, Observation, PipelineAction};

    let dir = tempfile::tempdir().unwrap();
    let db = open_with_mock_vector(&dir, Some(append_only_policy("autopsies")), None);

    let mut obs = Observation::from_text("Fixed `AuthModule` timeout by increasing pool size")
        .with_source(MemorySource::Agent);
    obs.table = Some("autopsies".into());
    let first = remember(&db, obs).unwrap();
    assert_eq!(first.action, PipelineAction::Stored);

    // Identical observation: default tables dedupe it; an append-only table
    // must store it again — every trial counts.
    let mut obs = Observation::from_text("Fixed `AuthModule` timeout by increasing pool size")
        .with_source(MemorySource::Agent);
    obs.table = Some("autopsies".into());
    let second = remember(&db, obs).unwrap();
    assert_eq!(
        second.action,
        PipelineAction::Stored,
        "append-only table must not dedupe or supersede repeated observations"
    );

    // Control: the same double-store into an unconfigured table is deduped.
    let mut obs = Observation::from_text("Fixed `AuthModule` timeout by increasing pool size")
        .with_source(MemorySource::Agent);
    obs.table = Some("notes".into());
    let n1 = remember(&db, obs).unwrap();
    assert_eq!(n1.action, PipelineAction::Stored);
    let mut obs = Observation::from_text("Fixed `AuthModule` timeout by increasing pool size")
        .with_source(MemorySource::Agent);
    obs.table = Some("notes".into());
    let n2 = remember(&db, obs).unwrap();
    assert_eq!(
        n2.action,
        PipelineAction::Ignored,
        "default tables keep duplicate detection"
    );
}
