//! One set of supersede rules for every writer, and compaction that keeps the
//! history the memory extension promises.
//!
//! - `detect_conflicts` demotes records only through `Axil::mark_superseded`,
//!   so the lifecycle policy, same-table and pin rules hold there too.
//! - Medium-confidence conflicts need a changed value, not just similarity.
//! - The brain pipeline weighs every candidate instead of stopping at the
//!   first contradiction.
//! - Superseded facts in the memory extension's `_`-prefixed tables survive
//!   compaction, so `history()` still returns every version.

use axil_core::config::{CompactMode, LifecycleConfig, TableLifecycle};
use axil_core::consolidation::ConflictResult;
use axil_core::{remember, Axil, HealingConfig, MemorySource, Observation, PipelineAction};
use axil_memory::AgentMemory;
use serde_json::json;

/// Deterministic mock embedder: texts that mention the same subset of `auth`
/// / `timeout` / `pool` embed identically (cosine 1.0). A trailing `~` nudges the
/// vector slightly off-axis (cosine ~0.98 to its un-nudged twin) so a test can
/// order two candidates by similarity.
struct FeatureEmbedder;

impl axil_core::TextEmbedder for FeatureEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let t = text.to_lowercase();
        Ok(vec![
            if t.contains("auth") { 1.0 } else { 0.0 },
            if t.contains("timeout") { 1.0 } else { 0.0 },
            if t.contains("pool") { 1.0 } else { 0.0 },
            1.0,
            if t.ends_with('~') { 0.4 } else { 0.0 },
        ])
    }
}

fn open_with(
    dir: &tempfile::TempDir,
    lifecycle: Option<LifecycleConfig>,
    threshold: Option<f32>,
) -> Axil {
    let path = dir.path().join("test.axil");
    let vector = axil_vector::VectorEngine::open(&path, 5).unwrap();
    let mut builder = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(FeatureEmbedder));
    if let Some(l) = lifecycle {
        builder = builder.with_lifecycle(l);
    }
    if let Some(t) = threshold {
        builder = builder.with_supersede_threshold(t);
    }
    builder.build().unwrap()
}

/// A handle whose insert path never auto-supersedes (threshold above 1.0),
/// so a test controls exactly which writer demotes what.
/// `detect_conflicts` has its own conflict threshold and is unaffected.
fn open(dir: &tempfile::TempDir, lifecycle: Option<LifecycleConfig>) -> Axil {
    open_with(dir, lifecycle, Some(1.5))
}

fn superseded(db: &Axil, id: &axil_core::RecordId) -> bool {
    axil_core::is_superseded_record(&db.get(id).unwrap().unwrap())
}

fn only_supersede_off(table: &str) -> LifecycleConfig {
    let mut cfg = LifecycleConfig::default();
    cfg.tables.insert(
        table.to_string(),
        TableLifecycle {
            supersede: false,
            decay: true,
            compact: CompactMode::Auto,
        },
    );
    cfg
}

const AFFIRMED: &str = "auth timeout is enabled in `login_flow`";
const NEGATED: &str = "auth timeout is not enabled in `login_flow`";

// ── detect_conflicts goes through the shared supersede rules ───────────

#[test]
fn detect_conflicts_supersedes_in_a_default_table() {
    // Control: an asymmetric-negation conflict on a shared entity supersedes.
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, None);
    let old = db
        .insert("autopsies", json!({"summary": AFFIRMED}))
        .unwrap();
    let new = db.insert("autopsies", json!({"summary": NEGATED})).unwrap();

    let conflicts = db.detect_conflicts(&new.id).unwrap();
    assert!(
        matches!(conflicts.as_slice(), [ConflictResult::Supersedes { .. }]),
        "{conflicts:?}"
    );
    assert!(superseded(&db, &old.id));
}

#[test]
fn detect_conflicts_never_demotes_in_a_supersede_false_table() {
    // `supersede = false` alone (compaction still on): a demotion here would
    // be purged by the next auto-heal, losing an append-only trial.
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, Some(only_supersede_off("autopsies")));
    let old = db
        .insert("autopsies", json!({"summary": AFFIRMED}))
        .unwrap();
    let new = db.insert("autopsies", json!({"summary": NEGATED})).unwrap();

    let conflicts = db.detect_conflicts(&new.id).unwrap();
    assert!(
        matches!(conflicts.as_slice(), [ConflictResult::Contradicts { .. }]),
        "a refused supersede is surfaced for review, got {conflicts:?}"
    );
    assert!(!superseded(&db, &old.id));

    db.heal_all(&HealingConfig::default(), false).unwrap();
    assert!(
        db.get(&old.id).unwrap().is_some(),
        "the trial must survive auto-heal"
    );
}

#[test]
fn detect_conflicts_ignores_other_tables_and_pinned_records() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, None);
    let elsewhere = db.insert("notes", json!({"summary": AFFIRMED})).unwrap();
    let pinned = db
        .insert(
            "autopsies",
            json!({"summary": AFFIRMED, "_importance_pinned": true}),
        )
        .unwrap();
    let new = db.insert("autopsies", json!({"summary": NEGATED})).unwrap();

    let conflicts = db.detect_conflicts(&new.id).unwrap();
    assert!(!superseded(&db, &elsewhere.id), "cross-table demotion");
    assert!(!superseded(&db, &pinned.id), "pinned records are absolute");
    assert!(
        matches!(
            conflicts.as_slice(),
            [ConflictResult::Contradicts { existing_record_id, .. }] if *existing_record_id == pinned.id
        ),
        "only the same-table record conflicts, and only for review: {conflicts:?}"
    );
}

#[test]
fn mark_superseded_enforces_the_shared_rules() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, Some(only_supersede_off("autopsies")));

    // Policy: supersede = false refuses.
    let a1 = db
        .insert("autopsies", json!({"summary": "trial 1"}))
        .unwrap();
    let a2 = db
        .insert("autopsies", json!({"summary": "trial 2"}))
        .unwrap();
    assert!(!db.mark_superseded(&a1.id, &a2, None).unwrap());

    // Cross-table refuses; same table succeeds exactly once.
    let f1 = db.insert("facts", json!({"summary": "fact 1"})).unwrap();
    let f2 = db.insert("facts", json!({"summary": "fact 2"})).unwrap();
    assert!(!db.mark_superseded(&f1.id, &a2, None).unwrap());
    assert!(db.mark_superseded(&f1.id, &f2, None).unwrap());
    assert!(
        !db.mark_superseded(&f1.id, &f2, None).unwrap(),
        "already superseded"
    );
    assert_eq!(
        db.get(&f1.id).unwrap().unwrap().data["_superseded_by"],
        json!(f2.id.to_string())
    );

    // Recency: an older record never demotes a newer one.
    let later = chrono::Utc::now() + chrono::Duration::hours(1);
    let f3 = db
        .insert_at("facts", json!({"summary": "fact 3"}), later)
        .unwrap();
    assert!(!db.mark_superseded(&f3.id, &f2, None).unwrap());
}

// ── medium-confidence conflicts need a changed value ───────────────────

#[test]
fn restatement_is_not_stored_as_a_contradiction() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(&dir, None, None);
    let observe = |text: &str| {
        let mut obs = Observation::from_text(text).with_source(MemorySource::Agent);
        obs.table = Some("notes".into());
        remember(&db, obs).unwrap()
    };

    observe("Auth uses `JWT` for sessions");
    let second = observe("Session tokens in auth are `JWT`");
    assert_eq!(second.action, PipelineAction::Stored);
    assert_eq!(second.confidence, 1.0, "{}", second.reason);
    let stored = second.record.unwrap();
    assert!(stored.data.get("_contradicts").is_none(), "{}", stored.data);

    // A changed value under the same framing is still surfaced.
    observe("Auth token lifetime is `15m` in `login_flow`");
    let changed = observe("Auth token lifetime is `30m` in `login_flow`");
    assert_eq!(changed.confidence, 0.7, "{}", changed.reason);
    assert!(changed.record.unwrap().data.get("_contradicts").is_some());
}

// ── brain resolution weighs every candidate ────────────────────────────

#[test]
fn a_later_duplicate_outranks_an_earlier_contradiction() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(&dir, None, None);
    let text = "auth timeout is `45s` in `login_flow`";

    // Two existing memories in a `_` table (no insert-path side effects),
    // embedded by hand so their similarity order is fixed: the value-changed
    // one is the closest match, the exact duplicate comes second.
    let changed = db
        .insert(
            "_obs",
            json!({"summary": "auth timeout is `30s` in `login_flow`"}),
        )
        .unwrap();
    db.embed_text(&changed.id, "auth timeout is `30s` in `login_flow`")
        .unwrap();
    let duplicate = db.insert("_obs", json!({"summary": text})).unwrap();
    db.embed_text(&duplicate.id, &format!("{text}~")).unwrap();

    let mut obs = Observation::from_text(text).with_source(MemorySource::Agent);
    obs.table = Some("_obs".into());
    let outcome = remember(&db, obs).unwrap();
    assert_eq!(
        outcome.action,
        PipelineAction::Ignored,
        "the observation is already stored — duplicate wins: {}",
        outcome.reason
    );
    assert!(
        outcome.reason.contains(&duplicate.id.to_string()),
        "{}",
        outcome.reason
    );
}

// ── compaction keeps memory-extension history ──────────────────────────

#[test]
fn know_know_heal_keeps_both_versions_in_history() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(&dir, None, None);
    let mem = AgentMemory::new(&db);

    let first = mem.semantic().know("svc", "port 8080", None).unwrap();
    mem.semantic().know("svc", "port 9090", None).unwrap();
    assert!(
        superseded(&db, &first.id),
        "the second fact supersedes the first"
    );

    db.heal_all(&HealingConfig::default(), false).unwrap();

    let history = mem.semantic().history("svc").unwrap();
    let facts: Vec<&str> = history
        .iter()
        .map(|r| r.data["fact"].as_str().unwrap())
        .collect();
    assert_eq!(facts, ["port 8080", "port 9090"]);
}

#[test]
fn legacy_meta_superseded_facts_survive_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, None);
    let legacy = db
        .insert(
            "_entities",
            json!({"entity": "svc", "fact": "port 7070", "_meta": {"superseded": true}}),
        )
        .unwrap();
    let stale = db
        .insert("facts", json!({"summary": "old fact", "_superseded": true}))
        .unwrap();

    let report = db.compact().unwrap();
    assert!(
        db.get(&legacy.id).unwrap().is_some(),
        "extension history kept"
    );
    assert!(
        db.get(&stale.id).unwrap().is_none(),
        "user tables purge as before"
    );
    assert_eq!(report.purged_superseded, 1);
}
