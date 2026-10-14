//! The "possible contradiction" review pass, wired with a scripted judge so
//! no model is needed: what it judges, what it writes, and what it must never
//! touch.
//!
//! - A flag is a `possibly_contradicts` edge (newer → older) plus a review
//!   row; neither record changes, and nothing is superseded.
//! - The judge reads `(older, newer)`, only for older candidates.
//! - A pass with nothing to judge never loads the judge.
//! - Dismissing closes a flag for good; superseding the older record closes
//!   it too; boot shows one line only while a flag is open.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axil_core::util::edge_types::POSSIBLY_CONTRADICTS;
use axil_core::{
    collect_extension_blocks, Axil, AxilError, Direction, PairJudge, PairVerdict, RecordId,
    ReviewOptions, CONTRADICTION_REVIEW_TABLE,
};
use axil_graph::AxilBuilderGraphExt;
use serde_json::{json, Value};

/// Texts about the same subject (`auth`, `deploy` or `pool`) embed alike;
/// different subjects are far apart, below the candidate floor.
struct SubjectEmbedder;

impl axil_core::TextEmbedder for SubjectEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let t = text.to_lowercase();
        Ok(vec![
            if t.contains("auth") { 1.0 } else { 0.0 },
            if t.contains("deploy") { 1.0 } else { 0.0 },
            if t.contains("pool") { 1.0 } else { 0.0 },
            0.05,
        ])
    }
}

/// Flags a pair when the newer text says "now"; remembers what it was asked.
#[derive(Default)]
struct ScriptedJudge {
    seen: Mutex<Vec<(String, String)>>,
}

impl PairJudge for ScriptedJudge {
    fn judge(&self, pairs: &[(&str, &str)]) -> axil_core::Result<Vec<PairVerdict>> {
        let mut seen = self.seen.lock().unwrap();
        Ok(pairs
            .iter()
            .map(|&(older, newer)| {
                seen.push((older.to_string(), newer.to_string()));
                let flagged = newer.contains(" now ");
                PairVerdict {
                    probability: if flagged { 0.9 } else { 0.1 },
                    flagged,
                }
            })
            .collect())
    }

    fn threshold(&self) -> f64 {
        0.5
    }

    fn name(&self) -> &str {
        "scripted@1"
    }

    fn calibration(&self) -> &str {
        "scripted-calibration"
    }
}

/// A judge shared with the test, so it can be inspected after a pass.
struct Shared(Arc<ScriptedJudge>);

impl PairJudge for Shared {
    fn judge(&self, pairs: &[(&str, &str)]) -> axil_core::Result<Vec<PairVerdict>> {
        self.0.judge(pairs)
    }
    fn threshold(&self) -> f64 {
        self.0.threshold()
    }
    fn name(&self) -> &str {
        self.0.name()
    }
    fn calibration(&self) -> &str {
        self.0.calibration()
    }
}

/// A judge that fails every call.
struct BrokenJudge;

impl PairJudge for BrokenJudge {
    fn judge(&self, _: &[(&str, &str)]) -> axil_core::Result<Vec<PairVerdict>> {
        Err(AxilError::plugin("inference failed"))
    }
    fn threshold(&self) -> f64 {
        0.5
    }
    fn name(&self) -> &str {
        "broken"
    }
    fn calibration(&self) -> &str {
        "none"
    }
}

fn open(dir: &tempfile::TempDir) -> Axil {
    let path = dir.path().join("memory.axil");
    let vector = axil_vector::VectorEngine::open(&path, 4).unwrap();
    Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(SubjectEmbedder))
        .with_graph_engine()
        .unwrap()
        // No insert-time near-duplicate supersede: same-subject texts embed
        // identically here, and this test is about what the review does.
        .with_supersede_threshold(1.5)
        .build()
        .unwrap()
}

fn store(db: &Axil, table: &str, summary: &str) -> RecordId {
    db.insert(table, json!({ "summary": summary })).unwrap().id
}

fn data(db: &Axil, id: &RecordId) -> Value {
    db.get(id).unwrap().unwrap().data
}

fn review(db: &Axil, judge: &Arc<ScriptedJudge>) -> axil_core::ReviewReport {
    let judge = judge.clone();
    db.review_new_records(&ReviewOptions::default(), move || {
        Ok(Box::new(Shared(judge)) as Box<dyn PairJudge>)
    })
    .unwrap()
}

#[test]
fn a_flag_is_an_edge_and_a_review_row_and_changes_no_record() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let old = store(&db, "decisions", "The auth timeout is 30 seconds.");
    let other = store(&db, "decisions", "Deploys happen at 5pm.");
    let new = store(&db, "decisions", "The auth timeout is now 60 seconds.");
    let before: Vec<Value> = [&old, &other, &new]
        .iter()
        .map(|id| data(&db, id))
        .collect();

    let judge = Arc::new(ScriptedJudge::default());
    let report = review(&db, &judge);
    assert_eq!(report.records_reviewed, 3);
    assert_eq!(report.pairs_judged, 1, "only the auth pair is a candidate");
    assert_eq!(report.flagged, 1);
    assert_eq!(report.pending, 0);
    assert_eq!(report.model.as_deref(), Some("scripted@1"));

    // Neither record (nor the bystander) changed; nothing was superseded.
    let after: Vec<Value> = [&old, &other, &new]
        .iter()
        .map(|id| data(&db, id))
        .collect();
    assert_eq!(before, after);
    for id in [&old, &other, &new] {
        assert!(!axil_core::is_superseded_record(
            &db.get(id).unwrap().unwrap()
        ));
    }

    let edges = db
        .edges(&new, Some(POSSIBLY_CONTRADICTS), Direction::Out)
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].to, old);
    let props = &edges[0].properties;
    assert_eq!(props["p"], json!(0.9));
    assert_eq!(props["threshold"], json!(0.5));
    assert_eq!(props["model"], json!("scripted@1"));
    assert_eq!(props["calibration"], json!("scripted-calibration"));
    assert!(props["similarity"].as_f64().unwrap() > 0.9);

    let open_flags = db.possible_contradictions().unwrap();
    assert_eq!(open_flags.len(), 1);
    assert_eq!(open_flags[0].newer.id, new);
    assert_eq!(open_flags[0].older.id, old);
    assert_eq!(open_flags[0].calibration, "scripted-calibration");

    // One row per judged record, unflagged pairs and all.
    assert_eq!(db.count(CONTRADICTION_REVIEW_TABLE).unwrap(), 3);
}

#[test]
fn the_judge_reads_the_older_record_first_and_never_a_newer_one() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    store(&db, "decisions", "The auth timeout is 30 seconds.");
    store(&db, "decisions", "The auth timeout is now 60 seconds.");

    let judge = Arc::new(ScriptedJudge::default());
    review(&db, &judge);
    // The newer record is judged against the older; the older has nothing
    // older to be judged against, so the pair is scored once.
    let seen = judge.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        [(
            "The auth timeout is 30 seconds.".to_string(),
            "The auth timeout is now 60 seconds.".to_string()
        )]
    );
}

#[test]
fn a_pass_with_nothing_due_never_loads_the_judge() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let loads = AtomicUsize::new(0);
    let refuse = || -> axil_core::Result<Box<dyn PairJudge>> {
        loads.fetch_add(1, Ordering::SeqCst);
        Err(AxilError::plugin("must not load"))
    };
    // Empty database.
    let report = db
        .review_new_records(&ReviewOptions::default(), refuse)
        .unwrap();
    assert_eq!(report.load_ms, None);

    // Everything already judged.
    store(&db, "decisions", "The auth timeout is 30 seconds.");
    review(&db, &Arc::new(ScriptedJudge::default()));
    let report = db
        .review_new_records(&ReviewOptions::default(), refuse)
        .unwrap();
    assert_eq!(report.records_reviewed, 0);
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}

#[test]
fn the_window_tables_and_cap_bound_a_pass_newest_first() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let month_ago = chrono::Utc::now() - chrono::Duration::days(30);
    let ancient = db
        .insert_at(
            "decisions",
            json!({"summary": "Pool size is 4."}),
            month_ago,
        )
        .unwrap()
        .id;
    let error = store(&db, "errors", "The auth pool deploy failed.");
    let internal = store(&db, "_beliefs", "The auth timeout is 30 seconds.");
    let a = store(&db, "decisions", "The pool size is 8.");
    let b = store(&db, "context", "Deploys go through the deploy queue.");
    let c = store(&db, "rules", "Never deploy on Fridays.");

    let opts = ReviewOptions {
        max_records: 2,
        ..ReviewOptions::default()
    };
    let due: Vec<RecordId> = db
        .records_due_for_review(&opts)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(due, [c.clone(), b.clone(), a.clone()], "newest first");
    for skipped in [&ancient, &error, &internal] {
        assert!(!due.contains(skipped));
    }

    let judge = Arc::new(ScriptedJudge::default());
    let shared = judge.clone();
    let report = db
        .review_new_records(&opts, move || {
            Ok(Box::new(Shared(shared)) as Box<dyn PairJudge>)
        })
        .unwrap();
    assert_eq!(report.records_reviewed, 2);
    assert_eq!(report.pending, 1);
    let next = db.records_due_for_review(&opts).unwrap();
    assert_eq!(next.len(), 1);
    assert_eq!(next[0].id, a);

    // An old record can still be judged on request.
    db.review_possible_contradictions(&ancient, judge.as_ref(), 5)
        .unwrap();
}

#[test]
fn dismissing_closes_the_flag_for_good() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let old = store(&db, "decisions", "The auth timeout is 30 seconds.");
    let new = store(&db, "decisions", "The auth timeout is now 60 seconds.");
    let judge = Arc::new(ScriptedJudge::default());
    review(&db, &judge);

    assert!(
        !db.dismiss_possible_contradiction(&old, &new).unwrap(),
        "wrong direction"
    );
    assert!(db.dismiss_possible_contradiction(&new, &old).unwrap());
    assert!(
        !db.dismiss_possible_contradiction(&new, &old).unwrap(),
        "already closed"
    );
    assert!(db.possible_contradictions().unwrap().is_empty());
    assert!(db
        .edges(&new, Some(POSSIBLY_CONTRADICTS), Direction::Out)
        .unwrap()
        .is_empty());

    // The score stays in the log, and judging again keeps the dismissal.
    db.review_possible_contradictions(&new, judge.as_ref(), 5)
        .unwrap();
    assert!(db.possible_contradictions().unwrap().is_empty());
    assert!(db
        .edges(&new, Some(POSSIBLY_CONTRADICTS), Direction::Out)
        .unwrap()
        .is_empty());
    let rows = db.contradiction_reviews().unwrap();
    let row = rows
        .iter()
        .find(|r| r.data["record"] == json!(new.as_str()))
        .unwrap();
    assert_eq!(row.data["pairs"][0]["probability"], json!(0.9));
    assert_eq!(row.data["dismissed"], json!([old.as_str()]));
    assert_eq!(
        db.count(CONTRADICTION_REVIEW_TABLE).unwrap(),
        2,
        "rewritten, not duplicated"
    );
}

#[test]
fn superseding_the_older_record_closes_the_flag() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let old = store(&db, "decisions", "The auth timeout is 30 seconds.");
    let new = store(&db, "decisions", "The auth timeout is now 60 seconds.");
    review(&db, &Arc::new(ScriptedJudge::default()));
    assert_eq!(db.possible_contradictions().unwrap().len(), 1);

    let newer = db.get(&new).unwrap().unwrap();
    assert!(db.mark_superseded(&old, &newer, None).unwrap());
    assert!(db.possible_contradictions().unwrap().is_empty());
}

#[test]
fn boot_shows_one_line_only_while_a_flag_is_open() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    let block = |db: &Axil| {
        collect_extension_blocks(db)
            .into_iter()
            .find(|(id, _)| id == "possible_contradictions")
            .map(|(_, text)| text)
    };
    let old = store(&db, "decisions", "The auth timeout is 30 seconds.");
    let new = store(&db, "decisions", "The auth timeout is now 60 seconds.");
    assert_eq!(block(&db), None);

    review(&db, &Arc::new(ScriptedJudge::default()));
    let text = block(&db).expect("an open flag shows in boot");
    assert!(text.contains("1 possible contradiction"), "{text}");
    assert!(text.contains("axil contradictions"), "{text}");
    assert_eq!(text.lines().count(), 2, "a heading and one line: {text}");

    db.dismiss_possible_contradiction(&new, &old).unwrap();
    assert_eq!(block(&db), None);
}

#[test]
fn a_judge_error_stops_the_pass_and_writes_nothing_for_that_record() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    store(&db, "decisions", "The auth timeout is 30 seconds.");
    store(&db, "decisions", "The auth timeout is now 60 seconds.");
    let err = db
        .review_new_records(&ReviewOptions::default(), || {
            Ok(Box::new(BrokenJudge) as Box<dyn PairJudge>)
        })
        .unwrap_err();
    assert!(err.to_string().contains("inference failed"), "{err}");
    // The newest record failed first, so nothing was recorded and both are
    // still due next time.
    assert_eq!(db.count(CONTRADICTION_REVIEW_TABLE).unwrap(), 0);
    assert_eq!(
        db.records_due_for_review(&ReviewOptions::default())
            .unwrap()
            .len(),
        2
    );
}
