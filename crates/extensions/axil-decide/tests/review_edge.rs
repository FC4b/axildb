//! The U1 judge behind the review pass, end to end on a real database: a
//! real update gets a `possibly_contradicts` edge, a refinement that still
//! holds does not, and no record is changed.
//!
//! Needs the pinned U1 model (87 MB), so it is ignored by default:
//! `cargo test --release -p axil-decide --test review_edge -- --ignored`.
//! Candidates come from a keyword embedder, so no embedding model is needed.

use axil_core::util::edge_types::POSSIBLY_CONTRADICTS;
use axil_core::{Axil, Direction, PairJudge, ReviewOptions};
use axil_decide::{Calibration, PairClassifier, NLI_DEBERTA_V3_XSMALL_QINT8 as U1};
use axil_graph::AxilBuilderGraphExt;
use serde_json::json;

/// Records about CI embed alike, records about deploys alike, and the two
/// subjects far apart, below the candidate floor.
struct SubjectEmbedder;

impl axil_core::TextEmbedder for SubjectEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        Ok(vec![
            if text.contains("CI") { 1.0 } else { 0.0 },
            if text.contains("Deploys") { 1.0 } else { 0.0 },
            0.05,
        ])
    }
}

#[test]
#[ignore = "needs the pinned U1 model: run with --ignored"]
fn the_u1_judge_flags_an_update_and_changes_nothing() {
    let model_dir = U1.default_dir().expect("home directory");
    U1.download(&model_dir).expect("U1 model");
    let judge = PairClassifier::load(&U1, &model_dir, Calibration::u1()).expect("load U1");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    let vector = axil_vector::VectorEngine::open(&path, 3).unwrap();
    let db = Axil::open(&path)
        .with_vector_index(Box::new(vector))
        .with_embedder(Box::new(SubjectEmbedder))
        .with_graph_engine()
        .unwrap()
        .with_supersede_threshold(1.5)
        .build()
        .unwrap();

    // Two pairs from the U1 repo set: r1-08, an update the judge flags
    // (p 0.769 in tests/fixtures/u1_parity.json), and r0-18, a refinement
    // where both still hold (p 0.194).
    let store = |s: &str| db.insert("decisions", json!({ "summary": s })).unwrap();
    let no_ci = store("The project has no CI.");
    let five_pm = store("Deploys happen at 5pm.");
    let ci = store(
        "CI runs build, tests and the recall-quality gate on every PR via .github/workflows/ci.yml.",
    );
    let weekdays = store("Deploys happen at 5pm on weekdays; hotfixes can ship any time.");

    let report = db
        .review_new_records(&ReviewOptions::default(), || {
            Ok(Box::new(judge) as Box<dyn PairJudge>)
        })
        .unwrap();
    assert_eq!(report.records_reviewed, 4);
    assert_eq!(report.pairs_judged, 2);
    assert_eq!(report.flagged, 1);

    let edges = db
        .edges(&ci.id, Some(POSSIBLY_CONTRADICTS), Direction::Out)
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].to, no_ci.id);
    let p = edges[0].properties["p"].as_f64().unwrap();
    assert!((p - 0.769).abs() < 1e-3, "p = {p}");
    assert_eq!(edges[0].properties["calibration"], json!("u1@2026-10-04"));
    assert!(db
        .edges(&weekdays.id, Some(POSSIBLY_CONTRADICTS), Direction::Out)
        .unwrap()
        .is_empty());

    for record in [&no_ci, &five_pm, &ci, &weekdays] {
        let now = db.get(&record.id).unwrap().unwrap();
        assert_eq!(now.data, record.data, "{} changed", record.id);
    }
}
