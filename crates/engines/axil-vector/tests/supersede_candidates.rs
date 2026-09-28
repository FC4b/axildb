//! Supersession with real embeddings: similarity nominates candidates above a
//! recall-oriented floor, and the judge decides. Real updates score well below
//! the old 0.92 gate under bge-small, so these pairs never reached the judge
//! before. Needs the `embed` feature and the bge-small model, so it runs with
//! `cargo test -p axil-vector --features embed` (locally and nightly), not in
//! the model-free per-PR job.

#![cfg(feature = "embed")]

use axil_core::{Axil, RecordId, SUPERSEDE_CANDIDATE_FLOOR};
use axil_vector::models::EmbeddingModel;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

fn store(db: &Axil, summary: &str) -> RecordId {
    db.insert("decisions", json!({ "summary": summary }))
        .unwrap()
        .id
}

fn superseded(db: &Axil, id: &RecordId) -> bool {
    db.get(id).unwrap().unwrap().data.get("_superseded") == Some(&json!(true))
}

#[test]
fn a_real_update_below_the_old_gate_is_caught_and_look_alikes_are_not() {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("memory.axil"))
        .with_embedder_model(EmbeddingModel::BgeSmall)
        .unwrap()
        .build()
        .unwrap();

    let intel = store(
        &db,
        "Releases ship six targets, including x86_64-apple-darwin for Intel Macs.",
    );
    let deploys = store(&db, "Deploys happen at 5pm.");
    let dark = store(&db, "The user prefers dark mode in the editor.");

    // An update (cosine ~0.84 under bge-small: a candidate, never past 0.92).
    let dropped = store(
        &db,
        "Dropped the x86_64-apple-darwin release target; releases now ship five triples.",
    );
    let new = db.get(&dropped).unwrap().unwrap();
    let candidates = db.supersede_candidates(&new, 5).unwrap();
    let (_, similarity) = candidates
        .iter()
        .find(|(r, _)| r.id == intel)
        .expect("the replaced record must be a candidate");
    assert!(*similarity >= SUPERSEDE_CANDIDATE_FLOOR && *similarity < 0.92);
    db.detect_conflicts(&dropped).unwrap();
    assert!(
        superseded(&db, &intel),
        "the update must retire the old claim"
    );

    // A refinement and a preference about another subject both still hold.
    let refined = store(
        &db,
        "Deploys happen at 5pm on weekdays; hotfixes can ship any time.",
    );
    let light = store(&db, "The user prefers light mode for printed docs.");
    db.detect_conflicts(&refined).unwrap();
    db.detect_conflicts(&light).unwrap();
    assert!(
        !superseded(&db, &deploys),
        "a refinement does not retire the fact"
    );
    assert!(
        !superseded(&db, &dark),
        "a different subject does not retire the fact"
    );
}
