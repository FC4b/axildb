//! Guards against CLI paths that destroy or poison the default vector store:
//! `reembed --table` (preserving the other tables, crash recovery, unreadable
//! stores) and `store --vector` into the embedder-owned default space.
//!
//! Each test drives the built `axil` binary against a temp database. Tests
//! that really re-embed need the bge-small model on disk and skip loudly when
//! it is absent, so they never download one.

#![cfg(feature = "embed")]

use std::path::{Path, PathBuf};
use std::process::Command;

use axil_core::plugin::VectorIndex;
use axil_core::RecordId;
use axil_vector::models::EmbeddingModel;
use axil_vector::VectorEngine;
use serde_json::Value;

const DIMS: usize = 384;

fn axil(db: &Path, args: &[&str]) -> (String, String, i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_axil"))
        .arg("--db")
        .arg(db)
        .args(args)
        .env_remove("AXIL_DB")
        .current_dir(db.parent().unwrap())
        .output()
        .expect("failed to run axil");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

fn parse_json(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not JSON ({e}): {stdout}"))
}

/// `axil store`, returning the new record's id.
fn store(db: &Path, table: &str, data: &str, extra: &[&str]) -> RecordId {
    let mut args = vec!["store", table, data];
    args.extend_from_slice(extra);
    let (stdout, stderr, code) = axil(db, &args);
    assert_eq!(code, 0, "store failed: {stderr}");
    RecordId::from_string(parse_json(&stdout)["id"].as_str().unwrap()).unwrap()
}

fn vec_path(db: &Path) -> PathBuf {
    axil_vector::vector_db_path(db)
}

fn backup_path(db: &Path) -> PathBuf {
    vec_path(db).with_extension("reembed-bak")
}

/// Each id's vector in the default store, read straight from the file.
fn stored_vectors(db: &Path, ids: &[&RecordId]) -> Vec<Option<Vec<f32>>> {
    let engine = VectorEngine::open(db, DIMS).unwrap();
    ids.iter()
        .map(|id| engine.get_vector(id).unwrap())
        .collect()
}

fn model_available() -> bool {
    let available = axil_vector::download::is_model_available(&EmbeddingModel::BgeSmall);
    if !available {
        eprintln!("skipping: bge-small model not downloaded");
    }
    available
}

/// Two `decisions` and two `notes` records, all embedded into the default
/// store. Returns `(decision ids, note ids)`.
fn seed_embedded(db: &Path) -> (Vec<RecordId>, Vec<RecordId>) {
    let embed = ["--embed", "summary"];
    let decisions = vec![
        store(
            db,
            "decisions",
            r#"{"summary":"Use redb as the embedded storage engine"}"#,
            &embed,
        ),
        store(
            db,
            "decisions",
            r#"{"summary":"Ship the CLI as a single static binary"}"#,
            &embed,
        ),
    ];
    let notes = vec![
        store(
            db,
            "notes",
            r#"{"summary":"Grocery list: apples, oat milk, coffee beans"}"#,
            &embed,
        ),
        store(
            db,
            "notes",
            r#"{"summary":"The dentist appointment moved to Thursday"}"#,
            &embed,
        ),
    ];
    (decisions, notes)
}

#[test]
fn scoped_reembed_preserves_other_tables_vectors() {
    if !model_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let (decisions, notes) = seed_embedded(&db);
    let before = stored_vectors(&db, &[&notes[0], &notes[1]]);
    assert!(
        before.iter().all(Option::is_some),
        "setup: notes must be embedded"
    );

    let (stdout, stderr, code) = axil(
        &db,
        &[
            "reembed",
            "--model",
            "bge-small",
            "--field",
            "summary",
            "--table",
            "decisions",
        ],
    );
    assert_eq!(code, 0, "reembed failed: {stderr}");
    let report = parse_json(&stdout);
    assert_eq!(report["reembedded"], 2, "report: {report}");
    assert_eq!(report["preserved_other_tables"], 2, "report: {report}");

    assert_eq!(
        stored_vectors(&db, &[&notes[0], &notes[1]]),
        before,
        "a --table decisions re-embed must leave the notes vectors untouched"
    );
    assert!(stored_vectors(&db, &[&decisions[0], &decisions[1]])
        .iter()
        .all(Option::is_some));
    assert!(
        !backup_path(&db).exists(),
        "backup must be removed after success"
    );
}

#[test]
fn scoped_reembed_refuses_an_unreadable_vector_store() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    store(
        &db,
        "decisions",
        r#"{"summary":"Use redb as the embedded storage engine"}"#,
        &[],
    );
    store(
        &db,
        "notes",
        r#"{"summary":"The dentist appointment moved to Thursday"}"#,
        &[],
    );
    let garbage = b"not a redb file".to_vec();
    std::fs::write(vec_path(&db), &garbage).unwrap();

    let (_, stderr, code) = axil(
        &db,
        &[
            "reembed",
            "--model",
            "bge-small",
            "--field",
            "summary",
            "--table",
            "decisions",
        ],
    );
    assert_ne!(
        code, 0,
        "a scoped re-embed over an unreadable store must fail"
    );
    assert!(stderr.contains("vector store"), "stderr: {stderr}");
    assert_eq!(
        std::fs::read(vec_path(&db)).unwrap(),
        garbage,
        "the existing store must be left exactly as it was"
    );
    assert!(!backup_path(&db).exists());
}

/// A run killed mid-rebuild leaves the original store in the backup slot and
/// a partial store live. Seed that state: the partial holds a wrong vector
/// for one note and nothing for the other.
fn simulate_interrupted_reembed(db: &Path, original: &[(RecordId, Vec<f32>)]) {
    {
        let engine = VectorEngine::open(db, DIMS).unwrap();
        for (id, v) in original {
            engine.add(id.clone(), v).unwrap();
        }
    }
    std::fs::rename(vec_path(db), backup_path(db)).unwrap();
    let partial = VectorEngine::open(db, DIMS).unwrap();
    let mut wrong = vec![0.0_f32; DIMS];
    wrong[DIMS - 1] = 1.0;
    partial.add(original[0].0.clone(), &wrong).unwrap();
}

fn unit(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIMS];
    v[i] = 1.0;
    v
}

#[test]
fn reembed_restores_the_backup_of_an_interrupted_run_before_anything_else() {
    // Model-free: an empty work set returns before any embedding, which is
    // exactly when a leftover backup used to be ignored.
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let a = store(&db, "notes", r#"{"summary":"first note"}"#, &[]);
    let b = store(&db, "notes", r#"{"summary":"second note"}"#, &[]);
    simulate_interrupted_reembed(&db, &[(a.clone(), unit(0)), (b.clone(), unit(1))]);

    let (_, stderr, code) = axil(
        &db,
        &[
            "reembed",
            "--model",
            "bge-small",
            "--field",
            "no_such_field",
            "--table",
            "notes",
        ],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(
        !backup_path(&db).exists(),
        "the backup must be moved back, not left behind"
    );
    assert_eq!(
        stored_vectors(&db, &[&a, &b]),
        vec![Some(unit(0)), Some(unit(1))],
        "the original vectors must be live again"
    );
}

#[test]
fn scoped_reembed_after_an_interrupted_run_carries_over_the_original_vectors() {
    if !model_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let (decisions, notes) = seed_embedded(&db);
    let original: Vec<(RecordId, Vec<f32>)> = notes
        .iter()
        .zip(stored_vectors(&db, &[&notes[0], &notes[1]]))
        .map(|(id, v)| (id.clone(), v.unwrap()))
        .collect();
    simulate_interrupted_reembed(&db, &original);

    let (stdout, stderr, code) = axil(
        &db,
        &[
            "reembed",
            "--model",
            "bge-small",
            "--field",
            "summary",
            "--table",
            "decisions",
        ],
    );
    assert_eq!(code, 0, "reembed failed: {stderr}");
    assert_eq!(parse_json(&stdout)["preserved_other_tables"], 2);
    assert!(!backup_path(&db).exists());
    assert_eq!(
        stored_vectors(&db, &[&notes[0], &notes[1]]),
        original
            .into_iter()
            .map(|(_, v)| Some(v))
            .collect::<Vec<_>>(),
        "other tables must come from the backup, not the half-built store"
    );
    assert!(stored_vectors(&db, &[&decisions[0], &decisions[1]])
        .iter()
        .all(Option::is_some));
}

#[test]
fn raw_vector_for_the_default_space_must_match_the_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");

    let (_, stderr, code) = axil(&db, &["store", "t", r#"{"n":"a"}"#, "--vector", "[1,0,0]"]);
    assert_ne!(
        code, 0,
        "a 3-dim vector must not claim the 384-dim default space"
    );
    assert!(
        stderr.contains("--space"),
        "error should point at --space: {stderr}"
    );
    assert!(!vec_path(&db).exists(), "no default store may be created");

    // The database stays usable, and a named space still takes any length.
    store(&db, "t", r#"{"n":"b"}"#, &[]);
    let (stdout, stderr, code) = axil(
        &db,
        &[
            "store",
            "t",
            r#"{"n":"c"}"#,
            "--vector",
            "[1,0,0]",
            "--space",
            "fp",
        ],
    );
    assert_eq!(code, 0, "named-space store failed: {stderr}");
    assert_eq!(parse_json(&stdout)["vector_dims"], 3);
    store(&db, "t", r#"{"n":"d"}"#, &[]);
}
