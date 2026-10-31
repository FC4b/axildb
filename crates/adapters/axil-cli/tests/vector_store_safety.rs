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

/// Where earlier versions set the live store aside while they rebuilt it.
fn backup_path(db: &Path) -> PathBuf {
    vec_path(db).with_extension("reembed-bak")
}

/// Where a re-embed builds the store that replaces the live one.
fn rebuild_path(db: &Path) -> PathBuf {
    db.with_file_name(format!(
        "{}.vec-rebuild",
        db.file_name().unwrap().to_str().unwrap()
    ))
}

/// The lock a re-embed holds for its whole run.
fn lock_path(db: &Path) -> PathBuf {
    db.with_file_name(format!(
        "{}.reembed-lock",
        db.file_name().unwrap().to_str().unwrap()
    ))
}

/// Each id's vector in the default store, read straight from the file.
fn stored_vectors(db: &Path, ids: &[&RecordId]) -> Vec<Option<Vec<f32>>> {
    let engine = VectorEngine::open(db, DIMS).unwrap();
    ids.iter()
        .map(|id| engine.get_vector(id).unwrap())
        .collect()
}

/// Whether `model` is on disk. Without it a test that embeds skips, and says
/// so on stderr past the test harness's capture; when
/// `AXIL_TEST_REQUIRE_MODELS` names the model (comma-separated, or `all`), it
/// fails instead, so a CI job that provides the model can't pass by skipping.
fn model_on_disk(model: &EmbeddingModel) -> bool {
    if axil_vector::download::is_model_available(model) {
        return true;
    }
    let name = model.name();
    let required = std::env::var("AXIL_TEST_REQUIRE_MODELS").unwrap_or_default();
    assert!(
        !required
            .split(',')
            .any(|m| m.trim() == name || m.trim() == "all"),
        "{name} is not downloaded, and AXIL_TEST_REQUIRE_MODELS requires it"
    );
    use std::io::Write as _;
    let _ = writeln!(std::io::stderr(), "skipping: {name} model not downloaded");
    false
}

fn model_available() -> bool {
    model_on_disk(&EmbeddingModel::BgeSmall)
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
        !rebuild_path(&db).exists(),
        "the replacement is moved into place"
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

fn unit(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIMS];
    v[i] = 1.0;
    v
}

/// A store with `vectors` in it, as the live store of `db`.
fn live_store(db: &Path, vectors: &[(RecordId, Vec<f32>)]) {
    let engine = VectorEngine::open(db, DIMS).unwrap();
    for (id, v) in vectors {
        engine.add(id.clone(), v).unwrap();
    }
}

/// Earlier versions rebuilt the store in place, with the original set aside
/// as `<db>.reembed-bak`. After one of them was interrupted, which copy is
/// right depends on what was written since, so a re-embed stops and says so
/// rather than guess, and touches neither.
#[test]
fn reembed_refuses_the_backup_an_earlier_version_left() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let a = store(&db, "notes", r#"{"summary":"first note"}"#, &[]);
    live_store(&db, &[(a.clone(), unit(0))]);
    std::fs::rename(vec_path(&db), backup_path(&db)).unwrap();
    live_store(&db, &[(a.clone(), unit(1))]);

    for args in [
        &["reembed"][..],
        &["reembed", "--field", "summary"][..],
        &["reembed", "--field", "no_such_field", "--table", "notes"][..],
    ] {
        let (_, stderr, code) = axil(&db, args);
        assert_ne!(code, 0, "{args:?} must stop");
        assert!(
            stderr.contains("earlier version") && stderr.contains("reembed-bak"),
            "{args:?}: {stderr}"
        );
    }
    assert_eq!(stored_vectors(&db, &[&a]), vec![Some(unit(1))]);
    let backup = VectorEngine::open_space_at(&backup_path(&db), None).unwrap();
    assert_eq!(backup.get_vector(&a).unwrap(), Some(unit(0)));
}

/// A re-embed of a database that isn't there names it, and creates nothing:
/// neither a database nor the lock file beside it, whose directory may not
/// exist either.
#[test]
fn reembed_of_a_missing_database_names_it() {
    let dir = tempfile::tempdir().unwrap();
    for db in [
        dir.path().join("memory.axil"),
        dir.path().join("no-such-dir").join("memory.axil"),
    ] {
        for args in [&["reembed"][..], &["reembed", "--field", "summary"][..]] {
            let output = Command::new(env!("CARGO_BIN_EXE_axil"))
                .arg("--db")
                .arg(&db)
                .args(args)
                .env_remove("AXIL_DB")
                .current_dir(dir.path())
                .output()
                .expect("failed to run axil");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_ne!(output.status.code(), Some(0), "{args:?} must stop");
            assert!(
                stderr.contains("no database at") && stderr.contains(&*db.to_string_lossy()),
                "{args:?}: {stderr}"
            );
        }
        assert!(!db.exists());
        assert!(!lock_path(&db).exists());
    }
    assert!(!dir.path().join("no-such-dir").exists());
}

/// A run killed mid-rebuild leaves the live store as it was, and an
/// unfinished replacement beside it, which no open reads and the next
/// re-embed deletes.
#[test]
fn reembed_deletes_the_replacement_an_interrupted_run_left() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let a = store(&db, "notes", r#"{"summary":"first note"}"#, &[]);
    live_store(&db, &[(a.clone(), unit(0))]);
    {
        let partial =
            axil_vector::create_store_for_model_at(&rebuild_path(&db), &EmbeddingModel::BgeSmall)
                .unwrap();
        partial.add(a.clone(), &unit(1)).unwrap();
    }

    // Model-free: an empty work set returns before any embedding.
    let (_, stderr, code) = axil(
        &db,
        &["reembed", "--field", "no_such_field", "--table", "notes"],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stderr.contains("interrupted re-embed"), "{stderr}");
    assert!(!rebuild_path(&db).exists());
    assert_eq!(stored_vectors(&db, &[&a]), vec![Some(unit(0))]);
}

/// While one re-embed runs, a second stops before it reads or deletes
/// anything: it would otherwise take the running one's replacement for an
/// interrupted run's and delete it, or move its own over the live store. The
/// lock is held here the way the running one holds it.
#[test]
fn a_second_reembed_stops_while_one_runs() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let a = store(&db, "notes", r#"{"summary":"first note"}"#, &[]);
    live_store(&db, &[(a.clone(), unit(0))]);
    {
        let partial =
            axil_vector::create_store_for_model_at(&rebuild_path(&db), &EmbeddingModel::BgeSmall)
                .unwrap();
        partial.add(a.clone(), &unit(1)).unwrap();
    }
    let running = std::fs::File::create(lock_path(&db)).unwrap();
    running.lock().unwrap();

    for args in [
        &["reembed"][..],
        &["reembed", "--field", "no_such_field", "--table", "notes"][..],
    ] {
        let (_, stderr, code) = axil(&db, args);
        assert_ne!(code, 0, "{args:?} must stop");
        assert!(
            stderr.contains("another `axil reembed`"),
            "{args:?}: {stderr}"
        );
    }
    assert!(
        rebuild_path(&db).exists(),
        "the running one's replacement is left alone"
    );
    assert_eq!(stored_vectors(&db, &[&a]), vec![Some(unit(0))]);

    // Once it is gone, the next one goes ahead (model-free: an empty work
    // set returns before any embedding), and releases the lock as it ends.
    drop(running);
    let (_, stderr, code) = axil(
        &db,
        &["reembed", "--field", "no_such_field", "--table", "notes"],
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(!rebuild_path(&db).exists());
    assert!(!lock_path(&db).exists());
}

#[test]
fn scoped_reembed_after_an_interrupted_run_keeps_the_live_vectors() {
    if !model_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let (decisions, notes) = seed_embedded(&db);
    let original = stored_vectors(&db, &[&notes[0], &notes[1]]);
    {
        let partial =
            axil_vector::create_store_for_model_at(&rebuild_path(&db), &EmbeddingModel::BgeSmall)
                .unwrap();
        partial.add(notes[0].clone(), &unit(DIMS - 1)).unwrap();
    }

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
    assert!(!rebuild_path(&db).exists());
    assert_eq!(
        stored_vectors(&db, &[&notes[0], &notes[1]]),
        original,
        "other tables come from the live store, not the unfinished one"
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

#[test]
fn add_vector_to_the_default_space_must_match_the_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let id = store(&db, "t", r#"{"n":"a"}"#, &[]).to_string();

    // `--dimensions` would otherwise size a brand-new default store to fit.
    let (_, stderr, code) = axil(&db, &["add-vector", &id, "[1,0,0]", "--dimensions", "3"]);
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
    let (stdout, stderr, code) = axil(&db, &["add-vector", &id, "[1,0,0]", "--space", "fp"]);
    assert_eq!(code, 0, "named-space add-vector failed: {stderr}");
    assert_eq!(parse_json(&stdout)["dimensions"], 3);
    store(&db, "t", r#"{"n":"c"}"#, &[]);
}
