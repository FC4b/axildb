//! The default vector store must have the embedding model's dimensions: every
//! open attaches the model from `axil.toml` and refuses a store of any other
//! size. `init` and `install` size a new store for that model, and `heal`,
//! which can neither create nor resize a store, names the command that can.
//!
//! None of these runs opens an embedder, so no model is loaded or downloaded.
//! `HOME` points at the temp dir so a user-level `~/.config/axil/config.toml`
//! cannot change the configured model.

#![cfg(feature = "embed")]

use std::path::Path;
use std::process::{Command, Output};

use axil_core::VectorIndex;
use serde_json::Value;

/// Run `axil` with `args` from `cwd`, isolated from the caller's environment.
fn axil_in(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axil"))
        .args(args)
        .env_remove("AXIL_DB")
        .env_remove("CLAUDE_PROJECT_DIR")
        .env("HOME", cwd)
        .current_dir(cwd)
        .output()
        .expect("failed to run axil")
}

fn stdout_json(output: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not JSON ({e}): {stdout}"))
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn configure_model(dir: &Path, model: &str) {
    std::fs::write(
        dir.join("axil.toml"),
        format!("[database]\nembedding_model = \"{model}\"\n"),
    )
    .unwrap();
}

fn stored_dims(db: &Path) -> Option<usize> {
    axil_vector::read_stored_dimensions(db).unwrap()
}

#[test]
fn init_sizes_the_vector_store_for_the_configured_model() {
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "bge-base");
    let db = dir.path().join("memory.axil");

    let output = axil_in(dir.path(), &["init", db.to_str().unwrap()]);
    assert!(output.status.success(), "init failed: {}", stderr(&output));
    assert_eq!(stdout_json(&output)["vector_dims"], 768);
    assert_eq!(stored_dims(&db), Some(768));
}

#[test]
fn init_uses_bge_small_dimensions_without_a_configured_model() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");

    let output = axil_in(dir.path(), &["init", db.to_str().unwrap()]);
    assert!(output.status.success(), "init failed: {}", stderr(&output));
    assert_eq!(stored_dims(&db), Some(384));
}

#[test]
fn init_vector_dims_flag_overrides_the_configured_model() {
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "bge-base");
    let db = dir.path().join("memory.axil");

    let output = axil_in(
        dir.path(),
        &["init", db.to_str().unwrap(), "--vector-dims", "384"],
    );
    assert!(output.status.success(), "init failed: {}", stderr(&output));
    assert_eq!(stored_dims(&db), Some(384));
    // The configured model makes 768-dimension vectors, so this store will
    // not open until axil.toml changes: say so.
    let err = stderr(&output);
    assert!(
        err.contains("--vector-dims 384 does not match the embedding model bge-base-en-v1.5"),
        "init should warn about the mismatch: {err}"
    );
}

#[test]
fn init_vector_dims_flag_matching_the_configured_model_is_quiet() {
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "bge-base");
    let db = dir.path().join("memory.axil");

    let output = axil_in(
        dir.path(),
        &["init", db.to_str().unwrap(), "--vector-dims", "768"],
    );
    assert!(output.status.success(), "init failed: {}", stderr(&output));
    assert_eq!(stored_dims(&db), Some(768));
    assert!(
        !stderr(&output).contains("--vector-dims"),
        "no warning expected: {}",
        stderr(&output)
    );
}

#[test]
fn install_sizes_the_vector_store_for_the_configured_model() {
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "bge-base");

    let output = axil_in(dir.path(), &["install", "--no-agents-md"]);
    assert!(
        output.status.success(),
        "install failed: {}",
        stderr(&output)
    );
    assert_eq!(
        stored_dims(&dir.path().join(".axil/memory.axil")),
        Some(768)
    );
}

#[test]
fn heal_reindex_reports_a_missing_vector_store() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let db_arg = db.to_str().unwrap();
    // A plain store (no `--embed`) creates no vector store.
    let output = axil_in(
        dir.path(),
        &[
            "--db",
            db_arg,
            "store",
            "notes",
            r#"{"summary":"a note to embed"}"#,
        ],
    );
    assert!(output.status.success(), "store failed: {}", stderr(&output));
    assert_eq!(stored_dims(&db), None, "setup: no vector store");

    let output = axil_in(dir.path(), &["--db", db_arg, "heal", "--reindex"]);
    assert!(output.status.success(), "heal failed: {}", stderr(&output));
    let report = stdout_json(&output);
    assert_eq!(
        report["vector_store"]["status"], "missing",
        "report: {report}"
    );
    let fix = report["vector_store"]["fix"].as_str().unwrap();
    assert!(fix.contains(&format!("axil init {db_arg}")), "fix: {fix}");
    assert!(fix.contains("384 dimensions"), "fix: {fix}");
}

#[test]
fn heal_refuses_a_vector_store_sized_for_another_model() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let db_arg = db.to_str().unwrap();
    // Created at bge-small's size, then the config moves to bge-base.
    let output = axil_in(dir.path(), &["init", db_arg]);
    assert!(output.status.success(), "init failed: {}", stderr(&output));
    configure_model(dir.path(), "bge-base");

    for args in [&["heal", "--reindex"][..], &["heal"][..]] {
        let mut full = vec!["--db", db_arg];
        full.extend_from_slice(args);
        let output = axil_in(dir.path(), &full);
        assert!(
            !output.status.success(),
            "{args:?} must fail on a 384-dim store"
        );
        let err = stderr(&output);
        assert!(err.contains("384-dimension"), "{args:?}: {err}");
        assert!(err.contains("768-dimension"), "{args:?}: {err}");
        assert!(
            err.contains("reembed --model bge-base-en-v1.5 --field"),
            "{args:?} should name the reembed command: {err}"
        );
    }
    assert_eq!(
        stored_dims(&db),
        Some(384),
        "heal must leave the store alone"
    );
}

#[test]
fn heal_reports_repairing_a_vector_store_that_was_not_closed_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let live = dir.path().join("live.axil");
    let db = dir.path().join("memory.axil");
    // A store as a killed writer leaves it: copied while its engine still
    // holds it open, so it has committed data but was never closed. Sized
    // for bge-base while the config is bge-small, so heal stops before the
    // open (which would load a model) and after the probe that repairs it.
    let engine = axil_vector::VectorEngine::open(&live, 768).unwrap();
    engine.add(axil_core::RecordId::new(), &[0.5; 768]).unwrap();
    std::fs::copy(
        axil_vector::vector_db_path(&live),
        axil_vector::vector_db_path(&db),
    )
    .unwrap();
    drop(engine);

    let output = axil_in(dir.path(), &["--db", db.to_str().unwrap(), "heal"]);
    assert!(!output.status.success(), "a 768-dim store must stop heal");
    let err = stderr(&output);
    assert!(
        err.contains("was not closed cleanly; repaired it"),
        "heal's probe repaired the store, so heal must say so: {err}"
    );
    assert!(err.contains("768-dimension"), "{err}");
}
