//! A vector store that is dirty, busy, or corrupt must never make recall
//! silently keyword-only. A dirty store (left by a killed writer) is repaired
//! on open; a corrupt one is reported by doctor, heal, boot, and the recall
//! context block that hooks inject.

use std::path::{Path, PathBuf};
use std::process::Command;

use axil_core::{Axil, BootOptions, DegradedEngine, Severity, VectorIndex};
use serde_json::{json, Value};

fn axil_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("target/debug/axil");
    path.set_extension(std::env::consts::EXE_EXTENSION);
    assert!(
        path.exists(),
        "axil binary not found at {}. Run `cargo build -p axildb` first.",
        path.display()
    );
    path
}

/// Run `axil --db <db> <args>` from the db's directory, so no `axil.toml`
/// above the temp dir can change the result. Returns (stdout, stderr, code).
fn axil(db: &Path, args: &[&str]) -> (String, String, i32) {
    let out = Command::new(axil_bin())
        .current_dir(db.parent().unwrap())
        .arg("--db")
        .arg(db)
        .args(args)
        .env_remove("AXIL_DB")
        .output()
        .expect("failed to run axil");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// A core DB holding one decision, and a `.vec` file that is not a redb file.
fn db_with_corrupt_vector_store(dir: &Path) -> PathBuf {
    let path = dir.join("memory.axil");
    Axil::open(&path)
        .build()
        .unwrap()
        .insert(
            "decisions",
            json!({"summary": "the connection pool size is 32"}),
        )
        .unwrap();
    std::fs::write(axil_vector::vector_db_path(&path), b"not a redb file").unwrap();
    path
}

fn vector_warning() -> DegradedEngine {
    DegradedEngine {
        engine: "vector".into(),
        reason: "memory.axil.vec could not be opened".into(),
        fix: "rebuild it".into(),
    }
}

#[test]
fn a_degraded_engine_is_reported_by_doctor_heal_and_boot() {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("memory.axil"))
        .with_degraded_engine(vector_warning())
        .build()
        .unwrap();
    assert_eq!(db.degraded_engines(), &[vector_warning()]);

    let doctor = db.doctor().unwrap();
    let check = doctor
        .checks
        .iter()
        .find(|c| c.name == "vector_engine")
        .expect("doctor must report the unavailable engine");
    assert_eq!(check.status, Severity::Error);
    assert_eq!(check.fix.as_deref(), Some("rebuild it"));
    assert_eq!(doctor.status, Severity::Error);

    let problem = db
        .detect_problems()
        .into_iter()
        .find(|p| p.detector == "engine_unavailable")
        .expect("heal must see the unavailable engine");
    assert_eq!(problem.severity, Severity::Error);
    assert!(!problem.auto_fixable);

    let boot = serde_json::to_value(db.boot(BootOptions::default()).unwrap()).unwrap();
    assert_eq!(boot["degraded"][0]["engine"], "vector");
}

#[test]
fn a_healthy_db_reports_nothing_degraded() {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("memory.axil")).build().unwrap();
    assert!(db.degraded_engines().is_empty());
    let boot = serde_json::to_value(db.boot(BootOptions::default()).unwrap()).unwrap();
    assert!(
        boot.get("degraded").is_none(),
        "no key when healthy: {boot}"
    );
}

#[test]
fn cli_reports_a_corrupt_vector_store_everywhere() {
    let dir = tempfile::tempdir().unwrap();
    let db = db_with_corrupt_vector_store(dir.path());

    let (stdout, stderr, code) = axil(&db, &["doctor"]);
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert_eq!(report["status"], "error", "{stdout}");
    assert_eq!(code, 2, "doctor exits 2 on an error");
    assert!(
        report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["name"] == "vector_engine" && c["status"] == "error"),
        "{stdout}"
    );
    assert!(stderr.contains("vector engine unavailable"), "{stderr}");

    let (stdout, _, code) = axil(&db, &["boot"]);
    assert_eq!(code, 0, "boot still works without vectors");
    let boot: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert!(
        boot["degraded"][0]
            .as_str()
            .is_some_and(|w| w.starts_with("vector engine unavailable")),
        "{stdout}"
    );

    let (stdout, _, _) = axil(&db, &["boot", "--boot-format", "narrative"]);
    assert!(
        stdout.contains("WARNING: vector engine unavailable"),
        "{stdout}"
    );

    let (stdout, _, _) = axil(&db, &["boot", "--boot-format", "compact"]);
    let boot: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert!(boot["degraded"][0].is_string(), "{stdout}");

    // The hook's per-prompt block: the warning leads, even with no hits.
    let (stdout, _, _) = axil(
        &db,
        &[
            "recall",
            "nothing matches this",
            "--recall-format",
            "context-block",
        ],
    );
    assert!(
        stdout.contains("# Warning: vector engine unavailable"),
        "{stdout}"
    );

    let (stdout, _, _) = axil(&db, &["heal", "--reindex", "--dry-run"]);
    let heal: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert!(heal["degraded"][0].is_string(), "{stdout}");
}

/// Uses the default embedding model (bge-small, 384 dims), so like the other
/// model-backed tests in this crate it runs locally and nightly, not per-PR.
#[test]
fn cli_repairs_a_vector_store_left_dirty_by_a_killed_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    let id = Axil::open(&path)
        .build()
        .unwrap()
        .insert(
            "decisions",
            json!({"summary": "the connection pool size is 32"}),
        )
        .unwrap()
        .id;

    // Copy the store while its engine holds it open: committed data, no clean close.
    let live = dir.path().join("live.axil");
    let engine = axil_vector::VectorEngine::open(&live, 384).unwrap();
    let mut v = vec![0.0f32; 384];
    v[0] = 1.0;
    engine.add(id, &v).unwrap();
    std::fs::copy(
        axil_vector::vector_db_path(&live),
        axil_vector::vector_db_path(&path),
    )
    .unwrap();
    drop(engine);

    let (stdout, stderr, _) = axil(&path, &["doctor"]);
    assert!(
        stderr.contains("was not closed cleanly; repaired it"),
        "{stderr}"
    );
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let vector = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "vector_index")
        .unwrap_or_else(|| panic!("vector engine must be attached after repair: {stdout}"));
    assert!(
        vector["detail"].as_str().unwrap().starts_with("1 vectors"),
        "{vector}"
    );

    // Repaired for good: the next open is quiet.
    let (_, stderr, _) = axil(&path, &["doctor"]);
    assert!(!stderr.contains("repaired"), "{stderr}");
}

#[test]
fn boot_topic_recalls_with_a_vector_store_attached() {
    // `boot --topic` used to open the database a second time for its recall,
    // which found this process's own writer lock and failed whenever a vector
    // store existed.
    use axil_vector::AxilBuilderVectorExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    Axil::open(&path)
        .with_vector(384)
        .unwrap()
        .build()
        .unwrap()
        .insert(
            "decisions",
            json!({"summary": "the connection pool size is 32"}),
        )
        .unwrap();

    let (stdout, stderr, code) = axil(&path, &["boot", "--topic", "connection pool"]);
    assert_eq!(code, 0, "{stderr}");
    let boot: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert!(boot["topic_recall"].is_array(), "{stdout}");
}
