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

/// The `name` check from a `doctor` JSON report.
fn doctor_check(report: &Value, name: &str) -> Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("doctor reports no {name} check: {report}"))
        .clone()
}

/// Uses the default embedding model (bge-small, 384 dims), so like the other
/// model-backed tests in this crate it runs locally and nightly, not per-PR.
#[test]
fn doctor_reports_orphaned_vectors_and_heal_reindex_removes_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    Axil::open(&path).build().unwrap();
    // A vector with no record behind it, as a delete that never reached the
    // vector store leaves one.
    let engine = axil_vector::VectorEngine::open(&path, 384).unwrap();
    let mut v = vec![0.0f32; 384];
    v[0] = 1.0;
    engine.add(axil_core::RecordId::new(), &v).unwrap();
    drop(engine);

    let (stdout, stderr, code) = axil(&path, &["doctor"]);
    assert_eq!(
        code, 2,
        "one orphan in one vector is an error: {stdout}{stderr}"
    );
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let check = doctor_check(&report, "orphaned_vectors");
    assert_eq!(check["status"], "error", "{check}");
    assert!(
        check["detail"]
            .as_str()
            .unwrap()
            .starts_with("1 of 1 vectors"),
        "{check}"
    );
    assert_eq!(check["fix"], "axil heal --orphans", "{check}");

    let (stdout, stderr, code) = axil(&path, &["heal", "--reindex"]);
    assert_eq!(code, 0, "{stderr}");
    let heal: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert!(
        heal["actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["action"] == "clean_orphaned_vectors"),
        "{stdout}"
    );

    let (stdout, _, _) = axil(&path, &["doctor"]);
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let check = doctor_check(&report, "orphaned_vectors");
    assert_eq!(check["status"], "ok", "{check}");
}

/// Uses the default embedding model (bge-small, 384 dims), so like the other
/// model-backed tests in this crate it runs locally and nightly, not per-PR.
#[test]
fn a_cli_delete_with_the_vector_engine_disabled_leaves_no_orphan() {
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
    let engine = axil_vector::VectorEngine::open(&path, 384).unwrap();
    let mut v = vec![0.0f32; 384];
    v[0] = 1.0;
    engine.add(id.clone(), &v).unwrap();
    drop(engine);

    // The operator turns the engine off for a while and deletes meanwhile.
    let config = dir.path().join("axil.toml");
    std::fs::write(&config, "[engines]\ndisabled = [\"vec\"]\n").unwrap();
    let (_, stderr, code) = axil(&path, &["delete", id.as_str()]);
    assert_eq!(code, 0, "{stderr}");
    std::fs::remove_file(&config).unwrap();

    // The next ordinary command with the engine back applies the note
    // (doctor only inspects, so it leaves notes for such a command).
    let (_, stderr, code) = axil(&path, &["list", "decisions"]);
    assert_eq!(code, 0, "{stderr}");
    let (stdout, _, _) = axil(&path, &["doctor"]);
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let check = doctor_check(&report, "orphaned_vectors");
    assert_eq!(check["status"], "ok", "{check}");
    assert_eq!(check["detail"], "0 vectors, no orphans", "{check}");
}

/// Uses the default embedding model (bge-small, 384 dims), so like the other
/// model-backed tests in this crate it runs locally and nightly, not per-PR.
#[test]
fn only_heal_embed_internal_embeds_code_proxies_left_without_a_vector() {
    use axil_vector::AxilBuilderVectorExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    // A store exists, but the proxy is written past it, as the indexer did
    // while the store could not be opened.
    drop(axil_vector::VectorEngine::open(&path, 384).unwrap());
    let proxy = Axil::open(&path)
        .build()
        .unwrap()
        .insert(
            "_idx_code_proxies",
            json!({"proxy_id": "p1", "path": "src/pool.rs", "proxy_text": "fn pool_size"}),
        )
        .unwrap()
        .id;

    let (stdout, _, code) = axil(&path, &["doctor"]);
    assert_eq!(code, 1, "a warning, not an error: {stdout}");
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let check = doctor_check(&report, "vector_index");
    assert_eq!(check["status"], "warning", "{check}");
    assert_eq!(check["fix"], "axil heal --embed-internal", "{check}");

    let (_, stderr, code) = axil(&path, &["heal", "--reindex"]);
    assert_eq!(code, 0, "{stderr}");
    let (stdout, _, _) = axil(&path, &["doctor"]);
    assert!(
        stdout.contains("1 of 1 _idx_code_proxies row(s) missing an embedding"),
        "heal --reindex must leave internal rows alone: {stdout}"
    );
    let (stdout, stderr, code) = axil(&path, &["heal", "--embed-internal"]);
    assert_eq!(code, 0, "{stderr}");
    let heal: Value = serde_json::from_str(&stdout).expect(&stdout);
    assert_eq!(
        heal["actions"][0]["action"], "reembed_missing_internal",
        "{stdout}"
    );

    let db = Axil::open(&path).with_vector(384).unwrap().build().unwrap();
    assert!(db.get_vector(&proxy).unwrap().is_some());
}

/// Uses the default embedding model (bge-small, 384 dims), so like the other
/// model-backed tests in this crate it runs locally and nightly, not per-PR.
#[test]
fn inspection_commands_leave_delete_notes_and_vectors_alone() {
    use axil_vector::AxilBuilderVectorExt;
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
    let engine = axil_vector::VectorEngine::open(&path, 384).unwrap();
    let mut v = vec![0.0f32; 384];
    v[0] = 1.0;
    engine.add(id.clone(), &v).unwrap();
    drop(engine);
    // Deleted while the store was detached: a note waits for the next open.
    assert!(Axil::open(&path).build().unwrap().delete(&id).unwrap());

    for args in [
        &["doctor"][..],
        &["heal", "--orphans", "--dry-run"],
        &["heal", "--reindex", "--dry-run"],
        &["stats"],
    ] {
        let (stdout, stderr, _) = axil(&path, args);
        assert!(!stdout.is_empty(), "{args:?}: {stderr}");
    }
    let (stdout, _, _) = axil(&path, &["doctor"]);
    let report: Value = serde_json::from_str(&stdout).expect(&stdout);
    let check = doctor_check(&report, "orphaned_vectors");
    assert!(
        check["detail"]
            .as_str()
            .unwrap()
            .contains("1 delete(s) made while the vector store was detached are not applied"),
        "{check}"
    );
    // The vector is still on disk: nothing above applied the note.
    let store = axil_vector::VectorEngine::open(&path, 384).unwrap();
    assert!(store.get_vector(&id).unwrap().is_some());
    drop(store);

    // A real heal applies it.
    let (_, stderr, code) = axil(&path, &["heal", "--orphans"]);
    assert_eq!(code, 0, "{stderr}");
    let db = Axil::open(&path).with_vector(384).unwrap().build().unwrap();
    assert_eq!(db.get_vector(&id).unwrap(), None);
}
