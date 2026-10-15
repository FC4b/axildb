//! The default vector store records the embedding model that built it, and
//! every command checks that record against the model `axil.toml`
//! configures. A store holding another model's vectors is refused before any
//! model loads, so most runs here need no model files; the ones that embed
//! need bge-small on disk and skip loudly without it, so they never download
//! one.
//!
//! The model-free runs point `HOME` at the temp dir, so a user-level config
//! cannot change the configured model. The ones that embed keep `HOME`,
//! where the model is installed.

#![cfg(feature = "embed")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use axil_core::plugin::VectorIndex;
use axil_core::{Axil, RecordId};
use axil_vector::models::EmbeddingModel;
use axil_vector::{open_store_for_model, read_stored_model, ModelStore, VectorEngine};
use serde_json::Value;

const BGE: &str = "bge-small-en-v1.5";
const GRANITE: &str = "granite-embedding-small-english-r2";

/// Run `axil` from `cwd` with `HOME` there too.
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

/// Run `axil` from `cwd` with the caller's `HOME`, where models live.
fn axil_with_models(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axil"))
        .args(args)
        .env_remove("AXIL_DB")
        .env_remove("CLAUDE_PROJECT_DIR")
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

/// A unit vector of `dims` along the first axis.
fn unit(dims: usize) -> Vec<f32> {
    let mut v = vec![0.0; dims];
    v[0] = 1.0;
    v
}

/// A database holding one decision whose vector the store holds as `model`'s,
/// recorded as an open for that model records it. Returns the database path
/// and the decision's id.
fn database_embedded_by(dir: &Path, model: &EmbeddingModel) -> (PathBuf, RecordId) {
    let db = dir.join("memory.axil");
    let id = Axil::open(&db)
        .build()
        .unwrap()
        .insert(
            "decisions",
            serde_json::json!({"summary": "Use redb as the embedded storage engine"}),
        )
        .unwrap()
        .id;
    let ModelStore::Open(engine) = open_store_for_model(&db, model).unwrap() else {
        panic!("a new store is never refused");
    };
    engine.add(id.clone(), &unit(model.dimensions())).unwrap();
    (db, id)
}

fn stored_vectors(db: &Path) -> usize {
    read_stored_model(db).unwrap().unwrap().vectors
}

fn stored_model(db: &Path) -> Option<String> {
    read_stored_model(db).unwrap().unwrap().model
}

#[test]
fn every_command_refuses_a_store_of_another_model_and_keeps_working() {
    let dir = tempfile::tempdir().unwrap();
    // Embedded by granite; axil.toml names no model, so bge-small is
    // configured, and both write 384-dimension vectors.
    let (db, _) = database_embedded_by(dir.path(), &EmbeddingModel::GraniteSmallR2);
    let db_arg = db.to_str().unwrap();
    let rebuild = format!("axil --db {db_arg} reembed");

    // Recall answers from its other signals, and says why once, naming
    // both models and the command that rebuilds the store.
    let output = axil_in(
        dir.path(),
        &["--db", db_arg, "recall", "redb storage engine"],
    );
    assert!(
        output.status.success(),
        "recall failed: {}",
        stderr(&output)
    );
    let err = stderr(&output);
    assert!(err.contains(GRANITE) && err.contains(BGE), "{err}");
    assert!(err.contains(&rebuild), "{err}");
    assert_eq!(
        err.matches("vector engine unavailable").count(),
        1,
        "said once: {err}"
    );

    // boot carries it, so an agent meets it at session start.
    let output = axil_in(dir.path(), &["--db", db_arg, "boot"]);
    assert!(output.status.success(), "boot failed: {}", stderr(&output));
    let boot = stdout_json(&output);
    assert!(
        boot["degraded"][0]
            .as_str()
            .is_some_and(|w| w.contains(GRANITE) && w.contains(&rebuild)),
        "{boot}"
    );

    // doctor fails on it.
    let output = axil_in(dir.path(), &["--db", db_arg, "doctor"]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let report = stdout_json(&output);
    let check = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "vector_engine")
        .unwrap_or_else(|| panic!("doctor must report the refusal: {report}"));
    let detail = check["detail"].as_str().unwrap();
    assert!(detail.contains(GRANITE) && detail.contains(BGE), "{detail}");

    // A plain store still works, and embeds nothing into the store.
    let output = axil_in(
        dir.path(),
        &[
            "--db",
            db_arg,
            "store",
            "decisions",
            r#"{"summary":"Ship the CLI as a single static binary"}"#,
        ],
    );
    assert!(output.status.success(), "store failed: {}", stderr(&output));
    assert_eq!(stored_vectors(&db), 1);

    // An explicit embed is refused, and so is a raw vector for the default
    // space: either would put a vector of one model beside another's.
    let raw = serde_json::to_string(&unit(384)).unwrap();
    for args in [
        &[
            "store",
            "decisions",
            r#"{"summary":"embed me"}"#,
            "--embed",
            "summary",
        ][..],
        &["store", "notes", r#"{"n":"a"}"#, "--vector", &raw][..],
    ] {
        let mut full = vec!["--db", db_arg];
        full.extend_from_slice(args);
        let output = axil_in(dir.path(), &full);
        assert!(!output.status.success(), "{args:?} must be refused");
        let err = stderr(&output);
        assert!(
            err.contains(GRANITE) && err.contains(&rebuild),
            "{args:?}: {err}"
        );
    }
    assert_eq!(stored_vectors(&db), 1);

    // info lists it, as the other reports do.
    let output = axil_in(dir.path(), &["--db", db_arg, "info"]);
    assert!(output.status.success(), "info failed: {}", stderr(&output));
    let info = stdout_json(&output);
    assert!(
        info["degraded"][0]
            .as_str()
            .is_some_and(|w| w.contains(GRANITE) && w.contains(&rebuild)),
        "{info}"
    );

    // A heal that would embed can't, and names what rebuilds the store.
    for args in [
        &["heal", "--reindex"][..],
        &["heal", "--embed-internal"][..],
        &["heal", "--rechunk"][..],
    ] {
        let mut full = vec!["--db", db_arg];
        full.extend_from_slice(args);
        let output = axil_in(dir.path(), &full);
        assert!(!output.status.success(), "{args:?} must stop");
        assert!(
            stderr(&output).contains(&rebuild),
            "{args:?}: {}",
            stderr(&output)
        );
    }
    // Any other runs without the vector engine, and says so.
    for args in [
        &["heal", "--compact"][..],
        &["heal", "--dry-run"][..],
        &["heal", "--reindex", "--dry-run"][..],
    ] {
        let mut full = vec!["--db", db_arg];
        full.extend_from_slice(args);
        let output = axil_in(dir.path(), &full);
        assert!(
            output.status.success(),
            "{args:?} failed: {}",
            stderr(&output)
        );
        let report = stdout_json(&output);
        assert!(
            report["degraded"][0]
                .as_str()
                .is_some_and(|w| w.contains(GRANITE)),
            "{args:?}: {report}"
        );
    }

    // Nothing above changed the store.
    assert_eq!(stored_model(&db).as_deref(), Some(GRANITE));
    assert_eq!(stored_vectors(&db), 1);
}

#[test]
fn a_store_of_another_width_is_refused_the_same_way() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = database_embedded_by(dir.path(), &EmbeddingModel::BgeBase);
    let db_arg = db.to_str().unwrap();

    // Before model records this failed every open; now it is a refusal.
    let output = axil_in(dir.path(), &["--db", db_arg, "recall", "redb"]);
    assert!(
        output.status.success(),
        "recall failed: {}",
        stderr(&output)
    );
    let err = stderr(&output);
    assert!(
        err.contains("bge-base-en-v1.5") && err.contains("768-dimension"),
        "{err}"
    );
    assert!(err.contains("reembed"), "{err}");
}

/// A database holding one decision whose `dims`-wide vector sits in a store
/// as an older version wrote it: no model recorded. Returns the database
/// path and the decision's id.
fn legacy_database(dir: &Path, dims: usize) -> (PathBuf, RecordId) {
    let db = dir.join("memory.axil");
    let id = Axil::open(&db)
        .build()
        .unwrap()
        .insert(
            "decisions",
            serde_json::json!({"summary": "Use redb as the embedded storage engine"}),
        )
        .unwrap()
        .id;
    VectorEngine::open(&db, dims)
        .unwrap()
        .add(id.clone(), &unit(dims))
        .unwrap();
    (db, id)
}

/// granite came after model records, so no store from before them holds its
/// vectors: taking one for granite's would be the very switch to catch.
#[test]
fn a_store_from_before_model_records_is_refused_to_granite() {
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "granite-small");
    let (db, _) = legacy_database(dir.path(), 384);

    let output = axil_in(
        dir.path(),
        &["--db", db.to_str().unwrap(), "recall", "redb"],
    );
    assert!(
        output.status.success(),
        "recall failed: {}",
        stderr(&output)
    );
    let err = stderr(&output);
    assert!(
        err.contains("does not name") && err.contains(GRANITE),
        "{err}"
    );
    assert!(err.contains("reembed"), "{err}");
    // The refusal writes nothing.
    assert_eq!(stored_model(&db), None);
}

/// Before model records a custom model named in axil.toml was never loaded
/// (opens fell back to bge-small), so its store may hold either model's.
#[test]
fn a_store_from_before_model_records_is_refused_to_a_custom_model() {
    let dir = tempfile::tempdir().unwrap();
    // A model registered with `axil model-add`, in this HOME's registry. Its
    // file need not exist: the refusal comes before the model loads.
    std::fs::create_dir_all(dir.path().join(".axil")).unwrap();
    std::fs::write(
        dir.path().join(".axil/custom_models.json"),
        r#"{"mine": {"path": "/models/mine/model.onnx", "dimensions": 384,
            "pooling": "cls", "max_seq_len": 512}}"#,
    )
    .unwrap();
    configure_model(dir.path(), "mine");
    let (db, _) = legacy_database(dir.path(), 384);

    let output = axil_in(
        dir.path(),
        &["--db", db.to_str().unwrap(), "recall", "redb"],
    );
    assert!(
        output.status.success(),
        "recall failed: {}",
        stderr(&output)
    );
    let err = stderr(&output);
    assert!(
        err.contains("custom:/models/mine/model.onnx") && err.contains("custom model"),
        "{err}"
    );
    assert!(err.contains("reembed"), "{err}");
    assert_eq!(stored_model(&db), None);
}

/// A bge-base database written before model records keeps its vectors on
/// upgrade, with a one-time notice in case another model built the store.
#[test]
fn a_store_from_before_model_records_is_taken_for_its_configured_model() {
    if !model_on_disk(&EmbeddingModel::BgeBase) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    configure_model(dir.path(), "bge-base");
    let (db, _) = legacy_database(dir.path(), 768);

    let output = axil_with_models(dir.path(), &["--db", db.to_str().unwrap(), "doctor"]);
    let err = stderr(&output);
    assert!(!err.contains("vector engine unavailable"), "{err}");
    assert!(
        err.contains("[axil vector] note:") && err.contains("bge-base-en-v1.5"),
        "{err}"
    );
    let report = stdout_json(&output);
    assert!(
        report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["name"] == "vector_index"
                && c["detail"]
                    .as_str()
                    .is_some_and(|d| d.contains("model=bge-base-en-v1.5"))),
        "{report}"
    );
    assert_eq!(stored_model(&db), None, "reading recorded nothing");
}

#[test]
fn a_scoped_reembed_refuses_a_store_of_another_model() {
    let dir = tempfile::tempdir().unwrap();
    let (db, id) = database_embedded_by(dir.path(), &EmbeddingModel::GraniteSmallR2);
    let output = axil_in(
        dir.path(),
        &[
            "--db",
            db.to_str().unwrap(),
            "reembed",
            "--field",
            "summary",
            "--table",
            "decisions",
        ],
    );
    assert!(!output.status.success(), "a scoped re-embed must stop");
    let err = stderr(&output);
    assert!(err.contains("cannot re-embed only one table"), "{err}");
    assert_eq!(stored_model(&db).as_deref(), Some(GRANITE));
    assert_eq!(
        VectorEngine::open(&db, 384)
            .unwrap()
            .get_vector(&id)
            .unwrap(),
        Some(unit(384)),
        "the store must be left as it was"
    );
}

/// A default-config database written before model records upgrades with no
/// action: the store is taken as bge-small's, quietly, and recorded so by
/// its next vector write.
#[test]
fn a_store_from_before_model_records_upgrades_quietly_under_the_default_model() {
    if !model_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = legacy_database(dir.path(), 384);
    let db_arg = db.to_str().unwrap();

    let output = axil_with_models(dir.path(), &["--db", db_arg, "doctor"]);
    let err = stderr(&output);
    assert!(!err.contains("vector engine unavailable"), "{err}");
    assert!(
        !err.contains("[axil vector] note"),
        "the default model is silent: {err}"
    );
    let report = stdout_json(&output);
    let check = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "vector_index")
        .unwrap_or_else(|| panic!("the vector engine must attach: {report}"));
    let detail = check["detail"].as_str().unwrap();
    assert!(detail.contains(&format!("model={BGE} (assumed")), "{check}");
    // Reading wrote nothing.
    assert_eq!(stored_model(&db), None);

    let output = axil_with_models(
        dir.path(),
        &[
            "--db",
            db_arg,
            "store",
            "decisions",
            r#"{"summary":"Ship the CLI as a single static binary"}"#,
        ],
    );
    assert!(output.status.success(), "store failed: {}", stderr(&output));
    assert_eq!(stored_model(&db).as_deref(), Some(BGE));
    assert_eq!(stored_vectors(&db), 2);
}

/// `reembed` with no `--field` rebuilds every vector owner (memory records
/// and internal rows alike) under the configured model and records it.
#[test]
fn reembed_rebuilds_every_owner_under_the_configured_model() {
    if !model_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let (db, decision) = database_embedded_by(dir.path(), &EmbeddingModel::GraniteSmallR2);
    let db_arg = db.to_str().unwrap();
    // A code-index row with a vector, and a memory record stored while the
    // store was refused, with none.
    let (proxy, later) = {
        let core = Axil::open(&db).build().unwrap();
        let proxy = core
            .insert(
                "_idx_code_proxies",
                serde_json::json!({"proxy_text": "fn open_store_for_model(db_path, model)"}),
            )
            .unwrap()
            .id;
        let later = core
            .insert(
                "decisions",
                serde_json::json!({"summary": "Ship the CLI as a single static binary"}),
            )
            .unwrap()
            .id;
        (proxy, later)
    };
    {
        let ModelStore::Open(engine) =
            open_store_for_model(&db, &EmbeddingModel::GraniteSmallR2).unwrap()
        else {
            panic!("the store is granite's");
        };
        let mut v = unit(384);
        v[1] = 1.0;
        engine.add(proxy.clone(), &v).unwrap();
    }

    let output = axil_with_models(dir.path(), &["--db", db_arg, "reembed"]);
    assert!(
        output.status.success(),
        "reembed failed: {}",
        stderr(&output)
    );
    let report = stdout_json(&output);
    assert_eq!(report["model"], BGE, "{report}");
    assert_eq!(report["previous_model"], GRANITE, "{report}");
    assert_eq!(report["memory_records"], 2, "{report}");
    assert_eq!(report["internal_rows"], 1, "{report}");
    assert_eq!(report["reembedded"], 3, "{report}");

    assert_eq!(stored_model(&db).as_deref(), Some(BGE));
    let engine = VectorEngine::open(&db, 384).unwrap();
    for id in [&decision, &proxy, &later] {
        let v = engine.get_vector(id).unwrap();
        assert!(
            v.is_some_and(|v| v != unit(384)),
            "{id} must hold a bge-small vector"
        );
    }
    drop(engine);
    assert!(!dir.path().join("memory.axil.vec-rebuild").exists());

    // The refusal is gone.
    let output = axil_with_models(dir.path(), &["--db", db_arg, "recall", "redb"]);
    assert!(
        output.status.success(),
        "recall failed: {}",
        stderr(&output)
    );
    assert!(
        !stderr(&output).contains("vector engine unavailable"),
        "{}",
        stderr(&output)
    );
}

/// A store from before model records is taken to hold the configured
/// model's vectors, so a command naming another model of its width must not
/// take it for that model's: a bge-base store adopted by nomic would carry
/// bge-base vectors into a store that records nomic.
#[test]
fn a_named_model_does_not_take_over_a_store_from_before_model_records() {
    for (configured, dims, named, named_id, configured_id) in [
        (
            "bge-base",
            768,
            "nomic",
            "nomic-embed-text-v1.5",
            "bge-base-en-v1.5",
        ),
        (
            "bge-small",
            384,
            "bge-small-int8",
            "bge-small-en-v1.5-int8",
            BGE,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        configure_model(dir.path(), configured);
        let (db, id) = legacy_database(dir.path(), dims);
        let db_arg = db.to_str().unwrap();
        let id = id.to_string();

        for args in [
            &["embed", &id, "summary", "--model", named][..],
            &[
                "reembed",
                "--model",
                named,
                "--field",
                "summary",
                "--table",
                "decisions",
            ][..],
        ] {
            let mut full = vec!["--db", db_arg];
            full.extend_from_slice(args);
            let output = axil_in(dir.path(), &full);
            assert!(!output.status.success(), "{args:?} must be refused");
            let err = stderr(&output);
            assert!(
                err.contains(named_id) && err.contains(configured_id),
                "{args:?}: {err}"
            );
            assert!(err.contains("drop --model"), "{args:?}: {err}");
        }
        assert_eq!(stored_model(&db), None, "nothing recorded");
        assert_eq!(stored_vectors(&db), 1, "nothing added");
        assert!(!dir.path().join("memory.axil.vec-rebuild").exists());
    }
}

/// `embed --model` names a model other than the configured one: with no
/// store it would start one of that model's vectors, which every open under
/// the configured model refuses.
#[test]
fn embed_with_another_model_does_not_start_a_store() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let id = Axil::open(&db)
        .build()
        .unwrap()
        .insert(
            "decisions",
            serde_json::json!({"summary": "Use redb as the embedded storage engine"}),
        )
        .unwrap()
        .id
        .to_string();
    let output = axil_in(
        dir.path(),
        &[
            "--db",
            db.to_str().unwrap(),
            "embed",
            &id,
            "summary",
            "--model",
            "granite-small",
        ],
    );
    assert!(!output.status.success(), "embed --model must be refused");
    let err = stderr(&output);
    assert!(err.contains("would start the vector store"), "{err}");
    assert!(err.contains("drop --model"), "{err}");
    assert!(
        !axil_vector::vector_db_path(&db).exists(),
        "no store created"
    );

    // Nor does it fill an empty store that records the named model already.
    let ModelStore::Open(engine) =
        open_store_for_model(&db, &EmbeddingModel::GraniteSmallR2).unwrap()
    else {
        panic!("a new store is never refused");
    };
    drop(engine);
    let output = axil_in(
        dir.path(),
        &[
            "--db",
            db.to_str().unwrap(),
            "embed",
            &id,
            "summary",
            "--model",
            "granite-small",
        ],
    );
    assert!(!output.status.success(), "embed --model must be refused");
    assert!(
        stderr(&output).contains("would start the vector store"),
        "{}",
        stderr(&output)
    );
    assert_eq!(stored_vectors(&db), 0);
}

/// Raw vector commands on the default space check the store's model like
/// every other open: a store of another model takes no vector and answers no
/// search, and a store that records none records the configured model with
/// its first vector.
#[test]
fn raw_vector_commands_check_the_stores_model() {
    let dir = tempfile::tempdir().unwrap();
    let (db, id) = database_embedded_by(dir.path(), &EmbeddingModel::GraniteSmallR2);
    let db_arg = db.to_str().unwrap();
    let id = id.to_string();
    let raw = serde_json::to_string(&unit(384)).unwrap();
    for args in [
        &["add-vector", &id, &raw][..],
        &["similar", "--vector", &raw][..],
        &["similar", "--id", &id][..],
        &["search-vector", &raw][..],
    ] {
        let mut full = vec!["--db", db_arg];
        full.extend_from_slice(args);
        let output = axil_in(dir.path(), &full);
        assert!(!output.status.success(), "{args:?} must be refused");
        let err = stderr(&output);
        assert!(
            err.contains(GRANITE) && err.contains("reembed"),
            "{args:?}: {err}"
        );
    }
    assert_eq!(stored_model(&db).as_deref(), Some(GRANITE));
    assert_eq!(stored_vectors(&db), 1);

    // An empty store that names no model, as `init --vector-dims` leaves one.
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("memory.axil");
    let id = Axil::open(&db)
        .build()
        .unwrap()
        .insert("notes", serde_json::json!({"n": "a"}))
        .unwrap()
        .id
        .to_string();
    drop(VectorEngine::open(&db, 384).unwrap());
    let output = axil_in(
        dir.path(),
        &["--db", db.to_str().unwrap(), "add-vector", &id, &raw],
    );
    assert!(
        output.status.success(),
        "add-vector failed: {}",
        stderr(&output)
    );
    assert_eq!(stored_model(&db).as_deref(), Some(BGE));
    assert_eq!(stored_vectors(&db), 1);
}
