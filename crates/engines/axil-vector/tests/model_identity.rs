//! The embedding-model check, end to end through [`Axil`]: a store holding
//! another model's vectors is refused before the model loads, so none of
//! these tests needs model files; a store from before model records is taken
//! over or refused by the configured model; and a rebuild re-embeds every
//! vector owner into a store that records the new model. A bag-of-words
//! embedder stands in for a real model wherever text is embedded.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::path::Path;

use axil_core::plugin::{TextEmbedder, VectorIndex};
use axil_core::{Axil, RecordId, Severity};
use axil_vector::models::EmbeddingModel;
use axil_vector::{
    open_store_for_model, read_stored_model, stored_vector_ids, vector_db_path, with_model_store,
    AxilBuilderVectorExt, ModelStore, VectorEngine,
};
use serde_json::json;

const DIMS: usize = 384;
const BGE: &str = "bge-small-en-v1.5";
const GRANITE: &str = "granite-embedding-small-english-r2";

/// Words hashed into a 384-dimension bag: an embedder that needs no model.
struct Bag;

impl TextEmbedder for Bag {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        let mut v = vec![0.0f32; DIMS];
        for word in text.split_whitespace() {
            let bucket = word.to_lowercase().bytes().map(usize::from).sum::<usize>() % (DIMS - 1);
            v[bucket] += 1.0;
        }
        v[DIMS - 1] = 1.0;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(v.iter().map(|x| x / norm).collect())
    }
}

/// The default vector store opened for `model`, which must not refuse it.
fn store_for(path: &Path, model: &EmbeddingModel) -> VectorEngine {
    match open_store_for_model(path, model).unwrap() {
        ModelStore::Open(engine) => engine,
        ModelStore::Refused(m) => panic!("refused: {m:?}"),
        _ => unreachable!(),
    }
}

/// A handle that embeds with [`Bag`] into the store opened for `model`, as
/// an open that loads `model` would.
fn open_as(path: &Path, model: &EmbeddingModel) -> Axil {
    Axil::open(path)
        .with_vector_index(Box::new(store_for(path, model)))
        .with_embedder(Box::new(Bag))
        .build()
        .unwrap()
}

/// A database whose store holds one memory record's vector, embedded under
/// `model`. Returns the record's id.
fn database_built_by(path: &Path, model: &EmbeddingModel) -> RecordId {
    let db = open_as(path, model);
    let record = db
        .insert(
            "decisions",
            json!({"summary": "Use redb as the embedded storage engine"}),
        )
        .unwrap();
    assert!(
        db.get_vector(&record.id).unwrap().is_some(),
        "setup: insert embeds"
    );
    record.id
}

thread_local! {
    static CAUGHT: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

/// Keeps the diagnostic lines of a thread that asked for them; every other
/// thread's go to stderr, as they would with no sink.
fn collect(args: fmt::Arguments<'_>) {
    let line = args.to_string();
    let kept = CAUGHT.with(|c| {
        c.borrow_mut()
            .as_mut()
            .map(|lines| lines.push(line.clone()))
    });
    if kept.is_none() {
        eprintln!("{line}");
    }
}

/// Run `f`, and return what it returned with the diagnostic lines it printed.
fn printed_by<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    // A process sets its sink once; every test here sets the same one.
    let _ = axil_core::diag::set_diagnostic_sink(collect);
    CAUGHT.with(|c| *c.borrow_mut() = Some(Vec::new()));
    let value = f();
    (value, CAUGHT.with(|c| c.borrow_mut().take()).unwrap())
}

/// A store as one written before model records: `dims` wide, one vector, no
/// model named.
fn legacy_store(path: &Path, dims: usize) {
    let raw = VectorEngine::open(path, dims).unwrap();
    let mut v = vec![0.0; dims];
    v[0] = 1.0;
    raw.add(RecordId::new(), &v).unwrap();
}

#[test]
fn a_legacy_store_is_taken_for_its_configured_model_with_a_notice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    // A bge-base database from before model records.
    legacy_store(&path, 768);
    configure(dir.path(), "bge-base");

    let (store, printed) =
        printed_by(|| open_store_for_model(&path, &EmbeddingModel::BgeBase).unwrap());
    assert!(
        matches!(store, ModelStore::Open(_)),
        "bge-base takes it over"
    );
    drop(store);
    assert_eq!(printed.len(), 1, "{printed:?}");
    let notice = &printed[0];
    assert!(notice.starts_with("[axil vector] note:"), "{notice}");
    assert!(
        notice.contains("bge-base-en-v1.5") && notice.contains("reembed"),
        "{notice}"
    );

    // Said once per store, and reading recorded nothing.
    let (store, printed) =
        printed_by(|| open_store_for_model(&path, &EmbeddingModel::BgeBase).unwrap());
    assert!(matches!(store, ModelStore::Open(_)));
    drop(store);
    assert!(printed.is_empty(), "{printed:?}");
    assert_eq!(read_stored_model(&path).unwrap().unwrap().model, None);
}

/// An axil.toml naming `model` beside the database in `dir`.
fn configure(dir: &Path, model: &str) {
    std::fs::write(
        dir.join("axil.toml"),
        format!("[database]\nembedding_model = \"{model}\"\n"),
    )
    .unwrap();
}

/// When axil.toml names a model, the CLI's hooks embedded a store from before
/// model records with it, so a library caller opening the database with
/// another model of the same width, custom or built-in, is refused the store
/// rather than writing its vectors beside theirs.
#[test]
fn a_model_passed_in_does_not_take_over_the_configured_models_legacy_store() {
    let custom = |dir: &Path| EmbeddingModel::Custom {
        path: dir.join("mine/model.onnx"),
        dimensions: DIMS,
        pooling: axil_vector::models::PoolingStrategy::Cls,
        max_seq_len: 512,
    };
    let dir = tempfile::tempdir().unwrap();
    for passed in [custom(dir.path()), EmbeddingModel::BgeSmallInt8] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.axil");
        legacy_store(&path, DIMS);
        configure(dir.path(), "bge-small");

        let (store, printed) = printed_by(|| open_store_for_model(&path, &passed).unwrap());
        let ModelStore::Refused(m) = store else {
            panic!("{passed} must not take over bge-small's store");
        };
        assert_eq!(m.configured.as_deref(), Some(BGE), "{passed}");
        assert!(printed.is_empty(), "{printed:?}");
        let stored = read_stored_model(&path).unwrap().unwrap();
        assert_eq!((stored.model, stored.vectors), (None, 1), "{passed}");
    }
}

/// A library caller that passed a custom model to its opens embedded with
/// it, so with no model configured its store from before model records is
/// taken for that model's, with a notice, and records the model with its
/// next vector.
#[test]
fn a_custom_model_passed_in_takes_over_its_legacy_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    legacy_store(&path, DIMS);
    let custom = EmbeddingModel::Custom {
        path: dir.path().join("mine/model.onnx"),
        dimensions: DIMS,
        pooling: axil_vector::models::PoolingStrategy::Cls,
        max_seq_len: 512,
    };

    let (store, printed) = printed_by(|| open_store_for_model(&path, &custom).unwrap());
    let ModelStore::Open(store) = store else {
        panic!("the custom model takes it over");
    };
    assert_eq!(printed.len(), 1, "{printed:?}");
    assert!(
        printed[0].starts_with("[axil vector] note:")
            && printed[0].contains(&custom.store_identity()),
        "{printed:?}"
    );
    let mut v = vec![0.0; DIMS];
    v[1] = 1.0;
    store.add(RecordId::new(), &v).unwrap();
    drop(store);
    let stored = read_stored_model(&path).unwrap().unwrap();
    assert_eq!(stored.model, Some(custom.store_identity()));
    assert_eq!(stored.vectors, 2);
}

#[test]
fn the_default_model_takes_over_a_legacy_store_silently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    legacy_store(&path, DIMS);
    let (store, printed) =
        printed_by(|| open_store_for_model(&path, &EmbeddingModel::BgeSmall).unwrap());
    assert!(matches!(store, ModelStore::Open(_)));
    assert!(printed.is_empty(), "{printed:?}");
}

#[test]
fn a_legacy_store_is_refused_to_granite_with_a_warning() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    legacy_store(&path, DIMS);
    let (db, printed) = printed_by(|| {
        Axil::open(&path)
            .with_embedder_model(EmbeddingModel::GraniteSmallR2)
            .unwrap()
            .build()
            .unwrap()
    });
    assert!(!db.has_vector_index());
    assert_eq!(printed.len(), 1, "{printed:?}");
    assert!(
        printed[0].starts_with("[axil vector] warning:")
            && printed[0].contains(GRANITE)
            && printed[0].contains("reembed"),
        "{}",
        printed[0]
    );
}

#[test]
fn another_models_store_is_refused_before_the_model_loads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    let id = database_built_by(&path, &EmbeddingModel::BgeSmall);

    // granite writes 384-dimension vectors too, so nothing about the store's
    // size tells the two apart. The refusal comes before the model would
    // load (or download), which is why this needs no model files.
    let db = Axil::open(&path)
        .with_embedder_model(EmbeddingModel::GraniteSmallR2)
        .unwrap()
        .build()
        .unwrap();
    assert!(!db.has_vector_index());
    assert!(!db.has_embedder());
    let degraded = db.degraded_engines();
    assert_eq!(degraded.len(), 1, "{degraded:?}");
    assert_eq!(degraded[0].engine, "vector");
    assert!(
        degraded[0].reason.contains(BGE) && degraded[0].reason.contains(GRANITE),
        "the reason names both models: {}",
        degraded[0].reason
    );
    assert!(degraded[0].fix.contains("reembed"), "{}", degraded[0].fix);

    // Vector search is refused, and says why.
    for err in [
        db.similar_to("redb", 5).unwrap_err().to_string(),
        db.similar_to_vector(&[1.0; DIMS], 5)
            .unwrap_err()
            .to_string(),
    ] {
        assert!(err.contains(GRANITE) && err.contains("reembed"), "{err}");
    }
    // Recall answers from its other signals.
    db.recall("redb storage engine", 5, None).unwrap();

    // Nothing new reaches the store: not on insert, not on request.
    let fresh = db
        .insert(
            "decisions",
            json!({"summary": "Ship the CLI as a single static binary"}),
        )
        .unwrap();
    assert!(db.embed_text(&fresh.id, "a single static binary").is_err());
    drop(db);
    let stored = read_stored_model(&path).unwrap().unwrap();
    assert_eq!(stored.model.as_deref(), Some(BGE));
    assert_eq!(stored.vectors, 1, "only the bge-small vector");
    let ids = stored_vector_ids(&path).unwrap();
    assert_eq!(ids, vec![id]);
}

#[test]
fn doctor_and_info_name_the_stores_model_and_a_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    database_built_by(&path, &EmbeddingModel::GraniteSmallR2);

    let db = open_as(&path, &EmbeddingModel::GraniteSmallR2);
    let report = db.doctor().unwrap();
    let check = report
        .checks
        .iter()
        .find(|c| c.name == "vector_index")
        .expect("a vector_index check");
    assert!(
        check.detail.contains(&format!("model={GRANITE}")),
        "{}",
        check.detail
    );
    let info = db.info().unwrap();
    assert_eq!(info.plugins["vector"]["model"], GRANITE);
    drop(db);

    // The same store opened by a database configured for bge-small.
    let db = Axil::open(&path)
        .with_embedder_model(EmbeddingModel::BgeSmall)
        .unwrap()
        .build()
        .unwrap();
    let report = db.doctor().unwrap();
    let check = report
        .checks
        .iter()
        .find(|c| c.name == "vector_engine")
        .expect("a vector_engine check");
    assert_eq!(check.status, Severity::Error);
    assert!(
        check.detail.contains(GRANITE) && check.detail.contains(BGE),
        "{}",
        check.detail
    );
    assert!(check.fix.as_deref().unwrap().contains("reembed"));
    assert_eq!(report.status, Severity::Error);
}

#[test]
fn a_store_created_for_a_model_without_loading_it_records_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    let db = with_model_store(Axil::open(&path), &EmbeddingModel::GraniteSmallR2)
        .unwrap()
        .build()
        .unwrap();
    assert!(db.has_vector_index());
    drop(db);
    assert_eq!(
        read_stored_model(&path).unwrap().unwrap().model.as_deref(),
        Some(GRANITE)
    );
}

/// Rows of the internal tables whose writers embed something other than
/// their whole text, as those writers store them, with the text each writer
/// embeds. The texts are spelled out rather than derived, so they pin the
/// writers' formats.
fn writer_rows() -> Vec<(&'static str, serde_json::Value, &'static str)> {
    vec![
        // The cache matches questions against questions: never the answer.
        (
            "_cache_entries",
            json!({
                "question": "how do I rebuild the vector store",
                "answer": "Run reembed; it builds the new store beside the old one",
                "created_at": "2026-10-15T08:00:00+00:00",
                "valid_until": "2026-11-15T08:00:00+00:00",
                "hit_count": 0,
                "code_refs": [],
            }),
            "how do I rebuild the vector store",
        ),
        (
            "_preferences",
            json!({
                "key": "editor",
                "value": "vim",
                "source": "user",
                "synthetic_doc": "The user prefers vim as their editor",
            }),
            "editor: vim",
        ),
        (
            "_procedures",
            json!({
                "pattern_name": "release",
                "description": "tag, build and publish the crates",
                "confidence": 0.5,
                "applications": 1,
            }),
            "release: tag, build and publish the crates",
        ),
        (
            "_entities",
            json!({
                "entity": "redb",
                "fact": "an embedded key-value store in pure Rust",
                "source": "docs",
            }),
            "redb: an embedded key-value store in pure Rust",
        ),
    ]
}

#[test]
fn a_rebuild_re_embeds_every_owner_and_records_the_new_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");

    // A bge-small database with every kind of vector owner.
    let (memory, proxy, written, raw, note, undocumented) = {
        let db = open_as(&path, &EmbeddingModel::BgeSmall);
        let memory = db
            .insert(
                "decisions",
                json!({"summary": "Use redb as the embedded storage engine"}),
            )
            .unwrap();
        // An internal row its writer embeds, from the field it embeds.
        let proxy = db
            .insert(
                "_idx_code_proxies",
                json!({"proxy_text": "fn open_store_for_model(db_path, model)"}),
            )
            .unwrap();
        db.embed_text(&proxy.id, "fn open_store_for_model(db_path, model)")
            .unwrap();
        // One with no vector, as code-index rows written while the store was
        // unavailable are: a rebuild leaves it so.
        let undocumented = db
            .insert(
                "_dep_docs",
                json!({"content": "redb: an embedded database"}),
            )
            .unwrap();
        // Rows of the tables whose writers embed part of a row, embedded as
        // those writers embed them.
        let mut written = Vec::new();
        for (table, data, text) in writer_rows() {
            let row = db.insert(table, data).unwrap();
            assert!(db.embed_row(&row).unwrap(), "{table}: its writer embeds it");
            assert_eq!(
                db.get_vector(&row.id).unwrap(),
                Some(Bag.embed(text).unwrap()),
                "{table}: embedded from {text:?}"
            );
            written.push((row.id, text));
        }
        // A row of a table Axil knows no writer of, embedded by its caller
        // (`store --embed`).
        let note = db
            .insert("_notes", json!({"text": "remember the milk"}))
            .unwrap();
        db.embed_text(&note.id, "remember the milk").unwrap();
        // A vector its caller supplied, for a row with no text.
        let raw = db.insert("_points", json!({"n": 3})).unwrap();
        db.add_vector(&raw.id, &Bag.embed("a point").unwrap())
            .unwrap();
        (
            memory.id,
            proxy.id,
            written,
            raw.id,
            note.id,
            undocumented.id,
        )
    };
    // A memory record stored while the vector engine was off has no vector.
    let detached = Axil::open(&path)
        .build()
        .unwrap()
        .insert(
            "decisions",
            json!({"summary": "Ship the CLI as a single static binary"}),
        )
        .unwrap()
        .id;

    // What `axil reembed` does: read the owners, set the old store aside,
    // and embed them into a new store for the new model.
    let owners: HashSet<RecordId> = stored_vector_ids(&path).unwrap().into_iter().collect();
    assert_eq!(
        owners.len(),
        8,
        "setup: memory, proxy, four writers' rows, note and raw vectors"
    );
    std::fs::rename(vector_db_path(&path), dir.path().join("old.vec")).unwrap();

    let db = open_as(&path, &EmbeddingModel::GraniteSmallR2);
    let mut last = (0, 0);
    let report = db
        .embed_vector_owners(&owners, &mut |done, total| last = (done, total))
        .unwrap();
    assert_eq!(
        (report.memory, report.internal, report.other, report.dropped),
        (2, 5, 1, 1),
        "{report:?}"
    );
    assert_eq!((report.failed, report.first_error), (0, None));
    assert_eq!(last, (8, 8));

    let mut expected: Vec<(&RecordId, &str)> = vec![
        (&memory, "Use redb as the embedded storage engine"),
        (&detached, "Ship the CLI as a single static binary"),
        (&proxy, "fn open_store_for_model(db_path, model)"),
        (&note, "remember the milk"),
    ];
    expected.extend(written.iter().map(|(id, text)| (id, *text)));
    for (id, text) in expected {
        assert_eq!(
            db.get_vector(id).unwrap(),
            Some(Bag.embed(text).unwrap()),
            "{id} is re-embedded from {text:?}, the text its writer embeds"
        );
    }
    assert_eq!(db.get_vector(&raw).unwrap(), None, "no text to embed");
    assert_eq!(db.get_vector(&undocumented).unwrap(), None);
    drop(db);
    assert_eq!(
        read_stored_model(&path).unwrap().unwrap().model.as_deref(),
        Some(GRANITE)
    );
}

/// A store opened raw and then given a model, the way a library caller may
/// put one together, is checked like a store opened for the model: one that
/// holds another model's vectors is not attached.
#[test]
fn a_raw_store_given_another_model_is_not_attached() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    database_built_by(&path, &EmbeddingModel::BgeSmall);

    let (db, printed) = printed_by(|| {
        let engine = VectorEngine::open(&path, DIMS)
            .unwrap()
            .with_model(EmbeddingModel::GraniteSmallR2)
            .unwrap();
        Axil::open(&path)
            .with_vector_and_embedder(engine)
            .build()
            .unwrap()
    });
    assert!(!db.has_vector_index() && !db.has_embedder());
    let degraded = db.degraded_engines();
    assert_eq!(degraded.len(), 1, "{degraded:?}");
    assert!(
        degraded[0].reason.contains(BGE) && degraded[0].reason.contains(GRANITE),
        "{}",
        degraded[0].reason
    );
    assert_eq!(printed.len(), 1, "{printed:?}");
    let err = db
        .similar_to_vector(&[1.0; DIMS], 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains(GRANITE), "{err}");
    drop(db);
    assert_eq!(
        read_stored_model(&path).unwrap().unwrap().model.as_deref(),
        Some(BGE)
    );
}

#[test]
fn a_rebuild_needs_an_attached_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.axil");
    let db = Axil::open(&path).build().unwrap();
    assert!(db
        .embed_vector_owners(&HashSet::new(), &mut |_, _| {})
        .is_err());
}
