//! End-to-end checks through [`Axil`] for the vector engine's lazy graph build
//! and for the repair of on-disk vector rows that can never load.

use std::path::Path;
use std::sync::{Arc, Mutex};

use axil_core::plugin::{TextEmbedder, VectorIndex};
use axil_core::{Axil, HealingConfig, RecordId};
use axil_vector::{vector_db_path, vector_space_db_path, VectorEngine};
use serde_json::json;

const VECTORS_TABLE: redb::TableDefinition<&str, &[u8]> = redb::TableDefinition::new("vectors");

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Write raw rows straight into a store's vectors table, bypassing every
/// insert-time check (what an older version, or corruption, leaves behind).
fn inject_rows(vec_path: &Path, rows: &[(&str, Vec<u8>)]) {
    let db = redb::Database::open(vec_path).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut table = txn.open_table(VECTORS_TABLE).unwrap();
        for (key, bytes) in rows {
            table.insert(*key, bytes.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();
}

/// Opens real [`VectorEngine`] spaces while keeping a handle to each, so a
/// test can inspect the engines `Axil` opened internally.
#[derive(Default)]
struct RecordingFactory {
    opened: Mutex<Vec<Arc<VectorEngine>>>,
}

impl axil_core::VectorSpaceFactory for RecordingFactory {
    fn open_space(
        &self,
        main_path: &Path,
        space: &str,
        dim: Option<usize>,
    ) -> axil_core::Result<Arc<dyn VectorIndex>> {
        let engine = Arc::new(VectorEngine::open_space_at(
            &vector_space_db_path(main_path, space),
            dim,
        )?);
        self.opened.lock().unwrap().push(engine.clone());
        Ok(engine)
    }

    fn space_names(&self, main_path: &Path) -> axil_core::Result<Vec<String>> {
        axil_vector::list_vector_space_names(main_path)
    }

    fn space_meta(&self, main_path: &Path, space: &str) -> axil_core::Result<(usize, usize)> {
        axil_core::VectorSpaceFactory::space_meta(
            &axil_vector::VectorSpaceFactory,
            main_path,
            space,
        )
    }
}

fn fingerprint(i: usize) -> [f32; 4] {
    let t = i as f32;
    [t.sin(), t.cos(), (t * 0.3).sin(), 1.0]
}

#[test]
fn record_delete_fan_out_never_builds_space_graphs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fanout.axil");
    let mut ids = Vec::new();
    {
        let db = axil_vector::with_vector_spaces(Axil::open(&path))
            .build()
            .unwrap();
        // Above the exact-scan threshold, so a search would need the graph.
        for i in 0..200 {
            let rec = db.insert("fingerprints", json!({ "n": i })).unwrap();
            db.add_vector_in("fp", &rec.id, &fingerprint(i)).unwrap();
            ids.push(rec.id);
        }
        // A second space that never held the id deleted below.
        let other = db.insert("fingerprints", json!({ "n": -1 })).unwrap();
        db.add_vector_in("other", &other.id, &[1.0, 0.0]).unwrap();
    }

    let factory = Arc::new(RecordingFactory::default());
    let db = Axil::open(&path)
        .with_vector_space_factory(factory.clone())
        .build()
        .unwrap();
    db.insert("fingerprints", json!({ "n": 1000 })).unwrap();
    assert!(db.delete(&ids[0]).unwrap());

    let opened = factory.opened.lock().unwrap().clone();
    assert_eq!(opened.len(), 2, "the delete fan-out visits every space");
    assert!(
        opened.iter().all(|e| !e.is_graph_built()),
        "a record delete must not build any space's graph"
    );
    assert_eq!(db.get_vector_in("fp", &ids[0]).unwrap(), None);
    assert_eq!(opened.iter().map(|e| e.vector_count()).sum::<usize>(), 200);

    // Searching is what builds a graph — and it no longer sees the deleted id.
    let hits = db.similar_in("fp", &fingerprint(1), 1).unwrap();
    assert_eq!(hits[0].0.id, ids[1]);
    assert!(opened.iter().any(|e| e.is_graph_built()));
}

fn load_skips(db: &Axil) -> Option<axil_core::ProblemDetection> {
    db.detect_problems()
        .into_iter()
        .find(|p| p.detector == "vector_load_skips")
}

fn open_plain(path: &Path) -> Axil {
    Axil::open(path)
        .with_vector_index(Box::new(VectorEngine::open(path, 3).unwrap()))
        .build()
        .unwrap()
}

#[test]
fn heal_clears_unloadable_vector_rows_for_good() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heal.axil");
    let keep = {
        let db = open_plain(&path);
        let rec = db.insert("notes", json!({ "summary": "kept vector" })).unwrap();
        db.add_vector(&rec.id, &[1.0, 0.0, 0.0]).unwrap();
        rec.id
    };
    // An unparsable id, and a wrong-dimension row for a record that no longer
    // exists: neither is in the live index (so orphan cleanup can't see it)
    // nor a live record (so re-embedding never touches it).
    inject_rows(
        &vector_db_path(&path),
        &[
            ("not-a-record-id", to_bytes(&[1.0, 0.0, 0.0])),
            (RecordId::new().as_str(), to_bytes(&[1.0, 0.0])),
        ],
    );

    {
        let db = open_plain(&path);
        let problem = load_skips(&db).expect("unloadable rows must be reported");
        assert!(problem.auto_fixable, "the purge needs no embedder");
        let report = db.heal_all(&HealingConfig::default(), false).unwrap();
        assert!(
            report
                .actions
                .iter()
                .any(|a| a.action == "purge_unloadable_vectors"),
            "{:?}",
            report.actions
        );
        assert!(report.healed);
        assert!(load_skips(&db).is_none(), "heal must clear it in-process");
    }

    let db = open_plain(&path);
    assert!(load_skips(&db).is_none(), "and for every later open");
    assert_eq!(db.get_vector(&keep).unwrap(), Some(vec![1.0, 0.0, 0.0]));
}

/// Embeds every text to the same fixed direction.
struct FixedEmbedder;

impl TextEmbedder for FixedEmbedder {
    fn embed(&self, _text: &str) -> axil_core::Result<Vec<f32>> {
        Ok(vec![0.0, 1.0, 0.0])
    }
}

#[test]
fn reindex_path_purges_then_reembeds_live_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reindex.axil");
    let live = open_plain(&path)
        .insert("notes", json!({ "summary": "needs a vector" }))
        .unwrap()
        .id;
    // An older version accepted an all-zero vector for this live record.
    inject_rows(
        &vector_db_path(&path),
        &[
            (live.as_str(), to_bytes(&[0.0, 0.0, 0.0])),
            ("garbage", to_bytes(&[1.0, 0.0, 0.0])),
        ],
    );

    {
        let db = Axil::open(&path)
            .with_vector_index(Box::new(VectorEngine::open(&path, 3).unwrap()))
            .with_embedder(Box::new(FixedEmbedder))
            .build()
            .unwrap();
        assert!(load_skips(&db).is_some());
        // What `axil heal --reindex` runs: vector_rebuild, then reembed_missing.
        db.vector_rebuild().unwrap();
        let (reembedded, _) = db.reembed_missing().unwrap();
        assert_eq!(reembedded, 1, "the live record gets a fresh vector");
        assert!(load_skips(&db).is_none());
    }

    let db = open_plain(&path);
    assert!(load_skips(&db).is_none());
    assert_eq!(db.get_vector(&live).unwrap(), Some(vec![0.0, 1.0, 0.0]));
}

#[test]
fn heal_purges_unloadable_rows_in_named_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spaces.axil");
    let fp = {
        let db = axil_vector::with_vector_spaces(open_builder(&path))
            .build()
            .unwrap();
        let rec = db.insert("fingerprints", json!({ "n": 1 })).unwrap();
        db.add_vector_in("fp", &rec.id, &[1.0, 0.0]).unwrap();
        rec.id
    };
    inject_rows(
        &vector_space_db_path(&path, "fp"),
        &[(RecordId::new().as_str(), to_bytes(&[0.0, 0.0]))],
    );

    {
        let db = axil_vector::with_vector_spaces(open_builder(&path))
            .build()
            .unwrap();
        db.vector_rebuild().unwrap();
    }
    let space = VectorEngine::open_space_at(&vector_space_db_path(&path, "fp"), None).unwrap();
    assert_eq!(space.skipped_at_load(), 0);
    assert_eq!(space.get_vector(&fp).unwrap(), Some(vec![1.0, 0.0]));
}

fn open_builder(path: &Path) -> axil_core::db::AxilBuilder {
    Axil::open(path).with_vector_index(Box::new(VectorEngine::open(path, 3).unwrap()))
}
