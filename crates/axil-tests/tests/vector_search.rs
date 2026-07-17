use axil_core::{Axil, RecordId};
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

fn temp_vector_db(dims: usize) -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path)
        .with_vector(dims)
        .unwrap()
        .build()
        .unwrap();
    (db, dir)
}

#[test]
fn add_vector_and_search() {
    let (db, _dir) = temp_vector_db(3);

    let r1 = db
        .insert("sessions", json!({"summary": "auth bug fix"}))
        .unwrap();
    let r2 = db
        .insert("sessions", json!({"summary": "deploy pipeline"}))
        .unwrap();
    let r3 = db
        .insert("sessions", json!({"summary": "unrelated task"}))
        .unwrap();

    db.add_vector(&r1.id, &[1.0, 0.0, 0.0]).unwrap();
    db.add_vector(&r2.id, &[0.9, 0.1, 0.0]).unwrap();
    db.add_vector(&r3.id, &[0.0, 0.0, 1.0]).unwrap();

    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.id, r1.id);
    assert!((results[0].1 - 1.0).abs() < 0.01);
    assert_eq!(results[1].0.id, r2.id);
    assert!(results[1].1 > 0.9);
}

#[test]
fn vector_persistence_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persist.axil");

    let record_id;
    {
        let db = Axil::open(&path).with_vector(3).unwrap().build().unwrap();

        let r = db.insert("data", json!({"text": "hello world"})).unwrap();
        record_id = r.id.clone();
        db.add_vector(&r.id, &[0.5, 0.5, 0.0]).unwrap();
    }

    {
        // Reopen with auto-detected dimensions.
        let db = Axil::open(&path)
            .with_vector_auto()
            .unwrap()
            .build()
            .unwrap();

        let results = db.similar_to_vector(&[0.5, 0.5, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, record_id);
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }
}

#[test]
fn delete_removes_vector() {
    let (db, _dir) = temp_vector_db(3);

    let r = db.insert("items", json!({"v": 1})).unwrap();
    db.add_vector(&r.id, &[1.0, 0.0, 0.0]).unwrap();

    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(results.len(), 1);

    db.delete(&r.id).unwrap();

    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn vector_search_with_different_dimensions() {
    let (db, _dir) = temp_vector_db(128);

    let r1 = db.insert("items", json!({"name": "a"})).unwrap();
    let r2 = db.insert("items", json!({"name": "b"})).unwrap();

    let mut v1 = vec![0.0_f32; 128];
    v1[0] = 1.0;
    let mut v2 = vec![0.0_f32; 128];
    v2[1] = 1.0;

    db.add_vector(&r1.id, &v1).unwrap();
    db.add_vector(&r2.id, &v2).unwrap();

    let results = db.similar_to_vector(&v1, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, r1.id);
}

#[test]
fn no_vector_index_errors_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_vec.axil");
    let db = Axil::open(&path).build().unwrap();

    let r = db.insert("items", json!({})).unwrap();
    assert!(db.add_vector(&r.id, &[1.0]).is_err());
    assert!(db.similar_to_vector(&[1.0], 5).is_err());
    assert!(!db.has_vector_index());
}

#[test]
fn query_builder_without_vector_still_works() {
    let (db, _dir) = temp_vector_db(3);

    db.insert("items", json!({"score": 10})).unwrap();
    db.insert("items", json!({"score": 20})).unwrap();

    let results = db
        .query()
        .table("items")
        .where_field("score", axil_core::Op::Gt, json!(15))
        .exec()
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data["score"], 20);
}

#[test]
fn top_k_zero_returns_empty() {
    let (db, _dir) = temp_vector_db(3);
    let r = db.insert("items", json!({})).unwrap();
    db.add_vector(&r.id, &[1.0, 0.0, 0.0]).unwrap();

    let results = db.similar_to_vector(&[1.0, 0.0, 0.0], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn add_vector_rejects_nonexistent_record() {
    let (db, _dir) = temp_vector_db(3);
    let fake_id = RecordId::new();
    let result = db.add_vector(&fake_id, &[1.0, 0.0, 0.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn embed_field_without_embedder_returns_clear_error() {
    let (db, _dir) = temp_vector_db(3);
    let r = db.insert("items", json!({"text": "hello"})).unwrap();

    let result = db.embed_field(&r.id, "text");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("embedder"));
}

#[test]
fn embed_text_without_embedder_returns_clear_error() {
    let (db, _dir) = temp_vector_db(3);
    let r = db.insert("items", json!({})).unwrap();

    let result = db.embed_text(&r.id, "some text");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("embedder"));
}

#[test]
fn similar_to_without_embedder_returns_clear_error() {
    let (db, _dir) = temp_vector_db(3);

    let result = db.similar_to("test query", 5);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("embedder"));
}

#[test]
fn embed_text_rejects_nonexistent_record() {
    let (db, _dir) = temp_vector_db(3);
    let fake_id = RecordId::new();

    let result = db.embed_text(&fake_id, "hello");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn embed_field_rejects_missing_field() {
    let (db, _dir) = temp_vector_db(3);
    let r = db.insert("items", json!({"other": 42})).unwrap();

    let result = db.embed_field(&r.id, "nonexistent");
    assert!(result.is_err());
}

#[test]
fn add_vector_rejects_nan() {
    let (db, _dir) = temp_vector_db(3);
    let r = db.insert("items", json!({})).unwrap();

    let result = db.add_vector(&r.id, &[1.0, f32::NAN, 0.0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NaN"));
}

// ── New tests ─────────────────────────────────────────────

#[test]
fn files_returns_correct_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).with_vector(3).unwrap().build().unwrap();

    let files = db.files();
    assert!(
        files.len() >= 2,
        "expected at least core + vec files, got {:?}",
        files
    );
    assert!(files.contains(&path));

    let vec_path = {
        let mut p = path.as_os_str().to_owned();
        p.push(".vec");
        std::path::PathBuf::from(p)
    };
    assert!(files.contains(&vec_path));
}

#[test]
fn database_size_includes_all_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).with_vector(3).unwrap().build().unwrap();

    let size = db.database_size();
    assert!(size > 0);
}

#[test]
fn info_returns_unified_stats() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).with_vector(3).unwrap().build().unwrap();

    db.insert("sessions", json!({"summary": "test"})).unwrap();
    db.insert("patterns", json!({"name": "x"})).unwrap();

    let info = db.info().unwrap();
    assert_eq!(info.total_records, 2);
    assert_eq!(info.tables.len(), 2);
    assert!(info.total_size > 0);
    assert!(info.files.len() >= 2);
}

#[test]
fn with_vector_auto_detects_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auto.axil");

    // Create with explicit dimensions.
    {
        let db = Axil::open(&path).with_vector(64).unwrap().build().unwrap();
        let r = db.insert("items", json!({})).unwrap();
        let v = vec![0.1_f32; 64];
        db.add_vector(&r.id, &v).unwrap();
    }

    // Reopen with auto-detect.
    {
        let db = Axil::open(&path)
            .with_vector_auto()
            .unwrap()
            .build()
            .unwrap();

        assert!(db.has_vector_index());
        let v = vec![0.1_f32; 64];
        let results = db.similar_to_vector(&v, 1).unwrap();
        assert_eq!(results.len(), 1);
    }
}

#[test]
fn with_vector_auto_fails_without_vec_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_vec.axil");

    // Create a plain database without vector.
    let _db = Axil::open(&path).build().unwrap();

    // with_vector_auto should fail.
    let result = Axil::open(&path).with_vector_auto();
    assert!(result.is_err());
}

#[test]
fn read_stored_dimensions_works() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dims.axil");

    // No vec store yet.
    assert_eq!(axil_vector::read_stored_dimensions(&path).unwrap(), None);

    // Create vector store and drop it so the file is released.
    {
        let _db = Axil::open(&path).with_vector(256).unwrap().build().unwrap();
    }

    assert_eq!(
        axil_vector::read_stored_dimensions(&path).unwrap(),
        Some(256)
    );
}

#[test]
fn path_getter_returns_correct_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("path_test.axil");
    let db = Axil::open(&path).build().unwrap();
    assert_eq!(db.path(), path);
}

// ── Named vector spaces ───────────────────────────────────────────

/// A DB holding a 384-dim default space + a 5-dim `fp` space must keep them
/// isolated (writes/searches never cross) and persist both across reopen.
#[test]
fn named_space_isolation_and_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spaces.axil");

    let (def_id, fp_id);
    {
        let db = Axil::open(&path)
            .with_vector(384)
            .unwrap()
            .with_vector_spaces()
            .build()
            .unwrap();

        let d = db.insert("docs", json!({"summary": "text embedding record"})).unwrap();
        let f = db.insert("strats", json!({"summary": "strategy fingerprint"})).unwrap();
        def_id = d.id.clone();
        fp_id = f.id.clone();

        // Default space: 384-dim.
        let mut dv = vec![0.0_f32; 384];
        dv[0] = 1.0;
        db.add_vector(&def_id, &dv).unwrap();

        // Named `fp` space: 5-dim. Independent dimension, no collision.
        db.add_vector_in("fp", &fp_id, &[1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        // Search does not cross spaces.
        let def_hits = db.similar_to_vector(&dv, 5).unwrap();
        assert_eq!(def_hits.len(), 1);
        assert_eq!(def_hits[0].0.id, def_id);

        let fp_hits = db.similar_in("fp", &[1.0, 0.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(fp_hits.len(), 1);
        assert_eq!(fp_hits[0].0.id, fp_id);
    }

    // Reopen: both spaces persist with their own dimensions.
    {
        let db = Axil::open(&path)
            .with_vector_auto()
            .unwrap()
            .with_vector_spaces()
            .build()
            .unwrap();

        let mut dv = vec![0.0_f32; 384];
        dv[0] = 1.0;
        let def_hits = db.similar_to_vector(&dv, 5).unwrap();
        assert_eq!(def_hits.len(), 1);
        assert_eq!(def_hits[0].0.id, def_id);

        let fp_hits = db.similar_in("fp", &[1.0, 0.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(fp_hits.len(), 1);
        assert_eq!(fp_hits[0].0.id, fp_id);

        // Round-trip the stored fingerprint.
        let stored = db.get_vector_in("fp", &fp_id).unwrap().unwrap();
        assert_eq!(stored, vec![1.0, 0.0, 0.0, 0.0, 0.0]);

        // vector_spaces() reports the named space (default is not "named").
        let spaces = db.vector_spaces().unwrap();
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].name, "fp");
        assert_eq!(spaces[0].dimensions, 5);
        assert_eq!(spaces[0].count, 1);
    }
}

/// A second write to a space with a different dimension errors.
#[test]
fn named_space_dimension_mismatch_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.axil");
    let db = Axil::open(&path)
        .with_vector(3)
        .unwrap()
        .with_vector_spaces()
        .build()
        .unwrap();

    let a = db.insert("s", json!({})).unwrap();
    let b = db.insert("s", json!({})).unwrap();

    db.add_vector_in("fp", &a.id, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    // Wrong length for the same (cached) space.
    let err = db.add_vector_in("fp", &b.id, &[1.0, 0.0]).unwrap_err();
    assert!(
        err.to_string().contains("dimension mismatch"),
        "unexpected error: {err}"
    );
}

/// Space names must match `[a-z0-9_-]{1,32}`.
#[test]
fn named_space_invalid_name_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("badname.axil");
    let db = Axil::open(&path)
        .with_vector(3)
        .unwrap()
        .with_vector_spaces()
        .build()
        .unwrap();
    let r = db.insert("s", json!({})).unwrap();
    assert!(db.add_vector_in("Bad Name", &r.id, &[1.0, 0.0, 0.0]).is_err());
    assert!(db.add_vector_in("", &r.id, &[1.0, 0.0, 0.0]).is_err());
    assert!(db
        .add_vector_in(&"x".repeat(33), &r.id, &[1.0, 0.0, 0.0])
        .is_err());
}

/// A ~0.97-cosine twin ranks just below the exact self-match and well above an
/// orthogonal decoy — the near-duplicate detection the CLI `similar --threshold`
/// flag is built on.
#[test]
fn named_space_near_duplicate_ranks_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("neardup.axil");
    let db = Axil::open(&path)
        .with_vector(8)
        .unwrap()
        .with_vector_spaces()
        .build()
        .unwrap();

    let a = db.insert("fp", json!({"name": "a"})).unwrap();
    let b = db.insert("fp", json!({"name": "b"})).unwrap();
    let c = db.insert("fp", json!({"name": "c"})).unwrap();

    let va = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // cos(va, vb) = 1/sqrt(1.0625) ≈ 0.970.
    let vb = [1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let vc = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    db.add_vector_in("fp", &a.id, &va).unwrap();
    db.add_vector_in("fp", &b.id, &vb).unwrap();
    db.add_vector_in("fp", &c.id, &vc).unwrap();

    let hits = db.similar_in("fp", &va, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0.id, a.id);
    assert!((hits[0].1 - 1.0).abs() < 0.01);
    assert_eq!(hits[1].0.id, b.id);
    assert!(hits[1].1 > 0.9, "twin score {} should exceed 0.9", hits[1].1);
    assert_eq!(hits[2].0.id, c.id);
    assert!(hits[2].1 < 0.5, "decoy score {} should be low", hits[2].1);
}

/// Named-space operations error clearly when no factory is registered.
#[test]
fn named_space_without_factory_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nofactory.axil");
    let db = Axil::open(&path).with_vector(3).unwrap().build().unwrap();
    let r = db.insert("s", json!({})).unwrap();
    assert!(db.add_vector_in("fp", &r.id, &[1.0, 0.0, 0.0]).is_err());
    // With no factory, listing is simply empty (not an error).
    assert!(db.vector_spaces().unwrap().is_empty());
}
