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
