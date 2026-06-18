use axil_core::{Axil, RecordId};
use chrono::{TimeZone, Utc};
use serde_json::json;

fn temp_db() -> (Axil, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.axil");
    let db = Axil::open(&path).build().unwrap();
    (db, dir)
}

#[test]
fn crud_insert_get() {
    let (db, _dir) = temp_db();
    let record = db
        .insert("sessions", json!({"summary": "fixed auth bug"}))
        .unwrap();
    let fetched = db.get(&record.id).unwrap().unwrap();
    assert_eq!(fetched.id, record.id);
    assert_eq!(fetched.table, "sessions");
    assert_eq!(fetched.data["summary"], "fixed auth bug");
}

#[test]
fn crud_update() {
    let (db, _dir) = temp_db();
    let record = db.insert("sessions", json!({"v": 1})).unwrap();
    let updated = db
        .update(&record.id, json!({"v": 2, "extra": true}))
        .unwrap();
    assert_eq!(updated.data["v"], 2);
    assert_eq!(updated.data["extra"], true);
    assert!(updated.updated_at >= record.created_at);
}

#[test]
fn crud_insert_at_preserves_timestamp() {
    let (db, _dir) = temp_db();
    let created_at = Utc.with_ymd_and_hms(2023, 5, 20, 2, 22, 0).unwrap();
    let record = db
        .insert_at(
            "sessions",
            json!({"summary": "historical import"}),
            created_at,
        )
        .unwrap();
    let fetched = db.get(&record.id).unwrap().unwrap();
    assert_eq!(fetched.created_at, created_at);
    assert_eq!(fetched.updated_at, created_at);
}

#[test]
fn crud_delete() {
    let (db, _dir) = temp_db();
    let record = db.insert("sessions", json!({})).unwrap();
    assert!(db.delete(&record.id).unwrap());
    assert!(db.get(&record.id).unwrap().is_none());
    // Double delete returns false.
    assert!(!db.delete(&record.id).unwrap());
}

#[test]
fn crud_list() {
    let (db, _dir) = temp_db();
    db.insert("items", json!({"a": 1})).unwrap();
    db.insert("items", json!({"a": 2})).unwrap();
    db.insert("other", json!({"b": 1})).unwrap();

    let items = db.list("items").unwrap();
    assert_eq!(items.len(), 2);

    let other = db.list("other").unwrap();
    assert_eq!(other.len(), 1);
}

#[test]
fn crud_not_found() {
    let (db, _dir) = temp_db();
    let id = RecordId::new();
    assert!(db.get(&id).unwrap().is_none());
    assert!(!db.delete(&id).unwrap());
}

#[test]
fn multiple_tables_same_db() {
    let (db, _dir) = temp_db();
    db.insert("sessions", json!({"type": "session"})).unwrap();
    db.insert("decisions", json!({"type": "decision"})).unwrap();
    db.insert("patterns", json!({"type": "pattern"})).unwrap();

    let mut tables = db.tables().unwrap();
    tables.sort();
    assert_eq!(tables, vec!["decisions", "patterns", "sessions"]);
}

#[test]
fn persistence_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persist.axil");

    let id = {
        let db = Axil::open(&path).build().unwrap();
        let r = db.insert("data", json!({"persisted": true})).unwrap();
        r.id
    };

    // Reopen the database.
    let db = Axil::open(&path).build().unwrap();
    let fetched = db.get(&id).unwrap().unwrap();
    assert_eq!(fetched.data["persisted"], true);
    assert_eq!(db.total_records().unwrap(), 1);
}

#[test]
fn batch_insert() {
    let (db, _dir) = temp_db();
    let items = vec![
        json!({"name": "a"}),
        json!({"name": "b"}),
        json!({"name": "c"}),
    ];
    let records = db.insert_batch("items", items).unwrap();
    assert_eq!(records.len(), 3);

    let all = db.list("items").unwrap();
    assert_eq!(all.len(), 3);

    // Each record should be retrievable.
    for r in &records {
        assert!(db.get(&r.id).unwrap().is_some());
    }
}

#[test]
fn batch_insert_empty() {
    let (db, _dir) = temp_db();
    let records = db.insert_batch("items", vec![]).unwrap();
    assert!(records.is_empty());
}

#[test]
fn database_info() {
    let (db, _dir) = temp_db();
    db.insert("a", json!({})).unwrap();
    db.insert("a", json!({})).unwrap();
    db.insert("b", json!({})).unwrap();

    assert_eq!(db.total_records().unwrap(), 3);
    assert_eq!(db.count("a").unwrap(), 2);
    assert_eq!(db.count("b").unwrap(), 1);
}
