//! The `_entities` key index must leave auto-linking's results unchanged.
//!
//! Each fixture runs twice: once through the index and once with the index
//! switched off, which sends `auto_link` down the full-scan path it always
//! used.
//! Record ids differ between the two databases, so the runs are compared by
//! record bodies — every `_entities` row in list order, and the body of each
//! entity every stored memory links to.

use axil_core::{Axil, Direction, Record, RecordId};
use axil_graph::AxilBuilderGraphExt;
use serde_json::{json, Value};

fn open_with_graph(path: &std::path::Path) -> Axil {
    Axil::open(path)
        .with_graph_engine()
        .unwrap()
        .build()
        .unwrap()
}

const PHASE_A: [&str; 3] = [
    "Fixed bug in `AuthModule` by updating `auth_config` for Redis",
    "The `ConnectionPool` in src/db.rs leaked under PostgreSQL load",
    "Switched Redis client to `ConnectionPool::with_timeout` and ONNX Runtime",
];

const PHASE_B: [&str; 3] = [
    "Redis and PostgreSQL both need `AuthModule` changes",
    "`auth_config` now reads ONNX Runtime paths; see `ConnectionPool`",
    "New `RateLimiter` wraps Redis",
];

/// Store a memory the way the CLI does, so `Axil::insert` auto-links it.
fn store(db: &Axil, summary: &str) -> RecordId {
    db.insert(
        "decisions",
        json!({ "summary": summary, "file": "src/db.rs" }),
    )
    .unwrap()
    .id
}

/// Entity bodies each memory mentions, sorted so edge order doesn't matter.
fn mentions(db: &Axil, id: &RecordId) -> Vec<String> {
    let mut out: Vec<String> = db
        .neighbors(id, Some("mentions"), Direction::Out)
        .unwrap()
        .into_iter()
        .map(|r| r.data.to_string())
        .collect();
    out.sort();
    out
}

#[derive(Debug, PartialEq)]
struct Outcome {
    entities: Vec<Value>,
    mentions: Vec<Vec<String>>,
}

fn run_fixture(use_index: bool) -> Outcome {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.axil");
    let db = open_with_graph(&path);
    if !use_index {
        db.storage().set_entity_key_index_enabled(false);
    }

    let mut memories = Vec::new();
    for text in PHASE_A {
        memories.push(store(&db, text));
    }

    // Second copies of every entity so far, under ids that sort *before*
    // the originals but list after them: a later lookup must pick the copy.
    let existing = db.storage().list("_entities", usize::MAX, 0).unwrap();
    assert!(!existing.is_empty(), "fixture text must extract entities");
    for (i, row) in existing.iter().enumerate() {
        let mut data = row.data.clone();
        data["copy"] = json!(i);
        let mut copy = Record::new("_entities", data);
        copy.id = RecordId(format!("0000-copy-{i:03}"));
        db.storage().insert(&copy).unwrap();
    }
    // A pre-migration style row keyed only by name, sharing a key with a
    // canonical row, and one canonical_id rewrite (the SCIP upgrade shape).
    let first_key = existing[0].data["canonical_id"]
        .as_str()
        .unwrap()
        .to_string();
    db.storage()
        .insert(&Record::new(
            "_entities",
            json!({ "name": first_key, "legacy": true }),
        ))
        .unwrap();
    let rewritten = &existing[existing.len() - 1];
    let mut upgraded = rewritten.data.clone();
    upgraded["canonical_id"] = json!("scip:rust:fixture/Upgraded#");
    db.update(&rewritten.id, upgraded).unwrap();

    for text in PHASE_B {
        memories.push(store(&db, text));
    }

    assert_eq!(
        db.storage().entity_key_index_ready().unwrap(),
        use_index,
        "the run must use the path it is testing"
    );

    Outcome {
        entities: db
            .storage()
            .list("_entities", usize::MAX, 0)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect(),
        mentions: memories.iter().map(|id| mentions(&db, id)).collect(),
    }
}

#[test]
fn auto_link_links_the_same_entities_with_and_without_the_index() {
    let indexed = run_fixture(true);
    let scanned = run_fixture(false);
    assert!(
        indexed.mentions.iter().all(|m| !m.is_empty()),
        "every fixture memory should mention something: {:?}",
        indexed.mentions
    );
    // Phase B must hit the duplicated keys, or the fixture proves nothing
    // about which duplicate wins.
    assert!(
        indexed.mentions[3..]
            .iter()
            .flatten()
            .any(|m| m.contains("\"copy\"")),
        "phase B should link to copied entities: {:?}",
        indexed.mentions
    );
    assert_eq!(indexed, scanned);
}

/// A store written without the index gets it from the first insert that
/// auto-links, not from opening: commands that never resolve entities (the
/// hook's lookups among them) must not pay for, or commit, the build.
#[test]
fn the_first_auto_link_builds_the_index_and_opening_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.axil");
    {
        let db = open_with_graph(&path);
        db.storage().set_entity_key_index_enabled(false);
        for text in PHASE_A {
            store(&db, text);
        }
    }

    {
        let db = open_with_graph(&path);
        db.search_text("AuthModule", 5).ok();
        db.list("decisions").unwrap();
        assert!(!db.storage().entity_key_index_ready().unwrap());
    }

    let db = open_with_graph(&path);
    assert!(!db.storage().entity_key_index_ready().unwrap());
    let before = db.storage().list("_entities", usize::MAX, 0).unwrap().len();
    store(&db, "Unrelated note with no entities");
    assert!(
        !db.storage().entity_key_index_ready().unwrap(),
        "nothing to resolve, nothing to build"
    );
    store(&db, PHASE_A[0]);
    assert_eq!(
        db.storage().list("_entities", usize::MAX, 0).unwrap().len(),
        before
    );
    assert!(db.storage().entity_key_index_ready().unwrap());

    let rows = db.storage().list("_entities", usize::MAX, 0).unwrap();
    let mut scan = std::collections::HashMap::new();
    for row in &rows {
        let key = row.data["canonical_id"]
            .as_str()
            .or_else(|| row.data["name"].as_str());
        if let Some(key) = key {
            scan.insert(key.to_string(), row.id.clone());
        }
    }
    let keys: Vec<&str> = scan.keys().map(String::as_str).collect();
    let indexed = db
        .storage()
        .lookup_entity_keys(&keys)
        .unwrap()
        .into_found()
        .unwrap();
    assert_eq!(indexed, scan);

    // Linking a repeat mention reuses the migrated rows instead of adding
    // new ones.
    store(&db, PHASE_A[0]);
    assert_eq!(
        db.storage().list("_entities", usize::MAX, 0).unwrap().len(),
        rows.len()
    );
}

/// Writes to a store the way an axil binary that predates the key index
/// does: the `records` and `table_index` tables, whose layout every version
/// shares, and nothing else. Such a binary is on the same file whenever the
/// hooks run the `axil` on PATH while an agent runs a newer build.
mod older_binary {
    use axil_core::{Record, RecordId};
    use redb::{ReadableTable, TableDefinition};
    use serde_json::Value;
    use std::path::Path;

    const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records");
    const TABLE_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("table_index");

    fn listed(idx: &impl ReadableTable<&'static str, &'static [u8]>, table: &str) -> Vec<String> {
        idx.get(table)
            .unwrap()
            .map(|g| serde_json::from_slice(g.value()).unwrap())
            .unwrap_or_default()
    }

    /// Add an `_entities` row, as an older binary's auto-link does for an
    /// entity it has not seen before.
    pub fn insert_entity(path: &Path, body: Value) -> RecordId {
        let record = Record::new("_entities", body);
        let db = redb::Database::create(path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            let bytes = record.to_bytes().unwrap();
            records
                .insert(record.id.as_str(), bytes.as_slice())
                .unwrap();
            let mut idx = txn.open_table(TABLE_INDEX).unwrap();
            let mut ids = listed(&idx, "_entities");
            ids.push(record.id.as_str().to_string());
            let bytes = serde_json::to_vec(&ids).unwrap();
            idx.insert("_entities", bytes.as_slice()).unwrap();
        }
        txn.commit().unwrap();
        record.id
    }

    /// Delete an `_entities` row, as `axil delete <id>` from an older binary
    /// does to the core file. Its graph edges are left alone: the checks
    /// below are about which entity rows resolution finds.
    pub fn delete_entity(path: &Path, id: &RecordId) {
        let db = redb::Database::create(path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut records = txn.open_table(RECORDS).unwrap();
            records.remove(id.as_str()).unwrap();
            let mut idx = txn.open_table(TABLE_INDEX).unwrap();
            let mut ids = listed(&idx, "_entities");
            ids.retain(|rid| rid != id.as_str());
            let bytes = serde_json::to_vec(&ids).unwrap();
            idx.insert("_entities", bytes.as_slice()).unwrap();
        }
        txn.commit().unwrap();
    }
}

/// Store a bare summary, as `axil store decisions '{"summary": ...}'` does.
fn store_summary(db: &Axil, summary: &str) -> RecordId {
    db.insert("decisions", json!({ "summary": summary }))
        .unwrap()
        .id
}

fn entity_key(row: &Record) -> String {
    row.data["canonical_id"]
        .as_str()
        .or_else(|| row.data["name"].as_str())
        .unwrap_or_default()
        .to_string()
}

/// The `_entities` rows auto-linking creates for `summary` in an empty
/// store. Entity creation is the same in every version, so these are the
/// rows an older binary's `store` of the same text adds.
fn entity_rows_for(summary: &str) -> Vec<Value> {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_graph(&dir.path().join("rows.axil"));
    store_summary(&db, summary);
    db.storage()
        .list("_entities", usize::MAX, 0)
        .unwrap()
        .into_iter()
        .map(|r| r.data)
        .collect()
}

fn rows_keyed(db: &Axil, key: &str) -> Vec<Record> {
    db.storage()
        .list("_entities", usize::MAX, 0)
        .unwrap()
        .into_iter()
        .filter(|r| entity_key(r) == key)
        .collect()
}

/// An older binary stores a memory whose entities this build's index has
/// never seen. Resolving against the index as built would create a second
/// row for each of them; the next store must find the older binary's rows.
#[test]
fn an_older_binarys_new_entities_are_found_not_duplicated() {
    use axil_core::EntityKeyIndexStatus;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.axil");
    {
        let db = open_with_graph(&path);
        store_summary(&db, "We chose `redis_cache` for SessionCache");
        assert!(db.storage().entity_key_index_ready().unwrap(), "built");
    }

    let older_text = "`kafka_broker` carries AuditEvents";
    let mut older_rows = Vec::new();
    for body in entity_rows_for(older_text) {
        older_rows.push(older_binary::insert_entity(&path, body));
    }
    assert_eq!(older_rows.len(), 2, "kafka_broker and AuditEvents");

    let db = open_with_graph(&path);
    assert_eq!(
        db.storage().entity_key_index_status().unwrap(),
        EntityKeyIndexStatus::Stale
    );
    let rec = store_summary(&db, "`kafka_broker` partitions raised for AuditEvents");
    for id in &older_rows {
        let row = db.get(id).unwrap().expect("older binary's row");
        assert_eq!(
            rows_keyed(&db, &entity_key(&row)).len(),
            1,
            "one row per entity, as with the older binary alone"
        );
    }
    let mut linked: Vec<RecordId> = db
        .neighbors(&rec, Some("mentions"), Direction::Out)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    linked.sort();
    older_rows.sort();
    assert_eq!(linked, older_rows, "the new memory links the older rows");
    assert!(db.storage().entity_key_index_ready().unwrap(), "repaired");
}

/// An older binary deletes an entity this build's index holds. Resolving
/// against the index as built would link the next mention to the deleted
/// id, leaving no entity row; the next store must create a live one.
#[test]
fn an_entity_an_older_binary_deleted_is_recreated_not_dangling() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.axil");
    let deleted = {
        let db = open_with_graph(&path);
        store_summary(&db, "`zorblax_frobnicator` owns the queue");
        let rows = rows_keyed(&db, "zorblax_frobnicator");
        assert_eq!(rows.len(), 1);
        assert!(db.storage().entity_key_index_ready().unwrap());
        rows[0].id.clone()
    };

    older_binary::delete_entity(&path, &deleted);

    let db = open_with_graph(&path);
    let rec = store_summary(&db, "`zorblax_frobnicator` partitions raised");
    let rows = rows_keyed(&db, "zorblax_frobnicator");
    assert_eq!(
        rows.len(),
        1,
        "a new entity row, as with the older binary alone"
    );
    assert_ne!(rows[0].id, deleted);
    let linked: Vec<RecordId> = db
        .neighbors(&rec, Some("mentions"), Direction::Out)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(linked, vec![rows[0].id.clone()], "the mention is live");
    assert!(db.storage().entity_key_index_ready().unwrap(), "repaired");
}
