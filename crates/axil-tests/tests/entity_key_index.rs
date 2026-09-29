//! The `_entities` key index must leave auto-linking's results unchanged.
//!
//! Each fixture runs twice: once through the index and once with the index
//! dropped, which sends `auto_link` down the full-scan path it always used.
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
    assert!(db.storage().entity_key_index_ready().unwrap());
    if !use_index {
        db.storage().drop_entity_key_index().unwrap();
        assert!(!db.storage().entity_key_index_ready().unwrap());
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

#[test]
fn opening_a_store_without_the_index_builds_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.axil");
    {
        let db = open_with_graph(&path);
        db.storage().drop_entity_key_index().unwrap();
        for text in PHASE_A {
            store(&db, text);
        }
        assert!(!db.storage().entity_key_index_ready().unwrap());
    }

    let db = open_with_graph(&path);
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
    let indexed = db.storage().lookup_entity_keys(&keys).unwrap().unwrap();
    assert_eq!(indexed, scan);

    // Linking a repeat mention reuses the migrated rows instead of adding
    // new ones.
    store(&db, PHASE_A[0]);
    assert_eq!(
        db.storage().list("_entities", usize::MAX, 0).unwrap().len(),
        rows.len()
    );
}
