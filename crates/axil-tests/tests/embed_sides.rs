//! Which side of an asymmetric embedding model each path embeds on. Models
//! such as nomic-embed-text are trained with one prompt for search queries and
//! another for stored text: a search embeds a query, while stored text, and
//! text compared with stored text as an equal (duplicate and supersession
//! checks), embeds a passage.

use std::sync::{Arc, Mutex};

use axil_core::brain::{remember, Observation, PipelineAction};
use axil_core::{Axil, TextEmbedder};
use axil_vector::VectorEngine;
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Query,
    Passage,
}

type Log = Arc<Mutex<Vec<(Side, String)>>>;

/// A bag of words in the first half of the vector for passages and in the
/// second half for queries, so a query never matches a passage, not even its
/// own text: any check that embeds on the wrong side finds nothing.
struct Sided(Log);

const HALF: usize = 8;

fn bag(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; HALF];
    for word in text.split_whitespace() {
        let bucket = word.to_lowercase().bytes().map(usize::from).sum::<usize>() % (HALF - 1);
        v[bucket] += 1.0;
    }
    v[HALF - 1] = 1.0;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

impl TextEmbedder for Sided {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.0
            .lock()
            .unwrap()
            .push((Side::Passage, text.to_string()));
        let mut v = bag(text);
        v.extend(std::iter::repeat(0.0).take(HALF));
        Ok(v)
    }

    fn embed_query(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.0.lock().unwrap().push((Side::Query, text.to_string()));
        let mut v = vec![0.0; HALF];
        v.extend(bag(text));
        Ok(v)
    }
}

fn sided_db() -> (Axil, Log, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sides.axil");
    let log: Log = Arc::default();
    let db = Axil::open(&path)
        .with_vector_index(Box::new(VectorEngine::open(&path, 2 * HALF).unwrap()))
        .with_embedder(Box::new(Sided(log.clone())))
        .build()
        .unwrap();
    (db, log, dir)
}

fn sides_for(log: &Log, text: &str) -> Vec<Side> {
    log.lock()
        .unwrap()
        .iter()
        .filter(|(_, t)| t == text)
        .map(|(s, _)| *s)
        .collect()
}

#[test]
fn searches_embed_a_query_and_stored_text_a_passage() {
    let (db, log, _dir) = sided_db();
    let rec = db
        .insert(
            "notes",
            json!({"summary": "connection pool timeout under load"}),
        )
        .unwrap();
    db.embed_field(&rec.id, "summary").unwrap();
    assert!(!sides_for(&log, "connection pool timeout under load").is_empty());
    assert!(
        !sides_for(&log, "connection pool timeout under load").contains(&Side::Query),
        "stored text embeds as a passage"
    );

    db.similar_to("pool timeout", 5).unwrap();
    db.query().similar_to("why the pool", 5).exec().unwrap();
    db.embed_query("pool exhaustion").unwrap();
    for q in ["pool timeout", "why the pool", "pool exhaustion"] {
        assert_eq!(sides_for(&log, q), vec![Side::Query], "{q} is a search");
    }

    db.similar_to_passage("pool timeout again", 5).unwrap();
    db.embed_passage("a cached question").unwrap();
    for p in ["pool timeout again", "a cached question"] {
        assert_eq!(
            sides_for(&log, p),
            vec![Side::Passage],
            "{p} compares as an equal"
        );
    }
}

#[test]
fn remembering_the_same_text_twice_finds_the_duplicate() {
    let (db, _log, _dir) = sided_db();
    let text = "Decided to cap the connection pool at 32 because the database \
                rejects more than 40 concurrent sessions under load";
    let first = remember(&db, Observation::from_text(text)).unwrap();
    assert_eq!(first.action, PipelineAction::Stored, "{}", first.reason);

    // A query-side probe would score 0 against the stored passage here and
    // store the same memory twice.
    let second = remember(&db, Observation::from_text(text)).unwrap();
    assert_eq!(second.action, PipelineAction::Ignored, "{}", second.reason);
    assert!(
        second.reason.starts_with("duplicate of"),
        "{}",
        second.reason
    );
}

#[test]
fn supersession_compares_a_new_record_as_a_passage() {
    let (db, log, _dir) = sided_db();
    let old = db
        .insert("decisions", json!({"summary": "use redb for core storage"}))
        .unwrap();
    db.embed_field(&old.id, "summary").unwrap();
    let new = db
        .insert(
            "decisions",
            json!({"summary": "use redb for the core storage layer"}),
        )
        .unwrap();

    log.lock().unwrap().clear();
    axil_memory::SupersedeEngine::new(&db)
        .check_and_supersede(&new)
        .unwrap();
    let sides: Vec<Side> = log.lock().unwrap().iter().map(|(s, _)| *s).collect();
    assert!(!sides.is_empty(), "the check embeds the new record");
    assert!(!sides.contains(&Side::Query), "never as a query: {sides:?}");
}
