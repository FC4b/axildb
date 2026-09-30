//! Recall's ranking must not depend on how many results were asked for, and
//! recency must not decide which candidates get the query-time-chunk blend.
//!
//! - `recall(q, k)` is a prefix of `recall(q, k')` for any `k < k'` up to the
//!   candidate pool. A pool that grew with `k` let graph proximity (links to
//!   the rest of the pool) lift well-connected records over the answer.
//! - The chunk blend goes to the most relevant candidates, ranked without
//!   recency. When the window followed the plain fused order, a relevant old
//!   record that recency pushed below it was never rescored.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axil_core::scoring::QtcConfig;
use axil_core::{Axil, RecallConfig, ScoreWeights, TextEmbedder};
use axil_graph::GraphEngine;
use axil_vector::VectorEngine;
use chrono::{Duration, Utc};
use serde_json::json;

const DIMS: usize = 32;

/// Bag of words: a few words own a dimension each, every other word hashes
/// into the rest, so cosines follow word overlap and are easy to reason about.
fn bag(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    for word in text.split_whitespace() {
        let w = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        let dim = match w.as_str() {
            "connection" => 0,
            "pool" => 1,
            "timeout" => 2,
            _ => 3 + w.bytes().map(usize::from).sum::<usize>() % (DIMS - 3),
        };
        v[dim] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[derive(Default)]
struct Bag {
    /// Texts embedded through `embed_batch`, which query-time chunking uses.
    batched_texts: AtomicUsize,
}

impl TextEmbedder for Bag {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        Ok(bag(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> axil_core::Result<Vec<Vec<f32>>> {
        self.batched_texts.fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts.iter().map(|t| bag(t)).collect())
    }
}

fn open(path: &std::path::Path, embedder: Arc<Bag>, graph: bool) -> Axil {
    let mut builder = Axil::open(path)
        .with_vector_index(Box::new(VectorEngine::open(path, DIMS).unwrap()))
        .with_embedder(Box::new(SharedBag(embedder)));
    if graph {
        builder = builder.with_graph_index(Arc::new(GraphEngine::open(path).unwrap()));
    }
    builder.build().unwrap()
}

struct SharedBag(Arc<Bag>);

impl TextEmbedder for SharedBag {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.0.embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> axil_core::Result<Vec<Vec<f32>>> {
        self.0.embed_batch(texts)
    }
}

/// The CLI's recall configuration: vector/recency split at the default alpha,
/// query-time chunks on.
fn cli_config() -> RecallConfig {
    RecallConfig {
        weights: ScoreWeights::with_vector_recency_split(0.7),
        qtc: Some(QtcConfig::default()),
        ..Default::default()
    }
}

fn ids(db: &Axil, query: &str, top_k: usize, cfg: RecallConfig) -> Vec<String> {
    db.recall(query, top_k, Some(cfg))
        .unwrap()
        .into_iter()
        .map(|r| r.record.id.to_string())
        .collect()
}

/// Deterministic pseudo-random stream so the corpus is the same every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

#[test]
fn a_smaller_top_k_returns_a_prefix_of_a_larger_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pool.axil");
    let db = open(&path, Arc::default(), true);

    const VOCAB: &[&str] = &[
        "connection", "pool", "timeout", "cache", "retry", "index", "vector", "graph", "hook",
        "daemon", "socket", "budget", "boot", "commit", "schema", "entity", "drain", "lock",
    ];
    let mut rng = Lcg(7);
    let mut records = Vec::new();
    let now = Utc::now();
    for i in 0..300 {
        let words: Vec<&str> = (0..6)
            .map(|_| VOCAB[rng.next() as usize % VOCAB.len()])
            .collect();
        let rec = db
            .insert_at(
                "notes",
                json!({ "summary": format!("{} note {i}", words.join(" ")) }),
                now - Duration::hours(rng.next() as i64 % 2000),
            )
            .unwrap();
        records.push(rec.id);
    }
    // A few hubs linked to many records, so graph proximity depends on which
    // records share the pool.
    for hub in 0..4 {
        for _ in 0..80 {
            let other = &records[rng.next() as usize % records.len()];
            if other != &records[hub] {
                db.relate(&records[hub], "relates_to", other, None).unwrap();
            }
        }
    }

    for query in [
        "connection pool timeout",
        "vector index budget",
        "daemon socket lock",
        "boot schema commit",
        "hook drain retry",
    ] {
        let k5 = ids(&db, query, 5, cli_config());
        let k10 = ids(&db, query, 10, cli_config());
        let k25 = ids(&db, query, 25, cli_config());
        let k40 = ids(&db, query, 40, cli_config());
        assert_eq!(k5, k10[..5], "{query}: top 5 of k=10 differs from k=5");
        assert_eq!(k10, k25[..10], "{query}: top 10 of k=25 differs from k=10");
        assert_eq!(k25, k40[..25], "{query}: top 25 of k=40 differs from k=25");
    }
}

#[test]
fn an_old_record_that_matches_best_is_not_stuck_below_a_rescoring_window() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("window.axil");
    let db = open(&path, Arc::default(), false);
    let now = Utc::now();

    // Cosine with the query ≈ 0.87, but 400 days old: the fused score, which
    // carries recency, ranks it below every recent record.
    let old = db
        .insert_at(
            "decisions",
            json!({"summary": "connection pool timeout retry"}),
            now - Duration::days(400),
        )
        .unwrap();
    // Cosine ≈ 0.67 and brand new: more of these than the old 20-record
    // rescoring window.
    for i in 0..30 {
        db.insert_at(
            "commits",
            json!({"summary": format!("connection pool cache{i}")}),
            now - Duration::minutes(i),
        )
        .unwrap();
    }

    let query = "connection pool timeout";
    let ranked = ids(&db, query, 5, cli_config());
    assert_eq!(
        ranked.first(),
        Some(&old.id.to_string()),
        "the best content match ranks first: {ranked:?}"
    );

    // With room for one record in the window, the blend goes to the most
    // relevant record, not to the newest one that leads the fused order.
    let narrow = RecallConfig {
        qtc: Some(QtcConfig {
            top_k: 1,
            ..QtcConfig::default()
        }),
        ..cli_config()
    };
    let results = db.recall(query, 5, Some(narrow)).unwrap();
    let blended: Vec<String> = results
        .iter()
        .filter(|r| r.explanation.signals.iter().any(|(n, _)| n == "qtc_best_chunk"))
        .map(|r| r.record.id.to_string())
        .collect();
    assert_eq!(blended, vec![old.id.to_string()]);
}

#[test]
fn only_the_window_is_embedded_at_query_time() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("budget.axil");
    // Insert with FTS only, so nothing gets a vector.
    {
        let db = Axil::open(&path)
            .with_fts_index(Arc::new(axil_fts::FtsEngine::open(&path).unwrap()))
            .build()
            .unwrap();
        for i in 0..12 {
            db.insert("notes", json!({"summary": format!("connection pool note {i}")}))
                .unwrap();
        }
    }

    let bag = Arc::new(Bag::default());
    let db = Axil::open(&path)
        .with_vector_index(Box::new(VectorEngine::open(&path, DIMS).unwrap()))
        .with_embedder(Box::new(SharedBag(bag.clone())))
        .with_fts_index(Arc::new(axil_fts::FtsEngine::open(&path).unwrap()))
        .build()
        .unwrap();

    let cfg = RecallConfig {
        qtc: Some(QtcConfig {
            top_k: 3,
            ..QtcConfig::default()
        }),
        ..cli_config()
    };
    let results = db.recall("connection pool", 10, Some(cfg)).unwrap();
    assert!(!results.is_empty(), "FTS still finds the records");
    // Each record is one short chunk, so three texts means three records.
    assert_eq!(
        bag.batched_texts.load(Ordering::Relaxed),
        3,
        "records with no stored chunk vectors are embedded only inside the window"
    );
}
