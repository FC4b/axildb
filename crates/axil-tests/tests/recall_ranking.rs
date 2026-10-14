//! Recall's ranking must not depend on how many results were asked for, and
//! recency must not decide which candidates get the query-time-chunk blend.
//!
//! - `recall(q, k)` is a prefix of `recall(q, k')` for any `k < k'` up to the
//!   candidate pool. A pool that grew with `k` let graph proximity (links to
//!   the rest of the pool) lift well-connected records over the answer.
//! - The chunk blend goes to the most relevant candidates, ranked without
//!   recency. When the window followed the plain fused order, a relevant old
//!   record that recency pushed below it was never rescored.
//! - Code-index rows, which code-search embeds into the same vector index,
//!   take no place among recall's vector candidates.
//! - Equal scores come back in record-id order, the same in every process,
//!   from recall and from full-text search.
//! - Graph proximity counts only for a query that names an entity, and only
//!   for the candidates linked to it.
//! - A long record gets a vector per dialog turn or fact, and a database
//!   written with older pieces can be re-split.
//! - A query that names a span of time keeps the candidates inside it, and
//!   ranks on relevance alone when none are. A record's `event_time`, when
//!   it has one, says where in time it falls.

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

#[test]
fn code_index_vectors_take_no_place_among_memory_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("code.axil");
    let db = open(&path, Arc::default(), false);

    let memory = db
        .insert(
            "decisions",
            json!({"summary": "raise the connection pool timeout for slow hosts"}),
        )
        .unwrap();
    // More code proxies than recall's vector candidates, every one a closer
    // match than the memory, embedded the way the indexer embeds them.
    for i in 0..200 {
        let text = format!("connection pool timeout f{i}");
        let proxy = db
            .insert(
                "_idx_code_proxies",
                json!({"proxy_text": text, "path": format!("src/f{i}.rs")}),
            )
            .unwrap();
        db.embed_text(&proxy.id, &text).unwrap();
    }

    let results = db
        .recall("connection pool timeout", 5, Some(cli_config()))
        .unwrap();
    let tables: Vec<&str> = results.iter().map(|r| r.record.table.as_str()).collect();
    assert!(
        !tables.iter().any(|t| t.starts_with("_idx_")),
        "code-index rows are not memory recall results: {tables:?}"
    );
    assert_eq!(
        results.first().map(|r| &r.record.id),
        Some(&memory.id),
        "the memory is found past the code rows"
    );
    // Code-search reads the same vectors through `similar_to`, unchanged.
    let similar = db.similar_to("connection pool timeout", 5).unwrap();
    assert!(similar.iter().all(|(r, _)| r.table == "_idx_code_proxies"));
}

#[test]
fn equal_scores_come_back_in_record_id_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ties.axil");
    let db = Axil::open(&path)
        .with_vector_index(Box::new(VectorEngine::open(&path, DIMS).unwrap()))
        .with_embedder(Box::new(SharedBag(Arc::default())))
        .with_fts_index(Arc::new(axil_fts::FtsEngine::open(&path).unwrap()))
        .build()
        .unwrap();

    // The same text at the same time, each in a table of its own so none
    // supersedes another: every signal scores them alike.
    let at = Utc::now() - Duration::days(1);
    let mut expected: Vec<String> = (0..12)
        .map(|i| {
            db.insert_at(
                &format!("notes{i}"),
                json!({"summary": "connection pool timeout"}),
                at,
            )
            .unwrap()
            .id
            .to_string()
        })
        .collect();
    // Re-index half of them, so the full-text index holds them after the
    // others and its own tie order is no longer the id order.
    for id in &expected[..6] {
        let id = axil_core::RecordId::from_string(id).unwrap();
        let record = db.get(&id).unwrap().unwrap();
        db.update(&id, record.data).unwrap();
    }
    expected.sort();

    // Recall gathers candidates in hash maps, whose order changes with every
    // map, so twenty runs would show any tie left to that order.
    let query = "connection pool timeout";
    for _ in 0..20 {
        assert_eq!(ids(&db, query, 12, cli_config()), expected, "recall");
        let fts: Vec<String> = db
            .search_text(query, 12)
            .unwrap()
            .into_iter()
            .map(|(r, _)| r.id.to_string())
            .collect();
        assert_eq!(fts, expected, "full-text search");
    }
    drop(db);

    // Without vectors, recall's candidates come in the full-text index's
    // order alone.
    let db = Axil::open(&path)
        .with_fts_index(Arc::new(axil_fts::FtsEngine::open(&path).unwrap()))
        .build()
        .unwrap();
    for _ in 0..20 {
        assert_eq!(
            ids(&db, query, 12, cli_config()),
            expected,
            "recall over full-text search only"
        );
    }
}

#[test]
fn a_named_span_keeps_the_candidates_inside_it() {
    use chrono::TimeZone;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("span.axil");
    let db = open(&path, Arc::default(), false);
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    let march = db
        .insert_at(
            "decisions",
            json!({"summary": "raise the connection pool timeout"}),
            Utc.with_ymd_and_hms(2026, 3, 12, 9, 0, 0).unwrap(),
        )
        .unwrap();
    let april = db
        .insert_at(
            "commits",
            json!({"summary": "connection pool timeout"}),
            now - Duration::days(1),
        )
        .unwrap();
    let cfg = RecallConfig {
        now,
        ..cli_config()
    };

    let unscoped = ids(&db, "connection pool timeout", 5, cfg.clone());
    assert_eq!(unscoped.first(), Some(&april.id.to_string()));
    assert_eq!(
        ids(&db, "connection pool timeout in March", 5, cfg.clone()),
        vec![march.id.to_string()]
    );
    // Nothing falls in January: the span is set aside.
    assert_eq!(
        ids(&db, "connection pool timeout in January", 5, cfg).len(),
        2
    );
}

#[test]
fn a_record_falls_in_time_where_its_event_time_says() {
    use chrono::TimeZone;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("event.axil");
    let db = open(&path, Arc::default(), false);
    let now = Utc.with_ymd_and_hms(2026, 4, 10, 12, 0, 0).unwrap();
    // Written yesterday, about a release in March.
    let release = db
        .insert_at(
            "context",
            json!({"summary": "the connection pool release", "event_time": "2026-03-12"}),
            now - Duration::days(1),
        )
        .unwrap();
    // Written in March about something that happened then too.
    let march = db
        .insert_at(
            "decisions",
            json!({"summary": "raise the connection pool timeout"}),
            Utc.with_ymd_and_hms(2026, 3, 20, 9, 0, 0).unwrap(),
        )
        .unwrap();
    // Written in April, no event_time.
    db.insert_at(
        "commits",
        json!({"summary": "connection pool release"}),
        now - Duration::days(2),
    )
    .unwrap();
    let cfg = RecallConfig {
        now,
        ..cli_config()
    };

    let mut in_march = ids(&db, "connection pool release in March", 5, cfg);
    in_march.sort();
    let mut expected = vec![release.id.to_string(), march.id.to_string()];
    expected.sort();
    assert_eq!(in_march, expected);
}

#[test]
fn graph_proximity_needs_an_entity_the_query_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.axil");
    let db = open(&path, Arc::default(), true);
    let now = Utc::now();
    let mut named = Vec::new();
    for i in 0..3 {
        let summary = format!("AuthService connection pool timeout {i}");
        let rec = db
            .insert_at("notes", json!({ "summary": summary }), now - Duration::hours(i))
            .unwrap();
        named.push(rec.id);
    }
    let mut all = named.clone();
    for i in 0..5 {
        let summary = format!("connection pool timeout plain{i}");
        let rec = db
            .insert_at("notes", json!({ "summary": summary }), now - Duration::hours(i))
            .unwrap();
        all.push(rec.id);
    }
    // Every record linked to every other: there are links to count.
    for a in &all {
        for b in &all {
            if a < b {
                db.relate(a, "relates_to", b, None).unwrap();
            }
        }
    }

    let lifted = |query: &str| -> Vec<String> {
        let mut ids: Vec<String> = db
            .recall(query, 8, Some(cli_config()))
            .unwrap()
            .into_iter()
            .filter(|r| {
                r.explanation
                    .signals
                    .iter()
                    .any(|(name, value)| name == "graph_proximity" && *value > 0.0)
            })
            .map(|r| r.record.id.to_string())
            .collect();
        ids.sort();
        ids
    };
    // No entity named: no graph signal, however linked the records are.
    assert_eq!(lifted("connection pool timeout"), Vec::<String>::new());
    // Naming one: the records linked to it, and only those.
    let mut expected: Vec<String> = named.iter().map(ToString::to_string).collect();
    expected.sort();
    assert_eq!(lifted("AuthService connection pool timeout"), expected);
}

#[test]
fn a_long_record_gets_a_vector_per_turn() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("turns.axil");
    let db = open(&path, Arc::default(), false);
    let mut turns: Vec<String> = (0..6)
        .map(|i| format!("user: tell me about topic{i} {}", "and more words ".repeat(30)))
        .collect();
    turns.insert(3, format!("assistant: {}", "the connection pool timeout is thirty seconds ".repeat(12)));
    let text = turns.join("\n");
    let session = db.insert("sessions", json!({ "full_text": text })).unwrap();

    let unit_count = |db: &Axil| -> usize {
        db.get(&session.id)
            .unwrap()
            .unwrap()
            .metadata
            .and_then(|m| m.get("_recall_chunk_ids").cloned())
            .and_then(|ids| ids.as_array().map(Vec::len))
            .unwrap_or(0)
    };
    let units = axil_core::util::recall_units(&text);
    assert_eq!(units.len(), turns.len(), "one unit per turn: {units:#?}");
    assert_eq!(unit_count(&db), units.len());

    // Re-splitting replaces the units rather than adding to them.
    assert_eq!(db.rebuild_recall_units().unwrap(), 1);
    assert_eq!(unit_count(&db), units.len());
    let hits = db
        .recall("connection pool timeout", 1, Some(cli_config()))
        .unwrap();
    assert_eq!(hits[0].record.id, session.id);
}
