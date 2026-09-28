//! LongMemEval benchmark runner for Axil.
//!
//! Measures retrieval accuracy on the LongMemEval dataset:
//! 1. For each question, ingests haystack sessions into a fresh Axil DB
//! 2. Queries using the question text via vector search / FTS / ask / recall
//! 3. Compares retrieved records against ground-truth answer sessions
//! 4. Reports recall@k, precision@k, hit rate per category, and per-question misses
//!
//! Usage:
//!   cargo run -p longmemeval-bench -- [OPTIONS]
//!
//! Options:
//!   --variant oracle|s|m    Dataset variant (default: oracle)
//!   --limit N               Max questions to evaluate (default: all)
//!   --strategy vector|fts|ask|recall  Retrieval strategy (default: vector)
//!   --rerank off|cross-encoder        Optional reranking for recall results
//!   --top-k N               Results per query (default: 5)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Shared multi-thread Tokio runtime for `--strategy ask` (uses
/// `ask_parallel`). Lazily initialised on first use; rayon workers all
/// share the same runtime so we don't pay builder cost per question.
fn ask_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime")
    })
}

use clap::{Parser, ValueEnum};
use chrono::{DateTime, NaiveDateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::plugin::TextEmbedder;
use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use axil_vector::embed::Embedder;
use axil_vector::models::EmbeddingModel;
use axil_vector::AxilBuilderVectorExt;

// ── Shared embedder wrapper ─────────────────────────────────────────

/// `TextEmbedder` wrapper around a shared `Arc<Embedder>`.
///
/// Loading the ONNX session is expensive (~2-5 s). Sharing one Embedder
/// across all questions amortises that cost. The inner `Embedder` has an
/// internal `Mutex<Session>` so concurrent `embed()` calls serialise at
/// the model boundary, but per-question DB setup, FTS indexing, and HNSW
/// work happen in parallel.
struct SharedEmbedder(Arc<Embedder>);

impl TextEmbedder for SharedEmbedder {
    fn embed(&self, text: &str) -> axil_core::Result<Vec<f32>> {
        self.0
            .embed(text)
            .map_err(axil_core::error::AxilError::plugin)
    }

    fn embed_batch(&self, texts: &[&str]) -> axil_core::Result<Vec<Vec<f32>>> {
        self.0
            .embed_batch_impl(texts)
            .map_err(axil_core::error::AxilError::plugin)
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum BenchRerankMode {
    #[default]
    Off,
    CrossEncoder,
}

// ── CLI ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "longmemeval-bench", about = "LongMemEval benchmark for Axil")]
struct Args {
    /// Dataset variant: oracle, s, or m.
    #[arg(long, default_value = "oracle")]
    variant: String,

    /// Max questions to evaluate (0 = all).
    #[arg(long, default_value = "0")]
    limit: usize,

    /// Retrieval strategy: vector, fts, ask, recall.
    #[arg(long, default_value = "vector")]
    strategy: String,

    /// Number of results per query.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Optional reranking mode for recall candidates.
    #[arg(long, value_enum, default_value_t = BenchRerankMode::Off)]
    rerank: BenchRerankMode,

    /// Cross-encoder model path for `--rerank cross-encoder`.
    #[arg(long, default_value = "models/cross-encoder.onnx")]
    rerank_model: String,

    /// Only rerank the top K retrieved candidates.
    #[arg(long, default_value = "20")]
    rerank_top_k: usize,

    /// Path to dataset directory.
    #[arg(long, default_value = "benchmarks/longmemeval/data")]
    data_dir: PathBuf,

    /// Print per-question details.
    #[arg(long)]
    verbose: bool,

    /// Worker threads for parallel question evaluation. 0 = rayon default (num_cpus).
    #[arg(long, default_value = "0")]
    jobs: usize,

    /// Embedding model: bge-small (default), bge-base, or nomic.
    #[arg(long, default_value = "bge-small")]
    model: String,

    /// Algorithmic query expansion: pull aliases + graph neighbors for entities in the query.
    #[arg(long)]
    expand: bool,

    /// Write each question's retrieved context (compact and full) plus the
    /// gold answer as JSONL, for end-to-end QA with a reader model
    /// (`benchmarks/e2e-qa/qa.py`).
    #[arg(long)]
    dump_context: Option<PathBuf>,

    /// Override recall score weights for the `recall` and `recall-qtc`
    /// strategies, e.g. `vector=0.7,recency=0.3`; unnamed weights keep their
    /// defaults. Used to measure the weights the CLI builds.
    #[arg(long)]
    weights: Option<String>,

    /// Number of graph neighbors to expand per entity.
    #[arg(long, default_value = "3")]
    expand_neighbors: usize,
}

/// Replicated from axil-cli/src/main.rs::expand_query — same algorithm, but
/// kept here so the bench doesn't pull in the CLI crate. Returns the original
/// query when no entities resolve or no expansion terms are found.
fn expand_query(db: &axil_core::Axil, query: &str, neighbors: usize) -> String {
    use axil_core::Direction;
    use std::collections::{BTreeSet, HashMap};

    let entities = axil_core::entity::extract_entities(query);
    if entities.is_empty() {
        return query.to_string();
    }

    let mem = axil_memory::AgentMemory::new(db);
    let semantic = mem.semantic();
    let by_name: HashMap<String, axil_core::Record> = db
        .list("_entities")
        .unwrap_or_default()
        .into_iter()
        .filter_map(|r| {
            let n = r.data.get("name").and_then(|v| v.as_str())?.to_ascii_lowercase();
            Some((n, r))
        })
        .collect();

    let mut extras: BTreeSet<String> = BTreeSet::new();
    for entity in &entities {
        let canonical = semantic.resolve(&entity.name).ok().flatten()
            .unwrap_or_else(|| entity.name.clone());
        if canonical != entity.name {
            extras.insert(canonical.clone());
        }
        if let Ok(aliases) = semantic.aliases(&canonical) {
            for a in aliases {
                if a != entity.name && a.len() >= 2 {
                    extras.insert(a);
                }
            }
        }
        if let Some(seed) = by_name.get(&canonical.to_ascii_lowercase()) {
            if let Ok(hop) = db.neighbors(&seed.id, None, Direction::Out) {
                for n in hop.into_iter().take(neighbors.saturating_mul(3)) {
                    if let Some(name) = n.data.get("name").and_then(|v| v.as_str()) {
                        if name.len() >= 2 && !name.eq_ignore_ascii_case(&entity.name) {
                            extras.insert(name.to_string());
                        }
                    }
                }
            }
        }
    }

    if extras.is_empty() {
        return query.to_string();
    }
    format!("{query} {}", extras.into_iter().collect::<Vec<_>>().join(" "))
}

// ── Dataset types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Question {
    question_id: String,
    question_type: String,
    question: String,
    answer: Value,
    #[serde(default)]
    question_date: String,
    #[serde(default)]
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: bool,
}

// ── Results ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct CategoryStats {
    total: usize,
    hits: usize,
    recall_sum: f64,
    precision_sum: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    benchmark: String,
    variant: String,
    strategy: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<String>,
    rerank: String,
    top_k: usize,
    total_questions: usize,
    overall: OverallStats,
    by_category: HashMap<String, CategoryReport>,
    misses: Vec<MissReport>,
}

#[derive(Debug, Serialize)]
struct OverallStats {
    hit_rate: f64,
    avg_recall: f64,
    avg_precision: f64,
    #[serde(flatten)]
    harder: HarderReport,
}

#[derive(Debug, Serialize)]
struct CategoryReport {
    total: usize,
    hits: usize,
    hit_rate: f64,
    avg_recall: f64,
    avg_precision: f64,
    #[serde(flatten)]
    harder: HarderReport,
}

/// Characters of a hit's summary the CLI's default (compact) recall format
/// shows the agent (`format_recall_results` truncates to 200).
const COMPACT_SUMMARY_CHARS: usize = 200;

/// Rank depth for NDCG.
const NDCG_DEPTH: usize = 10;

/// Per-question measures beyond "was any answer session retrieved".
#[derive(Debug, Default, Clone, Copy)]
struct Harder {
    /// 1.0 when every answer session is in the top-k.
    recall_all: f64,
    /// NDCG@10 over the session ranking, binary relevance.
    ndcg: f64,
    /// Share of the answer turns whose text the agent would actually read
    /// in the compact format (the first 200 chars of each hit)...
    turn_recall_compact: f64,
    /// ...and in the full record text.
    turn_recall_full: f64,
    /// Tokens (bytes / 4) of the top-k hits in each format.
    tokens_compact: f64,
    tokens_full: f64,
}

#[derive(Debug, Default)]
struct HarderSums {
    n: usize,
    sum: Harder,
}

impl HarderSums {
    fn add(&mut self, h: Harder) {
        self.n += 1;
        self.sum.recall_all += h.recall_all;
        self.sum.ndcg += h.ndcg;
        self.sum.turn_recall_compact += h.turn_recall_compact;
        self.sum.turn_recall_full += h.turn_recall_full;
        self.sum.tokens_compact += h.tokens_compact;
        self.sum.tokens_full += h.tokens_full;
    }

    fn report(&self) -> HarderReport {
        let n = self.n.max(1) as f64;
        let mean = |x: f64| x / n;
        // Evidence found per 1k tokens shown: a ratio of the means, so a few
        // tiny answers can't dominate it.
        let per_1k = |recall: f64, tokens: f64| {
            if tokens > 0.0 {
                recall / tokens * 1000.0
            } else {
                0.0
            }
        };
        HarderReport {
            recall_all: mean(self.sum.recall_all),
            ndcg_at_10: mean(self.sum.ndcg),
            turn_recall_compact: mean(self.sum.turn_recall_compact),
            turn_recall_full: mean(self.sum.turn_recall_full),
            tokens_compact: mean(self.sum.tokens_compact),
            tokens_full: mean(self.sum.tokens_full),
            evidence_per_1k_tokens_compact: per_1k(
                self.sum.turn_recall_compact,
                self.sum.tokens_compact,
            ),
            evidence_per_1k_tokens_full: per_1k(self.sum.turn_recall_full, self.sum.tokens_full),
        }
    }
}

#[derive(Debug, Serialize)]
struct HarderReport {
    recall_all: f64,
    ndcg_at_10: f64,
    turn_recall_compact: f64,
    turn_recall_full: f64,
    tokens_compact: f64,
    tokens_full: f64,
    evidence_per_1k_tokens_compact: f64,
    evidence_per_1k_tokens_full: f64,
}

#[derive(Debug, Serialize)]
struct MissReport {
    question_id: String,
    question_type: String,
    question_date: String,
    recall: f64,
    precision: f64,
    retrieved_sessions: Vec<String>,
    answer_sessions: Vec<String>,
}

#[derive(Debug)]
struct QuestionResult {
    category: String,
    recall: f64,
    precision: f64,
    harder: Harder,
    miss: Option<MissReport>,
    context: Option<QaContext>,
}

/// What a reader model gets for one question: the retrieved top-k sessions
/// as the CLI's compact output shows them and in full.
#[derive(Debug, Serialize)]
struct QaContext {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    answer: String,
    compact: String,
    full: String,
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let data_file = args.data_dir.join(match args.variant.as_str() {
        "s" => "longmemeval_s_cleaned.json",
        "m" => "longmemeval_m_cleaned.json",
        _ => "longmemeval_oracle.json",
    });

    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║  LongMemEval Benchmark for Axil                 ║");
    eprintln!("╠══════════════════════════════════════════════════╣");
    eprintln!("║  Variant:  {:<38}║", args.variant);
    eprintln!("║  Strategy: {:<38}║", args.strategy);
    eprintln!("║  Rerank:   {:<38}║", format!("{:?}", args.rerank).to_lowercase());
    eprintln!("║  Top-K:    {:<38}║", args.top_k);
    eprintln!("║  Data:     {:<38}║", data_file.display());
    eprintln!("╚══════════════════════════════════════════════════╝");
    eprintln!();

    if matches!(args.rerank, BenchRerankMode::CrossEncoder) && !cross_encoder_compiled() {
        eprintln!("warning: cross-encoder rerank requested, but this binary was built without the `rerank` feature; base recall order will be used");
    }

    // Load dataset. Read the whole file into memory first, then parse from
    // the slice — `serde_json::from_reader` does unbuffered byte-at-a-time
    // reads, which is markedly slower than `from_slice` on the 265 MB file.
    eprintln!("[1/3] Loading dataset...");
    let bytes = std::fs::read(&data_file).expect("cannot read dataset file");
    let questions: Vec<Question> = serde_json::from_slice(&bytes).expect("invalid JSON");
    let total = if args.limit > 0 { questions.len().min(args.limit) } else { questions.len() };
    eprintln!("       {} questions loaded, evaluating {}", questions.len(), total);

    // Preload the embedder once — this is the heaviest per-question cost
    // when each question opens a fresh Axil DB.
    eprintln!("[2/3] Loading embedder ({})...", args.model);
    let embedding_model = match args.model.as_str() {
        "bge-base" => EmbeddingModel::BgeBase,
        "nomic" => EmbeddingModel::Nomic,
        _ => EmbeddingModel::BgeSmall,
    };
    let embedder = match Embedder::new(embedding_model.clone()) {
        Ok(e) => Arc::new(e),
        Err(err) => {
            eprintln!("failed to load embedder: {err}");
            std::process::exit(1);
        }
    };

    // Configure rayon thread pool.
    let jobs = if args.jobs == 0 {
        num_cpus::get().min(8)
    } else {
        args.jobs
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("rayon pool");
    eprintln!("       {} worker thread(s)", jobs);

    // Evaluate in parallel.
    eprintln!("[3/4] Evaluating {} questions...", total);
    let done = AtomicUsize::new(0);
    let results: Vec<QuestionResult> = pool.install(|| {
        questions
            .par_iter()
            .take(total)
            .map(|q| {
                let result = evaluate_question(q, &args, &embedder, &embedding_model);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 10 == 0 || n == 1 {
                    eprint!("\r       {}/{}", n, total);
                }
                if args.verbose {
                    eprintln!(
                        "\n  {} [{}] recall={:.2} precision={:.2} {}",
                        q.question_id,
                        q.question_type,
                        result.recall,
                        result.precision,
                        if result.recall > 0.0 { "HIT" } else { "MISS" }
                    );
                }
                result
            })
            .collect()
    });
    eprintln!("\r       {}/{} done.          ", total, total);

    // Aggregate.
    let mut category_stats: HashMap<String, CategoryStats> = HashMap::new();
    let mut category_harder: HashMap<String, HarderSums> = HashMap::new();
    let mut contexts: Vec<QaContext> = Vec::new();
    let mut overall_harder = HarderSums::default();
    let mut total_recall = 0.0f64;
    let mut total_precision = 0.0f64;
    let mut total_hits = 0usize;
    let mut misses = Vec::new();
    for result in results {
        let QuestionResult {
            category,
            recall,
            precision,
            harder,
            miss,
            context,
        } = result;
        if let Some(context) = context {
            contexts.push(context);
        }
        overall_harder.add(harder);
        category_harder
            .entry(category.clone())
            .or_default()
            .add(harder);
        let hit = recall > 0.0;
        total_recall += recall;
        total_precision += precision;
        if hit {
            total_hits += 1;
        }
        let stats = category_stats.entry(category).or_default();
        stats.total += 1;
        if hit {
            stats.hits += 1;
        }
        stats.recall_sum += recall;
        stats.precision_sum += precision;
        if let Some(miss) = miss {
            misses.push(miss);
        }
    }

    // Build report
    eprintln!("[4/4] Results:");
    eprintln!();

    let report = BenchmarkReport {
        benchmark: "LongMemEval".to_string(),
        variant: args.variant.clone(),
        strategy: args.strategy.clone(),
        weights: args.weights.clone(),
        rerank: format!("{:?}", args.rerank).to_lowercase(),
        top_k: args.top_k,
        total_questions: total,
        overall: OverallStats {
            hit_rate: total_hits as f64 / total as f64,
            avg_recall: total_recall / total as f64,
            avg_precision: total_precision / total as f64,
            harder: overall_harder.report(),
        },
        by_category: category_stats
            .into_iter()
            .map(|(cat, stats)| {
                let harder = category_harder
                    .get(&cat)
                    .map(HarderSums::report)
                    .unwrap_or_else(|| HarderSums::default().report());
                (cat, CategoryReport {
                    total: stats.total,
                    hits: stats.hits,
                    hit_rate: stats.hits as f64 / stats.total as f64,
                    avg_recall: stats.recall_sum / stats.total as f64,
                    avg_precision: stats.precision_sum / stats.total as f64,
                    harder,
                })
            })
            .collect(),
        misses,
    };

    if let Some(path) = &args.dump_context {
        contexts.sort_by(|a, b| a.question_id.cmp(&b.question_id));
        let lines: Vec<String> = contexts
            .iter()
            .map(|c| serde_json::to_string(c).unwrap())
            .collect();
        std::fs::write(path, lines.join("\n") + "\n").expect("write --dump-context file");
        eprintln!(
            "       wrote {} contexts to {}",
            contexts.len(),
            path.display()
        );
    }

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

// ── Per-question evaluation ────────────────────────────────────────

fn evaluate_question(q: &Question, args: &Args, embedder: &Arc<Embedder>, model: &EmbeddingModel) -> QuestionResult {
    use std::time::Instant;
    let t0 = Instant::now();

    // Create a temp DB
    let tmp = tempfile::tempdir().expect("tmpdir");
    let db_path = tmp.path().join("bench.axil");

    let dims = model.dimensions();
    let builder = match Axil::open(&db_path).with_vector(dims) {
        Ok(b) => b,
        Err(_) => return question_failure(q, Vec::new(), Vec::new()),
    };
    let builder = builder.with_embedder(Box::new(SharedEmbedder(embedder.clone())));
    // FTS is only consulted by the `fts` and `ask` strategies. For
    // `vector`/`recall` the index is never read — building it just burns
    // insert time (Tantivy add + commit per record) on dead weight.
    let needs_fts = matches!(args.strategy.as_str(), "fts" | "ask");
    let builder = if needs_fts {
        match builder.with_fts_engine() {
            Ok(b) => b,
            Err(_) => return question_failure(q, Vec::new(), Vec::new()),
        }
    } else {
        builder
    };
    let db = match builder.build() {
        Ok(db) => db,
        Err(_) => return question_failure(q, Vec::new(), Vec::new()),
    };
    let t_db = t0.elapsed();

    let mut record_to_session: HashMap<axil_core::RecordId, String> = HashMap::new();
    let mut items: Vec<(Value, chrono::DateTime<chrono::Utc>)> = Vec::new();
    let mut session_ids: Vec<String> = Vec::new();
    let mut session_texts: Vec<String> = Vec::new();
    let now = chrono::Utc::now();

    for (si, session) in q.haystack_sessions.iter().enumerate() {
        let session_id = format!("session_{}", si);
        let date = q.haystack_dates.get(si).map(|s| s.as_str()).unwrap_or("");
        let text = session_text(session);
        session_texts.push(text.clone());
        let summary_end = text.floor_char_boundary(text.len().min(500));
        let data = json!({
            "session_id": session_id,
            "summary": &text[..summary_end],
            "full_text": text,
            "date": date,
        });
        let ts = parse_bench_datetime(date).unwrap_or(now);
        items.push((data, ts));
        session_ids.push(session_id);
    }
    let n_sessions = items.len();
    let t_prep = t0.elapsed();

    // `insert_batch_at` (not `insert_batch_raw`) so the index-time chunking
    // runs during ingestion. `_recall_chunks` + chunk vectors are what the
    // QTC fast path consults at query time.
    let records = db.insert_batch_at("sessions", items).unwrap_or_default();
    let t_insert = t0.elapsed();

    // No explicit FTS indexing here. When the strategy needs FTS, the
    // plugin's batch indexer (run inside `insert_batch_at`) already
    // auto-indexed every string field, including `full_text`. The old
    // `index_text` call re-indexed identical content and paid a second
    // Tantivy commit per record for nothing.
    for (i, record) in records.iter().enumerate() {
        if let Some(sid) = session_ids.get(i) {
            record_to_session.insert(record.id.clone(), sid.clone());
        }
    }
    let t_fts = t0.elapsed();

    eprintln!("    [timing] db={:.1}s prep={:.1}s insert={:.1}s fts={:.1}s sessions={}",
        t_db.as_secs_f64(),
        (t_prep - t_db).as_secs_f64(),
        (t_insert - t_prep).as_secs_f64(),
        (t_fts - t_insert).as_secs_f64(),
        n_sessions);

    let fetch_k = args.top_k.saturating_mul(8).max(40);

    // Optional algorithmic query expansion: append entity aliases + graph
    // neighbors to the question text. Only runs when --expand is set.
    let question_text: String = if args.expand {
        expand_query(&db, &q.question, args.expand_neighbors)
    } else {
        q.question.clone()
    };
    let q_text = question_text.as_str();

    // Query chunk records, then collapse to top-k unique sessions.
    let retrieved_hits: Vec<(axil_core::RecordId, f32)> = match args.strategy.as_str() {
        "fts" => {
            db.search_text(q_text, fetch_k)
                .unwrap_or_default()
                .into_iter()
                .map(|(r, score)| (r.id, score))
                .collect()
        }
        "ask" => {
            let db_arc = Arc::new(db);
            let result = ask_runtime().block_on(axil_indexer::ask::ask_parallel(
                Arc::clone(&db_arc), q_text, fetch_k, None,
            ));
            match result {
                Ok(r) => r.results.iter()
                    .filter_map(|v| {
                        let id = v.get("id").and_then(|id| id.as_str())?;
                        let score = v.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                        Some((id, score))
                    })
                    .filter_map(|(id, score)| axil_core::RecordId::from_string(id).ok().map(|rid| (rid, score)))
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
        "recall" => {
            let mut cfg = axil_core::RecallConfig::default();
            apply_weights(&mut cfg, args);
            if let Some(now) = parse_question_now(q) {
                cfg.now = now;
            }
            let mut results = db.recall(q_text, fetch_k, Some(cfg)).unwrap_or_default();
            maybe_rerank_recall_results(q_text, &mut results, args);
            results
                .into_iter()
                .map(|r| (r.record.id, r.score))
                .collect()
        }
        "recall-qtc" => {
            let mut cfg = axil_core::RecallConfig::default();
            apply_weights(&mut cfg, args);
            if let Some(now) = parse_question_now(q) {
                cfg.now = now;
            }
            cfg.qtc = Some(axil_core::scoring::QtcConfig::default());
            let mut results = db.recall(q_text, fetch_k, Some(cfg)).unwrap_or_default();
            maybe_rerank_recall_results(q_text, &mut results, args);
            results
                .into_iter()
                .map(|r| (r.record.id, r.score))
                .collect()
        }
        "oracle-answer" => {
            let oracle_query = oracle_answer_text(q);
            let q_str = if oracle_query.trim().is_empty() { question_text.clone() } else { oracle_query };
            db.similar_to(&q_str, fetch_k)
                .unwrap_or_default()
                .into_iter()
                .map(|(r, score)| (r.id, score))
                .collect()
        }
        _ => {
            db.similar_to(q_text, fetch_k)
                .unwrap_or_default()
                .into_iter()
                .map(|(r, score)| (r.id, score))
                .collect()
        }
    };

    let t_query = t0.elapsed();
    eprintln!("    [timing] query={:.1}s total={:.1}s",
        (t_query - t_fts).as_secs_f64(),
        t_query.as_secs_f64());

    let ranked = collapse_to_sessions(
        retrieved_hits,
        &record_to_session,
        args.top_k.max(NDCG_DEPTH),
    );
    let retrieved_sessions: Vec<String> = ranked.iter().take(args.top_k).cloned().collect();

    // The answer_session_ids in LongMemEval use a different naming scheme.
    // We need to check if any retrieved session contains answer content.
    // Since oracle variant has only answer sessions, session indices map directly.
    // For a fair comparison: check if the sessions that CONTAIN answer turns were retrieved.
    let answer_session_indices: Vec<usize> = q.haystack_sessions
        .iter()
        .enumerate()
        .filter(|(_, session)| session.iter().any(|t| t.has_answer))
        .map(|(i, _)| i)
        .collect();

    let answer_session_tags: Vec<String> = answer_session_indices
        .iter()
        .map(|i| format!("session_{}", i))
        .collect();

    let harder = harder_metrics(
        q,
        &session_texts,
        &ranked,
        &retrieved_sessions,
        &answer_session_tags,
    );

    let context = args
        .dump_context
        .is_some()
        .then(|| qa_context(q, &session_texts, &retrieved_sessions));

    // Compute recall and precision
    if answer_session_tags.is_empty() {
        return QuestionResult {
            category: q.question_type.clone(),
            recall: 1.0,
            precision: 0.0,
            harder,
            miss: None,
            context,
        };
    }

    let hits = answer_session_tags
        .iter()
        .filter(|a| retrieved_sessions.iter().any(|r| r == *a))
        .count();

    let recall = hits as f64 / answer_session_tags.len() as f64;

    let precision = if retrieved_sessions.is_empty() {
        0.0
    } else {
        let p_hits = retrieved_sessions
            .iter()
            .filter(|r| answer_session_tags.iter().any(|a| a == *r))
            .count();
        p_hits as f64 / retrieved_sessions.len() as f64
    };

    let miss = if recall > 0.0 {
        None
    } else {
        Some(MissReport {
            question_id: q.question_id.clone(),
            question_type: q.question_type.clone(),
            question_date: q.question_date.clone(),
            recall,
            precision,
            retrieved_sessions: retrieved_sessions.clone(),
            answer_sessions: answer_session_tags.clone(),
        })
    };

    QuestionResult {
        category: q.question_type.clone(),
        recall,
        precision,
        harder,
        miss,
        context,
    }
}

/// Apply `--weights name=value,...` on top of the default weights.
fn apply_weights(cfg: &mut axil_core::RecallConfig, args: &Args) {
    let Some(spec) = args.weights.as_deref() else {
        return;
    };
    let w = &mut cfg.weights;
    for pair in spec.split(',').filter(|p| !p.trim().is_empty()) {
        let (name, value) = pair.split_once('=').expect("--weights takes name=value pairs");
        let value: f32 = value.trim().parse().expect("--weights value must be a number");
        let slot = match name.trim() {
            "vector" => &mut w.vector,
            "recency" => &mut w.recency,
            "graph" => &mut w.graph,
            "keyword" => &mut w.keyword,
            "feedback" => &mut w.feedback,
            "temporal" => &mut w.temporal,
            "preference" => &mut w.preference,
            other => panic!("unknown weight {other:?}"),
        };
        *slot = value;
    }
}

/// The retrieved sessions in rank order, headed by their date (temporal
/// questions need it), cut to the compact form and in full.
fn qa_context(q: &Question, session_texts: &[String], top_k: &[String]) -> QaContext {
    let mut compact = Vec::new();
    let mut full = Vec::new();
    for tag in top_k {
        let Some(i) = tag
            .strip_prefix("session_")
            .and_then(|i| i.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(text) = session_texts.get(i) else {
            continue;
        };
        let date = q
            .haystack_dates
            .get(i)
            .map(String::as_str)
            .unwrap_or("unknown date");
        let end = text.floor_char_boundary(text.len().min(COMPACT_SUMMARY_CHARS));
        compact.push(format!("[Session {i}, {date}] {}", &text[..end]));
        full.push(format!("[Session {i}, {date}]\n{text}"));
    }
    QaContext {
        question_id: q.question_id.clone(),
        question_type: q.question_type.clone(),
        question: q.question.clone(),
        question_date: q.question_date.clone(),
        answer: match &q.answer {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
        compact: compact.join("\n"),
        full: full.join("\n\n"),
    }
}

/// The measures session hit/recall can't see. Questions with no answer
/// session (abstention) count as fully found, like `recall` does.
fn harder_metrics(
    q: &Question,
    session_texts: &[String],
    ranked: &[String],
    top_k: &[String],
    answers: &[String],
) -> Harder {
    let index = |tag: &str| {
        tag.strip_prefix("session_")
            .and_then(|i| i.parse::<usize>().ok())
    };
    let shown = |text: &str| -> String {
        let end = text.floor_char_boundary(text.len().min(COMPACT_SUMMARY_CHARS));
        text[..end].to_string()
    };
    let tokens = |f: &dyn Fn(&str) -> String| -> f64 {
        top_k
            .iter()
            .filter_map(|t| index(t).and_then(|i| session_texts.get(i)))
            .map(|text| f(text).len() as f64 / 4.0)
            .sum()
    };
    let tokens_compact = tokens(&|t: &str| shown(t));
    let tokens_full = tokens(&|t: &str| t.to_string());

    if answers.is_empty() {
        return Harder {
            recall_all: 1.0,
            ndcg: 1.0,
            turn_recall_compact: 1.0,
            turn_recall_full: 1.0,
            tokens_compact,
            tokens_full,
        };
    }

    let found = answers.iter().filter(|a| top_k.contains(a)).count();
    let dcg: f64 = ranked
        .iter()
        .take(NDCG_DEPTH)
        .enumerate()
        .filter(|(_, tag)| answers.contains(tag))
        .map(|(rank, _)| 1.0 / (rank as f64 + 2.0).log2())
        .sum();
    let ideal: f64 = (0..answers.len().min(NDCG_DEPTH))
        .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
        .sum();

    // Answer turns, and whether each is inside what the agent reads.
    let mut evidence = 0usize;
    let (mut in_compact, mut in_full) = (0usize, 0usize);
    for (si, session) in q.haystack_sessions.iter().enumerate() {
        let tag = format!("session_{si}");
        let retrieved = top_k.contains(&tag);
        for turn in session.iter().filter(|t| t.has_answer) {
            evidence += 1;
            if !retrieved {
                continue;
            }
            in_full += 1;
            let line = format!("{}: {}", turn.role, turn.content);
            if shown(&session_texts[si]).contains(&line) {
                in_compact += 1;
            }
        }
    }
    let share = |n: usize| {
        if evidence == 0 {
            1.0
        } else {
            n as f64 / evidence as f64
        }
    };

    Harder {
        recall_all: if found == answers.len() { 1.0 } else { 0.0 },
        ndcg: if ideal > 0.0 { dcg / ideal } else { 0.0 },
        turn_recall_compact: share(in_compact),
        turn_recall_full: share(in_full),
        tokens_compact,
        tokens_full,
    }
}

fn question_failure(q: &Question, retrieved_sessions: Vec<String>, answer_sessions: Vec<String>) -> QuestionResult {
    QuestionResult {
        category: q.question_type.clone(),
        recall: 0.0,
        precision: 0.0,
        harder: Harder::default(),
        context: None,
        miss: Some(MissReport {
            question_id: q.question_id.clone(),
            question_type: q.question_type.clone(),
            question_date: q.question_date.clone(),
            recall: 0.0,
            precision: 0.0,
            retrieved_sessions,
            answer_sessions,
        }),
    }
}

fn maybe_rerank_recall_results(
    query: &str,
    results: &mut Vec<axil_core::RecallResult>,
    args: &Args,
) {
    if !matches!(args.rerank, BenchRerankMode::CrossEncoder) || results.is_empty() {
        return;
    }

    let original_ids: Vec<String> = results.iter().map(|r| r.record.id.to_string()).collect();
    let mut values: Vec<Value> = results
        .iter()
        .map(|result| {
            json!({
                "id": result.record.id.to_string(),
                "summary": axil_core::util::searchable_text(&result.record.data),
                "score": result.score,
            })
        })
        .collect();

    let config = axil_indexer::rerank::RerankConfig {
        enabled: true,
        model: args.rerank_model.clone(),
        top_k_rerank: args.rerank_top_k,
    };

    if let Err(err) = axil_indexer::rerank::rerank(query, &mut values, &config) {
        eprintln!("[rerank] cross-encoder failed: {err} — returning base recall order");
        return;
    }

    let mut by_id: HashMap<String, axil_core::RecallResult> = results
        .drain(..)
        .map(|result| (result.record.id.to_string(), result))
        .collect();
    let mut reranked = Vec::with_capacity(by_id.len());

    for value in &values {
        let Some(id) = value.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(mut result) = by_id.remove(id) else {
            continue;
        };
        if let Some(score) = value.get("rerank_score").and_then(|v| v.as_f64()) {
            result.score = score as f32;
        }
        reranked.push(result);
    }

    for id in original_ids {
        if let Some(result) = by_id.remove(&id) {
            reranked.push(result);
        }
    }

    *results = reranked;
}

#[cfg(feature = "rerank")]
fn cross_encoder_compiled() -> bool {
    true
}

#[cfg(not(feature = "rerank"))]
fn cross_encoder_compiled() -> bool {
    false
}

fn parse_bench_datetime(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y/%m/%d (%a) %H:%M")
        .ok()
        .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
}

fn parse_question_now(q: &Question) -> Option<DateTime<Utc>> {
    parse_bench_datetime(&q.question_date).or_else(|| {
        q.haystack_dates
            .iter()
            .filter_map(|s| parse_bench_datetime(s))
            .max()
    })
}

// Query-Time Chunk reranking has moved to axil-core (see
// `axil_core::scoring::QtcConfig` + `Axil::recall()` with `qtc: Some(...)`).
// The bench calls the library path for the `recall-qtc` strategy.

fn session_text(session: &[Turn]) -> String {
    session
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Concatenate the text of every turn flagged `has_answer:true` across all
/// haystack sessions — the text that literally contains the answer. Using
/// this as the retrieval query measures whether the embedder can *at all*
/// surface the answer-bearing session when query and answer use the same
/// wording. It is the upper bound on any retrieval-only improvement.
fn oracle_answer_text(q: &Question) -> String {
    q.haystack_sessions
        .iter()
        .flat_map(|s| s.iter())
        .filter(|t| t.has_answer)
        .map(|t| t.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn collapse_to_sessions(
    hits: Vec<(axil_core::RecordId, f32)>,
    record_to_session: &HashMap<axil_core::RecordId, String>,
    top_k: usize,
) -> Vec<String> {
    let mut best_by_session: HashMap<String, (f32, usize)> = HashMap::new();

    for (rank, (record_id, score)) in hits.into_iter().enumerate() {
        let Some(session_id) = record_to_session.get(&record_id) else {
            continue;
        };
        let entry = best_by_session
            .entry(session_id.clone())
            .or_insert((score, rank));
        if score > entry.0 || (score == entry.0 && rank < entry.1) {
            *entry = (score, rank);
        }
    }

    let mut sessions: Vec<(String, (f32, usize))> = best_by_session.into_iter().collect();
    sessions.sort_by(|a, b| {
        b.1.0
            .partial_cmp(&a.1.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.1.cmp(&b.1.1))
    });
    sessions.into_iter().take(top_k).map(|(session_id, _)| session_id).collect()
}
