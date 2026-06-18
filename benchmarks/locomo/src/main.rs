//! LoCoMo benchmark runner for Axil.
//!
//! Measures retrieval quality on the LoCoMo dataset:
//! 1. For each conversation, ingests all messages into a fresh Axil DB
//! 2. For each question, queries using vector search / FTS / ask / recall strategy
//! 3. Checks if ground-truth answer appears in retrieved context (substring match)
//! 4. Also measures evidence-turn recall (fraction of evidence turns retrieved)
//! 5. Reports per-category accuracy + MemScore
//!
//! Usage:
//!   cargo run -p locomo-bench -- [OPTIONS]
//!
//! Options:
//!   --data-dir PATH          Path to dataset directory (default: benchmarks/locomo/data)
//!   --limit N                Max conversations to evaluate (default: all)
//!   --strategy vector|fts|ask|recall  Retrieval strategy (default: vector)
//!   --top-k N                Results per query (default: 5)
//!   --verbose                Print per-question details

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;

use axil_core::Axil;
use axil_fts::AxilBuilderFtsExt;
use axil_vector::models::EmbeddingModel;
use axil_vector::AxilBuilderVectorExt;

// ── CLI ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "locomo-bench", about = "LoCoMo benchmark for Axil")]
struct Args {
    /// Path to dataset directory containing locomo10.json.
    #[arg(long, default_value = "benchmarks/locomo/data")]
    data_dir: PathBuf,

    /// Max conversations to evaluate (0 = all).
    #[arg(long, default_value = "0")]
    limit: usize,

    /// Retrieval strategy: vector, fts, ask, recall.
    #[arg(long, default_value = "vector")]
    strategy: String,

    /// Number of results per query.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Print per-question details.
    #[arg(long)]
    verbose: bool,
}

// ── Dataset types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Conversation {
    #[allow(dead_code)]
    conversation_id: String,
    conversations: Vec<Turn>,
    questions: Vec<Question>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct Question {
    question_id: String,
    category: String,
    question: String,
    answer: String,
    #[serde(default)]
    evidence_turn_indices: Vec<usize>,
}

// ── Results ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct CategoryStats {
    total: usize,
    hits: usize,
    evidence_recall_sum: f64,
    latency_ms: Vec<f64>,
    context_tokens: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    memscore: String,
    breakdown: HashMap<String, f64>,
    latency_p50_ms: u64,
    latency_p99_ms: u64,
    avg_context_tokens: usize,
    total_questions: usize,
    excluded_adversarial: usize,
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let data_file = args.data_dir.join("locomo10.json");

    eprintln!("{}", "=".repeat(54));
    eprintln!("  LoCoMo Benchmark for Axil");
    eprintln!("{}", "=".repeat(54));
    eprintln!("  Strategy: {}", args.strategy);
    eprintln!("  Top-K:    {}", args.top_k);
    eprintln!("  Data:     {}", data_file.display());
    eprintln!("{}", "=".repeat(54));
    eprintln!();

    // Load dataset
    eprintln!("[1/3] Loading dataset...");
    let file = std::fs::File::open(&data_file).expect("cannot open dataset file");
    let conversations: Vec<Conversation> =
        serde_json::from_reader(file).expect("invalid JSON");
    let conv_count = if args.limit > 0 {
        conversations.len().min(args.limit)
    } else {
        conversations.len()
    };
    eprintln!(
        "       {} conversations loaded, evaluating {}",
        conversations.len(),
        conv_count
    );

    // Evaluate
    eprintln!("[2/3] Evaluating...");
    let mut category_stats: HashMap<String, CategoryStats> = HashMap::new();
    let mut all_latencies: Vec<f64> = Vec::new();
    let mut all_context_tokens: Vec<usize> = Vec::new();
    let mut total_questions = 0usize;
    let mut excluded_adversarial = 0usize;

    for (ci, conv) in conversations.iter().take(conv_count).enumerate() {
        eprint!(
            "\r       conversation {}/{}",
            ci + 1,
            conv_count
        );

        // Build a temp DB for this conversation
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db_path = tmp.path().join("bench.axil");

        let builder = Axil::open(&db_path);
        let builder = match builder.with_embedder_model(EmbeddingModel::BgeSmall) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("\n  WARN: embedder init failed: {e}");
                continue;
            }
        };
        let builder = match builder.with_fts_plugin() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("\n  WARN: FTS init failed: {e}");
                continue;
            }
        };
        let db = match builder.build() {
            Ok(db) => db,
            Err(e) => {
                eprintln!("\n  WARN: DB build failed: {e}");
                continue;
            }
        };

        // Ingest all turns as individual records
        let mut turn_record_ids: Vec<Option<axil_core::RecordId>> = Vec::new();

        for (ti, turn) in conv.conversations.iter().enumerate() {
            let data = json!({
                "turn_index": ti,
                "role": turn.role,
                "content": turn.content,
            });

            match db.insert("messages", data) {
                Ok(record) => {
                    let _ = db.embed_field(&record.id, "content");
                    let _ = db.index_text(&record.id, "content", &turn.content);
                    turn_record_ids.push(Some(record.id));
                }
                Err(_) => {
                    turn_record_ids.push(None);
                }
            }
        }

        // Evaluate each question
        for q in &conv.questions {
            // Exclude adversarial from official score
            if q.category == "adversarial" {
                excluded_adversarial += 1;
                continue;
            }

            total_questions += 1;

            let start = Instant::now();

            // Retrieve
            let retrieved: Vec<(axil_core::Record, f32)> = match args.strategy.as_str() {
                "fts" => db
                    .search_text(&q.question, args.top_k)
                    .unwrap_or_default(),
                "ask" => {
                    match axil_indexer::ask::ask(&db, &q.question, args.top_k) {
                        Ok(result) => result
                            .results
                            .iter()
                            .filter_map(|v| {
                                v.get("id")
                                    .and_then(|id| id.as_str())
                                    .and_then(|id| axil_core::RecordId::from_string(id).ok())
                            })
                            .filter_map(|id| {
                                db.get(&id).ok().flatten().map(|r| (r, 0.0))
                            })
                            .collect(),
                        Err(_) => vec![],
                    }
                }
                "recall" => db
                    .recall(&q.question, args.top_k, None)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|result| (result.record, result.score))
                    .collect(),
                _ => {
                    // vector (default)
                    db.similar_to(&q.question, args.top_k)
                        .unwrap_or_default()
                }
            };

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Collect retrieved text for answer-in-context check
            let mut context_text = String::new();
            let mut retrieved_turn_indices: Vec<usize> = Vec::new();

            for (record, _score) in &retrieved {
                if let Some(content) = record.data.get("content").and_then(|v| v.as_str()) {
                    context_text.push_str(content);
                    context_text.push('\n');
                }
                if let Some(ti) = record.data.get("turn_index").and_then(|v| v.as_u64()) {
                    retrieved_turn_indices.push(ti as usize);
                }
            }

            // Token count estimate: text.len() / 4
            let context_tokens = context_text.len() / 4;

            // Hit: ground-truth answer appears in retrieved context (case-insensitive)
            let answer_lower = q.answer.to_lowercase();
            let context_lower = context_text.to_lowercase();
            let hit = context_lower.contains(&answer_lower);

            // Evidence recall: fraction of evidence turns that were retrieved
            let evidence_recall = if q.evidence_turn_indices.is_empty() {
                if hit { 1.0 } else { 0.0 }
            } else {
                let found = q
                    .evidence_turn_indices
                    .iter()
                    .filter(|idx| retrieved_turn_indices.contains(idx))
                    .count();
                found as f64 / q.evidence_turn_indices.len() as f64
            };

            // Accumulate stats
            let stats = category_stats.entry(q.category.clone()).or_default();
            stats.total += 1;
            if hit {
                stats.hits += 1;
            }
            stats.evidence_recall_sum += evidence_recall;
            stats.latency_ms.push(elapsed_ms);
            stats.context_tokens.push(context_tokens);

            all_latencies.push(elapsed_ms);
            all_context_tokens.push(context_tokens);

            if args.verbose {
                eprintln!(
                    "\n  {} [{}] hit={} evidence_recall={:.2} {:.1}ms {}tok",
                    q.question_id,
                    q.category,
                    hit,
                    evidence_recall,
                    elapsed_ms,
                    context_tokens
                );
            }
        }
    }
    eprintln!(
        "\r       {}/{} conversations done.          ",
        conv_count, conv_count
    );

    // Build report
    eprintln!("[3/3] Results:");
    eprintln!();

    let total_hits: usize = category_stats.values().map(|s| s.hits).sum();
    let overall_accuracy = if total_questions > 0 {
        total_hits as f64 / total_questions as f64 * 100.0
    } else {
        0.0
    };

    let latency_p50 = percentile(&mut all_latencies, 50.0);
    let latency_p99 = percentile(&mut all_latencies, 99.0);

    let avg_context_tokens = if all_context_tokens.is_empty() {
        0
    } else {
        all_context_tokens.iter().sum::<usize>() / all_context_tokens.len()
    };

    let mut breakdown: HashMap<String, f64> = HashMap::new();
    for (cat, stats) in &category_stats {
        let acc = if stats.total > 0 {
            stats.hits as f64 / stats.total as f64 * 100.0
        } else {
            0.0
        };
        breakdown.insert(cat.clone(), round1(acc));
    }

    let memscore = format!(
        "{:.0}% / {:.0}ms / {}tok",
        overall_accuracy, latency_p50, avg_context_tokens
    );

    let report = BenchmarkReport {
        memscore,
        breakdown,
        latency_p50_ms: latency_p50 as u64,
        latency_p99_ms: latency_p99 as u64,
        avg_context_tokens,
        total_questions,
        excluded_adversarial,
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

// ── Helpers ────────────────────────────────────────────────────────

/// Compute a percentile from a mutable slice (sorts in place).
fn percentile(data: &mut [f64], pct: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * (data.len() - 1) as f64).round() as usize;
    data[idx.min(data.len() - 1)]
}

/// Round to 1 decimal place.
fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
