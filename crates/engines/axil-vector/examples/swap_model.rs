//! Re-embed a copy of a database under another embedding model, keeping the
//! set of records that have a vector.
//!
//! An embedding-model A/B on the dogfood eval needs each model's vectors for
//! the records the snapshot embeds, and no others. `axil reembed` embeds one
//! field of every record that has it, and `axil heal --reindex` with
//! `--embed-internal` embeds every row that could have a vector; both change
//! which rows compete in recall's vector channel, which moves the eval as
//! much as a model does (restoring 2,962 code-index vectors took the dogfood
//! hit@1 from 0.70 to 0.51). This keeps the membership and swaps the model:
//!
//! 1. Every record of `--from` with a vector is an owner. A vector whose
//!    record is gone is dropped, as `axil heal --reindex` drops it.
//! 2. Each owner is embedded into `--to` with `--model`: a memory record's
//!    `searchable_text`, which insert embeds, or for an internal row the
//!    field its writer embeds (`EMBEDDED_INTERNAL_FIELDS` in axil-core).
//!
//! `--to` must be a copy of the same database without its `.vec` file. Put an
//! `axil.toml` naming the model beside it, so `axil` embeds queries with the
//! same model. Rebuilding with the model `--from` was embedded with checks
//! the text choice: each fresh vector is compared with the stored one, per
//! table, in the report's `same_model_check`.
//!
//! ```text
//! cargo run --release -p axil-vector --features embed --example swap_model -- \
//!     --from <snapshot copy>/memory.axil --to <bare copy>/memory.axil \
//!     --model granite-small [--from-model bge-small]
//! ```
//!
//! Prints one JSON report on stdout.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use axil_core::{util, Axil, Record, RecordId};
use axil_vector::models::EmbeddingModel;
use axil_vector::AxilBuilderVectorExt;
use serde_json::json;

/// The text each internal table's writer embeds; mirrors
/// `EMBEDDED_INTERNAL_FIELDS` in axil-core's `db.rs`.
const INTERNAL_FIELDS: &[(&str, &str)] = &[
    ("_idx_code_proxies", "proxy_text"),
    ("_idx_files", "summary"),
    ("_idx_modules", "summary"),
    ("_idx_project", "summary"),
    ("_dep_docs", "content"),
    ("_recall_chunks", "content"),
    ("_checkpoint_records", "summary"),
];

/// A fresh vector counts as the stored one re-made when at least this close.
const SAME_COSINE: f32 = 0.99;

/// Owners embedded per `embed_fields_batch` call.
const SLICE: usize = 64;

struct Args {
    from: PathBuf,
    to: PathBuf,
    model: EmbeddingModel,
    from_model: EmbeddingModel,
}

fn parse_args() -> Args {
    let mut from = None;
    let mut to = None;
    let mut model = None;
    let mut from_model = EmbeddingModel::BgeSmall;
    let mut args = std::env::args().skip(1);
    let model_named = |name: Option<String>| {
        let name = name.expect("a model name");
        EmbeddingModel::from_name(&name).unwrap_or_else(|| panic!("unknown model {name}"))
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--from" => from = args.next().map(PathBuf::from),
            "--to" => to = args.next().map(PathBuf::from),
            "--model" => model = Some(model_named(args.next())),
            "--from-model" => from_model = model_named(args.next()),
            other => panic!("unknown argument {other}"),
        }
    }
    Args {
        from: from.expect("--from <db>"),
        to: to.expect("--to <db>"),
        model: model.expect("--model <name>"),
        from_model,
    }
}

/// The text insert or the row's writer embeds, and where it came from.
fn owner_text(record: &Record) -> (String, &'static str) {
    if !record.table.starts_with('_') {
        return (util::searchable_text(&record.data), "searchable_text");
    }
    let field = INTERNAL_FIELDS
        .iter()
        .find(|(table, _)| *table == record.table)
        .map(|(_, field)| *field);
    match field.and_then(|f| record.data.get(f)?.as_str().map(|s| (s, f))) {
        Some((text, field)) => (text.to_string(), field),
        None => (
            util::searchable_text(&record.data),
            "searchable_text (no writer field)",
        ),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn main() {
    let args = parse_args();
    assert!(
        !axil_vector::vector_db_path(&args.to).exists(),
        "{} already has a vector store; --to must be a copy without one",
        args.to.display()
    );

    let source = Axil::open(&args.from)
        .with_vector_auto()
        .and_then(|b| b.build())
        .expect("open --from with its vector store");
    let stored_vectors = source.vector_count().unwrap_or(0);
    let mut owners: Vec<(Record, Vec<f32>)> = Vec::new();
    for record in source.storage().scan_all_records().expect("scan --from") {
        if let Some(vector) = source.get_vector(&record.id).expect("read vector") {
            owners.push((record, vector));
        }
    }
    drop(source);

    let target = Axil::open(&args.to)
        .with_embedder_model(args.model.clone())
        .and_then(|b| b.build())
        .expect("open --to with the new model");

    let texts: Vec<(String, &'static str)> = owners.iter().map(|(r, _)| owner_text(r)).collect();
    let mut by_table: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for ((record, _), (_, source)) in owners.iter().zip(&texts) {
        *by_table
            .entry(record.table.as_str())
            .or_default()
            .entry(source)
            .or_default() += 1;
    }

    let entries: Vec<(&RecordId, &str)> = owners
        .iter()
        .zip(&texts)
        .map(|((r, _), (t, _))| (&r.id, t.as_str()))
        .collect();
    let started = Instant::now();
    for (i, slice) in entries.chunks(SLICE).enumerate() {
        target.embed_fields_batch(slice).expect("embed owners");
        eprint!("\r  {}/{}", (i * SLICE + slice.len()), entries.len());
    }
    eprintln!();
    let embed_secs = started.elapsed().as_secs_f64();
    let text_bytes: usize = texts.iter().map(|(t, _)| t.len()).sum();

    let same_model = args.model.name() == args.from_model.name();
    let mut check: BTreeMap<&str, (usize, usize, f32)> = BTreeMap::new();
    if same_model {
        for (record, stored) in &owners {
            let fresh = target
                .get_vector(&record.id)
                .expect("read rebuilt vector")
                .expect("every owner was embedded");
            let cos = cosine(&fresh, stored);
            let entry = check
                .entry(record.table.as_str())
                .or_insert((0, 0, f32::INFINITY));
            entry.0 += 1;
            entry.1 += usize::from(cos >= SAME_COSINE);
            entry.2 = entry.2.min(cos);
        }
    }
    // Cosine between memory records of one table, under the new model. The
    // supersede threshold (0.92) and the candidate floor (0.65) are absolute
    // cosines, so a model that places unrelated text closer together moves
    // how many pairs cross them.
    let mut tables: BTreeMap<&str, Vec<Vec<f32>>> = BTreeMap::new();
    for (record, _) in owners.iter().filter(|(r, _)| !r.table.starts_with('_')) {
        if let Some(v) = target.get_vector(&record.id).expect("read rebuilt vector") {
            tables.entry(record.table.as_str()).or_default().push(v);
        }
    }
    let mut pairs: Vec<f32> = Vec::new();
    for vectors in tables.values() {
        for (i, a) in vectors.iter().enumerate() {
            for b in &vectors[i + 1..] {
                pairs.push(cosine(a, b));
            }
        }
    }
    pairs.sort_by(f32::total_cmp);
    let quantile = |q: f64| {
        let i = ((pairs.len().saturating_sub(1)) as f64 * q).round() as usize;
        pairs.get(i).copied().unwrap_or(f32::NAN)
    };
    let share_at_least =
        |t: f32| pairs.iter().filter(|&&c| c >= t).count() as f64 / pairs.len().max(1) as f64;
    let pair_report = json!({
        "pairs": pairs.len(),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "max": pairs.last().copied().unwrap_or(f32::NAN),
        "share_at_least_0.65": share_at_least(0.65),
        "share_at_least_0.92": share_at_least(0.92),
    });

    let rebuilt = target.vector_count().unwrap_or(0);
    drop(target);
    let vec_bytes = std::fs::metadata(axil_vector::vector_db_path(&args.to))
        .map(|m| m.len())
        .unwrap_or(0);

    let report = json!({
        "from": args.from.display().to_string(),
        "to": args.to.display().to_string(),
        "model": args.model.name(),
        "dimensions": args.model.dimensions(),
        "max_seq_len": args.model.max_seq_len(),
        "from_vectors": stored_vectors,
        "owners": owners.len(),
        "dropped_without_record": stored_vectors.saturating_sub(owners.len()),
        "rebuilt_vectors": rebuilt,
        "owners_by_table_and_text": by_table,
        "text_bytes": text_bytes,
        "embed_seconds": (embed_secs * 1000.0).round() / 1000.0,
        "embed_ms_per_owner": (embed_secs * 1e6 / owners.len().max(1) as f64).round() / 1000.0,
        "vec_file_bytes": vec_bytes,
        "same_table_memory_pair_cosines": pair_report,
        "same_model_check": if same_model {
            json!({
                "from_model": args.from_model.name(),
                "threshold": SAME_COSINE,
                "by_table": check
                    .iter()
                    .map(|(t, (n, same, min))| {
                        (t.to_string(), json!({"n": n, "same": same, "min_cosine": min}))
                    })
                    .collect::<serde_json::Map<_, _>>(),
            })
        } else {
            serde_json::Value::Null
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
