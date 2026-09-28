//! Control arm of the Phase 29 bake-off: run Axil's production supersession
//! heuristic on the U1 pairs and print one JSON line per pair.
//!
//! Mirrors `Axil::detect_conflicts`: cosine similarity of the two texts under
//! the default bge-small embedder, then `check_conflict(new, existing, sim)`.
//! Retrieval is assumed to have surfaced the pair (the bake-off measures the
//! judgement, not the candidate search).

use std::io::{BufRead, BufReader, Write};

use axil_core::{check_conflict, judge_conflict, ConflictResult, Record};
use axil_vector::embed::Embedder;
use axil_vector::models::EmbeddingModel;
use serde_json::{json, Value};

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--full-path") {
        let path = args
            .get(1)
            .expect("usage: typed-decisions-heuristic --full-path <u1.jsonl>");
        full_path(path);
        return;
    }
    let path = args
        .first()
        .expect("usage: typed-decisions-heuristic <u1.jsonl>");
    let embedder = Embedder::new(EmbeddingModel::BgeSmall).expect("load bge-small");
    let file = std::fs::File::open(path).expect("open set");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line).expect("parse row");
        let old_text = row["old"].as_str().unwrap_or_default();
        let new_text = row["new"].as_str().unwrap_or_default();
        let old = Record::new("decisions", json!({ "summary": old_text }));
        let new = Record::new("decisions", json!({ "summary": new_text }));
        let sim = cosine(
            &embedder.embed(old_text).expect("embed old"),
            &embedder.embed(new_text).expect("embed new"),
        );
        let name = |r: ConflictResult| match r {
            ConflictResult::Novel => "novel",
            ConflictResult::Supersedes { .. } => "supersedes",
            ConflictResult::Contradicts { .. } => "contradicts",
        };
        // `verdict` is production (0.92 gate); `judge` is the same rules with
        // no gate, so any candidate floor can be evaluated offline.
        let verdict = name(check_conflict(&new, &old, sim));
        let judge = name(judge_conflict(&new, &old, sim));
        writeln!(
            out,
            "{}",
            json!({ "id": row["id"], "similarity": sim, "verdict": verdict, "judge": judge })
        )
        .expect("write");
    }
}

/// The supersession path end to end, in a real database: every pair's old
/// text is stored first, then each new text asks
/// `Axil::supersede_candidates` for its candidates and `judge_conflict`
/// rules on each. Prints one JSON line per pair: whether the true
/// predecessor was a candidate, whether the judge would supersede it, and
/// how many *other* candidates the judge would supersede (the harm case).
fn full_path(path: &str) {
    use axil_core::Axil;
    use axil_vector::AxilBuilderVectorExt;

    let rows: Vec<Value> = BufReader::new(std::fs::File::open(path).expect("open set"))
        .lines()
        .map(|l| l.expect("read line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("parse row"))
        .collect();

    let dir = tempfile::tempdir().expect("tempdir");
    let db = Axil::open(dir.path().join("u1.axil"))
        .with_embedder_model(EmbeddingModel::BgeSmall)
        .expect("vector store")
        .build()
        .expect("open db");

    // One record per distinct old text; pairs can share an old record.
    let mut old_ids: std::collections::HashMap<String, axil_core::RecordId> =
        std::collections::HashMap::new();
    for row in &rows {
        let text = row["old"].as_str().unwrap_or_default().to_string();
        if !old_ids.contains_key(&text) {
            let rec = db
                .insert("decisions", json!({ "summary": text }))
                .expect("insert old");
            old_ids.insert(text, rec.id);
        }
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for row in &rows {
        let true_old = &old_ids[row["old"].as_str().unwrap_or_default()];
        let new = Record::new("decisions", json!({ "summary": row["new"] }));
        let candidates = db.supersede_candidates(&new, 5).expect("candidates");
        let mut candidate = false;
        let mut supersedes_true = false;
        let mut supersedes_other = 0;
        for (existing, sim) in &candidates {
            let is_true = &existing.id == true_old;
            candidate |= is_true;
            if matches!(
                judge_conflict(&new, existing, *sim),
                ConflictResult::Supersedes { .. }
            ) {
                if is_true {
                    supersedes_true = true;
                } else {
                    supersedes_other += 1;
                }
            }
        }
        writeln!(
            out,
            "{}",
            json!({
                "id": row["id"],
                "candidates": candidates.len(),
                "candidate": candidate,
                "supersedes_true": supersedes_true,
                "supersedes_other": supersedes_other,
            })
        )
        .expect("write");
    }
}
