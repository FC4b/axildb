//! Control arm of the Phase 29 bake-off: run Axil's production supersession
//! heuristic on the U1 pairs and print one JSON line per pair.
//!
//! Mirrors `Axil::detect_conflicts`: cosine similarity of the two texts under
//! the default bge-small embedder, then `check_conflict(new, existing, sim)`.
//! Retrieval is assumed to have surfaced the pair (the bake-off measures the
//! judgement, not the candidate search).

use std::io::{BufRead, BufReader, Write};

use axil_core::{check_conflict, ConflictResult, Record};
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
    let path = std::env::args().nth(1).expect("usage: typed-decisions-heuristic <u1.jsonl>");
    let embedder = Embedder::new(EmbeddingModel::BgeSmall).expect("load bge-small");
    let file = std::fs::File::open(&path).expect("open set");
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
        let verdict = match check_conflict(&new, &old, sim) {
            ConflictResult::Novel => "novel",
            ConflictResult::Supersedes { .. } => "supersedes",
            ConflictResult::Contradicts { .. } => "contradicts",
        };
        writeln!(out, "{}", json!({ "id": row["id"], "similarity": sim, "verdict": verdict }))
            .expect("write");
    }
}
