//! Bulk filesystem-ingest benchmark.
//!
//! Mirrors the shape of `axil ingest <dir>`: generate a synthetic corpus of
//! markdown docs in a temp dir, then walk + read + paragraph-chunk + insert
//! each one into a fresh temp database. Everything is generated in-process, so
//! the bench is hermetic — no external corpus and no committed throughput
//! number; Criterion measures the wall time per run.
//!
//! The DB is opened without a vector engine, so this isolates the storage hot
//! path (chunk + insert) and stays fast/offline (no ONNX model download). It is
//! the dominant cost for the prose corpora `axil ingest` targets.

use std::fs;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::json;
use tempfile::TempDir;

use axil_core::Axil;

/// One synthetic markdown doc: a heading plus `paras` paragraphs, sized so it
/// chunks into more than one record (exercises the per-chunk insert loop).
fn synthetic_doc(idx: usize, paras: usize) -> String {
    let mut s = format!("# Note {idx}\n\n");
    for p in 0..paras {
        s.push_str(&format!(
            "Paragraph {p} of note {idx}. The auth refactor moved AuthModule to \
             OAuth2 using the standard JWT library, and the ingest pipeline walks \
             a directory chunking each file on paragraph boundaries.\n\n"
        ));
    }
    s
}

/// Write `n` synthetic `.md` docs into a fresh temp dir and return it.
fn synthetic_corpus(n: usize, paras: usize) -> TempDir {
    let dir = TempDir::new().expect("failed to create corpus dir");
    for i in 0..n {
        fs::write(dir.path().join(format!("doc-{i}.md")), synthetic_doc(i, paras))
            .expect("failed to write synthetic doc");
    }
    dir
}

/// Paragraph chunker matching the ingest body's behavior: split on blank lines,
/// merge paragraphs up to `max_bytes`, hard-wrapping any oversized paragraph.
fn chunk_text(text: &str, max_bytes: usize) -> Vec<String> {
    let max_bytes = max_bytes.max(4);
    if text.len() <= max_bytes {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for para in text.split("\n\n") {
        if para.is_empty() {
            continue;
        }
        if current.len() + para.len() + 2 > max_bytes && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if para.len() > max_bytes {
            for slice in para.as_bytes().chunks(max_bytes) {
                chunks.push(String::from_utf8_lossy(slice).into_owned());
            }
        } else {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Run the ingest pipeline shape over a pre-built corpus into a fresh DB.
fn ingest_corpus(corpus: &std::path::Path, chunk_bytes: usize) -> usize {
    let dir = TempDir::new().expect("failed to create db dir");
    let db = Axil::open(dir.path().join("bench.axil"))
        .build()
        .expect("failed to open db");
    let mut chunks_written = 0usize;
    for entry in fs::read_dir(corpus).expect("read_dir failed").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for (chunk_idx, chunk) in chunk_text(&content, chunk_bytes).iter().enumerate() {
            db.insert(
                "notes",
                json!({
                    "path": path.display().to_string(),
                    "chunk_idx": chunk_idx,
                    "content": chunk,
                    "source": "ingest",
                }),
            )
            .expect("insert failed");
            chunks_written += 1;
        }
    }
    chunks_written
}

fn bench_bulk_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_ingest");
    // A handful of file counts; each doc chunks into ~3 records at 2000 bytes.
    for files in [50usize, 200, 500] {
        // Build the corpus once, outside the measured section.
        let corpus = synthetic_corpus(files, 6);
        group.bench_with_input(BenchmarkId::from_parameter(files), &files, |b, _| {
            b.iter(|| black_box(ingest_corpus(corpus.path(), 2000)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bulk_ingest);
criterion_main!(benches);
