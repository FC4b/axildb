//! Test-only vector engine: deterministic embeddings and exact cosine search,
//! so recall and superseding are testable without ONNX.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axil_core::plugin::{Capability, Engine, TextEmbedder, VectorIndex};
use axil_core::{Axil, Record, RecordId, Result};

/// Embedding width. Each distinct word gets its own dimension (assigned on
/// first sight), so similarity is plain bag-of-words cosine with no hash
/// collisions.
const DIMS: usize = 256;

#[derive(Default)]
struct Inner {
    vectors: Mutex<HashMap<String, Vec<f32>>>,
    vocab: Mutex<HashMap<String, usize>>,
    embed_calls: AtomicUsize,
}

/// Shared handle onto the mock engine: the database owns one clone, the test
/// keeps another to inspect call counts or plant raw index entries.
#[derive(Clone, Default)]
pub(crate) struct MockVectors(Arc<Inner>);

impl MockVectors {
    /// Number of `embed` calls served so far.
    pub(crate) fn embed_calls(&self) -> usize {
        self.0.embed_calls.load(Ordering::SeqCst)
    }

    /// The embedding of `text`, without counting it as an embed call.
    pub(crate) fn vector_for(&self, text: &str) -> Vec<f32> {
        let mut vocab = self.0.vocab.lock().unwrap();
        let mut v = vec![0.0_f32; DIMS];
        for word in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let next = vocab.len();
            let idx = *vocab.entry(word.to_string()).or_insert(next);
            assert!(idx < DIMS, "mock vocabulary exhausted");
            v[idx] = 1.0;
        }
        v
    }

    /// Plant an index entry whose id resolves to no record — what a deleted
    /// record's lingering vector looks like to a search.
    pub(crate) fn plant_orphan(&self, vector: Vec<f32>) {
        self.0
            .vectors
            .lock()
            .unwrap()
            .insert(RecordId::new().to_string(), vector);
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

impl Engine for MockVectors {
    fn name(&self) -> &str {
        "mock-vectors"
    }
    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::VectorSearch]
    }
    fn on_record_insert(&self, _record: &Record) -> Result<()> {
        Ok(())
    }
    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        self.0.vectors.lock().unwrap().remove(&id.to_string());
        Ok(())
    }
}

impl VectorIndex for MockVectors {
    fn add(&self, id: RecordId, vector: &[f32]) -> Result<()> {
        self.0
            .vectors
            .lock()
            .unwrap()
            .insert(id.to_string(), vector.to_vec());
        Ok(())
    }
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>> {
        let mut scored: Vec<(RecordId, f32)> = self
            .0
            .vectors
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(id, v)| {
                RecordId::from_string(id)
                    .ok()
                    .map(|rid| (rid, cosine(query, v)))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_str().cmp(b.0.as_str()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
    fn count(&self) -> usize {
        self.0.vectors.lock().unwrap().len()
    }
    fn dimensions(&self) -> usize {
        DIMS
    }
}

impl TextEmbedder for MockVectors {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.0.embed_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.vector_for(text))
    }
}

/// A database with the mock vector engine attached, plus the test's handle.
pub(crate) fn vector_db() -> (Axil, MockVectors, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mock = MockVectors::default();
    let db = Axil::open(dir.path().join("mock.axil"))
        .with_vector_and_embedder(mock.clone())
        .build()
        .unwrap();
    (db, mock, dir)
}
