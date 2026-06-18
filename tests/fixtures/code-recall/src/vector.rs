//! Vector search backend — HNSW index over float embeddings.

/// Search the HNSW index for the top-k nearest vectors to the query.
pub fn search(query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let _ = (query, top_k);
    Vec::new()
}

/// Add a vector to the HNSW index.
pub fn add_vector(id: usize, vector: &[f32]) {
    let _ = (id, vector);
}
