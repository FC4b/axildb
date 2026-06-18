//! Recall scoring helpers — combines vector, FTS, and recency signals.

/// Combine vector similarity and full-text scores into a single rank.
pub fn rank(vector: f32, fts: f32, recency: f32) -> f32 {
    0.5 * vector + 0.3 * fts + 0.2 * recency
}

/// Apply a path/symbol exact-match boost to the base score.
pub fn boost_exact_match(base: f32, exact: bool) -> f32 {
    if exact { base + 0.10 } else { base }
}
