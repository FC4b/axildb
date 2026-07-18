//! Shared utilities used across modules.

use crate::record::Record;

/// Extract searchable text from a record's data payload.
pub fn record_text(record: &Record) -> String {
    searchable_text(&record.data)
}

/// Extract the best long-form text for retrieval and embedding.
///
/// Prefers richer fields such as `full_text` and `content` before shorter
/// summaries, so long records are ranked using the text the user will
/// actually search for.
pub fn searchable_text(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            for key in &[
                "full_text",
                "content",
                "text",
                "description",
                "message",
                "summary",
                "fact",
                "error",
                "statement",
            ] {
                if let Some(serde_json::Value::String(s)) = map.get(*key) {
                    return s.clone();
                }
            }
            // Fallback: concatenate all string values
            map.values()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        other => other.to_string(),
    }
}

/// Extract searchable text from a JSON value (same logic as `record_text` but without Record wrapper).
pub fn value_text(data: &serde_json::Value) -> String {
    searchable_text(data)
}

/// Split long text into overlapping chunks for retrieval scoring.
///
/// For short text this returns a single chunk. Chunks respect UTF-8
/// boundaries so keyword scoring can safely work on arbitrary input.
pub fn overlapping_chunks(text: &str, max_bytes: usize, overlap_bytes: usize) -> Vec<String> {
    if text.is_empty() || text.len() <= max_bytes || max_bytes == 0 {
        return vec![text.to_string()];
    }

    let overlap = overlap_bytes.min(max_bytes.saturating_sub(1));
    let step = max_bytes.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < text.len() {
        let mut end = (start + max_bytes).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            break;
        }
        chunks.push(text[start..end].to_string());
        if end == text.len() {
            break;
        }
        start = (start + step).min(text.len());
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

/// Backward-compatible helper used by older call sites that just need one
/// representative text payload.
pub fn value_text_legacy(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            for key in &[
                "summary",
                "content",
                "text",
                "description",
                "message",
                "fact",
                "error",
            ] {
                if let Some(serde_json::Value::String(s)) = map.get(*key) {
                    return s.clone();
                }
            }
            map.values()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        other => other.to_string(),
    }
}

/// Cosine similarity between two vectors.
///
/// Returns 0.0 for mismatched lengths, empty vectors, or zero-norm vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
}

/// Truncate a string to a maximum byte length, respecting UTF-8 char boundaries.
pub fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// True when `AXIL_VERBOSE=1` (or `=true`). Engine crates use this to
/// silence startup banners during a normal run while leaving them
/// available for debugging.
pub fn verbose_logging_enabled() -> bool {
    std::env::var("AXIL_VERBOSE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Emit `msg` to stderr at most once per process, and only when
/// `AXIL_VERBOSE=1`. The caller owns the `Once` so multiple banners
/// don't share a single fire — each distinct call site gets its own
/// static `Once`.
pub fn log_once_if_verbose(once: &std::sync::Once, msg: &str) {
    once.call_once(|| {
        if verbose_logging_enabled() {
            eprintln!("{msg}");
        }
    });
}

/// Compute normalized text similarity using word overlap (Jaccard-like).
///
/// Returns 0.0–1.0 where 1.0 means identical word sets.
pub fn word_jaccard(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Extract a string array from a JSON value's field.
pub fn extract_str_array(data: &serde_json::Value, field: &str) -> Vec<String> {
    data.get(field)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Edge type constants used by intelligence features.
pub mod edge_types {
    /// Record mentions an entity.
    pub const MENTIONS: &str = "mentions";
    /// Two records are related by vector similarity.
    pub const RELATED_TO: &str = "related_to";
    /// Newer fact supersedes an older one.
    pub const SUPERSEDES: &str = "supersedes";
    /// Two facts contradict each other.
    pub const CONTRADICTS: &str = "contradicts";
    /// Source facts consolidated into a summary record.
    pub const CONSOLIDATED_INTO: &str = "consolidated_into";
    /// A record was derived from another (lineage chains). Create with
    /// `axil link <child> derived_from <parent> --props '{...}'`; walk with
    /// `axil lineage`.
    pub const DERIVED_FROM: &str = "derived_from";
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_text_prefers_summary() {
        let r = Record::new("t", json!({"summary": "hello", "other": "world"}));
        assert_eq!(record_text(&r), "hello");
    }

    #[test]
    fn searchable_text_prefers_full_text_over_summary() {
        let text = searchable_text(&json!({
            "summary": "short summary",
            "full_text": "longer full body",
        }));
        assert_eq!(text, "longer full body");
    }

    #[test]
    fn record_text_fallback_to_all_strings() {
        let r = Record::new("t", json!({"a": "one", "b": "two", "c": 42}));
        let text = record_text(&r);
        assert!(text.contains("one"));
        assert!(text.contains("two"));
    }

    #[test]
    fn record_text_plain_string() {
        let r = Record::new("t", json!("plain text"));
        assert_eq!(record_text(&r), "plain text");
    }

    #[test]
    fn overlapping_chunks_split_long_text() {
        let text = "a".repeat(5000);
        let chunks = overlapping_chunks(&text, 1600, 400);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| !c.is_empty() && c.len() <= 1600));
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_mismatched_len() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }
}
