//! Shared utilities used across modules.

use crate::record::Record;

/// Extract searchable text from a record's data payload.
pub fn record_text(record: &Record) -> String {
    searchable_text(&record.data)
}

/// The fields [`searchable_text`] reads, in priority order. The first one
/// holding a string is the record's text. A record with none of them falls
/// back to every string value joined, ids and timestamps included; exported
/// so diagnostics can tell those records apart without copying the list.
pub const SEARCHABLE_TEXT_KEYS: &[&str] = &[
    "full_text",
    "content",
    "text",
    "description",
    "message",
    "summary",
    "fact",
    "error",
    "statement",
];

/// Extract the best long-form text for retrieval and embedding.
///
/// Prefers richer fields such as `full_text` and `content` before shorter
/// summaries, so long records are ranked using the text the user will
/// actually search for.
pub fn searchable_text(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            for key in SEARCHABLE_TEXT_KEYS {
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

/// Longest recall chunk, in bytes. A record whose [`searchable_text`] is
/// longer gets one extra vector per [`recall_units`] piece, besides its
/// own; exported so diagnostics that rebuild a record's vectors split its
/// text the same way insert does.
pub const RECALL_CHUNK_MAX_BYTES: usize = 1600;

/// Bytes each recall chunk shares with the one before it, where a line
/// longer than [`RECALL_CHUNK_MAX_BYTES`] is cut into overlapping pieces.
pub const RECALL_CHUNK_OVERLAP_BYTES: usize = 400;

/// Shortest recall unit, in bytes: a shorter line joins the line after it,
/// so a one-word reply rides with the turn it answers instead of taking a
/// vector of its own.
pub const RECALL_UNIT_MIN_BYTES: usize = 400;

/// The pieces of a record's text that get a vector of their own besides
/// the record's: empty when the text fits in one [`RECALL_CHUNK_MAX_BYTES`]
/// chunk.
///
/// Longer text splits at its lines, so each unit is a dialog turn or a fact
/// (a line, joined with the next ones until it holds
/// [`RECALL_UNIT_MIN_BYTES`]). A question about one turn then meets that
/// turn alone, not a fixed window that cuts it in half and pads it with its
/// neighbours. A unit longer than a chunk is cut into [`overlapping_chunks`],
/// so text without line breaks splits as it did before.
pub fn recall_units(text: &str) -> Vec<String> {
    if text.len() <= RECALL_CHUNK_MAX_BYTES {
        return Vec::new();
    }
    let mut units = Vec::new();
    let mut flush = |unit: &mut String| {
        if unit.len() > RECALL_CHUNK_MAX_BYTES {
            units.extend(overlapping_chunks(
                unit,
                RECALL_CHUNK_MAX_BYTES,
                RECALL_CHUNK_OVERLAP_BYTES,
            ));
        } else {
            units.push(std::mem::take(unit));
        }
        unit.clear();
    };
    let mut unit = String::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if !unit.is_empty() {
            unit.push('\n');
        }
        unit.push_str(line);
        if unit.len() >= RECALL_UNIT_MIN_BYTES {
            flush(&mut unit);
        }
    }
    if !unit.is_empty() {
        flush(&mut unit);
    }
    units
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
            crate::diagln!("{msg}");
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
    /// A newer record that a judge flagged as possibly replacing or
    /// contradicting an older one (newer → older). A suggestion to review,
    /// written by `Axil::review_possible_contradictions`; it changes neither
    /// record.
    pub const POSSIBLY_CONTRADICTS: &str = "possibly_contradicts";
    /// A record (the fix) resolves an `errors` record. Written only by
    /// `Axil::resolve_error`, which also removes it when the error is reopened.
    pub const RESOLVES: &str = "resolves";
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
    fn short_text_has_no_recall_units() {
        assert!(recall_units("user: hi\nassistant: hello").is_empty());
        assert!(recall_units(&"a".repeat(RECALL_CHUNK_MAX_BYTES)).is_empty());
    }

    #[test]
    fn recall_units_follow_turns() {
        let user = format!("user: {}", "where should we eat ".repeat(25));
        let reply = format!("assistant: {}", "try the noodle place ".repeat(25));
        let text = [user.as_str(), "user: ok", reply.as_str(), &user, &reply].join("\n");
        let units = recall_units(&text);
        assert_eq!(units.len(), 4, "{units:#?}");
        assert_eq!(units[0], user);
        // A short line joins the line after it.
        assert_eq!(units[1], format!("user: ok\n{reply}"));
        assert_eq!(units[2], user);
        assert_eq!(units[3], reply);
    }

    #[test]
    fn a_long_line_is_cut_into_overlapping_pieces() {
        let long = "x".repeat(4000);
        let units = recall_units(&long);
        assert_eq!(
            units,
            overlapping_chunks(&long, RECALL_CHUNK_MAX_BYTES, RECALL_CHUNK_OVERLAP_BYTES)
        );
        let text = format!("{}\n{long}", "y".repeat(500));
        let units = recall_units(&text);
        assert_eq!(units[0], "y".repeat(500));
        assert!(units[1..].iter().all(|u| u.len() <= RECALL_CHUNK_MAX_BYTES));
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
