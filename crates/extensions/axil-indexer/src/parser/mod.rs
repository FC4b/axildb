//! Language-specific source code parsers.
//!
//! Each parser extracts structured information from source files:
//! - Doc comments and module-level descriptions
//! - Public function/type signatures
//! - Import/export lists
//! - Pattern detection (error handling, HTTP routes, tests, etc.)

pub mod generic;
pub mod python;
pub mod rust;
pub mod typescript;

use crate::scanner::Language;

/// Information extracted from a source file by a language parser.
#[derive(Debug, Clone, Default)]
pub struct ParsedFile {
    /// 1-2 sentence summary of the file's purpose.
    pub summary: String,
    /// True when `summary` is a fallback (a line or symbol count, the first
    /// plain comment in the file, or "no summary available") rather than
    /// text drawn from doc comments, detected patterns, types or exports.
    /// It reads like a description but says little about what the file is
    /// for, so agent-facing summaries such as `axil boot` leave it out.
    pub summary_low_confidence: bool,
    /// Exported/public symbols.
    pub exports: Vec<String>,
    /// Import statements (module/crate names).
    pub imports: Vec<String>,
    /// Key types defined in this file (e.g. "Claims struct", "AuthError enum").
    pub key_types: Vec<String>,
    /// Extracted symbol information.
    pub symbols: Vec<ParsedSymbol>,
    /// Detected patterns (e.g. "error_handling", "http_handler", "tests").
    pub patterns: Vec<String>,
    /// Module-level doc comment, if any.
    pub module_doc: Option<String>,
}

/// A parsed public symbol (function, struct, enum, trait, etc.).
#[derive(Debug, Clone)]
pub struct ParsedSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub line: usize,
    pub signature: String,
    pub doc: Option<String>,
}

/// Kind of symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Interface,
    Class,
    Type,
    Constant,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::Class => "class",
            Self::Type => "type",
            Self::Constant => "constant",
        }
    }
}

/// Find the 1-indexed line number for a byte offset in source text.
pub(super) fn find_line(source: &str, offset: usize) -> usize {
    source[..offset].lines().count() + 1
}

/// Return the text inside the first balanced `{ … }` block at or after `from`,
/// or `None` when no balanced block is found. Naive brace counting (ignores
/// braces inside strings/comments) — good enough to harvest a member-name
/// digest for proxy enrichment, where a little over- or under-capture only
/// nudges the embedded keyword set.
pub(super) fn brace_match_body(source: &str, from: usize) -> Option<&str> {
    let open = from + source[from..].find('{')?;
    let mut depth = 0i32;
    for (i, c) in source[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Generate a summary from parsed file data with pattern-specific labels.
///
/// Shared logic across all language parsers. Each parser provides its own
/// `pattern_labels` mapping (e.g. `"http_handler"` → `"HTTP request handlers"`)
/// and a `last_resort` fallback string for when nothing else works. Returns
/// the summary and whether it is low-confidence, which is exactly when it
/// came from `last_resort`.
pub(super) fn generate_summary_common(
    file: &ParsedFile,
    pattern_labels: &[(&str, &str)],
    last_resort: impl Fn() -> String,
) -> (String, bool) {
    let confident = |s: String| (s, false);

    // Priority 1: Module-level doc comment
    if let Some(ref doc) = file.module_doc {
        let first = doc.split('.').next().unwrap_or(doc).trim();
        if !first.is_empty() {
            return confident(first.to_string());
        }
    }

    // Priority 2: Best symbol doc comment
    let best_doc = file
        .symbols
        .iter()
        .filter_map(|s| s.doc.as_deref())
        .filter(|d| d.len() > 10)
        .max_by_key(|d| d.len());

    if let Some(doc) = best_doc {
        let first = doc.split('.').next().unwrap_or(doc).trim();
        if first.len() > 15 {
            if !file.key_types.is_empty() {
                let types: Vec<&str> = file.key_types.iter().take(2).map(|s| s.as_str()).collect();
                return confident(format!("{first}. Defines {}", types.join(", ")));
            }
            return confident(first.to_string());
        }
    }

    // Priority 3: Pattern-based description
    let mut desc_parts = Vec::new();
    for pat in &file.patterns {
        for (key, label) in pattern_labels {
            if pat.as_str() == *key {
                desc_parts.push(label.to_string());
                break;
            }
        }
        // Also include trait impls directly
        if pat.starts_with("impl ") {
            desc_parts.push(pat.clone());
        }
    }

    // Priority 4: Key types
    if !file.key_types.is_empty() && desc_parts.is_empty() {
        let types: Vec<&str> = file.key_types.iter().take(3).map(|s| s.as_str()).collect();
        desc_parts.push(format!("defines {}", types.join(", ")));
    }

    // Priority 5: Exports
    if desc_parts.is_empty() && !file.exports.is_empty() {
        let top: Vec<&str> = file.exports.iter().take(3).map(|s| s.as_str()).collect();
        desc_parts.push(format!("provides {}", top.join(", ")));
    }

    if !desc_parts.is_empty() {
        confident(desc_parts.join("; "))
    } else {
        (last_resort(), true)
    }
}

/// Whether `summary` has the shape of a parser or module fallback: empty or
/// punctuation only, "no summary available", "N lines of <Lang> code", or
/// comma-separated counts such as "2 functions, 1 types, 40 lines" or
/// "3 files".
///
/// For records indexed before `summary_low_confidence` existed. The
/// first-comment fallback cannot be told apart from a real summary by its
/// text, so this catches only the machine-shaped ones.
pub fn is_fallback_summary(summary: &str) -> bool {
    let s = summary.trim();
    if s.chars()
        .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
    {
        return true;
    }
    if s == "no summary available" {
        return true;
    }
    let count_of = |part: &str, nouns: &[&str]| {
        let mut words = part.split_whitespace();
        matches!(
            (words.next(), words.next(), words.next()),
            (Some(n), Some(noun), None) if n.parse::<u64>().is_ok() && nouns.contains(&noun)
        )
    };
    let words: Vec<&str> = s.split_whitespace().collect();
    if let [n, "lines", "of", _, "code"] = words.as_slice() {
        if n.parse::<u64>().is_ok() {
            return true;
        }
    }
    s.split(", ")
        .all(|part| count_of(part, &["functions", "types", "lines", "files"]))
}

/// The summary of an indexed file or module record worth showing an agent,
/// or `None` when all it has is a fallback.
///
/// A record tagged `summary_low_confidence` has nothing to show. A module
/// whose files mix confident and fallback summaries carries
/// `confident_summary`, built from the confident ones only, and that is
/// preferred. Otherwise `summary` is used with any fallback-shaped parts
/// ([`is_fallback_summary`]) dropped, which also cleans module summaries
/// indexed before the tag existed.
pub fn confident_summary(data: &serde_json::Value) -> Option<String> {
    let text = |key: &str| data.get(key).and_then(serde_json::Value::as_str);
    if data
        .get("summary_low_confidence")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return None;
    }
    if let Some(c) = text("confident_summary").filter(|c| !c.trim().is_empty()) {
        return Some(c.to_string());
    }
    let parts: Vec<&str> = text("summary")?
        .split(". ")
        .map(str::trim)
        .filter(|p| !is_fallback_summary(p.trim_end_matches('.')))
        .collect();
    (!parts.is_empty()).then(|| parts.join(". "))
}

/// Parse a source file and extract structured information.
pub fn parse_file(source: &str, language: Language, include_private: bool) -> ParsedFile {
    match language {
        Language::Rust => rust::parse(source, include_private),
        Language::TypeScript | Language::JavaScript => typescript::parse(source, include_private),
        Language::Python => python::parse(source, include_private),
        _ => generic::parse(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parsers_tag_fallback_summaries() {
        let rust = parse_file("fn a() {}\n", Language::Rust, false);
        assert_eq!(rust.summary, "1 lines of Rust code");
        assert!(rust.summary_low_confidence);

        let comment = parse_file(
            "// TODO: split this file\nfn a() {}\n",
            Language::Rust,
            false,
        );
        assert_eq!(comment.summary, "TODO: split this file");
        assert!(comment.summary_low_confidence, "a first comment is a guess");

        let documented = parse_file("//! Token cache.\nfn a() {}\n", Language::Rust, false);
        assert!(!documented.summary_low_confidence);

        let python = parse_file("x = 1\n", Language::Python, false);
        assert_eq!(python.summary, "no summary available");
        assert!(python.summary_low_confidence);

        let generic = generic::parse("package main\nfunc main() {}\n");
        assert!(generic.summary_low_confidence, "{}", generic.summary);
        let generic_doc = generic::parse("// Package auth checks tokens.\npackage auth\n");
        assert!(!generic_doc.summary_low_confidence);
    }

    #[test]
    fn fallback_shapes_are_recognized() {
        for s in [
            "",
            " . ",
            "no summary available",
            "242 lines of Rust code",
            "2 functions, 1 types, 40 lines",
            "0 lines",
            "3 files",
        ] {
            assert!(is_fallback_summary(s), "{s:?}");
        }
        for s in [
            "JWT auth middleware",
            "tests",
            "defines Claims struct",
            "3 files: a, b",
        ] {
            assert!(!is_fallback_summary(s), "{s:?}");
        }
    }

    #[test]
    fn confident_summary_skips_fallbacks() {
        assert_eq!(
            confident_summary(&json!({"summary": "3 files", "summary_low_confidence": true})),
            None
        );
        assert_eq!(
            confident_summary(
                &json!({"summary": "a. 2 lines of Rust code", "confident_summary": "a"})
            )
            .as_deref(),
            Some("a")
        );
        // A module indexed before the tag: fallback-shaped parts are dropped.
        assert_eq!(
            confident_summary(
                &json!({"summary": "242 lines of Rust code. . Encryption benchmark"})
            )
            .as_deref(),
            Some("Encryption benchmark")
        );
        assert_eq!(
            confident_summary(&json!({"summary": "12 lines of Rust code"})),
            None
        );
    }
}
