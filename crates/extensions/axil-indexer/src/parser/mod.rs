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
/// and a `last_resort` fallback string for when nothing else works.
pub(super) fn generate_summary_common(
    file: &ParsedFile,
    pattern_labels: &[(&str, &str)],
    last_resort: impl Fn() -> String,
) -> String {
    // Priority 1: Module-level doc comment
    if let Some(ref doc) = file.module_doc {
        let first = doc.split('.').next().unwrap_or(doc).trim();
        if !first.is_empty() {
            return first.to_string();
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
                return format!("{first}. Defines {}", types.join(", "));
            }
            return first.to_string();
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
        desc_parts.join("; ")
    } else {
        last_resort()
    }
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
