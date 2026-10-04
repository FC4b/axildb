//! Pattern-based entity extraction — no LLM required.
//!
//! Extracts entities from record text using deterministic patterns:
//! file paths, CamelCase/snake_case identifiers, backtick-wrapped code,
//! quoted strings, and project names. Also handles code symbols
//! (`fn foo`, `auth::login`, `class User`, …) with a language hint so
//! cross-language collisions can be disambiguated downstream. Declaration
//! keywords such as `trait` and `class` are English words too, so those
//! patterns only read code: fenced blocks, inline code, and code-like lines.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Type of extracted entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// File path: `/src/auth/login.rs`
    File,
    /// Code identifier: `AuthModule`, `auth_module`
    Code,
    /// Project name
    Project,
    /// Quoted or backtick-wrapped string
    Reference,
    /// A code symbol extracted via a language-aware pattern.
    /// `lang_hint` is `"rust" | "python" | "ts" | "go" | "java" | ...`
    /// when the regex class gives us a strong signal; `None` otherwise.
    CodeSymbol { lang_hint: Option<String> },
}

/// Build a provisional canonical id for a code-symbol entity that
/// has not yet been grounded in a SCIP index.
///
/// The id is stable across re-extraction of the same `(name, lang, file)`
/// tuple, so later SCIP ingest can rewrite it without duplicating records.
pub fn provisional_canonical_id(name: &str, lang: Option<&str>, file: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update(b"|");
    hasher.update(lang.unwrap_or("").as_bytes());
    hasher.update(b"|");
    hasher.update(file.unwrap_or("").as_bytes());
    let digest = hasher.finalize();
    // 40 hex chars is plenty to avoid collisions while staying short.
    let hex: String = digest.iter().take(20).map(|b| format!("{b:02x}")).collect();
    format!("provisional:{hex}")
}

/// Canonical id for a non-code entity. Today identical to the normalized name —
/// future revisions may namespace natural-language entities the same way
/// code symbols are namespaced.
pub fn natural_canonical_id(name: &str) -> String {
    name.to_string()
}

/// An extracted entity with its type and source text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity {
    /// The entity name (normalized).
    pub name: String,
    /// The type of entity.
    pub entity_type: EntityType,
    /// Original text as found in the source.
    pub source_text: String,
}

/// Extract entities from a text string.
///
/// Priority order (higher priority wins for overlapping extractions):
/// 1. Backtick-wrapped code: `` `AuthModule` `` → exact entity
/// 2. File paths: `/src/auth/login.rs` → file entity
/// 3. CamelCase identifiers: `AuthModule` → code entity
/// 4. snake_case identifiers: `auth_module` → code entity
/// 5. Quoted strings (3+ words): potential reference entity
pub fn extract_entities(text: &str) -> Vec<Entity> {
    let mut entities = Vec::new();
    let mut seen = HashSet::new();

    // 1. Backtick-wrapped code
    extract_backtick_entities(text, &mut entities, &mut seen);

    // 2. File paths
    extract_file_paths(text, &mut entities, &mut seen);

    // 3. CamelCase identifiers
    extract_camel_case(text, &mut entities, &mut seen);

    // 4. snake_case identifiers
    extract_snake_case(text, &mut entities, &mut seen);

    // 5. Quoted strings
    extract_quoted_strings(text, &mut entities, &mut seen);

    extract_code_symbols(text, &mut entities, &mut seen);

    entities
}

fn extract_backtick_entities(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    // Inline code only: a fenced block is code to read, not a name.
    for (line, fenced) in lines_with_fences(text) {
        if fenced {
            continue;
        }
        for span in inline_code_spans(line) {
            let trimmed = span.trim();
            // Markdown syntax quoted in backticks (`- [ ]`, `**`) names nothing.
            if trimmed.len() > 200 || !trimmed.chars().any(char::is_alphanumeric) {
                continue;
            }
            let key = trimmed.to_lowercase();
            if seen.insert(key) {
                let entity_type = if looks_like_path(trimmed) {
                    EntityType::File
                } else {
                    EntityType::Code
                };
                entities.push(Entity {
                    name: normalize_entity_name(trimmed),
                    entity_type,
                    source_text: trimmed.to_string(),
                });
            }
        }
    }
}

/// The lines of `text`, each paired with whether it sits inside a fenced
/// code block (```` ``` ```` or `~~~`). The fence lines themselves are left out.
fn lines_with_fences(text: &str) -> Vec<(&str, bool)> {
    let mut lines = Vec::new();
    let mut fence: Option<&str> = None;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let marker = ["```", "~~~"].into_iter().find(|m| trimmed.starts_with(m));
        match (fence, marker) {
            (None, Some(m)) => fence = Some(m),
            (Some(open), Some(m)) if open == m => fence = None,
            _ => lines.push((line, fence.is_some())),
        }
    }
    lines
}

/// The closed inline code spans of one line: `` `a` and `b` `` gives `a`, `b`.
/// An unclosed backtick opens nothing.
fn inline_code_spans(line: &str) -> Vec<&str> {
    let parts: Vec<&str> = line.split('`').collect();
    (1..parts.len().saturating_sub(1))
        .step_by(2)
        .map(|i| parts[i])
        .filter(|span| !span.is_empty())
        .collect()
}

fn extract_file_paths(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    for word in text.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| {
            c == ','
                || c == '.'
                || c == ';'
                || c == ':'
                || c == ')'
                || c == '('
                || c == '"'
                || c == '\''
        });
        if looks_like_path(trimmed) {
            let key = trimmed.to_lowercase();
            if seen.insert(key) {
                entities.push(Entity {
                    name: normalize_file_path(trimmed),
                    entity_type: EntityType::File,
                    source_text: trimmed.to_string(),
                });
            }
        }
    }
}

fn extract_camel_case(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    for word in split_into_identifiers(text) {
        if is_camel_case(word) && word.len() >= 4 {
            let key = word.to_lowercase();
            if seen.insert(key) {
                entities.push(Entity {
                    name: camel_to_snake(word),
                    entity_type: EntityType::Code,
                    source_text: word.to_string(),
                });
            }
        }
    }
}

fn extract_snake_case(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    for word in split_into_identifiers(text) {
        if is_snake_case(word) && word.len() >= 4 && word.contains('_') {
            let key = word.to_lowercase();
            if seen.insert(key) {
                entities.push(Entity {
                    name: word.to_lowercase(),
                    entity_type: EntityType::Code,
                    source_text: word.to_string(),
                });
            }
        }
    }
}

fn extract_quoted_strings(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    // Double-quoted strings
    extract_between(text, '"', '"', entities, seen);
    // Single-quoted strings
    extract_between(text, '\'', '\'', entities, seen);
}

fn extract_between(
    text: &str,
    open: char,
    close: char,
    entities: &mut Vec<Entity>,
    seen: &mut HashSet<String>,
) {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == open {
            let start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != close {
                i += 1;
            }
            if i < chars.len() {
                let content: String = chars[start..i].iter().collect();
                let trimmed = content.trim();
                // Only extract meaningful quoted strings (3+ words)
                let word_count = trimmed.split_whitespace().count();
                if word_count >= 3 && trimmed.len() <= 200 {
                    let key = trimmed.to_lowercase();
                    if seen.insert(key) {
                        entities.push(Entity {
                            name: trimmed.to_string(),
                            entity_type: EntityType::Reference,
                            source_text: trimmed.to_string(),
                        });
                    }
                }
            }
        }
        i += 1;
    }
}

/// Check if a string looks like a file path.
fn looks_like_path(s: &str) -> bool {
    if s.len() < 3 {
        return false;
    }
    // Must start with / or ./ or ../ or contain /
    let has_path_prefix = s.starts_with('/') || s.starts_with("./") || s.starts_with("../");
    let has_extension = s
        .rfind('.')
        .map(|dot| {
            let ext = &s[dot + 1..];
            !ext.is_empty() && ext.len() <= 10 && ext.chars().all(|c| c.is_alphanumeric())
        })
        .unwrap_or(false);

    // Path-like if it has a path prefix, or has both slashes and an extension
    has_path_prefix || (s.contains('/') && has_extension)
}

/// Check if a string is CamelCase (at least 2 uppercase letters).
fn is_camel_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let upper_count = s.chars().filter(|c| c.is_uppercase()).count();
    let has_lower = s.chars().any(|c| c.is_lowercase());
    upper_count >= 2 && has_lower && s.chars().all(|c| c.is_alphanumeric())
}

/// Check if a string is snake_case.
fn is_snake_case(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_')
        && !s.starts_with('_')
        && !s.ends_with('_')
}

/// Convert CamelCase to snake_case, handling consecutive uppercase (acronyms).
///
/// `AuthModule` → `auth_module`, `HTTPClient` → `http_client`
fn camel_to_snake(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 {
            let prev_upper = chars[i - 1].is_uppercase();
            let next_lower = chars.get(i + 1).is_some_and(|nc| nc.is_lowercase());
            // Insert underscore before: start of new word (after lowercase),
            // or before last letter of an acronym followed by lowercase
            if !prev_upper || next_lower {
                result.push('_');
            }
        }
        result.push(c.to_lowercase().next().unwrap_or(c));
    }
    result
}

/// Split text into potential identifiers.
fn split_into_identifiers(text: &str) -> Vec<&str> {
    // Split on whitespace and common delimiters, keeping alphanumeric+underscore tokens
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.len() >= 4)
        .collect()
}

/// Normalize an entity name (lowercase, trim).
fn normalize_entity_name(s: &str) -> String {
    s.trim().to_string()
}

/// Normalize a file path entity (strip leading ./, keep the rest).
fn normalize_file_path(s: &str) -> String {
    s.strip_prefix("./").unwrap_or(s).to_string()
}

// ── Language-aware code-symbol extraction ─────────────────────

use regex::Regex;
use std::sync::LazyLock;

/// A single code-symbol pattern: language hint + compiled regex.
/// Capture group 1 is the symbol name.
struct SymbolPattern {
    lang: &'static str,
    re: Regex,
}

/// Rust-qualified paths (`auth::login`, `Mod::Ty::method`). `::` does not
/// occur in English, so this pattern reads all text.
static QUALIFIED_PATH: LazyLock<SymbolPattern> = LazyLock::new(|| SymbolPattern {
    lang: "rust",
    re: Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b").unwrap(),
});

/// Declaration patterns. Their keywords (`trait`, `class`, `function`,
/// `public`, …) are also English words, so they only read code segments (see
/// [`code_segments`]).
static DECLARATION_PATTERNS: LazyLock<Vec<SymbolPattern>> = LazyLock::new(|| {
    vec![
        // Rust: `fn name`, `struct Name`, `impl Name`, `trait Name`
        SymbolPattern {
            lang: "rust",
            re: Regex::new(r"\b(?:fn|struct|impl|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
                .unwrap(),
        },
        // Python: `def name`, `class Name`
        SymbolPattern {
            lang: "python",
            re: Regex::new(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
        },
        // TypeScript/JavaScript: `function name`, `class Name`
        SymbolPattern {
            lang: "ts",
            re: Regex::new(r"\b(?:function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)").unwrap(),
        },
        // Go: `func name` and `func (recv *T) Method`
        SymbolPattern {
            lang: "go",
            re: Regex::new(r"\bfunc\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
        },
        // Java/Kotlin: `public int name(`, `private Foo name(`
        SymbolPattern {
            lang: "java",
            re: Regex::new(
                r"\b(?:public|private|protected|internal)\s+(?:static\s+)?[A-Za-z_][A-Za-z0-9_<>,\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            )
            .unwrap(),
        },
    ]
});

/// A line that opens with a declaration (`pub fn parse(`, `class User:`,
/// `def run(self):`) and carries code punctuation after it.
static DECLARATION_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:(?:pub(?:\([^)]*\))?|export|async|unsafe|const|public|private|protected|internal|static|abstract|final|override)\s+)*(?:fn|struct|impl|trait|enum|def|class|function|func|interface)\s+\S.*[(){}:<;=]",
    )
    .unwrap()
});

/// English words that follow a declaration keyword in prose ("a trait with
/// …", "the class that …"). Outside fenced blocks a declaration that names
/// one of these is prose, not a symbol.
const PROSE_WORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "but", "by", "can", "could", "did", "do", "does", "each", "every", "for",
    "from", "had", "has", "have", "here", "how", "if", "in", "into", "is", "it", "its", "just",
    "may", "might", "must", "no", "not", "now", "of", "on", "one", "only", "or", "our", "should",
    "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "without", "would", "you", "your",
];

/// A stretch of text the declaration patterns may read.
struct CodeSegment<'a> {
    text: &'a str,
    /// Inside a fenced block: real code, read as is. Anything else may be
    /// prose that only looks like code, so names that are [`PROSE_WORDS`]
    /// are skipped.
    fenced: bool,
    /// From a prose line: a match counts only when its name is shaped like
    /// an identifier (`impl Decoder`, `fn parse_token`) or is called or
    /// parameterized right away (`def login(u)`, `struct Pool<T>`). See
    /// [`prose_declaration_counts`].
    prose: bool,
}

/// Split `text` into the segments the declaration patterns read: lines in
/// fenced blocks, code-like lines, inline code spans, and the remaining prose
/// lines (whose matches must look like code, see [`CodeSegment::prose`]).
fn code_segments(text: &str) -> Vec<CodeSegment<'_>> {
    let mut segments = Vec::new();
    for (line, fenced) in lines_with_fences(text) {
        if fenced || is_code_like_line(line) {
            segments.push(CodeSegment {
                text: line,
                fenced,
                prose: false,
            });
            continue;
        }
        for span in inline_code_spans(line) {
            segments.push(CodeSegment {
                text: span,
                fenced: false,
                prose: false,
            });
        }
        segments.push(CodeSegment {
            text: line,
            fenced: false,
            prose: true,
        });
    }
    segments
}

/// Whether a line outside a fenced block reads as code: it ends a statement
/// or block, or it opens with a declaration followed by code punctuation.
fn is_code_like_line(line: &str) -> bool {
    let t = line.trim();
    t.ends_with('{') || t.ends_with('}') || t.ends_with(';') || DECLARATION_LINE.is_match(t)
}

fn extract_code_symbols(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    // Order matters: qualified paths come first so `auth::login` wins over
    // `login` alone.
    for caps in QUALIFIED_PATH.re.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_code_symbol(QUALIFIED_PATH.lang, m.as_str(), entities, seen);
        }
    }
    let segments = code_segments(text);
    for pattern in DECLARATION_PATTERNS.iter() {
        for segment in &segments {
            for caps in pattern.re.captures_iter(segment.text) {
                let Some(m) = caps.get(1) else { continue };
                let name = m.as_str();
                if segment.prose && !prose_declaration_counts(pattern, segment.text, m) {
                    continue;
                }
                if !segment.fenced && PROSE_WORDS.contains(&name.to_ascii_lowercase().as_str()) {
                    continue;
                }
                push_code_symbol(pattern.lang, name, entities, seen);
            }
        }
    }
}

/// Whether a declaration match on a prose line names code. Prose puts plain
/// words after `trait`, `class` and `function` ("a trait with …"), so the
/// name must be called or parameterized right away (`def login(u)`,
/// `struct Pool<T>`), or be shaped like an identifier: a capital, an
/// underscore or a digit (`impl Decoder`, `fn parse_token`). The Java
/// pattern spans several words before the name, so on prose it only takes
/// the call shape.
fn prose_declaration_counts(pattern: &SymbolPattern, line: &str, name: regex::Match<'_>) -> bool {
    if line[name.end()..].starts_with(['(', '<']) {
        return true;
    }
    pattern.lang != "java"
        && name
            .as_str()
            .chars()
            .any(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

fn push_code_symbol(
    lang: &'static str,
    name: &str,
    entities: &mut Vec<Entity>,
    seen: &mut HashSet<String>,
) {
    if name.len() < 2 || name.len() > 200 {
        return;
    }
    // Collision key is scoped by language so `login` (Rust) and
    // `login` (Python) don't silently merge.
    let key = format!("sym:{}:{}", lang, name.to_lowercase());
    if seen.insert(key) {
        entities.push(Entity {
            name: name.to_string(),
            entity_type: EntityType::CodeSymbol {
                lang_hint: Some(lang.to_string()),
            },
            source_text: name.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_backtick_entities() {
        let text = "Fixed bug in `AuthModule` by updating `auth_config`";
        let entities = extract_entities(text);
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"AuthModule"));
        assert!(names.contains(&"auth_config"));
    }

    #[test]
    fn extract_file_path() {
        let text = "Modified /src/auth/login.rs for the fix";
        let entities = extract_entities(text);
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::File && e.name.contains("auth/login.rs")));
    }

    #[test]
    fn extract_relative_path() {
        let text = "Updated ./src/main.rs with new config";
        let entities = extract_entities(text);
        assert!(entities.iter().any(|e| e.entity_type == EntityType::File));
    }

    #[test]
    fn extract_camel_case_identifier() {
        let text = "The AuthModule handles authentication";
        let entities = extract_entities(text);
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Code && e.name == "auth_module"));
    }

    #[test]
    fn extract_snake_case_identifier() {
        let text = "Updated auth_module for the fix";
        let entities = extract_entities(text);
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Code && e.name == "auth_module"));
    }

    #[test]
    fn extract_quoted_string() {
        let text = "Error message was \"connection timed out unexpectedly\"";
        let entities = extract_entities(text);
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Reference));
    }

    #[test]
    fn no_short_quoted_strings() {
        let text = "Set \"active\" flag";
        let entities = extract_entities(text);
        // "active" is only 1 word, should not be extracted
        assert!(!entities
            .iter()
            .any(|e| e.entity_type == EntityType::Reference));
    }

    #[test]
    fn deduplication() {
        let text = "`AuthModule` is the AuthModule class";
        let entities = extract_entities(text);
        // Should only appear once (backtick version wins)
        let auth_count = entities
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains("authmodule") || e.name.contains("auth_module")
            })
            .count();
        assert_eq!(auth_count, 1);
    }

    #[test]
    fn camel_to_snake_basic() {
        assert_eq!(camel_to_snake("AuthModule"), "auth_module");
        assert_eq!(camel_to_snake("HTTPClient"), "http_client");
    }

    #[test]
    fn empty_text_no_entities() {
        let entities = extract_entities("");
        assert!(entities.is_empty());
    }

    // ── tests ────────────────────────────────────────────────

    fn code_syms(text: &str) -> Vec<(String, Option<String>)> {
        extract_entities(text)
            .into_iter()
            .filter_map(|e| match e.entity_type {
                EntityType::CodeSymbol { lang_hint } => Some((e.name, lang_hint)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn extract_rust_qualified_symbol() {
        let syms = code_syms("Fixed the timeout in auth::login after refactor");
        assert!(syms
            .iter()
            .any(|(n, l)| n == "auth::login" && l.as_deref() == Some("rust")));
    }

    #[test]
    fn extract_rust_fn_struct() {
        let syms = code_syms("fn parse_token builds struct TokenRing via impl Decoder");
        assert!(syms.iter().any(|(n, _)| n == "parse_token"));
        assert!(syms.iter().any(|(n, _)| n == "TokenRing"));
        assert!(syms.iter().any(|(n, _)| n == "Decoder"));
    }

    #[test]
    fn prose_mints_no_declaration_symbols() {
        for text in [
            "Made it a trait with a default method",
            "the class that owns the pool is gone",
            "a function returns early when the struct is empty",
            "impl the fix in the enum handling, then def the plan",
            "the public API and private helpers for doing (x)",
            "the public API lives in Store (see the docs)",
        ] {
            let syms = code_syms(text);
            assert!(syms.is_empty(), "{text:?} minted {syms:?}");
        }
    }

    #[test]
    fn checklist_syntax_mints_nothing() {
        let entities = extract_entities(
            "- [ ] write tests\n- [x] ship it\nUse `- [ ]` for open items, `**` for bold",
        );
        assert!(entities.is_empty(), "minted {entities:?}");
    }

    #[test]
    fn fenced_code_reads_every_declaration() {
        let syms = code_syms(
            "Builder:\n```rust\npub trait Store\nfn with(self) -> Self\n```\nThe trait with it.",
        );
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Store" && l.as_deref() == Some("rust")));
        // Real code may name a method `with`; only prose loses it.
        assert!(syms.iter().any(|(n, _)| n == "with"));
        assert_eq!(syms.len(), 2, "{syms:?}");
    }

    #[test]
    fn fenced_block_is_not_a_backtick_entity() {
        let entities =
            extract_entities("Fix:\n```rust\nlet pool_size = 32;\n```\nThen `PoolConfig`.");
        assert!(
            entities.iter().all(|e| !e.name.contains('\n')),
            "a fenced block became an entity: {entities:?}"
        );
        assert!(entities.iter().any(|e| e.name == "PoolConfig"));
    }

    #[test]
    fn code_like_lines_read_outside_fences() {
        let syms = code_syms(
            "Signature now:\n    pub fn open(path: &Path) -> Result<Self> {\nclass Pool:",
        );
        assert!(syms
            .iter()
            .any(|(n, l)| n == "open" && l.as_deref() == Some("rust")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Pool" && l.as_deref() == Some("python")));
    }

    #[test]
    fn prose_reads_declarations_that_look_like_code() {
        let syms = code_syms(
            "Call def login(u) first; the struct Pool<T> holds them; a trait with (care). \
             Then impl Adapter for HttpAdapter and fn parse_v2 in the class that loads it.",
        );
        assert!(syms
            .iter()
            .any(|(n, l)| n == "login" && l.as_deref() == Some("python")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Pool" && l.as_deref() == Some("rust")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Adapter" && l.as_deref() == Some("rust")));
        assert!(syms.iter().any(|(n, _)| n == "parse_v2"));
        for word in ["with", "that"] {
            assert!(!syms.iter().any(|(n, _)| n == word), "{word}: {syms:?}");
        }
    }

    #[test]
    fn inline_code_skips_prose_words() {
        let syms = code_syms("Wrote `impl with` by mistake, meant `impl Store`");
        assert!(syms.iter().any(|(n, _)| n == "Store"));
        assert!(!syms.iter().any(|(n, _)| n == "with"), "{syms:?}");
    }

    #[test]
    fn extract_python_def_class() {
        let syms = code_syms("def login(user): pass\nclass UserManager: pass");
        assert!(syms
            .iter()
            .any(|(n, l)| n == "login" && l.as_deref() == Some("python")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "UserManager" && l.as_deref() == Some("python")));
    }

    #[test]
    fn extract_ts_function_class() {
        let syms = code_syms("function createSession() {}\nclass Router {}");
        assert!(syms
            .iter()
            .any(|(n, l)| n == "createSession" && l.as_deref() == Some("ts")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Router" && l.as_deref() == Some("ts")));
    }

    #[test]
    fn extract_go_func_and_method() {
        let syms = code_syms("func Login(u string) error {}\nfunc (s *Server) Serve() {}");
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Login" && l.as_deref() == Some("go")));
        assert!(syms
            .iter()
            .any(|(n, l)| n == "Serve" && l.as_deref() == Some("go")));
    }

    #[test]
    fn cross_language_login_distinct_keys() {
        // Same display name 'login' appearing via different language
        // patterns must both be captured (lang_hint disambiguates).
        let syms = code_syms("In Python: def login(u)\nIn Rust: auth::login");
        let py = syms
            .iter()
            .find(|(n, l)| n == "login" && l.as_deref() == Some("python"));
        let rs = syms
            .iter()
            .find(|(n, l)| n == "auth::login" && l.as_deref() == Some("rust"));
        assert!(py.is_some(), "python login missing");
        assert!(rs.is_some(), "rust auth::login missing");
    }

    #[test]
    fn provisional_canonical_id_is_stable_and_scoped() {
        let a = provisional_canonical_id("login", Some("rust"), Some("src/auth.rs"));
        let b = provisional_canonical_id("login", Some("rust"), Some("src/auth.rs"));
        let c = provisional_canonical_id("login", Some("python"), Some("src/auth.rs"));
        let d = provisional_canonical_id("login", Some("rust"), Some("src/other.rs"));
        assert_eq!(a, b, "same tuple must produce same id");
        assert_ne!(a, c, "language scope must disambiguate");
        assert_ne!(a, d, "file scope must disambiguate");
        assert!(a.starts_with("provisional:"));
    }

    /// Pin the JSON shape of `EntityType::CodeSymbol`. `axil-scip`'s
    /// provisional-upgrade loader reads `entity_type.code_symbol.lang_hint`;
    /// if serde ever stops snake-casing this variant, the upgrade path
    /// would silently start missing all `(name, lang)` matches.
    #[test]
    fn code_symbol_serializes_as_snake_case() {
        let v = serde_json::to_value(EntityType::CodeSymbol {
            lang_hint: Some("rust".to_string()),
        })
        .unwrap();
        let obj = v.as_object().expect("variant serializes to an object");
        assert!(
            obj.contains_key("code_symbol"),
            "expected `code_symbol` key; got {v}",
        );
        let inner = obj.get("code_symbol").and_then(|v| v.as_object()).unwrap();
        assert_eq!(
            inner.get("lang_hint").and_then(|v| v.as_str()),
            Some("rust"),
        );
    }
}
