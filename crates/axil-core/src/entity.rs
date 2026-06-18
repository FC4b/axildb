//! Pattern-based entity extraction — no LLM required.
//!
//! Extracts entities from record text using deterministic patterns:
//! file paths, CamelCase/snake_case identifiers, backtick-wrapped code,
//! quoted strings, and project names. Also handles Phase 13 code symbols
//! (`fn foo`, `auth::login`, `class User`, …) with a language hint so
//! cross-language collisions can be disambiguated downstream.

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
    let mut start = 0;
    while let Some(open) = text[start..].find('`') {
        let open_pos = start + open + 1;
        if open_pos >= text.len() {
            break;
        }
        // Skip double/triple backticks (code blocks)
        if text[open_pos..].starts_with('`') {
            start = open_pos + 1;
            continue;
        }
        if let Some(close) = text[open_pos..].find('`') {
            let content = &text[open_pos..open_pos + close];
            let trimmed = content.trim();
            if !trimmed.is_empty() && trimmed.len() <= 200 {
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
            start = open_pos + close + 1;
        } else {
            break;
        }
    }
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

static CODE_SYMBOL_PATTERNS: LazyLock<Vec<SymbolPattern>> = LazyLock::new(|| {
    // Order matters: Rust-qualified paths must come before bare fn/struct so
    // `auth::login` wins over `login` alone.
    vec![
        // Rust: qualified path `auth::login`, `Mod::Ty::method`
        SymbolPattern {
            lang: "rust",
            re: Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b").unwrap(),
        },
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

fn extract_code_symbols(text: &str, entities: &mut Vec<Entity>, seen: &mut HashSet<String>) {
    for pattern in CODE_SYMBOL_PATTERNS.iter() {
        for caps in pattern.re.captures_iter(text) {
            let Some(m) = caps.get(1) else { continue };
            let name = m.as_str();
            if name.len() < 2 || name.len() > 200 {
                continue;
            }
            // Collision key is scoped by language so `login` (Rust) and
            // `login` (Python) don't silently merge.
            let key = format!("sym:{}:{}", pattern.lang, name.to_lowercase());
            if seen.insert(key) {
                entities.push(Entity {
                    name: name.to_string(),
                    entity_type: EntityType::CodeSymbol {
                        lang_hint: Some(pattern.lang.to_string()),
                    },
                    source_text: name.to_string(),
                });
            }
        }
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

    // ── Phase 13.1 tests ────────────────────────────────────────────────

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
