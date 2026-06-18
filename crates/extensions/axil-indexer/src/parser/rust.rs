//! Rust source file parser.
//!
//! Extracts pub fn, pub struct, pub enum, pub trait, mod, use statements,
//! doc comments (/// and //!), and detects patterns like error types,
//! trait impls, and derive macros.

use regex::Regex;
use std::sync::LazyLock;

use super::{ParsedFile, ParsedSymbol, SymbolKind};

static RE_MODULE_DOC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^//! (.+)$").unwrap());

static RE_PUB_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*pub(?:\(crate\))?\s+(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^\{]+))?\s*\{").unwrap()
});

static RE_PUB_STRUCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*pub\s+struct\s+(\w+)").unwrap());

static RE_PUB_ENUM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*pub\s+enum\s+(\w+)").unwrap());

static RE_PUB_TRAIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*pub\s+trait\s+(\w+)").unwrap());

static RE_PUB_TYPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*pub\s+type\s+(\w+)").unwrap());

static RE_PUB_CONST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[ \t]*pub\s+(?:const|static)\s+(\w+)").unwrap());

static RE_USE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^use\s+([\w:]+)").unwrap());

static RE_IMPL_TRAIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^impl(?:<[^>]*>)?\s+(\w+)\s+for\s+(\w+)").unwrap());

static RE_DERIVE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"#\[derive\(([^)]+)\)\]").unwrap());

static RE_PRIV_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^\{]+))?\s*\{").unwrap()
});

/// Matches an `impl` header, capturing the Self type (group 1) — the type
/// after `for` for a trait impl, otherwise the inherent-impl type. Generics
/// and lifetimes on either side are skipped.
static RE_IMPL_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^impl(?:<[^>]*>)?\s+(?:[\w:]+(?:<[^>]*>)?\s+for\s+)?(\w+)").unwrap()
});

/// Matches a method/function name anywhere (used to harvest a digest from a
/// trait or impl body).
static RE_FN_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bfn\s+(\w+)").unwrap());

pub fn parse(source: &str, include_private: bool) -> ParsedFile {
    let mut file = ParsedFile::default();
    let lines: Vec<&str> = source.lines().collect();

    // Module-level doc comments (//!)
    let mut module_docs = Vec::new();
    for cap in RE_MODULE_DOC.captures_iter(source) {
        module_docs.push(cap[1].to_string());
    }
    if !module_docs.is_empty() {
        file.module_doc = Some(module_docs.join(" "));
    }

    // Collect doc comments above each line for lookup
    let doc_map = build_doc_map(&lines);

    // Map each type → method names from its impl blocks, so a no-doc-comment
    // struct/enum embeds more than its breadcrumb. Method names (validate,
    // resolve, as_sql) are the concept terms conceptual queries actually use.
    let impl_methods = build_impl_method_map(source);

    // Public functions
    for cap in RE_PUB_FN.captures_iter(source) {
        let name = cap[1].to_string();
        let params = cap[2].trim().to_string();
        let ret = cap.get(3).map(|m| m.as_str().trim().to_string());
        let line = find_line(source, cap.get(0).unwrap().start());

        let sig = format_fn_sig(&name, &params, ret.as_deref());
        let doc = doc_map.get(&line).cloned();

        file.exports.push(name.clone());
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Function,
            line,
            signature: sig,
            doc,
        });
    }

    // Private functions (if requested)
    if include_private {
        for cap in RE_PRIV_FN.captures_iter(source) {
            let full_match = cap.get(0).unwrap().as_str();
            // Skip if it's actually a pub fn (already captured above)
            if full_match.trim_start().starts_with("pub") {
                continue;
            }
            let name = cap[1].to_string();
            let params = cap[2].trim().to_string();
            let ret = cap.get(3).map(|m| m.as_str().trim().to_string());
            let line = find_line(source, cap.get(0).unwrap().start());
            let sig = format_fn_sig(&name, &params, ret.as_deref());
            let doc = doc_map.get(&line).cloned();

            file.symbols.push(ParsedSymbol {
                name,
                kind: SymbolKind::Function,
                line,
                signature: sig,
                doc,
            });
        }
    }

    // Structs
    for cap in RE_PUB_STRUCT.captures_iter(source) {
        let name = cap[1].to_string();
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = doc_map.get(&line).cloned();
        let signature = type_signature("struct", &name, impl_methods.get(&name));
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} struct"));
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Struct,
            line,
            signature,
            doc,
        });
    }

    // Enums
    for cap in RE_PUB_ENUM.captures_iter(source) {
        let name = cap[1].to_string();
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = doc_map.get(&line).cloned();
        let signature = type_signature("enum", &name, impl_methods.get(&name));
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} enum"));
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Enum,
            line,
            signature,
            doc,
        });
    }

    // Traits — harvest the method digest from the trait body itself.
    for cap in RE_PUB_TRAIT.captures_iter(source) {
        let name = cap[1].to_string();
        let m0 = cap.get(0).unwrap();
        let line = find_line(source, m0.start());
        let doc = doc_map.get(&line).cloned();
        let trait_methods = brace_match_body(source, m0.start()).map(collect_fn_names);
        let signature = type_signature("trait", &name, trait_methods.as_ref());
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} trait"));
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Trait,
            line,
            signature,
            doc,
        });
    }

    // Type aliases
    for cap in RE_PUB_TYPE.captures_iter(source) {
        let name = cap[1].to_string();
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} type"));
    }

    // Constants
    for cap in RE_PUB_CONST.captures_iter(source) {
        let name = cap[1].to_string();
        file.exports.push(name);
    }

    // Imports (use statements)
    for cap in RE_USE.captures_iter(source) {
        let path = cap[1].to_string();
        // Extract the crate name (first segment)
        if let Some(crate_name) = path.split("::").next() {
            if !["crate", "self", "super"].contains(&crate_name)
                && !file.imports.contains(&crate_name.to_string())
            {
                file.imports.push(crate_name.to_string());
            }
        }
    }

    // Pattern detection
    detect_patterns(source, &mut file);

    // Generate summary
    file.summary = generate_summary(&file, &lines);

    file
}

fn format_fn_sig(name: &str, params: &str, ret: Option<&str>) -> String {
    let params_clean = params
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ");
    match ret {
        Some(r) => format!("fn {name}({params_clean}) -> {r}"),
        None => format!("fn {name}({params_clean})"),
    }
}

/// Build a map of line number → accumulated doc comment above that line.
fn build_doc_map(lines: &[&str]) -> std::collections::HashMap<usize, String> {
    let mut map = std::collections::HashMap::new();
    let mut doc_lines: Vec<String> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(doc) = trimmed.strip_prefix("/// ") {
            doc_lines.push(doc.to_string());
        } else if trimmed == "///" {
            doc_lines.push(String::new());
        } else if !doc_lines.is_empty() {
            // Attach doc to this line (1-indexed)
            let doc = doc_lines.join(" ").trim().to_string();
            if !doc.is_empty() {
                map.insert(i + 1, doc);
            }
            doc_lines.clear();
        }
    }
    map
}

use super::{brace_match_body, find_line};

/// Build a type signature enriched with a method-name digest. When the type has
/// no associated methods the bare `kind name` head is used (still richer than
/// the old empty signature). Mirrors the Python parser's class digest.
fn type_signature(kind: &str, name: &str, methods: Option<&Vec<String>>) -> String {
    let head = format!("{kind} {name}");
    match methods {
        Some(m) if !m.is_empty() => {
            let top: Vec<&str> = m.iter().take(12).map(String::as_str).collect();
            format!("{head} — methods: {}", top.join(", "))
        }
        _ => head,
    }
}

/// Map each Self type to the method names declared across its `impl` blocks
/// (inherent + trait impls) in this file. Cross-file impls are not seen — a
/// per-file parser limitation that's fine for a keyword digest.
fn build_impl_method_map(source: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for cap in RE_IMPL_HEADER.captures_iter(source) {
        let self_ty = cap[1].to_string();
        if let Some(body) = brace_match_body(source, cap.get(0).unwrap().start()) {
            let entry = map.entry(self_ty).or_default();
            for name in collect_fn_names(body) {
                if !entry.contains(&name) {
                    entry.push(name);
                }
            }
        }
    }
    map
}

/// Collect deduped function/method names from a trait or impl body, in source
/// order. Skips `__`-style and operator names are not a concern in Rust.
fn collect_fn_names(body: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cap in RE_FN_NAME.captures_iter(body) {
        let name = cap[1].to_string();
        if seen.insert(name.clone()) {
            names.push(name);
        }
    }
    names
}

fn detect_patterns(source: &str, file: &mut ParsedFile) {
    // Error handling patterns
    if source.contains("thiserror") || source.contains("#[error(") {
        file.patterns.push("error_handling".to_string());
    }

    // HTTP handler patterns
    if source.contains("#[get(")
        || source.contains("#[post(")
        || source.contains("#[put(")
        || source.contains("#[delete(")
        || source.contains("HttpRequest")
        || source.contains("HttpResponse")
        || source.contains("axum::")
        || source.contains("actix_web")
    {
        file.patterns.push("http_handler".to_string());
    }

    // Database patterns
    if source.contains("sqlx::")
        || source.contains("diesel::")
        || source.contains("sea_orm")
        || source.contains("rusqlite")
    {
        file.patterns.push("database".to_string());
    }

    // Test patterns
    if source.contains("#[test]") || source.contains("#[tokio::test]") {
        file.patterns.push("tests".to_string());
    }

    // Trait impls
    for cap in RE_IMPL_TRAIT.captures_iter(source) {
        let trait_name = cap[1].to_string();
        let type_name = cap[2].to_string();
        file.patterns
            .push(format!("impl {trait_name} for {type_name}"));
    }

    // Derive macros
    for cap in RE_DERIVE.captures_iter(source) {
        let derives: Vec<&str> = cap[1].split(',').map(|s| s.trim()).collect();
        for d in derives {
            if ["Serialize", "Deserialize", "Clone", "Debug"].contains(&d) {
                continue; // Too common to track
            }
            file.patterns.push(format!("derive({d})"));
        }
    }
}

const RUST_PATTERN_LABELS: &[(&str, &str)] = &[
    ("error_handling", "error types"),
    ("http_handler", "HTTP request handlers"),
    ("database", "database queries"),
    ("tests", "tests"),
];

fn generate_summary(file: &ParsedFile, lines: &[&str]) -> String {
    super::generate_summary_common(file, RUST_PATTERN_LABELS, || {
        // Rust-specific fallback: first comment, then line count
        for line in lines.iter().take(20) {
            let trimmed = line.trim();
            if let Some(comment) = trimmed.strip_prefix("// ") {
                if !comment.is_empty() && !comment.starts_with("---") && !comment.starts_with("──")
                {
                    return comment.to_string();
                }
            }
        }
        format!("{} lines of Rust code", lines.len())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pub_fn() {
        let src = r#"
/// Validates a JWT token.
pub fn validate_token(token: &str, secret: &[u8]) -> Result<Claims, AuthError> {
    todo!()
}
"#;
        let result = parse(src, false);
        assert!(result.exports.contains(&"validate_token".to_string()));
        assert_eq!(result.symbols.len(), 1);
        assert_eq!(result.symbols[0].kind, SymbolKind::Function);
        assert!(result.symbols[0].doc.as_ref().unwrap().contains("JWT"));
    }

    #[test]
    fn parse_pub_struct_and_enum() {
        let src = r#"
/// User claims from JWT.
pub struct Claims {
    pub user_id: String,
}

pub enum AuthError {
    Expired,
    Invalid,
}
"#;
        let result = parse(src, false);
        assert!(result.exports.contains(&"Claims".to_string()));
        assert!(result.exports.contains(&"AuthError".to_string()));
        assert!(result.key_types.contains(&"Claims struct".to_string()));
        assert!(result.key_types.contains(&"AuthError enum".to_string()));
    }

    #[test]
    fn parse_module_doc() {
        let src = r#"//! JWT auth middleware — validates tokens and extracts claims.

pub fn validate() {}
"#;
        let result = parse(src, false);
        assert!(result.module_doc.is_some());
        assert!(result.summary.contains("JWT auth middleware"));
    }

    #[test]
    fn parse_imports() {
        let src = r#"
use std::path::Path;
use serde::Serialize;
use crate::db;
"#;
        let result = parse(src, false);
        assert!(result.imports.contains(&"std".to_string()));
        assert!(result.imports.contains(&"serde".to_string()));
        assert!(!result.imports.contains(&"crate".to_string()));
    }

    #[test]
    fn struct_signature_includes_impl_method_digest() {
        let src = r#"
pub struct SqlCompiler {
    dialect: String,
}

impl SqlCompiler {
    pub fn compile(&self, q: &Query) -> String { String::new() }
    fn as_sql(&self) -> String { String::new() }
}
"#;
        let result = parse(src, false);
        let s = result.symbols.iter().find(|s| s.name == "SqlCompiler").unwrap();
        assert!(s.signature.contains("methods:"), "got: {}", s.signature);
        assert!(s.signature.contains("compile"));
        assert!(s.signature.contains("as_sql"));
    }

    #[test]
    fn struct_signature_includes_trait_impl_methods() {
        let src = r#"
pub struct Foo;

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut Formatter) -> Result { Ok(()) }
}
"#;
        let result = parse(src, false);
        let s = result.symbols.iter().find(|s| s.name == "Foo").unwrap();
        assert!(s.signature.contains("fmt"), "got: {}", s.signature);
    }

    #[test]
    fn trait_signature_includes_method_digest() {
        let src = r#"
pub trait Resolver {
    fn resolve(&self, url: &str) -> Match;
    fn reverse(&self, name: &str) -> String;
}
"#;
        let result = parse(src, false);
        let t = result.symbols.iter().find(|s| s.name == "Resolver").unwrap();
        assert!(t.signature.contains("resolve"), "got: {}", t.signature);
        assert!(t.signature.contains("reverse"));
    }

    #[test]
    fn type_without_methods_has_bare_signature() {
        let src = "pub struct Claims { pub user_id: String }";
        let result = parse(src, false);
        let s = result.symbols.iter().find(|s| s.name == "Claims").unwrap();
        assert_eq!(s.signature, "struct Claims");
    }

    #[test]
    fn detect_error_handling() {
        let src = r#"
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("not found")]
    NotFound,
}
"#;
        let result = parse(src, false);
        assert!(result.patterns.contains(&"error_handling".to_string()));
    }
}
