//! TypeScript/JavaScript source file parser.
//!
//! Extracts export function, export class, export interface, export type,
//! JSDoc comments, and detects patterns like React components, API routes,
//! and middleware.

use regex::Regex;
use std::sync::LazyLock;

use super::{ParsedFile, ParsedSymbol, SymbolKind};

static RE_EXPORT_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^export\s+(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?\s*\{").unwrap()
});

static RE_EXPORT_CONST_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^export\s+const\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\(?").unwrap()
});

static RE_EXPORT_CLASS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^export\s+(?:default\s+)?class\s+(\w+)").unwrap());

static RE_EXPORT_INTERFACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^export\s+(?:default\s+)?interface\s+(\w+)").unwrap());

static RE_EXPORT_TYPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^export\s+type\s+(\w+)").unwrap());

static RE_EXPORT_ENUM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^export\s+(?:const\s+)?enum\s+(\w+)").unwrap());

static RE_IMPORT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?m)^import\s+.*?\s+from\s+['"]([^'"]+)['"]"#).unwrap());

static RE_JSDOC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/\*\*\s*\n([\s\S]*?)\*/").unwrap());

pub fn parse(source: &str, _include_private: bool) -> ParsedFile {
    let mut file = ParsedFile::default();
    let lines: Vec<&str> = source.lines().collect();

    // JSDoc comments map: line → doc
    let doc_map = build_jsdoc_map(source);

    // Export functions
    for cap in RE_EXPORT_FN.captures_iter(source) {
        let name = cap[1].to_string();
        let params = cap[2].trim().to_string();
        let ret = cap.get(3).map(|m| m.as_str().trim().to_string());
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = find_doc(&doc_map, line);

        let sig = match ret {
            Some(r) => format!("function {name}({params}): {r}"),
            None => format!("function {name}({params})"),
        };

        file.exports.push(name.clone());
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Function,
            line,
            signature: sig,
            doc,
        });
    }

    // Export const (arrow functions, etc.)
    for cap in RE_EXPORT_CONST_FN.captures_iter(source) {
        let name = cap[1].to_string();
        if !file.exports.contains(&name) {
            let line = find_line(source, cap.get(0).unwrap().start());
            let doc = find_doc(&doc_map, line);
            file.exports.push(name.clone());
            file.symbols.push(ParsedSymbol {
                name,
                kind: SymbolKind::Function,
                line,
                signature: String::new(),
                doc,
            });
        }
    }

    // Classes
    for cap in RE_EXPORT_CLASS.captures_iter(source) {
        let name = cap[1].to_string();
        let m0 = cap.get(0).unwrap();
        let line = find_line(source, m0.start());
        let doc = find_doc(&doc_map, line);
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} class"));
        file.symbols.push(ParsedSymbol {
            name: name.clone(),
            kind: SymbolKind::Class,
            line,
            signature: member_signature(&format!("class {name}"), source, m0.start()),
            doc,
        });
    }

    // Interfaces
    for cap in RE_EXPORT_INTERFACE.captures_iter(source) {
        let name = cap[1].to_string();
        let m0 = cap.get(0).unwrap();
        let line = find_line(source, m0.start());
        let doc = find_doc(&doc_map, line);
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} interface"));
        file.symbols.push(ParsedSymbol {
            name: name.clone(),
            kind: SymbolKind::Interface,
            line,
            signature: member_signature(&format!("interface {name}"), source, m0.start()),
            doc,
        });
    }

    // Type aliases
    for cap in RE_EXPORT_TYPE.captures_iter(source) {
        let name = cap[1].to_string();
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} type"));
    }

    // Enums
    for cap in RE_EXPORT_ENUM.captures_iter(source) {
        let name = cap[1].to_string();
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = find_doc(&doc_map, line);
        file.exports.push(name.clone());
        file.key_types.push(format!("{name} enum"));
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Enum,
            line,
            signature: String::new(),
            doc,
        });
    }

    // Imports
    for cap in RE_IMPORT.captures_iter(source) {
        let module = cap[1].to_string();
        // Skip relative imports
        if !module.starts_with('.') && !file.imports.contains(&module) {
            file.imports.push(module);
        }
    }

    // File-level comment (first few lines)
    for line in lines.iter().take(5) {
        let trimmed = line.trim();
        if let Some(comment) = trimmed.strip_prefix("// ") {
            if file.module_doc.is_none() && !comment.is_empty() {
                file.module_doc = Some(comment.to_string());
            }
        } else if trimmed.starts_with("/**") {
            // Already handled by JSDoc
            break;
        } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
            break;
        }
    }

    // Pattern detection
    detect_patterns(source, &mut file);

    file.summary = generate_summary(&file);
    file
}

fn build_jsdoc_map(source: &str) -> Vec<(usize, String)> {
    let mut map = Vec::new();
    for cap in RE_JSDOC.captures_iter(source) {
        let doc_text = &cap[1];
        let end_offset = cap.get(0).unwrap().end();
        let line = find_line(source, end_offset);

        // Clean JSDoc: remove leading * and @tags
        let clean: Vec<&str> = doc_text
            .lines()
            .map(|l| l.trim().trim_start_matches('*').trim())
            .filter(|l| !l.is_empty() && !l.starts_with('@'))
            .collect();

        if !clean.is_empty() {
            map.push((line, clean.join(" ")));
        }
    }
    map
}

fn find_doc(doc_map: &[(usize, String)], target_line: usize) -> Option<String> {
    // Find the closest doc comment that ends just before this line
    doc_map
        .iter()
        .rev()
        .find(|(line, _)| *line <= target_line && target_line - *line <= 2)
        .map(|(_, doc)| doc.clone())
}

use super::{brace_match_body, find_line};

/// Build a class/interface signature enriched with a member-name digest, so a
/// type with no JSDoc embeds more than its breadcrumb. Method and field names
/// (`validateToken`, `resolve`, `userId`) are the concept terms conceptual
/// queries actually use. Mirrors the Python parser's class digest.
fn member_signature(head: &str, source: &str, decl_offset: usize) -> String {
    let members = ts_member_names(source, decl_offset, 12);
    if members.is_empty() {
        head.to_string()
    } else {
        format!("{head} — members: {}", members.join(", "))
    }
}

/// Names of methods/fields declared directly inside a class or interface body
/// (one nesting level deep — method bodies are skipped). Naive but sufficient
/// for a keyword digest.
fn ts_member_names(source: &str, decl_offset: usize, max: usize) -> Vec<String> {
    let Some(body) = brace_match_body(source, decl_offset) else {
        return Vec::new();
    };
    let mut depth = 0i32;
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in body.lines() {
        let start_depth = depth;
        for c in line.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        // Only harvest members at the body's top level, not inside a method.
        if start_depth == 0 {
            if let Some(name) = leading_member_name(line) {
                if seen.insert(name.clone()) {
                    out.push(name);
                    if out.len() >= max {
                        break;
                    }
                }
            }
        }
    }
    out
}

/// Extract the leading member name from a class/interface body line, or `None`
/// if the line isn't a member declaration. Strips access/modifier keywords,
/// then accepts an identifier immediately followed by `(`, `<`, `:`, `?`, or
/// `=` (method, generic method, field, optional field, or arrow-fn field).
fn leading_member_name(line: &str) -> Option<String> {
    let mut t = line.trim_start();
    if t.is_empty()
        || t.starts_with("//")
        || t.starts_with('/')
        || t.starts_with('*')
        || t.starts_with('}')
        || t.starts_with('@')
    {
        return None;
    }
    // Strip leading modifier keywords (each followed by whitespace).
    loop {
        let stripped = ["public", "private", "protected", "readonly", "static", "abstract",
            "async", "override", "declare", "get", "set"]
            .iter()
            .find_map(|kw| {
                t.strip_prefix(kw)
                    .filter(|rest| rest.starts_with(char::is_whitespace))
                    .map(str::trim_start)
            });
        match stripped {
            Some(rest) => t = rest,
            None => break,
        }
    }
    let name: String = t
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect();
    if name.is_empty() || name == "constructor" {
        return None;
    }
    // Reject control-flow keywords that can appear at body top level (rare).
    if matches!(
        name.as_str(),
        "if" | "for" | "while" | "switch" | "return" | "else" | "do" | "try" | "catch"
    ) {
        return None;
    }
    let after = t[name.len()..].trim_start();
    let next = after.chars().next()?;
    if matches!(next, '(' | '<' | ':' | '?' | '=' | '!') {
        Some(name)
    } else {
        None
    }
}

fn detect_patterns(source: &str, file: &mut ParsedFile) {
    // React patterns
    if source.contains("React.")
        || source.contains("useState")
        || source.contains("useEffect")
        || source.contains("jsx")
        || source.contains("tsx")
    {
        file.patterns.push("react_component".to_string());
    }

    // API routes (Next.js, Express, etc.)
    if source.contains("app.get(")
        || source.contains("app.post(")
        || source.contains("router.get(")
        || source.contains("router.post(")
        || source.contains("export async function GET")
        || source.contains("export async function POST")
    {
        file.patterns.push("api_route".to_string());
    }

    // Middleware
    if source.contains("middleware")
        || source.contains("(req, res, next)")
        || source.contains("NextResponse")
    {
        file.patterns.push("middleware".to_string());
    }

    // Tests
    if source.contains("describe(")
        || source.contains("it(")
        || source.contains("test(")
        || source.contains("expect(")
    {
        file.patterns.push("tests".to_string());
    }
}

const TS_PATTERN_LABELS: &[(&str, &str)] = &[
    ("react_component", "React component"),
    ("api_route", "API route handler"),
    ("middleware", "middleware"),
    ("tests", "tests"),
];

fn generate_summary(file: &ParsedFile) -> String {
    super::generate_summary_common(file, TS_PATTERN_LABELS, || {
        "no summary available".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_export_function() {
        let src = r#"
/**
 * Validates user authentication token.
 */
export async function validateToken(token: string): Promise<Claims> {
    return verify(token);
}
"#;
        let result = parse(src, false);
        assert!(result.exports.contains(&"validateToken".to_string()));
        assert_eq!(result.symbols.len(), 1);
        assert_eq!(result.symbols[0].kind, SymbolKind::Function);
    }

    #[test]
    fn parse_export_class() {
        let src = r#"
export class AuthService {
    validate() {}
}
"#;
        let result = parse(src, false);
        assert!(result.exports.contains(&"AuthService".to_string()));
        assert!(result.key_types.contains(&"AuthService class".to_string()));
    }

    #[test]
    fn class_signature_includes_member_digest() {
        let src = r#"
export class AuthService {
    private store: Store;
    async validateToken(token: string): Promise<Claims> { return verify(token); }
    refresh() {}
}
"#;
        let result = parse(src, false);
        let c = result.symbols.iter().find(|s| s.name == "AuthService").unwrap();
        assert!(c.signature.contains("members:"), "got: {}", c.signature);
        assert!(c.signature.contains("validateToken"));
        assert!(c.signature.contains("refresh"));
        assert!(c.signature.contains("store"));
        // a call inside a method body must not leak into the digest
        assert!(!c.signature.contains("verify"), "got: {}", c.signature);
    }

    #[test]
    fn interface_signature_includes_member_digest() {
        let src = r#"
export interface Resolver {
    resolve(url: string): Match;
    pattern: string;
}
"#;
        let result = parse(src, false);
        let i = result.symbols.iter().find(|s| s.name == "Resolver").unwrap();
        assert!(i.signature.contains("resolve"), "got: {}", i.signature);
        assert!(i.signature.contains("pattern"));
    }

    #[test]
    fn parse_imports() {
        let src = r#"
import express from 'express';
import { Router } from 'express';
import './styles.css';
"#;
        let result = parse(src, false);
        assert!(result.imports.contains(&"express".to_string()));
        assert!(!result.imports.iter().any(|i| i.starts_with('.')));
    }

    #[test]
    fn detect_react() {
        let src = "import React from 'react';\nconst [state, setState] = useState(0);";
        let result = parse(src, false);
        assert!(result.patterns.contains(&"react_component".to_string()));
    }
}
