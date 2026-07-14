//! Python source file parser.
//!
//! Extracts def, class, __all__, docstrings, type hints,
//! and detects patterns like FastAPI routes, Django views, and dataclasses.

use regex::Regex;
use std::sync::LazyLock;

use super::{ParsedFile, ParsedSymbol, SymbolKind};

static RE_CLASS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^class\s+(\w+)(\([^)]*\))?:").unwrap());

static RE_DEF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?\s*:").unwrap());

static RE_ASYNC_DEF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^async\s+def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?\s*:").unwrap()
});

static RE_ALL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?m)^__all__\s*=\s*\[([^\]]*)\]"#).unwrap());

static RE_IMPORT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(?:from\s+(\S+)\s+)?import\s+(\S+)").unwrap());

pub fn parse(source: &str, include_private: bool) -> ParsedFile {
    let mut file = ParsedFile::default();
    let lines: Vec<&str> = source.lines().collect();

    // Module-level docstring (first non-comment, non-empty line)
    file.module_doc = extract_module_docstring(&lines);

    // __all__ list
    let explicit_exports: Vec<String> = extract_all_list(source);

    // Classes
    for cap in RE_CLASS.captures_iter(source) {
        let name = cap[1].to_string();
        if !include_private && name.starts_with('_') {
            continue;
        }
        let m0 = cap.get(0).unwrap();
        let line = find_line(source, m0.start());
        let doc = extract_docstring_after(source, m0.end());

        // Give the class proxy real matchable surface: its base classes and
        // a method-name digest. A no-docstring class would otherwise embed
        // as just its breadcrumb (13 tokens), so conceptual queries like
        // "url resolver resolve view" never rank it. Method names ARE the
        // concept terms (resolve, compile, as_sql, ...). See the context-ab
        // recall-quality experiment.
        let bases = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let members = extract_class_members(source, m0.start());
        let mut signature = format!("class {name}{bases}");
        if !members.is_empty() {
            let top: Vec<&str> = members.iter().take(12).map(String::as_str).collect();
            signature.push_str(" — methods: ");
            signature.push_str(&top.join(", "));
        }

        file.exports.push(name.clone());
        file.key_types.push(format!("{name} class"));
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Class,
            line,
            signature,
            doc,
        });
    }

    // Functions (sync)
    for cap in RE_DEF.captures_iter(source) {
        let name = cap[1].to_string();
        if !include_private && name.starts_with('_') {
            continue;
        }
        let params = cap[2].trim().to_string();
        let ret = cap.get(3).map(|m| m.as_str().to_string());
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = extract_docstring_after(source, cap.get(0).unwrap().end());

        // Check if this is a top-level function (not indented = not a method)
        let line_start = &source[..cap.get(0).unwrap().start()];
        let indent = line_start.len() - line_start.trim_end().len();
        let is_method = indent > 0 && !line_start.ends_with('\n') || {
            let last_line_start = line_start.rfind('\n').map(|p| p + 1).unwrap_or(0);
            source[last_line_start..cap.get(0).unwrap().start()].starts_with([' ', '\t'])
        };

        if !is_method {
            file.exports.push(name.clone());
        }

        let sig = match ret {
            Some(r) => format!("def {name}({params}) -> {r}"),
            None => format!("def {name}({params})"),
        };

        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Function,
            line,
            signature: sig,
            doc,
        });
    }

    // Async functions
    for cap in RE_ASYNC_DEF.captures_iter(source) {
        let name = cap[1].to_string();
        if !include_private && name.starts_with('_') {
            continue;
        }
        let params = cap[2].trim().to_string();
        let ret = cap.get(3).map(|m| m.as_str().to_string());
        let line = find_line(source, cap.get(0).unwrap().start());
        let doc = extract_docstring_after(source, cap.get(0).unwrap().end());

        file.exports.push(name.clone());
        let sig = match ret {
            Some(r) => format!("async def {name}({params}) -> {r}"),
            None => format!("async def {name}({params})"),
        };
        file.symbols.push(ParsedSymbol {
            name,
            kind: SymbolKind::Function,
            line,
            signature: sig,
            doc,
        });
    }

    // Methods declared inside class bodies → `Class.method` symbols. The
    // RE_DEF/RE_ASYNC_DEF passes above are column-anchored (`^def`) and so only
    // ever see top-level functions; without this pass a method-level query like
    // `as_sql` never resolves `SQLCompiler.as_sql`. See `extract_methods` for the
    // scoping and visibility rules.
    file.symbols.extend(extract_methods(source, include_private));

    // Imports
    for cap in RE_IMPORT.captures_iter(source) {
        let module = cap
            .get(1)
            .or(cap.get(2))
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        // Get root package
        let root = module.split('.').next().unwrap_or(&module).to_string();
        if !root.is_empty() && !file.imports.contains(&root) {
            file.imports.push(root);
        }
    }

    // If __all__ is defined, filter exports
    if !explicit_exports.is_empty() {
        file.exports = explicit_exports;
    }

    // Pattern detection
    detect_patterns(source, &mut file);

    file.summary = generate_summary(&file);
    file
}

fn extract_module_docstring(lines: &[&str]) -> Option<String> {
    let mut i = 0;
    // Skip shebang, encoding, and blank lines
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }
        break;
    }

    if i >= lines.len() {
        return None;
    }

    let line = lines[i].trim();
    // Single-line docstring
    if (line.starts_with("\"\"\"") && line.ends_with("\"\"\"") && line.len() > 6)
        || (line.starts_with("'''") && line.ends_with("'''") && line.len() > 6)
    {
        return Some(line[3..line.len() - 3].trim().to_string());
    }

    // Multi-line docstring
    if line.starts_with("\"\"\"") || line.starts_with("'''") {
        let delim = &line[..3];
        let mut doc = line[3..].to_string();
        i += 1;
        while i < lines.len() {
            if lines[i].contains(delim) {
                let before = lines[i].split(delim).next().unwrap_or("").trim();
                if !before.is_empty() {
                    doc.push(' ');
                    doc.push_str(before);
                }
                break;
            }
            doc.push(' ');
            doc.push_str(lines[i].trim());
            i += 1;
        }
        let trimmed = doc.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }

    None
}

fn extract_all_list(source: &str) -> Vec<String> {
    if let Some(cap) = RE_ALL.captures(source) {
        let items = &cap[1];
        items
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim().trim_matches(|c| c == '"' || c == '\'').trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Collect the method names defined directly in a class body, in source
/// order, deduped, skipping dunder/private. Used to enrich the class proxy's
/// embedded text so conceptual queries can match by behavior, not just name.
///
/// Scope is determined by indentation: lines more indented than the `class`
/// line are the body; the first non-blank line at or below the class indent
/// ends it.
fn extract_class_members(source: &str, class_match_start: usize) -> Vec<String> {
    let line_start = source[..class_match_start]
        .rfind('\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    let class_indent = source[line_start..class_match_start]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .count();

    let mut members = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut lines = source[class_match_start..].lines();
    lines.next(); // skip the `class ...:` line itself
    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - trimmed.len();
        if indent <= class_indent {
            break; // dedent → end of class body
        }
        let body = trimmed.strip_prefix("async ").unwrap_or(trimmed);
        if let Some(rest) = body.strip_prefix("def ") {
            let mname: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !mname.is_empty() && !mname.starts_with("__") && seen.insert(mname.clone()) {
                members.push(mname);
            }
        }
    }
    members
}

/// Collect `Class.method` symbols for every `def`/`async def` declared directly
/// in a class body. Names are qualified (`SQLCompiler.as_sql`,
/// `QuerySet._fetch_all`) so the proxy builder mints one searchable proxy per
/// method and the owning class travels with the name into the breadcrumb.
///
/// Class membership is tracked with an indentation stack built *only* from
/// `class`/`def` header lines: a `def` is a method when the innermost open block
/// is a class; a `def` whose innermost block is another `def` (i.e. a function
/// nested inside a method) is not. Ignoring every non-header line is what keeps
/// the stack honest — a multi-line signature's continuation lines (`self,`,
/// `):`) can't prematurely close a scope, which is the only way a nested helper
/// could otherwise be mis-attributed as a method.
///
/// Like `extract_class_members` and the column-anchored regexes, this does not
/// track string literals, so a `def`-shaped line inside a docstring is a rare
/// false positive — the accepted heuristic level for this parser.
///
/// Visibility mirrors the top-level passes: dunder methods are skipped except
/// `__init__`; a single-underscore private method (or any method of a
/// single-underscore private class) is emitted only when `include_private`,
/// exactly as a `_`-prefixed top-level `def`/`class` is.
fn extract_methods(source: &str, include_private: bool) -> Vec<ParsedSymbol> {
    struct Scope {
        indent: usize,
        /// `Some` for a `class` header, `None` for a `def` header.
        class_name: Option<String>,
    }

    let mut stack: Vec<Scope> = Vec::new();
    let mut out = Vec::new();
    let mut offset = 0usize;
    let mut line_no = 0usize;

    for raw in source.split_inclusive('\n') {
        line_no += 1;
        offset += raw.len(); // now points at the start of the next line

        let line = raw.trim_end();
        let trimmed = line.trim_start();

        // Only class/def headers drive the stack; everything else (blank lines,
        // comments, statements, signature continuation lines) is ignored.
        let class_name = class_header_name(trimmed);
        let def_info = if class_name.is_none() {
            def_header_name(trimmed)
        } else {
            None
        };
        if class_name.is_none() && def_info.is_none() {
            continue;
        }

        let indent = line.len() - trimmed.len();
        while stack.last().map_or(false, |s| s.indent >= indent) {
            stack.pop();
        }

        if let Some(name) = class_name {
            stack.push(Scope {
                indent,
                class_name: Some(name),
            });
            continue;
        }

        let (name, is_async) = def_info.expect("def_info present when class_name absent");
        if let Some(class_name) = stack.last().and_then(|s| s.class_name.as_deref()) {
            let class_private = !include_private && class_name.starts_with('_');
            if !class_private && keep_method(&name, include_private) {
                out.push(ParsedSymbol {
                    name: format!("{class_name}.{name}"),
                    kind: SymbolKind::Function,
                    line: line_no,
                    signature: method_signature(trimmed, is_async, &name),
                    doc: extract_docstring_after(source, offset),
                });
            }
        }
        // Push the def's own scope so a function nested inside it is not seen
        // as a method of the enclosing class.
        stack.push(Scope {
            indent,
            class_name: None,
        });
    }

    out
}

/// Parse `class Name` from a line's leading text, returning the class name.
/// Requires whitespace after `class` so `classmethod`/`class_var` don't match.
fn class_header_name(trimmed: &str) -> Option<String> {
    let rest = trimmed.strip_prefix("class")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let name: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Parse `def name` / `async def name` from a line's leading text, returning
/// the bare name and whether it is `async`. Requires whitespace after the
/// keywords so `default`/`def_foo` don't match.
fn def_header_name(trimmed: &str) -> Option<(String, bool)> {
    let (is_async, rest) = match trimmed.strip_prefix("async") {
        Some(r) if r.starts_with(char::is_whitespace) => (true, r.trim_start()),
        _ => (false, trimmed),
    };
    let rest = rest.strip_prefix("def")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let name: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some((name, is_async))
}

/// Whether a method should be indexed. Dunders are dropped except `__init__`;
/// single-underscore private methods follow `include_private`. Mirrors the
/// `_`-prefix handling of the top-level function pass.
fn keep_method(name: &str, include_private: bool) -> bool {
    if name == "__init__" {
        return true;
    }
    if name.starts_with("__") && name.ends_with("__") {
        return false; // dunder other than __init__
    }
    if name.starts_with('_') {
        return include_private;
    }
    true
}

/// Best-effort single-line method signature in the same shape the top-level
/// passes emit (`def name(params)` / `... -> ret`). Params/return are captured
/// from the header line only; a multi-line signature degrades to `name(...)` —
/// the same naive capture the top-level `[^)]*` regex performs.
fn method_signature(trimmed: &str, is_async: bool, name: &str) -> String {
    let head = if is_async { "async def" } else { "def" };
    let params = trimmed
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(p, _)| p.trim());
    let ret = trimmed
        .split_once("->")
        .map(|(_, r)| r.split(':').next().unwrap_or(r).trim())
        .filter(|r| !r.is_empty());
    match (params, ret) {
        (Some(p), Some(r)) => format!("{head} {name}({p}) -> {r}"),
        (Some(p), None) => format!("{head} {name}({p})"),
        (None, _) => format!("{head} {name}(...)"),
    }
}

fn extract_docstring_after(source: &str, offset: usize) -> Option<String> {
    let rest = &source[offset..];
    // Look for a docstring on the next line(s)
    let trimmed = rest.trim_start();
    if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
        let delim = &trimmed[..3];
        let after_delim = &trimmed[3..];
        // Single-line docstring
        if let Some(end) = after_delim.find(delim) {
            return Some(after_delim[..end].trim().to_string());
        }
        // Multi-line: take first line
        let first_line = after_delim.lines().next().unwrap_or("").trim();
        if !first_line.is_empty() {
            return Some(first_line.to_string());
        }
    }
    None
}

use super::find_line;

fn detect_patterns(source: &str, file: &mut ParsedFile) {
    // FastAPI
    if source.contains("@app.get")
        || source.contains("@app.post")
        || source.contains("@router.get")
        || source.contains("@router.post")
        || source.contains("FastAPI")
    {
        file.patterns.push("fastapi_route".to_string());
    }

    // Django
    if source.contains("django")
        || source.contains("models.Model")
        || source.contains("views.")
        || source.contains("HttpResponse")
    {
        file.patterns.push("django".to_string());
    }

    // Dataclasses
    if source.contains("@dataclass") || source.contains("dataclasses") {
        file.patterns.push("dataclass".to_string());
    }

    // Pydantic
    if source.contains("BaseModel") && source.contains("pydantic") {
        file.patterns.push("pydantic_model".to_string());
    }

    // Tests
    if source.contains("def test_") || source.contains("pytest") || source.contains("unittest") {
        file.patterns.push("tests".to_string());
    }
}

const PY_PATTERN_LABELS: &[(&str, &str)] = &[
    ("fastapi_route", "FastAPI route handlers"),
    ("django", "Django views/models"),
    ("dataclass", "dataclass definitions"),
    ("pydantic_model", "Pydantic models"),
    ("tests", "tests"),
];

fn generate_summary(file: &ParsedFile) -> String {
    super::generate_summary_common(file, PY_PATTERN_LABELS, || {
        "no summary available".to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_functions() {
        let src = r#"
def validate_token(token: str) -> bool:
    """Validates a JWT token."""
    return True

class AuthService:
    def check(self):
        pass
"#;
        let result = parse(src, false);
        assert!(result.exports.contains(&"validate_token".to_string()));
        assert!(result.exports.contains(&"AuthService".to_string()));
    }

    #[test]
    fn parse_module_docstring() {
        let src = r#"
"""Authentication module for JWT token handling."""

def validate():
    pass
"#;
        let result = parse(src, false);
        assert!(result.module_doc.is_some());
        assert!(result.summary.contains("Authentication module"));
    }

    #[test]
    fn parse_all_list() {
        let src = r#"
__all__ = ["validate", "AuthService"]

def validate():
    pass

def _internal():
    pass

class AuthService:
    pass
"#;
        let result = parse(src, false);
        assert_eq!(result.exports, vec!["validate", "AuthService"]);
    }

    #[test]
    fn parse_methods_qualified_names() {
        let src = r#"
class SQLCompiler:
    def as_sql(self, with_limits=True):
        """Return the SQL string and params."""
        return ""

    async def execute_sql(self):
        return None
"#;
        let result = parse(src, false);
        let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"SQLCompiler.as_sql"), "{names:?}");
        assert!(names.contains(&"SQLCompiler.execute_sql"), "{names:?}");
        // The class itself is still a symbol.
        assert!(names.contains(&"SQLCompiler"));
        // Methods do not leak into module exports.
        assert!(
            !result.exports.iter().any(|e| e.contains('.')),
            "exports: {:?}",
            result.exports
        );
        // async marker + docstring are carried through.
        let m = result
            .symbols
            .iter()
            .find(|s| s.name == "SQLCompiler.execute_sql")
            .unwrap();
        assert!(m.signature.starts_with("async def"), "{}", m.signature);
        let a = result
            .symbols
            .iter()
            .find(|s| s.name == "SQLCompiler.as_sql")
            .unwrap();
        assert_eq!(a.doc.as_deref(), Some("Return the SQL string and params."));
    }

    #[test]
    fn parse_methods_init_kept_dunders_skipped() {
        let src = r#"
class Widget:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return "w"

    def __repr__(self):
        return "w"

    def render(self):
        return self.x
"#;
        let result = parse(src, false);
        let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Widget.__init__"), "{names:?}");
        assert!(names.contains(&"Widget.render"), "{names:?}");
        assert!(
            !names.contains(&"Widget.__str__") && !names.contains(&"Widget.__repr__"),
            "dunders should be skipped: {names:?}"
        );
    }

    #[test]
    fn parse_methods_private_follows_index_private_flag() {
        let src = r#"
class QuerySet:
    def _fetch_all(self):
        return []

    def all(self):
        return self
"#;
        // Default (public only): private method excluded, mirroring top-level.
        let public = parse(src, false);
        let pub_names: Vec<&str> = public.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            !pub_names.contains(&"QuerySet._fetch_all"),
            "private method leaked: {pub_names:?}"
        );
        assert!(pub_names.contains(&"QuerySet.all"), "{pub_names:?}");

        // index_private: the private method is emitted.
        let private = parse(src, true);
        let priv_names: Vec<&str> = private.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            priv_names.contains(&"QuerySet._fetch_all"),
            "{priv_names:?}"
        );
    }

    #[test]
    fn parse_methods_nested_class_and_nested_function() {
        let src = r#"
class Outer:
    class Meta:
        def inner_method(self):
            pass

    def outer_method(self):
        def helper():
            return 1
        return helper()
"#;
        let result = parse(src, false);
        let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
        // Method of the nested class attributes to the nearest enclosing class.
        assert!(names.contains(&"Meta.inner_method"), "{names:?}");
        assert!(names.contains(&"Outer.outer_method"), "{names:?}");
        // A function nested inside a method is NOT a method.
        assert!(
            !names.iter().any(|n| n.ends_with(".helper")) && !names.contains(&"helper"),
            "nested helper mis-attributed: {names:?}"
        );
    }

    #[test]
    fn parse_methods_multiline_signature_and_private_class() {
        let src = r#"
class _Internal:
    def should_be_hidden(self):
        pass

class Compiler:
    def compile(
        self,
        query,
    ) -> str:
        def local_only():
            return 0
        return ""
"#;
        let result = parse(src, false);
        let names: Vec<&str> = result.symbols.iter().map(|s| s.name.as_str()).collect();
        // Multi-line signature is still captured (degrades to `(...)`).
        assert!(names.contains(&"Compiler.compile"), "{names:?}");
        // A function nested in a multi-line-signature method is not a method.
        assert!(
            !names.iter().any(|n| n.ends_with(".local_only")),
            "{names:?}"
        );
        // Methods of a single-underscore private class are hidden by default.
        assert!(
            !names.iter().any(|n| n.starts_with("_Internal.")),
            "private-class method leaked: {names:?}"
        );
    }

    #[test]
    fn detect_fastapi() {
        let src = r#"
from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
async def get_users():
    return []
"#;
        let result = parse(src, false);
        assert!(result.patterns.contains(&"fastapi_route".to_string()));
    }
}
