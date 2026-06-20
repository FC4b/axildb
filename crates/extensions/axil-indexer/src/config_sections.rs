//! P1: structural section splitters for config files.
//!
//! Coding projects carry config files (Cargo.toml, package.json,
//! pyproject.toml, CI workflows) whose internal structure is *exactly*
//! the level of granularity an agent wants to retrieve at — "where are
//! the dev-dependencies declared?", "what does the build script run?".
//! Without this, the file proxy returns the whole file body and forces
//! the agent to re-read the raw source.
//!
//! This module produces lightweight `ParsedSection` records (the same
//! shape used by the markdown splitter) for two formats today:
//! - TOML top-level tables: `[package]`, `[dependencies]`,
//!   `[dev-dependencies]`, `[tool.poetry]`, `[workspace.dependencies]`,
//!   etc. Subtables are kept on the parent until a sibling appears, so
//!   `[dependencies]` and `[dependencies.serde]` remain in one section.
//! - JSON top-level keys: e.g. `package.json`'s `scripts`,
//!   `dependencies`, `devDependencies`. Each top-level key becomes one
//!   section, line ranges from the raw text.
//!
//! YAML is intentionally deferred — tantivy/serde_yaml is a heavier
//! dependency and most of the practical YAML in coding projects (CI
//! workflows) shares the JSON shape closely enough that the JSON
//! splitter handles a large fraction of the value once a future YAML
//! pass converts to JSON-equivalent.
//!
//! The output uses `crate::markdown::ParsedSection` so downstream code
//! (proxy builder, recall) doesn't need a second type.

use crate::markdown::ParsedSection;

/// Split a TOML source into top-level table sections.
///
/// Behavior:
/// - One `ParsedSection` per top-level table header `[name]` or
///   `[[name]]` (array of tables). Subtables (`[name.sub]`) are folded
///   into the parent until a new top-level table starts.
/// - The text *before* any table header (the so-called "implicit root"
///   table) becomes a synthetic section named `(root)` when it carries
///   any non-comment content.
/// - Lines and offsets are 1-based.
pub fn split_toml_sections(source: &str) -> Vec<ParsedSection> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    // Walk lines once. A header is "sub-table" when its dotted name has
    // a parent that already opened a section above and no other top-level
    // table has appeared since — those rows fold into the running parent
    // rather than starting a new section. `[[bin]]` arrays of tables are
    // distinct entries: each `[[bin]]` deliberately starts a new section.
    let mut header_lines: Vec<(usize, String)> = Vec::new();
    let mut current_top: Option<String> = None;
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let parsed = parse_toml_header(trimmed);
        let (top, is_array_of_tables, was_subtable) = match parsed {
            Some(p) => p,
            None => continue,
        };
        if was_subtable && current_top.as_deref() == Some(top.as_str()) && !is_array_of_tables {
            // Sub-table of the running section — fold in.
            continue;
        }
        header_lines.push((idx, top.clone()));
        current_top = Some(top);
    }

    let mut sections: Vec<ParsedSection> = Vec::new();

    // Implicit-root section: text before the first header.
    let first_header_line = header_lines.first().map(|(i, _)| *i).unwrap_or(lines.len());
    if has_real_content(&lines[..first_header_line]) {
        sections.push(ParsedSection {
            depth: 1,
            heading_path: vec!["(root)".to_string()],
            heading: "(root)".to_string(),
            line_start: 1,
            line_end: first_header_line.max(1),
            body: lines[..first_header_line].join("\n").trim().to_string(),
        });
    }

    sections.extend(emit_top_level_sections(&lines, &header_lines));
    sections
}

/// Emit one `ParsedSection` per top-level header. Bodies span from the
/// line after the header to the line before the next header (or EOF).
/// Shared by the TOML and YAML splitters since they both produce a flat
/// list of `(line_idx, top_name)` headers.
fn emit_top_level_sections(lines: &[&str], header_lines: &[(usize, String)]) -> Vec<ParsedSection> {
    let mut sections = Vec::with_capacity(header_lines.len());
    for (i, (line_idx, name)) in header_lines.iter().enumerate() {
        let end_idx = header_lines
            .get(i + 1)
            .map(|(idx, _)| *idx)
            .unwrap_or(lines.len());
        let body = lines[line_idx + 1..end_idx].join("\n").trim().to_string();
        sections.push(ParsedSection {
            depth: 1,
            heading_path: vec![name.clone()],
            heading: name.clone(),
            line_start: line_idx + 1,
            line_end: end_idx.max(line_idx + 1),
            body,
        });
    }
    sections
}

/// Parse a TOML table header. Returns `(top_level_name,
/// is_array_of_tables, was_subtable)`. `was_subtable` is true when the
/// raw header had a dotted name (e.g. `[a.b]`) — used by the splitter
/// to fold sub-tables into the running parent section.
///
/// Strips an optional trailing `# comment` before bracket-checking so
/// a header like `[dependencies] # third-party crates` still parses.
fn parse_toml_header(line: &str) -> Option<(String, bool, bool)> {
    let trimmed = line.trim_end();
    // Drop trailing inline comment. TOML basic strings can't span lines
    // so we don't need a string-aware tokenizer here — the first `#`
    // outside of nothing is the comment start. Strip from the first
    // unescaped `#` outside any quoted run is overkill for headers
    // (table headers can't contain `#` in a basic key), so a plain
    // first-`#` split is safe.
    let content = trimmed.split('#').next().unwrap_or("").trim_end();
    if !(content.starts_with('[') && content.ends_with(']')) {
        return None;
    }
    let (inner, is_array_of_tables) = if content.starts_with("[[") && content.ends_with("]]") {
        (&content[2..content.len() - 2], true)
    } else {
        (&content[1..content.len() - 1], false)
    };
    if inner.is_empty() {
        return None;
    }
    let was_subtable = inner.contains('.');
    let top = inner.split('.').next()?.trim();
    if top.is_empty() {
        return None;
    }
    Some((top.to_string(), is_array_of_tables, was_subtable))
}

fn has_real_content(lines: &[&str]) -> bool {
    lines.iter().any(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with('#') && !t.starts_with("//")
    })
}

/// Split a YAML source into top-level key sections.
///
/// Targeted at CI workflow files (`.github/workflows/*.yml`) and other
/// flat top-level-keyed YAML. Sections are emitted for every line whose
/// indentation is column 0 and shape is `key:`.
///
/// Limitations (deliberate, dependency-free):
/// - Only block-style YAML at the document root. Multi-document streams
///   (`---` separated) are treated as one stream — sections from later
///   documents accumulate alongside the first.
/// - Comments inside string values aren't tracked. A `#` after a colon
///   stops the value parse for the heading line, which matches actual
///   YAML semantics.
/// - Quoted keys (`"foo bar":`) are kept as-is including quotes; this
///   matches what an agent expects to see in the breadcrumb.
pub fn split_yaml_sections(source: &str) -> Vec<ParsedSection> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut header_lines: Vec<(usize, String)> = Vec::new();
    for (idx, raw) in lines.iter().enumerate() {
        // Top-level keys are at column 0 (no leading whitespace), not
        // a document marker / sequence dash, and contain a colon
        // followed by either end-of-line or whitespace.
        if raw.starts_with(' ') || raw.starts_with('\t') {
            continue;
        }
        let trimmed_left = raw;
        if trimmed_left.is_empty()
            || trimmed_left.starts_with('#')
            || trimmed_left.starts_with("---")
            || trimmed_left.starts_with("...")
            || trimmed_left.starts_with("- ")
        {
            continue;
        }
        if let Some(name) = parse_yaml_top_level_key(trimmed_left) {
            header_lines.push((idx, name));
        }
    }

    if header_lines.is_empty() {
        return Vec::new();
    }

    emit_top_level_sections(&lines, &header_lines)
}

/// Parse a YAML top-level key from a line starting at column 0. Returns
/// `Some(key)` when the line shape is `key:` followed by EOL or
/// whitespace (with an optional inline value or comment).
fn parse_yaml_top_level_key(line: &str) -> Option<String> {
    let line = line.trim_end();
    let colon_idx = line.find(':')?;
    let key_raw = &line[..colon_idx];
    if key_raw.is_empty() || key_raw.contains(' ') || key_raw.contains('\t') {
        return None;
    }
    // Either end-of-line right after `:`, or whitespace.
    let after = &line[colon_idx + 1..];
    if !after.is_empty() && !after.starts_with(' ') && !after.starts_with('\t') {
        // `foo:bar` is not a YAML key/value — likely a URL or similar.
        return None;
    }
    Some(key_raw.to_string())
}

/// Split a JSON source into top-level key sections.
///
/// Designed for `package.json`, `tsconfig.json`, etc. — finds top-level
/// keys at brace depth 1 inside the outermost object and emits one
/// section per key. Falls back to a single `(root)` section when the
/// file is not an object or the source is not parseable as JSON-shaped
/// text.
///
/// This is a deliberately tiny, dependency-free walker: it does not
/// build a real JSON AST (we'd need `serde_json::from_str` and that
/// silently drops trailing commas etc.). Strings, comments, and braces
/// inside strings are respected enough to identify top-level keys.
pub fn split_json_sections(source: &str) -> Vec<ParsedSection> {
    let bytes = source.as_bytes();
    let mut sections: Vec<ParsedSection> = Vec::new();

    // Find first `{` at depth 0.
    let mut idx = match bytes.iter().position(|&b| b == b'{') {
        Some(i) => i + 1,
        None => return sections,
    };
    let mut depth: i32 = 1; // inside the outermost object
    let mut in_str = false;
    let mut esc = false;

    // Track current key-section being built so we can compute its line
    // range when we pop back to depth 0.
    struct Pending {
        name: String,
        start_byte: usize,
    }
    let mut pending: Option<Pending> = None;

    while idx < bytes.len() {
        let b = bytes[idx];
        if esc {
            esc = false;
            idx += 1;
            continue;
        }
        if in_str {
            match b {
                b'\\' => esc = true,
                b'"' => in_str = false,
                _ => {}
            }
            idx += 1;
            continue;
        }
        match b {
            b'"' => {
                if depth == 1 && pending.is_none() {
                    // Start of a top-level key.
                    if let Some((name, after_idx)) = read_json_string(bytes, idx) {
                        // Skip to ':' then to the value start.
                        let value_start = skip_to_colon_value(bytes, after_idx);
                        pending = Some(Pending {
                            name,
                            start_byte: value_start.unwrap_or(after_idx),
                        });
                        idx = after_idx;
                        continue;
                    }
                }
                in_str = true;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    // End of outermost object — finalize any pending.
                    if let Some(p) = pending.take() {
                        push_json_section(source, p.start_byte, idx, &p.name, &mut sections);
                    }
                    break;
                }
            }
            b',' => {
                if depth == 1 {
                    if let Some(p) = pending.take() {
                        push_json_section(source, p.start_byte, idx, &p.name, &mut sections);
                    }
                }
            }
            _ => {}
        }
        idx += 1;
    }

    sections
}

fn read_json_string(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    if bytes.get(start)? != &b'"' {
        return None;
    }
    let mut i = start + 1;
    let mut name = String::new();
    let mut esc = false;
    while i < bytes.len() {
        let b = bytes[i];
        if esc {
            // Best-effort: keep the escape literal in the name.
            name.push(b as char);
            esc = false;
        } else if b == b'\\' {
            esc = true;
        } else if b == b'"' {
            return Some((name, i + 1));
        } else {
            name.push(b as char);
        }
        i += 1;
    }
    None
}

fn skip_to_colon_value(bytes: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i < bytes.len() && bytes[i] != b':' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    Some(i)
}

fn push_json_section(
    source: &str,
    start_byte: usize,
    end_byte: usize,
    name: &str,
    out: &mut Vec<ParsedSection>,
) {
    let line_start = byte_to_line(source, start_byte);
    let line_end = byte_to_line(source, end_byte);
    let body = source
        .get(start_byte..end_byte)
        .unwrap_or("")
        .trim()
        .to_string();
    out.push(ParsedSection {
        depth: 1,
        heading_path: vec![name.to_string()],
        heading: name.to_string(),
        line_start,
        line_end,
        body,
    });
}

fn byte_to_line(source: &str, byte_idx: usize) -> usize {
    let cap = byte_idx.min(source.len());
    source[..cap].bytes().filter(|b| *b == b'\n').count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_top_level_tables() {
        let src = r#"# leading comment
foo = "bar"

[package]
name = "axil"
version = "0.1.0"

[dependencies]
serde = "1"
tokio = "1"

[dependencies.serde_json]
version = "1"

[dev-dependencies]
tempfile = "3"
"#;
        let sections = split_toml_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(names.contains(&"(root)"), "missing root: {names:?}");
        assert!(names.contains(&"package"));
        assert!(names.contains(&"dependencies"));
        assert!(names.contains(&"dev-dependencies"));
        // [dependencies.serde_json] is a sub-table — it should NOT spawn
        // its own top-level section.
        assert!(!names.contains(&"dependencies.serde_json"));
        // The `[dependencies]` body should include the sub-table since
        // it lives between `[dependencies]` and the next top-level.
        let deps = sections
            .iter()
            .find(|s| s.heading == "dependencies")
            .unwrap();
        assert!(deps.body.contains("serde_json"));
    }

    #[test]
    fn toml_array_of_tables() {
        let src = "[[bin]]\nname = \"a\"\n\n[[bin]]\nname = \"b\"\n";
        let sections = split_toml_sections(src);
        let bins: Vec<_> = sections.iter().filter(|s| s.heading == "bin").collect();
        assert_eq!(bins.len(), 2);
    }

    #[test]
    fn toml_header_with_inline_comment_is_recognized() {
        let src = "[dependencies] # third-party crates\nserde = \"1\"\n";
        let sections = split_toml_sections(src);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "dependencies");
        assert!(sections[0].body.contains("serde"));
    }

    #[test]
    fn toml_no_headers_emits_root_only() {
        let src = "name = \"x\"\nversion = \"0.1\"\n";
        let sections = split_toml_sections(src);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "(root)");
    }

    #[test]
    fn json_top_level_keys() {
        let src = r#"{
  "name": "demo",
  "version": "1.0.0",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  },
  "dependencies": {
    "react": "19"
  }
}"#;
        let sections = split_json_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"version"));
        assert!(names.contains(&"scripts"));
        assert!(names.contains(&"dependencies"));
        let scripts = sections.iter().find(|s| s.heading == "scripts").unwrap();
        assert!(scripts.body.contains("build"));
        assert!(scripts.body.contains("test"));
    }

    #[test]
    fn json_skips_keys_inside_nested_objects() {
        let src = r#"{ "a": 1, "outer": { "inner": 2 }, "b": 3 }"#;
        let sections = split_json_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        // `inner` is at depth 2 — must not be picked up as a top-level
        // key.
        assert!(!names.contains(&"inner"));
        assert!(names.contains(&"outer"));
    }

    #[test]
    fn json_handles_escaped_quotes_in_keys() {
        let src = r#"{ "weird\"key": 1, "ok": 2 }"#;
        let sections = split_json_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("weird")));
        assert!(names.contains(&"ok"));
    }

    #[test]
    fn json_non_object_returns_empty() {
        assert!(split_json_sections("[1, 2, 3]").is_empty());
        assert!(split_json_sections("not json").is_empty());
    }

    #[test]
    fn yaml_top_level_keys_for_ci_workflow() {
        let src = r#"name: CI

on:
  push:
    branches: [main]
  pull_request: {}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo build
  test:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
"#;
        let sections = split_yaml_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"on"));
        assert!(names.contains(&"jobs"));
        // `build:` and `test:` are nested under `jobs:` — not top-level.
        assert!(!names.contains(&"build"));
        assert!(!names.contains(&"test"));
        let jobs = sections.iter().find(|s| s.heading == "jobs").unwrap();
        assert!(jobs.body.contains("build"));
        assert!(jobs.body.contains("test"));
    }

    #[test]
    fn yaml_skips_comments_and_doc_markers() {
        let src = "# leading comment\n---\nname: CI\n# inline comment\n";
        let sections = split_yaml_sections(src);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "name");
    }

    #[test]
    fn yaml_url_lines_are_not_keys() {
        // `url:https://...` is not a YAML key (no whitespace after colon)
        // we should reject it instead of fabricating a section.
        let src = "url:https://example.com\nname: ok\n";
        let sections = split_yaml_sections(src);
        let names: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert!(!names.contains(&"url"));
        assert!(names.contains(&"name"));
    }
}
