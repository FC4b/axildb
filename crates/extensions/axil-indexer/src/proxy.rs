//! Structural code proxies (Phase 13b).
//!
//! A `CodeProxy` is a small, structure-aware record that points back to an
//! exact file, symbol, or markdown section. Proxies are embedded and FTS-
//! indexed so agent recall can retrieve compact pointers (path + line +
//! breadcrumb + signature) before reading raw source.
//!
//! Identity (`proxy_id`) is deterministic on
//! `(project | path | kind | canonical_id_or_symbol | sig_hash_or_heading_path)`
//! so re-indexing unchanged content does not produce duplicates, and so an
//! agent can compare logical identities across re-indexes without depending
//! on the storage `RecordId`.
//!
//! Line numbers are *navigation hints* and are deliberately excluded from
//! identity — moving a symbol down by adding lines above it must not create
//! a second logical proxy.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::token;

pub const TABLE_CODE_PROXIES: &str = "_idx_code_proxies";

/// Re-export of `axil_core::code_refs::TABLE_CODE_REFS_INDEX`.
///
/// The reverse index over `data.code_refs[]` arrays lives in axil-core so
/// `Axil::insert` / `Axil::update` can sync it transparently for every
/// caller (CLI, MCP, embedded library).
pub use axil_core::code_refs::TABLE_CODE_REFS_INDEX;

/// Re-export of `axil_core::code_refs::anchor_keys`.
pub use axil_core::code_refs::anchor_keys as code_ref_anchor_keys;

/// Default proxy text token budget. Proxies should stay compact so that
/// recall returns many pointers without exhausting the agent's context.
pub const DEFAULT_PROXY_TOKEN_BUDGET: usize = 256;

/// Kind of structural proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProxyKind {
    File,
    Symbol,
    Section,
}

impl ProxyKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Symbol => "symbol",
            Self::Section => "section",
        }
    }
}

/// A structural proxy record built by the indexer.
///
/// `proxy_text` is what gets embedded and FTS-indexed; the pointer fields
/// (`path`, `line_start`, `symbol`, `breadcrumb`, ...) are what the agent
/// receives back at recall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProxy {
    pub kind: ProxyKind,
    pub project: Option<String>,
    pub module: Option<String>,
    pub path: String,
    pub language: Option<String>,
    pub symbol: Option<String>,
    pub signature: Option<String>,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
    /// SCIP-style canonical identifier when known. For markdown sections
    /// this holds the heading path like `path#H1>H2>H3`.
    pub canonical_id: Option<String>,
    pub breadcrumb: String,
    pub summary: Option<String>,
    pub proxy_text: String,
    pub source_record: Option<String>,
    /// Stable logical identity — see module docs.
    pub proxy_id: String,
    pub tokens: usize,
}

/// Inputs to build a single proxy. The builder is shared between the project
/// indexer (file/symbol proxies) and the markdown splitter (section proxies),
/// and is reusable from CLI/tests.
#[derive(Debug, Clone, Default)]
pub struct ProxyInput<'a> {
    pub kind: ProxyKind,
    pub project: Option<&'a str>,
    pub module: Option<&'a str>,
    pub path: &'a str,
    pub language: Option<&'a str>,
    pub symbol: Option<&'a str>,
    pub signature: Option<&'a str>,
    pub line_start: Option<usize>,
    pub line_end: Option<usize>,
    pub canonical_id: Option<&'a str>,
    pub summary: Option<&'a str>,
    pub doc: Option<&'a str>,
    /// File-level proxies should pass top imports/exports/key types here so
    /// that the embedded `proxy_text` carries enough structural keywords to
    /// be discoverable via vector and FTS. Each list is best-effort and is
    /// truncated to keep proxy_text inside the token budget.
    pub imports: &'a [String],
    pub exports: &'a [String],
    pub key_types: &'a [String],
    pub heading_path: Option<&'a [String]>,
    pub source_record: Option<&'a str>,
    pub token_budget: usize,
}

impl Default for ProxyKind {
    fn default() -> Self {
        Self::File
    }
}

/// Build a `CodeProxy` from input. Always succeeds — fields the parser
/// could not provide are simply omitted. The returned proxy is ready to
/// insert into `_idx_code_proxies`.
pub fn build_proxy(input: ProxyInput<'_>) -> CodeProxy {
    let breadcrumb = build_breadcrumb(&input);
    let signature_hash = signature_hash(input.signature);
    let heading_path_str = input.heading_path.map(|h| h.join(">"));
    let proxy_id = build_proxy_id(
        input.project.unwrap_or(""),
        input.path,
        input.kind,
        input.canonical_id.or(input.symbol).unwrap_or(""),
        signature_hash
            .as_deref()
            .or(heading_path_str.as_deref())
            .unwrap_or(""),
    );

    let budget = if input.token_budget == 0 {
        DEFAULT_PROXY_TOKEN_BUDGET
    } else {
        input.token_budget
    };
    let proxy_text = build_proxy_text(&input, &breadcrumb, budget);
    let tokens = token::estimate_tokens(&proxy_text);

    CodeProxy {
        kind: input.kind,
        project: input.project.map(str::to_string),
        module: input.module.map(str::to_string),
        path: input.path.to_string(),
        language: input.language.map(str::to_string),
        symbol: input.symbol.map(str::to_string),
        signature: input.signature.map(str::to_string),
        line_start: input.line_start,
        line_end: input.line_end,
        canonical_id: input.canonical_id.map(str::to_string),
        breadcrumb,
        summary: input.summary.map(str::to_string),
        proxy_text,
        source_record: input.source_record.map(str::to_string),
        proxy_id,
        tokens,
    }
}

/// Render a proxy as JSON suitable for `Axil::insert(TABLE_CODE_PROXIES, ...)`.
///
/// Field names are kept stable: external callers (recall, MCP, CLI) read
/// these from the stored record.
pub fn proxy_to_record(proxy: &CodeProxy) -> Value {
    let mut data = serde_json::Map::new();
    data.insert("proxy_id".into(), json!(proxy.proxy_id));
    data.insert("kind".into(), json!(proxy.kind.as_str()));
    data.insert("path".into(), json!(proxy.path));
    data.insert("breadcrumb".into(), json!(proxy.breadcrumb));
    data.insert("proxy_text".into(), json!(proxy.proxy_text));
    data.insert("tokens".into(), json!(proxy.tokens));
    if let Some(p) = &proxy.project {
        data.insert("project".into(), json!(p));
    }
    if let Some(m) = &proxy.module {
        data.insert("module".into(), json!(m));
    }
    if let Some(l) = &proxy.language {
        data.insert("language".into(), json!(l));
    }
    if let Some(s) = &proxy.symbol {
        data.insert("symbol".into(), json!(s));
    }
    if let Some(sig) = &proxy.signature {
        data.insert("signature".into(), json!(sig));
    }
    if let Some(start) = proxy.line_start {
        data.insert("line_start".into(), json!(start));
    }
    if let Some(end) = proxy.line_end {
        data.insert("line_end".into(), json!(end));
    }
    if let Some(c) = &proxy.canonical_id {
        data.insert("canonical_id".into(), json!(c));
    }
    if let Some(s) = &proxy.summary {
        data.insert("summary".into(), json!(s));
    }
    if let Some(src) = &proxy.source_record {
        data.insert("source_record".into(), json!(src));
    }
    Value::Object(data)
}

fn build_breadcrumb(input: &ProxyInput<'_>) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(p) = input.project {
        if !p.is_empty() {
            parts.push(p);
        }
    }
    if let Some(m) = input.module {
        if !m.is_empty() {
            parts.push(m);
        }
    }
    let last_component = std::path::Path::new(input.path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(input.path);
    parts.push(last_component);
    if let Some(s) = input.symbol {
        parts.push(s);
    }
    if let Some(headings) = input.heading_path {
        for h in headings {
            parts.push(h.as_str());
        }
    }
    parts.join(" > ")
}

/// Produce the searchable text body for a proxy.
///
/// Format:
/// ```text
/// project > module > file > symbol
/// signature                       (when present)
/// summary / doc                   (when present)
/// keywords (imports/exports/...)  (file-level only)
/// ```
///
/// Keeps the result under `token_budget` by trimming trailing keyword/doc
/// content first; never trims the breadcrumb or signature.
pub fn build_proxy_text(input: &ProxyInput<'_>, breadcrumb: &str, token_budget: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(breadcrumb.to_string());

    if let Some(sig) = input.signature {
        lines.push(sig.trim().to_string());
    }

    let summary_or_doc = input
        .summary
        .or(input.doc)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(s) = summary_or_doc {
        lines.push(squeeze_lines(s));
    }

    if matches!(input.kind, ProxyKind::File) {
        let mut keyword_parts: Vec<String> = Vec::new();
        if !input.exports.is_empty() {
            let top: Vec<&str> = input.exports.iter().take(8).map(String::as_str).collect();
            keyword_parts.push(format!("exports: {}", top.join(", ")));
        }
        if !input.key_types.is_empty() {
            let top: Vec<&str> = input.key_types.iter().take(6).map(String::as_str).collect();
            keyword_parts.push(format!("types: {}", top.join(", ")));
        }
        if !input.imports.is_empty() {
            let top: Vec<&str> = input.imports.iter().take(8).map(String::as_str).collect();
            keyword_parts.push(format!("imports: {}", top.join(", ")));
        }
        if !keyword_parts.is_empty() {
            lines.push(keyword_parts.join(" | "));
        }
    }

    let text = lines.join("\n");
    let max_chars = token_budget.saturating_mul(4);
    if max_chars > 0 && text.len() > max_chars {
        return axil_core::util::truncate_str(&text, max_chars).to_string();
    }
    text
}

fn squeeze_lines(s: &str) -> String {
    s.split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_signature(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn short_hash(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn signature_hash(sig: Option<&str>) -> Option<String> {
    sig.map(normalize_signature).map(|s| short_hash(&s))
}

/// Phase 13b.8 P1: backfill report — counts proxies that gained a
/// canonical_id (and a new proxy_id) after a SCIP ingest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackfillReport {
    pub examined: usize,
    pub upgraded: usize,
    pub ambiguous: usize,
}

/// Phase 13b.8 P1: rewrite `_idx_code_proxies` rows that did not have a
/// SCIP `canonical_id` when they were built so they pick up the canonical
/// id (and a new stable `proxy_id`) once SCIP data exists in the DB.
///
/// Lookup key: for each symbol proxy without `canonical_id`, find a
/// `_scip_aliases` row whose `alias == proxy.symbol` and
/// `scope == "file:{proxy.path}"`. The file-scope is unambiguous because
/// SCIP emits one alias per `(symbol-name, file)` pair on definition. If
/// more than one alias matches, the proxy is left alone (ambiguous), so
/// no silent canonical-id collisions.
///
/// `proxy_id` is recomputed using the existing identity rule (project,
/// path, kind, canonical_id, signature_hash). The old proxy is updated
/// in-place — Phase 13b's identity rule (canonical_id dominates) means
/// the new id is the durable one going forward; old code_refs that
/// pointed at the regex-only proxy_id continue to work via path/symbol
/// fallback in `related_memories_for_proxies`.
pub fn backfill_canonical_ids_from_scip(db: &axil_core::Axil) -> axil_core::Result<BackfillReport> {
    use std::collections::HashMap;

    let aliases = db.list(axil_core::SCIP_ALIAS_TABLE).unwrap_or_default();
    if aliases.is_empty() {
        return Ok(BackfillReport::default());
    }
    // Index aliases by `(file_path, symbol_name)` -> Vec<canonical_id>.
    // We only consider `file:` scoped aliases — `lang:` and `global` scopes
    // span multiple files and cannot uniquely identify a proxy.
    let mut by_file_name: HashMap<(String, String), Vec<String>> = HashMap::new();
    for r in &aliases {
        let alias = match r.data.get("alias").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let scope = match r.data.get("scope").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let cid = match r.data.get("canonical_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let path = match scope.strip_prefix("file:") {
            Some(p) => p,
            None => continue,
        };
        by_file_name
            .entry((path.to_string(), alias.to_string()))
            .or_default()
            .push(cid.to_string());
    }

    if by_file_name.is_empty() {
        return Ok(BackfillReport::default());
    }

    let proxies = db.list(TABLE_CODE_PROXIES)?;
    let mut report = BackfillReport::default();
    for proxy in proxies {
        if proxy.data.get("kind").and_then(|v| v.as_str()) != Some(ProxyKind::Symbol.as_str()) {
            continue;
        }
        let already = proxy
            .data
            .get("canonical_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty() && !s.starts_with("provisional:"))
            .is_some();
        if already {
            continue;
        }
        report.examined += 1;
        let path = match proxy.data.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => continue,
        };
        let symbol = match proxy.data.get("symbol").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let candidates = match by_file_name.get(&(path.clone(), symbol.clone())) {
            Some(c) => c,
            None => continue,
        };
        if candidates.len() != 1 {
            // Ambiguous — leave the proxy alone rather than picking one
            // arbitrarily and silently collapsing two distinct symbols.
            report.ambiguous += 1;
            continue;
        }
        let canonical = &candidates[0];
        let mut new_data = proxy.data.clone();
        new_data["canonical_id"] = serde_json::Value::String(canonical.clone());

        // Recompute proxy_id using the same identity rule as build_proxy.
        let project = new_data
            .get("project")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let signature = new_data
            .get("signature")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let sig_hash = signature_hash(signature).unwrap_or_default();
        let new_proxy_id = build_proxy_id(&project, &path, ProxyKind::Symbol, canonical, &sig_hash);
        new_data["proxy_id"] = serde_json::Value::String(new_proxy_id);

        db.update(&proxy.id, new_data)?;
        report.upgraded += 1;
    }
    Ok(report)
}

/// Build the stable logical identity of a proxy. See module docs.
///
/// Inputs are deliberately separated so any call site can compute the same
/// id from primitive strings (no dependency on `CodeProxy` itself).
pub fn build_proxy_id(
    project_or_member: &str,
    path: &str,
    kind: ProxyKind,
    canonical_id_or_symbol: &str,
    signature_or_heading_hash: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(project_or_member.as_bytes());
    hasher.update(b"|");
    hasher.update(path.as_bytes());
    hasher.update(b"|");
    hasher.update(kind.as_str().as_bytes());
    hasher.update(b"|");
    hasher.update(canonical_id_or_symbol.as_bytes());
    hasher.update(b"|");
    hasher.update(signature_or_heading_hash.as_bytes());
    let digest = hasher.finalize();
    digest.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_symbol_input<'a>(
        path: &'a str,
        sym: &'a str,
        sig: &'a str,
        line: usize,
    ) -> ProxyInput<'a> {
        ProxyInput {
            kind: ProxyKind::Symbol,
            project: Some("axildb"),
            module: Some("crates/axil-core"),
            path,
            language: Some("rust"),
            symbol: Some(sym),
            signature: Some(sig),
            line_start: Some(line),
            line_end: None,
            canonical_id: None,
            summary: None,
            doc: Some("Combines vector, FTS, graph, feedback, and QTC scoring for agent recall."),
            imports: &[],
            exports: &[],
            key_types: &[],
            heading_path: None,
            source_record: None,
            token_budget: DEFAULT_PROXY_TOKEN_BUDGET,
        }
    }

    #[test]
    fn rust_symbol_proxy_has_breadcrumb_and_signature() {
        let input = rust_symbol_input(
            "crates/axil-core/src/db.rs",
            "Axil::recall",
            "pub fn recall(query: &str) -> Result<...>",
            2930,
        );
        let proxy = build_proxy(input);
        assert_eq!(proxy.kind, ProxyKind::Symbol);
        assert!(proxy.breadcrumb.contains("axildb"));
        assert!(proxy.breadcrumb.contains("db.rs"));
        assert!(proxy.breadcrumb.contains("Axil::recall"));
        assert!(proxy.proxy_text.contains("pub fn recall"));
        assert!(proxy.proxy_text.contains("agent recall"));
        assert!(proxy.tokens > 0);
        assert_eq!(proxy.line_start, Some(2930));
    }

    #[test]
    fn proxy_id_is_stable_across_line_movements() {
        // moving the symbol down 50 lines must not change the logical id.
        let a = build_proxy(rust_symbol_input(
            "crates/axil-core/src/db.rs",
            "Axil::recall",
            "pub fn recall(q: &str)",
            2930,
        ));
        let b = build_proxy(rust_symbol_input(
            "crates/axil-core/src/db.rs",
            "Axil::recall",
            "pub fn recall(q: &str)",
            2980,
        ));
        assert_eq!(a.proxy_id, b.proxy_id);
    }

    #[test]
    fn proxy_id_changes_when_signature_changes() {
        let a = build_proxy(rust_symbol_input("p.rs", "f", "pub fn f(a: i32)", 1));
        let b = build_proxy(rust_symbol_input(
            "p.rs",
            "f",
            "pub fn f(a: i32, b: i32)",
            1,
        ));
        assert_ne!(a.proxy_id, b.proxy_id);
    }

    #[test]
    fn proxy_id_changes_when_path_changes() {
        let a = build_proxy(rust_symbol_input("a.rs", "f", "pub fn f()", 1));
        let b = build_proxy(rust_symbol_input("b.rs", "f", "pub fn f()", 1));
        assert_ne!(a.proxy_id, b.proxy_id);
    }

    #[test]
    fn file_proxy_includes_imports_exports() {
        let imports = vec!["serde".into(), "tokio".into()];
        let exports = vec!["build".into(), "Recall".into()];
        let key_types = vec!["RecallResult".into()];
        let input = ProxyInput {
            kind: ProxyKind::File,
            project: Some("axildb"),
            module: Some("crates/axil-indexer"),
            path: "crates/axil-indexer/src/recall.rs",
            language: Some("rust"),
            symbol: None,
            signature: None,
            line_start: Some(1),
            line_end: Some(640),
            canonical_id: None,
            summary: Some("Agent-optimized recall across index tables"),
            doc: None,
            imports: &imports,
            exports: &exports,
            key_types: &key_types,
            heading_path: None,
            source_record: None,
            token_budget: DEFAULT_PROXY_TOKEN_BUDGET,
        };
        let proxy = build_proxy(input);
        assert!(proxy.proxy_text.contains("exports: build, Recall"));
        assert!(proxy.proxy_text.contains("imports: serde, tokio"));
        assert!(proxy.proxy_text.contains("types: RecallResult"));
        assert_eq!(proxy.kind, ProxyKind::File);
    }

    #[test]
    fn section_proxy_uses_heading_path() {
        let headings = vec!["Phase 13b".to_string(), "Identity".to_string()];
        let input = ProxyInput {
            kind: ProxyKind::Section,
            project: Some("axildb"),
            module: None,
            path: "tasks/phase-13b.md",
            language: Some("markdown"),
            symbol: Some("Identity"),
            signature: None,
            line_start: Some(84),
            line_end: Some(106),
            canonical_id: Some("tasks/phase-13b.md#Phase 13b>Identity"),
            summary: None,
            doc: Some("proxy_id is sha256(...). Lines are navigation hints."),
            imports: &[],
            exports: &[],
            key_types: &[],
            heading_path: Some(&headings),
            source_record: None,
            token_budget: DEFAULT_PROXY_TOKEN_BUDGET,
        };
        let proxy = build_proxy(input);
        assert_eq!(proxy.kind, ProxyKind::Section);
        assert!(proxy.breadcrumb.contains("Phase 13b"));
        assert!(proxy.breadcrumb.contains("Identity"));
        assert!(proxy.canonical_id.unwrap().contains("Phase 13b>Identity"));
    }

    #[test]
    fn python_and_typescript_symbol_proxies_build() {
        // Python
        let py = build_proxy(ProxyInput {
            kind: ProxyKind::Symbol,
            project: Some("svc"),
            module: Some("api"),
            path: "api/auth.py",
            language: Some("python"),
            symbol: Some("login"),
            signature: Some("def login(user: str, pw: str) -> Token"),
            line_start: Some(42),
            doc: Some("Authenticate a user."),
            token_budget: DEFAULT_PROXY_TOKEN_BUDGET,
            ..Default::default()
        });
        assert!(py.proxy_text.contains("def login"));
        assert!(!py.proxy_id.is_empty());

        // TypeScript
        let ts = build_proxy(ProxyInput {
            kind: ProxyKind::Symbol,
            project: Some("web"),
            module: Some("src/auth"),
            path: "src/auth/login.ts",
            language: Some("typescript"),
            symbol: Some("loginUser"),
            signature: Some("export async function loginUser(c: Creds): Promise<Token>"),
            line_start: Some(10),
            doc: Some("Issue a session token."),
            token_budget: DEFAULT_PROXY_TOKEN_BUDGET,
            ..Default::default()
        });
        assert!(ts.proxy_text.contains("loginUser"));
    }

    #[test]
    fn token_budget_caps_proxy_text_length() {
        // budget=8 tokens ≈ 32 chars
        let input = ProxyInput {
            kind: ProxyKind::Symbol,
            project: Some("p"),
            path: "f.rs",
            language: Some("rust"),
            symbol: Some("very_long_function_name_here"),
            signature: Some(
                "pub fn very_long_function_name_here(a: i32, b: i32, c: i32, d: i32) -> Result<()>",
            ),
            doc: Some("Documentation that should be truncated because the budget is tiny"),
            token_budget: 8,
            ..Default::default()
        };
        let proxy = build_proxy(input);
        assert!(proxy.proxy_text.len() <= 8 * 4);
    }

    #[test]
    fn signature_normalization_collapses_run_whitespace() {
        // Run-length whitespace normalization: extra interior spaces and
        // tabs/newlines collapse. (Tokens themselves are not lexed —
        // adjacent punctuation must remain attached for the hash to match.)
        let a = build_proxy_id(
            "p",
            "f.rs",
            ProxyKind::Symbol,
            "f",
            &short_hash(&normalize_signature("pub  fn   f(a: i32)")),
        );
        let b = build_proxy_id(
            "p",
            "f.rs",
            ProxyKind::Symbol,
            "f",
            &short_hash(&normalize_signature("pub\tfn\nf(a: i32)")),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn proxy_record_round_trips_through_json() {
        let proxy = build_proxy(rust_symbol_input("p.rs", "f", "pub fn f()", 1));
        let json = proxy_to_record(&proxy);
        assert_eq!(json["kind"], "symbol");
        assert_eq!(json["path"], "p.rs");
        assert_eq!(json["proxy_id"], proxy.proxy_id);
        assert!(json["proxy_text"].is_string());
    }
}
