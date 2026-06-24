//! Identifier-aware query classification for recall.
//!
//! A cheap, high-precision classifier that decides whether a recall query is an
//! *exact-identifier lookup* (a UUID, a file path, a code symbol, a hostname/URL,
//! or a long numeric/hex id) rather than a natural-language question. The result
//! lets [`crate::db::Axil::recall`] tilt Reciprocal Rank Fusion toward the
//! full-text-search list for identifier queries, so an exact lexical match is not
//! diluted by its semantic neighbors. Natural-language queries classify as
//! [`QueryClass::NaturalLanguage`] and leave fusion byte-identical to pure RRF.
//!
//! Precision is the design priority: a false positive (firing on prose) would
//! silently re-rank ordinary recall, so every rule errs toward *not* firing when
//! a token merely looks word-like. The classifier never falls back to "identifier"
//! on doubt.

use std::sync::LazyLock;

use regex::Regex;

/// What an identifier query matched on. Ordered by detection priority — the
/// classifier reports the first kind that fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentifierKind {
    /// Standard 8-4-4-4-12 hex UUID.
    Uuid,
    /// `http(s)://…` URL or a bare dotted hostname (`api.example.com`).
    HostnameUrl,
    /// A filesystem-ish path: multiple `/`-separated segments, or a single token
    /// carrying a file extension (`src/db.rs`, `./a/b`, `config.toml`).
    Path,
    /// A code symbol: `fn login`, `Foo::bar`, `mod::path`, or a single token that
    /// is unambiguously code-shaped (snake_case, CamelCase, or `::`-qualified).
    CodeSymbol,
    /// A long numeric or hex identifier (ULID, long decimal id), not a small count.
    NumericId,
}

impl IdentifierKind {
    /// Stable lowercase tag for profile/explain output, e.g. `"uuid"`.
    pub fn tag(self) -> &'static str {
        match self {
            IdentifierKind::Uuid => "uuid",
            IdentifierKind::HostnameUrl => "hostname",
            IdentifierKind::Path => "path",
            IdentifierKind::CodeSymbol => "symbol",
            IdentifierKind::NumericId => "numeric-id",
        }
    }
}

/// Classification of a recall query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryClass {
    /// An exact-identifier lookup. FTS is tilted up during fusion.
    Identifier(IdentifierKind),
    /// Ordinary natural-language query. Fusion is unchanged (pure RRF).
    NaturalLanguage,
}

impl QueryClass {
    /// `true` when the query is an identifier lookup.
    pub fn is_identifier(self) -> bool {
        matches!(self, QueryClass::Identifier(_))
    }

    /// The matched identifier kind, if any.
    pub fn identifier_kind(self) -> Option<IdentifierKind> {
        match self {
            QueryClass::Identifier(k) => Some(k),
            QueryClass::NaturalLanguage => None,
        }
    }

    /// Stable tag for profile/explain output: `"identifier:<kind>"` or
    /// `"natural-language"`.
    pub fn tag(self) -> String {
        match self {
            QueryClass::Identifier(k) => format!("identifier:{}", k.tag()),
            QueryClass::NaturalLanguage => "natural-language".to_string(),
        }
    }
}

// ── Precompiled detection patterns ────────────────────────────────────────

/// Whole-string UUID (allowing surrounding whitespace via trim before match).
static UUID_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$").unwrap()
});

/// `http://` / `https://` URL anchored at the start of the (trimmed) query.
static URL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^https?://\S+$").unwrap());

/// A single dotted hostname token: `a.b`, `api.example.com`. Each label is
/// alphanumeric/hyphen; at least two labels; the final label (TLD-ish) is
/// alphabetic so a decimal like `3.14` does not register as a host.
static HOSTNAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$").unwrap()
});

/// A `::`-qualified code path: `Foo::bar`, `mod::path::Item`. Each segment is an
/// identifier; at least one `::` separator.
static QUALIFIED_SYMBOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+$").unwrap()
});

/// A leading code keyword followed by an identifier: `fn login`, `struct User`,
/// `def handler`, `class Foo`, `func Serve`, `impl Bar`, `trait T`, `enum E`.
static KEYWORD_SYMBOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:fn|struct|impl|trait|enum|def|class|func|mod|interface|type)\s+[A-Za-z_$][A-Za-z0-9_$]*$")
        .unwrap()
});

/// A single bare code token (no spaces) that is unambiguously code-shaped:
/// snake_case (has `_`), or a multi-hump identifier with at least one internal
/// uppercase boundary (camelCase / PascalCase like `getUser`, `HttpClient`).
/// A single all-lowercase dictionary word (`login`) deliberately does NOT match.
static SNAKE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*$").unwrap());
static CAMEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*$").unwrap());

/// A function-call shaped token: `foo()`, `bar(arg)`.
static CALL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*\([^)]*\)$").unwrap());

/// A long numeric or hex identifier: ≥ 7 chars, all digits, or ≥ 10 chars of
/// `[0-9a-z]` with at least one digit (ULID/Crockford-base32-ish). Kept tight so
/// a plain year ("2026") or small count never registers.
static NUMERIC_ID_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[0-9]{7,}$").unwrap());
static ULID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^[0-9A-Z]{16,}$").unwrap());

/// Classify a recall query as an exact-identifier lookup or natural language.
///
/// This is a pure function and cheap enough to run once per recall. It is biased
/// toward [`QueryClass::NaturalLanguage`]: anything that reads like a phrase, or a
/// single ordinary word, classifies as natural language and leaves fusion
/// untouched.
pub fn classify_query(query: &str) -> QueryClass {
    let q = query.trim();
    if q.is_empty() {
        return QueryClass::NaturalLanguage;
    }

    // Whole-string identifiers (no internal whitespace).
    if !q.contains(char::is_whitespace) {
        if UUID_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::Uuid);
        }
        if URL_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::HostnameUrl);
        }
        // A slash-bearing token is unambiguously a path (`src/db.rs`, `./a/b`),
        // checked before hostname so a path is never mistaken for a host.
        if looks_like_slash_path(q) {
            return QueryClass::Identifier(IdentifierKind::Path);
        }
        // A dotted, slash-free token is structurally ambiguous between a
        // hostname (`api.example.com`) and a `file.ext` (`config.toml`). A known
        // file extension breaks the tie toward Path; otherwise the multi-label
        // dotted shape reads as a hostname. Either way it's still an Identifier
        // and the FTS tilt behaves identically — only the reported kind differs.
        if has_known_file_extension(q) {
            return QueryClass::Identifier(IdentifierKind::Path);
        }
        if HOSTNAME_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::HostnameUrl);
        }
        if looks_like_file_token(q) {
            return QueryClass::Identifier(IdentifierKind::Path);
        }
        if QUALIFIED_SYMBOL_RE.is_match(q) || CALL_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::CodeSymbol);
        }
        if NUMERIC_ID_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::NumericId);
        }
        // ULID/base32 long id, but exclude all-alpha words (those are caught by
        // the symbol rules or left as natural language).
        if ULID_RE.is_match(q) && q.chars().any(|c| c.is_ascii_digit()) {
            return QueryClass::Identifier(IdentifierKind::NumericId);
        }
        // Single code-shaped token: snake_case or camelCase/PascalCase. A plain
        // lowercase word (`login`, `timeout`) is intentionally NOT a symbol.
        if SNAKE_RE.is_match(q) || CAMEL_RE.is_match(q) {
            return QueryClass::Identifier(IdentifierKind::CodeSymbol);
        }
        return QueryClass::NaturalLanguage;
    }

    // Multi-token queries: only the explicit `<keyword> <name>` symbol form
    // fires. Everything else with spaces is treated as natural language so a
    // short English phrase never trips the identifier tilt.
    if KEYWORD_SYMBOL_RE.is_match(q) {
        return QueryClass::Identifier(IdentifierKind::CodeSymbol);
    }

    QueryClass::NaturalLanguage
}

/// Common source/config/doc file extensions, used only to break the
/// `file.ext` vs `host.tld` tie toward Path. Not exhaustive — an unknown
/// extension on a dotted token simply reports as a hostname (still an
/// identifier, same tilt).
const KNOWN_FILE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "kt", "c", "h", "cpp", "cc", "hpp", "cs",
    "rb", "php", "swift", "scala", "clj", "ex", "exs", "toml", "json", "yaml", "yml", "lock", "md",
    "txt", "cfg", "ini", "env", "sh", "bash", "zsh", "sql", "html", "css", "scss", "xml", "proto",
    "csv", "tsv", "log",
];

/// A dotted, slash-free token whose final segment is a known file extension.
fn has_known_file_extension(token: &str) -> bool {
    token
        .rsplit_once('.')
        .map(|(stem, ext)| {
            !stem.is_empty() && KNOWN_FILE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
        })
        .unwrap_or(false)
}

/// A whitespace-free token reads as a slash path when it has multiple
/// slash-separated segments or an explicit relative/absolute prefix.
fn looks_like_slash_path(token: &str) -> bool {
    let has_slash_segments = token.matches('/').count() >= 1
        && token.split('/').filter(|s| !s.is_empty()).count() >= 2;
    has_slash_segments || token.starts_with("./") || token.starts_with("../")
}

/// A bare, slash-free `file.ext` token: an alphabetic-led stem, a single dot, and
/// a short alphanumeric extension carrying at least one letter. Excludes
/// dotted-hostname shapes (multiple dots) and pure version strings (`1.2.3`),
/// which are filtered out before this is reached.
fn looks_like_file_token(token: &str) -> bool {
    let Some((stem, ext)) = token.rsplit_once('.') else {
        return false;
    };
    let ext_ok = (1..=8).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
        && ext.chars().any(|c| c.is_ascii_alphabetic());
    let stem_ok = !stem.is_empty()
        && stem
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic() || c == '_')
            .unwrap_or(false)
        // a stem that is itself dotted is a hostname, not a file token
        && !stem.contains('.');
    ext_ok && stem_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(q: &str) -> Option<IdentifierKind> {
        classify_query(q).identifier_kind()
    }

    #[test]
    fn uuid_fires() {
        assert_eq!(
            kind("550e8400-e29b-41d4-a716-446655440000"),
            Some(IdentifierKind::Uuid)
        );
        // case-insensitive + surrounding whitespace
        assert_eq!(
            kind("  6BA7B810-9DAD-11D1-80B4-00C04FD430C8  "),
            Some(IdentifierKind::Uuid)
        );
    }

    #[test]
    fn url_and_hostname_fire() {
        assert_eq!(
            kind("https://example.com/path?x=1"),
            Some(IdentifierKind::HostnameUrl)
        );
        assert_eq!(kind("api.example.com"), Some(IdentifierKind::HostnameUrl));
        assert_eq!(kind("foo.bar"), Some(IdentifierKind::HostnameUrl));
    }

    #[test]
    fn paths_fire() {
        assert_eq!(kind("src/db.rs"), Some(IdentifierKind::Path));
        assert_eq!(kind("./a/b"), Some(IdentifierKind::Path));
        assert_eq!(kind("crates/axil-core/src/lib.rs"), Some(IdentifierKind::Path));
        assert_eq!(kind("config.toml"), Some(IdentifierKind::Path));
        assert_eq!(kind("Cargo.lock"), Some(IdentifierKind::Path));
    }

    #[test]
    fn code_symbols_fire() {
        assert_eq!(kind("fn login"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("struct User"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("def handler"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("Foo::bar"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("axil_core::recall"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("get_user"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("getUserById"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("HttpClient"), Some(IdentifierKind::CodeSymbol));
        assert_eq!(kind("compute()"), Some(IdentifierKind::CodeSymbol));
    }

    #[test]
    fn numeric_ids_fire() {
        assert_eq!(kind("1234567"), Some(IdentifierKind::NumericId));
        assert_eq!(
            kind("01KV4TA1DYAACCXVK53Z6NRPYN"),
            Some(IdentifierKind::NumericId)
        );
    }

    #[test]
    fn natural_language_does_not_fire() {
        // Precision guard: a battery of ordinary queries must stay NL.
        let nl = [
            "how does auth work",
            "what did we decide about caching",
            "fix the login timeout bug",
            "why is recall slow",
            "vector search performance",
            "login",            // single dictionary word, not a symbol
            "timeout",          // ditto
            "database",         // ditto
            "the auth flow",
            "memory consolidation strategy",
            "2026",             // year, not a numeric id
            "42",               // small count
            "1.2.3",            // version string, not a path/host
            "v1.5",             // version-ish
            "what is RRF",
            "explain the fusion algorithm",
            "store a decision",
            "Phase 20 recall discipline",
            "How do I run the tests?",
        ];
        for q in nl {
            assert_eq!(
                classify_query(q),
                QueryClass::NaturalLanguage,
                "query unexpectedly classified as identifier: {q:?} -> {}",
                classify_query(q).tag()
            );
        }
    }

    #[test]
    fn tags_are_stable() {
        assert_eq!(
            classify_query("550e8400-e29b-41d4-a716-446655440000").tag(),
            "identifier:uuid"
        );
        assert_eq!(classify_query("fn login").tag(), "identifier:symbol");
        assert_eq!(classify_query("src/db.rs").tag(), "identifier:path");
        assert_eq!(classify_query("how does it work").tag(), "natural-language");
    }

    #[test]
    fn empty_is_natural_language() {
        assert_eq!(classify_query(""), QueryClass::NaturalLanguage);
        assert_eq!(classify_query("   "), QueryClass::NaturalLanguage);
    }
}
