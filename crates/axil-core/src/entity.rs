//! Pattern-based entity extraction — no LLM required.
//!
//! Extracts entities from record text using deterministic patterns:
//! file paths, CamelCase/snake_case identifiers and backtick-wrapped code.
//! Also handles code symbols
//! (`fn foo`, `auth::login`, `class User`, …) with a language hint so
//! cross-language collisions can be disambiguated downstream. Declaration
//! keywords such as `trait` and `class` are English words too, so those
//! patterns only read code: fenced blocks, inline code, and code-like lines.

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ops::Range;
use std::sync::LazyLock;

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
    /// A name that fits no other type: what an LLM extractor returns for
    /// people and concepts. The patterns never produce it.
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
/// Patterns run in priority order, and a later pattern never mints a part of
/// a name an earlier one took at the same spot (`host_linker` inside
/// `WasmHost::host_linker`):
/// 1. Backtick-wrapped code: `` `AuthModule` `` → exact entity. A span with
///    spaces (a command line, an expression) is not one name; the patterns
///    below read the names inside it.
/// 2. Commands: `cargo test`, `axil recall`, `git worktree add` → code entity
/// 3. Long flags: `--dry-run`, `--top-k` (without a value) → code entity
/// 4. Qualified paths: `auth::login`, `axil-core::brain::MemoryType` → code symbol
/// 5. Git branches: `feat/open-backlog`, `origin/main` → code entity
/// 6. File paths: `/src/auth/login.rs` → file entity
/// 7. Hyphenated names: `recall-for-file`, `bge-m3`, `cargo-component` → code entity
/// 8. Constants and env vars: `BRUTE_FORCE_MAX`, `$GOMODCACHE`, `HOME` → code entity
/// 9. CamelCase identifiers: `AuthModule` → code entity
/// 10. snake_case identifiers: `auth_module` → code entity
/// 10. Declarations in code: `fn parse_token(` → code symbol
///
/// Quoted strings are not entities: a quoted phrase is an error message or an
/// example, and linking notes on it joins notes that merely quote the same
/// words.
pub fn extract_entities(text: &str) -> Vec<Entity> {
    let mut x = Extraction::new(text);
    extract_backtick_entities(&mut x);
    extract_commands(&mut x);
    extract_long_flags(&mut x);
    extract_qualified_paths(&mut x);
    extract_branches(&mut x);
    extract_file_paths(&mut x);
    extract_hyphenated_names(&mut x);
    extract_constants(&mut x);
    extract_camel_case(&mut x);
    extract_snake_case(&mut x);
    extract_declarations(&mut x);
    x.entities
}

/// One extraction in progress: what has been found so far, and which parts
/// of the text are already taken.
struct Extraction<'a> {
    text: &'a str,
    entities: Vec<Entity>,
    /// Dedup keys of the entities found so far.
    seen: HashSet<String>,
    /// Byte ranges of `text` that hold a whole name. A shorter pattern that
    /// matches inside one would mint a fragment of that name, so the
    /// identifier patterns skip them; the same word elsewhere in the text
    /// still counts.
    claimed: Vec<Range<usize>>,
    /// Byte ranges that are code by their markup: inline code spans and the
    /// lines of fenced blocks.
    code: Vec<Range<usize>>,
}

impl<'a> Extraction<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            entities: Vec::new(),
            seen: HashSet::new(),
            claimed: Vec::new(),
            code: Vec::new(),
        }
    }

    fn is_code(&self, at: usize) -> bool {
        self.code.iter().any(|c| c.contains(&at))
    }

    /// The byte range of `part` within the text. `part` must be a slice of
    /// the text (every pattern here matches on such slices).
    fn range_of(&self, part: &str) -> Range<usize> {
        let base = self.text.as_ptr() as usize;
        let start = part.as_ptr() as usize - base;
        debug_assert!(start + part.len() <= self.text.len());
        start..start + part.len()
    }

    fn is_claimed(&self, range: &Range<usize>) -> bool {
        self.claimed
            .iter()
            .any(|c| c.start < range.end && range.start < c.end)
    }

    fn claim(&mut self, range: Range<usize>) {
        self.claimed.push(range);
    }

    /// Keep `entity` unless an entity with the same `key` was found already.
    fn push(&mut self, key: String, entity: Entity) {
        if self.seen.insert(key) {
            self.entities.push(entity);
        }
    }
}

fn extract_backtick_entities(x: &mut Extraction<'_>) {
    for (line, fenced) in lines_with_fences(x.text) {
        // A fenced block is code to read, not a name; only the command
        // pattern reads it.
        if fenced {
            x.code.push(x.range_of(line));
            continue;
        }
        for span in inline_code_spans(line) {
            x.code.push(x.range_of(span));
            let trimmed = span.trim();
            if trimmed.contains(char::is_whitespace) {
                continue;
            }
            let name = trim_code_name(trimmed);
            // Markdown syntax quoted in backticks (`- [ ]`, `**`) names
            // nothing, and neither does a keyword or a literal (`true`, `0`).
            if name.len() > 200 || !name.chars().any(char::is_alphanumeric) || is_code_literal(name)
            {
                continue;
            }
            x.claim(x.range_of(trimmed));
            let entity_type = if is_file_like(name) {
                EntityType::File
            } else {
                EntityType::Code
            };
            x.push(
                name.to_lowercase(),
                Entity {
                    name: normalize_entity_name(name),
                    entity_type,
                    source_text: name.to_string(),
                },
            );
        }
    }
}

/// The name an inline code span holds: without a `$` sigil (`$HOME`), call
/// arguments (`Axil::open(path)`) or a `:line` suffix (`db.rs:611`).
fn trim_code_name(span: &str) -> &str {
    let name = span.strip_prefix('$').unwrap_or(span);
    let name = match name.find('(') {
        Some(open) if open > 0 && name.ends_with(')') => &name[..open],
        _ => name,
    };
    match LINE_SUFFIX.find(name) {
        Some(m) if m.start() > 0 => &name[..m.start()],
        _ => name,
    }
}

/// A `:line` or `:line-line` suffix after a file name.
static LINE_SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r":\d+(?:-\d+)?$").unwrap());

/// Keywords, basic types, literals and generic method names: code that
/// names nothing a note would link on, even in backticks.
const CODE_LITERALS: &[&str] = &[
    "arc", "async", "await", "bool", "box", "build", "char", "class", "const", "def", "default",
    "dyn", "enum", "err", "f32", "f64", "false", "fn", "func", "function", "get", "i128", "i16",
    "i32", "i64", "i8", "impl", "isize", "let", "main", "mod", "mut", "new", "nil", "none", "null",
    "ok", "open", "option", "parse", "pub", "rc", "result", "run", "self", "set", "some", "str",
    "string", "struct", "trait", "true", "u128", "u16", "u32", "u64", "u8", "usize", "vec", "with",
];

fn is_code_literal(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let digits = lower.replace('_', "");
    CODE_LITERALS.binary_search(&lower.as_str()).is_ok()
        || digits.parse::<f64>().is_ok()
        || digits
            .strip_prefix("0x")
            .is_some_and(|hex| !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()))
}

// ── Commands ──────────────────────────────────────────────────

/// A program whose command lines notes quote.
struct Program {
    name: &'static str,
    /// Its subcommands, sorted. Outside code markup a word after the
    /// program counts only if listed: "cargo resolves the lockfile" is
    /// prose, "cargo test passes" names `cargo test`.
    subcommands: &'static [&'static str],
    /// Subcommands that take one more subcommand word (`git worktree add`),
    /// sorted.
    groups: &'static [&'static str],
}

/// Every top-level `axil` subcommand, sorted (a CLI test keeps it in step
/// with the command tree).
pub const AXIL_COMMANDS: &[&str] = &[
    "about",
    "activity",
    "add-vector",
    "agg",
    "analytics",
    "ask",
    "auto-capture",
    "auto-link",
    "batch-insert",
    "belief-history",
    "beliefs",
    "believe",
    "bench",
    "boot",
    "brain",
    "brain-banner",
    "brain-eval",
    "branch",
    "bridge",
    "brief",
    "cache",
    "capture",
    "checkpoint",
    "close-session",
    "code-context",
    "code-recall-bench",
    "code-search",
    "compact",
    "config",
    "connections",
    "consent",
    "consolidate",
    "context",
    "context-savings",
    "contradictions",
    "create",
    "decay",
    "delete",
    "dep-docs",
    "deps",
    "detect",
    "detect-conflicts",
    "diff",
    "doctor",
    "doubt",
    "edges",
    "embed",
    "entity-alias",
    "entity-history",
    "entity-merge",
    "entity-merge-canonical",
    "entity-resolve",
    "entity-resolve-scoped",
    "episodes",
    "episodes-similar",
    "explain",
    "explain-code-hit",
    "export",
    "ext",
    "extensions",
    "extract-entities",
    "fact-confirm",
    "fact-reject",
    "features",
    "fts",
    "fts-optimize",
    "get",
    "heal",
    "health-report",
    "history",
    "hook",
    "how",
    "impact",
    "import",
    "importance",
    "index",
    "index-stats",
    "index-text",
    "infer",
    "info",
    "ingest",
    "ingest-scip",
    "init",
    "insert",
    "inspect-memory",
    "install",
    "know",
    "know-about",
    "know-list",
    "learn",
    "lineage",
    "link",
    "list",
    "llm",
    "log",
    "maintain",
    "mcp",
    "memory-policy",
    "memory-pressure",
    "metrics-snapshot",
    "migrate-provenance",
    "model-add",
    "model-bench",
    "model-download",
    "model-list",
    "model-remove",
    "neighbors",
    "observe",
    "patterns",
    "pin",
    "prefer",
    "prefetch",
    "profile",
    "project-model",
    "ql",
    "query",
    "recall",
    "recall-across",
    "recall-for-entity",
    "recall-for-file",
    "recall-index",
    "redact",
    "reembed",
    "reflect",
    "refresh",
    "reindex",
    "remember",
    "report",
    "resolve",
    "retention",
    "retro",
    "revise-beliefs",
    "rule",
    "rules",
    "schedule",
    "scip",
    "search",
    "search-vector",
    "self",
    "serve",
    "session",
    "session-heal",
    "similar",
    "since",
    "skill",
    "slow-queries",
    "snapshot",
    "stats",
    "store",
    "sync",
    "tables",
    "timeline",
    "trace-memory",
    "trace-record",
    "traverse",
    "trends",
    "ttl-clear",
    "ttl-set",
    "unlink",
    "unpin",
    "update",
    "user-contract",
    "verify",
    "warm-up",
    "why",
    "why-fact",
    "why-recalled",
    "why-remembered",
    "why-revised",
    "worker",
    "workspace",
];

/// The `axil` subcommands that take a subcommand of their own, sorted.
pub const AXIL_COMMAND_GROUPS: &[&str] = &[
    "brain",
    "branch",
    "bridge",
    "cache",
    "capture",
    "config",
    "consent",
    "contradictions",
    "deps",
    "ext",
    "extensions",
    "hook",
    "llm",
    "mcp",
    "project-model",
    "report",
    "retention",
    "rule",
    "rules",
    "schedule",
    "scip",
    "self",
    "session",
    "skill",
    "user-contract",
    "worker",
    "workspace",
];

static PROGRAMS: &[Program] = &[
    Program {
        name: "axil",
        subcommands: AXIL_COMMANDS,
        groups: AXIL_COMMAND_GROUPS,
    },
    Program {
        name: "brew",
        subcommands: &["info", "install", "list", "uninstall", "update", "upgrade"],
        groups: &[],
    },
    Program {
        name: "cargo",
        subcommands: &[
            "add",
            "audit",
            "bench",
            "build",
            "check",
            "clean",
            "clippy",
            "component",
            "deny",
            "doc",
            "expand",
            "fetch",
            "fix",
            "fmt",
            "generate-lockfile",
            "init",
            "install",
            "llvm-cov",
            "metadata",
            "new",
            "nextest",
            "package",
            "publish",
            "remove",
            "run",
            "rustc",
            "rustdoc",
            "search",
            "semver-checks",
            "test",
            "tree",
            "uninstall",
            "update",
            "vendor",
            "yank",
        ],
        groups: &[],
    },
    Program {
        name: "codex",
        subcommands: &["exec", "login", "review"],
        groups: &[],
    },
    Program {
        name: "docker",
        subcommands: &[
            "build", "compose", "exec", "images", "logs", "ps", "pull", "push", "rm", "run", "stop",
        ],
        groups: &["compose"],
    },
    Program {
        name: "gh",
        subcommands: &[
            "api", "auth", "browse", "cache", "gist", "issue", "label", "pr", "release", "repo",
            "run", "search", "secret", "workflow",
        ],
        groups: &[
            "auth", "cache", "gist", "issue", "label", "pr", "release", "repo", "run", "secret",
            "workflow",
        ],
    },
    Program {
        name: "git",
        subcommands: &[
            "add",
            "am",
            "apply",
            "archive",
            "bisect",
            "blame",
            "branch",
            "checkout",
            "cherry-pick",
            "clean",
            "clone",
            "commit",
            "config",
            "describe",
            "diff",
            "fetch",
            "format-patch",
            "gc",
            "grep",
            "init",
            "log",
            "ls-files",
            "merge",
            "mv",
            "notes",
            "pull",
            "push",
            "rebase",
            "reflog",
            "remote",
            "reset",
            "restore",
            "rev-parse",
            "revert",
            "rm",
            "shortlog",
            "show",
            "stash",
            "status",
            "submodule",
            "switch",
            "tag",
            "worktree",
        ],
        groups: &[
            "bisect",
            "notes",
            "remote",
            "stash",
            "submodule",
            "worktree",
        ],
    },
    Program {
        name: "go",
        subcommands: &[
            "build", "fmt", "generate", "get", "install", "mod", "run", "test", "vet", "work",
        ],
        groups: &["mod", "work"],
    },
    Program {
        name: "npm",
        subcommands: &[
            "audit", "ci", "exec", "init", "install", "link", "pack", "publish", "run", "start",
            "test", "update",
        ],
        groups: &["run"],
    },
    Program {
        name: "opencode",
        subcommands: &["run"],
        groups: &[],
    },
    Program {
        name: "pip",
        subcommands: &[
            "download",
            "freeze",
            "install",
            "list",
            "show",
            "uninstall",
            "wheel",
        ],
        groups: &[],
    },
    Program {
        name: "pip3",
        subcommands: &[
            "download",
            "freeze",
            "install",
            "list",
            "show",
            "uninstall",
            "wheel",
        ],
        groups: &[],
    },
    Program {
        name: "pnpm",
        subcommands: &[
            "add", "build", "exec", "install", "publish", "remove", "run", "test",
        ],
        groups: &["run"],
    },
    Program {
        name: "python",
        subcommands: &[],
        groups: &[],
    },
    Program {
        name: "python3",
        subcommands: &[],
        groups: &[],
    },
    Program {
        name: "rg",
        subcommands: &[],
        groups: &[],
    },
    Program {
        name: "rustfmt",
        subcommands: &[],
        groups: &[],
    },
    Program {
        name: "rustup",
        subcommands: &[
            "component",
            "default",
            "install",
            "override",
            "show",
            "target",
            "toolchain",
            "update",
        ],
        groups: &["component", "target", "toolchain"],
    },
    Program {
        name: "uv",
        subcommands: &[
            "add", "build", "init", "lock", "pip", "publish", "python", "remove", "run", "sync",
            "tool", "venv",
        ],
        groups: &["pip", "python", "tool"],
    },
    Program {
        name: "yarn",
        subcommands: &["add", "build", "install", "remove", "run", "test"],
        groups: &["run"],
    },
];

/// A program name at the start of a word, with what follows it on the line.
static PROGRAM_WORD: LazyLock<Regex> = LazyLock::new(|| {
    let names: Vec<&str> = PROGRAMS.iter().map(|p| p.name).collect();
    let pattern = format!(
        r#"(?m)(?:^|[\s`(\[{{"'])({})(?:[ \t]+|$|`)"#,
        names.join("|")
    );
    Regex::new(&pattern).unwrap()
});

/// One subcommand word: lowercase, digits and inner hyphens, ended by
/// whitespace, closing markup or sentence punctuation.
static SUBCOMMAND_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?:$|[\s`)\],.;:!?])").unwrap());

/// Commands: a program and its subcommand words, up to the first flag,
/// argument or placeholder (`cargo test -p x` gives `cargo test`). In code
/// markup any subcommand-shaped word counts and a program alone is a
/// command (`python3 run.py`); in prose only listed subcommands count.
fn extract_commands(x: &mut Extraction<'_>) {
    let text = x.text;
    for caps in PROGRAM_WORD.captures_iter(text) {
        let Some(m) = caps.get(1) else { continue };
        let Some(program) = PROGRAMS.iter().find(|p| p.name == m.as_str()) else {
            continue;
        };
        let in_code = x.is_code(m.start());
        let mut end = m.end();
        let mut words = 0;
        while words < 2 {
            let rest = &text[end..];
            let gap = rest.len() - rest.trim_start_matches([' ', '\t']).len();
            let Some(found) = (gap > 0)
                .then(|| SUBCOMMAND_WORD.find(&rest[gap..]))
                .flatten()
            else {
                break;
            };
            let word = found
                .as_str()
                .trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
            let counts = if words == 0 {
                !program.subcommands.is_empty()
                    && (in_code || program.subcommands.binary_search(&word).is_ok())
            } else {
                !PROSE_WORDS.contains(&word)
            };
            if !counts {
                break;
            }
            end += gap + word.len();
            words += 1;
            if program.groups.binary_search(&word).is_err() {
                break;
            }
        }
        if words == 0 && !in_code {
            continue;
        }
        let command = &text[m.start()..end];
        if x.is_claimed(&(m.start()..end)) {
            continue;
        }
        x.claim(m.start()..end);
        let name = command.split_whitespace().collect::<Vec<_>>().join(" ");
        x.push(
            name.to_lowercase(),
            Entity {
                name: name.clone(),
                entity_type: EntityType::Code,
                source_text: name,
            },
        );
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

/// A git branch or remote ref: a work-type prefix (`feat/`, `fix/`,
/// `review/`, ...) or a remote (`origin/`, `upstream/`), then a name.
/// `docs/`, `test/` and `build/` are left to the path pattern, since they
/// open directories far more often than branches.
static BRANCH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:^|[\s`(\[{"',])((?:bench|bugfix|chore|ci|exp|feat|feature|fix|hotfix|origin|perf|refactor|release|review|spike|upstream|wip)/[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*)"#,
    )
    .unwrap()
});

fn extract_branches(x: &mut Extraction<'_>) {
    let text = x.text;
    for caps in BRANCH.captures_iter(text) {
        let Some(m) = caps.get(1) else { continue };
        let name = m.as_str().trim_end_matches(['.', '-', '_']);
        let range = m.start()..m.start() + name.len();
        // A file under such a directory is a path, not a branch.
        if x.is_claimed(&range) || is_file_name(name.rsplit('/').next().unwrap_or(name)) {
            continue;
        }
        x.claim(range);
        x.push(
            name.to_lowercase(),
            Entity {
                name: name.to_string(),
                entity_type: EntityType::Code,
                source_text: name.to_string(),
            },
        );
    }
}

fn extract_file_paths(x: &mut Extraction<'_>) {
    let text = x.text;
    for word in text.split_whitespace() {
        let trimmed = trim_path_word(word);
        let trimmed = match LINE_SUFFIX.find(trimmed) {
            Some(m) if m.start() > 0 => &trimmed[..m.start()],
            _ => trimmed,
        };
        // A slash list of file names (`lib.rs/main.rs`) names each file.
        let parts: Vec<&str> = trimmed.split('/').collect();
        if parts.len() > 1 && parts.iter().all(|p| is_file_name(p)) {
            for part in parts {
                push_file(x, part);
            }
            continue;
        }
        let range = x.range_of(trimmed);
        if trimmed.is_empty() || x.is_claimed(&range) {
            continue;
        }
        let next_word = text[range.end..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches(|c: char| !c.is_alphanumeric());
        if looks_like_path(trimmed)
            || is_file_name(trimmed)
            || is_directory(trimmed)
            || is_companion_file(trimmed, next_word)
        {
            push_file(x, trimmed);
        }
    }
}

fn push_file(x: &mut Extraction<'_>, path: &str) {
    x.claim(x.range_of(path));
    x.push(
        path.to_lowercase(),
        Entity {
            name: normalize_file_path(path),
            entity_type: EntityType::File,
            source_text: path.to_string(),
        },
    );
}

/// Extensions of files a project keeps, sorted. A name with one of these
/// is a file without a slash (`db.rs`, `axil.toml`, `memory.axil.ts`);
/// any other dotted word (`crates.io`, `e.g`, `v1.2`) is not.
const FILE_EXTENSIONS: &[&str] = &[
    "axil", "bash", "bin", "c", "cc", "cfg", "cjs", "conf", "cpp", "css", "csv", "db", "dll",
    "dylib", "exe", "fts", "gif", "go", "graph", "gz", "h", "hpp", "html", "ini", "ipynb", "java",
    "jpg", "js", "json", "jsonl", "jsx", "kt", "lock", "log", "md", "mjs", "mv2", "onnx",
    "parquet", "pdf", "php", "png", "proto", "py", "rb", "rs", "scip", "scss", "sh", "so", "sql",
    "svelte", "svg", "swift", "tar", "toml", "ts", "tsv", "tsx", "txt", "vec", "vue", "wasm",
    "wit", "xml", "yaml", "yml", "zip", "zsh",
];

/// Dot-files named without a stem, sorted.
const DOT_FILES: &[&str] = &[
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".gitattributes",
    ".gitignore",
    ".gitmodules",
    ".npmrc",
];

/// The companion files of an Axil database, named by suffix alone ("the
/// .vec store"). `.ts` is also TypeScript's extension, so it counts only
/// before "store" or "file".
const COMPANION_SUFFIXES: &[&str] = &[".axil", ".fts", ".graph", ".vec"];

/// Directories that open a project-relative path, sorted. A slash path
/// without an extension is a file path only when it starts with one of
/// these, a dot-directory or `~` (`crates/adapters/axil-cli`), since
/// `decisions/errors/context` and `Linux/Windows` are lists of words.
const PROJECT_DIRS: &[&str] = &[
    "apps",
    "assets",
    "benches",
    "benchmarks",
    "bin",
    "build",
    "config",
    "crates",
    "dist",
    "doc",
    "docs",
    "examples",
    "lib",
    "node_modules",
    "packages",
    "scripts",
    "src",
    "target",
    "tasks",
    "test",
    "tests",
    "tmp",
    "tools",
    "vendor",
];

/// Characters a file or directory name is made of; `*` and `<…>` keep a
/// glob or a path template (`hubness-<date>.json`) whole.
fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '*' | '<' | '>' | '~')
}

/// A file name without a directory: a stem with a letter, then a known
/// extension (`db.rs`, `.axil.vec`, `memory_*.rs`), or a known dot-file.
fn is_file_name(s: &str) -> bool {
    if DOT_FILES.contains(&s) {
        return true;
    }
    let Some(dot) = s.rfind('.') else {
        return false;
    };
    let (stem, ext) = (&s[..dot], &s[dot + 1..]);
    FILE_EXTENSIONS.binary_search(&ext).is_ok()
        && stem.chars().any(|c| c.is_ascii_alphabetic())
        && !stem.starts_with('-')
        && stem.chars().all(is_path_char)
}

/// A directory: a path ending in `/` (`target/`, `benchmarks/results/`), or
/// a slash path without an extension that starts at a project directory, a
/// dot-directory or `~`.
fn is_directory(s: &str) -> bool {
    let segments: Vec<&str> = s.trim_end_matches('/').split('/').collect();
    let first = segments[0];
    let well_formed = segments
        .iter()
        .all(|seg| !seg.is_empty() && seg.chars().all(is_path_char))
        && first.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 2;
    if !well_formed {
        return false;
    }
    if s.ends_with('/') {
        return true;
    }
    // `build/test` and `docs/CI` are word lists even though each word can
    // name a directory, so a project directory needs a third segment or a
    // segment no plain word has.
    let name_like = segments.len() > 2
        || segments[1..]
            .iter()
            .any(|seg| seg.contains(|c: char| !c.is_ascii_alphabetic()));
    segments.len() > 1
        && (first.starts_with('.')
            || first.starts_with('~')
            || PROJECT_DIRS.binary_search(&first).is_ok() && name_like)
}

/// Whether one word names a file or directory by any of the file patterns.
fn is_file_like(s: &str) -> bool {
    looks_like_path(s) || is_file_name(s) || is_directory(s) || is_companion_file(s, "")
}

/// An Axil companion file named by its suffix (`.vec`, `.axil.graph`).
fn is_companion_file(s: &str, next_word: &str) -> bool {
    COMPANION_SUFFIXES.contains(&s) || s == ".ts" && matches!(next_word, "store" | "file")
}

/// A hyphenated word: letters first, then `-`-joined segments that may hold
/// digits, dots and underscores (`nli-deberta-v3-xsmall`, `glm-5.3-flash`,
/// `x86_64-apple-darwin`).
static HYPHENATED_WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s`(\[{"',])([A-Za-z][A-Za-z0-9_.]*(?:-[A-Za-z0-9_.]+)+)"#).unwrap()
});

/// A word and a number or version (`Tier-1`, `BUG-10`, `utf-8`, `top-5`,
/// `post-1.2.0`): a plan label, a measurement or a release, not a name.
static WORD_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z]+(?:-v?\d+(?:\.\d+)*[a-z]?)+$").unwrap());

/// Hyphenated names. Most hyphenated words are English compounds
/// (`free-form`, `one-shot`), so only three shapes count: an `axil`
/// subcommand (`recall-for-file`), a name with a version or size in it
/// (`bge-m3`, `gpt-4o-mini`, `axildb-v1.2.0`), and a `cargo-` or `scip-`
/// tool (`cargo-component`, `scip-go`).
fn extract_hyphenated_names(x: &mut Extraction<'_>) {
    let text = x.text;
    for caps in HYPHENATED_WORD.captures_iter(text) {
        let Some(m) = caps.get(1) else { continue };
        let word = m.as_str().trim_end_matches(['.', '_']);
        let range = m.start()..m.start() + word.len();
        // A range of labels (`BUG-1..8`) is bookkeeping.
        if x.is_claimed(&range) || text[range.end..].starts_with("::") || word.contains("..") {
            continue;
        }
        let versioned = word.contains(|c: char| c.is_ascii_digit()) && !WORD_NUMBER.is_match(word);
        let tool = word.starts_with("cargo-") || word.starts_with("scip-");
        let subcommand = AXIL_COMMANDS.binary_search(&word).is_ok();
        if !(versioned || tool || subcommand) {
            continue;
        }
        x.claim(range);
        x.push(
            word.to_lowercase(),
            Entity {
                name: word.to_string(),
                entity_type: EntityType::Code,
                source_text: word.to_string(),
            },
        );
    }
}

/// An ALL_CAPS name joined by underscores (`CARGO_REGISTRY_TOKEN`), or any
/// ALL_CAPS name after a `$` sigil (`$GOMODCACHE`, `${HOME}`).
static CONSTANT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\$\{?([A-Z][A-Z0-9_]*[A-Z0-9])\b|\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b").unwrap()
});

/// Environment variables whose names have no underscore, sorted. Other
/// ALL_CAPS words without one are emphasis (`NOT FIXED`, `DONE`).
const ENV_VARS: &[&str] = &[
    "EDITOR", "HOME", "LANG", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "USER",
];

static CAPS_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[A-Z]{2,}\b").unwrap());

/// Constants and environment variables, without a `$` sigil.
fn extract_constants(x: &mut Extraction<'_>) {
    let text = x.text;
    let sigiled = CONSTANT
        .captures_iter(text)
        .filter_map(|caps| caps.get(1).or_else(|| caps.get(2)));
    let known = CAPS_WORD
        .find_iter(text)
        .filter(|m| ENV_VARS.binary_search(&m.as_str()).is_ok());
    let found: Vec<regex::Match<'_>> = sigiled.chain(known).collect();
    for m in found {
        if x.is_claimed(&m.range()) {
            continue;
        }
        x.claim(m.range());
        x.push(
            m.as_str().to_lowercase(),
            Entity {
                name: m.as_str().to_string(),
                entity_type: EntityType::Code,
                source_text: m.as_str().to_string(),
            },
        );
    }
}

fn extract_camel_case(x: &mut Extraction<'_>) {
    for word in split_into_identifiers(x.text) {
        if is_camel_case(word) && word.len() >= 4 && !x.is_claimed(&x.range_of(word)) {
            x.push(
                word.to_lowercase(),
                Entity {
                    name: camel_to_snake(word),
                    entity_type: EntityType::Code,
                    source_text: word.to_string(),
                },
            );
        }
    }
}

fn extract_snake_case(x: &mut Extraction<'_>) {
    for word in split_into_identifiers(x.text) {
        if is_snake_case(word)
            && word.len() >= 4
            && word.contains('_')
            && !x.is_claimed(&x.range_of(word))
        {
            x.push(
                word.to_lowercase(),
                Entity {
                    name: word.to_lowercase(),
                    entity_type: EntityType::Code,
                    source_text: word.to_string(),
                },
            );
        }
    }
}

/// Strip the punctuation prose wraps around a path. A trailing `.` ends the
/// sentence, but a leading one names a dot-directory or dot-file
/// (`.claude/CLAUDE.md`, `./src`), so only an ellipsis is stripped there.
fn trim_path_word(word: &str) -> &str {
    let wrapper = |c: char| {
        matches!(
            c,
            ',' | ';' | ':' | ')' | '(' | '"' | '\'' | '`' | '[' | ']'
        )
    };
    // `**` is bold markup; a lone `*` belongs to a glob (`docs/*`).
    let word = word.trim_start_matches("**").trim_end_matches("**");
    let start = word.trim_start_matches(wrapper);
    let start = if start.starts_with("...") {
        start.trim_start_matches('.')
    } else {
        start
    };
    start.trim_end_matches(|c: char| wrapper(c) || c == '.')
}

/// Check if a string looks like a file path.
fn looks_like_path(s: &str) -> bool {
    if s.len() < 3 {
        return false;
    }
    // Must start with / or ./ or ../ or contain /
    let has_path_prefix =
        s.starts_with('/') || s.starts_with("./") || s.starts_with("../") || s.starts_with("~/");
    let has_extension = s
        .rfind('.')
        .map(|dot| {
            let ext = &s[dot + 1..];
            !ext.is_empty()
                && ext.len() <= 10
                && ext.chars().all(|c| c.is_alphanumeric())
                && ext.chars().any(|c| c.is_alphabetic())
        })
        .unwrap_or(false);
    // A query string or an assignment is not a path (`/recall?q=x`).
    if s.contains(['?', '&', '=']) {
        return false;
    }

    // Path-like if it has a path prefix, or has both slashes and an
    // extension; numbers with slashes (`3.1/3.2`) have no extension.
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

/// Check if a string is snake_case. One leading underscore is allowed, for
/// system tables and private fields (`_entities`, `_scip_aliases`).
fn is_snake_case(s: &str) -> bool {
    let body = s.strip_prefix('_').unwrap_or(s);
    body.chars()
        .all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_')
        && body.starts_with(|c: char| c.is_lowercase())
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

/// A long flag where a word starts: `--top-k`, `--no-default-features`.
/// A value (`--format=json`) is not part of it, and `--` alone or a
/// markdown rule (`---`) is not a flag.
static LONG_FLAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s`(\[{"',/|])(--[a-z0-9]+(?:[-_][a-z0-9]+)*)\b"#).unwrap()
});

fn extract_long_flags(x: &mut Extraction<'_>) {
    let text = x.text;
    for caps in LONG_FLAG.captures_iter(text) {
        let Some(m) = caps.get(1) else { continue };
        if x.is_claimed(&m.range()) {
            continue;
        }
        x.claim(m.range());
        x.push(
            m.as_str().to_string(),
            Entity {
                name: m.as_str().to_string(),
                entity_type: EntityType::Code,
                source_text: m.as_str().to_string(),
            },
        );
    }
}

// ── Language-aware code-symbol extraction ─────────────────────

/// A single code-symbol pattern: language hint + compiled regex.
/// Capture group 1 is the symbol name.
struct SymbolPattern {
    lang: &'static str,
    re: Regex,
}

/// Rust-qualified paths (`auth::login`, `Mod::Ty::method`). `::` does not
/// occur in English, so this pattern reads all text. The first segment may be
/// a hyphenated crate name (`axil-core::brain`), which a path names in prose
/// even though Rust code spells it with underscores.
static QUALIFIED_PATH: LazyLock<SymbolPattern> = LazyLock::new(|| SymbolPattern {
    lang: "rust",
    re: Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b")
        .unwrap(),
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

/// A line that opens with a declaration: a keyword, then a name followed
/// right away by code punctuation (`pub fn parse(`, `struct Pool<`,
/// `class User:`, `trait Store: Send`, `struct Unit;`), or a Go method
/// receiver (`func (s *Server)`).
static DECLARATION_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:(?:pub(?:\([^)]*\))?|export|default|async|unsafe|const|public|private|protected|internal|static|abstract|final|override)\s+)*(?:func\s*\(|(?:fn|struct|impl|trait|enum|def|class|function|func|interface)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*(?:[(<{:=;]|->))",
    )
    .unwrap()
});

/// English words that follow a declaration keyword in prose ("a trait with
/// …", "the class that …"). A match on a prose line that names one of these
/// is prose, not a symbol; code (`fn with(self) -> Self`) keeps it.
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
    /// From a prose line: a match counts only when its name is shaped like
    /// an identifier (`impl Decoder`, `fn parse_token`) or is called or
    /// parameterized right away (`def login(u)`, `struct Pool<T>`), and is
    /// not one of the [`PROSE_WORDS`]. See [`prose_declaration_counts`].
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
                prose: false,
            });
            continue;
        }
        for span in inline_code_spans(line) {
            segments.push(CodeSegment {
                text: span,
                prose: false,
            });
        }
        segments.push(CodeSegment {
            text: line,
            prose: true,
        });
    }
    segments
}

/// Whether a line outside a fenced block reads as code: it opens with a
/// declaration (see [`DECLARATION_LINE`]). Punctuation alone proves nothing;
/// prose ends lines with `;` and `:` too.
fn is_code_like_line(line: &str) -> bool {
    DECLARATION_LINE.is_match(line.trim())
}

fn extract_qualified_paths(x: &mut Extraction<'_>) {
    let text = x.text;
    for caps in QUALIFIED_PATH.re.captures_iter(text) {
        let Some(m) = caps.get(1) else { continue };
        let mut end = m.end();
        // A macro keeps its `!` (`bindgen!`); `a::b != c` compares.
        if text[end..].starts_with('!') && !text[end + 1..].starts_with('=') {
            end += 1;
        }
        let path = &text[m.start()..end];
        x.claim(m.start()..end);
        push_code_symbol(x, QUALIFIED_PATH.lang, path);
    }
}

fn extract_declarations(x: &mut Extraction<'_>) {
    let segments = code_segments(x.text);
    for pattern in DECLARATION_PATTERNS.iter() {
        for segment in &segments {
            for caps in pattern.re.captures_iter(segment.text) {
                let Some(m) = caps.get(1) else { continue };
                if segment.prose && !prose_declaration_counts(pattern, segment.text, m) {
                    continue;
                }
                push_code_symbol(x, pattern.lang, m.as_str());
            }
        }
    }
}

/// Whether a declaration match on a prose line names code. Prose puts plain
/// words after `trait`, `class` and `function` ("a trait with …"), so the
/// name must be called or parameterized right away (`def login(u)`,
/// `struct Pool<T>`), or be shaped like an identifier: a capital, an
/// underscore or a digit (`impl Decoder`, `fn parse_token`). Either way it
/// must not be one of the [`PROSE_WORDS`] ("a trait with(…)"). The Java
/// pattern spans several words before the name, so on prose it only takes
/// the call shape.
fn prose_declaration_counts(pattern: &SymbolPattern, line: &str, name: regex::Match<'_>) -> bool {
    if PROSE_WORDS.contains(&name.as_str().to_ascii_lowercase().as_str()) {
        return false;
    }
    if line[name.end()..].starts_with(['(', '<']) {
        return true;
    }
    pattern.lang != "java"
        && name
            .as_str()
            .chars()
            .any(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
}

fn push_code_symbol(x: &mut Extraction<'_>, lang: &'static str, name: &str) {
    if name.len() < 2 || name.len() > 200 {
        return;
    }
    // Collision key is scoped by language so `login` (Rust) and
    // `login` (Python) don't silently merge.
    x.push(
        format!("sym:{}:{}", lang, name.to_lowercase()),
        Entity {
            name: name.to_string(),
            entity_type: EntityType::CodeSymbol {
                lang_hint: Some(lang.to_string()),
            },
            source_text: name.to_string(),
        },
    );
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
    fn file_paths_keep_a_leading_dot() {
        let text = "Read .claude/CLAUDE.md, then (.axil/memory.axil.vec). See ../hooks/x.sh too.";
        let files: Vec<String> = extract_entities(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::File)
            .map(|e| e.source_text)
            .collect();
        for path in [
            ".claude/CLAUDE.md",
            ".axil/memory.axil.vec",
            "../hooks/x.sh",
        ] {
            assert!(files.iter().any(|f| f == path), "{path}: {files:?}");
        }
        // An ellipsis before a path is prose, not part of the name.
        let files: Vec<String> = extract_entities("and so on ...src/main.rs")
            .into_iter()
            .map(|e| e.source_text)
            .collect();
        assert_eq!(files, vec!["src/main.rs".to_string()]);
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
    fn quoted_phrases_are_not_entities() {
        for text in [
            "Error message was \"connection timed out unexpectedly\"",
            "It printed 'no summary available' and the user's notes don't say why",
        ] {
            let entities = extract_entities(text);
            assert!(entities.is_empty(), "{text:?} minted {entities:?}");
        }
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
    fn punctuation_does_not_make_prose_code() {
        for text in [
            "The trait with a default method;",
            "trait with a default method;",
            "We kept the class that wins {",
            "a function returns early }",
            "enum with three cases: on, off, auto",
        ] {
            let syms = code_syms(text);
            assert!(syms.is_empty(), "{text:?} minted {syms:?}");
        }
        // A declaration line is code, so its names count as written.
        let syms = code_syms("struct Unit;\ntrait Store: Send + Sync\nfn is(self) -> bool;");
        for name in ["Unit", "Store", "is"] {
            assert!(syms.iter().any(|(n, _)| n == name), "{name}: {syms:?}");
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
    fn code_outside_fences_keeps_names_that_are_english_words() {
        // An unfenced signature, a code-like line and an inline span are code:
        // a method named `with` or `is` is a real symbol there.
        for text in [
            "Builder method:\nfn with(self) -> Self",
            "Builder method:\n    pub fn with(self, n: usize) -> Self {",
            "Call `fn with(self)` last.",
        ] {
            let syms = code_syms(text);
            assert!(
                syms.iter()
                    .any(|(n, l)| n == "with" && l.as_deref() == Some("rust")),
                "{text:?} lost `with`: {syms:?}"
            );
        }
        // On a prose line the same word stays prose, even in call shape.
        let syms = code_syms("Then the builder takes a trait with(care) in mind");
        assert!(!syms.iter().any(|(n, _)| n == "with"), "{syms:?}");
    }

    fn sources(text: &str) -> Vec<String> {
        extract_entities(text)
            .into_iter()
            .map(|e| e.source_text)
            .collect()
    }

    #[test]
    fn parts_of_a_name_are_not_minted_beside_it() {
        let found = sources(
            "WasmHost::host_linker() builds it; see crates/axil-core/benches/core_benchmarks.rs \
             and `auth_config`.",
        );
        for part in ["WasmHost", "host_linker", "core_benchmarks"] {
            assert!(!found.iter().any(|f| f == part), "{part}: {found:?}");
        }
        assert!(
            found.iter().any(|f| f == "WasmHost::host_linker"),
            "{found:?}"
        );
        // The same word on its own elsewhere in the text is still a mention.
        let found = sources("GraphEngine::open fails, so GraphEngine retries.");
        assert!(found.iter().any(|f| f == "GraphEngine"), "{found:?}");
    }

    #[test]
    fn qualified_paths_keep_a_hyphenated_crate_and_a_macro_bang() {
        let found = sources(
            "axil-runtime::atomic_write calls wasmtime::component::bindgen! once; \
             a::b != c is a comparison.",
        );
        for name in [
            "axil-runtime::atomic_write",
            "wasmtime::component::bindgen!",
            "a::b",
        ] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for fragment in ["runtime::atomic_write", "atomic_write", "a::b!"] {
            assert!(
                !found.iter().any(|f| f == fragment),
                "{fragment}: {found:?}"
            );
        }
    }

    #[test]
    fn word_lists_are_sorted_for_binary_search() {
        let mut lists: Vec<(&str, &[&str])> = vec![
            ("CODE_LITERALS", CODE_LITERALS),
            ("AXIL_COMMANDS", AXIL_COMMANDS),
            ("AXIL_COMMAND_GROUPS", AXIL_COMMAND_GROUPS),
            ("FILE_EXTENSIONS", FILE_EXTENSIONS),
            ("PROJECT_DIRS", PROJECT_DIRS),
            ("ENV_VARS", ENV_VARS),
        ];
        for p in PROGRAMS {
            lists.push((p.name, p.subcommands));
            lists.push((p.name, p.groups));
            for group in p.groups {
                assert!(p.subcommands.contains(group), "{}: {group}", p.name);
            }
        }
        for (name, list) in lists {
            assert!(list.windows(2).all(|w| w[0] < w[1]), "{name} is not sorted");
        }
    }

    #[test]
    fn backticked_command_lines_give_the_command_not_the_line() {
        let found = sources(
            "Build with `cargo build --release -p axildb`, then `axil since 1h` and \
             `rustfmt --edition 2021 --emit stdout`; `git worktree add ../wt dev` too.",
        );
        for name in ["cargo build", "axil since", "rustfmt", "git worktree add"] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        assert!(
            !found.iter().any(|f| f.contains(" -")),
            "a whole command line: {found:?}"
        );
    }

    #[test]
    fn prose_commands_need_a_known_subcommand() {
        let found = sources(
            "Verified with cargo publish --dry-run, ran gh workflow run release.yml and \
             axil heal --orphans; cargo resolves the lockfile and the axil binary stays.",
        );
        for name in ["cargo publish", "gh workflow run", "axil heal"] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for prose in ["cargo resolves", "axil binary"] {
            assert!(!found.iter().any(|f| f == prose), "{prose}: {found:?}");
        }
    }

    #[test]
    fn backtick_names_drop_calls_line_suffixes_and_literals() {
        let found = sources(
            "`code_segments()` and `Axil::open(path)` at `db.rs:611`; `$GOMODCACHE`; \
             `true`, `0`, `with`, `mod` and `default` name nothing.",
        );
        for name in ["code_segments", "Axil::open", "db.rs", "GOMODCACHE"] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for literal in ["true", "0", "with", "mod", "default"] {
            assert!(!found.iter().any(|f| f == literal), "{literal}: {found:?}");
        }
    }

    #[test]
    fn long_flags_without_their_values() {
        let found = sources(
            "Run it with --top-k 5 --format=json; `--no-default-features` and --save/--compare \
             both work -- but --- and a-b--c are not flags.",
        );
        for flag in [
            "--top-k",
            "--format",
            "--no-default-features",
            "--save",
            "--compare",
        ] {
            assert!(found.iter().any(|f| f == flag), "{flag}: {found:?}");
        }
        assert!(
            !found.iter().any(|f| f == "--" || f == "---" || f == "--c"),
            "{found:?}"
        );
    }

    #[test]
    fn file_names_without_a_directory() {
        let found = sources(
            "See db.rs:611 and calibrate_u1.py, then the .vec store, memory.axil.ts, a .ts store, \
             .axil.vec, target/ and crates/adapters/axil-cli; lib.rs/main.rs too.",
        );
        for file in [
            "db.rs",
            "calibrate_u1.py",
            ".vec",
            "memory.axil.ts",
            ".ts",
            ".axil.vec",
            "target/",
            "crates/adapters/axil-cli",
            "lib.rs",
            "main.rs",
        ] {
            assert!(found.iter().any(|f| f == file), "{file}: {found:?}");
        }
        // The stem of a file name is not a second entity.
        assert!(!found.iter().any(|f| f == "calibrate_u1"), "{found:?}");
        for prose in [
            "crates.io stays, e.g. v1.2 and decisions/errors/context or Linux/Windows",
            "all .ts files",
            "plan items 28.25/28.26 over http/1.1, GET /recall?q=x&k=5, build/test and docs/CI",
        ] {
            let files: Vec<Entity> = extract_entities(prose)
                .into_iter()
                .filter(|e| e.entity_type == EntityType::File)
                .collect();
            assert!(files.is_empty(), "{prose:?} minted {files:?}");
        }
    }

    #[test]
    fn hyphenated_names_but_not_english_compounds() {
        let found = sources(
            "recall-for-file uses nli-deberta-v3-xsmall and glm-5.3-flash on \
             x86_64-apple-darwin; cargo-component builds it. A one-shot, free-form, \
             8192-token pass for Tier-1 (BUG-10, BUG-1..8) at top-5 over utf-8 text, post-1.2.0.",
        );
        for name in [
            "recall-for-file",
            "nli-deberta-v3-xsmall",
            "glm-5.3-flash",
            "x86_64-apple-darwin",
            "cargo-component",
        ] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for word in [
            "one-shot",
            "free-form",
            "8192-token",
            "Tier-1",
            "BUG-10",
            "BUG-1..8",
            "top-5",
            "utf-8",
            "post-1.2.0",
        ] {
            assert!(!found.iter().any(|f| f == word), "{word}: {found:?}");
        }
        assert!(!found.iter().any(|f| f == "x86_64"), "{found:?}");
    }

    #[test]
    fn constants_and_env_vars_but_not_emphasis() {
        let found = sources(
            "Rotated CARGO_REGISTRY_TOKEN; with AXIL_DB unset it reads $GOMODCACHE and HOME. \
             VERDICT: NOT FIXED, DONE.",
        );
        for name in ["CARGO_REGISTRY_TOKEN", "AXIL_DB", "GOMODCACHE", "HOME"] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for word in ["VERDICT", "NOT", "FIXED", "DONE", "$GOMODCACHE"] {
            assert!(!found.iter().any(|f| f == word), "{word}: {found:?}");
        }
    }

    #[test]
    fn branch_names_are_code_not_paths() {
        let entities = extract_entities(
            "Merged feat/phase-29-supersede-split and review/recall-core, then synced \
             origin/main. fix/notes.md is a file.",
        );
        for name in [
            "feat/phase-29-supersede-split",
            "review/recall-core",
            "origin/main",
        ] {
            assert!(
                entities
                    .iter()
                    .any(|e| e.source_text == name && e.entity_type == EntityType::Code),
                "{name}: {entities:?}"
            );
        }
        assert!(
            !entities
                .iter()
                .any(|e| e.source_text == "daemon-d2" || e.source_text == "main"),
            "{entities:?}"
        );
        assert!(entities
            .iter()
            .any(|e| e.source_text == "fix/notes.md" && e.entity_type == EntityType::File));
    }

    #[test]
    fn backticked_file_names_are_files() {
        let entities =
            extract_entities("Edit `db.rs`, `target/` and `.axil.vec`, not `AuthModule`.");
        for name in ["db.rs", "target/", ".axil.vec"] {
            assert!(
                entities
                    .iter()
                    .any(|e| e.source_text == name && e.entity_type == EntityType::File),
                "{name}: {entities:?}"
            );
        }
        assert!(entities
            .iter()
            .any(|e| e.source_text == "AuthModule" && e.entity_type == EntityType::Code));
    }

    #[test]
    fn underscore_prefixed_tables_are_snake_case() {
        let found = sources("auto_link writes _entities and _scip_aliases, not __init__ or _x_.");
        for name in ["auto_link", "_entities", "_scip_aliases"] {
            assert!(found.iter().any(|f| f == name), "{name}: {found:?}");
        }
        for word in ["__init__", "_x_"] {
            assert!(!found.iter().any(|f| f == word), "{word}: {found:?}");
        }
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
