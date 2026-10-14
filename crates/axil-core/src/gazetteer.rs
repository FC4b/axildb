//! Known names for entity extraction.
//!
//! Product, library, tool and language names (`redb`, `GitHub`, `Claude
//! Code`, `release-plz`) have no shape a pattern can see: they are plain
//! words. A [`Gazetteer`] lists them, so [`crate::entity::extract_entities_with`]
//! can find them. It holds three kinds of names:
//!
//! - a built-in list of well-known tools, languages, platforms, services,
//!   models and named techniques ([`Gazetteer::builtin`]);
//! - the project's own crates and packages, and the name they share
//!   (`axil` for `axil-core`, `axil-vector`, ...), read from its manifests
//!   ([`Gazetteer::load_project_dir`]);
//! - the project's direct dependencies, from the same manifests or from the
//!   database's `_deps` rows ([`Gazetteer::add_dependency`]).
//!
//! A project-derived name that is also an English word or a generic
//! technical noun (`log`, `time`, `regex`, `toml`) is left out, since every
//! note using the word would become a mention of the package.
//!
//! [`crate::Axil`] builds one per handle on first use (see
//! [`crate::Axil::entity_gazetteer`]); the plain
//! [`crate::entity::extract_entities`] uses none.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use crate::entity::EntityType;

/// One known name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KnownName {
    /// The spelling every mention is stored under, so `Github` and `GitHub`
    /// link to one entity.
    pub(crate) name: String,
    pub(crate) entity_type: EntityType,
    /// Match only this exact spelling. Set for names that are English words
    /// in lowercase (`Go`, `Rust`, `Windows`, `Cursor`).
    pub(crate) exact: bool,
}

/// A set of known names, matched case-insensitively unless marked exact.
#[derive(Debug, Clone, Default)]
pub struct Gazetteer {
    /// Lowercased name to its entry.
    names: HashMap<String, KnownName>,
    /// Lowercased `<project>-` prefixes: a hyphenated word under one
    /// (`axil-learn`) is one of the project's own components.
    prefixes: Vec<String>,
    /// The most words any name has (`Claude Code` has two).
    max_words: usize,
}

/// Tools, languages, platforms, services, file-format ecosystems and models,
/// matched in any case.
const BUILTIN_PROJECTS: &[&str] = &[
    // Languages and runtimes.
    "C++",
    "Deno",
    "Elixir",
    "Erlang",
    "Golang",
    "Haskell",
    "JavaScript",
    "Kotlin",
    "Lua",
    "Node.js",
    "PowerShell",
    "Python",
    "TypeScript",
    "WebAssembly",
    "Zig",
    // Platforms, hardware and file systems.
    "Android",
    "APFS",
    "CUDA",
    "Debian",
    "Docker",
    "FreeBSD",
    "Homebrew",
    "iOS",
    "Kubernetes",
    "Linux",
    "M1",
    "M1 Max",
    "M1 Pro",
    "M2",
    "M3",
    "M4",
    "macOS",
    "NTFS",
    "ROCm",
    "Ubuntu",
    "WSL",
    // Formats and protocol ecosystems that are products in their own right.
    "LLVM",
    "LSP",
    "MCP",
    "ONNX",
    "ONNX Runtime",
    "OpenTelemetry",
    "OTLP",
    "Protobuf",
    "SCIP",
    "WASI",
    "WASM",
    "gRPC",
    // Databases and search engines.
    "DuckDB",
    "Elasticsearch",
    "FAISS",
    "LanceDB",
    "LMDB",
    "Lucene",
    "Milvus",
    "MySQL",
    "Neo4j",
    "pgvector",
    "Pinecone",
    "Postgres",
    "PostgreSQL",
    "Qdrant",
    "Redis",
    "RocksDB",
    "SQLite",
    "SurrealDB",
    "SurrealKV",
    "Weaviate",
    // Services and registries.
    "Anthropic",
    "AWS",
    "Azure",
    "Bitbucket",
    "Cloudflare",
    "crates.io",
    "docs.rs",
    "GCP",
    "GitHub",
    "GitHub Actions",
    "GitLab",
    "Hugging Face",
    "HuggingFace",
    "LiteLLM",
    "npm",
    "Ollama",
    "OpenAI",
    "OpenRouter",
    "PyPI",
    "Vercel",
    // Developer tools.
    "Bazel",
    "cargo",
    "cargo-component",
    "cargo-nextest",
    "clippy",
    "CMake",
    "conda",
    "esbuild",
    "git",
    "jq",
    "Jest",
    "Neovim",
    "pip",
    "Playwright",
    "pytest",
    "React",
    "release-plz",
    "rg",
    "ripgrep",
    "rust-analyzer",
    "rustc",
    "rustfmt",
    "rustup",
    "Svelte",
    "Tailwind",
    "tmux",
    "Valgrind",
    "Vite",
    "VS Code",
    "wasm-tools",
    "wasmtime",
    "Webpack",
    "Xcode",
    // Agents, assistants and memory systems.
    "ChatGPT",
    "Claude Code",
    "Codex",
    "Cognee",
    "DeepSeek",
    "Graphiti",
    "HelixDB",
    "LangChain",
    "Letta",
    "LlamaIndex",
    "Mem0",
    "MemGPT",
    "Memvid",
    "OpenCode",
    "Qwen",
    // Models.
    "all-MiniLM-L6-v2",
    "bge-base",
    "bge-base-en-v1.5",
    "bge-large",
    "bge-m3",
    "bge-small",
    "bge-small-en-v1.5",
    "ColBERT",
    "DeBERTa",
    "GLiNER",
    "GLiNER2",
    "GPT-4",
    "GPT-4o",
    "GPT-5",
    "MiniLM",
    "ModernBERT",
    "nomic",
    "nomic-embed-text",
    "nomic-embed-text-v1.5",
    "RoBERTa",
    "SPLADE",
    // Benchmarks and datasets.
    "BEIR",
    "HotpotQA",
    "LoCoMo",
    "LongMemEval",
    "LongMemEval-S",
    "MS MARCO",
    "MTEB",
    "SWE-bench",
];

/// Names that are English words in another case (`go`, `rust`, `windows`,
/// `cursor`), matched only as spelled here.
const BUILTIN_EXACT_PROJECTS: &[&str] = &[
    "Aider",
    "Alpine",
    "Apple Silicon",
    "BERT",
    "BGE",
    "Bash",
    "Bun",
    "Chroma",
    "Claude",
    "Cline",
    "Component Model",
    "Copilot",
    "Criterion",
    "Cursor",
    "E5",
    "GLM",
    "GTE",
    "Gemini",
    "Gemma",
    "Go",
    "HF",
    "Haiku",
    "Hindsight",
    "Intel Mac",
    "Java",
    "Llama",
    "Mistral",
    "Opus",
    "Poetry",
    "Rosetta",
    "Ruby",
    "Rust",
    "Sonnet",
    "Swift",
    "Vue",
    "WIT",
    "Whisper",
    "Windows",
    "Windsurf",
    "Zep",
];

/// Named algorithms and techniques, matched only as spelled here.
const BUILTIN_CONCEPTS: &[&str] = &[
    "BM25",
    "HNSW",
    "MinHash",
    "PageRank",
    "Platt scaling",
    "RRF",
    "SimHash",
    "TF-IDF",
];

/// Words a project-derived name may not be on its own; see the file.
static COMMON_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    include_str!("gazetteer_common_words.txt")
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(str::split_whitespace)
        .collect()
});

/// Whether a project-derived name would also match ordinary prose: an English
/// word, a generic technical noun, or too short to be distinctive.
fn is_common_word(name: &str) -> bool {
    name.len() < 3 || COMMON_WORDS.contains(name.to_ascii_lowercase().as_str())
}

impl Gazetteer {
    /// An empty gazetteer.
    pub fn new() -> Self {
        Self::default()
    }

    /// The built-in names: well-known tools, languages, platforms, services,
    /// models, benchmarks and named techniques.
    pub fn builtin() -> Self {
        let mut g = Self::new();
        for name in BUILTIN_PROJECTS {
            g.insert(name, EntityType::Project);
        }
        for name in BUILTIN_EXACT_PROJECTS {
            g.insert_exact(name, EntityType::Project);
        }
        for name in BUILTIN_CONCEPTS {
            g.insert_exact(name, EntityType::Reference);
        }
        g
    }

    /// The built-in names plus what the project at `dir` declares (see
    /// [`Gazetteer::load_project_dir`]).
    pub fn for_project_dir(dir: &Path) -> Self {
        let mut g = Self::builtin();
        g.load_project_dir(dir);
        g
    }

    /// Add a name matched in any case.
    pub fn insert(&mut self, name: &str, entity_type: EntityType) {
        self.put(name, entity_type, false);
    }

    /// Add a name matched only as spelled.
    pub fn insert_exact(&mut self, name: &str, entity_type: EntityType) {
        self.put(name, entity_type, true);
    }

    fn put(&mut self, name: &str, entity_type: EntityType, exact: bool) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.max_words = self.max_words.max(name.split_whitespace().count());
        self.names.insert(
            name.to_lowercase(),
            KnownName {
                name: name.to_string(),
                entity_type,
                exact,
            },
        );
    }

    /// Add a dependency of the project, as a `project` entity. Returns
    /// whether it was added: a name that is an English word or a generic
    /// noun (`log`, `regex`), or already known, is skipped.
    pub fn add_dependency(&mut self, name: &str) -> bool {
        if is_common_word(name) || self.names.contains_key(&name.to_lowercase()) {
            return false;
        }
        self.insert(name, EntityType::Project);
        true
    }

    /// Add one of the project's own crates or packages, as a `code` entity.
    /// Returns whether it was added (see [`Gazetteer::add_dependency`]).
    pub fn add_project_crate(&mut self, name: &str) -> bool {
        if is_common_word(name) {
            return false;
        }
        self.insert(name, EntityType::Code);
        true
    }

    /// Add the project's own name (`axil`): a `project` entity, and the
    /// prefix of its components (`axil-learn` is code).
    pub fn add_project_name(&mut self, name: &str) {
        if is_common_word(name) {
            return;
        }
        self.insert(name, EntityType::Project);
        let prefix = format!("{}-", name.to_lowercase());
        if !self.prefixes.contains(&prefix) {
            self.prefixes.push(prefix);
        }
    }

    /// Read the project at `dir`: the crates, packages and direct
    /// dependencies its `Cargo.toml` (with workspace members),
    /// `package.json` and `pyproject.toml` declare, and the name its crates
    /// share. Missing or unreadable manifests are skipped. Returns how many
    /// names were added.
    pub fn load_project_dir(&mut self, dir: &Path) -> usize {
        let before = self.names.len();
        let mut manifest = ProjectManifest::default();
        read_cargo_workspace(dir, &mut manifest);
        read_package_json(dir, &mut manifest);
        read_pyproject(dir, &mut manifest);
        let own: HashSet<String> = manifest.crates.iter().map(|c| c.to_lowercase()).collect();
        for name in &manifest.crates {
            self.add_project_crate(name);
        }
        for name in &manifest.dependencies {
            if !own.contains(&name.to_lowercase()) {
                self.add_dependency(name);
            }
        }
        if let Some(name) = shared_prefix(&manifest.crates) {
            self.add_project_name(&name);
        }
        self.names.len() - before
    }

    /// How many names it holds.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether it holds no names.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && self.prefixes.is_empty()
    }

    /// The most words a name has.
    pub(crate) fn max_words(&self) -> usize {
        self.max_words
    }

    /// The entry `candidate` is a mention of, if any.
    pub(crate) fn lookup(&self, candidate: &str) -> Option<&KnownName> {
        let known = self.names.get(&candidate.to_lowercase())?;
        (!known.exact || known.name == candidate).then_some(known)
    }

    /// Whether a hyphenated word is one of the project's own components
    /// (`axil-learn` under `axil`).
    pub(crate) fn is_project_component(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.prefixes.iter().any(|p| {
            lower.len() > p.len()
                && lower.starts_with(p.as_str())
                && lower[p.len()..].starts_with(|c: char| c.is_ascii_alphabetic())
        })
    }
}

/// What a project's manifests declare.
#[derive(Default)]
struct ProjectManifest {
    /// The project's own crates and packages.
    crates: Vec<String>,
    /// Their direct dependencies.
    dependencies: Vec<String>,
}

fn read_toml(path: &Path) -> Option<toml::Value> {
    toml::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// The dependency names of one Cargo manifest: `[dependencies]`,
/// `[dev-dependencies]`, `[build-dependencies]`, the same under
/// `[target.*]`, and `[workspace.dependencies]`.
fn cargo_dependencies(manifest: &toml::Value, out: &mut Vec<String>) {
    let tables = ["dependencies", "dev-dependencies", "build-dependencies"];
    let mut push = |table: Option<&toml::Value>| {
        if let Some(deps) = table.and_then(toml::Value::as_table) {
            out.extend(deps.keys().cloned());
        }
    };
    for t in tables {
        push(manifest.get(t));
    }
    push(
        manifest
            .get("workspace")
            .and_then(|w| w.get("dependencies")),
    );
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            for t in tables {
                push(target.get(t));
            }
        }
    }
}

/// The root `Cargo.toml` at `dir` and, for a workspace, every member's
/// (a trailing `/*` member pattern is expanded one level).
fn read_cargo_workspace(dir: &Path, out: &mut ProjectManifest) {
    let Some(root) = read_toml(&dir.join("Cargo.toml")) else {
        return;
    };
    let mut manifests = vec![root.clone()];
    let members = root
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for member in members.iter().filter_map(toml::Value::as_str) {
        let dirs: Vec<std::path::PathBuf> = match member.strip_suffix("/*") {
            Some(parent) => std::fs::read_dir(dir.join(parent))
                .map(|entries| entries.flatten().map(|e| e.path()).collect())
                .unwrap_or_default(),
            None => vec![dir.join(member)],
        };
        for member_dir in dirs {
            if let Some(manifest) = read_toml(&member_dir.join("Cargo.toml")) {
                manifests.push(manifest);
            }
        }
    }
    for manifest in &manifests {
        if let Some(name) = manifest
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(toml::Value::as_str)
        {
            out.crates.push(name.to_string());
        }
        cargo_dependencies(manifest, &mut out.dependencies);
    }
}

/// `package.json` at `dir`: its name and the keys of its dependency maps.
fn read_package_json(dir: &Path, out: &mut ProjectManifest) {
    let Some(json) = std::fs::read_to_string(dir.join("package.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    else {
        return;
    };
    if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
        out.crates.push(name.to_string());
    }
    for key in ["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(deps) = json.get(key).and_then(|v| v.as_object()) {
            out.dependencies.extend(deps.keys().cloned());
        }
    }
}

/// `pyproject.toml` at `dir`: `[project] name` and the package names of
/// `[project] dependencies` (a PEP 508 string up to its first version,
/// extra or marker).
fn read_pyproject(dir: &Path, out: &mut ProjectManifest) {
    let Some(project) =
        read_toml(&dir.join("pyproject.toml")).and_then(|t| t.get("project").cloned())
    else {
        return;
    };
    if let Some(name) = project.get("name").and_then(toml::Value::as_str) {
        out.crates.push(name.to_string());
    }
    let deps = project.get("dependencies").and_then(toml::Value::as_array);
    for spec in deps.into_iter().flatten().filter_map(toml::Value::as_str) {
        let name = spec
            .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
            .next()
            .unwrap_or("");
        if !name.is_empty() {
            out.dependencies.push(name.to_string());
        }
    }
}

/// The name a project's crates share as their first hyphen- or
/// underscore-separated word (`axil` for `axil-core`, `axil-vector`, ...):
/// the most common such word, when at least two crates and at least half of
/// them carry it.
fn shared_prefix(crates: &[String]) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in crates {
        if let Some((first, _)) = name.split_once(['-', '_']) {
            *counts.entry(first.to_lowercase()).or_default() += 1;
        }
    }
    let (word, count) = counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))?;
    (count >= 2 && count * 2 >= crates.len()).then_some(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_names_match_in_any_case_unless_exact() {
        let g = Gazetteer::builtin();
        assert_eq!(g.lookup("github").map(|k| k.name.as_str()), Some("GitHub"));
        assert_eq!(
            g.lookup("Claude Code").map(|k| k.name.as_str()),
            Some("Claude Code")
        );
        assert!(g.lookup("Go").is_some());
        for english in ["go", "rust", "windows", "cursor"] {
            assert!(g.lookup(english).is_none(), "{english}");
        }
        assert_eq!(g.max_words(), 2);
    }

    #[test]
    fn common_words_are_not_project_names() {
        let mut g = Gazetteer::new();
        for name in ["log", "regex", "toml", "time", "criterion", "cc"] {
            assert!(!g.add_dependency(name), "{name}");
        }
        for name in ["redb", "tantivy", "anyhow", "serde_json"] {
            assert!(g.add_dependency(name), "{name}");
        }
    }

    #[test]
    fn shared_prefix_is_the_projects_name() {
        let crates: Vec<String> = ["axil-core", "axil-vector", "axildb", "axil-tests"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(shared_prefix(&crates).as_deref(), Some("axil"));
        let lone: Vec<String> = vec!["tool-core".into(), "other".into(), "misc".into()];
        assert_eq!(shared_prefix(&lone), None);
    }

    #[test]
    fn a_cargo_workspace_gives_its_crates_dependencies_and_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\", \"tool\"]\n\
             [workspace.dependencies]\nredb = \"3\"\nregex = \"1\"\n",
        )
        .unwrap();
        for (path, name, deps) in [
            ("crates/acme-core", "acme-core", "tantivy = \"0.22\"\n"),
            (
                "crates/acme-cli",
                "acme-cli",
                "clap = \"4\"\nacme-core = { path = \"../acme-core\" }\n",
            ),
            (
                "tool",
                "acme-tool",
                "[target.'cfg(unix)'.dependencies]\nnix = \"0.29\"\nrustix = \"1\"\n",
            ),
        ] {
            std::fs::create_dir_all(root.join(path)).unwrap();
            let deps = if deps.starts_with('[') {
                deps.to_string()
            } else {
                format!("[dependencies]\n{deps}")
            };
            std::fs::write(
                root.join(path).join("Cargo.toml"),
                format!("[package]\nname = \"{name}\"\n{deps}"),
            )
            .unwrap();
        }
        let g = Gazetteer::for_project_dir(root);
        for (name, ty) in [
            ("acme-core", EntityType::Code),
            ("acme-cli", EntityType::Code),
            ("acme", EntityType::Project),
            ("redb", EntityType::Project),
            ("tantivy", EntityType::Project),
            ("clap", EntityType::Project),
            ("rustix", EntityType::Project),
        ] {
            assert_eq!(
                g.lookup(name).map(|k| k.entity_type.clone()),
                Some(ty),
                "{name}"
            );
        }
        for word in ["regex", "nix"] {
            assert!(g.lookup(word).is_none(), "{word}");
        }
        assert!(g.is_project_component("acme-docs"));
        assert!(!g.is_project_component("acme-"));
    }
}
