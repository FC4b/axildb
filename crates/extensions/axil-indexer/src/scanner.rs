//! Project scanner — walks a directory tree and detects project type.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};

use axil_core::IndexConfig;

/// Detected project type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectType {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    CSharp,
    Unknown,
}

impl ProjectType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Java => "java",
            Self::CSharp => "csharp",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ProjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Information about a scanned project.
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub root: PathBuf,
    pub project_type: ProjectType,
    pub name: String,
}

/// A source file discovered during scanning.
#[derive(Debug, Clone)]
pub struct ScannedFile {
    /// Absolute path.
    pub path: PathBuf,
    /// Path relative to project root.
    pub rel_path: String,
    /// Detected language for this file.
    pub language: Language,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Last modified time.
    pub modified: Option<SystemTime>,
}

/// Language of a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    CSharp,
    Markdown,
    Toml,
    Yaml,
    Json,
    Other,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Java => "java",
            Self::CSharp => "csharp",
            Self::Markdown => "markdown",
            Self::Toml => "toml",
            Self::Yaml => "yaml",
            Self::Json => "json",
            Self::Other => "other",
        }
    }

    /// Whether this language has a dedicated parser.
    pub fn has_parser(&self) -> bool {
        matches!(
            self,
            Self::Rust | Self::TypeScript | Self::JavaScript | Self::Python
        )
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "rs" => Self::Rust,
            "ts" | "tsx" => Self::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "py" | "pyi" => Self::Python,
            "go" => Self::Go,
            "java" => Self::Java,
            "cs" => Self::CSharp,
            "md" | "mdx" => Self::Markdown,
            "toml" => Self::Toml,
            "yaml" | "yml" => Self::Yaml,
            "json" => Self::Json,
            _ => Self::Other,
        }
    }
}

/// Directories that are always skipped.
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".eggs",
    "coverage",
    ".coverage",
    ".nyc_output",
];

/// File extensions for binary/generated files to skip.
const SKIP_EXTENSIONS: &[&str] = &[
    "exe", "dll", "so", "dylib", "o", "a", "lib", "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg",
    "webp", "mp3", "mp4", "wav", "avi", "mov", "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    "wasm", "pyc", "pyo", "class", "lock", // lockfiles are not useful for understanding
    "min.js", "min.css", "map", // source maps
];

/// Detect the primary project type from marker files.
pub fn detect_project_type(root: &Path) -> ProjectType {
    if root.join("Cargo.toml").exists() {
        ProjectType::Rust
    } else if root.join("tsconfig.json").exists() {
        ProjectType::TypeScript
    } else if root.join("package.json").exists() {
        ProjectType::JavaScript
    } else if root.join("pyproject.toml").exists()
        || root.join("setup.py").exists()
        || root.join("requirements.txt").exists()
    {
        ProjectType::Python
    } else if root.join("go.mod").exists() {
        ProjectType::Go
    } else if root.join("pom.xml").exists() || root.join("build.gradle").exists() {
        ProjectType::Java
    } else if has_extension_in_dir(root, "csproj") {
        ProjectType::CSharp
    } else {
        ProjectType::Unknown
    }
}

fn has_extension_in_dir(dir: &Path, ext: &str) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == ext) {
                return true;
            }
        }
    }
    false
}

/// Detect the project name from the root directory or manifest files.
pub fn detect_project_name(root: &Path, project_type: ProjectType) -> String {
    match project_type {
        ProjectType::Rust => {
            if let Ok(contents) = std::fs::read_to_string(root.join("Cargo.toml")) {
                if let Ok(parsed) = contents.parse::<toml::Table>() {
                    // Workspace name or package name
                    if let Some(pkg) = parsed.get("package") {
                        if let Some(name) = pkg.get("name").and_then(|v| v.as_str()) {
                            return name.to_string();
                        }
                    }
                    if let Some(ws) = parsed.get("workspace") {
                        if let Some(pkg) = ws.get("package") {
                            if let Some(name) = pkg.get("name").and_then(|v| v.as_str()) {
                                return name.to_string();
                            }
                        }
                    }
                }
            }
        }
        ProjectType::TypeScript | ProjectType::JavaScript => {
            if let Ok(contents) = std::fs::read_to_string(root.join("package.json")) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&contents) {
                    if let Some(name) = parsed.get("name").and_then(|v| v.as_str()) {
                        return name.to_string();
                    }
                }
            }
        }
        ProjectType::Python => {
            if let Ok(contents) = std::fs::read_to_string(root.join("pyproject.toml")) {
                if let Ok(parsed) = contents.parse::<toml::Table>() {
                    if let Some(proj) = parsed.get("project") {
                        if let Some(name) = proj.get("name").and_then(|v| v.as_str()) {
                            return name.to_string();
                        }
                    }
                }
            }
        }
        ProjectType::Go => {
            if let Ok(contents) = std::fs::read_to_string(root.join("go.mod")) {
                if let Some(line) = contents.lines().next() {
                    if let Some(module) = line.strip_prefix("module ") {
                        let parts: Vec<&str> = module.trim().rsplit('/').collect();
                        if let Some(name) = parts.first() {
                            return name.to_string();
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Fallback: directory name
    root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// A source file [`scan_files`] left out for exceeding `max_file_size_kb`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    /// Path relative to the scan root, with `/` separators.
    pub rel_path: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// What [`scan_files`] found under a project root.
#[derive(Debug, Clone, Default)]
pub struct Scan {
    /// Files to index, sorted by relative path.
    pub files: Vec<ScannedFile>,
    /// Source files left out for size, largest first. Their symbols are
    /// missing from code-search and fts and nothing else says why, so a
    /// caller that indexes should report them: see [`skipped_large_warning`].
    pub skipped_large: Vec<SkippedFile>,
}

/// The one-line warning for files a scan left out for size, or `None` when
/// it left out none. Names the three largest.
pub fn skipped_large_warning(skipped: &[SkippedFile], max_file_size_kb: u64) -> Option<String> {
    if skipped.is_empty() {
        return None;
    }
    let shown: Vec<String> = skipped
        .iter()
        .take(3)
        .map(|f| format!("{} ({} KB)", f.rel_path, f.size_bytes / 1024))
        .collect();
    let more = skipped.len().saturating_sub(3);
    let more_note = if more > 0 { format!(", +{more} more") } else { String::new() };
    Some(format!(
        "[axil-index] skipped {} file(s) over max_file_size_kb = {} KB: {}{} — their symbols will NOT be searchable; raise [index].max_file_size_kb in axil.toml to include them",
        skipped.len(),
        max_file_size_kb,
        shown.join(", "),
        more_note,
    ))
}

/// A database's own files: its core file and every companion named after it
/// (`memory.axil.vec`, `memory.axil.ts`, `memory.axil.fts/`, named vector
/// spaces). `memory.axil.ts` would otherwise read as TypeScript.
struct DatabaseFiles {
    /// The directory holding the database, canonicalized.
    dir: PathBuf,
    /// The core file's name.
    name: std::ffi::OsString,
}

impl DatabaseFiles {
    fn new(db_path: &Path) -> Option<Self> {
        let name = db_path.file_name()?.to_os_string();
        let parent = db_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let dir = std::fs::canonicalize(parent).ok()?;
        Some(Self { dir, name })
    }

    fn contains(&self, path: &Path) -> bool {
        let Some(name) = path.file_name() else {
            return false;
        };
        let named_after = name
            .as_encoded_bytes()
            .strip_prefix(self.name.as_encoded_bytes())
            .is_some_and(|rest| rest.is_empty() || rest[0] == b'.');
        // Only an entry named like the database pays for a canonicalize.
        named_after
            && path
                .parent()
                .and_then(|p| std::fs::canonicalize(p).ok())
                .is_some_and(|dir| dir == self.dir)
    }
}

/// Scan a project directory for source files.
///
/// Respects `.gitignore` and `.axilignore`. Skips binary files,
/// generated files, and directories listed in [`SKIP_DIRS`]. Given the
/// database the scan is for, also skips that database's own files, wherever
/// in the tree it lives.
///
/// Files over `max_file_size_kb` are not indexed and not printed about: they
/// come back in [`Scan::skipped_large`], so a read path that only checks
/// freshness stays quiet while an index run can warn.
pub fn scan_files(root: &Path, config: &IndexConfig, db_path: Option<&Path>) -> Scan {
    let skip_dirs: HashSet<&str> = SKIP_DIRS.iter().copied().collect();
    let skip_exts: HashSet<&str> = SKIP_EXTENSIONS.iter().copied().collect();
    let max_size = config.max_file_size_kb * 1024;

    // Build additional ignore patterns from config
    let extra_ignores: Vec<&str> = config.ignore.iter().map(|s| s.as_str()).collect();

    let mut walker = WalkBuilder::new(root);
    walker
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(false)
        .git_exclude(true)
        // The index root is the ignore boundary. Without this, a project
        // nested inside another repo inherits that repo's .gitignore — e.g. a
        // checkout under a workspace whose .gitignore has `models/` silently
        // loses every `**/models/` subtree (Django's db/models vanished this
        // way). The project's own .gitignore/.axilignore still apply.
        .parents(false);

    // Respect .axilignore files (same syntax as .gitignore)
    walker.add_custom_ignore_filename(".axilignore");

    if let Some(db_files) = db_path.and_then(DatabaseFiles::new) {
        walker.filter_entry(move |entry| !db_files.contains(entry.path()));
    }

    let mut files = Vec::new();
    let mut skipped_large: Vec<SkippedFile> = Vec::new();

    for entry in walker.build().flatten() {
        let path = entry.path();

        // Skip directories in skip list
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if skip_dirs.contains(name) {
                    continue;
                }
            }
            continue;
        }

        // Skip files matching extra ignore patterns.
        // Normalize to forward slashes: Windows `to_string_lossy` yields
        // `src\main.rs`, which would make the index non-portable across
        // OSes and break the `/tests/` skip patterns below.
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        if extra_ignores.iter().any(|pat| {
            // Simple glob: "*.pb.rs" or "generated/"
            if let Some(suffix) = pat.strip_prefix('*') {
                rel.ends_with(suffix)
            } else if pat.ends_with('/') {
                rel.starts_with(pat.trim_end_matches('/'))
            } else {
                rel.starts_with(pat) || rel.contains(pat)
            }
        }) {
            continue;
        }

        // Skip by extension
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if skip_exts.contains(ext) {
            continue;
        }

        // Detect language
        let language = Language::from_extension(ext);

        // Skip non-source files in "auto" mode
        if config.languages.contains(&"auto".to_string()) {
            if matches!(language, Language::Other) {
                continue;
            }
        } else {
            // Only index configured languages
            let lang_str = language.as_str();
            if !config.languages.iter().any(|l| l == lang_str) {
                continue;
            }
        }

        // Skip test files if configured
        if !config.index_tests {
            let lower = rel.to_lowercase();
            if lower.contains("/tests/")
                || lower.contains("/test/")
                || lower.contains("_test.")
                || lower.ends_with("_test.rs")
                || lower.ends_with("_test.py")
                || lower.ends_with(".test.ts")
                || lower.ends_with(".test.js")
                || lower.ends_with(".spec.ts")
                || lower.ends_with(".spec.js")
            {
                continue;
            }
        }

        // Check file size
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let size = metadata.len();
        if size > max_size {
            // A silently missing file poisons recall: queries for its symbols
            // return nothing and nothing says why. Record it for the caller.
            skipped_large.push(SkippedFile {
                rel_path: rel,
                size_bytes: size,
            });
            continue;
        }

        let modified = metadata.modified().ok();

        files.push(ScannedFile {
            path: path.to_path_buf(),
            rel_path: rel,
            language,
            size_bytes: size,
            modified,
        });
    }

    skipped_large.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    // Sort by path for deterministic output
    files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Scan {
        files,
        skipped_large,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_detection() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("xyz"), Language::Other);
    }

    #[test]
    fn project_type_display() {
        assert_eq!(ProjectType::Rust.as_str(), "rust");
        assert_eq!(ProjectType::TypeScript.as_str(), "typescript");
    }

    #[test]
    fn scan_ignores_parent_repo_gitignore() {
        // A project nested inside another git repo must not inherit that
        // repo's ignore rules: only ignore files at or below the index root
        // apply. The parent here ignores `models/`, which would otherwise
        // swallow the project's `models/` source directory.
        let parent = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(parent.path().join(".git")).unwrap();
        std::fs::write(parent.path().join(".gitignore"), "models/\n").unwrap();

        let proj = parent.path().join("proj");
        std::fs::create_dir_all(proj.join("models")).unwrap();
        std::fs::write(proj.join("models").join("a.py"), "def f():\n    pass\n").unwrap();
        std::fs::write(proj.join("b.py"), "def g():\n    pass\n").unwrap();

        let config = IndexConfig::default();
        let files = scan_files(&proj, &config, None).files;
        let rels: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(rels.contains(&"models/a.py"), "parent .gitignore leaked: {rels:?}");
        assert!(rels.contains(&"b.py"), "{rels:?}");

        // The project's OWN .gitignore still applies.
        std::fs::write(proj.join(".gitignore"), "models/\n").unwrap();
        std::fs::create_dir_all(proj.join(".git")).unwrap();
        let files = scan_files(&proj, &config, None).files;
        let rels: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(!rels.contains(&"models/a.py"), "own .gitignore ignored: {rels:?}");
        assert!(rels.contains(&"b.py"), "{rels:?}");
    }

    #[test]
    fn scan_respects_config() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(src.join("big.rs"), "x".repeat(200_000)).unwrap(); // > 100KB

        // Pin the cap explicitly: this tests that the config is respected,
        // not what the shipped default happens to be.
        let mut config = IndexConfig::default();
        config.max_file_size_kb = 100;
        let scan = scan_files(dir.path(), &config, None);

        assert_eq!(scan.files.len(), 1); // big.rs should be skipped
        assert_eq!(scan.files[0].rel_path, "src/main.rs");
        assert_eq!(
            scan.skipped_large,
            [SkippedFile {
                rel_path: "src/big.rs".into(),
                size_bytes: 200_000,
            }]
        );
        let warning = skipped_large_warning(&scan.skipped_large, 100).unwrap();
        assert!(
            warning.starts_with("[axil-index] skipped 1 file(s) over max_file_size_kb = 100 KB: src/big.rs (195 KB)"),
            "{warning}"
        );
        assert_eq!(skipped_large_warning(&[], 100), None);
    }

    #[test]
    fn scan_skips_the_database_and_its_companions() {
        // A database kept in the tree, not under a hidden `.axil/`. Its
        // timeseries companion ends in `.ts`, so by extension it is
        // TypeScript; small enough here that only the exclusion keeps it out.
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(data.join("memory.axil.fts")).unwrap();
        let db = data.join("memory.axil");
        std::fs::write(&db, "").unwrap();
        std::fs::write(data.join("memory.axil.ts"), "export const x = 1;\n").unwrap();
        std::fs::write(data.join("memory.axil.vec.code"), "").unwrap();
        std::fs::write(data.join("memory.axil.fts/notes.py"), "x = 1\n").unwrap();
        // Named like the database but not its file: still source.
        std::fs::write(data.join("memory.axilx.ts"), "export const y = 2;\n").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub fn f() {}\n").unwrap();
        let config = IndexConfig::default();

        let rels = |db_path: Option<&Path>| -> Vec<String> {
            scan_files(dir.path(), &config, db_path)
                .files
                .into_iter()
                .map(|f| f.rel_path)
                .collect()
        };
        assert_eq!(rels(Some(&db)), ["data/memory.axilx.ts", "lib.rs"]);
        // Found by its canonical location, however the path was spelled.
        let spelled = dir.path().join("data/../data/memory.axil");
        assert_eq!(rels(Some(&spelled)), ["data/memory.axilx.ts", "lib.rs"]);
        assert!(rels(None).contains(&"data/memory.axil.ts".to_string()));
    }
}
