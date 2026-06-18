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

/// Scan a project directory for source files.
///
/// Respects `.gitignore` and `.axilignore`. Skips binary files,
/// generated files, and directories listed in [`SKIP_DIRS`].
pub fn scan_files(root: &Path, config: &IndexConfig) -> Vec<ScannedFile> {
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
        .git_exclude(true);

    // Respect .axilignore files (same syntax as .gitignore)
    walker.add_custom_ignore_filename(".axilignore");

    let mut files = Vec::new();

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

    // Sort by path for deterministic output
    files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    files
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
    fn scan_respects_config() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(src.join("big.rs"), "x".repeat(200_000)).unwrap(); // > 100KB

        let config = IndexConfig::default();
        let files = scan_files(dir.path(), &config);

        assert_eq!(files.len(), 1); // big.rs should be skipped
        assert_eq!(files[0].rel_path, "src/main.rs");
    }
}
