//! Local doc extraction — read a dependency's documentation from the
//! copy already on disk after `cargo build` / `npm install`.
//!
//! This is the default, headline path: the dependency source is already
//! present (Rust in the shared registry cache, npm under the project's
//! `node_modules/`), it is an exact version match, and reading it costs
//! no network round-trip.

use std::path::{Path, PathBuf};

use crate::manifest::{Dependency, Ecosystem};

/// Minimum useful local-doc length in characters. A dependency whose
/// on-disk docs are shorter is flagged sparse — the caller should
/// consider a web fallback (Phase 16 P0.5) rather than ingest it
/// near-empty.
pub const MIN_LOCAL_DOC_CHARS: usize = 400;

/// Upper bound on changelog text read from disk. Changelogs list their
/// newest entries first, so a truncated tail still covers the recent
/// migrations that matter.
pub const MAX_CHANGELOG_CHARS: usize = 80_000;

/// Documentation extracted from a dependency's on-disk copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedDoc {
    /// The raw documentation text (markdown).
    pub text: String,
    /// Path the text was read from.
    pub source_ref: String,
    /// True when the text is below [`MIN_LOCAL_DOC_CHARS`].
    pub sparse: bool,
}

/// Why local extraction produced no documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotFound {
    /// The dependency is not installed on disk.
    NotInstalled,
    /// The dependency is unpinned — no resolved version to locate.
    Unpinned,
    /// Installed, but no documentation file could be found.
    NoDocs,
}

impl NotFound {
    /// Lowercase identifier for status output.
    pub fn as_str(&self) -> &'static str {
        match self {
            NotFound::NotInstalled => "not_installed",
            NotFound::Unpinned => "unpinned",
            NotFound::NoDocs => "no_docs",
        }
    }
}

/// Extract a dependency's docs from its on-disk copy.
///
/// `project_root` locates `node_modules/` for npm deps; Cargo deps come
/// from the shared registry cache and ignore it.
pub fn extract_local_doc(dep: &Dependency, project_root: &Path) -> Result<ExtractedDoc, NotFound> {
    let version = dep.version.as_deref().ok_or(NotFound::Unpinned)?;
    let (text, source_ref) = match dep.ecosystem {
        Ecosystem::Cargo => extract_cargo(&dep.name, version)?,
        Ecosystem::Npm => extract_npm(&dep.name, project_root)?,
        Ecosystem::Python => extract_python(&dep.name, version, project_root)?,
        Ecosystem::Go => extract_go(&dep.name, version)?,
        Ecosystem::Java => extract_java(&dep.name, version)?,
    };
    let sparse = text.chars().count() < MIN_LOCAL_DOC_CHARS;
    Ok(ExtractedDoc {
        text,
        source_ref,
        sparse,
    })
}

/// Read a dependency's changelog from its on-disk copy, if it ships one
/// — the raw material for Phase 16 P1.b migration notes.
///
/// Cargo, npm and Go dependencies vendor their source (and usually a
/// `CHANGELOG.md`) on disk. Python wheels and Maven artifacts do not
/// reliably bundle one, so they return `None`.
pub fn extract_changelog(dep: &Dependency, project_root: &Path) -> Option<String> {
    let dir = dep_dir(dep, project_root)?;
    read_changelog(&dir)
}

// ── Rust / Cargo ────────────────────────────────────────────────────────────

/// Extract docs for a Cargo dependency from the registry source cache.
fn extract_cargo(name: &str, version: &str) -> Result<(String, String), NotFound> {
    let src_root = cargo_registry_src().ok_or(NotFound::NotInstalled)?;
    let crate_dir = find_crate_dir_in(&src_root, name, version).ok_or(NotFound::NotInstalled)?;

    let mut text = String::new();
    let mut source_ref = crate_dir.display().to_string();

    if let Some((readme, path)) = read_readme(&crate_dir) {
        text.push_str(readme.trim());
        source_ref = path;
    }
    if let Some(doc) = read_crate_doc_comment(&crate_dir.join("src").join("lib.rs")) {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&doc);
    }
    if text.trim().is_empty() {
        return Err(NotFound::NoDocs);
    }
    Ok((text, source_ref))
}

/// `<CARGO_HOME>/registry/src` (or `~/.cargo/registry/src`).
fn cargo_registry_src() -> Option<PathBuf> {
    let cargo_home = match std::env::var_os("CARGO_HOME") {
        Some(h) => PathBuf::from(h),
        None => axil_core::home_dir()?.join(".cargo"),
    };
    Some(cargo_home.join("registry").join("src"))
}

/// Find `<name>-<version>/` under any registry subdirectory of `src_root`.
///
/// The registry subdir name (`index.crates.io-<hash>`, …) is not stable,
/// so every child is probed.
fn find_crate_dir_in(src_root: &Path, name: &str, version: &str) -> Option<PathBuf> {
    let target = format!("{name}-{version}");
    for entry in std::fs::read_dir(src_root).ok()?.flatten() {
        let candidate = entry.path().join(&target);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Extract the crate-level `//!` doc comment from a `lib.rs`.
fn read_crate_doc_comment(lib_rs: &Path) -> Option<String> {
    let content = std::fs::read_to_string(lib_rs).ok()?;
    let mut doc = String::new();
    for line in content.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("//!") {
            doc.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            doc.push('\n');
        } else if t.is_empty() || t.starts_with("#![") {
            // Blank lines and inner attributes may sit among crate docs.
            continue;
        } else if !doc.is_empty() {
            // First real item — the crate doc block has ended.
            break;
        }
    }
    let doc = doc.trim();
    if doc.is_empty() {
        None
    } else {
        Some(doc.to_string())
    }
}

// ── npm ─────────────────────────────────────────────────────────────────────

/// Extract docs for an npm dependency from `node_modules/`.
fn extract_npm(name: &str, project_root: &Path) -> Result<(String, String), NotFound> {
    // `node_modules/<name>` — `name` may be a `@scope/pkg`. Reads follow
    // symlinks, so pnpm's `.pnpm` content store resolves transparently.
    let pkg_dir = project_root.join("node_modules").join(name);
    if !pkg_dir.is_dir() {
        return Err(NotFound::NotInstalled);
    }

    let mut text = String::new();
    let mut source_ref = pkg_dir.display().to_string();

    if let Some((readme, path)) = read_readme(&pkg_dir) {
        text.push_str(readme.trim());
        source_ref = path;
    }
    if text.trim().is_empty() {
        if let Some(desc) = read_npm_description(&pkg_dir.join("package.json")) {
            text.push_str(&desc);
        }
    }
    if text.trim().is_empty() {
        return Err(NotFound::NoDocs);
    }
    Ok((text, source_ref))
}

/// Read the `description` field of a `package.json`.
fn read_npm_description(pkg_json: &Path) -> Option<String> {
    let content = std::fs::read_to_string(pkg_json).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("description")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
}

// ── Python ──────────────────────────────────────────────────────────────────

/// Extract docs for a Python dependency from a project virtualenv.
fn extract_python(
    name: &str,
    version: &str,
    project_root: &Path,
) -> Result<(String, String), NotFound> {
    let site_packages = find_site_packages(project_root).ok_or(NotFound::NotInstalled)?;
    let dist_info = find_dist_info(&site_packages, name, version).ok_or(NotFound::NotInstalled)?;
    let metadata = dist_info.join("METADATA");
    let raw = std::fs::read_to_string(&metadata).map_err(|_| NotFound::NoDocs)?;
    // METADATA is RFC-822: headers, a blank line, then the long
    // description (the published README).
    let body = raw
        .split_once("\n\n")
        .map(|(_, body)| body)
        .unwrap_or("")
        .trim();
    if body.is_empty() {
        return Err(NotFound::NoDocs);
    }
    Ok((body.to_string(), metadata.display().to_string()))
}

/// Locate a project's `site-packages` directory inside its virtualenv.
fn find_site_packages(project_root: &Path) -> Option<PathBuf> {
    for venv in [".venv", "venv", "env"] {
        let base = project_root.join(venv);
        if !base.is_dir() {
            continue;
        }
        // Windows layout.
        let win = base.join("Lib").join("site-packages");
        if win.is_dir() {
            return Some(win);
        }
        // Unix layout: lib/python3.X/site-packages.
        if let Ok(entries) = std::fs::read_dir(base.join("lib")) {
            for entry in entries.flatten() {
                let sp = entry.path().join("site-packages");
                if sp.is_dir() {
                    return Some(sp);
                }
            }
        }
    }
    None
}

/// Find the `<name>-<version>.dist-info` directory, matching the package
/// name under PEP 503 normalization (case- and separator-insensitive).
fn find_dist_info(site_packages: &Path, name: &str, version: &str) -> Option<PathBuf> {
    let norm = |s: &str| s.to_lowercase().replace(['_', '.'], "-");
    let want = norm(name);
    let suffix = format!("-{version}");
    for entry in std::fs::read_dir(site_packages).ok()?.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        let Some(stem) = file_name.strip_suffix(".dist-info") else {
            continue;
        };
        let Some(pkg) = stem.strip_suffix(&suffix) else {
            continue;
        };
        if norm(pkg) == want {
            return Some(entry.path());
        }
    }
    None
}

// ── Go ──────────────────────────────────────────────────────────────────────

/// Extract docs for a Go module from the module cache.
fn extract_go(module: &str, version: &str) -> Result<(String, String), NotFound> {
    let cache = go_mod_cache().ok_or(NotFound::NotInstalled)?;
    let dir = cache.join(format!("{}@{version}", escape_go_path(module)));
    if !dir.is_dir() {
        return Err(NotFound::NotInstalled);
    }
    match read_readme(&dir) {
        Some((readme, path)) => Ok((readme.trim().to_string(), path)),
        None => Err(NotFound::NoDocs),
    }
}

/// `$GOMODCACHE`, else `$GOPATH/pkg/mod`, else `~/go/pkg/mod`.
fn go_mod_cache() -> Option<PathBuf> {
    if let Some(cache) = std::env::var_os("GOMODCACHE") {
        return Some(PathBuf::from(cache));
    }
    if let Some(gopath) = std::env::var_os("GOPATH") {
        return Some(PathBuf::from(gopath).join("pkg").join("mod"));
    }
    axil_core::home_dir().map(|h| h.join("go").join("pkg").join("mod"))
}

/// Escape a module path for the Go module cache: an uppercase letter
/// `X` becomes `!x`, so mixed-case modules do not collide on a
/// case-insensitive filesystem.
fn escape_go_path(module: &str) -> String {
    let mut out = String::with_capacity(module.len());
    for c in module.chars() {
        if c.is_ascii_uppercase() {
            out.push('!');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

// ── Java ────────────────────────────────────────────────────────────────────

/// Extract docs for a Java dependency from the local Maven repository.
///
/// Reads the `<description>` from the artifact's `.pom`. The bundled
/// Javadoc JAR is not unpacked — many artifacts ship none, and those
/// that do are better served by the web fallback.
fn extract_java(name: &str, version: &str) -> Result<(String, String), NotFound> {
    let (group, artifact) = name.split_once(':').ok_or(NotFound::NotInstalled)?;
    let mut dir = maven_repo().ok_or(NotFound::NotInstalled)?;
    for segment in group.split('.') {
        dir = dir.join(segment);
    }
    dir = dir.join(artifact).join(version);
    if !dir.is_dir() {
        return Err(NotFound::NotInstalled);
    }
    let pom = dir.join(format!("{artifact}-{version}.pom"));
    let description = std::fs::read_to_string(&pom)
        .ok()
        .and_then(|xml| pom_description(&xml))
        .ok_or(NotFound::NoDocs)?;
    Ok((description, pom.display().to_string()))
}

/// `~/.m2/repository`.
fn maven_repo() -> Option<PathBuf> {
    axil_core::home_dir().map(|h| h.join(".m2").join("repository"))
}

/// Extract the `<description>` text from a `pom.xml`.
fn pom_description(xml: &str) -> Option<String> {
    let start = xml.find("<description>")? + "<description>".len();
    let end = xml[start..].find("</description>")? + start;
    let text = xml[start..end].trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

// ── shared ──────────────────────────────────────────────────────────────────

/// The on-disk directory for a dependency, when it exists as a plain
/// source tree — Cargo, npm and Go. Python (a `dist-info` of installed
/// metadata) and Java (a Maven `.m2` artifact) have no equivalent
/// source directory and return `None`.
fn dep_dir(dep: &Dependency, project_root: &Path) -> Option<PathBuf> {
    let version = dep.version.as_deref()?;
    let dir = match dep.ecosystem {
        Ecosystem::Cargo => find_crate_dir_in(&cargo_registry_src()?, &dep.name, version)?,
        Ecosystem::Npm => project_root.join("node_modules").join(&dep.name),
        Ecosystem::Go => go_mod_cache()?.join(format!("{}@{version}", escape_go_path(&dep.name))),
        Ecosystem::Python | Ecosystem::Java => return None,
    };
    dir.is_dir().then_some(dir)
}

/// Read the first changelog-like file in `dir`, capped at
/// [`MAX_CHANGELOG_CHARS`].
fn read_changelog(dir: &Path) -> Option<String> {
    for name in [
        "CHANGELOG.md",
        "Changelog.md",
        "changelog.md",
        "CHANGELOG",
        "CHANGELOG.markdown",
        "CHANGES.md",
        "CHANGES",
        "HISTORY.md",
        "NEWS.md",
        "RELEASES.md",
    ] {
        let Ok(content) = std::fs::read_to_string(dir.join(name)) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }
        return Some(content.chars().take(MAX_CHANGELOG_CHARS).collect());
    }
    None
}

/// Read the first README-like file in `dir`, returning `(text, path)`.
fn read_readme(dir: &Path) -> Option<(String, String)> {
    for name in [
        "README.md",
        "Readme.md",
        "readme.md",
        "README.markdown",
        "README.rst",
        "README.txt",
        "README",
    ] {
        let path = dir.join(name);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !content.trim().is_empty() {
                return Some((content, path.display().to_string()));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{DepKind, Dependency};
    use std::fs;

    fn npm_dep(name: &str, version: Option<&str>) -> Dependency {
        Dependency {
            name: name.to_string(),
            ecosystem: Ecosystem::Npm,
            kind: DepKind::Direct,
            declared_range: "*".to_string(),
            version: version.map(str::to_string),
        }
    }

    #[test]
    fn finds_a_crate_dir_under_a_registry_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("registry").join("src");
        let registry = src.join("index.crates.io-abc123");
        let crate_dir = registry.join("serde-1.0.210");
        fs::create_dir_all(&crate_dir).unwrap();

        let found = find_crate_dir_in(&src, "serde", "1.0.210");
        assert_eq!(found.as_deref(), Some(crate_dir.as_path()));
        assert!(find_crate_dir_in(&src, "serde", "9.9.9").is_none());
    }

    #[test]
    fn extracts_npm_readme() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("node_modules").join("react");
        fs::create_dir_all(&pkg).unwrap();
        let body = format!("# React\n\n{}", "A JavaScript library. ".repeat(40));
        fs::write(pkg.join("README.md"), &body).unwrap();

        let doc = extract_local_doc(&npm_dep("react", Some("19.2.4")), dir.path()).unwrap();
        assert!(doc.text.contains("JavaScript library"));
        assert!(!doc.sparse, "a long README is not sparse");
        assert!(doc.source_ref.contains("README.md"));
    }

    #[test]
    fn npm_description_fallback_flags_sparse() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("node_modules").join("tiny");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(
            pkg.join("package.json"),
            r#"{"name":"tiny","description":"A tiny package."}"#,
        )
        .unwrap();

        let doc = extract_local_doc(&npm_dep("tiny", Some("1.0.0")), dir.path()).unwrap();
        assert_eq!(doc.text, "A tiny package.");
        assert!(doc.sparse, "a one-line description is below the threshold");
    }

    #[test]
    fn missing_and_unpinned_deps_report_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            extract_local_doc(&npm_dep("absent", Some("1.0.0")), dir.path()),
            Err(NotFound::NotInstalled)
        );
        assert_eq!(
            extract_local_doc(&npm_dep("noversion", None), dir.path()),
            Err(NotFound::Unpinned)
        );
    }

    #[test]
    fn reads_a_crate_level_doc_comment() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("lib.rs");
        fs::write(
            &lib,
            "//! Crate summary line.\n//!\n//! More detail.\n#![allow(dead_code)]\npub fn x() {}\n",
        )
        .unwrap();
        let doc = read_crate_doc_comment(&lib).unwrap();
        assert!(doc.contains("Crate summary line."));
        assert!(doc.contains("More detail."));
        assert!(!doc.contains("pub fn"));
    }

    #[test]
    fn go_path_escaping() {
        assert_eq!(escape_go_path("github.com/foo/bar"), "github.com/foo/bar");
        assert_eq!(
            escape_go_path("github.com/BurntSushi/toml"),
            "github.com/!burnt!sushi/toml"
        );
    }

    #[test]
    fn pom_description_extraction() {
        let xml = "<project><description>A handy library.</description></project>";
        assert_eq!(pom_description(xml).as_deref(), Some("A handy library."));
        assert_eq!(pom_description("<project></project>"), None);
    }

    #[test]
    fn extracts_python_metadata_from_a_venv() {
        let dir = tempfile::tempdir().unwrap();
        let sp = dir.path().join(".venv").join("Lib").join("site-packages");
        // dist-info name casing differs from the requirement name.
        let dist = sp.join("Requests-2.31.0.dist-info");
        fs::create_dir_all(&dist).unwrap();
        let body = "Requests is an HTTP library. ".repeat(30);
        fs::write(
            dist.join("METADATA"),
            format!("Metadata-Version: 2.1\nName: requests\n\n{body}"),
        )
        .unwrap();

        let dep = Dependency {
            name: "requests".to_string(),
            ecosystem: Ecosystem::Python,
            kind: DepKind::Direct,
            declared_range: "*".to_string(),
            version: Some("2.31.0".to_string()),
        };
        let doc = extract_local_doc(&dep, dir.path()).unwrap();
        assert!(doc.text.contains("HTTP library"));
        assert!(doc.source_ref.ends_with("METADATA"));
    }

    #[test]
    fn extracts_npm_changelog() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("node_modules").join("react");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(
            pkg.join("CHANGELOG.md"),
            "# Changelog\n\n## 19.2.0\n\nBreaking: removed the legacy API.\n",
        )
        .unwrap();

        let changelog = extract_changelog(&npm_dep("react", Some("19.2.4")), dir.path()).unwrap();
        assert!(changelog.contains("removed the legacy API"));

        // A package with no changelog file → None.
        let bare = dir.path().join("node_modules").join("bare");
        fs::create_dir_all(&bare).unwrap();
        assert!(extract_changelog(&npm_dep("bare", Some("1.0.0")), dir.path()).is_none());
    }
}
