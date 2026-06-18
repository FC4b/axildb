//! Manifest detection + dependency parsing.
//!
//! Walks a repository, finds every Cargo / npm manifest, and parses the
//! dependencies it declares. Version *resolution* (pinning each dep to a
//! lockfile version) is [`crate::resolve`]'s job — the [`Dependency`]
//! values produced here carry `version: None`.

use std::path::{Path, PathBuf};

use crate::DocsError;

/// A dependency ecosystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Ecosystem {
    /// Rust — `Cargo.toml` + `Cargo.lock`.
    Cargo,
    /// Node — `package.json` + `package-lock.json` / `yarn.lock` / `pnpm-lock.yaml`.
    Npm,
    /// Python — `requirements.txt` / `pyproject.toml` + `uv.lock` / `poetry.lock`.
    Python,
    /// Go — `go.mod` (self-pinning) + `go.sum`.
    Go,
    /// Java — `pom.xml` (self-pinning).
    Java,
}

impl Ecosystem {
    /// Lowercase identifier used in stored records and CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Ecosystem::Cargo => "cargo",
            Ecosystem::Npm => "npm",
            Ecosystem::Python => "python",
            Ecosystem::Go => "go",
            Ecosystem::Java => "java",
        }
    }

    /// Parse a lowercase ecosystem identifier (the inverse of [`as_str`]).
    ///
    /// [`as_str`]: Ecosystem::as_str
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cargo" => Some(Ecosystem::Cargo),
            "npm" => Some(Ecosystem::Npm),
            "python" => Some(Ecosystem::Python),
            "go" => Some(Ecosystem::Go),
            "java" => Some(Ecosystem::Java),
            _ => None,
        }
    }
}

/// How a dependency is declared in its manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DepKind {
    /// A normal runtime dependency.
    Direct,
    /// A dev/test-only dependency (`[dev-dependencies]`, `devDependencies`).
    Dev,
    /// A build-script dependency (`[build-dependencies]`).
    Build,
    /// A transitive dependency — present in the lockfile but not
    /// declared in any manifest (Phase 16 P1.a).
    Transitive,
}

impl DepKind {
    /// Lowercase identifier used in stored records and CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            DepKind::Direct => "direct",
            DepKind::Dev => "dev",
            DepKind::Build => "build",
            DepKind::Transitive => "transitive",
        }
    }
}

/// A single dependency declared in a manifest.
///
/// `version` is `None` until [`crate::resolve_dependencies`] pins it from
/// the lockfile; a dependency that stays `None` after resolution is
/// *unpinned* (no lockfile exists, or it is absent from the lockfile).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Dependency {
    /// Package name as it appears in the registry.
    pub name: String,
    /// Ecosystem this dependency belongs to.
    pub ecosystem: Ecosystem,
    /// Whether it is a runtime, dev or build dependency.
    pub kind: DepKind,
    /// The version range as written in the manifest (e.g. `^19.2.4`,
    /// `1`), or a marker (`workspace` / `path` / `git`) when the version
    /// is pinned somewhere other than an inline range.
    pub declared_range: String,
    /// Exact resolved version from the lockfile, or `None` if unpinned.
    pub version: Option<String>,
}

/// A manifest file discovered in a repository, with its companion lockfile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedManifest {
    /// Path to the manifest file.
    pub path: PathBuf,
    /// Ecosystem the manifest belongs to.
    pub ecosystem: Ecosystem,
    /// The lockfile that pins this manifest's dependencies, if one exists.
    pub lockfile: Option<PathBuf>,
}

/// Directory names never descended into during detection.
const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git"];

/// Walk `root` and return every Cargo / npm manifest found.
///
/// `target/`, `node_modules/`, `.git/` and dotfile directories are
/// skipped. Results are sorted by path for deterministic output.
pub fn detect_manifests(root: &Path) -> Vec<DetectedManifest> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if file_type.is_dir() {
                if SKIP_DIRS.contains(&name.as_ref()) || name.starts_with('.') {
                    continue;
                }
                stack.push(entry.path());
            } else if file_type.is_file() {
                let ecosystem = match name.as_ref() {
                    "Cargo.toml" => Ecosystem::Cargo,
                    "package.json" => Ecosystem::Npm,
                    "requirements.txt" | "pyproject.toml" => Ecosystem::Python,
                    "go.mod" => Ecosystem::Go,
                    "pom.xml" => Ecosystem::Java,
                    _ => continue,
                };
                let lockfile = match ecosystem {
                    // A Cargo workspace keeps one `Cargo.lock` at the
                    // root, so member manifests walk up to find it.
                    Ecosystem::Cargo => find_lockfile(&dir, &["Cargo.lock"], true),
                    Ecosystem::Npm => find_lockfile(
                        &dir,
                        &["package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
                        false,
                    ),
                    Ecosystem::Python => {
                        find_lockfile(&dir, &["uv.lock", "poetry.lock", "Pipfile.lock"], false)
                    }
                    Ecosystem::Go => find_lockfile(&dir, &["go.sum"], false),
                    // pom.xml carries exact <version>s itself.
                    Ecosystem::Java => None,
                };
                out.push(DetectedManifest {
                    path: entry.path(),
                    ecosystem,
                    lockfile,
                });
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Find the first of `names` in `dir` — and, when `walk_up`, its ancestors.
fn find_lockfile(dir: &Path, names: &[&str], walk_up: bool) -> Option<PathBuf> {
    let mut cur = Some(dir.to_path_buf());
    while let Some(d) = cur {
        for n in names {
            let candidate = d.join(n);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        if !walk_up {
            return None;
        }
        cur = d.parent().map(Path::to_path_buf);
    }
    None
}

/// Parse a detected manifest into its declared dependencies.
///
/// The returned dependencies are unresolved (`version: None`); call
/// [`crate::resolve_dependencies`] to pin them.
pub fn parse_manifest(manifest: &DetectedManifest) -> Result<Vec<Dependency>, DocsError> {
    match manifest.ecosystem {
        Ecosystem::Cargo => parse_cargo_manifest(&manifest.path),
        Ecosystem::Npm => parse_npm_manifest(&manifest.path),
        Ecosystem::Python => parse_python_manifest(&manifest.path),
        Ecosystem::Go => parse_go_manifest(&manifest.path),
        Ecosystem::Java => parse_java_manifest(&manifest.path),
    }
}

/// Parse `[dependencies]`, `[dev-dependencies]` and `[build-dependencies]`
/// from a `Cargo.toml`.
pub fn parse_cargo_manifest(path: &Path) -> Result<Vec<Dependency>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let value: toml::Value = toml::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut deps = Vec::new();
    for (section, kind) in [
        ("dependencies", DepKind::Direct),
        ("dev-dependencies", DepKind::Dev),
        ("build-dependencies", DepKind::Build),
    ] {
        let Some(table) = value.get(section).and_then(toml::Value::as_table) else {
            continue;
        };
        for (name, spec) in table {
            deps.push(Dependency {
                name: name.clone(),
                ecosystem: Ecosystem::Cargo,
                kind,
                declared_range: cargo_dep_range(spec),
                version: None,
            });
        }
    }
    Ok(deps)
}

/// Extract a Cargo dependency's declared range from its TOML spec.
///
/// A spec is either a bare version string or a table; tables that pin
/// the version elsewhere (workspace inheritance, path, git) yield a
/// marker so the caller can tell why no inline range is available.
fn cargo_dep_range(spec: &toml::Value) -> String {
    match spec {
        toml::Value::String(s) => s.clone(),
        toml::Value::Table(t) => {
            if t.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
                "workspace".to_string()
            } else if let Some(v) = t.get("version").and_then(toml::Value::as_str) {
                v.to_string()
            } else if t.contains_key("path") {
                "path".to_string()
            } else if t.contains_key("git") {
                "git".to_string()
            } else {
                "*".to_string()
            }
        }
        _ => "*".to_string(),
    }
}

/// Parse `dependencies` and `devDependencies` from a `package.json`.
pub fn parse_npm_manifest(path: &Path) -> Result<Vec<Dependency>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut deps = Vec::new();
    for (section, kind) in [
        ("dependencies", DepKind::Direct),
        ("devDependencies", DepKind::Dev),
    ] {
        let Some(obj) = json.get(section).and_then(serde_json::Value::as_object) else {
            continue;
        };
        for (name, range) in obj {
            deps.push(Dependency {
                name: name.clone(),
                ecosystem: Ecosystem::Npm,
                kind,
                declared_range: range.as_str().unwrap_or("*").to_string(),
                version: None,
            });
        }
    }
    Ok(deps)
}

/// Parse a Python `requirements.txt` or `pyproject.toml`.
///
/// requirements.txt: one PEP 508 requirement per line. pyproject.toml:
/// the PEP 621 `[project].dependencies` array. An `==` pin is captured
/// as the resolved version; any other operator leaves the dependency
/// unpinned for the lockfile to resolve.
pub fn parse_python_manifest(path: &Path) -> Result<Vec<Dependency>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let is_pyproject = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n == "pyproject.toml")
        .unwrap_or(false);

    let requirements: Vec<String> = if is_pyproject {
        let value: toml::Value = toml::from_str(&text).map_err(|e| DocsError::at(path, e))?;
        value
            .get("project")
            .and_then(|p| p.get("dependencies"))
            .and_then(toml::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        text.lines().map(str::to_string).collect()
    };

    Ok(requirements
        .iter()
        .filter_map(|line| parse_python_requirement(line))
        .collect())
}

/// Parse one PEP 508 requirement line into a [`Dependency`].
fn parse_python_requirement(line: &str) -> Option<Dependency> {
    let line = line.split('#').next().unwrap_or("").trim();
    if line.is_empty() || line.starts_with('-') {
        return None;
    }
    // Drop any environment marker (`; python_version < "3.9"`).
    let line = line.split(';').next().unwrap_or("").trim();
    // The name ends at the first operator, extras bracket, or space.
    let end = line
        .find(|c: char| matches!(c, '=' | '<' | '>' | '~' | '!' | '[' | ' ' | '('))
        .unwrap_or(line.len());
    let name = line[..end].trim();
    if name.is_empty() {
        return None;
    }
    let spec = line[end..].trim();
    // An exact `==X` pin yields a resolved version.
    let version = spec
        .split("==")
        .nth(1)
        .map(|v| v.split(',').next().unwrap_or(v).trim().to_string())
        .filter(|v| !v.is_empty() && !v.contains('*'));
    Some(Dependency {
        name: name.to_string(),
        ecosystem: Ecosystem::Python,
        kind: DepKind::Direct,
        declared_range: if spec.is_empty() {
            "*".to_string()
        } else {
            spec.to_string()
        },
        version,
    })
}

/// Parse a Go `go.mod` file's `require` directives.
///
/// `go.mod` records exact versions, so every parsed dependency is
/// already pinned. `// indirect` (transitive) requires are skipped.
pub fn parse_go_manifest(path: &Path) -> Result<Vec<Dependency>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let mut deps = Vec::new();
    let mut in_block = false;
    for raw in text.lines() {
        let indirect = raw.contains("// indirect");
        let line = raw.split("//").next().unwrap_or("").trim();
        if line == "require (" {
            in_block = true;
            continue;
        }
        if in_block && line == ")" {
            in_block = false;
            continue;
        }
        let spec = if in_block {
            line
        } else if let Some(rest) = line.strip_prefix("require ") {
            rest.trim()
        } else {
            continue;
        };
        if spec.is_empty() || indirect {
            continue;
        }
        let mut parts = spec.split_whitespace();
        let (Some(module), Some(version)) = (parts.next(), parts.next()) else {
            continue;
        };
        deps.push(Dependency {
            name: module.to_string(),
            ecosystem: Ecosystem::Go,
            kind: DepKind::Direct,
            declared_range: version.to_string(),
            version: Some(version.to_string()),
        });
    }
    Ok(deps)
}

/// Parse a Maven `pom.xml`'s `<dependency>` blocks.
///
/// Each yields a `groupId:artifactId` name. A literal `<version>` is
/// captured as the resolved version; a `${property}` placeholder or an
/// absent version leaves the dependency unpinned. This is a deliberately
/// minimal scan — it reads conventional pom XML, not arbitrary XML.
pub fn parse_java_manifest(path: &Path) -> Result<Vec<Dependency>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let mut deps = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find("<dependency>") {
        let after = &rest[start + "<dependency>".len()..];
        let Some(end) = after.find("</dependency>") else {
            break;
        };
        let block = &after[..end];
        rest = &after[end + "</dependency>".len()..];

        let (Some(group), Some(artifact)) = (
            xml_tag_text(block, "groupId"),
            xml_tag_text(block, "artifactId"),
        ) else {
            continue;
        };
        let raw_version = xml_tag_text(block, "version");
        let version = raw_version
            .filter(|v| !v.is_empty() && !v.starts_with("${"))
            .map(str::to_string);
        let kind = if xml_tag_text(block, "scope") == Some("test") {
            DepKind::Dev
        } else {
            DepKind::Direct
        };
        deps.push(Dependency {
            name: format!("{group}:{artifact}"),
            ecosystem: Ecosystem::Java,
            kind,
            declared_range: raw_version.unwrap_or("*").to_string(),
            version,
        });
    }
    Ok(deps)
}

/// Extract the trimmed text of the first `<tag>…</tag>` in `xml`.
fn xml_tag_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_cargo_dependency_kinds_and_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("Cargo.toml");
        fs::write(
            &manifest,
            r#"
[package]
name = "demo"

[dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }
local = { path = "../local" }
shared = { workspace = true }

[dev-dependencies]
tempfile = "3"

[build-dependencies]
cc = "1"
"#,
        )
        .unwrap();

        let deps = parse_cargo_manifest(&manifest).unwrap();
        let find = |n: &str| deps.iter().find(|d| d.name == n).unwrap();

        assert_eq!(find("serde").declared_range, "1.0");
        assert_eq!(find("serde").kind, DepKind::Direct);
        assert_eq!(find("serde").ecosystem, Ecosystem::Cargo);
        assert_eq!(find("serde").version, None);
        assert_eq!(find("tokio").declared_range, "1");
        assert_eq!(find("local").declared_range, "path");
        assert_eq!(find("shared").declared_range, "workspace");
        assert_eq!(find("tempfile").kind, DepKind::Dev);
        assert_eq!(find("cc").kind, DepKind::Build);
    }

    #[test]
    fn parses_npm_dependency_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("package.json");
        fs::write(
            &manifest,
            r#"{
  "name": "demo",
  "dependencies": { "react": "^19.0.0", "@scope/pkg": "1.2.3" },
  "devDependencies": { "vite": "^8.0.0" }
}"#,
        )
        .unwrap();

        let deps = parse_npm_manifest(&manifest).unwrap();
        let find = |n: &str| deps.iter().find(|d| d.name == n).unwrap();

        assert_eq!(find("react").declared_range, "^19.0.0");
        assert_eq!(find("react").kind, DepKind::Direct);
        assert_eq!(find("react").ecosystem, Ecosystem::Npm);
        assert_eq!(find("@scope/pkg").declared_range, "1.2.3");
        assert_eq!(find("vite").kind, DepKind::Dev);
    }

    #[test]
    fn detects_manifests_and_skips_target() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        fs::write(dir.path().join("Cargo.lock"), "").unwrap();

        let ui = dir.path().join("ui");
        fs::create_dir(&ui).unwrap();
        fs::write(ui.join("package.json"), "{}").unwrap();

        // A Cargo.toml under target/ must be ignored.
        let target = dir.path().join("target");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("Cargo.toml"), "[package]\nname = \"junk\"\n").unwrap();

        let found = detect_manifests(dir.path());
        assert_eq!(found.len(), 2, "target/ Cargo.toml should be skipped");

        let cargo = found
            .iter()
            .find(|m| m.ecosystem == Ecosystem::Cargo)
            .unwrap();
        assert!(cargo.lockfile.is_some(), "Cargo.lock should be found");

        let npm = found
            .iter()
            .find(|m| m.ecosystem == Ecosystem::Npm)
            .unwrap();
        assert!(npm.lockfile.is_none(), "ui/ has no lockfile");
    }

    #[test]
    fn parses_python_requirements() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("requirements.txt");
        fs::write(
            &manifest,
            "# a comment\nrequests==2.31.0\nflask>=3.0\nblack[d]==24.1.0\n-e .\nrich\n",
        )
        .unwrap();
        let deps = parse_python_manifest(&manifest).unwrap();
        let find = |n: &str| deps.iter().find(|d| d.name == n).unwrap();
        assert_eq!(find("requests").version.as_deref(), Some("2.31.0"));
        assert_eq!(find("requests").ecosystem, Ecosystem::Python);
        assert_eq!(find("flask").version, None, ">= is not an exact pin");
        assert_eq!(find("black").version.as_deref(), Some("24.1.0"));
        assert_eq!(find("rich").declared_range, "*");
        assert!(deps.iter().all(|d| d.name != "-e"), "options are skipped");
    }

    #[test]
    fn parses_go_mod_skipping_indirect() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("go.mod");
        fs::write(
            &manifest,
            "module example.com/app\n\ngo 1.22\n\nrequire (\n\tgithub.com/foo/bar v1.4.0\n\tgithub.com/baz/qux v0.2.1 // indirect\n)\n\nrequire github.com/solo/dep v2.0.0\n",
        )
        .unwrap();
        let deps = parse_go_manifest(&manifest).unwrap();
        let find = |n: &str| deps.iter().find(|d| d.name == n);
        assert_eq!(
            find("github.com/foo/bar").unwrap().version.as_deref(),
            Some("v1.4.0")
        );
        assert!(find("github.com/baz/qux").is_none(), "indirect is skipped");
        assert_eq!(
            find("github.com/solo/dep").unwrap().version.as_deref(),
            Some("v2.0.0")
        );
    }

    #[test]
    fn parses_pom_xml_dependencies() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("pom.xml");
        fs::write(
            &manifest,
            r#"<project>
  <dependencies>
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
      <version>33.0.0-jre</version>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>${junit.version}</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>"#,
        )
        .unwrap();
        let deps = parse_java_manifest(&manifest).unwrap();
        let guava = deps
            .iter()
            .find(|d| d.name == "com.google.guava:guava")
            .unwrap();
        assert_eq!(guava.version.as_deref(), Some("33.0.0-jre"));
        assert_eq!(guava.kind, DepKind::Direct);
        let junit = deps
            .iter()
            .find(|d| d.name == "org.junit.jupiter:junit-jupiter")
            .unwrap();
        assert_eq!(junit.version, None, "${{...}} placeholder is unpinned");
        assert_eq!(junit.kind, DepKind::Dev);
    }

    #[test]
    fn cargo_member_walks_up_to_workspace_lock() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("Cargo.lock"), "").unwrap();
        let member = dir.path().join("crates").join("inner");
        fs::create_dir_all(&member).unwrap();
        fs::write(member.join("Cargo.toml"), "[package]\nname = \"inner\"\n").unwrap();

        let found = detect_manifests(dir.path());
        let inner = found
            .iter()
            .find(|m| {
                m.path.ends_with("crates/inner/Cargo.toml")
                    || m.path.ends_with("crates\\inner\\Cargo.toml")
            })
            .unwrap();
        assert!(
            inner.lockfile.is_some(),
            "member crate should resolve the workspace-root Cargo.lock"
        );
    }
}
