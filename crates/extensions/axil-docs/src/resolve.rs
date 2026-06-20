//! Lockfile resolution — pin declared dependencies to exact versions.
//!
//! A manifest declares *ranges* (`^19.2.4`, `1`); the lockfile records
//! the *exact* version actually installed. Dep-docs must be pinned to
//! the locked version, never "latest", so this module reads the lockfile
//! and fills in [`Dependency::version`].

use std::collections::HashMap;
use std::path::Path;

use crate::manifest::{Dependency, DetectedManifest, Ecosystem};
use crate::DocsError;

/// Resolve every dependency in `deps` to its exact lockfile version.
///
/// Dependencies present in the manifest's lockfile get `version` set;
/// those absent — or every dependency, when the manifest has no lockfile
/// at all — keep `version: None` and are thereby reported as *unpinned*.
pub fn resolve_dependencies(
    manifest: &DetectedManifest,
    deps: &mut [Dependency],
) -> Result<(), DocsError> {
    let Some(lockfile) = manifest.lockfile.as_deref() else {
        return Ok(());
    };
    let versions = match manifest.ecosystem {
        Ecosystem::Cargo => load_cargo_lock(lockfile)?,
        Ecosystem::Npm => load_npm_lock(lockfile)?,
        Ecosystem::Python => load_python_lock(lockfile)?,
        // go.mod and pom.xml carry exact versions, so the parser already
        // pinned every dependency — there is nothing to resolve here.
        Ecosystem::Go | Ecosystem::Java => HashMap::new(),
    };
    for dep in deps.iter_mut() {
        // PyPI treats names case- and separator-insensitively; the lock
        // and the manifest can disagree (`Django` vs `django`).
        let key = if manifest.ecosystem == Ecosystem::Python {
            normalize_python_name(&dep.name)
        } else {
            dep.name.clone()
        };
        if let Some(version) = versions.get(&key) {
            dep.version = Some(version.clone());
        }
    }
    Ok(())
}

/// Every package recorded in a manifest's lockfile — the full
/// dependency closure as `name → exact version`.
///
/// This is the *transitive* universe: the lockfile
/// pins every installed package, direct and indirect alike. Returns an
/// empty map when the manifest has no lockfile, or for Go and Java —
/// their manifests pin versions inline, so there is no separate
/// lockfile of transitive versions to read.
pub fn lockfile_packages(
    manifest: &DetectedManifest,
) -> Result<HashMap<String, String>, DocsError> {
    let Some(lockfile) = manifest.lockfile.as_deref() else {
        return Ok(HashMap::new());
    };
    match manifest.ecosystem {
        Ecosystem::Cargo => load_cargo_lock(lockfile),
        Ecosystem::Npm => load_npm_lock(lockfile),
        Ecosystem::Python => load_python_lock(lockfile),
        Ecosystem::Go | Ecosystem::Java => Ok(HashMap::new()),
    }
}

/// PEP 503 name normalization: lowercase, with `_` and `.` folded to `-`.
fn normalize_python_name(name: &str) -> String {
    name.to_lowercase().replace(['_', '.'], "-")
}

/// Map crate name → resolved version from a `Cargo.lock`.
///
/// When a crate appears more than once (multiple versions coexist in the
/// tree) the first occurrence wins; precise per-range matching is a
/// later refinement and does not affect the common single-version case.
fn load_cargo_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let value: toml::Value = toml::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut map = HashMap::new();
    if let Some(packages) = value.get("package").and_then(toml::Value::as_array) {
        for pkg in packages {
            let name = pkg.get("name").and_then(toml::Value::as_str);
            let version = pkg.get("version").and_then(toml::Value::as_str);
            if let (Some(name), Some(version)) = (name, version) {
                map.entry(name.to_string())
                    .or_insert_with(|| version.to_string());
            }
        }
    }
    Ok(map)
}

/// Map package name → resolved version from an npm lockfile.
///
/// `package-lock.json` (v1/v2/v3), `yarn.lock` (v1 and Berry) and
/// `pnpm-lock.yaml` are all parsed.
fn load_npm_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    match path.file_name().and_then(|n| n.to_str()) {
        Some("package-lock.json") => load_package_lock_json(path),
        Some("yarn.lock") => load_yarn_lock(path),
        Some("pnpm-lock.yaml") => load_pnpm_lock(path),
        _ => Ok(HashMap::new()),
    }
}

/// Parse a `package-lock.json` (v1, v2 and v3 layouts).
fn load_package_lock_json(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut map = HashMap::new();
    // v2/v3: a flat `packages` map keyed by install path.
    if let Some(packages) = json.get("packages").and_then(serde_json::Value::as_object) {
        for (key, spec) in packages {
            // Keep only top-level installs — `node_modules/<name>` — and
            // drop nested copies whose path still contains `node_modules/`.
            let Some(name) = key.strip_prefix("node_modules/") else {
                continue;
            };
            if name.contains("node_modules/") {
                continue;
            }
            if let Some(version) = spec.get("version").and_then(serde_json::Value::as_str) {
                map.entry(name.to_string())
                    .or_insert_with(|| version.to_string());
            }
        }
    }
    // v1 fallback: a nested `dependencies` map keyed directly by name.
    if map.is_empty() {
        if let Some(deps) = json
            .get("dependencies")
            .and_then(serde_json::Value::as_object)
        {
            for (name, spec) in deps {
                if let Some(version) = spec.get("version").and_then(serde_json::Value::as_str) {
                    map.insert(name.clone(), version.to_string());
                }
            }
        }
    }
    Ok(map)
}

/// Parse a `yarn.lock` — both the Yarn v1 custom format and the Yarn
/// Berry (v2+) YAML format.
///
/// A key line (column 0, ending `:`) lists one or more `name@range`
/// specifiers; the next indented version line pins them all — v1 writes
/// `version "X"`, Berry writes `version: X`.
fn load_yarn_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let mut map = HashMap::new();
    let mut pending: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with(char::is_whitespace) {
            pending.clear();
            for entry in line.trim_end_matches(':').split(',') {
                if let Some(name) = npm_spec_name(entry.trim().trim_matches('"')) {
                    pending.push(name);
                }
            }
        } else if let Some(rest) = line
            .trim()
            .strip_prefix("version ")
            .or_else(|| line.trim().strip_prefix("version:"))
        {
            // v1 writes `version "1.2.3"`; Berry writes `version: 1.2.3`.
            let version = rest.trim().trim_matches('"');
            if !version.is_empty() {
                for name in pending.drain(..) {
                    map.entry(name).or_insert_with(|| version.to_string());
                }
            }
        }
    }
    Ok(map)
}

/// Parse a `pnpm-lock.yaml` by scanning its package keys.
///
/// pnpm encodes each package as a YAML key `name@version` (optionally
/// `/`-prefixed, with a `(peer@ver)` suffix). A deliberately minimal
/// line scan extracts those — no YAML dependency.
fn load_pnpm_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let mut map = HashMap::new();
    for line in text.lines() {
        if !line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some(key) = line.trim_end().strip_suffix(':') else {
            continue;
        };
        if let Some((name, version)) = pnpm_key_to_name_version(key) {
            map.entry(name).or_insert(version);
        }
    }
    Ok(map)
}

/// Extract the package name from an npm `name@range` specifier.
fn npm_spec_name(spec: &str) -> Option<String> {
    let at = if let Some(rest) = spec.strip_prefix('@') {
        rest.find('@').map(|i| i + 1)
    } else {
        spec.find('@')
    }?;
    let name = &spec[..at];
    (!name.is_empty()).then(|| name.to_string())
}

/// Split a pnpm package key (`/name@version(peer)`) into `(name, version)`.
fn pnpm_key_to_name_version(key: &str) -> Option<(String, String)> {
    let key = key
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .trim_start_matches('/');
    // Drop any trailing `(peer@ver)` suffix.
    let key = key.split('(').next().unwrap_or(key);
    let at = if let Some(rest) = key.strip_prefix('@') {
        rest.find('@').map(|i| i + 1)
    } else {
        key.rfind('@')
    }?;
    let (name, version) = (&key[..at], &key[at + 1..]);
    if name.is_empty() || version.is_empty() {
        None
    } else {
        Some((name.to_string(), version.to_string()))
    }
}

/// Map normalized package name → version from a Python lockfile.
///
/// `uv.lock` / `poetry.lock` (TOML `[[package]]`) and `Pipfile.lock`
/// (JSON `default` / `develop`) are all parsed.
fn load_python_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    match path.file_name().and_then(|n| n.to_str()) {
        Some("uv.lock") | Some("poetry.lock") => load_python_toml_lock(path),
        Some("Pipfile.lock") => load_pipfile_lock(path),
        _ => Ok(HashMap::new()),
    }
}

/// Parse a `uv.lock` / `poetry.lock` — TOML `[[package]]` arrays, the
/// same shape as `Cargo.lock`.
fn load_python_toml_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let value: toml::Value = toml::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut map = HashMap::new();
    if let Some(packages) = value.get("package").and_then(toml::Value::as_array) {
        for pkg in packages {
            let name = pkg.get("name").and_then(toml::Value::as_str);
            let version = pkg.get("version").and_then(toml::Value::as_str);
            if let (Some(name), Some(version)) = (name, version) {
                map.entry(normalize_python_name(name))
                    .or_insert_with(|| version.to_string());
            }
        }
    }
    Ok(map)
}

/// Parse a `Pipfile.lock` — JSON, with the resolved version stored as
/// an `==`-prefixed string under `default` / `develop`.
fn load_pipfile_lock(path: &Path) -> Result<HashMap<String, String>, DocsError> {
    let text = std::fs::read_to_string(path).map_err(|e| DocsError::at(path, e))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| DocsError::at(path, e))?;

    let mut map = HashMap::new();
    for section in ["default", "develop"] {
        let Some(obj) = json.get(section).and_then(serde_json::Value::as_object) else {
            continue;
        };
        for (name, spec) in obj {
            if let Some(version) = spec.get("version").and_then(serde_json::Value::as_str) {
                let version = version.trim().trim_start_matches("==").trim();
                if !version.is_empty() {
                    map.entry(normalize_python_name(name))
                        .or_insert_with(|| version.to_string());
                }
            }
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::DepKind;
    use std::fs;
    use std::path::PathBuf;

    fn dep(name: &str, ecosystem: Ecosystem) -> Dependency {
        Dependency {
            name: name.to_string(),
            ecosystem,
            kind: DepKind::Direct,
            declared_range: "1".to_string(),
            version: None,
        }
    }

    #[test]
    fn resolves_cargo_versions_and_leaves_unlisted_unpinned() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("Cargo.lock");
        fs::write(
            &lock,
            r#"
[[package]]
name = "serde"
version = "1.0.210"

[[package]]
name = "tokio"
version = "1.40.0"
"#,
        )
        .unwrap();

        let manifest = DetectedManifest {
            path: dir.path().join("Cargo.toml"),
            ecosystem: Ecosystem::Cargo,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("serde", Ecosystem::Cargo),
            dep("not-in-lock", Ecosystem::Cargo),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();

        assert_eq!(deps[0].version.as_deref(), Some("1.0.210"));
        assert_eq!(deps[1].version, None, "absent deps stay unpinned");
    }

    #[test]
    fn resolves_npm_versions_from_package_lock_v3() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("package-lock.json");
        fs::write(
            &lock,
            r#"{
  "lockfileVersion": 3,
  "packages": {
    "": { "name": "demo" },
    "node_modules/react": { "version": "19.2.4" },
    "node_modules/@scope/pkg": { "version": "1.2.3" },
    "node_modules/react/node_modules/nested": { "version": "9.9.9" }
  }
}"#,
        )
        .unwrap();

        let manifest = DetectedManifest {
            path: dir.path().join("package.json"),
            ecosystem: Ecosystem::Npm,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("react", Ecosystem::Npm),
            dep("@scope/pkg", Ecosystem::Npm),
            dep("nested", Ecosystem::Npm),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();

        assert_eq!(deps[0].version.as_deref(), Some("19.2.4"));
        assert_eq!(deps[1].version.as_deref(), Some("1.2.3"));
        assert_eq!(deps[2].version, None, "nested copies are not top-level");
    }

    #[test]
    fn resolves_python_versions_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("poetry.lock");
        fs::write(
            &lock,
            "[[package]]\nname = \"django\"\nversion = \"5.0.1\"\n",
        )
        .unwrap();
        let manifest = DetectedManifest {
            path: dir.path().join("requirements.txt"),
            ecosystem: Ecosystem::Python,
            lockfile: Some(lock),
        };
        // The manifest spells it `Django` — PEP 503 normalization must
        // still match the lowercase lock entry.
        let mut deps = vec![dep("Django", Ecosystem::Python)];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version.as_deref(), Some("5.0.1"));
    }

    #[test]
    fn resolves_yarn_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("yarn.lock");
        fs::write(
            &lock,
            "# yarn lockfile v1\n\
             \"react@^19.0.0\", react@^19.2.0:\n  version \"19.2.4\"\n  resolved \"x\"\n\n\
             \"@babel/core@^7.0.0\":\n  version \"7.23.9\"\n",
        )
        .unwrap();
        let manifest = DetectedManifest {
            path: dir.path().join("package.json"),
            ecosystem: Ecosystem::Npm,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("react", Ecosystem::Npm),
            dep("@babel/core", Ecosystem::Npm),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version.as_deref(), Some("19.2.4"));
        assert_eq!(deps[1].version.as_deref(), Some("7.23.9"));
    }

    #[test]
    fn resolves_yarn_berry_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("yarn.lock");
        // Berry's YAML format: `version: X` (no quotes) instead of v1's
        // `version "X"`. The leading `__metadata:` block must not leak.
        fs::write(
            &lock,
            "__metadata:\n  version: 8\n\n\
             \"react@npm:^19.0.0\":\n  version: 19.2.4\n  resolution: \"react@npm:19.2.4\"\n\n\
             \"@babel/core@npm:^7.0.0\":\n  version: 7.23.9\n",
        )
        .unwrap();
        let manifest = DetectedManifest {
            path: dir.path().join("package.json"),
            ecosystem: Ecosystem::Npm,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("react", Ecosystem::Npm),
            dep("@babel/core", Ecosystem::Npm),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version.as_deref(), Some("19.2.4"));
        assert_eq!(deps[1].version.as_deref(), Some("7.23.9"));
    }

    #[test]
    fn resolves_pnpm_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("pnpm-lock.yaml");
        fs::write(
            &lock,
            "lockfileVersion: '9.0'\n\npackages:\n\n  \
             react@19.2.4:\n    resolution: {integrity: sha512-x}\n\n  \
             '@babel/core@7.23.9':\n    resolution: {integrity: sha512-y}\n\n  \
             react-dom@19.2.4(react@19.2.4):\n    resolution: {integrity: sha512-z}\n",
        )
        .unwrap();
        let manifest = DetectedManifest {
            path: dir.path().join("package.json"),
            ecosystem: Ecosystem::Npm,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("react", Ecosystem::Npm),
            dep("@babel/core", Ecosystem::Npm),
            dep("react-dom", Ecosystem::Npm),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version.as_deref(), Some("19.2.4"));
        assert_eq!(deps[1].version.as_deref(), Some("7.23.9"));
        assert_eq!(
            deps[2].version.as_deref(),
            Some("19.2.4"),
            "the (peer) suffix is stripped"
        );
    }

    #[test]
    fn resolves_pipfile_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("Pipfile.lock");
        fs::write(
            &lock,
            r#"{"default":{"requests":{"version":"==2.31.0"}},"develop":{"pytest":{"version":"==8.0.0"}}}"#,
        )
        .unwrap();
        let manifest = DetectedManifest {
            path: dir.path().join("Pipfile"),
            ecosystem: Ecosystem::Python,
            lockfile: Some(lock),
        };
        let mut deps = vec![
            dep("requests", Ecosystem::Python),
            dep("pytest", Ecosystem::Python),
        ];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version.as_deref(), Some("2.31.0"));
        assert_eq!(deps[1].version.as_deref(), Some("8.0.0"));
    }

    #[test]
    fn no_lockfile_leaves_everything_unpinned() {
        let manifest = DetectedManifest {
            path: PathBuf::from("Cargo.toml"),
            ecosystem: Ecosystem::Cargo,
            lockfile: None,
        };
        let mut deps = vec![dep("serde", Ecosystem::Cargo)];
        resolve_dependencies(&manifest, &mut deps).unwrap();
        assert_eq!(deps[0].version, None);
    }
}
