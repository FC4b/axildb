//! First-party import scan (Phase 16 P1.a).
//!
//! Which external packages does the project's *own* source actually
//! `use` / `import`? Transitive-dependency doc ingestion is gated on
//! this: the lockfile closure is large, so a transitive dependency is
//! worth ingesting only when project code imports it directly.
//!
//! Only Cargo (`.rs`) and npm (`.js`/`.ts` family) are scanned — the
//! ecosystems whose import token maps cleanly to a package name.

use std::collections::HashSet;
use std::path::Path;

use crate::manifest::Ecosystem;

/// Cap on source files scanned, so the walk cannot run away on a
/// pathological tree.
const MAX_SCANNED_FILES: usize = 20_000;

/// Scan a project tree for the top-level package names its source
/// files import.
///
/// Returns an empty set for ecosystems other than Cargo and npm. The
/// caller intersects this with the lockfile's transitive closure, so a
/// stray non-package token (a stdlib path, a `crate::` segment) is
/// harmless — it simply matches nothing.
pub fn scan_project_imports(root: &Path, ecosystem: Ecosystem) -> HashSet<String> {
    let extensions: &[&str] = match ecosystem {
        Ecosystem::Cargo => &["rs"],
        Ecosystem::Npm => &["js", "jsx", "ts", "tsx", "mjs", "cjs"],
        Ecosystem::Python | Ecosystem::Go | Ecosystem::Java => return HashSet::new(),
    };
    let mut found = HashSet::new();
    let mut budget = MAX_SCANNED_FILES;
    walk(root, extensions, &mut budget, &mut |text| match ecosystem {
        Ecosystem::Cargo => collect_rust_imports(text, &mut found),
        Ecosystem::Npm => collect_npm_imports(text, &mut found),
        _ => {}
    });
    found
}

/// Recursively walk `dir`, passing the contents of every file whose
/// extension is in `exts` to `visit`. Skips build/vendor/VCS/hidden
/// directories.
fn walk(dir: &Path, exts: &[&str], budget: &mut usize, visit: &mut impl FnMut(&str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if *budget == 0 {
            return;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if matches!(name.as_ref(), "target" | "node_modules" | "vendor")
                || name.starts_with('.')
            {
                continue;
            }
            walk(&path, exts, budget, visit);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| exts.contains(&e))
            .unwrap_or(false)
        {
            if let Ok(text) = std::fs::read_to_string(&path) {
                visit(&text);
            }
            *budget -= 1;
        }
    }
}

/// Collect crate names from Rust `use` / `pub use` / `extern crate`
/// statements — the first `::`-delimited path segment.
fn collect_rust_imports(text: &str, out: &mut HashSet<String>) {
    for line in text.lines() {
        let t = line.trim_start();
        let rest = t
            .strip_prefix("use ")
            .or_else(|| t.strip_prefix("pub use "))
            .or_else(|| t.strip_prefix("extern crate "));
        let Some(rest) = rest else {
            continue;
        };
        let seg: String = rest
            .trim_start_matches("::")
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !seg.is_empty() {
            out.insert(seg);
        }
    }
}

/// Collect package names from npm `import`/`export … from "x"` and
/// `require("x")` / `import("x")` statements.
fn collect_npm_imports(text: &str, out: &mut HashSet<String>) {
    for line in text.lines() {
        let t = line.trim_start();
        let looks_like_import = t.starts_with("import ")
            || t.starts_with("import(")
            || t.starts_with("export ")
            || t.contains("require(");
        if !looks_like_import {
            continue;
        }
        if let Some(spec) = first_quoted(line) {
            if let Some(pkg) = npm_package_of(spec) {
                out.insert(pkg);
            }
        }
    }
}

/// The first single- or double-quoted substring on a line.
fn first_quoted(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let start = bytes.iter().position(|&b| b == b'"' || b == b'\'')?;
    let quote = bytes[start];
    let rel_end = bytes[start + 1..].iter().position(|&b| b == quote)?;
    line.get(start + 1..start + 1 + rel_end)
}

/// The package name an npm module specifier resolves to — `@scope/name`
/// for a scoped package, the first path segment otherwise. Relative and
/// absolute specifiers (`./x`, `/x`) are not packages.
fn npm_package_of(spec: &str) -> Option<String> {
    if spec.is_empty() || spec.starts_with('.') || spec.starts_with('/') {
        return None;
    }
    if let Some(scoped) = spec.strip_prefix('@') {
        let mut parts = scoped.splitn(3, '/');
        let scope = parts.next().filter(|s| !s.is_empty())?;
        let name = parts.next().filter(|s| !s.is_empty())?;
        Some(format!("@{scope}/{name}"))
    } else {
        let first = spec.split('/').next()?;
        (!first.is_empty()).then(|| first.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn rust_imports_pick_the_crate_segment() {
        let mut out = HashSet::new();
        collect_rust_imports(
            "use tokio::net::TcpListener;\n\
             pub use serde_json::Value;\n\
             extern crate libc;\n\
             use crate::internal::thing;\n\
             use ::anyhow::Result;\n\
             let x = 1;\n",
            &mut out,
        );
        assert!(out.contains("tokio"));
        assert!(out.contains("serde_json"));
        assert!(out.contains("libc"));
        assert!(out.contains("anyhow"), "leading :: is tolerated");
        assert!(out.contains("crate"), "noise is harmless — matches no dep");
    }

    #[test]
    fn npm_imports_pick_the_package() {
        let mut out = HashSet::new();
        collect_npm_imports(
            "import React from \"react\";\n\
             import {pad} from 'left-pad/lib';\n\
             export {x} from \"@babel/core\";\n\
             const fs = require('node:fs');\n\
             import local from './local';\n",
            &mut out,
        );
        assert!(out.contains("react"));
        assert!(out.contains("left-pad"), "subpath is stripped");
        assert!(out.contains("@babel/core"), "scoped package kept whole");
        assert!(out.contains("node:fs"));
        assert!(!out.iter().any(|p| p.starts_with('.')), "relative skipped");
    }

    #[test]
    fn npm_package_of_handles_scopes_and_subpaths() {
        assert_eq!(npm_package_of("left-pad").as_deref(), Some("left-pad"));
        assert_eq!(
            npm_package_of("left-pad/lib/x").as_deref(),
            Some("left-pad")
        );
        assert_eq!(
            npm_package_of("@scope/pkg/sub").as_deref(),
            Some("@scope/pkg")
        );
        assert_eq!(npm_package_of("./local"), None);
        assert_eq!(npm_package_of("@scope"), None);
    }

    #[test]
    fn scan_walks_the_tree_and_skips_vendor_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(
            src.join("main.rs"),
            "use tokio::spawn;\nuse bytes::Bytes;\n",
        )
        .unwrap();
        // A file under target/ must be ignored.
        let target = dir.path().join("target");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("gen.rs"), "use should_be_ignored::X;\n").unwrap();

        let found = scan_project_imports(dir.path(), Ecosystem::Cargo);
        assert!(found.contains("tokio") && found.contains("bytes"));
        assert!(!found.contains("should_be_ignored"), "target/ is skipped");
        // A non-Cargo/npm ecosystem yields nothing.
        assert!(scan_project_imports(dir.path(), Ecosystem::Go).is_empty());
    }
}
