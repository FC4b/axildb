//! Phase 17 P3 (deferred slice) — multi-manifest ingestion pipeline.
//!
//! The functions here previously lived as `axil-cli`-private helpers
//! (`deps_collect_unique`, `collect_gated_transitive`,
//! `deps_ingest_manifests`, `deps_sweep_removed`). They are pure
//! dep-doc domain logic — nothing about argv parsing or the
//! `Output` printer is in here — so they relocate cleanly to
//! `axil-docs`, which then lets `DocsExtension::handle_cli` cover
//! `deps sync` and `deps refresh` without depending on the CLI.
//!
//! Error mapping: every function returns `Result<_, DocsError>`. The
//! CLI adapts back to `anyhow::Result` with one `map_err` at the
//! call site.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use axil_core::Axil;
use serde_json::{json, Value};

use crate::imports::scan_project_imports;
use crate::ingest::{
    active_dep_version, diff_dep_docs, ingest_dep_docs, ingest_migration_note,
    sweep_removed_deps, DEFAULT_MAX_CHUNKS_PER_DEP,
};
use crate::local::{extract_changelog, extract_local_doc, NotFound};
use crate::manifest::{
    detect_manifests as detect_manifests_fn, parse_manifest, DepKind, Dependency, DetectedManifest,
    Ecosystem,
};
use crate::refresh::record_manifest_state;
use crate::resolve::{lockfile_packages, resolve_dependencies};
use crate::DocsError;

/// Collect the unique direct (and optionally dev) dependencies declared
/// across a set of manifests, dedup'd by `(ecosystem, name, version)`.
///
/// Skips `path = "..."` deps because they are workspace-local and
/// have no registry doc to extract.
pub fn collect_unique_deps(
    manifests: &[DetectedManifest],
    include_dev: bool,
) -> Result<Vec<Dependency>, DocsError> {
    let mut unique: BTreeMap<(String, String, String), Dependency> = BTreeMap::new();
    for manifest in manifests {
        let mut deps = parse_manifest(manifest)?;
        resolve_dependencies(manifest, &mut deps)?;
        for d in deps {
            if !include_dev && d.kind != DepKind::Direct {
                continue;
            }
            if d.declared_range == "path" {
                continue;
            }
            let key = (
                d.ecosystem.as_str().to_string(),
                d.name.clone(),
                d.version.clone().unwrap_or_default(),
            );
            unique.entry(key).or_insert(d);
        }
    }
    Ok(unique.into_values().collect())
}

/// Collect the transitive dependencies a project's own source actually
/// imports (Phase 16 P1.a).
///
/// The lockfile closure minus the manifest-declared deps is the
/// transitive universe; it is intersected with a scan of the project's
/// own source so only the transitive deps the code really `use`s /
/// `import`s come back. Cargo and npm only — other ecosystems yield
/// nothing (their import token does not map cleanly to a package name).
pub fn collect_gated_transitive(
    manifests: &[DetectedManifest],
    project_root: &Path,
) -> Result<Vec<Dependency>, DocsError> {
    // The full transitive universe: lockfile closure minus declared deps.
    let mut transitive: BTreeMap<(String, String), Dependency> = BTreeMap::new();
    for manifest in manifests {
        let closure = lockfile_packages(manifest)?;
        if closure.is_empty() {
            continue;
        }
        let declared: HashSet<String> = parse_manifest(manifest)?
            .into_iter()
            .map(|d| d.name)
            .collect();
        for (name, version) in closure {
            if declared.contains(&name) {
                continue;
            }
            let key = (manifest.ecosystem.as_str().to_string(), name.clone());
            transitive.entry(key).or_insert(Dependency {
                name,
                ecosystem: manifest.ecosystem,
                kind: DepKind::Transitive,
                declared_range: "transitive".to_string(),
                version: Some(version),
            });
        }
    }
    if transitive.is_empty() {
        return Ok(Vec::new());
    }

    // Gate on the project's own imports — scanned once per ecosystem.
    let mut imports: HashMap<&'static str, HashSet<String>> = HashMap::new();
    let mut kept = Vec::new();
    for dep in transitive.into_values() {
        let imported = imports
            .entry(dep.ecosystem.as_str())
            .or_insert_with(|| scan_project_imports(project_root, dep.ecosystem));
        let used = match dep.ecosystem {
            // Rust `use` statements write a crate's hyphens as underscores.
            Ecosystem::Cargo => imported.contains(&dep.name.replace('-', "_")),
            Ecosystem::Npm => imported.contains(&dep.name),
            _ => false,
        };
        if used {
            kept.push(dep);
        }
    }
    Ok(kept)
}

/// Detect, resolve, extract and ingest the registry dependencies of
/// `manifests`, then record each manifest's hash state for drift
/// detection.
///
/// With `transitive`, the transitive deps the project's own source
/// imports are ingested too (Phase 16 P1.a).
///
/// Returns the run summary as JSON. Schema:
/// ```text
/// {
///   "deps_seen":          <int>,
///   "ingested":           <int>,
///   "transitive":         <int>,
///   "chunks":             <int>,
///   "needs_web_fallback": [<dep>],
///   "not_installed":      [<dep>],
///   "migrations":         [<bump>],
///   "doc_diffs":          <int>,
/// }
/// ```
pub fn ingest_manifests(
    db: &Axil,
    manifests: &[DetectedManifest],
    project_root: &Path,
    transitive: bool,
) -> Result<Value, DocsError> {
    let mut to_ingest = collect_unique_deps(manifests, false)?;
    if transitive {
        let direct: HashSet<(String, String)> = to_ingest
            .iter()
            .map(|d| (d.ecosystem.as_str().to_string(), d.name.clone()))
            .collect();
        for t in collect_gated_transitive(manifests, project_root)? {
            if !direct.contains(&(t.ecosystem.as_str().to_string(), t.name.clone())) {
                to_ingest.push(t);
            }
        }
    }
    let unique = to_ingest;

    let mut ingested = 0usize;
    let mut transitive_ingested = 0usize;
    let mut chunks_total = 0usize;
    let mut needs_web_fallback: Vec<Value> = Vec::new();
    let mut not_installed: Vec<Value> = Vec::new();
    let mut migrations: Vec<Value> = Vec::new();
    let mut doc_diffs = 0usize;
    for dep in &unique {
        // Capture the previously-ingested version so a version bump can
        // be detected (P1.b changelog memory + P1.c doc diffing).
        let prev_version = active_dep_version(db, &dep.name, dep.ecosystem)?;
        match extract_local_doc(dep, project_root) {
            Ok(doc) => {
                let n = ingest_dep_docs(
                    db,
                    dep,
                    &doc.text,
                    "local",
                    DEFAULT_MAX_CHUNKS_PER_DEP,
                )?;
                ingested += 1;
                if dep.kind == DepKind::Transitive {
                    transitive_ingested += 1;
                }
                chunks_total += n;
                if doc.sparse {
                    needs_web_fallback.push(json!({
                        "name": dep.name,
                        "ecosystem": dep.ecosystem.as_str(),
                        "version": dep.version,
                    }));
                }
                // On a version bump: capture the changelog as
                // migration notes (P1.b) and diff the docs across the
                // bump (P1.c).
                if let Some(prev) = prev_version.as_deref() {
                    if Some(prev) != dep.version.as_deref() {
                        if let Some(changelog) = extract_changelog(dep, project_root) {
                            let m = ingest_migration_note(db, dep, prev, &changelog)?;
                            if m > 0 {
                                migrations.push(json!({
                                    "name": dep.name,
                                    "ecosystem": dep.ecosystem.as_str(),
                                    "from": prev,
                                    "to": dep.version,
                                    "chunks": m,
                                }));
                            }
                        }
                        if diff_dep_docs(db, dep, prev)? > 0 {
                            doc_diffs += 1;
                        }
                    }
                }
            }
            Err(NotFound::NotInstalled) => {
                not_installed.push(json!({
                    "name": dep.name,
                    "ecosystem": dep.ecosystem.as_str(),
                    "version": dep.version,
                }));
            }
            Err(_) => {}
        }
    }

    // Record each manifest's hash state so `deps refresh` has a baseline.
    for manifest in manifests {
        record_manifest_state(db, manifest)?;
    }

    Ok(json!({
        "deps_seen": unique.len(),
        "ingested": ingested,
        "transitive": transitive_ingested,
        "chunks": chunks_total,
        "needs_web_fallback": needs_web_fallback,
        "not_installed": not_installed,
        "migrations": migrations,
        "doc_diffs": doc_diffs,
    }))
}

/// Mark dependencies no longer declared in any manifest as `removed`
/// — the *removed* set of the P0.4 drift diff.
///
/// Keyed on the **full** detected manifest set, so a partial
/// `--if-stale` refresh never mistakes a still-declared dependency for
/// a removed one. Returns the names of the swept deps.
pub fn sweep_removed_for_manifests(
    db: &Axil,
    all_manifests: &[DetectedManifest],
) -> Result<Vec<String>, DocsError> {
    let current = collect_unique_deps(all_manifests, false)?;
    sweep_removed_deps(db, &current)
}

/// Convenience: re-export `detect_manifests` under the pipeline
/// namespace so a caller can do the whole sync flow without reaching
/// across multiple `axil_docs::` submodules.
pub use crate::manifest::detect_manifests;
// Suppress unused-import warning when only `detect_manifests` is
// surfaced via re-export. The function itself is used inside the crate
// via the explicit re-export above; the underscore alias here is a
// no-op for type checking.
#[allow(dead_code)]
fn _force_detect_manifests_visible() {
    let _ = detect_manifests_fn;
}
