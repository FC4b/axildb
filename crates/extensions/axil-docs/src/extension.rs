//! `axil-docs` as a Tier-2 [`Extension`] — the canonical reference
//! implementation for the rest of the project to model against.
//!
//! Implements every method on the [`axil_core::Extension`] trait:
//! identity + table-prefix registration, a boot block summarising
//! manifest drift, `mcp_tools` + `handle_mcp` covering the
//! `deps_status` and `dep_docs` tools, and `handle_cli` covering all
//! 5 `axil deps …` subcommands. JSON shapes mirror what `axil-cli`
//! and `axil-mcp` produce so Adapters can fall through (Path C)
//! without behavior drift.

use std::path::PathBuf;

use axil_core::{
    Axil, CliInvocation, CliOutput, Dispatch, Extension, McpCall, McpSurface, McpTool,
};
use axil_core::error::Result as CoreResult;
use serde_json::json;

use crate::{detect_manifests, manifest_drift, TABLE_DEPS, TABLE_DEP_MANIFESTS};

/// `axil-docs` as an Axil Extension.
///
/// Construct with [`DocsExtension::default`] and register through
/// [`axil_core::AxilBuilder::with_extension`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DocsExtension;

impl Extension for DocsExtension {
    fn id(&self) -> &str {
        "docs"
    }

    fn display_name(&self) -> &str {
        "Dependency Docs"
    }

    /// Tables owned by `axil-docs`: `_deps`, `_dep_docs`,
    /// `_dep_manifests`. The single declared prefix `_dep` is a prefix
    /// of all three.
    ///
    /// > **Note on naming:** the prefix is `_dep` (legacy —    /// > predates the convention). The `_deps` table name
    /// > without a trailing underscore is why the prefix isn't `_dep_`.
    /// > New Extensions should use `_<id>_` (e.g. `_myext_`); see
    /// > `docs/src/extending/extensions.md`.
    fn table_prefixes(&self) -> &[&str] {
        // `_dep` covers `_deps`, `_dep_docs`, `_dep_manifests`.
        // Asserted in this module's tests.
        &["_dep"]
    }

    /// Mirror the `dep_docs_freshness` block `axil-cli` already emits.
    ///
    /// Returns a one-line summary when at least one manifest is drifted
    /// *and* the project has synced at least once (the same gate the
    /// CLI uses). Returns `None` otherwise so the boot output stays
    /// quiet for projects that don't use dep-docs.
    ///
    /// The project root is derived from `db.path()` by walking up
    /// until we find a manifest-bearing directory — matching the CLI's
    /// `detect_project_root` heuristic, scoped to this Extension so
    /// the trait method stays self-contained.
    fn boot_block(&self, db: &Axil) -> Option<String> {
        // Gate: only nag once the project has actually run `deps sync`.
        let synced = db
            .list(TABLE_DEP_MANIFESTS)
            .map(|rows| !rows.is_empty())
            .unwrap_or(false);
        if !synced {
            return None;
        }

        let root = detect_project_root(db.path())?;
        let stale = detect_manifests(&root)
            .iter()
            .filter(|m| matches!(manifest_drift(db, m), Ok(d) if d.needs_sync()))
            .count();
        if stale == 0 {
            return None;
        }
        Some(format!(
            "dep-docs: {stale} manifest{} drifted — run `axil deps refresh --if-stale`",
            if stale == 1 { "" } else { "s" }
        ))
    }

    /// MCP tool surface for `axil-docs`.
    ///
    /// Today this exposes the read-only `deps_status` tool (the same
    /// shape `axil-mcp` already serves via its hard-coded handler).
    /// `dep_docs` is intentionally not yet included — it has parameter
    /// validation that's easier to migrate once `axil-mcp` switches to
    /// the composed-surface dispatch path. Until then `axil-mcp` keeps
    /// its hard-coded `dep_docs` handler; Path C's NotHandled fallback
    /// means everything keeps working.
    fn mcp_tools(&self) -> Option<McpSurface> {
        Some(McpSurface::new(vec![
            McpTool::new(
                "deps_status",
                "List the dependencies whose docs are in memory: name, resolved version, ecosystem and stored doc-chunk count.",
                json!({ "type": "object", "properties": {} }),
            ),
            McpTool::new(
                "dep_docs",
                "Scoped query over the dependency-doc memory. Returns version-pinned chunks for the project's dependencies.",
                json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Library question or API name to search for." },
                            "top_k": { "type": "integer", "description": "How many hits to return. Default 5." },
                            "dep":   { "type": "string", "description": "Restrict to a single dependency by name." },
                            "include_superseded": { "type": "boolean", "description": "Include archived chunks from superseded versions. Default false." }
                        },
                        "required": ["query"]
                    }),
            ),
        ]))
    }

    /// MCP dispatch handler.
    ///
    /// Routes `deps_status` to its existing logic (mirror of
    /// `axil-mcp::tools::handle_deps_status`). Any other tool name —
    /// including `dep_docs`, which still lives in `axil-mcp` until
    /// migration — returns `Dispatch::NotHandled` so the Adapter
    /// falls back to its hard-coded path.
    fn handle_mcp(
        &self,
        db: &Axil,
        call: &McpCall,
    ) -> CoreResult<Dispatch<serde_json::Value>> {
        match call.tool.as_str() {
            "deps_status" => {
                let rows = db.list(TABLE_DEPS)?;
                let deps: Vec<serde_json::Value> =
                    rows.iter().map(|r| r.data.clone()).collect();
                Ok(Dispatch::Handled(json!({
                    "synced_deps": deps.len(),
                    "deps": deps,
                })))
            }
            "dep_docs" => {
                let query = call
                    .params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| wrap_input_err("dep_docs: missing required parameter `query`"))?;
                let top_k = call
                    .params
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                let dep = call.params.get("dep").and_then(|v| v.as_str());
                let include_superseded = call
                    .params
                    .get("include_superseded")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let hits = crate::query_dep_docs(db, query, top_k, dep, include_superseded)
                    .map_err(wrap_io_err)?;
                let value = serde_json::to_value(&hits).unwrap_or_else(|_| json!([]));
                Ok(Dispatch::Handled(value))
            }
            // Path C — anything else is the Adapter's problem.
            _ => Ok(Dispatch::NotHandled),
        }
    }

    /// CLI dispatch handler.
    ///
    /// Handles `deps status [--path <dir>]`, mirroring `axil-cli`'s
    /// `run_deps`'s `DepsCommand::Status` JSON shape exactly. Path C
    /// semantics: every other subcommand returns `Dispatch::NotHandled`
    /// so `axil-cli` falls back to its hard-coded handler.
    ///
    /// Migrating more subcommands is a follow-up; `status` is the
    /// simplest read-only example and serves as the proof point for
    /// the dispatch contract.
    fn handle_cli(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        // We only claim the `deps` top-level command — anything else
        // is the Adapter's problem.
        if invocation.command_path.len() < 2 || invocation.command_path[0] != "deps" {
            return Ok(Dispatch::NotHandled);
        }
        match invocation.command_path[1].as_str() {
            "status" => self.handle_deps_status(db, invocation),
            "list" => self.handle_deps_list(invocation),
            "ingest" => self.handle_deps_ingest(db, invocation),
            "sync" => self.handle_deps_sync(db, invocation),
            "refresh" => self.handle_deps_refresh(db, invocation),
            _ => Ok(Dispatch::NotHandled),
        }
    }
}

impl DocsExtension {
    /// Migrated handler for `axil deps status [--path <dir>]`.
    /// JSON-shape compatible with axil-cli's pre-migration `run_deps`
    /// `DepsCommand::Status` branch.
    fn handle_deps_status(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        let path = parse_path_arg(&invocation.args).unwrap_or_else(|| PathBuf::from("."));

        let mut deps: Vec<serde_json::Value> =
            db.list(TABLE_DEPS)?.iter().map(|r| r.data.clone()).collect();
        deps.sort_by(|a, b| {
            let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
            an.cmp(bn)
        });
        let (manifests_json, _stale) = drift_report(db, &detect_manifests(&path), true)?;
        let value = json!({
            "synced_deps": deps.len(),
            "deps": deps,
            "manifests": manifests_json,
        });
        Ok(Dispatch::Handled(json_stdout(value)))
    }

    /// Migrated handler for `axil deps list [--path <dir>] [--dev]`.
    /// JSON-shape compatible with axil-cli's pre-migration `run_deps`
    /// `DepsCommand::List` branch. Read-only — does not open the DB.
    fn handle_deps_list(&self, invocation: &CliInvocation) -> CoreResult<Dispatch<CliOutput>> {
        let path = parse_path_arg(&invocation.args).unwrap_or_else(|| PathBuf::from("."));
        let include_dev = flag_set(&invocation.args, "--dev");

        let manifests = detect_manifests(&path);
        let list = match crate::collect_unique_deps(&manifests, include_dev) {
            Ok(v) => v,
            Err(e) => return Err(wrap_io_err(e)),
        };
        let deps_json: Vec<serde_json::Value> = list
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "ecosystem": d.ecosystem.as_str(),
                    "kind": d.kind.as_str(),
                    "declared_range": d.declared_range,
                    "version": d.version,
                    "pinned": d.version.is_some(),
                })
            })
            .collect();
        let value = json!({
            "manifests": manifests.len(),
            "dependencies": deps_json.len(),
            "deps": deps_json,
        });
        Ok(Dispatch::Handled(json_stdout(value)))
    }

    /// Migrated handler for
    /// `axil deps sync [--path <dir>] [--offline] [--transitive]`.
    /// JSON shape matches axil-cli's pre-migration `run_deps`
    /// `DepsCommand::Sync` branch.
    fn handle_deps_sync(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        let path = parse_path_arg(&invocation.args).unwrap_or_else(|| PathBuf::from("."));
        let transitive = flag_set(&invocation.args, "--transitive");
        // `--offline` is currently the only supported mode; kept for
        // forward-compat parity with the clap surface.
        let _offline = flag_set(&invocation.args, "--offline");

        let manifests = crate::detect_manifests(&path);
        let mut summary =
            crate::ingest_manifests(db, &manifests, &path, transitive).map_err(wrap_io_err)?;
        let removed = crate::sweep_removed_for_manifests(db, &manifests).map_err(wrap_io_err)?;
        summary["removed"] = json!(removed);
        summary["manifests"] = json!(manifests.len());
        Ok(Dispatch::Handled(json_stdout(summary)))
    }

    /// Migrated handler for
    /// `axil deps refresh [--path <dir>] [--if-stale] [--transitive]`.
    /// JSON shape matches axil-cli's pre-migration `run_deps`
    /// `DepsCommand::Refresh` branch.
    fn handle_deps_refresh(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        let path = parse_path_arg(&invocation.args).unwrap_or_else(|| PathBuf::from("."));
        let if_stale = flag_set(&invocation.args, "--if-stale");
        let transitive = flag_set(&invocation.args, "--transitive");

        let all = crate::detect_manifests(&path);
        let (drift_json, stale) = drift_report(db, &all, false)?;

        // `--if-stale` fast exit: nothing changed since the last sync.
        if if_stale && stale.is_empty() {
            let value = json!({
                "manifests": all.len(),
                "refreshed": 0,
                "status": "fresh",
            });
            return Ok(Dispatch::Handled(json_stdout(value)));
        }

        let to_sync: &[crate::DetectedManifest] = if if_stale { &stale } else { &all };
        let mut summary =
            crate::ingest_manifests(db, to_sync, &path, transitive).map_err(wrap_io_err)?;
        let removed = crate::sweep_removed_for_manifests(db, &all).map_err(wrap_io_err)?;
        summary["removed"] = json!(removed);
        summary["manifests"] = json!(all.len());
        summary["refreshed"] = json!(to_sync.len());
        summary["drift"] = json!(drift_json);
        Ok(Dispatch::Handled(json_stdout(summary)))
    }

    /// Migrated handler for
    /// `axil deps ingest --dep <name@version> --ecosystem <eco> [--file <path>] [--from-web]`.
    /// Reads the doc text from the named file, from stdin (if no file),
    /// or — when the `web-docs` cargo feature is enabled — from the
    /// network. Calls [`crate::ingest_dep_docs`] and returns the same
    /// JSON shape `axil-cli`'s `DepsCommand::Ingest` branch produces.
    fn handle_deps_ingest(
        &self,
        db: &Axil,
        invocation: &CliInvocation,
    ) -> CoreResult<Dispatch<CliOutput>> {
        // Required: --dep name@version, --ecosystem <eco>.
        let dep_arg = match named_arg(&invocation.args, "--dep") {
            Some(v) => v,
            None => return Err(wrap_input_err("deps ingest: --dep is required")),
        };
        let (name, version) = match dep_arg.rsplit_once('@') {
            Some(pair) => pair,
            None => return Err(wrap_input_err("deps ingest: --dep must be name@version")),
        };
        let eco_arg = match named_arg(&invocation.args, "--ecosystem") {
            Some(v) => v,
            None => return Err(wrap_input_err("deps ingest: --ecosystem is required")),
        };
        let eco = match crate::Ecosystem::from_str(&eco_arg) {
            Some(e) => e,
            None => return Err(wrap_input_err(&format!("unknown ecosystem: {eco_arg}"))),
        };

        let dependency = crate::Dependency {
            name: name.to_string(),
            ecosystem: eco,
            kind: crate::DepKind::Direct,
            declared_range: "agent".to_string(),
            version: Some(version.to_string()),
        };

        let from_web = flag_set(&invocation.args, "--from-web");
        let file = named_arg(&invocation.args, "--file");

        let (text, source): (String, &str) = if from_web {
            #[cfg(feature = "web-docs")]
            {
                match crate::fetch_web_doc(&dependency) {
                    Some(t) => (t, "web"),
                    None => {
                        return Err(wrap_input_err(&format!(
                            "deps ingest: web fetch returned no docs for {dep_arg}"
                        )))
                    }
                }
            }
            #[cfg(not(feature = "web-docs"))]
            {
                return Err(wrap_input_err(
                    "deps ingest --from-web requires the `web-docs` cargo feature",
                ));
            }
        } else if let Some(p) = file {
            let p = PathBuf::from(p);
            let t = std::fs::read_to_string(&p)
                .map_err(|e| wrap_input_err(&format!("reading {}: {e}", p.display())))?;
            (t, "agent")
        } else {
            // Adapter is responsible for stdin capture into invocation.stdin.
            match &invocation.stdin {
                Some(s) => (s.clone(), "agent"),
                None => {
                    return Err(wrap_input_err(
                        "deps ingest: doc text required via --file <path> or stdin",
                    ))
                }
            }
        };

        let n = crate::ingest_dep_docs(
            db,
            &dependency,
            &text,
            source,
            crate::DEFAULT_MAX_CHUNKS_PER_DEP,
        )
        .map_err(wrap_io_err)?;
        let value = json!({
            "dep": name,
            "version": version,
            "ecosystem": eco.as_str(),
            "chunks": n,
            "source": source,
        });
        Ok(Dispatch::Handled(json_stdout(value)))
    }
}

/// Walk `manifests` and compute their drift state, returning the
/// per-manifest JSON entries (the shape both `deps status` and
/// `deps refresh` emit) and the subset that needs re-syncing.
///
/// When `include_ecosystem` is true, each JSON entry has an
/// `ecosystem` field — `deps status` includes it; `deps refresh`
/// drops it to match the historical `run_deps` shape.
fn drift_report(
    db: &Axil,
    manifests: &[crate::DetectedManifest],
    include_ecosystem: bool,
) -> CoreResult<(Vec<serde_json::Value>, Vec<crate::DetectedManifest>)> {
    let mut report = Vec::with_capacity(manifests.len());
    let mut stale = Vec::new();
    for manifest in manifests {
        let drift = manifest_drift(db, manifest).map_err(wrap_io_err)?;
        let mut entry = json!({
            "path": manifest.path.display().to_string(),
            "drift": drift.as_str(),
        });
        if include_ecosystem {
            entry["ecosystem"] = json!(manifest.ecosystem.as_str());
        }
        report.push(entry);
        if drift.needs_sync() {
            stale.push(manifest.clone());
        }
    }
    Ok((report, stale))
}

/// Build a successful `CliOutput` containing pretty-printed JSON.
fn json_stdout(value: serde_json::Value) -> CliOutput {
    CliOutput {
        exit_code: 0,
        stdout: serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        stderr: String::new(),
    }
}

/// Map an `axil-docs` domain error into the host's plugin-error
/// variant. Used by every `handle_cli` / `handle_mcp` arm that needs
/// to propagate a `DocsError` (or any `Display`) through
/// `axil_core::Result`.
fn wrap_io_err<E: std::fmt::Display>(e: E) -> axil_core::AxilError {
    axil_core::AxilError::plugin(e.to_string())
}

/// CLI argument validation failure — semantically an
/// [`axil_core::AxilError::InvalidQuery`], surfaced as such so the
/// Adapter can map to a usage-error exit code.
fn wrap_input_err(msg: &str) -> axil_core::AxilError {
    axil_core::AxilError::InvalidQuery(msg.to_string())
}

/// `true` if `args` contains a bare `--<flag>` token.
fn flag_set(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Extract `--<name> <value>` (or `--<name>=<value>`) from a flat args
/// vector. Returns the first match.
fn named_arg(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    let prefix = format!("{name}=");
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
        if let Some(rest) = arg.strip_prefix(prefix.as_str()) {
            return Some(rest.to_string());
        }
    }
    None
}

// `collect_unique_deps` previously lived here; it is now in
// `crate::pipeline` and re-exported from the crate root. Callers in
// this module use `crate::collect_unique_deps`.

/// Parse `--path <value>` (or `--path=<value>`) from a flat CLI args
/// vector. Returns the first match.
fn parse_path_arg(args: &[String]) -> Option<PathBuf> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--path" {
            return iter.next().map(PathBuf::from);
        }
        if let Some(rest) = arg.strip_prefix("--path=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

/// Walk up from the database path looking for a directory that contains
/// at least one of the project markers `axil-docs` knows about. Returns
/// the project root, or `None` if none found.
///
/// Mirrors `axil-cli`'s `detect_project_root` but lives here so the
/// Extension impl is self-contained.
fn detect_project_root(db_path: &std::path::Path) -> Option<std::path::PathBuf> {
    let start = db_path.parent()?;
    // Markers we treat as project roots (lockfiles preferred, then
    // manifests, then VCS). Lockfiles first so workspace-root wins
    // over crate-member root.
    const MARKERS: &[&str] = &[
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "go.mod",
        "pom.xml",
        ".git",
    ];
    let mut cursor = start;
    loop {
        if MARKERS.iter().any(|m| cursor.join(m).exists()) {
            return Some(cursor.to_path_buf());
        }
        cursor = cursor.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TABLE_DEP_DOCS, TABLE_DEPS};

    #[test]
    fn id_and_display_name() {
        let ext = DocsExtension;
        assert_eq!(ext.id(), "docs");
        assert_eq!(ext.display_name(), "Dependency Docs");
    }

    #[test]
    fn table_prefix_covers_all_owned_tables() {
        let ext = DocsExtension;
        let prefixes = ext.table_prefixes();
        // The single declared prefix must be a prefix of every table
        // this Extension actually owns.
        assert!(prefixes.iter().any(|p| TABLE_DEPS.starts_with(p)));
        assert!(prefixes.iter().any(|p| TABLE_DEP_DOCS.starts_with(p)));
        assert!(prefixes.iter().any(|p| TABLE_DEP_MANIFESTS.starts_with(p)));
    }

    #[test]
    fn boot_block_quiet_when_never_synced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        // Brand-new DB — `_dep_manifests` is empty, boot_block must be
        // silent so `axil boot` doesn't nag projects that don't use
        // dep-docs.
        let ext = DocsExtension;
        assert!(ext.boot_block(&db).is_none());
    }

    #[test]
    fn registers_in_axil_builder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path)
            .with_extension(DocsExtension)
            .build()
            .unwrap();
        assert_eq!(db.extensions().len(), 1);
        assert_eq!(db.extensions()[0].id(), "docs");
        assert_eq!(db.extensions()[0].table_prefixes(), &["_dep"]);
    }

    // ---- — MCP dispatch tests ----

    #[test]
    fn mcp_surface_lists_deps_status() {
        let ext = DocsExtension;
        let surface = ext.mcp_tools().expect("DocsExtension must expose MCP tools");
        assert!(
            surface.tools.iter().any(|t| t.name == "deps_status"),
            "DocsExtension MCP surface should include `deps_status`",
        );
    }

    #[test]
    fn handle_mcp_deps_status_returns_empty_on_fresh_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let call = McpCall {
            tool: "deps_status".into(),
            params: serde_json::Value::Null,
        };
        let result = ext.handle_mcp(&db, &call).unwrap();
        match result {
            Dispatch::Handled(v) => {
                assert_eq!(v["synced_deps"], 0);
                assert!(v["deps"].is_array());
                assert_eq!(v["deps"].as_array().unwrap().len(), 0);
            }
            Dispatch::NotHandled => panic!("deps_status should be handled by DocsExtension"),
        }
    }

    #[test]
    fn handle_mcp_unknown_tool_declines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let call = McpCall {
            tool: "not_a_real_tool".into(),
            params: serde_json::Value::Null,
        };
        let result = ext.handle_mcp(&db, &call).unwrap();
        assert!(matches!(result, Dispatch::NotHandled));
    }

    // ---- — handle_cli tests ----

    #[test]
    fn handle_cli_unknown_path_declines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["something-else".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            ext.handle_cli(&db, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    /// byte-identical-shape verification.
    ///
    /// Asserts that the JSON shape `DocsExtension::handle_cli`
    /// produces for `deps status` matches what `axil-cli`'s
    /// hardcoded `run_deps` `DepsCommand::Status` branch produces:
    /// `{ synced_deps, deps: [sorted by name], manifests }`.
    /// If this test breaks, the dispatch swap is no longer
    /// behavior-identical and the corresponding hardcoded handler
    /// needs to be updated in lockstep.
    #[test]
    fn handle_cli_deps_status_shape_matches_axil_cli() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();

        // Seed the DB with two _deps rows in *unsorted* insertion
        // order so the sort assertion is meaningful.
        db.insert(
            crate::TABLE_DEPS,
            json!({
                "name": "zeta",
                "ecosystem": "cargo",
                "version": "1.0.0",
            }),
        )
        .unwrap();
        db.insert(
            crate::TABLE_DEPS,
            json!({
                "name": "alpha",
                "ecosystem": "cargo",
                "version": "2.0.0",
            }),
        )
        .unwrap();

        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "status".into()],
            args: vec!["--path".into(), dir.path().display().to_string()],
            stdin: None,
        };
        let value: serde_json::Value = match ext.handle_cli(&db, &inv).unwrap() {
            Dispatch::Handled(out) => serde_json::from_str(&out.stdout).unwrap(),
            Dispatch::NotHandled => panic!("expected Handled"),
        };

        // Top-level keys exactly match run_deps Status output.
        let obj = value.as_object().expect("top-level must be object");
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(|s| s.as_str()).collect();
        let expected: std::collections::BTreeSet<&str> =
            ["synced_deps", "deps", "manifests"].iter().copied().collect();
        assert_eq!(keys, expected, "top-level keys must match run_deps shape");

        assert_eq!(value["synced_deps"], 2);
        let deps = value["deps"].as_array().expect("deps must be array");
        assert_eq!(deps.len(), 2);
        // Sorted by name — alpha before zeta.
        assert_eq!(deps[0]["name"], "alpha");
        assert_eq!(deps[1]["name"], "zeta");
        assert!(value["manifests"].is_array());
    }

    #[test]
    fn handle_cli_deps_status_returns_empty_on_fresh_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "status".into()],
            args: vec!["--path".into(), dir.path().display().to_string()],
            stdin: None,
        };
        let result = ext.handle_cli(&db, &inv).unwrap();
        match result {
            Dispatch::Handled(out) => {
                assert_eq!(out.exit_code, 0);
                let value: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
                assert_eq!(value["synced_deps"], 0);
                assert!(value["deps"].is_array());
                assert!(value["manifests"].is_array());
            }
            Dispatch::NotHandled => panic!("deps status should be handled"),
        }
    }

    #[test]
    fn handle_cli_deps_unknown_subcommand_declines() {
        // A truly unknown deps subcommand should decline so the
        // Adapter falls back to its hardcoded path. All 5 real
        // variants (Status / List / Ingest / Sync / Refresh) are
        // now migrated.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "definitely-not-a-real-subcommand".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            ext.handle_cli(&db, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn flag_and_named_arg_helpers() {
        assert!(flag_set(&["--dev".into()], "--dev"));
        assert!(!flag_set(&["--other".into()], "--dev"));
        assert_eq!(
            named_arg(&["--dep".into(), "foo@1".into()], "--dep"),
            Some("foo@1".into())
        );
        assert_eq!(
            named_arg(&["--dep=foo@1".into()], "--dep"),
            Some("foo@1".into())
        );
        assert_eq!(named_arg(&[], "--dep"), None);
    }

    #[test]
    fn handle_cli_deps_list_returns_zero_for_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "list".into()],
            args: vec!["--path".into(), dir.path().display().to_string()],
            stdin: None,
        };
        let result = ext.handle_cli(&db, &inv).unwrap();
        match result {
            Dispatch::Handled(out) => {
                let value: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
                let obj = value.as_object().unwrap();
                let keys: std::collections::BTreeSet<&str> =
                    obj.keys().map(|s| s.as_str()).collect();
                let expected: std::collections::BTreeSet<&str> =
                    ["manifests", "dependencies", "deps"].iter().copied().collect();
                assert_eq!(keys, expected, "top-level keys must match run_deps List");
                assert_eq!(value["manifests"], 0);
                assert_eq!(value["dependencies"], 0);
            }
            Dispatch::NotHandled => panic!("deps list should be handled"),
        }
    }

    #[test]
    fn handle_cli_deps_sync_on_empty_dir() {
        // Empty dir → 0 manifests → ingest_manifests returns
        // {deps_seen:0,ingested:0,...}; sweep_removed returns [].
        // Final shape gets `manifests` and `removed` overlaid.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "sync".into()],
            args: vec!["--path".into(), dir.path().display().to_string()],
            stdin: None,
        };
        let value: serde_json::Value = match ext.handle_cli(&db, &inv).unwrap() {
            Dispatch::Handled(out) => serde_json::from_str(&out.stdout).unwrap(),
            Dispatch::NotHandled => panic!("expected Handled"),
        };
        // Shape sanity: every expected key present, no extras matter.
        for k in [
            "deps_seen",
            "ingested",
            "chunks",
            "needs_web_fallback",
            "not_installed",
            "migrations",
            "doc_diffs",
            "removed",
            "manifests",
        ] {
            assert!(value.get(k).is_some(), "deps sync output missing key `{k}`");
        }
        assert_eq!(value["manifests"], 0);
        assert_eq!(value["ingested"], 0);
    }

    #[test]
    fn handle_cli_deps_refresh_if_stale_fast_exit() {
        // No manifests + --if-stale → fast exit path with
        // {manifests:0, refreshed:0, status:"fresh"}.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "refresh".into()],
            args: vec![
                "--path".into(),
                dir.path().display().to_string(),
                "--if-stale".into(),
            ],
            stdin: None,
        };
        let value: serde_json::Value = match ext.handle_cli(&db, &inv).unwrap() {
            Dispatch::Handled(out) => serde_json::from_str(&out.stdout).unwrap(),
            Dispatch::NotHandled => panic!("expected Handled"),
        };
        assert_eq!(value["manifests"], 0);
        assert_eq!(value["refreshed"], 0);
        assert_eq!(value["status"], "fresh");
    }

    #[test]
    fn handle_cli_deps_ingest_requires_dep_arg() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "ingest".into()],
            args: vec![],
            stdin: None,
        };
        // Missing --dep → error
        assert!(ext.handle_cli(&db, &inv).is_err());
    }

    #[test]
    fn handle_cli_deps_ingest_from_stdin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let inv = CliInvocation {
            command_path: vec!["deps".into(), "ingest".into()],
            args: vec![
                "--dep".into(),
                "smoke@1.0.0".into(),
                "--ecosystem".into(),
                "cargo".into(),
            ],
            stdin: Some("# Smoke\n\nMinimal doc text for the smoke ingest test.\n".into()),
        };
        let result = ext.handle_cli(&db, &inv).unwrap();
        match result {
            Dispatch::Handled(out) => {
                let value: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
                assert_eq!(value["dep"], "smoke");
                assert_eq!(value["version"], "1.0.0");
                assert_eq!(value["ecosystem"], "cargo");
                assert_eq!(value["source"], "agent");
                assert!(
                    value["chunks"].as_u64().is_some(),
                    "chunks must be a non-negative integer"
                );
            }
            Dispatch::NotHandled => panic!("deps ingest should be handled"),
        }
    }

    #[test]
    fn handle_mcp_dep_docs_requires_query() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let call = McpCall {
            tool: "dep_docs".into(),
            params: serde_json::json!({}),
        };
        // Missing `query` → error
        assert!(ext.handle_mcp(&db, &call).is_err());
    }

    #[test]
    fn handle_mcp_dep_docs_empty_returns_handled_with_empty_array() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        let ext = DocsExtension;
        let call = McpCall {
            tool: "dep_docs".into(),
            params: serde_json::json!({"query": "anything", "top_k": 3}),
        };
        match ext.handle_mcp(&db, &call).unwrap() {
            Dispatch::Handled(v) => {
                assert!(v.is_array(), "dep_docs returns a JSON array of hits");
                assert_eq!(v.as_array().unwrap().len(), 0);
            }
            Dispatch::NotHandled => panic!("dep_docs should be handled"),
        }
    }

    #[test]
    fn mcp_surface_includes_dep_docs_tool() {
        let ext = DocsExtension;
        let surface = ext.mcp_tools().expect("DocsExtension must expose MCP tools");
        let names: std::collections::BTreeSet<&str> =
            surface.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains("deps_status"));
        assert!(names.contains("dep_docs"));
    }

    #[test]
    fn parse_path_arg_handles_both_forms() {
        assert_eq!(
            parse_path_arg(&["--path".into(), "/x".into()]),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            parse_path_arg(&["--path=/y".into()]),
            Some(PathBuf::from("/y"))
        );
        assert_eq!(parse_path_arg(&[]), None);
        assert_eq!(parse_path_arg(&["--other".into(), "/z".into()]), None);
    }

    #[test]
    fn end_to_end_dispatch_via_compose_helpers() {
        use axil_core::{compose_mcp_surface, dispatch_mcp};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path)
            .with_extension(DocsExtension)
            .build()
            .unwrap();

        // (1) Surface composition surfaces the deps_status tool.
        let surfaces = compose_mcp_surface(&db.extensions());
        assert!(
            surfaces
                .iter()
                .flat_map(|s| s.tools.iter())
                .any(|t| t.name == "deps_status"),
            "compose_mcp_surface must include DocsExtension's deps_status tool",
        );

        // (2) dispatch_mcp routes the call to DocsExtension::handle_mcp.
        let call = McpCall {
            tool: "deps_status".into(),
            params: serde_json::Value::Null,
        };
        let result = dispatch_mcp(&db, &db.extensions(), &call).unwrap();
        let value = match result {
            Dispatch::Handled(v) => v,
            Dispatch::NotHandled => panic!("dispatch_mcp must route to DocsExtension"),
        };
        assert_eq!(value["synced_deps"], 0);

        // (3) Unknown tool falls through.
        let call = McpCall {
            tool: "ghost".into(),
            params: serde_json::Value::Null,
        };
        let result = dispatch_mcp(&db, &db.extensions(), &call).unwrap();
        assert!(matches!(result, Dispatch::NotHandled));
    }
}
