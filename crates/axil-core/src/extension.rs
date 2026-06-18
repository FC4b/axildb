//! Phase 17 — `Extension` trait, the contract for Tier-2 extensibility.
//!
//! An Extension is a capability built on top of one or more Engines
//! (Tier 1). It owns prefixed tables in the core `.axil` file (not a
//! companion file), and optionally registers a CLI subcommand surface,
//! MCP tool surface, brain hooks, and drift-refresh logic.
//!
//! See `docs/src/extending/extensions.md` for the full authoring guide
//! and `axil-docs` for the reference implementation.
//!
//! This trait is contract-only in Phase 17 P1.1. Builder registration
//! and `Adapter` discovery wiring land in P1.2 / P1.3 — existing
//! Extensions (`axil-docs`, `axil-scip`, …) continue to work without
//! implementing this trait until they migrate in P3.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::db::Axil;
use crate::error::Result;

/// Stable interface for Tier-2 Extensions in Axil's extensibility model.
///
/// Every method except [`Extension::id`] has a default implementation.
/// Implement only what your Extension needs.
pub trait Extension: Send + Sync {
    /// Stable extension id, kebab-case, must match the crate name minus
    /// the `axil-` prefix (e.g. `axil-docs` → `"docs"`).
    fn id(&self) -> &str;

    /// Human-readable display name (used in `axil status`, `axil boot`).
    /// Defaults to [`Extension::id`].
    fn display_name(&self) -> &str {
        self.id()
    }

    /// Table-name prefixes this extension owns. The master coordinator
    /// will reject builder registration if two extensions claim
    /// overlapping prefixes. Convention: all tables start with
    /// `_<id>_*`.
    fn table_prefixes(&self) -> &[&str] {
        &[]
    }

    /// Optional CLI subcommand surface. Returned to a CLI Adapter at
    /// registration time. `None` = no CLI surface.
    fn cli_commands(&self) -> Option<CliSurface> {
        None
    }

    /// Optional MCP tool surface. Returned to an MCP Adapter at
    /// registration time. `None` = no MCP surface.
    fn mcp_tools(&self) -> Option<McpSurface> {
        None
    }

    /// Optional `axil boot` block. Called once per boot. Returning
    /// `Some(s)` appends `s` to the boot summary; `None` adds nothing.
    fn boot_block(&self, _db: &Axil) -> Option<String> {
        None
    }

    /// Optional drift/refresh entry point. Called by `axil refresh`,
    /// PostToolUse brain hooks, or any agent-initiated refresh.
    fn refresh(&self, _db: &Axil, _opts: RefreshOpts) -> Result<RefreshReport> {
        Ok(RefreshReport::default())
    }

    /// Optional `recall-for-file` contribution. Called when the agent
    /// is about to edit a file; the extension may surface relevant
    /// records (e.g. dep docs for the file's imports).
    fn recall_for_file(&self, _db: &Axil, _path: &Path) -> Result<Vec<Hit>> {
        Ok(vec![])
    }

    /// Phase 17 P1.3 — handle a CLI dispatch matched against this
    /// Extension's [`Extension::cli_commands`] surface.
    ///
    /// **Contract (Path C — hybrid):** the default impl returns
    /// `Ok(Dispatch::NotHandled)`, signalling that the calling Adapter
    /// should fall back to its hard-coded dispatch path. Extensions
    /// that own a CLI surface should override to actually run the
    /// matched subcommand. Returning `Dispatch::Handled(output)`
    /// short-circuits the Adapter's fallback.
    ///
    /// The Adapter is responsible for matching the user's argv against
    /// `cli_commands()` to route the call here; this method only needs
    /// to execute the matched subcommand.
    fn handle_cli(
        &self,
        _db: &Axil,
        _invocation: &CliInvocation,
    ) -> Result<Dispatch<CliOutput>> {
        Ok(Dispatch::NotHandled)
    }

    /// Phase 17 P1.3 — handle an MCP tool call matched against this
    /// Extension's [`Extension::mcp_tools`] surface.
    ///
    /// Same Path C contract as [`Extension::handle_cli`]: default impl
    /// returns `Ok(Dispatch::NotHandled)`, letting the Adapter fall
    /// back to its hard-coded path.
    fn handle_mcp(
        &self,
        _db: &Axil,
        _call: &McpCall,
    ) -> Result<Dispatch<serde_json::Value>> {
        Ok(Dispatch::NotHandled)
    }
}

/// Whether an Extension actually handled a dispatch, or punted to the
/// Adapter's fallback. Phase 17 Path C uses this for backwards-
/// compatible Extension dispatch: existing Adapters keep their
/// hard-coded paths and only run them when *no* registered Extension
/// claims the call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dispatch<T> {
    /// Extension handled this call — Adapter should surface `T`.
    Handled(T),
    /// Extension declined — Adapter should try its next dispatcher
    /// (typically a hard-coded fallback).
    NotHandled,
}

impl<T> Dispatch<T> {
    /// `true` if the Extension handled the dispatch.
    pub fn is_handled(&self) -> bool {
        matches!(self, Dispatch::Handled(_))
    }

    /// Convert into `Option<T>`, discarding the `NotHandled` signal.
    pub fn handled(self) -> Option<T> {
        match self {
            Dispatch::Handled(v) => Some(v),
            Dispatch::NotHandled => None,
        }
    }
}

/// What a CLI Adapter passes to [`Extension::handle_cli`].
///
/// The Adapter has already matched argv against the Extension's
/// [`CliSurface`]; `command_path` records what matched, and `args`
/// holds whatever came after the matched path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliInvocation {
    /// The matched subcommand path. For `axil deps sync --path foo`
    /// dispatched to the `docs` Extension, this is `["deps", "sync"]`.
    pub command_path: Vec<String>,
    /// Raw arguments after the matched subcommand path (positional +
    /// flags, in argv order).
    pub args: Vec<String>,
    /// Stdin contents, if the Adapter captured them. `None` when no
    /// stdin was piped or the Adapter doesn't capture stdin.
    pub stdin: Option<String>,
}

/// What an Extension returns from a successful CLI dispatch.
///
/// The Adapter translates this into its protocol's native output
/// (writes stdout/stderr and propagates exit code for `axil-cli`,
/// serializes to JSON for an HTTP Adapter, etc.).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliOutput {
    /// Exit code. `0` for success. Convention matches `axil-cli`:
    /// `1` for runtime error, `2` for usage error.
    pub exit_code: i32,
    /// Stdout text. Should not include a trailing newline; Adapters
    /// add one as appropriate.
    pub stdout: String,
    /// Stderr text. Same trailing-newline convention as `stdout`.
    pub stderr: String,
}

/// What an MCP Adapter passes to [`Extension::handle_mcp`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCall {
    /// Tool name from the MCP `tools/call` request (e.g.
    /// `"dep_docs"`).
    pub tool: String,
    /// Parameters from the MCP `tools/call` request — already JSON-
    /// schema-validated by the Adapter against the Extension's
    /// [`McpTool::input_schema`].
    pub params: serde_json::Value,
}

/// Adapter-agnostic description of an Extension's CLI subcommand surface.
///
/// A CLI Adapter (e.g. `axil-cli`) translates this into its native
/// argument-parsing structure (`clap::Command` for `axil-cli`). The
/// `Extension` trait stays free of `clap` so any Adapter can compose
/// its surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSurface {
    /// Top-level subcommand name (e.g. `"deps"` for `axil deps …`).
    pub command: String,
    /// One-line description for `--help`.
    pub about: String,
    /// Nested subcommands (e.g. `deps list`, `deps sync`).
    pub subcommands: Vec<CliSubcommand>,
}

/// A single nested subcommand under a [`CliSurface`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSubcommand {
    /// Subcommand name (e.g. `"list"`).
    pub name: String,
    /// One-line description.
    pub about: String,
    /// Positional and named arguments.
    pub args: Vec<CliArg>,
}

/// Description of a single CLI argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliArg {
    /// Argument name (positional name or long flag without the `--`).
    pub name: String,
    /// One-line description for `--help`.
    pub about: String,
    /// Whether the argument is required.
    pub required: bool,
    /// Whether the argument takes a value (flags do not; named args do).
    pub takes_value: bool,
}

/// Adapter-agnostic description of an Extension's MCP tool surface.
///
/// The MCP Adapter (`axil-mcp`) translates each tool into an MCP
/// `tools/list` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSurface {
    /// Tools the Extension exposes over MCP.
    pub tools: Vec<McpTool>,
}

/// A single tool exposed over MCP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Tool name (e.g. `"dep_docs"`).
    pub name: String,
    /// Description shown to MCP clients.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// Options passed to [`Extension::refresh`].
#[derive(Debug, Clone, Default)]
pub struct RefreshOpts {
    /// Only re-ingest items the Extension's drift detection flags as
    /// stale. When `false`, force a full refresh.
    pub if_stale: bool,
    /// Limit refresh to a specific working directory (for
    /// multi-project workspaces).
    pub path: Option<PathBuf>,
}

/// Summary report returned by [`Extension::refresh`].
#[derive(Debug, Clone, Default)]
pub struct RefreshReport {
    /// Items inspected during the refresh pass.
    pub inspected: usize,
    /// Items found stale (would be refreshed if `if_stale` is `true`,
    /// always refreshed otherwise).
    pub stale: usize,
    /// Items successfully re-ingested in this call.
    pub refreshed: usize,
    /// Human-readable per-item status lines, surfaced by the Adapter.
    pub details: Vec<String>,
}

/// A hit returned by [`Extension::recall_for_file`].
///
/// Intentionally minimal — the Adapter decides how to render it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// Table the hit lives in (e.g. `"_dep_docs"`).
    pub table: String,
    /// Record id within the table.
    pub id: String,
    /// Optional one-line summary for display.
    pub summary: Option<String>,
    /// Extension-defined relevance score, higher = more relevant.
    /// Adapters may merge or rerank across Extensions.
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal Extension impl to verify the trait surface compiles
    /// with only `id()` overridden.
    struct Minimal;

    impl Extension for Minimal {
        fn id(&self) -> &str {
            "minimal"
        }
    }

    #[test]
    fn minimal_extension_defaults() {
        let ext = Minimal;
        assert_eq!(ext.id(), "minimal");
        assert_eq!(ext.display_name(), "minimal");
        assert!(ext.table_prefixes().is_empty());
        assert!(ext.cli_commands().is_none());
        assert!(ext.mcp_tools().is_none());
    }

    #[test]
    fn refresh_report_default() {
        let r = RefreshReport::default();
        assert_eq!(r.inspected, 0);
        assert_eq!(r.stale, 0);
        assert_eq!(r.refreshed, 0);
        assert!(r.details.is_empty());
    }

    // ---- Phase 17 P1.3 — dispatch contract ----

    #[test]
    fn dispatch_helpers() {
        let h: Dispatch<i32> = Dispatch::Handled(42);
        let n: Dispatch<i32> = Dispatch::NotHandled;
        assert!(h.is_handled());
        assert!(!n.is_handled());
        assert_eq!(h.handled(), Some(42));
        assert_eq!(n.handled(), None);
    }

    #[test]
    fn dispatch_equality() {
        assert_eq!(Dispatch::<i32>::NotHandled, Dispatch::<i32>::NotHandled);
        assert_eq!(Dispatch::Handled(1), Dispatch::Handled(1));
        assert_ne!(Dispatch::Handled(1), Dispatch::Handled(2));
        assert_ne!(Dispatch::Handled(1), Dispatch::NotHandled);
    }

    #[test]
    fn default_handle_methods_decline() {
        // A minimal Extension that overrides nothing should decline
        // both dispatch surfaces — the Adapter then falls back to its
        // hard-coded path. This is the load-bearing behavior of
        // Path C, so it gets a dedicated test.
        let ext = Minimal;
        // Can't construct an Axil without a tempdir, but we can build
        // the invocation/call types and call them on a stub.
        let inv = CliInvocation {
            command_path: vec!["x".into()],
            args: vec![],
            stdin: None,
        };
        let call = McpCall {
            tool: "x".into(),
            params: serde_json::Value::Null,
        };
        // We need a real Axil to call handle_cli/handle_mcp because
        // they take `&Axil`. Build one in a temp dir.
        let dir = tempfile::tempdir().unwrap();
        let db = crate::Axil::open(dir.path().join("test.axil"))
            .build()
            .unwrap();
        match ext.handle_cli(&db, &inv).unwrap() {
            Dispatch::NotHandled => (),
            Dispatch::Handled(_) => panic!("default impl should decline"),
        }
        match ext.handle_mcp(&db, &call).unwrap() {
            Dispatch::NotHandled => (),
            Dispatch::Handled(_) => panic!("default impl should decline"),
        }
    }
}
