//! `Extension` trait, the contract for Tier-2 extensibility.
//!
//! An Extension is a capability built on top of one or more Engines
//! (Tier 1). It owns prefixed tables in the core `.axil` file (not a
//! companion file), and optionally registers a CLI subcommand surface,
//! MCP tool surface, brain hooks, and drift-refresh logic.
//!
//! See `docs/src/extending/extensions.md` for the full authoring guide
//! and `axil-docs` for the reference implementation.
//!
//! # Stability
//!
//! `Extension` and its support types ([`CliSurface`], [`CliSubcommand`],
//! [`CliArg`], [`CliInvocation`], [`CliOutput`], [`McpSurface`], [`McpTool`],
//! [`McpCall`], [`Dispatch`], [`Hit`], [`RefreshOpts`], [`RefreshReport`]) are
//! part of Axil's **1.0 stable SPI** — third-party Extensions compile against
//! them with a semver compatibility promise. The structs third parties
//! construct or receive are `#[non_exhaustive]` and exposed through
//! constructors/builders so new fields can be added without a breaking change.
//! Contrast with the `crate::plugin` Engine traits, which are explicitly
//! *unstable* (upstream-or-fork).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::db::Axil;
use crate::error::Result;

/// Stable interface for Tier-2 Extensions in Axil's extensibility model.
///
/// Every method except [`Extension::id`] has a default implementation.
/// Implement only what your Extension needs.
///
/// # WASM-bindable contract
///
/// This trait is the mirror for the WASM plugin ABI (the `axil:plugin` WIT
/// world). To keep a `WasmExtension` shim able to back it without a per-call
/// boundary crossing, the trait splits into two method classes:
///
/// - **Metadata** — [`id`](Extension::id), [`display_name`](Extension::display_name),
///   [`table_prefixes`](Extension::table_prefixes), [`cli_commands`](Extension::cli_commands),
///   [`mcp_tools`](Extension::mcp_tools). These are called on hot paths (e.g.
///   [`crate::dispatch_cli`] iterates `cli_commands()` for every Extension on
///   every CLI call) and MUST be cheap and statically derivable. A WASM host
///   fetches them **once at load** and caches the owned result, then serves the
///   borrow-returning methods from that cache. They must not perform expensive,
///   blocking, or fallible work at call time.
/// - **Handlers / contributions** — [`handle_cli`](Extension::handle_cli),
///   [`handle_mcp`](Extension::handle_mcp), [`refresh`](Extension::refresh),
///   [`recall_for_file`](Extension::recall_for_file), and the live render of
///   [`boot_block`](Extension::boot_block). These are the only methods a WASM
///   shim invokes across the sandbox boundary on demand, and are `Result`- or
///   `Option`-returning so a guest failure has somewhere to go.
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

    /// Handle a CLI dispatch matched against this Extension's
    /// [`Extension::cli_commands`] surface.
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

    /// Handle an MCP tool call matched against this Extension's
    /// [`Extension::mcp_tools`] surface.
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
/// Adapter's fallback. Path C uses this for backwards-compatible
/// Extension dispatch: existing Adapters keep their hard-coded paths
/// and only run them when *no* registered Extension claims the call.
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

impl CliInvocation {
    /// Construct a CLI invocation for dispatch to an Extension.
    pub fn new(
        command_path: Vec<String>,
        args: Vec<String>,
        stdin: Option<String>,
    ) -> Self {
        Self {
            command_path,
            args,
            stdin,
        }
    }
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

impl CliOutput {
    /// A successful (exit code 0) output carrying `stdout`.
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self {
            exit_code: 0,
            stdout: stdout.into(),
            stderr: String::new(),
        }
    }

    /// A failed output: `exit_code` (non-zero by convention) + `stderr`.
    pub fn err(exit_code: i32, stderr: impl Into<String>) -> Self {
        Self {
            exit_code,
            stdout: String::new(),
            stderr: stderr.into(),
        }
    }
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

impl McpCall {
    /// Construct an MCP call for dispatch to an Extension.
    pub fn new(tool: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            tool: tool.into(),
            params,
        }
    }
}

/// Adapter-agnostic description of an Extension's CLI subcommand surface.
///
/// A CLI Adapter (e.g. `axil-cli`) translates this into its native
/// argument-parsing structure (`clap::Command` for `axil-cli`). The
/// `Extension` trait stays free of `clap` so any Adapter can compose
/// its surface.
///
/// `#[non_exhaustive]`: construct via [`CliSurface::new`] + the
/// [`subcommand`](CliSurface::subcommand) builder so new fields can be
/// added without breaking third-party Extensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CliSurface {
    /// Top-level subcommand name (e.g. `"deps"` for `axil deps …`).
    pub command: String,
    /// One-line description for `--help`.
    pub about: String,
    /// Nested subcommands (e.g. `deps list`, `deps sync`).
    pub subcommands: Vec<CliSubcommand>,
}

impl CliSurface {
    /// A CLI surface for top-level `command` with no subcommands yet.
    pub fn new(command: impl Into<String>, about: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            about: about.into(),
            subcommands: Vec::new(),
        }
    }

    /// Append a subcommand (builder style).
    pub fn subcommand(mut self, sub: CliSubcommand) -> Self {
        self.subcommands.push(sub);
        self
    }

    /// Append many subcommands (builder style).
    pub fn subcommands(mut self, subs: impl IntoIterator<Item = CliSubcommand>) -> Self {
        self.subcommands.extend(subs);
        self
    }
}

/// A single nested subcommand under a [`CliSurface`].
///
/// `#[non_exhaustive]`: construct via [`CliSubcommand::new`] + the
/// [`arg`](CliSubcommand::arg) builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CliSubcommand {
    /// Subcommand name (e.g. `"list"`).
    pub name: String,
    /// One-line description.
    pub about: String,
    /// Positional and named arguments.
    pub args: Vec<CliArg>,
}

impl CliSubcommand {
    /// A subcommand named `name` with no arguments yet.
    pub fn new(name: impl Into<String>, about: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            about: about.into(),
            args: Vec::new(),
        }
    }

    /// Append an argument (builder style).
    pub fn arg(mut self, arg: CliArg) -> Self {
        self.args.push(arg);
        self
    }

    /// Append many arguments (builder style).
    pub fn args(mut self, args: impl IntoIterator<Item = CliArg>) -> Self {
        self.args.extend(args);
        self
    }
}

/// Description of a single CLI argument.
///
/// `#[non_exhaustive]`: construct via [`CliArg::new`] +
/// [`required`](CliArg::required) / [`takes_value`](CliArg::takes_value).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

impl CliArg {
    /// An optional, valueless argument named `name`. Use the builders to
    /// mark it [`required`](CliArg::required) or value-taking
    /// ([`takes_value`](CliArg::takes_value)).
    pub fn new(name: impl Into<String>, about: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            about: about.into(),
            required: false,
            takes_value: false,
        }
    }

    /// Set whether the argument is required (builder style).
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set whether the argument takes a value (builder style).
    pub fn takes_value(mut self, takes_value: bool) -> Self {
        self.takes_value = takes_value;
        self
    }
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

impl McpSurface {
    /// An MCP surface exposing `tools`.
    pub fn new(tools: Vec<McpTool>) -> Self {
        Self { tools }
    }
}

/// A single tool exposed over MCP.
///
/// `#[non_exhaustive]`: construct via [`McpTool::new`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct McpTool {
    /// Tool name (e.g. `"dep_docs"`).
    pub name: String,
    /// Description shown to MCP clients.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

impl McpTool {
    /// An MCP tool named `name` with the given description + input schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Options passed to [`Extension::refresh`].
///
/// `#[non_exhaustive]`: construct via [`RefreshOpts::new`] or
/// [`RefreshOpts::default`] + the builders.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RefreshOpts {
    /// Only re-ingest items the Extension's drift detection flags as
    /// stale. When `false`, force a full refresh.
    pub if_stale: bool,
    /// Limit refresh to a specific working directory (for
    /// multi-project workspaces).
    pub path: Option<PathBuf>,
}

impl RefreshOpts {
    /// Default options (full refresh, no path scope).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to refresh only stale items (builder style).
    pub fn if_stale(mut self, if_stale: bool) -> Self {
        self.if_stale = if_stale;
        self
    }

    /// Scope the refresh to a working directory (builder style).
    pub fn path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// Summary report returned by [`Extension::refresh`].
///
/// `#[non_exhaustive]`: build via [`RefreshReport::default`] and the
/// builder methods, or mutate the public fields after `default()`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
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

impl RefreshReport {
    /// Set the inspected/stale/refreshed counters (builder style).
    pub fn with_counts(mut self, inspected: usize, stale: usize, refreshed: usize) -> Self {
        self.inspected = inspected;
        self.stale = stale;
        self.refreshed = refreshed;
        self
    }

    /// Append a status detail line (builder style).
    pub fn detail(mut self, line: impl Into<String>) -> Self {
        self.details.push(line.into());
        self
    }
}

/// A hit returned by [`Extension::recall_for_file`].
///
/// Intentionally minimal — the Adapter decides how to render it.
/// `#[non_exhaustive]`: construct via [`Hit::new`] +
/// [`with_summary`](Hit::with_summary).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

impl Hit {
    /// A hit pointing at record `id` in `table` with relevance `score`.
    pub fn new(table: impl Into<String>, id: impl Into<String>, score: f32) -> Self {
        Self {
            table: table.into(),
            id: id.into(),
            summary: None,
            score,
        }
    }

    /// Attach a one-line display summary (builder style).
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }
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

    #[test]
    fn surface_builders_compose() {
        // The #[non_exhaustive] constructors must reproduce the same
        // shape an external Extension would build by hand.
        let surface = CliSurface::new("deps", "dependency docs").subcommand(
            CliSubcommand::new("sync", "sync deps").arg(
                CliArg::new("path", "project path")
                    .required(false)
                    .takes_value(true),
            ),
        );
        assert_eq!(surface.command, "deps");
        assert_eq!(surface.subcommands.len(), 1);
        assert_eq!(surface.subcommands[0].name, "sync");
        assert_eq!(surface.subcommands[0].args.len(), 1);
        assert!(surface.subcommands[0].args[0].takes_value);
        assert!(!surface.subcommands[0].args[0].required);

        let mcp = McpSurface::new(vec![McpTool::new(
            "dep_docs",
            "search dep docs",
            serde_json::json!({"type": "object"}),
        )]);
        assert_eq!(mcp.tools.len(), 1);
        assert_eq!(mcp.tools[0].name, "dep_docs");

        let hit = Hit::new("_dep_docs", "01ABC", 0.9).with_summary("serde derive");
        assert_eq!(hit.table, "_dep_docs");
        assert_eq!(hit.summary.as_deref(), Some("serde derive"));

        let opts = RefreshOpts::new().if_stale(true);
        assert!(opts.if_stale);

        let report = RefreshReport::default()
            .with_counts(3, 1, 1)
            .detail("refreshed serde");
        assert_eq!(report.inspected, 3);
        assert_eq!(report.details.len(), 1);
    }

    // ---- dispatch contract ----

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
        let inv = CliInvocation::new(vec!["x".into()], vec![], None);
        let call = McpCall::new("x", serde_json::Value::Null);
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
