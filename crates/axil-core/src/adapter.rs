//! Phase 17 — `Adapter` trait, the contract for Tier-3 extensibility.
//!
//! An Adapter is the interface between Axil and an external protocol
//! (CLI argv, MCP stdio JSON-RPC, HTTP, GraphQL, AxilQL, etc.). It
//! translates protocol calls to `Axil` API calls, maps errors back to
//! the protocol's native error type, and surfaces every registered
//! [`Extension`]'s commands/tools through its protocol.
//!
//! Adapters are stateless w.r.t. storage: they read and write through
//! `Axil`, never to their own files.
//!
//! See `docs/src/extending/adapters.md` for the full authoring guide.
//!
//! Every in-tree adapter implements this trait: `axil_cli::CliAdapter`
//! (argv), `axil_mcp::McpAdapter` (stdio JSON-RPC), and
//! `axil_ql::QlAdapter` (the AxilQL query frontend); the
//! `examples/http_adapter.rs` `HttpAdapter` is the out-of-tree template.
//! Helper functions for composing Extension surfaces
//! (`compose_cli_surface`, `compose_mcp_surface`) let an Adapter discover
//! every registered Extension's commands/tools without reaching into
//! internals.
//!
//! [`Extension`]: crate::extension::Extension

use std::sync::Arc;

use crate::db::Axil;
use crate::error::Result;
use crate::extension::{
    CliInvocation, CliOutput, CliSurface, Dispatch, Extension, McpCall, McpSurface,
};

/// Stable interface for Tier-3 Adapters in Axil's extensibility model.
pub trait Adapter {
    /// Stable adapter id (e.g. `"cli"`, `"mcp"`, `"http"`, `"axilql"`).
    fn id(&self) -> &str;

    /// The protocol this adapter serves.
    fn protocol(&self) -> Protocol;

    /// Called once after the `Axil` host is built. The adapter takes a
    /// shared handle to the host so multiple Adapters can run against
    /// the same database (e.g. CLI + HTTP in the same process).
    ///
    /// The adapter is expected to inspect registered Extensions via
    /// `db.extensions()` (once that lands in P1.2) and prepare its
    /// dispatch table here.
    fn bind(&mut self, db: Arc<Axil>) -> Result<()>;

    /// Run the adapter (blocking). For a CLI Adapter this parses argv
    /// and dispatches; for an MCP Adapter this runs the stdio JSON-RPC
    /// loop; for an HTTP Adapter this binds a socket and serves
    /// requests.
    fn run(self) -> Result<()>
    where
        Self: Sized;
}

/// Phase 17 P2.3 — compose every registered Extension's CLI surface
/// into a flat list.
///
/// An Adapter (typically `axil-cli`) calls this once after `bind()`
/// to learn which top-level subcommands the registered Extensions
/// claim. The Adapter is then responsible for routing matching argv
/// invocations to [`dispatch_cli`].
///
/// The returned `Vec` preserves registration order so deterministic
/// `--help` rendering is straightforward.
pub fn compose_cli_surface(extensions: &[Arc<dyn Extension>]) -> Vec<CliSurface> {
    extensions
        .iter()
        .filter_map(|e| e.cli_commands())
        .collect()
}

/// Phase 17 P2.3 — compose every registered Extension's MCP tool
/// surface into a flat list.
///
/// An Adapter (typically `axil-mcp`) calls this once after `bind()`
/// to learn which MCP tools the registered Extensions claim. The
/// Adapter then surfaces them in its `tools/list` response and routes
/// matching `tools/call` requests to [`dispatch_mcp`].
pub fn compose_mcp_surface(extensions: &[Arc<dyn Extension>]) -> Vec<McpSurface> {
    extensions
        .iter()
        .filter_map(|e| e.mcp_tools())
        .collect()
}

/// Phase 17 P2.3 — route a CLI invocation to whichever registered
/// Extension owns the matched top-level subcommand.
///
/// **Path C semantics:** the first Extension whose
/// [`CliSurface::command`] matches `invocation.command_path[0]` is
/// asked to [`Extension::handle_cli`]. If that returns
/// `Dispatch::Handled(output)`, the output is propagated up. If it
/// returns `Dispatch::NotHandled` (or the Extension has no matching
/// command), this function returns `Dispatch::NotHandled` so the
/// Adapter can fall through to its hard-coded path.
///
/// Returns `Err` only when an Extension's handler errors — a missing
/// match is `Ok(Dispatch::NotHandled)`, not an error.
pub fn dispatch_cli(
    db: &Axil,
    extensions: &[Arc<dyn Extension>],
    invocation: &CliInvocation,
) -> Result<Dispatch<CliOutput>> {
    let Some(top) = invocation.command_path.first() else {
        return Ok(Dispatch::NotHandled);
    };
    for ext in extensions {
        if let Some(surface) = ext.cli_commands() {
            if &surface.command == top {
                let result = ext.handle_cli(db, invocation)?;
                if result.is_handled() {
                    return Ok(result);
                }
                // Extension matched the top-level command but declined
                // — keep walking; another Extension may claim the same
                // top-level command for a different subcommand. This is
                // unusual but the builder doesn't enforce uniqueness on
                // top-level CLI names (only on table prefixes).
            }
        }
    }
    Ok(Dispatch::NotHandled)
}

/// Phase 17 P2.3 — route an MCP tool call to whichever registered
/// Extension owns the named tool. Same Path C semantics as
/// [`dispatch_cli`].
pub fn dispatch_mcp(
    db: &Axil,
    extensions: &[Arc<dyn Extension>],
    call: &McpCall,
) -> Result<Dispatch<serde_json::Value>> {
    for ext in extensions {
        if let Some(surface) = ext.mcp_tools() {
            if surface.tools.iter().any(|t| t.name == call.tool) {
                let result = ext.handle_mcp(db, call)?;
                if result.is_handled() {
                    return Ok(result);
                }
            }
        }
    }
    Ok(Dispatch::NotHandled)
}

/// Transport / protocol an Adapter serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Protocol {
    /// Command-line interface (argv → dispatch). `axil-cli`.
    Cli,
    /// Model Context Protocol over stdio JSON-RPC. `axil-mcp`.
    Mcp,
    /// HTTP REST/JSON.
    Http,
    /// GraphQL over HTTP.
    GraphQl,
    /// A query-language frontend (e.g. AxilQL).
    QueryLang,
    /// Anything else — the &'static str is the protocol's stable id.
    Custom(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test that a minimal Adapter impl compiles. We can't
    /// instantiate `Arc<Axil>` without a real database, so this only
    /// exercises the trait surface.
    struct DummyAdapter;

    impl Adapter for DummyAdapter {
        fn id(&self) -> &str {
            "dummy"
        }
        fn protocol(&self) -> Protocol {
            Protocol::Custom("dummy")
        }
        fn bind(&mut self, _db: Arc<Axil>) -> Result<()> {
            Ok(())
        }
        fn run(self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn adapter_id_and_protocol() {
        let a = DummyAdapter;
        assert_eq!(a.id(), "dummy");
        assert_eq!(a.protocol(), Protocol::Custom("dummy"));
    }

    #[test]
    fn protocol_equality() {
        assert_eq!(Protocol::Cli, Protocol::Cli);
        assert_ne!(Protocol::Cli, Protocol::Mcp);
        assert_eq!(Protocol::Custom("x"), Protocol::Custom("x"));
        assert_ne!(Protocol::Custom("x"), Protocol::Custom("y"));
    }

    // ---- Phase 17 P2.3 — compose + dispatch helpers ----

    use crate::extension::{
        CliInvocation, CliOutput, CliSurface, McpCall, McpSurface, McpTool,
    };

    /// Extension that claims `["echo"]` on CLI and `"ping"` on MCP,
    /// and actually handles them.
    struct EchoExt;
    impl Extension for EchoExt {
        fn id(&self) -> &str {
            "echo"
        }
        fn cli_commands(&self) -> Option<CliSurface> {
            Some(CliSurface {
                command: "echo".into(),
                about: "echo args".into(),
                subcommands: vec![],
            })
        }
        fn mcp_tools(&self) -> Option<McpSurface> {
            Some(McpSurface {
                tools: vec![McpTool {
                    name: "ping".into(),
                    description: "ping".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }],
            })
        }
        fn handle_cli(
            &self,
            _db: &Axil,
            inv: &CliInvocation,
        ) -> Result<Dispatch<CliOutput>> {
            Ok(Dispatch::Handled(CliOutput {
                exit_code: 0,
                stdout: inv.args.join(" "),
                stderr: String::new(),
            }))
        }
        fn handle_mcp(
            &self,
            _db: &Axil,
            call: &McpCall,
        ) -> Result<Dispatch<serde_json::Value>> {
            Ok(Dispatch::Handled(serde_json::json!({
                "tool": call.tool,
                "params": call.params,
            })))
        }
    }

    /// Extension that claims a CLI surface but always declines —
    /// proves the dispatch loop keeps walking past `NotHandled`.
    struct DecliningExt;
    impl Extension for DecliningExt {
        fn id(&self) -> &str {
            "declining"
        }
        fn cli_commands(&self) -> Option<CliSurface> {
            Some(CliSurface {
                command: "echo".into(),
                about: "claims echo but declines".into(),
                subcommands: vec![],
            })
        }
    }

    fn temp_axil() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::Axil::open(dir.path().join("test.axil"))
            .build()
            .unwrap();
        (db, dir)
    }

    #[test]
    fn compose_cli_surface_collects_in_order() {
        let exts: Vec<Arc<dyn Extension>> =
            vec![Arc::new(EchoExt), Arc::new(DecliningExt)];
        let surfaces = compose_cli_surface(&exts);
        assert_eq!(surfaces.len(), 2);
        assert_eq!(surfaces[0].command, "echo");
        assert_eq!(surfaces[1].command, "echo");
    }

    #[test]
    fn compose_cli_skips_extensions_with_no_surface() {
        struct NoSurface;
        impl Extension for NoSurface {
            fn id(&self) -> &str {
                "ns"
            }
        }
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(NoSurface), Arc::new(EchoExt)];
        let surfaces = compose_cli_surface(&exts);
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].command, "echo");
    }

    #[test]
    fn compose_mcp_surface_collects() {
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(EchoExt)];
        let surfaces = compose_mcp_surface(&exts);
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].tools[0].name, "ping");
    }

    #[test]
    fn dispatch_cli_routes_to_matching_extension() {
        let (db, _dir) = temp_axil();
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(EchoExt)];
        let inv = CliInvocation {
            command_path: vec!["echo".into()],
            args: vec!["hello".into(), "world".into()],
            stdin: None,
        };
        let result = dispatch_cli(&db, &exts, &inv).unwrap();
        match result {
            Dispatch::Handled(out) => {
                assert_eq!(out.exit_code, 0);
                assert_eq!(out.stdout, "hello world");
            }
            Dispatch::NotHandled => panic!("expected Handled"),
        }
    }

    #[test]
    fn dispatch_cli_returns_not_handled_when_no_match() {
        let (db, _dir) = temp_axil();
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(EchoExt)];
        let inv = CliInvocation {
            command_path: vec!["unknown".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            dispatch_cli(&db, &exts, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn dispatch_cli_walks_past_declining_extension() {
        // DecliningExt comes first and matches `echo` but uses the
        // default handle_cli (returns NotHandled). EchoExt comes
        // second and actually handles it. The dispatcher must walk
        // past the decliner.
        let (db, _dir) = temp_axil();
        let exts: Vec<Arc<dyn Extension>> =
            vec![Arc::new(DecliningExt), Arc::new(EchoExt)];
        let inv = CliInvocation {
            command_path: vec!["echo".into()],
            args: vec!["ok".into()],
            stdin: None,
        };
        let result = dispatch_cli(&db, &exts, &inv).unwrap();
        match result {
            Dispatch::Handled(out) => assert_eq!(out.stdout, "ok"),
            Dispatch::NotHandled => panic!("EchoExt should have handled this"),
        }
    }

    #[test]
    fn dispatch_mcp_routes_to_matching_tool() {
        let (db, _dir) = temp_axil();
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(EchoExt)];
        let call = McpCall {
            tool: "ping".into(),
            params: serde_json::json!({"x": 1}),
        };
        let result = dispatch_mcp(&db, &exts, &call).unwrap();
        match result {
            Dispatch::Handled(v) => {
                assert_eq!(v["tool"], "ping");
                assert_eq!(v["params"]["x"], 1);
            }
            Dispatch::NotHandled => panic!("expected Handled"),
        }
    }

    #[test]
    fn dispatch_mcp_returns_not_handled_for_unknown_tool() {
        let (db, _dir) = temp_axil();
        let exts: Vec<Arc<dyn Extension>> = vec![Arc::new(EchoExt)];
        let call = McpCall {
            tool: "unknown".into(),
            params: serde_json::Value::Null,
        };
        assert!(matches!(
            dispatch_mcp(&db, &exts, &call).unwrap(),
            Dispatch::NotHandled
        ));
    }
}
