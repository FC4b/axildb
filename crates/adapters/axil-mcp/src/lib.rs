//! MCP (Model Context Protocol) server for Axil.
//!
//! Implements a JSON-RPC 2.0 server over stdin/stdout, following the MCP specification.
//! The server exposes Axil database operations as MCP tools that can be called by
//! Claude Code and other MCP-compatible clients.
//!
//! # Protocol
//!
//! - Transport: stdin/stdout, newline-delimited JSON-RPC messages
//! - Lifecycle: `initialize` -> `initialized` notification -> tool calls -> `shutdown`
//! - Server capabilities: `{"tools": {}}`

pub mod protocol;
pub mod tools;

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

use axil_core::{Adapter, Axil, AxilError, Protocol};

use protocol::{
    InitializeResult, JsonRpcRequest, JsonRpcResponse, ServerCapabilities, ServerInfo,
    ToolCallResult, ToolDefinition, ToolsCapability, ToolsListResult, INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND, PARSE_ERROR,
};

/// Resolve the embedding model for `db_path` from the project's `axil.toml`,
/// falling back to BgeSmall when no config is present. Mirrors the CLI's
/// `resolve_embedding_model` so MCP and CLI agree on which model to load.
#[cfg(feature = "embed")]
fn resolve_embedding_model(db_path: &std::path::Path) -> axil_vector::models::EmbeddingModel {
    let dir = db_path.parent().unwrap_or(std::path::Path::new("."));
    if let Ok(config) = axil_core::load_config_from(dir) {
        if let Some(name) = config.database.embedding_model.as_deref() {
            if let Some(model) = axil_vector::models::EmbeddingModel::from_name(name) {
                return model;
            }
        }
    }
    axil_vector::models::EmbeddingModel::BgeSmall
}

/// Detect plugin companion files at `path` and attach each one whose
/// on-disk state is present. Mirrors the CLI's `attach_detected_engines`
/// helper so MCP clients get the same set of retrieval/graph/FTS
/// capabilities that `axil recall` / `axil search` / `axil link` use.
///
/// Gated by the crate's optional features (`vector`, `graph`, `fts`) so a
/// minimal build that disables all three still compiles and just ships a
/// CRUD-only MCP surface.
pub(crate) fn attach_detected_engines(
    #[allow(unused_mut)] mut builder: axil_core::AxilBuilder,
) -> anyhow::Result<axil_core::AxilBuilder> {
    #[allow(unused)]
    let path = builder.path().to_path_buf();

    #[cfg(feature = "vector")]
    {
        use axil_vector::AxilBuilderVectorExt;
        if let Ok(Some(_)) = axil_vector::read_stored_dimensions(&path) {
            // When `embed` is on, load the embedder so `db.recall()` and
            // auto-embed-on-insert work end-to-end. Resolve the model from
            // the same `axil.toml` the CLI uses so MCP and CLI agree on
            // which model to load — hard-coding BgeSmall would break any
            // DB built with nomic/bge-base/bge-m3/custom.
            #[cfg(feature = "embed")]
            {
                let model = resolve_embedding_model(&path);
                let with_embed = axil_core::Axil::open(&path).with_embedder_model(model);
                builder = match with_embed {
                    Ok(b) => b,
                    Err(_) => axil_core::Axil::open(&path).with_vector_auto()?,
                };
            }
            #[cfg(not(feature = "embed"))]
            {
                builder = builder.with_vector_auto()?;
            }
        }
    }

    #[cfg(feature = "graph")]
    {
        use axil_graph::AxilBuilderGraphExt;
        if axil_graph::has_graph_store(&path) {
            builder = builder.with_graph_engine()?;
        }
    }

    #[cfg(feature = "fts")]
    {
        use axil_fts::AxilBuilderFtsExt;
        if axil_fts::has_fts_store(&path) {
            builder = builder.with_fts_engine()?;
        }
    }

    // Register every enabled built-in Extension from the central bundle so the
    // MCP `dispatch_mcp` route finds them (`deps_status` flows through
    // DocsExtension::handle_mcp; checkpoint tools route through
    // CheckpointExtension). One registration site shared with the CLI + audit,
    // with the `[extensions] disabled` filter applied centrally.
    let config = path
        .parent()
        .and_then(|dir| axil_core::load_config_from(dir).ok())
        .unwrap_or_default();
    builder = axil_bundle::register_builtin_extensions(builder, &config);

    Ok(builder)
}

/// MCP server wrapping an Axil database.
///
/// Holds an `Arc<Axil>` so the same database can be shared with other Adapters
/// in-process (the [`Adapter`] contract). [`McpServer::new`] keeps the original
/// owned-`Axil` constructor working by wrapping it.
pub struct McpServer {
    db: Arc<Axil>,
}

impl McpServer {
    /// Create a new MCP server backed by the given Axil database.
    pub fn new(db: Axil) -> Self {
        Self { db: Arc::new(db) }
    }

    /// Create an MCP server over an already-shared database handle — the path
    /// the [`McpAdapter`] uses so several Adapters can share one `Axil`.
    pub fn from_arc(db: Arc<Axil>) -> Self {
        Self { db }
    }

    /// Open a database at the given path and create an MCP server.
    ///
    /// Attaches every companion plugin that has on-disk state:
    /// - `*.axil.vec` → vector plugin (with embedder when the `embed`
    ///   feature is enabled and a model is resolvable)
    /// - `*.axil.graph` → graph plugin
    /// - `*.axil.fts/` → FTS plugin
    ///
    /// Missing companions are silently skipped — tools that require an
    /// absent plugin return a structured error at call time instead of
    /// failing at open.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let builder = attach_detected_engines(Axil::open(path))?;
        let db = builder.build()?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Narrow accessor for integration tests that need to verify plugin
    /// attachment after `open()`. Marked hidden so it isn't part of the
    /// public surface documented for MCP users.
    #[doc(hidden)]
    pub fn db_for_tests(&self) -> &Axil {
        &self.db
    }

    /// Run the MCP server, reading JSON-RPC from stdin and writing responses to stdout.
    ///
    /// This method runs until stdin is closed or a shutdown request is received.
    pub async fn run(&self) -> anyhow::Result<()> {
        /// Per-message line cap (16 MB). Caps the malicious-client OOM
        /// vector where a single newline-less line is sent unbounded.
        /// A well-behaved JSON-RPC client emits one message per line
        /// and a 16 MB single message would be pathological anyway —
        /// the typical message is sub-kilobyte.
        const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

        let mut stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();

        loop {
            // Re-wrap `&mut stdin` with `take()` per iteration so each
            // line read is hard-bounded. BufReader is constructed
            // fresh — tokio::io::stdin() is already line-buffered at
            // the OS level for our purposes, so the extra buffering
            // layer mostly serves `read_until` semantics.
            let mut buf: Vec<u8> = Vec::with_capacity(4096);
            let n = {
                let limited = (&mut stdin).take(MAX_LINE_BYTES);
                let mut br = BufReader::new(limited);
                br.read_until(b'\n', &mut buf).await?
            };
            if n == 0 {
                break; // EOF
            }
            // If we read exactly MAX_LINE_BYTES without a trailing
            // newline, the client is feeding us an unbounded line —
            // surface a parse error and terminate the connection
            // (the host should reconnect on the next tool call).
            let truncated = n as u64 == MAX_LINE_BYTES && !buf.ends_with(b"\n");
            if truncated {
                let resp = JsonRpcResponse::error(
                    None,
                    PARSE_ERROR,
                    format!("line exceeds {MAX_LINE_BYTES}-byte cap"),
                );
                write_response(&mut stdout, &resp).await?;
                break;
            }
            let line = match std::str::from_utf8(&buf) {
                Ok(s) => s.trim().to_string(),
                Err(_) => {
                    let resp =
                        JsonRpcResponse::error(None, PARSE_ERROR, "Parse error: invalid UTF-8");
                    write_response(&mut stdout, &resp).await?;
                    continue;
                }
            };
            if line.is_empty() {
                continue;
            }

            // Parse the JSON-RPC message.
            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(_) => {
                    let resp = JsonRpcResponse::error(None, PARSE_ERROR, "Parse error");
                    write_response(&mut stdout, &resp).await?;
                    continue;
                }
            };

            let is_shutdown = request.method == "shutdown";

            // Dispatch based on method.
            let response = self.handle_request(&request);

            // Notifications (no id) get no response.
            if request.id.is_none() && !is_shutdown {
                continue;
            }

            if let Some(resp) = response {
                write_response(&mut stdout, &resp).await?;
            }

            if is_shutdown {
                break;
            }
        }

        Ok(())
    }

    /// Handle a single JSON-RPC request and return an optional response.
    fn handle_request(&self, req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
        match req.method.as_str() {
            "initialize" => Some(self.handle_initialize(req)),
            "initialized" => None, // Notification, no response.
            "shutdown" => Some(JsonRpcResponse::success(req.id.clone(), Value::Null)),
            "tools/list" => Some(self.handle_tools_list(req)),
            "tools/call" => Some(self.handle_tools_call(req)),
            _ => {
                // Unknown method.
                if req.id.is_some() {
                    Some(JsonRpcResponse::error(
                        req.id.clone(),
                        METHOD_NOT_FOUND,
                        format!("Method not found: {}", req.method),
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Handle the `initialize` request.
    fn handle_initialize(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        let result = InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability {},
            },
            server_info: ServerInfo {
                name: "axil-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };

        match serde_json::to_value(&result) {
            Ok(v) => JsonRpcResponse::success(req.id.clone(), v),
            Err(e) => JsonRpcResponse::error(
                req.id.clone(),
                INTERNAL_ERROR,
                format!("serialization error: {e}"),
            ),
        }
    }

    /// Handle the `tools/list` request.
    ///
    /// Phase 17 P3.3 (final): start from the static hardcoded list,
    /// then overlay tools from every registered Extension's
    /// [`axil_core::Extension::mcp_tools`]. Extension entries
    /// **replace** matching hardcoded entries (by tool name) so the
    /// `tools/list` output mirrors the dispatch contract: when the
    /// dispatcher routes a tool to an Extension, that Extension's
    /// description and schema win.
    fn handle_tools_list(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        let mut tools = tools::tool_definitions();
        for ext in self.db.extensions() {
            if let Some(surface) = ext.mcp_tools() {
                for t in surface.tools {
                    let def = ToolDefinition {
                        name: t.name.clone(),
                        description: t.description,
                        input_schema: t.input_schema,
                    };
                    if let Some(existing) = tools.iter_mut().find(|d| d.name == t.name) {
                        *existing = def;
                    } else {
                        tools.push(def);
                    }
                }
            }
        }
        let result = ToolsListResult { tools };

        match serde_json::to_value(&result) {
            Ok(v) => JsonRpcResponse::success(req.id.clone(), v),
            Err(e) => JsonRpcResponse::error(
                req.id.clone(),
                INTERNAL_ERROR,
                format!("serialization error: {e}"),
            ),
        }
    }

    /// Handle the `tools/call` request.
    fn handle_tools_call(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &req.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(req.id.clone(), INVALID_PARAMS, "missing params");
            }
        };

        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                return JsonRpcResponse::error(
                    req.id.clone(),
                    INVALID_PARAMS,
                    "missing 'name' in params",
                );
            }
        };

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let result: ToolCallResult = tools::dispatch(&self.db, tool_name, &arguments);

        match serde_json::to_value(&result) {
            Ok(v) => JsonRpcResponse::success(req.id.clone(), v),
            Err(e) => JsonRpcResponse::error(
                req.id.clone(),
                INTERNAL_ERROR,
                format!("serialization error: {e}"),
            ),
        }
    }
}

/// Tier-3 [`Adapter`] for MCP over stdio JSON-RPC.
///
/// Expresses the MCP server through Axil's stable Adapter contract: `bind` a
/// shared `Axil`, then `run` the blocking serve loop. It owns the tokio runtime
/// internally so a caller drives it synchronously — the `Adapter::run(self)`
/// shape — instead of managing async itself.
#[derive(Default)]
pub struct McpAdapter {
    db: Option<Arc<Axil>>,
}

impl McpAdapter {
    /// An unbound MCP adapter. Call [`Adapter::bind`] before [`Adapter::run`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl Adapter for McpAdapter {
    fn id(&self) -> &str {
        "mcp"
    }

    fn protocol(&self) -> Protocol {
        Protocol::Mcp
    }

    fn bind(&mut self, db: Arc<Axil>) -> axil_core::Result<()> {
        self.db = Some(db);
        Ok(())
    }

    fn run(self) -> axil_core::Result<()> {
        let db = self
            .db
            .ok_or_else(|| AxilError::plugin("MCP adapter run() called before bind()"))?;
        let server = McpServer::from_arc(db);
        // The serve loop is async; the Adapter contract is a synchronous
        // `run(self)`, so own a current-thread runtime and block on it. `axil`
        // builds no ambient runtime, so this never nests.
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AxilError::plugin(format!("tokio runtime init failed: {e}")))?;
        rt.block_on(server.run())
            .map_err(|e| AxilError::plugin(format!("MCP server error: {e}")))
    }
}

/// Write a JSON-RPC response as a newline-delimited JSON line to the writer.
async fn write_response<W: AsyncWriteExt + Unpin>(
    writer: &mut W,
    response: &JsonRpcResponse,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(response)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod adapter_tests {
    use super::*;

    #[test]
    fn mcp_adapter_identity() {
        let a = McpAdapter::new();
        assert_eq!(a.id(), "mcp");
        assert_eq!(a.protocol(), Protocol::Mcp);
    }

    #[test]
    fn run_before_bind_errors_instead_of_panicking() {
        // An unbound adapter refuses to run (no db, no runtime, no stdin read).
        assert!(McpAdapter::new().run().is_err());
    }

    #[test]
    fn bind_accepts_a_shared_db() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Axil::open(dir.path().join("m.axil")).build().unwrap());
        let mut a = McpAdapter::new();
        assert!(a.bind(db).is_ok());
    }
}
