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

/// Top-level routing guidance returned in the `initialize` response so the
/// client surfaces it to the model once per session. The tool surface overlaps
/// by design (code lookups vs memory recall vs session resume); without a map
/// the agent pays selection cost and mis-picks. Keep this terse — it ships on
/// every session init, not per call.
const SERVER_INSTRUCTIONS: &str = "\
Axil is the agent's persistent memory + code index for this project. Routing:
- Code: \"where/how is X\", or before editing a file/symbol → `code_context` (task-scoped bundle) or `code_search` (locate symbols). Prefer these over recall for code.
- Memory: past decisions, errors, or context → `recall`; expand one hit with `get <id>`. Time-scoped history → `query_history`.
- Write knowledge (do this after each unit of work): a decision → `remember_decision`; a bug+fix → `remember_error`; a preference → `set_preference`; anything else → `store`.
- Resume a session → `boot` (recent decisions, errors, checkpoint). Graph link between records → `link`.";

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
    // Config (from the db dir) governs which Engines attach (`[engines]
    // disabled`) and which Extensions register (`[extensions] disabled`) —
    // identical to the CLI's `attach_detected_engines`, so MCP and CLI expose
    // the same engine set for the same DB.
    let config = path
        .parent()
        .and_then(|dir| axil_core::load_config_from(dir).ok())
        .unwrap_or_default();

    #[cfg(feature = "vector")]
    {
        use axil_vector::AxilBuilderVectorExt;
        // Register the named-vector-space factory unconditionally so the
        // `add_vector` / `similar` tools can target named spaces even when the
        // default store is absent. Additive — existing vector paths unchanged.
        builder = axil_vector::with_vector_spaces(builder);
        if !config.is_engine_disabled("vec") {
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
    }

    #[cfg(feature = "graph")]
    {
        use axil_graph::AxilBuilderGraphExt;
        if !config.is_engine_disabled("graph") && axil_graph::has_graph_store(&path) {
            builder = builder.with_graph_engine()?;
        }
    }

    #[cfg(feature = "timeseries")]
    {
        use axil_timeseries::AxilBuilderTimeSeriesExt;
        if !config.is_engine_disabled("ts") && axil_timeseries::has_timeseries_store(&path) {
            builder = builder.with_timeseries_engine()?;
        }
    }

    #[cfg(feature = "fts")]
    {
        use axil_fts::AxilBuilderFtsExt;
        if !config.is_engine_disabled("fts") && axil_fts::has_fts_store(&path) {
            builder = builder.with_fts_engine()?;
        }
    }

    // Register every enabled built-in Extension from the central bundle so the
    // MCP `dispatch_mcp` route finds them (`deps_status` flows through
    // DocsExtension::handle_mcp; checkpoint tools route through
    // CheckpointExtension). One registration site shared with the CLI + audit,
    // with the `[extensions] disabled` filter applied centrally.
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
    ///
    /// Unlike [`McpServer::open`], this trusts the caller's handle as-is: it does
    /// **not** attach engines or register built-in/WASM Extensions. The caller is
    /// responsible for building a fully-configured `Axil` (as `axil-cli`'s `mcp`
    /// command does via `attach_detected_engines` + `register_installed_plugins`
    /// before binding) — otherwise tools that need a missing engine/extension
    /// return errors and plugin tools are absent from `tools/list`.
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
    ///
    /// Honors `[healing] event_log` from the database directory's `axil.toml`
    /// (with the `event-log` feature), as the CLI's open paths do.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let builder = attach_detected_engines(Axil::open(path))?;
        let db = builder.build()?;
        #[cfg(feature = "event-log")]
        if path
            .parent()
            .and_then(|dir| axil_core::load_config_from(dir).ok())
            .is_some_and(|config| config.healing.event_log)
        {
            db.set_event_log_enabled(true);
        }
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
    ///
    /// Pipelined requests keep their arrival order wherever it matters:
    /// read-only tool calls overlap, while a state-mutating call waits for
    /// every earlier request and holds back every later one. A handler that
    /// panics is answered with a JSON-RPC internal error for its id.
    ///
    /// A read from tokio's stdin can't be cancelled, so one stays parked on
    /// the blocking pool after this returns; a host that owns the runtime
    /// should end it with `shutdown_background` (as [`McpAdapter`] does)
    /// rather than a plain drop, which would wait for the next input line.
    pub async fn run(&self) -> anyhow::Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// The transport-agnostic body of [`McpServer::run`]: read newline-delimited
    /// JSON-RPC frames from `input` and write responses to `output` until EOF or
    /// a `shutdown` request.
    async fn serve<R, W>(&self, input: R, output: W) -> anyhow::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin,
    {
        /// Per-message line cap (16 MB). Caps the malicious-client OOM
        /// vector where a single newline-less line is sent unbounded.
        /// A well-behaved JSON-RPC client emits one message per line
        /// and a 16 MB single message would be pathological anyway —
        /// the typical message is sub-kilobyte.
        const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

        enum Incoming {
            Frame(String),
            TooLarge,
            Eof,
        }

        let mut stdout = output;

        // A dedicated reader task owns stdin so line reads never interleave
        // with response writes; frames flow to the loop as messages.
        let (in_tx, mut in_rx) = tokio::sync::mpsc::unbounded_channel::<Incoming>();
        tokio::spawn(async move {
            // One persistent reader for the whole connection. A client that
            // pipelines requests (several frames in one pipe chunk) gets them
            // all buffered here; a per-iteration BufReader would drop the
            // still-buffered frames on the floor when it goes out of scope,
            // silently swallowing every request after the first.
            let mut reader = BufReader::new(input);
            loop {
                // Re-wrap with `take()` per iteration so each line read is
                // hard-bounded without discarding the shared buffer.
                let mut buf: Vec<u8> = Vec::with_capacity(4096);
                let n = match (&mut reader)
                    .take(MAX_LINE_BYTES)
                    .read_until(b'\n', &mut buf)
                    .await
                {
                    Ok(n) => n,
                    Err(_) => break,
                };
                if n == 0 {
                    break; // EOF
                }
                // If we read exactly MAX_LINE_BYTES without a trailing
                // newline, the client is feeding us an unbounded line —
                // surface a parse error and terminate the connection
                // (the host should reconnect on the next tool call).
                if n as u64 == MAX_LINE_BYTES && !buf.ends_with(b"\n") {
                    let _ = in_tx.send(Incoming::TooLarge);
                    break;
                }
                let line = String::from_utf8_lossy(&buf).to_string();
                if in_tx.send(Incoming::Frame(line)).is_err() {
                    break;
                }
            }
        });

        // Completed responses flow back through this channel: dispatch runs
        // concurrently on the blocking pool, while every write stays
        // serialized here. A slow tool call therefore no longer blocks the
        // connection — later requests (and cancellations) are read while it
        // runs. JSON-RPC responses carry their id, so replying out of order
        // is well-defined for hosts.
        let (resp_tx, mut resp_rx) = tokio::sync::mpsc::unbounded_channel::<JsonRpcResponse>();

        // Requests are admitted in arrival order by a dispatcher task that
        // keeps pipelined requests causally ordered: read-only requests overlap
        // freely, but a state-mutating one runs only after everything before it
        // has finished and before anything after it starts. A pipelined
        // `store` → `recall` (or `delete` → `get`) observes the write.
        let (req_tx, req_rx) = tokio::sync::mpsc::unbounded_channel::<JsonRpcRequest>();
        tokio::spawn(dispatch_in_order(self.db.clone(), req_rx, resp_tx.clone()));

        loop {
            tokio::select! {
                biased;
                // Prefer draining finished responses over reading new frames
                // so a pipelining client gets replies as they complete.
                maybe_resp = resp_rx.recv() => {
                    if let Some(resp) = maybe_resp {
                        write_response(&mut stdout, &resp).await?;
                    }
                    continue;
                }
                incoming = in_rx.recv() => {
                    let line = match incoming {
                        // The reader task ended (stdin closed): stop reading.
                        None | Some(Incoming::Eof) => break,
                        Some(Incoming::TooLarge) => {
                            let resp = JsonRpcResponse::error(
                                None,
                                PARSE_ERROR,
                                format!("line exceeds {MAX_LINE_BYTES}-byte cap"),
                            );
                            write_response(&mut stdout, &resp).await?;
                            break;
                        }
                        Some(Incoming::Frame(line)) => line,
                    };

                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }

                    // Parse the JSON-RPC message.
                    let request: JsonRpcRequest = match serde_json::from_str(&line) {
                        Ok(req) => req,
                        Err(_) => {
                            let resp =
                                JsonRpcResponse::error(None, PARSE_ERROR, "Parse error");
                            write_response(&mut stdout, &resp).await?;
                            continue;
                        }
                    };

                    // Shutdown exits the loop immediately; the reply is written
                    // inline so it can never race the process teardown.
                    if request.method == "shutdown" {
                        let resp =
                            JsonRpcResponse::success(request.id.clone(), serde_json::Value::Null);
                        write_response(&mut stdout, &resp).await?;
                        break;
                    }

                    // Notifications (no id) get no response.
                    if request.id.is_none() {
                        continue;
                    }

                    // The dispatcher answers through `resp_tx` whenever the
                    // request completes.
                    if req_tx.send(request).is_err() {
                        break;
                    }
                }
            }
        }

        // Drain in-flight work before returning: a client that pipelines a
        // request and closes stdin still gets its answer.
        drop(req_tx);
        drop(resp_tx);
        while let Some(resp) = resp_rx.recv().await {
            write_response(&mut stdout, &resp).await?;
        }

        Ok(())
    }

    /// Drive a single raw JSON-RPC frame the way the stdio `run` loop does and
    /// return the serialized response line, or `None` for a notification (a
    /// request with no `id`) that takes no reply.
    ///
    /// This is the in-process equivalent of piping one newline-delimited frame
    /// into `axil --db <DB> mcp`: it parses the line, routes it through
    /// [`McpServer::handle_request`], and serializes the response back to a
    /// JSON string — without touching stdin/stdout. It lets a host (or a test)
    /// exercise the exact `initialize` / `tools/list` / `tools/call` frames the
    /// docs document. A malformed frame yields a JSON-RPC parse-error response
    /// rather than an `Err`, matching the serve loop's behavior.
    pub fn handle_frame(&self, frame: &str) -> Option<String> {
        let request: JsonRpcRequest = match serde_json::from_str(frame.trim()) {
            Ok(req) => req,
            Err(_) => {
                let resp = JsonRpcResponse::error(None, PARSE_ERROR, "Parse error");
                return Some(serde_json::to_string(&resp).unwrap_or_default());
            }
        };
        let resp = self.handle_request(&request)?;
        Some(serde_json::to_string(&resp).unwrap_or_default())
    }

    /// Handle a single JSON-RPC request and return an optional response.
    fn handle_request(&self, req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
        match req.method.as_str() {
            "initialize" => Some(self.handle_initialize(req)),
            "initialized" => None, // Notification, no response.
            // Client-side cancellation notice: the work is already dispatched
            // concurrently, so there is nothing to unwind here — the result
            // for a cancelled id is simply discarded by the host.
            "notifications/cancelled" => None,
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
            instructions: Some(SERVER_INSTRUCTIONS.into()),
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
    /// start from the static hardcoded list,
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
        let served = rt.block_on(server.run());
        // The stdin reader always has a read parked on tokio's blocking pool,
        // and that read can't be cancelled: dropping the runtime would wait on
        // it, so a client that sends `shutdown` but keeps stdin open would
        // never see the process exit. Every reply is already written by now.
        rt.shutdown_background();
        served.map_err(|e| AxilError::plugin(format!("MCP server error: {e}")))
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

/// Run requests in arrival order under a readers/writer discipline.
///
/// Consecutive read-only requests (see [`is_read_only_request`]) run
/// concurrently. Any other request is a barrier: it waits for every earlier
/// request to finish, and nothing admitted after it starts until it has.
async fn dispatch_in_order(
    db: Arc<Axil>,
    mut requests: tokio::sync::mpsc::UnboundedReceiver<JsonRpcRequest>,
    replies: tokio::sync::mpsc::UnboundedSender<JsonRpcResponse>,
) {
    let mut reads = tokio::task::JoinSet::new();
    while let Some(request) = requests.recv().await {
        // Reap finished reads so a long read-only stream doesn't pile up handles.
        while reads.try_join_next().is_some() {}
        if is_read_only_request(&request) {
            reads.spawn(answer(db.clone(), request, replies.clone()));
        } else {
            while reads.join_next().await.is_some() {}
            answer(db.clone(), request, replies.clone()).await;
        }
    }
    // Dropping a JoinSet aborts its tasks, which would lose their replies.
    while reads.join_next().await.is_some() {}
}

/// Run one request on the blocking pool and send its reply.
///
/// A panicking handler still gets an answer — a JSON-RPC internal error for
/// its id — so the client never waits out a timeout for a reply that would
/// otherwise never come.
async fn answer(
    db: Arc<Axil>,
    request: JsonRpcRequest,
    replies: tokio::sync::mpsc::UnboundedSender<JsonRpcResponse>,
) {
    let id = request.id.clone();
    let handled =
        tokio::task::spawn_blocking(move || McpServer { db }.handle_request(&request)).await;
    let response = match handled {
        Ok(response) => response,
        Err(e) => {
            let reason = match e.try_into_panic() {
                Ok(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "non-string panic payload".into());
                    format!("request handler panicked: {detail}")
                }
                Err(_) => "request handler was cancelled".into(),
            };
            Some(JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("internal error: {reason}"),
            ))
        }
    };
    if let Some(response) = response {
        let _ = replies.send(response);
    }
}

/// Whether `request` leaves the database untouched, so it may overlap with
/// other read-only requests. Anything not positively classified — an unknown
/// method, an unknown or plugin tool — counts as mutating, which only ever
/// costs concurrency, never ordering.
fn is_read_only_request(request: &JsonRpcRequest) -> bool {
    match request.method.as_str() {
        "initialize" | "tools/list" => true,
        "tools/call" => {
            let params = request.params.as_ref();
            let name = params.and_then(|p| p.get("name")).and_then(Value::as_str);
            let args = params.and_then(|p| p.get("arguments")).unwrap_or(&Value::Null);
            name.is_some_and(|name| tools::is_read_only_call(name, args))
        }
        _ => false,
    }
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

    /// Drift guard: every tool name the `initialize` routing instructions point
    /// an agent at must be a real, registered tool. A rename/removal that
    /// forgets to update `SERVER_INSTRUCTIONS` would otherwise ship stale
    /// guidance to non-existent tools.
    #[test]
    fn server_instructions_reference_only_real_tools() {
        let defs = crate::tools::tool_definitions();
        let names: std::collections::HashSet<&str> =
            defs.iter().map(|t| t.name.as_str()).collect();
        // The tools the instructions steer toward (backticked in the string).
        let referenced = [
            "code_context",
            "code_search",
            "recall",
            "get",
            "query_history",
            "remember_decision",
            "remember_error",
            "set_preference",
            "store",
            "boot",
            "link",
        ];
        for name in referenced {
            assert!(
                names.contains(name),
                "SERVER_INSTRUCTIONS references `{name}` but it is not a registered MCP tool — \
                 update the instructions or restore the tool"
            );
            assert!(
                SERVER_INSTRUCTIONS.contains(name),
                "test's `referenced` list names `{name}` but the instructions no longer mention it"
            );
        }
    }

    /// The MCP surface guide, embedded at compile time so the drift test can
    /// diff it against the runtime tool set without a working-dir-relative read.
    const MCP_DOC: &str = include_str!("../../../../docs/src/agents/mcp.md");

    /// Every tool the assembled server actually exposes: the built-in
    /// `tool_definitions()` **plus** each enabled Extension's `mcp_tools()`
    /// surface. `tool_definitions()` alone omits the Extension tools
    /// (`dep_docs`/`deps_status`, `checkpoint`/`checkpoint_show`), which only
    /// reach `tools/list` via `register_builtin_extensions`.
    fn assembled_tool_names() -> std::collections::BTreeSet<String> {
        let mut names: std::collections::BTreeSet<String> = crate::tools::tool_definitions()
            .into_iter()
            .map(|t| t.name)
            .collect();
        let config = axil_core::AxilConfig::default();
        for surface in axil_bundle::builtin_mcp_surfaces(&config) {
            for tool in surface.tools {
                names.insert(tool.name);
            }
        }
        // `recall_delta` only exists under the off-by-default `event-log` feature,
        // so it is documented in a prose subsection of mcp.md (not the tool
        // table) to keep the doc portable across feature sets. Exclude it from
        // the strict table-equality drift check; its presence is asserted
        // separately by the feature-gated `recall_delta_tool_*` tests.
        names.remove("recall_delta");
        names
    }

    /// Extract the documented tool names from `docs/src/agents/mcp.md`: the
    /// first cell of every tool-table row, recognised as a line of the form
    /// `| `name` | …`. Prose back-ticks (e.g. "prefer `recall`") are ignored
    /// because they aren't in leading-cell position.
    fn documented_tool_names() -> std::collections::BTreeSet<String> {
        MCP_DOC
            .lines()
            .filter_map(|line| {
                let cell = line.strip_prefix("| `")?;
                let name = cell.split('`').next()?;
                // A tool name is a bare identifier; the header separator row
                // (`|------|`) and multi-word cells never qualify.
                if !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Drift guard: `docs/src/agents/mcp.md` must document exactly the tools the
    /// assembled server exposes — built-ins **and** the enabled Extensions'
    /// tools. Adding or removing a tool without updating the doc table (or vice
    /// versa) fails here. The doc is the agent-facing contract; stale tool docs
    /// mislead every MCP client.
    #[test]
    fn mcp_doc_matches_assembled_tool_surface() {
        let runtime = assembled_tool_names();
        let documented = documented_tool_names();

        let undocumented: Vec<&String> = runtime.difference(&documented).collect();
        assert!(
            undocumented.is_empty(),
            "these assembled MCP tools are missing from docs/src/agents/mcp.md: {undocumented:?} \
             — add a table row (name, params, when-to-use)"
        );

        let phantom: Vec<&String> = documented.difference(&runtime).collect();
        assert!(
            phantom.is_empty(),
            "docs/src/agents/mcp.md documents tools the assembled server does not expose: \
             {phantom:?} — remove the row or restore the tool"
        );
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

    #[cfg(feature = "event-log")]
    #[test]
    fn open_honors_event_log_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("axil.toml"), "[healing]\nevent_log = true\n").unwrap();
        let server = McpServer::open(&dir.path().join("m.axil")).unwrap();
        assert!(server.db.event_log_enabled());
    }
}

#[cfg(test)]
mod serve_tests {
    use super::*;
    use axil_core::{Dispatch, Extension, McpCall, McpSurface, McpTool};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    /// How long a `slow` probe holds its request open — long enough that an
    /// unordered follow-up request would reliably finish first.
    const HOLD: Duration = Duration::from_millis(300);

    /// Extension that claims a few tool names to make request scheduling
    /// observable. It runs ahead of the built-in handlers and, except where it
    /// answers itself, falls through to them.
    #[derive(Default)]
    struct Probe {
        arrivals: Mutex<usize>,
        arrived: Condvar,
    }

    impl Extension for Probe {
        fn id(&self) -> &str {
            "probe"
        }

        fn mcp_tools(&self) -> Option<McpSurface> {
            let tool = |name: &str| McpTool::new(name, "scheduling probe", json!({}));
            Some(McpSurface::new(vec![
                tool("boom"),
                tool("recall"),
                tool("store"),
                tool("list"),
            ]))
        }

        fn handle_mcp(&self, _db: &Axil, call: &McpCall) -> axil_core::Result<Dispatch<Value>> {
            let slow = call.params.get("slow").is_some();
            match call.tool.as_str() {
                "boom" => panic!("probe: mutating handler panicked"),
                "recall" if call.params.get("panic").is_some() => {
                    panic!("probe: read-only handler panicked")
                }
                // Rendezvous: report whether a second `recall` arrived while
                // this one was still running.
                "recall" => {
                    let mut n = self.arrivals.lock().unwrap();
                    *n += 1;
                    self.arrived.notify_all();
                    let (_n, wait) = self
                        .arrived
                        .wait_timeout_while(n, Duration::from_secs(5), |n| *n < 2)
                        .unwrap();
                    Ok(Dispatch::Handled(json!({"overlapped": !wait.timed_out()})))
                }
                "store" | "list" if slow => {
                    std::thread::sleep(HOLD);
                    Ok(Dispatch::NotHandled)
                }
                _ => Ok(Dispatch::NotHandled),
            }
        }
    }

    fn call(id: u64, tool: &str, arguments: Value) -> String {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        })
        .to_string()
    }

    /// Pipeline `frames` into a fresh server in one write, close the input, and
    /// collect every response by id once the server has drained.
    async fn exchange(frames: &[String]) -> HashMap<u64, Value> {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("serve.axil"))
            .with_extension(Probe::default())
            .build()
            .unwrap();
        let server = McpServer::new(db);
        let input = format!("{}\n", frames.join("\n"));
        let mut output = Vec::new();
        server
            .serve(std::io::Cursor::new(input.into_bytes()), &mut output)
            .await
            .unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| {
                let v: Value = serde_json::from_str(line).unwrap();
                (v["id"].as_u64().expect("response carries its id"), v)
            })
            .collect()
    }

    fn reply(replies: &HashMap<u64, Value>, id: u64) -> &Value {
        replies
            .get(&id)
            .unwrap_or_else(|| panic!("no response for request {id}: {replies:?}"))
    }

    /// The tool payload of a successful `tools/call` response.
    fn payload(replies: &HashMap<u64, Value>, id: u64) -> Value {
        let resp = reply(replies, id);
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("request {id} did not succeed: {resp}"));
        serde_json::from_str(text).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handler_panic_answers_with_internal_error() {
        let replies = exchange(&[
            call(1, "boom", json!({})),
            call(2, "recall", json!({"query": "q", "panic": true})),
            call(3, "list", json!({"table": "notes"})),
        ])
        .await;
        for id in [1, 2] {
            assert_eq!(
                reply(&replies, id)["error"]["code"],
                INTERNAL_ERROR,
                "a panicking handler must still answer request {id}"
            );
        }
        assert_eq!(payload(&replies, 3), json!([]), "the server keeps serving");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_after_write_observes_the_write() {
        let replies = exchange(&[
            call(1, "store", json!({"table": "notes", "data": {"n": 1}, "slow": true})),
            call(2, "list", json!({"table": "notes"})),
        ])
        .await;
        assert!(reply(&replies, 1)["result"].is_object());
        assert_eq!(
            payload(&replies, 2).as_array().unwrap().len(),
            1,
            "a read pipelined after a write must not run before it"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_does_not_overtake_an_earlier_read() {
        let replies = exchange(&[
            call(1, "list", json!({"table": "notes", "slow": true})),
            call(2, "store", json!({"table": "notes", "data": {"n": 1}})),
        ])
        .await;
        assert_eq!(
            payload(&replies, 1),
            json!([]),
            "a write pipelined after a read must wait for it"
        );
        assert!(reply(&replies, 2)["result"].is_object());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_only_requests_still_overlap() {
        let replies = exchange(&[
            call(1, "recall", json!({"query": "a"})),
            call(2, "recall", json!({"query": "b"})),
        ])
        .await;
        for id in [1, 2] {
            assert_eq!(
                payload(&replies, id)["overlapped"],
                true,
                "read-only request {id} ran alone"
            );
        }
    }
}
