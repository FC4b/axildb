# Authoring an Adapter (Tier 3)

An **Adapter** is the interface between Axil and the outside world. It translates an external protocol (CLI argv, MCP stdio JSON-RPC, HTTP, AxilQL, GraphQL, gRPC) to and from `Axil::query()` calls. Adapters don't store anything themselves — they live on the process boundary.

This guide covers writing a new Adapter, using [`axil-cli`](../../../crates/axil-cli/) and [`axil-mcp`](../../../crates/axil-mcp/) as the reference implementations.

## When to write an Adapter

You're in the right tier if you want to:

- Expose Axil over a new protocol (HTTP REST, gRPC, GraphQL, WebSocket).
- Add a new query language frontend (`axil-ql` is the existing one).
- Wrap Axil as a backend for an existing tool (Slack bot, IDE plugin, web dashboard).

You're in the wrong tier if you want to:

- Add a new agent command — that's an [Extension](extensions.md)'s `cli_commands()` surface, which Adapters discover and surface automatically.
- Store something new — that's an [Extension](extensions.md) or an [Engine](engines.md).

## The contract

Adapters have a small, focused contract:

```rust
// crates/axil-core/src/adapter.rs

use crate::{Axil, Result};

pub trait Adapter {
    /// Stable adapter id (e.g. "cli", "mcp", "http", "axilql").
    fn id(&self) -> &str;

    /// The protocol this adapter serves.
    fn protocol(&self) -> Protocol;

    /// Called once after the Axil host is built. The adapter inspects
    /// db.extensions() and prepares its dispatch table.
    fn bind(&mut self, db: Axil) -> Result<()>;

    /// Run the adapter (blocking). For axil-cli this parses argv and
    /// dispatches; for axil-mcp this runs the stdio JSON-RPC loop;
    /// for an HTTP adapter this binds a socket and serves requests.
    fn run(self) -> Result<()>;
}

pub enum Protocol {
    Cli,
    Mcp,         // stdio JSON-RPC
    Http,        // REST / JSON
    GraphQl,
    QueryLang,   // AxilQL or similar
    Custom(&'static str),
}
```

That's the whole trait. An Adapter is responsible for three things: discovering Extensions, translating protocol calls to Axil API calls, and mapping errors back.

## Error mapping contract

Each Adapter is responsible for translating `axil_core::AxilError` into its protocol's native error type. This is documented, not enforced:

| Protocol | Error mapping |
|---|---|
| **CLI** | Exit code (2 for usage error, 1 for runtime error, 0 for success) + human-readable stderr. Errors with a `kind: "not_found"` may use a distinct exit code if your CLI conventions require it. |
| **MCP** | JSON-RPC error object: `{ code: -32000, message: <human readable>, data: { axil_error: <enum variant> } }`. Use the standard JSON-RPC codes for protocol-level errors (parse error, method not found). |
| **HTTP** | HTTP status code (4xx for client error, 5xx for Axil/storage error) + JSON body `{ error: <enum variant>, message: <human readable>, details: { … } }`. |
| **GraphQL** | `errors[]` array with `extensions.code` set to the Axil error variant. |
| **QueryLang** | `QueryError` with line/col when the source-location is known. |

The principle: surface the *kind* of error in a machine-readable field, and the *message* in a human-readable field. Don't lose the Axil error variant — clients often need it to retry intelligently.

## Surfacing Extensions

The most important Adapter responsibility is **discovering and surfacing every registered Extension's commands/tools** through the Adapter's protocol. Adapters do NOT hard-code knowledge of specific Extensions like `axil-docs` or `axil-scip`.

`axil-core` provides helpers for the common cases:

```rust
// Compose all registered Extensions' CLI subcommands into one clap::Command.
pub fn compose_cli_surface(extensions: &[Box<dyn Extension>]) -> clap::Command { ... }

// Compose all registered Extensions' MCP tools into one tool list.
pub fn compose_mcp_surface(extensions: &[Box<dyn Extension>]) -> Vec<McpTool> { ... }
```

For a new protocol, write your own composition. The pattern is:

```rust
impl Adapter for HttpAdapter {
    fn bind(&mut self, db: Axil) -> Result<()> {
        for ext in db.extensions() {
            for route in ext_to_http_routes(ext.as_ref()) {
                self.router = self.router.route(&route.path, route.handler);
            }
        }
        self.db = Some(db);
        Ok(())
    }
}
```

The Adapter author decides how their protocol *shape* maps to an Extension's surface. CLI uses subcommands; HTTP could use REST paths under `/ext/<id>/…`; GraphQL could use a namespaced field per Extension.

## Stateless w.r.t. storage

Adapters do not store anything. They:

- Read and write through `Axil` (the host).
- Hold no tables, no files, no databases of their own.
- May hold in-memory dispatch state (the composed surface, route table, etc.) — this is fine.

If you find yourself wanting persistent state in your Adapter, what you actually want is an Extension that the Adapter surfaces. Move the storage there.

## Hello-world Adapter

A minimal HTTP Adapter that surfaces `recall`:

```rust
// crates/axil-http/src/lib.rs
use axil_core::{Adapter, Axil, Protocol, Result};
use axum::{Router, routing::get, extract::State, Json};
use std::sync::Arc;

pub struct HttpAdapter {
    bind_addr: String,
    db: Option<Arc<Axil>>,
}

impl HttpAdapter {
    pub fn new(bind_addr: &str) -> Self {
        Self { bind_addr: bind_addr.to_string(), db: None }
    }
}

impl Adapter for HttpAdapter {
    fn id(&self) -> &str { "http" }
    fn protocol(&self) -> Protocol { Protocol::Http }

    fn bind(&mut self, db: Axil) -> Result<()> {
        self.db = Some(Arc::new(db));
        Ok(())
    }

    fn run(self) -> Result<()> {
        let db = self.db.expect("bind() must be called before run()");
        let app = Router::new()
            .route("/recall", get(recall_handler))
            .with_state(db);
        // … axum::Server::bind(…).serve(app.into_make_service())
        Ok(())
    }
}

async fn recall_handler(State(db): State<Arc<Axil>>, q: String) -> Json<RecallResponse> {
    let hits = db.recall(&q, 10).unwrap_or_default();
    Json(RecallResponse { hits })
}
```

Wire it from a host binary:

```rust
fn main() -> Result<()> {
    let db = Axil::open("./memory.axil")
        .with_vector_plugin()?
        .with_extension(axil_docs::DocsExtension::default())
        .build()?;

    let mut adapter = HttpAdapter::new("127.0.0.1:8080");
    adapter.bind(db)?;
    adapter.run()
}
```

## Reference implementations

| Adapter | Crate | Protocol |
|---|---|---|
| CLI | [`axil-cli`](../../../crates/axil-cli/) | argv → dispatch (clap) |
| MCP | [`axil-mcp`](../../../crates/axil-mcp/) | stdio JSON-RPC |
| AxilQL | [`axil-ql`](../../../crates/axil-ql/) | Query string → `QueryBuilder` |

Read `axil-mcp` first if you want the simplest end-to-end example (one transport, well-defined tool list). Read `axil-cli` for the more sprawling case (argument parsing, multiple subcommand groups, embedded Extension surfaces, brain-hook orchestration).

## What you don't have to do

- **No storage.** You don't own any tables, files, or databases.
- **No lifecycle hooks.** Adapters don't see record insert/update/delete events — those are for Engines.
- **No drift detection.** Extensions handle their own refresh; Adapters just call `ext.refresh()` when the protocol asks for it.
- **No hard-coded Extension knowledge.** Surface them via `compose_*` helpers or your own composition, but don't bake `axil-docs` knowledge into your dispatch.

## Path to upstream

Adapters are easier to maintain externally than Engines or Extensions because they sit at the process boundary and don't touch internal APIs. The phased path:

1. Build your `axil-X` Adapter as its own crate, depending on `axil-core` (and optionally specific Extensions if your Adapter needs them as compile-time deps).
2. Ship a host binary that wires together `axil-core` + your Adapter + whichever Extensions are relevant to your users.
3. If the Adapter is broadly useful, submit it upstream for inclusion in the main workspace.
