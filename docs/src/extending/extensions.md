# Authoring an Extension (Tier 2)

An **Extension** is a capability built on top of Axil's Engines. Extensions don't own a companion file — they live in the core `.axil` database with their own prefixed tables. They optionally expose CLI subcommands, MCP tools, brain hooks, and drift-refresh logic, and they're the primary third-party extensibility surface in Axil.

This guide walks through writing an Extension end-to-end, using [`axil-docs`](../../../crates/axil-docs/) (the dependency-doc memory) as the reference implementation.

## When to write an Extension

You're in the right tier if you want to:

- Ingest some external data (lockfiles, code-graph indexes, document corpora) and store it in Axil.
- Add a new agent command (`axil deps`, `axil ingest-scip`).
- Surface a new kind of recall (`axil dep-docs`, `axil code-search`).
- Hook into agent boot or file-edit events (PreToolUse, PostToolUse, `axil boot`).

You're in the wrong tier if you want to:

- Add a new fundamental index type → [Engine](engines.md).
- Expose Axil over a new protocol (HTTP, gRPC, GraphQL) → [Adapter](adapters.md).

## The `Extension` trait

```rust
// crates/axil-core/src/extension.rs

use crate::{Axil, Result, Record};
use std::path::Path;

pub trait Extension: Send + Sync {
    /// Stable extension id, kebab-case, must match the crate name minus
    /// the `axil-` prefix (e.g. `axil-docs` → `"docs"`).
    fn id(&self) -> &str;

    /// Human-readable display name (used in `axil status`, `axil boot`).
    fn display_name(&self) -> &str { self.id() }

    /// Table-name prefixes this extension owns. The master coordinator
    /// enforces that no two extensions claim overlapping prefixes.
    /// Convention: all extension tables start with `_<id>_*`.
    fn table_prefixes(&self) -> &[&str] { &[] }

    /// Optional CLI subcommands. Returned to `axil-cli` at registration.
    fn cli_commands(&self) -> Option<CliSurface> { None }

    /// Optional MCP tools. Returned to `axil-mcp` at registration.
    fn mcp_tools(&self) -> Option<McpSurface> { None }

    /// Optional `axil boot` block. Called once per boot; the returned
    /// string (if any) is appended to the boot summary.
    fn boot_block(&self, _db: &Axil) -> Option<String> { None }

    /// Optional drift/refresh entry point. Called by `axil refresh`,
    /// PostToolUse brain hooks, and any agent-initiated refresh.
    fn refresh(&self, _db: &Axil, _opts: RefreshOpts) -> Result<RefreshReport> {
        Ok(RefreshReport::default())
    }

    /// Optional `recall-for-file` contribution. Called when the agent
    /// is about to edit a file; the extension may surface relevant
    /// records.
    fn recall_for_file(&self, _db: &Axil, _path: &Path) -> Result<Vec<Hit>> {
        Ok(vec![])
    }
}
```

Every method except `id()` has a default implementation. Implement only what your Extension needs.

## Conventions you must follow

### Crate naming

Your crate is `axil-<name>`. The `<name>` is your Extension id. For the dependency-doc extension: `axil-docs` → id `"docs"`.

### Table prefix

All tables your Extension creates must start with `_<id>_`. Examples from the in-tree reference Extensions:

| Extension | Table prefix | Tables |
|---|---|---|
| `axil-docs` | `_dep_` *(legacy — predates this convention)* | `_dep_manifests`, `_deps`, `_dep_docs` |
| `axil-scip` | `_scip_` | `_scip_aliases` (and shared `_entities`) |
| `axil-indexer` | `_idx_` | `_idx_code_proxies`, `_idx_code_refs` |

The master coordinator panics at builder time if two registered Extensions claim overlapping prefixes. Pick a unique one.

> **Note on `axil-docs`:** Its prefix is `_dep_` (without the `_docs_` we'd recommend today) because it was built before the convention was formalized in Phase 17. New Extensions should use `_<id>_`.

### Cargo-feature gating

Your Extension is a Cargo feature on `axil-cli`. Default-on for generally useful Extensions, default-off for niche or experimental ones:

```toml
# crates/axil-cli/Cargo.toml
[features]
default = ["vector", "graph", "fts", "deps", "scip"]
myext = ["dep:axil-myext"]
full = ["vector", "graph", "fts", "deps", "scip", "myext"]
```

The Extension crate itself should have minimal default features so it can be enabled cleanly:

```toml
# crates/axil-myext/Cargo.toml
[dependencies]
axil-core = { path = "../axil-core" }

[features]
default = []
```

### Error type

Use `thiserror` for your error enum. Always include a `Db` variant that wraps `axil_core::AxilError`, so callers can propagate cleanly:

```rust
#[derive(Debug, thiserror::Error)]
pub enum MyExtError {
    #[error("{context}: {source}")]
    Parse {
        context: String,
        #[source] source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("database error: {0}")]
    Db(#[from] axil_core::AxilError),
}
```

## Hello-world Extension

A minimal Extension that adds an `axil hello` command and stores greetings in a `_hello_greetings` table:

```rust
// crates/axil-hello/src/lib.rs
use axil_core::{Axil, Extension, CliSurface, Result};

pub struct HelloExtension;

impl Extension for HelloExtension {
    fn id(&self) -> &str { "hello" }
    fn display_name(&self) -> &str { "Hello world" }
    fn table_prefixes(&self) -> &[&str] { &["_hello_"] }

    fn cli_commands(&self) -> Option<CliSurface> {
        Some(CliSurface::subcommand("hello")
            .about("Greet someone, store the greeting")
            .arg("name", "Who to greet"))
    }

    fn boot_block(&self, db: &Axil) -> Option<String> {
        let count = db.list("_hello_greetings").ok()?.len();
        Some(format!("👋 hello: {count} greetings stored"))
    }
}

pub fn greet(db: &Axil, name: &str) -> Result<()> {
    db.insert("_hello_greetings", serde_json::json!({
        "name": name,
        "at": chrono::Utc::now().to_rfc3339(),
    }))?;
    println!("Hello, {name}!");
    Ok(())
}
```

That's the whole contract. The Adapter (`axil-cli`) discovers your Extension at registration time, surfaces the `hello` subcommand, and dispatches to your code.

## The `axil-docs` reference implementation

Look at `crates/axil-docs/` for a production-grade Extension that exercises the full surface. The relevant patterns:

| Pattern | Where in `axil-docs` |
|---|---|
| Crate layout (modules per concern) | `manifest.rs`, `resolve.rs`, `local.rs`, `ingest.rs`, `refresh.rs`, `query.rs`, `web.rs` |
| Owned tables | `_dep_manifests`, `_deps`, `_dep_docs` (defined as constants in `ingest.rs`) |
| Drift detection | `refresh.rs` — content-hash the manifest + lockfile, store in `_dep_manifests`, re-ingest when the hash changes |
| Background refresh via brain hook | PostToolUse hook in `.claude/settings.json` calls `axil deps refresh --if-stale` on manifest edits |
| `axil boot` integration | The `dep_docs_freshness` block surfaces stale-doc warnings on agent boot |
| `recall-for-file` integration | A per-file pass surfaces dep docs for the file's imports |
| MCP parity | The `dep_docs` and `deps_status` tools mirror the CLI surface |
| Version history | A version bump archives old chunks instead of deleting (`archived: true`) and links via `superseded_by` |
| Feature gating | Default-on in `axil-cli` `default` and `full`; absent in `core` |

Read [`crates/axil-docs/src/lib.rs`](../../../crates/axil-docs/src/lib.rs) end-to-end before authoring a non-trivial Extension. It's the canonical shape.

## Drift / refresh pattern

Most Extensions that ingest external state need a refresh path. The pattern, copied from `axil-docs`:

1. Hash the external source(s) (manifest file, source tree, config) and store the hash in a `_<id>_state` table.
2. On `refresh()`, recompute the hash and compare. If unchanged, return early — no work.
3. If changed, recompute only what changed. Use content-hash dedup so unchanged sub-units (chunks, entities) skip re-embedding / re-indexing.
4. Mark old data as superseded rather than deleting if the agent might still need it (migration questions, history queries).

The `axil_docs::refresh::manifest_drift` function is the canonical implementation; copy its shape.

## Brain-hook integration

Extensions can opt into the agent brain (Phase 11) by:

1. Implementing `boot_block(db)` — the returned string is appended to `axil boot` output.
2. Implementing `recall_for_file(db, path)` — returned hits surface when the agent is about to edit a file.
3. Documenting any PostToolUse hooks the Extension expects to be installed in `.claude/settings.json` (e.g. `axil deps refresh --if-stale` on manifest writes). The hook itself is configured by the project, not the Extension — the Extension just provides the command.

## Cross-Extension graph edges

Extensions can write graph edges that bridge each other's records. For example, `axil-docs` writes `superseded_by` edges between dep-version rows, and could optionally write `documents` edges from memory records to dep-doc chunks.

Use the regular `db.relate(from, edge_type, to)` API. Edge types are not namespaced today — pick descriptive names (`superseded_by`, not `s_by`).

## Testing

Required:

- **Unit tests** in `crates/axil-myext/tests/` covering each module.
- **End-to-end smoke test** that exercises the full pipeline (ingest → store → query) through `Axil::open(…).with_extension(MyExtension)`.

Recommended:

- **Fixture-based regression test** mirroring `tests/fixtures/code-recall/` — small synthetic dataset with expected hits, gate script that fails CI on quality regression.

## What you don't have to do

- **No companion file.** Use the core storage — that's the whole point of being an Extension.
- **No master-coordinator coordination.** You don't see insert/update/delete hooks. If you need those, you're writing an Engine, not an Extension.
- **No ABI stability.** Today every Extension is a Cargo dependency, recompiled together with `axil-core`. You can use any internal API.
- **No registry / install step.** Extensions are linked at build time via Cargo features.

## Path to upstream

Open-source Extensions can be:

1. **Maintained externally** — your `axil-myext` crate lives in its own repo, depends on `axil-core` from crates.io, and consumers fork `axil-cli` to wire it in.
2. **Submitted upstream** — if generally useful, send a PR adding your crate to the workspace. The criteria are roughly: doesn't depend on niche external services, broadly applicable, well-tested, follows the conventions above.

The phased path: prove the Extension out-of-tree first, then upstream if the value is clear.
