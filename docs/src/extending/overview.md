# Extending Axil — The Three Tiers

Axil is built from a small core (`axil-core`) plus a fan of pluggable crates. To keep the architecture legible, the project distinguishes **three tiers** of extensibility, each with its own contract, conventions, and authoring guide.

| Tier | Name | What it is | Owns | 3rd-party? |
|---|---|---|---|---|
| 1 | **[Engine](engines.md)** | Storage substrate. Implements the `Plugin` trait, owns a companion file, gets `on_record_insert` / `on_record_update` / `on_record_delete` lifecycle hooks from the master coordinator. | A companion file (`*.axil.vec`, `*.axil.graph`, `*.axil.fts/`, `*.axil.ts`) | Possible, high bar — upstream-only in practice |
| 2 | **[Extension](extensions.md)** | Capability built on Engines. Owns prefixed tables in the core `.axil` file, optionally registers CLI subcommands, MCP tools, brain hooks, and drift-refresh logic. | Prefixed tables (`_dep_docs`, `_dep_manifests`, `_scip_aliases`, …) | **Yes — designed for it** |
| 3 | **[Adapter](adapters.md)** | Interface to the outside world. Translates an external protocol (CLI argv, MCP stdio, HTTP, AxilQL) to/from `Axil::query()`. | A protocol surface | **Yes — designed for it** |

The taxonomy is **the** organizing principle of the codebase. Every `axil-*` crate fits exactly one tier.

> **Naming note:** *Engine* is the canonical industry term for the substrate layer (SQLite/MySQL storage engines, Tantivy search engine, V8 JavaScript engine). In the trait API the same concept is spelled `Plugin` for historical compatibility — both words refer to the same thing.

## Crate classification

| Crate | Tier | Companion file / Tables |
|---|---|---|
| `axil-core` (`crates/axil-core/`) | Host (not a tier) | `*.axil` (core), traits, builder |
| `axil-vector` (`crates/engines/`) | Engine | `*.axil.vec` |
| `axil-graph` (`crates/engines/`) | Engine | `*.axil.graph` |
| `axil-fts` (`crates/engines/`) | Engine | `*.axil.fts/` |
| `axil-timeseries` (`crates/engines/`) | Engine | `*.axil.ts` |
| `axil-memory` (`crates/extensions/`) | Extension | agent memory patterns (TTL, supersede, recency) |
| `axil-indexer` (`crates/extensions/`) | Extension | project source indexer + section splitters |
| `axil-rerank` (`crates/extensions/`) | Extension | reranker (no tables — pure compute) |
| `axil-scip` (`crates/extensions/`) | Extension | `_scip_*`, `_entities`, `_entity_aliases` |
| `axil-docs` (`crates/extensions/`) | Extension *(reference impl)* | `_dep_manifests`, `_deps`, `_dep_docs` |
| `axil-workspace` (`crates/extensions/`) | Extension | `.axil-workspace.toml` + federation tables |
| `axil-cli` (`crates/adapters/`) | Adapter | (none — process boundary) |
| `axil-mcp` (`crates/adapters/`) | Adapter | (none — stdio JSON-RPC) |
| `axil-ql` (`crates/adapters/`) | Adapter | (none — query string → QueryBuilder) |

## Why three tiers?

Each tier has a different contract because each has a different risk profile and audience:

- **Engines** mutate the storage substrate. A bad Engine corrupts data. The contract is strict (lifecycle hooks, companion-file convention, ABI stability across all Engines), the audience is small (DB engine authors), and the path is upstream-only today.
- **Extensions** add capabilities on top of Engines using only the public `Axil` API. A bad Extension wastes tables or CPU but can't corrupt the substrate. The contract is advisory (a trait + naming conventions), the audience is medium (anyone who wants a new agent capability), and the path is third-party-friendly.
- **Adapters** translate protocols. A bad Adapter returns wrong results to one client but doesn't affect any other client or the substrate. The contract is small (translate to/from `Axil::query()`), the audience is large (anyone with a protocol they want Axil behind), and the path is third-party-friendly.

Putting all three in the same "plugin" bucket would mean the strictest contract (Engine) gets imposed on the loosest case (Adapter). Splitting them lets each tier have the right amount of structure.

## Quick decision tree

> **I want to add something to Axil. Which tier am I in?**

```
Do I need to store data in a new kind of index
(beyond k/v records, vectors, graph edges, FTS,
or time-series)?
│
├── Yes → Engine (Tier 1). See engines.md.
│         You'll implement Plugin + one of the index
│         traits, own a *.axil.<suffix> companion file,
│         and need master-coordinator coordination.
│
└── No → Am I exposing Axil to a new protocol
          (HTTP, gRPC, GraphQL, a new query language)?
          │
          ├── Yes → Adapter (Tier 3). See adapters.md.
          │         You'll translate the protocol to
          │         Axil::query() calls and surface every
          │         registered Extension's commands/tools.
          │
          └── No → Extension (Tier 2). See extensions.md.
                    You'll own prefixed tables, optionally
                    expose CLI/MCP surface, and integrate
                    with brain hooks and refresh.
```

## Authoring guides

- [Authoring Engines](engines.md) — Tier 1, the storage substrate.
- [Authoring Extensions](extensions.md) — Tier 2, capability extensions. `axil-docs` is the reference implementation.
- [Authoring Adapters](adapters.md) — Tier 3, protocol adapters. `axil-cli` and `axil-mcp` are the reference implementations.

## Stability — the 1.0 SPI

The extensibility surface is split into a **stable outer SPI** and an **unstable inner Engine API**, on purpose. That asymmetry is what lets the core add, drop, or swap storage Engines without breaking anyone building on the outer tiers.

| Surface | Stability | Who builds against it |
|---|---|---|
| `Extension` + its support types (`CliSurface`, `CliSubcommand`, `CliArg`, `McpSurface`, `McpTool`, `Hit`, `RefreshOpts`, `RefreshReport`, …) | **Stable** — semver-locked at 1.0; the structs third parties construct are `#[non_exhaustive]` with constructors/builders so they can grow additively | Extension authors (Tier 2) |
| `Adapter`, `Protocol`, `dispatch_cli` / `dispatch_mcp`, `compose_cli_surface` / `compose_mcp_surface`, the `Axil` builder + query API | **Stable** — semver-locked at 1.0 | Adapter authors (Tier 3) |
| `Plugin`, `VectorIndex`, `GraphIndex`, `SearchIndex`, `TimeSeriesIndex`, `TextEmbedder`, `Capability` | **Unstable** — no semver guarantee, may change in any release | Engine authors (Tier 1 — upstream-or-fork) |

The full stable set is enumerated in one place: the crate-level docs of `axil-core` (`lib.rs`).

## How to extend Axil — compile-time and runtime

There are now **two** ways to add an Extension, and you pick by whether you control the build:

**Compile-time (native, in-tree):**

- **One-line built-in registration.** `axil-bundle` is the single registry the CLI, the MCP server, and the workspace audit test all derive their Extension set from. Adding a built-in Extension is one line there, not three hand-wired sites.
- **Runtime enable/disable, no rebuild.** `axil extensions list` shows every compiled-in Extension and its state; `axil extensions disable <id>` / `enable <id>` toggle it via `[extensions] disabled` in `axil.toml`.
- **Zero CLI code.** A CLI-facing Extension's declared `CliSurface` is dispatched generically — no per-command code in `axil-cli`.
- **Clean Engine removal.** `axil compact --drop-engine <vector|graph|timeseries|fts>` deletes the orphaned companion file when an Engine is removed.

**Runtime (WASM, out-of-tree) — the `wasm-host` build:**

- **Drop-in plugins, no rebuild, no fork.** Build any Component-Model language to a `.wasm`, then `axil ext install foo.wasm`. It loads from `.axil/plugins/`, registers into the live database, and its commands/tools/boot-block work exactly like a native Extension — because a loaded plugin *is* "just another `dyn Extension`" (one `WasmExtension` shim across the WIT ABI, `axil:plugin@1.0.0`).
- **Sandboxed + capability-gated.** Ambient filesystem/network are denied by default (WASI off); CPU is bounded by Wasmtime fuel. A plugin is **deny-by-default** — it can run but cannot call back into Axil until the operator grants capabilities with `axil ext grant <plugin> <cap>` (`[plugins.<key>] capabilities` in `axil.toml`). Record writes are constrained to the plugin's declared table prefixes.
- **`axil ext install | list | remove | info | grant | revoke`.** One bad `.wasm` is quarantined (reported, never fatal). See [Authoring WASM plugins](wasm-plugins.md).

The `wasm-host` feature is **off by default** so the standard binary stays small (zero Wasmtime); build with `--features wasm-host` to opt in. Tier 1 (storage Engines) stays compile-time — Engines need direct master-coordinator access and are the storage hot path.

Load-time **ABI-version negotiation** (a precise error when a plugin's `axil:plugin@X.Y.Z` isn't one this host implements, instead of a raw link failure), a **compiled-module cache** (`.axil/plugins/.cache/`, ~16× faster repeat invocations — a deserialize instead of a recompile), and a **host-ABI conformance suite** (a real guest exercising every host import across the boundary) are in. Still ahead (Phase 22 polish): an ergonomic guest SDK (proc-macro) and a fuzz harness. None change the contract above.
