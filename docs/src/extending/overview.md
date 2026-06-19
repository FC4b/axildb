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

## What ships in 1.0 vs. what's next

The current model is **compile-time** — every Engine, Extension, and Adapter is a Cargo dependency baked into the binary at build time. Within that model, 1.0 ships:

- **One-line built-in registration.** `axil-bundle` is the single registry the CLI, the MCP server, and the workspace audit test all derive their Extension set from. Adding a built-in Extension is one line there, not three hand-wired sites.
- **Runtime enable/disable, no rebuild.** `axil extensions list` shows every compiled-in Extension and its state; `axil extensions disable <id>` / `enable <id>` toggle it via `[extensions] disabled` in `axil.toml`. A disabled Extension is skipped at registration — its CLI/MCP surface and boot block vanish until re-enabled.
- **Clean Engine removal.** `axil compact --drop-engine <vector|graph|timeseries|fts>` deletes the orphaned companion file left behind when an Engine is removed (rebuilt without its feature, or disabled).

Still ahead (tracked in the Phase 21–22 extensibility plan):

- **Dynamic CLI from `CliSurface`** — a CLI-facing Extension needing **zero** code in `axil-cli`: its declared surface is appended to the argument parser at runtime.
- **WASM runtime Extensions / Adapters** — load Extension and Adapter code from a `.wasm` component at runtime (`axil ext install foo.wasm`), sandboxed and ABI-versioned. Tier 1 stays compile-time because storage Engines need direct access to the master coordinator.
- **Capability sandboxing** — today every Extension and Adapter has full `Axil` access. A granted-capability model would let a loaded plugin declare (and the host enforce) exactly which host calls it may make.

Until WASM lands, the practical path for an **out-of-tree** third party is:

1. Write your Extension or Adapter as its own `axil-X` crate against the stable SPI above.
2. Register it in a host binary — fork `axil-cli`, or build your own host that calls `Axil::open(...).with_extension(...)`.
3. Submit it upstream for inclusion in the bundle if the work is generally useful.
