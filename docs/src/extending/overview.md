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

## Future work

The current model is **compile-time** — every Engine, Extension, and Adapter is a Cargo dependency baked into the binary at build time. There is no dynamic loading, no plugin registry, and no stable ABI for out-of-tree storage Engines.

Tracked future work:

- **WASM runtime Extensions / Adapters** — load Extension and Adapter code from a `.wasm` artifact at runtime. Tier 1 stays compile-time because storage Engines need direct access to the master coordinator.
- **Extension/Adapter registry** — a package-manager-style discovery surface (`axil ext install …`) once there's enough third-party code to make discovery a problem.
- **Capability sandboxing** — today, every Extension and Adapter has full `Axil` access. A future capability model would let Extensions declare the table prefixes / CLI surface / MCP tools they need, and the host would enforce it.

Until those land, the practical path for a third party is:

1. Write your Extension or Adapter as its own `axil-X` crate.
2. Fork `axil-cli` (or your host binary) and add your crate as a dependency.
3. Submit the upstream for inclusion if the work is generally useful.
