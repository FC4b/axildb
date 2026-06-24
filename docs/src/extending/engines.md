# Authoring an Engine (Tier 1)

> **Naming reminder:** This tier is an **Engine** in both the docs and the code (the `Engine` trait).

An **Engine** is the deepest extensibility tier in Axil — a storage substrate that implements the `Engine` trait, owns a companion file next to the core `.axil` database, and participates in the master coordinator's insert/update/delete lifecycle. Engines are how Axil supports vector search, graph edges, FTS, and time-series queries side-by-side in one binary.

This guide covers what writing a new Engine looks like. Read [the three-tier overview](overview.md) first if you haven't decided which tier you're actually in — most new functionality is an [Extension](extensions.md), not an Engine.

## Before you start

**Writing a new Engine is the highest-bar path in Axil.** Engines have direct access to the master coordinator's lifecycle, can corrupt the database if they get it wrong, and must coordinate with every other Engine on companion-file conventions. Build an Extension instead if any of these are true:

- You can express your storage need with the existing core record store, vector index, graph, FTS, or time-series.
- You don't need a separate file for write-concurrency or independent I/O.
- You're adding a new query pattern, not a new index type.

**Today the realistic path for a new Engine is upstream contribution.** There is no stable ABI for out-of-tree Engines. The trait surface is `pub` and you can write an Engine in your own crate, but `axil-cli` and the master coordinator are compiled with a fixed set of Engines today. If your Engine is broadly useful, the path is a PR to the main repo; if it's narrow, fork `axil-cli` and depend on your crate.

## The trait surface

Every Engine implements `Engine` plus zero or one index trait. The index traits live in `crates/axil-core/src/plugin.rs`.

```rust
use axil_core::{Engine, Capability, Record, RecordId, Result};

pub struct MyEngine { /* … */ }

impl Engine for MyEngine {
    fn name(&self) -> &str { "myengine" }

    fn capabilities(&self) -> Vec<Capability> {
        vec![/* declare what this engine provides */]
    }

    fn on_record_insert(&self, record: &Record) -> Result<()> {
        // Update internal index when a record is inserted into core.
        // Called by the master coordinator after the core write commits.
        Ok(())
    }

    fn on_record_update(&self, record: &Record) -> Result<()> {
        // Optional. Default is no-op. Override if you track mutable state.
        Ok(())
    }

    fn on_record_delete(&self, id: &RecordId) -> Result<()> {
        // Remove from your index. Must be idempotent.
        Ok(())
    }
}
```

If you're adding a new *kind* of index (not vector / graph / FTS / time-series), define a new trait alongside `VectorIndex`, `GraphIndex`, etc.:

```rust
pub trait MyIndex: Engine {
    fn add(&self, id: RecordId, payload: &MyPayload) -> Result<()>;
    fn query(&self, q: &MyQuery, limit: usize) -> Result<Vec<(RecordId, f32)>>;
}
```

## Companion file convention

Every Engine owns a single file (or directory) derived from the base database path by appending a suffix:

| Suffix | Engine | Notes |
|---|---|---|
| `.vec` | Vector | Single file (HNSW serialized) |
| `.graph` | Graph | Single redb file |
| `.fts/` | FTS | Directory (tantivy segments) |
| `.ts` | Time-series | Single redb file |

Pick a *new*, unique suffix. Update `COMPANION_SUFFIXES` in [crates/axil-core/src/db.rs](../../../crates/axil-core/src/db.rs) so `axil doctor` and database-discovery code recognize your companion file.

Open your companion file from a builder extension trait that mirrors the existing pattern:

```rust
// crates/axil-myengine/src/lib.rs
use axil_core::AxilBuilder;

pub trait AxilBuilderMyEngineExt {
    fn with_my_engine(self) -> Result<Self> where Self: Sized;
}

impl AxilBuilderMyEngineExt for AxilBuilder {
    fn with_my_engine(self) -> Result<Self> {
        let companion = axil_core::companion_path(self.path(), ".myengine");
        let engine = MyEngine::open(&companion)?;
        Ok(self.with_engine(Box::new(engine)))
    }
}
```

## Lifecycle contract

The master coordinator routes every mutation in this order:

1. Write to core storage (redb transaction commits).
2. For each registered Engine, call `on_record_insert` / `on_record_update` / `on_record_delete`.
3. Return success to the caller.

Implications:

- **Engine writes happen *after* the core write commits.** If your Engine crashes mid-write, the core has the record but your index doesn't. Make `on_record_insert` idempotent (retrying must be safe) and provide a rebuild path for crash recovery.
- **Engines see records in the order they were inserted into core.** Don't assume your Engine sees writes in causal order across processes — for that, use the time-series Engine's bucket ordering.
- **Failure is propagated.** If your `on_record_insert` returns `Err`, the master coordinator surfaces the error to the caller. The core write has already committed, so the caller sees a partial-write error. Design your Engine so that the *only* way `on_record_insert` fails is corruption of your companion file.

## Vector re-embedding is not automatic

If your Engine holds vectors derived from text fields (like `axil-vector` does), be aware that the master coordinator **does not** automatically re-embed on `on_record_update`. Callers must explicitly call `db.embed_field()` after updating records with embedded fields. This is intentional — Path A (the agent orchestrates the embed pipeline) lets the agent batch embeds across many updates.

If your Engine has the same property, document it the same way.

## Cargo-feature gating

Engines are opt-in via Cargo features on the umbrella crate or on `axil-cli`:

```toml
# crates/axil-cli/Cargo.toml
[features]
default = ["vector", "graph", "fts", "deps"]
myengine = ["dep:axil-myengine"]
full = ["vector", "graph", "fts", "deps", "myengine"]
```

A minimal `core` build should still compile without your Engine. Test this by running `cargo check -p axildb --no-default-features --features core`.

## Diagnostics integration

`axil doctor` and `axil status` query each registered Engine for health and stats. Implement these accessors:

```rust
impl MyEngine {
    pub fn stats(&self) -> serde_json::Value { /* records, file size, etc. */ }
    pub fn health_check(&self) -> Vec<CheckResult> { /* corruption probes */ }
}
```

The master coordinator picks them up automatically when your Engine is registered through `with_engine()`.

## Testing

Every Engine ships with:

- **Unit tests** in the crate (`crates/axil-myengine/tests/`).
- **Criterion benchmark** in `crates/axil-myengine/benches/` covering the hot path (add, query, delete).
- **Integration test** in `crates/axil-tests/` that exercises insert → engine update → query end-to-end through `Axil::open(…).with_my_engine()`.

The benchmark must run as part of `scripts/bench-check.sh` so regressions >5% break CI.

## Reference implementations

The four built-in Engines are the canonical examples:

- **Vector**: [crates/axil-vector/](../../../crates/axil-vector/) — `instant-distance` HNSW + ONNX embedding.
- **Graph**: [crates/axil-graph/](../../../crates/axil-graph/) — pure-Rust edge storage in redb.
- **FTS**: [crates/axil-fts/](../../../crates/axil-fts/) — tantivy wrapper with code tokenizer, fuzzy matching, field boosting.
- **Time-series**: [crates/axil-timeseries/](../../../crates/axil-timeseries/) — redb-backed bucketed ranges.

Start by reading `axil-fts` if you want the simplest end-to-end example, or `axil-vector` if you need the embedding-pipeline pattern.

## When *not* to write an Engine

If your storage need is any of these, you want an [Extension](extensions.md), not an Engine:

- "I need a new table for tracking X" → Extension with its own `_x_…` prefixed tables in core storage.
- "I need to ingest Y from disk and query it" → Extension, e.g. how `axil-docs` ingests dependency docs.
- "I need to surface a new CLI subcommand or MCP tool" → Extension's CLI/MCP surface.
- "I need to expose Axil over HTTP / GraphQL / gRPC" → [Adapter](adapters.md).

Engines exist only when you need a **fundamentally different index type** that doesn't fit the existing primitives.
