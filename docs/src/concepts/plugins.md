# Engines (Storage Plugins)

> **Naming note:** Axil calls this tier an **Engine** — the standard term SQLite, MySQL, and Tantivy use for the same architectural layer — and the trait is named `Engine` in code to match.

Axil's storage substrate is built from **Engines** — compile-time modules that implement the `Engine` trait, own a companion file next to the core `.axil` database, and receive insert/update/delete lifecycle hooks from the master coordinator.

Engines are one of three extensibility tiers in Axil. For the full taxonomy and where Engines fit alongside [Extensions](../extending/extensions.md) and [Adapters](../extending/adapters.md), see the [extending overview](../extending/overview.md).

## Engine traits

```rust
pub trait Engine: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
    fn on_record_insert(&self, record: &Record) -> Result<()>;
    fn on_record_update(&self, record: &Record) -> Result<()> { Ok(()) }
    fn on_record_delete(&self, id: &RecordId) -> Result<()>;
}

pub trait VectorIndex: Engine {
    fn add(&self, id: RecordId, vector: &[f32]) -> Result<()>;
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>>;
    fn count(&self) -> usize;
}

pub trait TextEmbedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub trait GraphIndex: Engine {
    fn relate(&self, from: RecordId, edge_type: &str, to: RecordId, props: Value) -> Result<RecordId>;
    fn traverse(&self, start: RecordId, path: &[TraversalStep]) -> Result<Vec<Record>>;
    fn neighbors(&self, id: RecordId, edge_type: Option<&str>, direction: Direction) -> Result<Vec<Record>>;
}

pub trait SearchIndex: Engine {
    fn index_text(&self, id: RecordId, field: &str, text: &str) -> Result<()>;
    fn search_text(&self, query: &str, limit: usize) -> Result<Vec<(RecordId, f32)>>;
}
```

`TextEmbedder` is intentionally separate from `VectorIndex` so an ANN-only Engine doesn't need to implement embedding, and embedding can be configured independently (local ONNX vs. external API).

## Using Engines

```rust
use axil_core::Axil;
use axil_vector::{models::EmbeddingModel, AxilBuilderVectorExt};
use axil_graph::AxilBuilderGraphExt;
use axil_fts::AxilBuilderFtsExt;

let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?
    .with_graph_engine()?
    .with_fts_engine()?
    .build()?;
```

Each Engine is enabled by a Cargo feature on `axil-cli` (or by depending on the crate directly in an embedded build).

## Engines shipped today

| Engine | Crate | Feature flag | Companion file | Backed by |
|--------|-------|-------------|----------------|-----------|
| Vector | `axil-vector` | `vector` / `embed` | `*.axil.vec` | `hnsw_rs` (incremental HNSW) + `ort` (ONNX) |
| Graph | `axil-graph` | `graph` | `*.axil.graph` | redb (built-in) |
| FTS | `axil-fts` | `fts` | `*.axil.fts/` | `tantivy` |
| Time-series | `axil-timeseries` | built-in | `*.axil.ts` | redb |

## Authoring a new Engine

Writing a new Engine is the highest-bar extensibility path in Axil. An Engine must:

1. Implement the base `Engine` trait plus one of the index traits (`VectorIndex`, `GraphIndex`, `SearchIndex`, `TimeSeriesIndex`) or define its own.
2. Manage a companion file using the `*.axil.<suffix>` naming convention.
3. Respect the master-coordinator lifecycle: insert/update/delete hooks must be idempotent and tolerant of out-of-order calls.
4. Ship a builder-extension trait (e.g. `AxilBuilderVectorExt`) so the Engine attaches via `Axil::open(…)` cleanly.

For the full authoring walkthrough, see [Authoring Engines](../extending/engines.md).

> **Practical note:** Today the realistic path for a new Engine is upstream contribution. There's no stable ABI for out-of-tree storage Engines yet — that's tracked as future work (see [extending overview](../extending/overview.md#future-work)).
