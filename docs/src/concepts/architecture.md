# Architecture

Axil is organized in layers, from interface down to storage:

```
┌─────────────────────────────────────────┐
│           Interface Layer               │
│  • CLI + Skills (primary for agents)    │
│  • AxilQL query language (verb-first)   │
│  • Embedded Rust lib                    │
│  • MCP Server (for non-CLI agents)      │
│  • HTTP API (optional)                  │
├─────────────────────────────────────────┤
│      LLM Intelligence (optional)        │
│  Path A: CLI + Skill — agent is the LLM │
│  Path B: LlmProvider trait — app gives  │
│          Axil an LLM callback           │
│  Path 0: No LLM — algorithmic (80%)    │
├─────────────────────────────────────────┤
│      Master Coordinator (Axil)          │
│  • Single entry: Axil::open(path)       │
│  • Owns all Engine storage lifecycle    │
│  • Routes mutations to all Engines      │
│  • Scoring: vector + graph + recency +  │
│    keyword + feedback → ranked recall   │
├─────────────────────────────────────────┤
│         Engine Trait System             │
│  • VectorEngine (HNSW) → *.axil.vec    │
│  • TextEmbedder (ONNX)  → model files  │
│  • GraphEngine (edges)  → *.axil.graph │
│  • FtsEngine (FTS)   → *.axil.fts/  │
├─────────────────────────────────────────┤
│         Core Storage Engine             │
│  • redb (embedded, ACID) → *.axil      │
│  • Records = typed docs with IDs        │
│  • Edges are records linking records    │
└─────────────────────────────────────────┘
```

## File layout

Each database is a set of companion files derived from the base path:

| File | Description |
|------|-------------|
| `*.axil` | Core storage (redb, ACID) |
| `*.axil.vec` | Vector index (HNSW) |
| `*.axil.graph` | Graph index (edges) |
| `*.axil.fts/` | Full-text search (Tantivy directory) |

`Axil::open()` is the master coordinator that creates and manages all files.

## Crate structure

| Crate | Tier | Purpose |
|-------|------|---------|
| `axil-core` | Host | Storage engine, records, query builder, trait surface for Engines/Extensions/Adapters |
| `axil-vector` | Engine | HNSW vector index + ONNX embedding |
| `axil-graph` | Engine | Graph storage (edges, traversal) |
| `axil-fts` | Engine | Full-text search (Tantivy wrapper) |
| `axil-timeseries` | Engine | Time-series index (range queries, bucketing) |
| `axil-memory` | Extension | Agent memory types (5 types + session) |
| `axil-docs` | Extension | Dependency documentation memory (Phase 16) |
| `axil-ql` | Adapter | AxilQL query language parser |
| `axil-mcp` | Adapter | MCP protocol server |
| `axil-cli` | Adapter | CLI binary |

> The three-tier model is documented in [Extending Axil](../extending/overview.md). "Engine" is the Tier-1 storage substrate, named the `Engine` trait in code.

## Record model

Every piece of data in Axil is a `Record`:

```rust
pub struct Record {
    pub id: RecordId,        // ULID (time-sorted, globally unique)
    pub table: String,       // Logical grouping
    pub data: Value,         // JSON payload
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

Records are stored in tables (like collections). Internal tables start with `_` (e.g., `_entities`, `_sessions`).

The `table` is the record's top-level category, and how you choose it (by
*function*, not topic) is what makes retrieval fast. See
[Memory Taxonomy](./memory-taxonomy.md) for the categorization model and the
`--type` facet filter.
