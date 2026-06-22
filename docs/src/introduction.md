# Axil

**Cognitive memory for AI agents. One binary. No LLM required.**

Axil is a lightweight, embeddable cognitive memory system built in Rust, purpose-built for AI agents. Your agent learns, remembers, synthesizes, and forgets — automatically.

Under the hood: vector search, knowledge graph, full-text search, and time-series queries, all in a single binary with a tiered extensibility model — Engines (storage substrates), Extensions (capabilities), and Adapters (protocols). See [Extending Axil](../extending/overview.md) for the full taxonomy.

## Why Axil?

Current AI agent memory solutions require external databases, LLM calls for basic operations, or are too heavy for embedded use. Axil fills the gap:

- **One binary** — no PostgreSQL, no Neo4j, no external services
- **No LLM required** — algorithmic entity extraction, consolidation, and inference
- **Embeddable** — use as a Rust library or standalone CLI
- **Tiered extensibility** — Engines (vector, graph, FTS, time-series), Extensions (memory, docs, scip, …), Adapters (CLI, MCP, AxilQL) — all opt-in via Cargo features
- **Agent-native** — 5 memory types, session lifecycle, auto-importance, decay
- **Source-available** — free for noncommercial use; commercial license available

## Key Features

| Feature | Description |
|---------|-------------|
| Vector search | HNSW index with local ONNX embeddings (BGE, Nomic) |
| Knowledge graph | Entity extraction, relationships, traversal |
| Full-text search | Tantivy-powered with fuzzy matching and snippets |
| 5 memory types | Working, semantic, episodic, procedural, preference |
| Auto-importance | Score records by entity density and structure |
| Memory decay | Time-based forgetting with configurable half-life |
| Belief system | High-level understanding from accumulated facts |
| Context-aware push | Proactive memory surfacing |
| AxilQL | Verb-first query language |
| MCP server | Model Context Protocol for agent integration |

## Quick Example

```bash
# Install
cargo install axildb

# Store a memory
axil init ./memory.axil
axil --db ./memory.axil store sessions '{"summary": "Fixed auth timeout"}'

# Recall similar memories
axil --db ./memory.axil recall "authentication issues" --top-k 5

# Run diagnostics
axil --db ./memory.axil doctor
```

```rust
use axil_core::Axil;
use axil_vector::{models::EmbeddingModel, AxilBuilderVectorExt};
use axil_graph::AxilBuilderGraphExt;
use axil_fts::AxilBuilderFtsExt;

let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?
    .with_graph_plugin()?
    .with_fts_plugin()?
    .build()?;

let session = db.insert("sessions", json!({
    "summary": "Fixed auth timeout bug",
}))?;

db.embed_field(&session.id, "summary")?;

let results = db.similar_to("auth error", 5)?;
```
