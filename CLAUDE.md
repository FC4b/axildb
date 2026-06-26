# Axil - CLAUDE.md

## Project Overview

**Axil** is a lightweight, embeddable cognitive memory system built in Rust, purpose-built for AI agents. Your agent learns, remembers, synthesizes, and forgets — automatically. Under the hood: vector search, knowledge graph, full-text search, and time-series queries, all in a single binary with a plugin architecture.

**Tagline:** *"Cognitive memory for AI agents. One binary. No LLM required."*

> **Storage model:** Core data lives in a single `.axil` file (redb, ACID). Plugins that need independent I/O (e.g. vector search) use companion files (`.axil.vec`) in the same directory for write-concurrency — similar to how SQLite uses WAL/SHM files. All files share a predictable naming convention and are managed as one logical database.

## Why This Exists

Current AI agent memory solutions have gaps:
- **Memvid**: Closest match (Rust, single-file, local embeddings) but NO knowledge graph, NO entity extraction, NO memory types, NO consolidation/inference. Smart doc store, not structured memory.
- **SurrealDB/Spectron**: Heavy (~50MB), BSL license, Spectron still unreleased as of April 2026
- **Hindsight**: Best benchmark scores (91.4% LongMemEval) but requires PostgreSQL + LLM, ~200MB+
- **HelixDB**: Rust + graph + vector but server-first (not embeddable), AGPL license, no plugin architecture
- **Mem0**: Python middleware requiring LLM + 3 external databases just to store a memory (51.6k stars)
- **Letta/MemGPT**: Full agent runtime, replaces your stack instead of plugging in
- **Zep/Graphiti**: Temporal KG, but requires Neo4j, Community Edition deprecated

Axil fills the gap: one Rust binary, cognitive memory with structured agent cognition (graph + 5 memory types + auto-importance + decay + beliefs + inference), no LLM required.

## Competitive Position

### Direct Competitors
- **Memvid** (Rust, 13.7k stars, Apache 2.0) — single-file .mv2, BM25+vector, multi-modal. Closest architecturally but NO graph, NO entity extraction, NO memory types, NO knowledge consolidation. Smart doc store, not structured memory.
- **HelixDB** (YC-backed, Rust, graph+vector) — server-first, not embeddable, AGPL, no plugin arch
- **SurrealDB/Spectron** (Rust, multi-model) — heavy (~50MB), BSL license, Spectron still unreleased as of April 2026
- **Hindsight** (by Vectorize.io, MIT) — 4-strategy parallel retrieval, 91.4% LongMemEval. But requires PostgreSQL + LLM.
- **CortexaDB** (Rust, embedded, AI memory) — early stage, limited features

### Memory Layer Competitors
- **Mem0** (Python, 51.6k stars) — requires LLM + external DBs, not a database itself
- **Zep/Graphiti** — temporal knowledge graph, strongest bi-temporal. Community Edition deprecated.
- **Letta** — full agent runtime, not a memory layer
- **Cognee** — open-source KG + vector, entity extraction built-in

### Axil's Unique Position
No existing solution combines: embeddable, Rust-native, plugin-based, knowledge graph + 5 memory types + entity extraction + inference, CLI-first, token-optimized, no LLM required, source-available (free for noncommercial use).

| Feature | Axil | Memvid | HelixDB | SurrealDB | Mem0 | Hindsight |
|---------|------|--------|---------|-----------|------|-----------|
| Embeddable (no server) | ✅ | ✅ (.mv2) | ❌ | Optional | ❌ | ❌ (PG) |
| Knowledge graph | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| 5 memory types | ✅ | ❌ | ❌ | Advertised | ❌ | Partial |
| Entity extraction | ✅ | ❌ | ❌ | Advertised | ✅ (Pro) | ✅ |
| Knowledge consolidation | ✅ | ❌ | ❌ | Advertised | ❌ | ✅ |
| Graph inference | ✅ | ❌ | ❌ | Advertised | ❌ | ❌ |
| Built-in local embedding | ✅ (BGE) | ✅ (BGE) | ✅ | ❌ | Via LLM | Via LLM |
| Token optimization | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Query explanation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Built-in diagnostics | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Relevance feedback | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Auto-linking (no LLM) | ✅ | ❌ | ❌ | Via LLM | ❌ | Via LLM |
| Memory consolidation (no LLM) | ✅ | ❌ | ❌ | Via LLM | ❌ | Via LLM |
| Auto-importance scoring | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Memory decay (active forgetting) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Belief system | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Context-aware push | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Auto-capture | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Requires LLM | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| License | PolyForm NC | Apache 2.0 | AGPL-3.0 | BSL 1.1 | Apache 2.0 | MIT |
| Binary size target | ~5-10MB | ~10-20MB | ~50MB+ | ~50MB | N/A | ~200MB+ |

## Architecture

> **Glossary (Phase 17):** Axil has three extensibility tiers — **Engine** (Tier 1, storage substrate; implements the `Engine` trait, owns a `*.axil.<suffix>` companion file), **Extension** (Tier 2, capability built on Engines; owns prefixed tables in the core `.axil`), **Adapter** (Tier 3, protocol surface to the outside world; no storage). Code and docs agree: the Tier-1 trait is `Engine`; the `AxilError::Plugin` error variant and the WASM `axil:plugin` ABI are unrelated and keep their names. See [docs/src/extending/overview.md](docs/src/extending/overview.md) for the full taxonomy.

```
┌─────────────────────────────────────────┐
│        Adapter Layer (Tier 3)           │
│  • axil-cli — CLI + Skills (primary)    │
│  • axil-mcp — MCP server (stdio)        │
│  • axil-ql  — AxilQL query language     │
│  • Embedded Rust lib (direct API)       │
│  • HTTP API (future)                    │
├─────────────────────────────────────────┤
│      Extension Layer (Tier 2)           │
│  • axil-docs — dependency-doc memory    │
│  • axil-scip — code-graph ingest        │
│  • axil-memory / indexer / rerank /     │
│    workspace — capabilities             │
│  Owns prefixed tables in core .axil.    │
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
│         Engine Layer (Tier 1)           │
│         (Plugin trait surface)          │
│  • VectorEngine (HNSW) → *.axil.vec    │
│  • TextEmbedder (ONNX)  → model files  │
│  • GraphEngine (edges)  → *.axil.graph │
│  • FtsEngine (FTS)   → *.axil.fts/  │
│  • TimeSeriesEngine     → *.axil.ts    │
├─────────────────────────────────────────┤
│         Core Storage                    │
│  • redb (embedded, ACID) → *.axil      │
│  • Records = typed docs with IDs        │
│  • Edges are records linking records    │
└─────────────────────────────────────────┘
```

> **File layout:** Each database is a set of companion files derived from the base path:
> `*.axil` (core), `*.axil.vec` (vectors), `*.axil.graph` (graph), `*.axil.fts/` (FTS).
> `Axil::open()` is the master that creates and coordinates all of them.

## Intelligence Design

Axil gets smarter without requiring an LLM. Two paths for adding LLM intelligence when desired:

### Path A: CLI + Skill (primary — for Claude Code and agent frameworks)

The agent IS the LLM. It orchestrates the pipeline via CLI commands. Axil stays dumb and fast.

```
/axil-store "Fixed auth timeout bug"
  → Claude extracts entities: ["auth", "timeout", "login_flow"]
  → axil insert ./memory.axil sessions '{"summary": "...", "entities": [...]}'
  → axil embed ./memory.axil $ID summary
  → axil relate ./memory.axil $ID "mentions" $AUTH_ENTITY_ID
```

### Path B: LlmProvider trait (for Rust library users without an agent)

```rust
let db = Axil::open("./memory.axil")
    .with_vector(384)
    .with_llm(Box::new(HttpLlm::new(endpoint, api_key, model)))
    .build()?;
// Axil calls LLM internally for entity extraction, consolidation
```

### Intelligence levels (all work without LLM)

| Level | Feature | LLM boost |
|-------|---------|-----------|
| 1 | Semantic search (vector similarity) | Not needed |
| 2 | Auto-routing (vector + time + graph in one query) | Not needed |
| 3 | Context-aware recall (recency + project + graph boost) | Not needed |
| 4 | Auto-linking (entity extraction, co-occurrence) | Pattern-based → LLM-enhanced |
| 5 | Memory consolidation (contradiction detection, superseding) | Template-based → LLM summaries |
| 6 | Query explanation (`explain`, `--profile`, bottleneck detection) | Not needed |
| 7 | Relevance feedback (learn from usage patterns) | Not needed |
| 8 | Auto-importance scoring (entity density, structural markers) | Not needed |
| 9 | Memory decay (time-based forgetting with half-life) | Not needed |
| 10 | Belief system (agent's high-level understanding) | Not needed |
| 11 | Context-aware push (proactive memory surfacing) | Not needed |
| 12 | Auto-capture (extract knowledge from actions) | Not needed |

## Tech Stack

- **Language**: Rust (2021 edition)
- **Core Storage**: `redb` (embedded, ACID, pure Rust). Core data in `.axil`, plugin data in companion files (`.axil.vec` etc.)
- **Vector Search**: `instant-distance` or `usearch` (HNSW)
- **Embeddings**: `ort` (ONNX Runtime) with bge-small-en-v1.5 (default), bge-base-en-v1.5, nomic-embed-text-v1.5 (configurable)
- **Full-Text Search**: `tantivy`
- **Serialization**: `serde` + `serde_json`
- **Graph**: Custom implementation (edges as records)
- **MCP Server**: `rmcp` or custom JSON-RPC over stdio
- **CLI**: `clap`
- **Async**: `tokio`
- **Observability**: `opentelemetry` + `opentelemetry-otlp` (optional `otel` feature, gRPC/tonic)
- **Benchmarks**: `criterion` (per-plugin + combined suite)

## Plugin System Design

Compile-time plugins via Cargo features (v1), WASM runtime plugins (future):

```toml
[features]
default = ["core"]
core = ["redb", "serde", "serde_json"]
vector = ["instant-distance", "ort"]
graph = []  # built-in, enabled via feature flag
fts = ["tantivy"]
otel = ["opentelemetry", "opentelemetry-otlp"]  # optional, zero overhead when disabled
full = ["vector", "graph", "fts"]
```

## Core Traits

```rust
pub trait Engine: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
    fn on_record_insert(&self, record: &Record) -> Result<()>;
    fn on_record_delete(&self, id: &RecordId) -> Result<()>;
}

pub trait VectorIndex: Engine {
    fn add(&self, id: RecordId, vector: &[f32]) -> Result<()>;
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(RecordId, f32)>>;
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

> **Note:** `TextEmbedder` is separate from `VectorIndex` so ANN-only plugins don't need to implement embedding, and embedding can be configured independently (local ONNX vs external API).

## Target API (Rust Embedded)

```rust
use axil::{Axil, Record};

// Single entry point — Axil creates all plugin storage internally
let db = Axil::open("./memory.axil")
    .with_vector(384)           // creates *.axil.vec
    .with_graph()               // creates *.axil.graph
    .with_fts()                 // creates *.axil.fts/
    .build()?;

// Store
let session = db.insert("session", json!({
    "summary": "Fixed auth timeout bug",
    "project": "my-app",
    "created_at": "2026-03-31T10:00:00Z",
}))?;

// Embed
db.embed_field(&session, "summary")?;

// Graph
db.relate(&session, "modified", &file_id)?;

// Combined query
let results = db.query()
    .similar_to("auth error", 5)
    .traverse("->modified->file")
    .where_field("created_at", ">", "2026-03-24T00:00:00Z")
    .exec()?;
```

## MCP Server Tools (for Claude Code integration)

```json
{
  "tools": [
    {
      "name": "recall",
      "description": "Semantic search + graph + time-based recall of past context",
      "params": { "query": "string", "top_k": "int", "time_range": "string?" }
    },
    {
      "name": "store",
      "description": "Store a session/decision/pattern with auto-embedding",
      "params": { "table": "string", "data": "object", "embed_fields": "string[]?" }
    },
    {
      "name": "link",
      "description": "Create a graph relationship between two records",
      "params": { "from": "string", "edge_type": "string", "to": "string", "props": "object?" }
    },
    {
      "name": "search",
      "description": "Full-text search across all indexed fields",
      "params": { "query": "string", "limit": "int?" }
    },
    {
      "name": "query_history",
      "description": "Time-based query of past sessions and decisions",
      "params": { "after": "datetime?", "before": "datetime?", "project": "string?" }
    }
  ]
}
```

## Agent Memory Patterns (Built-in)

Unlike raw databases, Axil includes agent-specific memory patterns:

### TTL / Expiry
Records can have a `valid_until` timestamp. Expired records are excluded from queries by default.

### Memory Superseding
When a new fact is stored, Axil can auto-detect semantically similar existing facts (via vector similarity > 0.92) and mark the old ones as superseded, linking them via graph edge `->supersedes->`.

### Recency-Weighted Recall
The `recall()` function combines vector similarity with recency scoring, so newer memories rank higher when relevance is equal.

### Session Lifecycle
```
Session starts → agent calls recall() for context
Working...     → agent calls store() for decisions/patterns
Session ends   → agent calls store() with session summary + auto-embed
```

## Development Phases

All core phases are complete. (Detailed phase specs live in Axil memory + the
local, gitignored `tasks/` dir — they are not shipped in the public repo.)

### Phase 1: Core + Document Store ✅
### Phase 2: Vector Search ✅
### Phase 2b: Master Coordinator ✅
### Phase 3: Graph ✅
### Phase 4: FTS + MCP ✅
- Tantivy FTS with field scoping, fuzzy search, snippets, code tokenizer, field boosting
- MCP server (stdio): recall, store, link, search, query_history, get, list, delete

### Phase 5: Agent Memory Patterns ✅
- TTL/expiry (`axil-memory/ttl.rs`), memory superseding (similarity >0.92 + graph edges)
- Recency-weighted recall (per-memory-type alpha blending), session lifecycle (start/log/end → episodic)

### Phase 5b: Diagnostics & Observability ✅
### Phase 5c: Self-Healing ✅ (mostly)
- Auto-compact, vector rebuild, health-report, heal, snapshot, trends, config
- Deferred: FTS auto-commit on idle, health-report --save/--compare, embedding drift detector

### Phase 5d: Active Memory ✅ (mostly)
- Entity extraction, disambiguation/resolve, connection strength, entity profiles, graph inference
- Background worker (`axil worker run/status`), pattern detection, memory branching (create/list/delete/diff)
- Deferred: branch switch/merge, multi-agent shared memory (basic `--agent` tagging works)

### Phase 5e: Intelligent Database ✅
### Phase 5f: LLM Provider Interface ✅
- `LlmProvider` trait + `HttpLlm` (OpenAI-compatible), LLM-enhanced entity extraction & consolidation
- Cost tracking, rate limiting, graceful fallback to algorithmic. CLI: `axil llm test/config/usage`

### Phase 6: Polish + Scale ✅ (mostly)
- Combined query engine (RRF fusion), batch insert, Criterion benchmarks, API polish
- LoCoMo benchmark (99% hit rate, 94.4% recall — *historical; see note below*),
  LongMemEval benchmark, A/B testing framework
- Deferred: examples (vector_search, graph_queries, agent_memory), documentation site. (CI/CD shipped post-1.0: `.github/workflows/ci.yml` build/test/recall-quality gate + `release-plz.yml` auto-publish.)

> **Benchmark harnesses are tracked and in-tree** (Phase 25). The LoCoMo /
> LongMemEval / SQLite-compare / vector-latency / criterion-suite harnesses
> live under `benchmarks/` with their sources committed (`git ls-files
> benchmarks/`); only generated `data/`, `target/`, and `out/` are gitignored.
> They are `exclude`d from the default workspace (so `cargo check --workspace`
> stays fast/clean), not removed — run one with
> `cargo run --release --manifest-path benchmarks/<name>/Cargo.toml`.
> Regeneration in CI splits by data dependency:
> - **Dataset-free, CI-gated:** `sqlite-compare` (reduced-n speedup floor) and
>   the needle-recall gate run on every PR; `bench-check.sh` (Criterion
>   `core`/`vector`/`graph`/`fts`, >5% latency regression) runs nightly.
> - **Dataset-gated, skip-loud:** LongMemEval / LoCoMo / ConvoMem need an
>   out-of-tree dataset; their gates emit a loud `::warning` skip when it's
>   absent (a green CI run never means they verified anything). Committed
>   500-question baselines live in `benchmarks/results/` (e.g.
>   `qtc-500.json` backs the 94.5% Recall-QTC figure); the LongMemEval gate
>   compares against them when the dataset is present.

### Phase 7c: Web UI ✅
- React 19 + Vite 6 + rust-embed. Database explorer with graph/vector viz, query console.

### Phase 7d: AxilQL ✅
### Phase 8a: Performance ✅
### Phase 8b: AI Agent Performance Optimizations ✅
- All 21 items complete (8b.1–8b.21): cascaded filtering, adaptive RRF, batch embedding,
  activation scoring, int8 quantization, Matryoshka dims, temporal edges, tiered memory,
  PageRank recall, deferred indexing, negation detection, mmap vectors, binary embeddings,
  snapshots, hook capture, entity extraction, token-budgeted recall, multi-agent, boot context

### Phase 9: Ship — Remaining Work
- Testing gaps, examples, feature polish, benchmarks. (CI/CD now shipped — `.github/workflows/ci.yml` + `release-plz.yml`.)

### Phase 10: Cognitive Memory ✅
- Auto-importance scoring on every insert (entity density, structural markers, complexity)
- Memory decay with configurable half-life (default 90 days), access-based reinforcement
- Memory pressure: Hot/Warm/Cold/Archived tiers, auto-archive below threshold
- Auto-capture: detect errors and decisions from text, store automatically
- Cognitive query: importance-weighted recall with `--min-importance` filter
- Belief system: `axil believe/doubt/beliefs`, auto-generate from high-importance facts
- Context-aware push: `axil boot --files/--entities/--error`, `axil recall-for-file`

### Phase 13: Code-Aware Memory (SCIP) ✅
- Canonical entity identity: `_entities.canonical_id` + scoped `_entity_aliases`, idempotent `Axil::open` migration
- Code-symbol extractor (Rust/Python/TS/Go/Java regex) with `lang_hint` so cross-language `login` stays distinct
- `axil-scip` crate parses SCIP protobuf via `prost`; emits `defined_in`/`references`/`implements`/`type_of` (direct) and `calls`/`imports` (heuristic) edges with `confidence` label
- Provisional entity upgrade: regex-extracted `provisional:<sha>` rows rewritten to SCIP canonical id on unambiguous match; ambiguous cases stay provisional (no silent merge)
- `axil ingest-scip <path> [--dry-run | --watch]` with size+mtime stabilization gate
- `axil scip refresh [--language <lang>] [--if-stale --max-age-days N] [--in-background] [--skip-ingest]` — detects every `(language, project dir)` pair via a bounded subfolder walk (`axil-cli/src/scip_detect.rs`, depth ≤ 4, `node_modules`/`target`/dot-dirs/gitignored dirs skipped), runs each indexer (rust-analyzer / scip-typescript / scip-python / scip-go / scip-java) from its own project dir, and ingests all outputs in one sweep — a polyglot monorepo (`frontend/package.json` + `backend/pyproject.toml`, no root marker) is fully covered. Single-project repos write `.axil/index.scip`; polyglot repos write per-project `.axil/index-<lang>-<dir>-<hash>.scip` (hash defeats lossy-slug sibling collisions; legacy single file retired only after a full-coverage sweep). Missing indexer binaries skip that project with an install hint (non-actionable for the `--if-stale` fast path, hard error with explicit `--language`); `--if-stale` is checked per project; a failed ingest deletes its output so the next refresh self-heals. Brain hook calls `--if-stale --in-background --quiet` on first PreToolUse so refresh is opportunistic and never blocks the agent (lock at `.axil/scip-refresh.lock`, child runs under `nohup` so it survives parent shell exit).
- `axil scip status` — reports detected `(language, project dir)` pairs with each project's expected output file and age, indexer presence on PATH, existing `.scip` files with age/symbol counts, and per-language install hints.
- `axil doctor` SCIP detection block: reports indexer/symbols/age; suggests installer (scip-rust, scip-python, scip-typescript, scip-go, scip-java) when index missing on code repo, and points at `axil scip refresh` as the one-liner
- `axil recall-for-entity` (BFS over call/ref/impl/type edges + `--trace-graph`) and Pass 4 on `recall-for-file` — surfaces memories about callers/callees when agent edits a symbol

### Phase 13b: Structural Code Recall ✅
- `_idx_code_proxies` table: structure-aware proxy records (file/symbol/section) with stable `proxy_id`, breadcrumb, signature, line range, optional SCIP `canonical_id`
- `axil code-search` / `code-context` / `explain-code-hit` + MCP parity (`code_search`, `code_context`)
- Pointer-attached memories: `axil store --code-ref <proxy_id|canonical_id|path:line>` resolves to `code_refs[]`; recall surfaces memories whose refs match returned proxies
- Markdown heading splitter + TOML/JSON/YAML section splitters share `axil_indexer::{split_sections, split_toml_sections, split_json_sections, split_yaml_sections}`
- Graph composition: `same_file` and `tests` edges between proxies; `--trace-graph` walks SCIP call/ref/impl edges from proxy hits via canonical_id bridge
- SCIP P0/P1: backfill provisional proxies via `_scip_aliases` (file-scoped), precise `line_end` from SCIP Definition `enclosing_range` stored on `_entities.def_line_*`
- `axil code-recall-bench` + `scripts/code-recall-gate.sh` regression gate (`tests/fixtures/code-recall/`); P0 quality measured (45% ctx-token reduction on Axil dogfood, top-3 file/symbol 0%→20%, p95 19ms→15ms)
- 13b.10 perf hardening: `ProxyDedupCache` skips re-embed on unchanged proxies (refreshes nav-only drift via `db.update`), one-shot `_entities` map kills O(symbols×entities), per-recall boost iterates fused entries instead of full proxy table, `_idx_code_refs` reverse index replaces N+1 walk over memory tables (auto-synced from `Axil::insert`/`update`/`delete` in `axil_core::code_refs`)

### Phase 16: Dependency Doc Memory ✅ (P0 — 5 ecosystems)
- New `crates/axil-docs/` crate behind the `deps` Cargo feature (in axil-cli `default`/`full`; excluded from minimal `core` builds — a true opt-out plugin)
- Version-pinned library docs in memory: detect manifests → resolve exact lockfile versions → extract docs from the on-disk dependency copy → chunk + embed + FTS into `_dep_docs`. `axil dep-docs "<query>"` returns version-correct docs with zero network calls.
- Three tables: `_dep_manifests` (drift state), `_deps` (resolved dependencies), `_dep_docs` (doc chunks, embedded + FTS-indexed)
- Five ecosystems — Cargo, npm, Python, Go, Java. Lockfiles parsed: `Cargo.lock`; `package-lock.json` / `yarn.lock` (v1 + Berry) / `pnpm-lock.yaml`; `uv.lock` / `poetry.lock` / `Pipfile.lock`. Go and Java pin versions inline in `go.mod` / `pom.xml`.
- Local extraction (Path 0, default): Rust from `~/.cargo/registry/src/...`, npm from `node_modules/`, Python from site-packages `*.dist-info`, Go from the module cache, Java from `~/.m2`; `CARGO_HOME` honored
- Drift detection: content-hashed manifest + lockfile in `_dep_manifests`; `axil deps refresh --if-stale` re-ingests only changed deps. The PostToolUse brain-hook fires a detached `deps refresh --if-stale` on any manifest/lockfile edit; `axil boot` has a `dep_docs_freshness` block; `recall-for-file` surfaces dep docs for a file's imports.
- Version history (P0.4): a version bump archives the old version's chunks (`archived: true` — kept for migration questions) and marks the old `_deps` row `superseded` + linked to the replacement; a dropped dependency is swept to `removed`. `axil dep-docs` hides archived chunks unless `--include-superseded`.
- Changelog memory (P1.b): on a version bump the dependency's `CHANGELOG.md` is read from its on-disk copy and stored as `migration`-tagged `_dep_docs` chunks (Cargo/npm/Go) — the agent can recall "what changed when we bumped X"; `DepDocHit.doc_kind` labels the hit
- Doc diffing (P1.c): a bump also stores a `doc_kind: "doc_diff"` chunk — the section-level added/removed/changed delta between the old and new docs, the *observed* change set (catches what authors omit from a changelog)
- Transitive deps (P1.a): `deps sync/refresh --transitive` ingests the transitive deps the project's own source actually imports — `imports::scan_project_imports` walks `.rs`/`.js`-family source and gates the lockfile closure against it (Cargo/npm); ingested as `kind: "transitive"`
- Web fallback: Path A — `axil deps ingest` accepts agent-fetched text (stdin/file, no feature flag); Path B — `web.rs` HTTP fetcher behind the default-off `web-docs` feature (npm registry; offline-first posture)
- CLI: `axil deps {list,sync,refresh,ingest,status}` + `axil dep-docs`; MCP parity: `dep_docs` + `deps_status` tools
- Deferred: `documents` graph edges (memory↔dep-doc chunks, no consumer yet); a dep-docs-specific CI gate (the repo now has CI — `.github/workflows/ci.yml` runs build/test/recall-quality on every PR — but it does not exercise dep-docs). Only P1 epic still open: 16.P1.d workspace-shared dep-doc cache (blocked on Phase 14, multi-project workspace).

### Phase 18: Session Checkpoint ✅
- New `crates/extensions/axil-checkpoint/` (Tier-2 Extension behind the `checkpoint` Cargo feature, in axil-cli `default`/`full`). Templated on `DocsExtension`.
- **Structured checkpoint record** replaces the free-text `context:session_summary` pattern. Fields: `goal`, `state`, `next_steps[]`, `open_questions[]`, `references[]` (typed pointers `{kind, ref, note?}`), optional `summary` headline. Stored in `_checkpoint_records` (prefix `_checkpoint_`) with a `session_checkpoint_for` graph edge to `_sessions`.
- **Two write modes:** `axil checkpoint '<json|->'` writes a mid-session snapshot (kind `snapshot`, owning session keeps running, auto-creates an active session when none exists); `--final` stamps kind `final`. MCP parity via `checkpoint` + `checkpoint_show` tools.
- **Implicit replay on boot:** Phase 17's `Extension::boot_block` is now actually wired — `axil_core::collect_extension_blocks(&db)` feeds both the `BootContext::CurrentScope.extension_blocks` map (v1 schema) and the default CLI flat-JSON boot path. `boot_to_narrative` renders contributed blocks at the top of `axil boot`, before rules/sessions/decisions. `CheckpointExtension::boot_block` emits "## Resume Here" — `references[]` of `kind: record` are resolved live so superseded decisions show current state, not stale snapshots.
- **Adapter fallback:** when no explicit checkpoint is stored, `derive_checkpoint_from_session` synthesizes one from the latest session's `summary`, `decisions_made`, `files_touched` plus unresolved errors, tagged `_(source: derived — no explicit checkpoint stored)_` so the agent knows the difference.
- New `/axil-checkpoint` skill (Matt Pocock /handoff principles adapted to Axil's persistent model: compact, reference don't duplicate, redact, tailor to focus). Both project CLAUDE.md templates updated to point the post-task store pattern at `axil checkpoint`.
- Tests: 26 in axil-checkpoint (types + write + derive + render + extension); 3 new in axil-core::boot (extension_blocks wiring); extension_audit picks up CheckpointExtension automatically (3/3 green).

## Project Structure

```
axil/
├── Cargo.toml              # workspace root
├── CLAUDE.md               # this file
├── README.md
├── LICENSE                 # PolyForm Noncommercial 1.0.0
├── LICENSING.md            # dual-license: NC free + commercial path
├── crates/
│   ├── axil-core/          # storage engine, record types, traits
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── db.rs       # Axil struct, open/build, doctor/stats/bench
│   │   │   ├── record.rs   # Record, RecordId types
│   │   │   ├── storage.rs  # redb storage backend
│   │   │   ├── query.rs    # query builder, explain, profiling
│   │   │   ├── plugin.rs   # Engine traits
│   │   │   ├── error.rs    # error types
│   │   │   ├── config.rs   # AxilConfig, axil.toml parsing
│   │   │   ├── metrics.rs  # Metrics collector, counters, latency tracking
│   │   │   ├── otel.rs     # OpenTelemetry instrumentation (behind `otel` feature)
│   │   │   └── diagnostics.rs # DoctorReport, DatabaseStats, BenchReport types
│   │   ├── benches/
│   │   │   └── core_benchmarks.rs  # Criterion: insert, get, filter, combined query
│   │   └── Cargo.toml
│   ├── axil-vector/        # vector search plugin
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── hnsw.rs     # HNSW index
│   │   │   ├── embed.rs    # ONNX embedding + MultiEmbedder
│   │   │   └── models.rs   # EmbeddingModel enum + custom model registry
│   │   ├── benches/
│   │   │   └── vector_benchmarks.rs  # Criterion: add, search, delete
│   │   └── Cargo.toml
│   ├── axil-graph/         # graph plugin
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── edge.rs     # edge storage
│   │   │   └── traverse.rs # traversal engine
│   │   ├── benches/
│   │   │   └── graph_benchmarks.rs  # Criterion: relate, neighbors, traverse
│   │   └── Cargo.toml
│   ├── axil-fts/           # full-text search plugin
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── index.rs    # tantivy wrapper
│   │   ├── benches/
│   │   │   └── fts_benchmarks.rs  # Criterion: index_text, search, fuzzy
│   │   └── Cargo.toml
│   ├── axil-scip/          # SCIP code-graph ingest (Phase 13)
│   │   ├── src/
│   │   │   ├── lib.rs      # ingest + provisional upgrade + watch stabilization
│   │   │   └── proto.rs    # hand-written prost messages (no build.rs)
│   │   └── Cargo.toml
│   ├── axil-docs/          # dependency doc memory (Phase 16)
│   │   ├── src/
│   │   │   ├── lib.rs      # crate root: DocsError, find_row/delete_rows_where
│   │   │   ├── manifest.rs # detect + parse manifests (5 ecosystems)
│   │   │   ├── resolve.rs  # pin deps to exact lockfile versions
│   │   │   ├── local.rs    # extract docs from the on-disk dep copy
│   │   │   ├── ingest.rs   # chunk → embed → FTS → _dep_docs
│   │   │   ├── refresh.rs  # manifest-hash drift detection
│   │   │   ├── query.rs    # scoped recall over _dep_docs
│   │   │   └── web.rs      # Path B HTTP fetcher (web-docs feature)
│   │   └── Cargo.toml
│   ├── axil-mcp/           # MCP server
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── server.rs   # MCP protocol handler
│   │   │   └── tools.rs    # tool definitions
│   │   └── Cargo.toml
│   ├── axil-ql/            # AxilQL query language parser
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── lexer.rs    # tokenizer
│   │   │   ├── parser.rs   # recursive descent parser
│   │   │   ├── ast.rs      # AST types
│   │   │   └── compiler.rs # AST → QueryBuilder
│   │   └── Cargo.toml
│   ├── axil-ui/            # React web UI (Vite + React 19)
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── components/ # React components
│   │   │   ├── pages/      # route pages
│   │   │   └── hooks/      # custom hooks
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── axil-cli/           # CLI binary
│       ├── src/
│       │   └── main.rs
│       └── Cargo.toml
├── benchmarks/
│   ├── criterion-suite/    # combined Criterion benchmarks (vector+graph+FTS hot paths)
│   ├── locomo/             # LoCoMo retrieval quality benchmark
│   └── longmemeval/        # LongMemEval retrieval accuracy benchmark
├── scripts/
│   └── bench-check.sh      # CI regression detection (>5% threshold)
├── models/                 # ONNX embedding models (gitignored, downloaded at build)
├── tests/                  # integration tests
└── examples/               # usage examples
```

## Coding Conventions

- Use `thiserror` for error types in library crates
- Use `anyhow` in binary crates (CLI, MCP server)
- Prefer `&str` over `String` in function params
- All public APIs must have doc comments
- Use `#[cfg(feature = "...")]` for optional plugins
- Tests in each crate + integration tests at workspace root
- Keep dependencies minimal — every dependency is a decision
- File extension for databases: `.axil`
- **Numbers integrity.** Every savings/compression/speed-up/reduction figure
  surfaced to a user (README, docs, CLI/MCP output) must be measured against a
  named baseline, labeled an estimate (naming its heuristic), or sourced to a
  committed benchmark. A bare number that can't be traced to one of those is a
  bug. See [Numbers integrity](docs/src/advanced/context-economics.md#numbers-integrity-policy).
- **No task/phase tags in code comments.** Don't prefix comments (or doc
  comments) with `Phase 20.2:`, `8b.19:`, task IDs, etc. Comments explain *why
  the code is the way it is* — which phase shipped it is git-history noise that
  goes stale. Write the rationale, drop the bookkeeping. (Phase numbers belong
  in `tasks/`, commit messages, and Axil memory — not the source.)

## Licensing & Business Model

Axil is **source-available, free for noncommercial use** under the
[PolyForm Noncommercial License](LICENSE); commercial use requires a separate
commercial license. The embedded engine + all plugins + CLI + MCP server run
fully standalone. See [LICENSING.md](LICENSING.md).

**Axil Atlas** — the multi-database team/sync control plane — is a separate
**closed, commercial** product and lives in a private repo. Pricing, tiering,
and go-to-market strategy live in the private `axil-atlas` repo, not here.

## References

- redb: https://github.com/cberner/redb
- instant-distance: https://github.com/InstantDomain/instant-distance
- tantivy: https://github.com/quickwit-oss/tantivy
- ort (ONNX Runtime): https://github.com/pykeio/ort
- MCP spec: https://modelcontextprotocol.io
- SurrealKV (alternative storage): https://github.com/surrealdb/surrealkv
- HelixDB (competitor): https://github.com/HelixDB/helix-db
- Mem0 (competitor): https://github.com/mem0ai/mem0
- Engram (competitor): https://github.com/Gentleman-Programming/engram
