<div align="center">

<pre>
 █████╗  ██╗  ██╗ ██╗ ██╗     
██╔══██╗ ╚██╗██╔╝ ██║ ██║     
███████║  ╚███╔╝  ██║ ██║     
██╔══██║  ██╔██╗  ██║ ██║     
██║  ██║ ██╔╝ ██╗ ██║ ███████╗
╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝ ╚══════╝
</pre>

### Agent memory in one local file. No server, no cloud, no LLM.

*Think SQLite, but for your agent's memory — a file you embed, not a database you run.*

**Local-first · ~5–10MB binary · vector + graph + full-text + time-series · MCP · 74–80% fewer context tokens**

[![CI](https://github.com/FC4b/axildb/actions/workflows/ci.yml/badge.svg)](https://github.com/FC4b/axildb/actions/workflows/ci.yml)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm--NC-blue.svg)](LICENSE)
![Built with Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg?logo=rust&logoColor=white)
![No LLM required](https://img.shields.io/badge/LLM-not_required-2ea44f.svg)
![Runs offline](https://img.shields.io/badge/runs-offline_·_one_binary-8a2be2.svg)

[**Token savings**](#token-savings--real-savings) · [**Install**](#install) · [**Quick start**](#quick-start) · [**Benchmarks**](#benchmarks) · [**Extensible**](#extensible-by-design) · [**Docs**](#documentation)

</div>

---

Your coding agent is brilliant and amnesiac. Every session it re-reads the same files, re-learns the same architecture, repeats the same mistakes — and **burns tokens (your money) doing it.** Axil is the second brain that fixes this: it remembers decisions, gotchas, and code structure across sessions and hands the agent the *right* memory at the right moment, instead of dumping the whole repo into context.

> **In a real, equal-correctness A/B test, agents answered the same coding questions with 74–80% fewer context tokens using Axil.** → [the numbers & caveats](#token-savings--real-savings)

- 🧠 **Remembers across sessions** — learn once, never re-read. Vector + knowledge graph + full-text + time-series, all in a single `.axil` file.
- 💸 **Frugal by design** — ~15:1 context compression; returns pointers, not file dumps. Fewer tokens, every turn.
- ⚡ **Embeddable & instant** — no Postgres, no cloud, no daemon. A ~5–10MB binary, <100 ms commands, fully offline.
- 🤖 **No LLM required** — local ONNX embeddings + rule-based cognition do ~80% with zero API calls; plug an LLM in for the rest.
- 🔗 **One memory, every tool** — the same portable `.axil` brain is read *and* written by Claude Code, Cursor, Windsurf, Codex, any MCP client, or your own Rust. No vendor lock-in.
- 🕸️ **Code-graph + doc memory others don't have** — a SCIP **code-graph** (knows callers/callees, not just text) and **version-pinned dependency-doc memory** (your *exact* lib versions, zero network), on an **Engine · Extension · Adapter** architecture you can extend.

---

## Token savings = real savings

Every token in your agent's context is billed — and re-billed on **every turn**. We measured the win with a real, end-to-end A/B test (not a synthetic estimate): clone one public repo into two identical sandboxes, give a coding agent the same "where is X / how does Y work" tasks in each — one with only `grep` + file reads, the other with only Axil — and count the context tokens each pulls in **to reach a verified-correct answer**:

| Corpus | Without Axil | With Axil | Reduction |
|--------|-------------:|----------:|----------:|
| **Django** (906 source files) | 10,763 tok | **2,111 tok** | **80%** |
| **Flask** (24 source files) | 16,225 tok | **4,300 tok** | **74%** |

On the common "where is X" question, Axil answers in **~100 tokens** (one pointer-shaped hit) where an unaided agent greps and reads several files — **75–90% fewer tokens per lookup**, every lookup, every session.

> ⚠️ **A specific experiment, not a guarantee.** Two open-source Python repos, a disciplined agent, measured at equal task-correctness. Real savings depend on repo size and question type — **largest on big codebases and semantic "where/how" questions**, near break-even on tiny repos where `grep` already nails it. Reproduce: `scripts/context-ab-setup.sh`. Full methodology and every run: [Context Economics](docs/src/advanced/context-economics.md).

## What you get

A complete cognitive memory system in one binary — vector search, knowledge graph, full-text search, and time-series, with real agent cognition on top:

**🧠 Cognitive memory (no LLM required)** — 5 memory types (working, semantic, episodic, procedural, preference) · auto-importance scoring · active forgetting (decay + reinforcement) · belief system · auto-capture of errors & decisions · consolidation & contradiction detection.

**🔎 Multi-model retrieval** — HNSW vector search (local ONNX/BGE) · a **temporal knowledge graph** (typed edges, traversal, entity extraction + inference, time-aware `as_of` queries — no Neo4j) · Tantivy full-text · time-series. One `recall()` fuses them all (RRF) with per-result score explanations.

**💻 Built for code agents** — structural code index + `code-search` / `code-context` (pointer-shaped, token-frugal) · SCIP cross-reference graph · version-pinned dependency-doc memory · structured session checkpoints · AxilQL · MCP server (full CLI parity).

**🔌 Optional LLM upgrade** — everything works 100% without an LLM. Plug one in (Claude / GPT / Ollama, or via Claude Code skills) to go from ~80% → ~95% on extraction & consolidation.

## How it works

One command turns any Claude Code / Cursor / Codex project into a memory-enriched agent:

```bash
axil install --claude-code --bootstrap   # wire hooks + skills AND index your code, in one shot
```

From then on, the loop runs itself:

<div align="center">
<pre>
   ┌───────────────┐      ┌────────────────────┐      ┌─────────────────────┐
   │   1. BOOT     │ ───▶ │    2. WORK         │ ───▶ │   3. CHECKPOINT     │
   │ inject recent │      │ recall on demand   │      │ write "resume here" │
   │ context from  │      │ + auto-capture     │      │ so the next session │
   │ .axil &lt;100 ms │      │ decisions &amp; errors │      │ resumes, not restart│
   └───────────────┘      └────────────────────┘      └─────────────────────┘
          ▲                                                       │
          ╰──────────────  next session boots from it  ◀──────────╯

   one .axil file · no server · no LLM · &lt;100 ms · 100% offline
</pre>
</div>

A PreToolUse hook injects context before each turn, file edits are auto-captured, and a Stop hook writes a checkpoint at the end — so the next session *resumes* instead of restarting. Or drive it by hand:

```bash
axil recall "<query>"       # cognitive recall (fusion + importance + decay)
axil code-search "<symbol>" # token-frugal code location
axil boot                   # the "resume here" context block
axil brief                  # today's summary, any time
```

## Benchmarks

**LongMemEval** — retrieval recall over 500 questions (top-k=5), vs comparable memory systems:

| System | Recall | No LLM | No server |
|--------|-------:|:------:|:---------:|
| MemPalace | 96.6% | ✅ | ✅ |
| **Axil — Recall-QTC** | **94.5%** | ✅ | ✅ |
| Hindsight | 91.4% | ❌ | ❌ (PostgreSQL) |
| **Axil — Recall (fusion)** | **90.9%** | ✅ | ✅ |
| Memvid | 85.7% | ✅ | ✅ |
| Mem0 | 68.4% | ❌ | ❌ (3 DBs) |
| Zep | 66.0% | ❌ | ❌ |

Axil hits **94.5% recall at ~1/8th the context-token cost** of comparable systems, runs **~173× faster vector search** than SQLite + sqlite-vec at 100k vectors, and answers in **<100 ms** from a **~5–10 MB** offline binary.

> Numbers from the in-tree harnesses (competitor/latency harnesses kept out-of-tree). → Full tables, per-category breakdown, and methodology: **[Benchmarks](docs/src/advanced/benchmarks.md)**.

## Extensible by design

Axil isn't a monolith — it's a small core with a **three-tier plugin model**, and its headline capabilities are built *on* it (code-graph and dependency-doc memory are just Extensions). The tiers are Cargo features, so you build exactly what you need:

```bash
# default = everything; or compose your own — a lean code-agent build:
cargo install axil-cli --no-default-features \
  --features "vector,embed,graph,fts,timeseries,memory,scip,deps,checkpoint,mcp,ql"
#   Engines    (Tier 1 · storage)      : vector embed graph fts timeseries
#   Extensions (Tier 2 · capabilities) : memory indexer scip deps checkpoint rerank
#   Adapters   (Tier 3 · surfaces)     : mcp ql http
```

- **Engines** — storage substrates; each owns a companion file beside your `.axil`.
- **Extensions** — capabilities on top (memory, SCIP code-graph, dependency-doc memory, rerank, checkpoints). Add your own without forking the core.
- **Adapters** — how the world talks to it (CLI, MCP, AxilQL) — same engine underneath, every surface in parity.

Tiers are chosen at **build time** (features, above) — drop a feature and it's compiled out: its commands disappear, but your `.axil` **data stays compatible and dormant** (re-add the feature later and it's live again — no migration). **Run-time** behavior is tuned in an optional `axil.toml` (project root, or `~/.config/axil/axil.toml`):

```toml
[database]
path = "./.axil/memory.axil"

[index]
embedding_model = "bge-small-en-v1.5"    # local ONNX, auto-downloaded
embedding_dimensions = 384

[fts]
default_limit = 10

[llm]                                     # optional — Axil works fully without it
endpoint = "https://api.openai.com/v1"
model = "gpt-4o-mini"                     # api_key via AXIL_LLM_API_KEY env var
```

→ Build your own: [Three Tiers](docs/src/extending/overview.md) · [Engines](docs/src/extending/engines.md) · [Extensions](docs/src/extending/extensions.md) · [Adapters](docs/src/extending/adapters.md) · [Configuration](docs/src/getting-started/configuration.md)

## Install

```bash
cargo install axil-cli                 # published crate (default features ≈ everything)

# or build from source:
git clone https://github.com/FC4b/axildb.git && cd axildb
cargo build --release -p axil-cli
```

Then run `axil install --claude-code --bootstrap` inside your project. Full options — feature flags, SCIP indexers, manual setup — in [Installation](docs/src/getting-started/installation.md).

## Quick start

**Path A — agent memory (recommended).** One command wires Axil in and indexes your code; from there the agent does the work (hooks inject context, capture edits, and checkpoint automatically). The DB auto-detects at `.axil/memory.axil`, so everyday commands need no `--db`:

```bash
axil install --claude-code --bootstrap   # hooks + skills + initial code index
axil boot                                 # "resume here" — recent decisions, errors, open threads
axil recall "auth timeout" --top-k 5      # cognitive recall (vector + graph + recency + keyword)
axil code-search "login handler"          # token-frugal "where is X?" — pointers, not file dumps
# the agent stores what it learns as it goes:
axil store decisions '{"summary":"Use JWT","reason":"simpler than OAuth","files":["auth.rs"]}'
axil checkpoint      '{"goal":"ship auth","state":"tests green","next_steps":["wire refresh"]}'
```

→ Using Cursor, Windsurf, Codex, or another MCP client? See the [Agent Integration guide](docs/src/agents/claude-code.md) and [MCP Server](docs/src/agents/mcp.md).

**Path B — standalone CLI.** Drive Axil directly as a memory store:

```bash
axil init ./memory.axil                                        # create a database
axil --db ./memory.axil store decisions '{"choice":"Use JWT"}' # store (any table + JSON)
axil --db ./memory.axil recall "auth issues" --top-k 5         # semantic recall (local ONNX, no key)
axil --db ./memory.axil fts "timeout error"                    # full-text search
axil --db ./memory.axil link <a> mentions <b>                  # knowledge-graph edge
axil --db ./memory.axil traverse <a> "->mentions->entity"      # multi-hop walk
axil --db ./memory.axil ql 'RECALL "auth error" TOP 5'         # AxilQL one-shot
```

> Tip: set `AXIL_DB=./memory.axil` to drop the `--db` flag. `axil --help` lists every command; `axil doctor` checks health. → [Quick Start](docs/src/getting-started/quick-start.md) · [CLI reference](docs/src/cli/data.md).

**Use it from Rust:**

```rust
use axil_core::Axil;
use axil_vector::{models::EmbeddingModel, AxilBuilderVectorExt};
use axil_graph::AxilBuilderGraphExt;
use axil_fts::AxilBuilderFtsExt;

let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?  // Engine: vector
    .with_graph_plugin()?                            // Engine: graph
    .with_fts_plugin()?                              // Engine: full-text
    .build()?;

let session = db.insert("sessions", serde_json::json!({ "summary": "Fixed auth timeout" }))?;
db.embed_field(&session.id, "summary")?;
let hits = db.query().similar_to("auth error", 5).exec()?;
```

→ [Embedded Usage](docs/src/api/embedded.md) · [Query Builder](docs/src/api/query-builder.md) · [Plugin Traits](docs/src/api/plugin-traits.md)

## Documentation

| Topic | Pages |
|-------|-------|
| **Getting started** | [Install](docs/src/getting-started/installation.md) · [Quick Start](docs/src/getting-started/quick-start.md) · [Configuration](docs/src/getting-started/configuration.md) |
| **Concepts** | [Architecture](docs/src/concepts/architecture.md) · [Memory Types](docs/src/concepts/memory-types.md) · [Engines](docs/src/concepts/plugins.md) · [Storage Model](docs/src/concepts/storage.md) |
| **CLI reference** | [Data](docs/src/cli/data.md) · [Memory](docs/src/cli/memory.md) · [Code Search](docs/src/cli/code-search.md) · [Diagnostics](docs/src/cli/diagnostics.md) · [AxilQL](docs/src/cli/axilql.md) · [Dependency Docs](docs/src/cli/dependency-docs.md) |
| **Agent integration** | [Claude Code](docs/src/agents/claude-code.md) · [MCP Server](docs/src/agents/mcp.md) · [Multi-Agent](docs/src/agents/multi-agent.md) |
| **Deep dives** | [Benchmarks](docs/src/advanced/benchmarks.md) · [Context Economics](docs/src/advanced/context-economics.md) · [Retrieval Pipeline](docs/src/advanced/retrieval-pipeline.md) · [Cognitive Memory](docs/src/advanced/cognitive.md) |
| **Extending Axil** | [Three Tiers](docs/src/extending/overview.md) · [Engines](docs/src/extending/engines.md) · [Extensions](docs/src/extending/extensions.md) · [Adapters](docs/src/extending/adapters.md) |

## Status & license

**Pre-release / active development.** Core engine, all plugins, agent memory, diagnostics, and benchmarks are implemented; examples and the hosted docs site are in progress.

**Source-available, free for noncommercial use** under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal projects, research, education, nonprofits, and evaluation. Commercial use requires a commercial license; see [LICENSING.md](LICENSING.md). **Axil Atlas** (the multi-database team/sync control plane) is a separate commercial product — the engine here runs fully standalone with no Atlas dependency.

## Star history

If Axil saves your agents tokens, a star helps others find it. ⭐

<a href="https://star-history.com/#FC4b/axildb&Date">
  <img src="https://api.star-history.com/svg?repos=FC4b/axildb&type=Date" alt="Star History Chart" width="600">
</a>

## Contributing

Contributions, feedback, and ideas are welcome — open an issue to start a discussion.

```bash
git clone https://github.com/FC4b/axildb.git && cd axildb
cargo build && cargo test
```
