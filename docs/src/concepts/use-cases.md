# Use Cases

Axil is one engine — a single embedded `.axil` file with vector + graph + FTS + 5 memory types — but it serves several distinct workloads. This page maps the most common use cases to the features and config you'd reach for.

## At a glance

| Use case | What you store | What you retrieve | Features used |
|---|---|---|---|
| [Agent memory](#agent-memory) | sessions, decisions, errors, context | "what did we decide about X?" | All cognitive primitives (decay, importance, supersede, beliefs, boot) |
| [RAG / knowledge base](#rag-knowledge-base) | documents, chunks | "what does the doc say about X?" | Vector + FTS + hybrid + QTC chunking |
| [Code-aware retrieval](#code-aware-retrieval) | source files, SCIP graph | "where is `login()` called from?" | SCIP ingest + graph traversal + entity extraction |
| [Personal AI](#personal-ai) | notes, conversations, preferences | "remember when I told you X?" | Episodic memory + preferences + decay |
| [Local-first agent state](#local-first-agent-state) | task graphs, intermediate results | "resume where we left off" | Boot context + sessions + branching |

---

## Agent memory

The flagship use case. Your agent learns, remembers, synthesizes, and forgets — automatically.

**What you store**
- Session summaries (`sessions` table)
- Decisions with rationale (`decisions`)
- Errors and fixes (`errors`)
- Architecture notes (`context`)
- User preferences (`preferences`)

**What you retrieve**
- Top-N relevant memories at session start (`axil boot`)
- Memories matching a topic (`axil recall "auth flow"`)
- Beliefs the agent has formed (`axil beliefs`)
- Records linked to a file or entity (`axil recall-for-file`, `axil recall-for-entity`)

**Why Axil specifically**
- 5 memory types built in (episodic, semantic, procedural, preference, belief)
- Auto-importance scoring + decay so old memories fade
- Auto-supersede on near-duplicates (cosine > 0.92)
- No LLM required for any of this — algorithmic by default, LLM-enhanced when configured

```bash
axil store decisions '{"summary":"chose redb over sled","reason":"ACID + maintained","files":["Cargo.toml"]}'
axil recall "storage choice" --top-k 5
axil boot --schema v1   # session-start context with budget discipline
```

---

## RAG / knowledge base

Static or slowly-changing document corpora — manuals, codebases, knowledge bases, support docs. Axil works as a single-file embedded RAG store.

**What you store**
- Document chunks (manual ingestion or `axil ingest`)
- Original document metadata (path, title, source)

**What you retrieve**
- Top-K chunks matching a query (vector + FTS hybrid via `recall`)
- Token-budgeted chunks for prompt assembly (`recall --budget 2000`)

**Recommended config for RAG**
Disable the cognitive features that assume dynamic state — RAG corpora don't supersede or decay:

```toml
# axil.toml
[mode]
preset = "rag"   # disables decay, supersede, auto-importance
```

Or programmatically:

```rust
let cfg = RecallConfig {
    qtc: Some(QtcConfig::default()),  // chunk-level reranking, the lift on long docs
    ..Default::default()
};
let hits = db.recall("how do I configure SSL?", 10, Some(cfg))?;
```

**Why Axil specifically vs Chroma/LanceDB**
- Single `.axil` file (no separate metadata DB)
- Hybrid vector + FTS + RRF fusion (most embedded RAG stores are vector-only)
- Query-time chunking (QTC) — 97.2% hit rate on LongMemEval-S
- Source-available, free for noncommercial use (commercial license available)
- Same engine doubles as agent memory if you ever add an agent loop on top

**Bench positioning** *(working note — needs validation)*
> Run benchmarks/bench-axil-vs-lancedb on a public RAG eval before publishing comparative numbers.

---

## Code-aware retrieval

Use Axil's SCIP integration to give a coding agent fast retrieval over a codebase: not just "files matching this query" but "calls/refs/impls of this symbol."

**What you store**
- Source files via `axil-indexer`
- A SCIP index — easiest path is `axil scip refresh` (auto-detects language, runs the right indexer, ingests in one step). Manual path: `axil ingest-scip path/to/index.scip`

**What you retrieve**
- Memories about a symbol (`axil recall-for-entity login`)
- Memories about a file's callers/callees (`axil recall-for-file src/auth.rs`)
- Trace through the code graph (`recall-for-entity --trace-graph`)

**Why Axil specifically**
- SCIP edges (`defined_in`, `references`, `calls`, `imports`, `implements`, `type_of`) become first-class graph relationships
- Cross-language disambiguation via `lang_hint` (Rust `login` ≠ Python `login`)
- Boot context surfaces relevant code memories when the agent edits a symbol

```bash
# Quickest setup — autodetects language, runs indexer, ingests:
axil scip refresh

# Or manual two-step:
scip-rust . --output index.scip && axil ingest-scip index.scip --watch

axil recall-for-file src/auth.rs --top-k 5 --trace-graph
```

---

## Personal AI

A long-running personal assistant that remembers conversations, preferences, and recurring topics across sessions and devices.

**What you store**
- Conversations (`sessions`)
- Preferences (`axil prefer favorite_editor '"vim"'`)
- Notes / journal (`context`)
- Beliefs the assistant has formed (`axil believe "user prefers terse responses"`)

**What you retrieve**
- "Did the user mention X recently?" (`recall` with recency weighting)
- "What does the user prefer for Y?" (`get preferences/<key>`)
- Decayed older memories surface less but aren't deleted

**Why Axil specifically**
- All data stays in one file you can back up to iCloud/Drive
- No vendor lock-in, no API key, no cloud account
- Cognitive primitives match how humans actually remember (recency, importance, decay)
- Local-first → privacy by default

---

## Local-first agent state

For complex agent workflows that span hours/days — long-running task graphs, intermediate results, branching exploration.

**What you store**
- Task / subtask records (custom table)
- Intermediate results
- Decision branches (`axil branch create alternative-approach`)

**What you retrieve**
- "Resume the task I was working on" (`axil boot --schema v1`)
- "What did I try last time that failed?" (`recall errors`)
- "Switch to the other exploration branch" (`axil branch switch`)

**Why Axil specifically**
- Branching memory (try an approach, fall back, compare)
- Session lifecycle: start → log → close, with `axil close-session`
- Boot context restores the agent's working state without re-prompting from scratch

---

## Combining use cases

The use cases above aren't mutually exclusive — same `.axil` file can hold both a static RAG corpus AND dynamic agent memory. The `_scope` field separates them:

```bash
axil store docs '{"_scope":"rag","title":"...","content":"..."}'
axil store decisions '{"_scope":"agent","summary":"chose..."}'
axil recall "X" --scope agent     # only agent memory
axil recall "X" --scope rag,agent # both
```

The cognitive primitives (decay, supersede) only apply to records that have the relevant fields — RAG records without `_importance` or `_superseded_at` are simply unaffected.

---

## What Axil is NOT a good fit for

Honest negatives so you don't waste time:

- **High-throughput OLTP** — redb is single-writer; if you need 10k+ writes/sec across many threads, use a server DB.
- **Multi-region replication** — no built-in sync today (see [Axil Sync roadmap](#)).
- **SQL-style analytics** — AxilQL is verb-first for retrieval, not aggregation. Use DuckDB for that.
- **Pure key-value cache** — Axil's overhead (vector index, graph, FTS) is wasted if you only need `get/set`.
- **Massive corpora (>10M records per table)** — works, but other engines (LanceDB, Vespa) are tuned for that scale.
