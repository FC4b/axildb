# Storage Model

## Core engine

Axil uses [redb](https://github.com/cberner/redb) as its core storage engine — an embedded, ACID-compliant, pure-Rust key-value store. Each database is a single `.axil` file.

## Records

All data is stored as `Record` structs:

- **ID**: ULID (Universally Unique Lexicographically Sortable Identifier) — time-sorted by default
- **Table**: Logical grouping (like a collection or bucket)
- **Data**: JSON payload (`serde_json::Value`)
- **Timestamps**: `created_at` and `updated_at` (UTC)

## Tables

Tables are created on first insert — no schema definition required.
Internal tables (prefixed with `_`) are used by Axil for metadata,
sessions, entities, etc. Tier-2 Extensions own prefixed tables in the
core `.axil` file (e.g. `_checkpoint_records`, `_dep_docs`,
`_idx_code_proxies`).

## Companion files

Tier-1 Engines store their data in companion files alongside
the main database:

| File | Owner | What it holds | Safe to delete? |
|------|-------|---------------|-----------------|
| `memory.axil` | core (redb) | All records, prefixed internal tables, Extension tables | **No** — the database itself |
| `memory.axil.vec` | `axil-vector` | HNSW graph + raw embeddings (mmap-friendly) | Yes — re-embed source records via `axil heal --reindex` |
| `memory.axil.graph` | `axil-graph` | Edge index for fast neighbor lookup | Yes — rebuilds from edge records |
| `memory.axil.fts/` | `axil-fts` | Tantivy index directory | Yes — rebuilds from record text via `axil heal --reindex` |
| `memory.axil.ts` | `axil-timeseries` | Time-series index (created_at b-tree) | Yes — rebuilds on demand |

### Why the split

Companion files exist for two reasons:

1. **Concurrency.** Vector search and FTS commits are expensive
   compared to a redb point-write. Keeping them in separate files lets
   each engine manage its own write path without holding the core redb
   transaction open.
2. **Format independence.** Tantivy needs a directory; HNSW prefers an
   mmap-able binary blob. Forcing them through redb would lose those
   properties.

Conceptually `memory.axil` and its companions are **one logical
database**. `Axil::open()` is the master that creates and coordinates
all of them. You should treat the whole set as a unit.

### Companion file lifecycle

- **Created** by `Axil::open(path).with_vector(...).with_graph()....build()` — the builder calls each engine's `init()` which creates its companion if missing.
- **Opened** read-only or read-write according to how the engine was attached.
- **Closed** when the `Axil` value drops. Cleanup is best-effort; redb's WAL is durable across crashes.

### Portability — what to copy

To move or back up a database, copy **the main file and each
companion** as one unit. Prefer the explicit list — a `memory.axil*`
glob silently over-captures unrelated files (e.g. `memory.axil2`,
`memory.axil.bak`, Tantivy temp files) when the directory holds more
than one database:

```bash
# Recommended — explicit list, no over-capture risk
cp -a memory.axil memory.axil.vec memory.axil.graph memory.axil.fts memory.axil.ts /backup/
```

Copying only `memory.axil` and re-opening will work but you'll lose:
- The HNSW graph structure (rebuilds, but slow on large vector sets)
- The FTS index (rebuilds from record text)
- The graph edge index (rebuilds from `_edges` records)

Axil has no internal atomic snapshot today. `axil branch create <name>`
sequentially `fs::copy`s the main `.axil` and each companion, with no
locking or snapshot boundary, so concurrent writers can produce a
mixed-time branch. Quiesce writers before copying, or use a
filesystem-level snapshot (ZFS, Btrfs) to get true atomicity.
(`axil snapshot` is a metrics-only command for trend tracking, not a
data snapshot — see
[Memory Hygiene](../advanced/memory-hygiene.md).)

## ACID guarantees

- **Atomicity**: Mutations to the core database are atomic via redb transactions
- **Consistency**: Schema-free — JSON records are always valid
- **Isolation**: One writer process at a time. redb takes an **exclusive** OS
  lock when a process opens the core `.axil` for writing, so a second writer
  fails fast with `AxilError::Busy` (`is_busy()` returns true) — there is **no**
  shared-WAL coordinator brokering concurrent writers. The exclusive lock also
  blocks a *read-only* open (redb's shared lock can't coexist with it), so a
  reader cannot read *through* a live writer. Because writers are short-lived,
  the hot read CLI commands (`boot`/`get`/`list`/`recall`/`code-search`) do a
  bounded busy-retry on the writable open first; if the writer is still active
  after the retry budget, they fall back to a read-only open of the last
  committed state (which succeeds only in the gap between writer sessions).
- **Durability**: fsync on commit (redb default)

Companion engines maintain their own consistency relative to the core.
A crash mid-write can leave a companion behind its core; `axil heal
--reindex` rebuilds drifted companions from the canonical records in
`memory.axil`.

## Sizes at scale

Rough magnitudes (see [Indexing & Scale](../advanced/indexing-and-scale.md) for measured numbers):

| Component | What dominates | Typical at 10k records |
|-----------|----------------|------------------------|
| `memory.axil` | record JSON + indexes | 10-50 MB |
| `memory.axil.vec` | 384-dim float32 embeddings | ~15 MB (fp32) / ~4 MB (int8) |
| `memory.axil.graph` | edge records | 1-5 MB |
| `memory.axil.fts/` | Tantivy postings + positions | 20-100 MB |

The FTS directory tends to be the largest at scale; `int8` embeddings
cut the vector file by ~4×.

## Branching

Databases can be branched for experimentation:

```bash
axil branch create experiment    # Full copy (all companion files)
axil branch diff experiment      # Compare table counts
axil branch merge experiment     # Merge back with conflict resolution
axil branch delete experiment    # Clean up
```

Branches are independent file copies — no shared state, no soft-links.
Disk-cheap for small DBs, proportional to total companion size for
large ones.
