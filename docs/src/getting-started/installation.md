# Installation

## Prebuilt binary (recommended)

A prebuilt `axil` — no Rust toolchain, no multi-minute compile, and the
**onnxruntime shared library is bundled next to the binary** so vector search
and embeddings work out of the box on every platform (including Windows). The
prebuilt archive ships the **default** feature set — every component in the
table below except `rerank`, `web-docs`, and `otel`.

### One-line install (shell)

Linux/macOS (POSIX sh — needs only `curl` + `tar`):

```bash
curl -fsSL https://raw.githubusercontent.com/FC4b/axildb/main/scripts/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/FC4b/axildb/main/scripts/install.ps1 | iex
```

Each script detects the platform, downloads the archive from the latest
release, and installs into `$AXIL_HOME/bin` (default `~/.axil/bin`), printing
any PATH step you need. Prebuilt archives cover Linux and Windows on x86_64 and
aarch64, and macOS on Apple Silicon; on an Intel Mac, build
[from source](#from-source) instead. Review the script before piping it into a shell —
it is short on purpose.

### Via cargo-binstall

[`cargo binstall`](https://github.com/cargo-bins/cargo-binstall) fetches the
prebuilt archive and installs the `axil` binary — no source build:

```bash
cargo binstall axildb
```

### Direct download

Grab the archive for your platform from the
[latest release](https://github.com/FC4b/axildb/releases/latest) —
`axildb-<target>.tar.gz` (or `axildb-<target>.zip` on Windows) — extract it, and
put the `axil` binary on your `PATH`. The `onnxruntime` shared library ships
**inside the archive next to `axil`**; keep the two together so the loader
resolves the bundled runtime first.

## From source

`cargo install` and `cargo build` compile from source — they need a C toolchain
and take a few minutes. Prefer the prebuilt binary (`cargo binstall` / archive
download) above unless you need a custom feature set.

```bash
cargo install axildb                                            # from crates.io, default features

# from a checkout:
git clone https://github.com/FC4b/axildb.git
cd axildb
cargo install --path crates/adapters/axil-cli                   # default features
cargo install --path crates/adapters/axil-cli --features full   # everything
```

`cargo build --release -p axildb` works too; the binary lands at `target/release/axil`.

> **Windows + ONNX.** A from-source `cargo install axildb` copies only `axil.exe`
> to `~/.cargo/bin` and leaves the `download-binaries`-produced `onnxruntime.dll`
> behind in `target/`. The Windows loader then finds a stale system
> `onnxruntime.dll` and the embedder fails ONNX init
> (`Failed to initialize ORT API … version [22] is not supported`). See the
> [Windows + ONNX](#windows--onnx) note below for the fix — or just use the
> prebuilt binary (`cargo binstall` / archive download), which bundles a
> matching runtime.

## Mixing binary versions on one database

Agent hooks run the `axil` that comes first on your `PATH` (usually
`~/.cargo/bin/axil`). If you or an agent also run a build from a checkout
(`./target/release/axil`) against the same database, two versions take turns
writing it. So whenever you start using a build from a checkout, install it on
your `PATH` at the same time:

```bash
cargo install --path crates/adapters/axil-cli   # from the same checkout
```

Mixing versions doesn't corrupt anything, but the graph file makes it cost
time. Newer builds keep per-node adjacency tables next to the edges in
`memory.axil.graph`, and older builds write edges without updating them. The
first graph operation of a newer build after an older one has written brings
the tables up to date:

| What the older build did since | What the newer build does first | CPU on the 161k-edge dogfood snapshot |
|---|---|---|
| Nothing | Checks a stamp (three lookups) | — |
| Only added edges (`link`, `store`, `ingest-scip`) | Reads just the added edges | 0.03 s after one `link` |
| Deleted any edge (`delete` of a linked record, `unlink`, `compact`) | Decodes every edge and diffs them against the tables | 0.49 s |
| Nothing ever: the tables don't exist yet | Builds them once (322,746 entries) | 1.05 s |

For comparison, the older build spends 0.52-0.62 s of CPU loading every edge
on each command that touches the graph. These are user + sys seconds measured
on a heavily loaded machine, so treat them as rough; see
`benchmarks/results/graph-adjacency-size-2026-09-30.json` in the repository.
The work runs in whichever process gets there first, often a hook with a short
timeout. If that process is killed partway through, the work it committed is
kept and the next one finishes the rest.

The tables also take room in the file. A file has two sizes: its length, which
`ls` and `axil info` show, and the disk it takes up, which `du` shows. redb
lengthens a file without writing the new part, and filesystems such as APFS
and ext4 give that unwritten part no disk until it is used. On the same
snapshot, building the tables fit in free pages the file already had: the data
in it went from 70.5 MB to 119.2 MB (redb's count of pages in use) while its
length and its disk use both stayed at 120.7 MB. A later `ingest-scip` that
added 25,507 edges then took the length to 241.4 MB and the disk use to
144.0 MB; the older build stays at 120.7 MB for both. redb doubles a file of
this size when it runs out of free pages.

`axil compact` compacts the graph file only when at most 40% of the disk it
takes up holds data, as after deleting many edges. The files above hold data
in 97-99% of theirs, so it leaves them alone. Compacting copies of them
anyway gave back 1.3-3.1% of their disk use and, because redb doubled each
packed file again when closing it, left them 16-97% longer.

## Picking components

Axil is assembled from three extensibility tiers — [Engines, Extensions, and Adapters](../extending/overview.md) — each behind a compile-time Cargo feature on `axil-cli`:

| Tier | Feature | Description | In `default` | In `full` |
|------|---------|-------------|:---:|:---:|
| Core | `core` | Core storage (redb) — always present | ✅ | ✅ |
| Engine | `vector` | Vector search (HNSW) → `*.axil.vec` | ✅ | ✅ |
| Engine | `embed` | Built-in ONNX embedder (BGE family) — implies `vector` | ✅ | ✅ |
| Engine | `graph` | Knowledge graph (edges, traversal) → `*.axil.graph` | ✅ | ✅ |
| Engine | `fts` | Full-text search (Tantivy) → `*.axil.fts/` | ✅ | ✅ |
| Engine | `timeseries` | Time-series queries → `*.axil.ts` | ✅ | ✅ |
| Extension | `indexer` | Structural code proxies (`code-search` / `code-context`) | ✅ | ✅ |
| Extension | `scip` | SCIP code-graph ingest — implies `graph` | ✅ | ✅ |
| Extension | `deps` | Dependency doc memory (version-pinned library docs) | ✅ | ✅ |
| Extension | `checkpoint` | Session checkpoints (structured resume state) | ✅ | ✅ |
| Extension | `memory` | Agent memory patterns (TTL, superseding, sessions) | ✅ | ✅ |
| Extension | `rerank` | Cross-encoder reranking — implies `indexer` | ❌ | ✅ |
| Adapter | `mcp` | MCP server (stdio) | ✅ | ✅ |
| Adapter | `ql` | AxilQL query language + REPL | ✅ | ✅ |
| Adapter | `http` | HTTP API server (axum) | ✅ | ✅ |
| Opt-in | `llm-http` | OpenAI-compatible `LlmProvider` (Path B intelligence) | ✅ | ✅ |
| Opt-in | `web-docs` | HTTP doc fetcher for `deps` | ❌ | ❌ |
| Opt-in | `otel` | OpenTelemetry instrumentation | ❌ | ❌ |

> `web-docs` and `otel` are excluded from `full` on purpose: Axil is offline-first, and observability should cost zero unless you ask for it. GPU execution providers (`cuda`, `directml`) live on `axil-vector` and are likewise explicit opt-ins.

### Select all

```bash
cargo install axildb --features full
```

### Pick and choose

```bash
cargo install axildb --no-default-features --features "core,vector,embed,graph,mcp"
```

Feature dependencies are wired in — `embed` pulls `vector`, `scip` pulls `graph`, `rerank` pulls `indexer`, `web-docs` pulls `deps` — so you can't compose a broken set.

### Minimal

```bash
cargo install axildb --no-default-features --features core
```

A pure document store: CRUD, queries, diagnostics — no vectors, graph, or FTS.

## Changing features later

Features are compile-time, so changing them means a rebuild — but the binary knows what it was built with and can compose the command for you:

```bash
axil features            # what's compiled in? (use --format table for humans)
axil features --wizard   # interactive picker → emits (and optionally runs) the cargo install command
```

The wizard seeds its selection from the current binary, enforces feature dependencies in both directions (dropping `graph` also drops `scip`), offers `a` / `d` / `m` presets (all / default / minimal), and finishes with the exact `cargo install … --force` command — run it on the spot or copy it for later. When run inside a source checkout it uses `--path crates/adapters/axil-cli`; otherwise it targets the published `axildb` crate.

## Embedding models

On first use with vector search, Axil downloads the default embedding model (`bge-small-en-v1.5`, ~128 MB). Models are cached in `~/.axil/models/`, and every file is checked against a pinned sha256 before it is installed.

Available models:

| Model | Dimensions | Download (approx.) | Quality |
|-------|-----------|------|---------|
| `bge-small-en-v1.5` | 384 | 128 MB | Good (default) |
| `bge-base-en-v1.5` | 768 | 416 MB | Better |
| `nomic-embed-text-v1.5` | 768 | 523 MB | Best |
| `granite-embedding-small-english-r2` (`granite-small`) | 384 | 188 MB | Experimental, see below |

The model is set with `[database] embedding_model` in `axil.toml`
(see [Configuration](configuration.md)). Changing it needs a rebuild of the
vector store; see [Switching embedding models](#switching-embedding-models).

### Switching embedding models

Vectors from two models can't be compared, even when they have the same
width: `bge-small-en-v1.5` and `granite-embedding-small-english-r2` both write
384 dimensions. So the vector store (`memory.axil.vec`) records the model
that built it, and every command that loads a model checks that record
against `[database] embedding_model` first. A store that holds another
model's vectors is refused, never mixed:

- each command prints one warning, naming the configured model, the model
  the store records (or that it records none), and the fix, and recall ranks
  without vectors (keyword, recency, graph and its other signals);
- nothing new is embedded into the store: `store --embed`, `embed`, and a raw
  `--vector` for the default space fail with the same message, as do the raw
  vector commands on it (`add-vector`, `similar`), and a plain `store` saves
  the record without a vector;
- `doctor` reports a `vector_engine` error, `boot` lists it under
  `degraded`, and the [daemon](../cli/daemon.md) leaves the database to
  one-shot commands, which report it every time, until it is fixed.

To switch, name the new model, then rebuild the store under it:

```bash
axil config set database.embedding_model granite-small
axil reembed
```

`axil reembed` (without `--field`) builds a new store beside the old one
(`memory.axil.vec-rebuild`) and records the new model in it. It re-embeds
every memory record, and every other row that had a vector: code-index rows,
dependency docs, recall chunks, checkpoints, cached answers, and the
episodes, sessions, entity facts, procedures and preferences of the memory
types, each from the text its writer embeds, and any other row (one stored
with `store <table> … --embed <field>`, or embedded with `axil embed <id>
<field>`) from its searchable text. An internal-table row that had no vector
keeps none, so a rebuild does not change which of those rows compete in
recall; one with no text to embed, such as a raw vector its caller supplied,
loses its vector, and the report counts it under `dropped`. The old store
stays in place, untouched, until the new one is complete, which then
replaces it in one rename. A failure, or an interruption at any point,
leaves the old store as it was: an unfinished `memory.axil.vec-rebuild` is
never read, and the next `reembed` deletes it. `--model` rebuilds under
another model than the configured one, after which every command refuses the
store until `axil.toml` names that model. To keep the old store instead, set
`embedding_model` back to the model the warning names.

`reembed` holds the database for the whole rebuild, which takes as long as
embedding every row does. Meanwhile other commands on it wait for the
writer lock and then fail as busy (`store` after about 10 s, `recall` at
once), and so do an agent's hooks, which stall or are killed for as long as
it runs: run it with no agent session or MCP server open on the database. A
second `reembed` of the same database stops at once, naming the one that
runs (it holds `memory.axil.reembed-lock`).

Earlier versions rebuilt the store in place and set the old one aside as
`memory.axil.reembed-bak`. If one of them was interrupted, that file is still
there, and which copy holds the right vectors depends on what was stored
since, so `reembed` stops and names both: move the backup back over
`memory.axil.vec` if nothing was stored since the interrupted run, otherwise
delete it, and run `reembed` again.

Each built-in model is recorded under its own name. bge-small's int8 export
(`bge-small-int8`) counts as a model of its own: its vectors approximate
bge-small's, but how closely has not been measured, so switching between the
two needs a rebuild too. A custom model (`axil model-add`) is recorded by the
path of its ONNX file, so moving the file counts as a change of model.

**Stores from before model records.** A store written by an older Axil names
no model, so the configured model is the only evidence of which one built
it. Such a store is taken to hold the configured model's vectors when it has
that model's width and the model is one Axil had then: `bge-small-en-v1.5`,
`bge-small-en-v1.5-int8`, `bge-base-en-v1.5`, `nomic-embed-text-v1.5`,
`bge-m3` or `gte-modernbert-base`. Nothing has to be done on upgrade, and
nothing is written until the next vector is, which records the model. Under
the default model this is silent; under any other, each command prints one
note (`[axil vector] note: …`) until that write, saying the store was taken
as that model's and naming `axil reembed` in case another model built it.
Meanwhile the [daemon](../cli/daemon.md) leaves the database to one-shot
commands, as it does for any open that prints: a daemon that meets the note
exits, and serves the database again only once a vector write has recorded
the model and `axil daemon start` starts it again.

Two configured models never take over such a store, and get the refusal
instead: granite, which came after model records, so no store from before
them holds its vectors; and a custom model `axil.toml` names, since before
model records the CLI and the MCP server never loaded a custom model named
there (bge-small embedded in its place), so the store may hold either
model's vectors. A store that holds no vectors yet is taken over by whichever
configured model of its width opens it.

A program using Axil as a library passes its model to its opens
(`with_embedder_model`, `VectorEngine::with_model`). When the database's
`axil.toml` names no model, the model passed in stands for the configured
one, with one difference: a custom model of the store's width takes it over
too, with the note, since that program really embedded with it, so it is the
best evidence of who built the store. When `axil.toml` names a model, the
CLI, the MCP server and an agent's hooks embed with that one, so an open
that passes any other model, built-in or custom, gets the refusal rather
than writing its vectors beside theirs.

> **Upgrade note.** A database whose `axil.toml` names a custom model was
> embedded with bge-small by earlier versions of the CLI and the MCP server,
> which did not load a custom model named there. After upgrading, its vector
> store is refused, with a warning naming `axil reembed`, and recall runs
> without vectors until `axil reembed` rebuilds the store with the custom
> model. A library caller that passed its custom model in directly keeps
> its store when `axil.toml` names no model: the next vector it writes
> records the model. When `axil.toml` names another model, the store is
> refused to it too, since the CLI may have added that model's vectors.

Only the configured model takes over a store from before model records. A
command that names a model with `--model` (`embed`, and `reembed --field …
--table …`, which keeps the other tables' vectors) can use a store only if
the store records that model, or holds no vectors yet for a scoped
`reembed`: a store from before model records is taken to hold the
configured model's vectors, so carrying them into a store that records the
named model would mix the two. `embed --model` with another model than the
configured one does not start a store either, since every open under the
configured model would refuse it. Both stop with the reason and the two
ways forward: drop `--model`, or name the model in `axil.toml` and run
`axil reembed`.

### granite-small is experimental

`granite-embedding-small-english-r2` is registered so it can be measured
against the default, not as a recommendation. The default stays
`bge-small-en-v1.5`. Before switching to granite, know two things:

- **Switching to or from granite needs a rebuild.** Nothing in a store's
  size tells granite's vectors from bge-small's; the model the store records
  does, and a store built by the other model is refused until `axil reembed`
  rebuilds it (see [Switching embedding models](#switching-embedding-models)).
- **The similarity thresholds were tuned on bge-small.** The 0.92
  auto-supersede threshold (`[healing] supersede_similarity_threshold`) and
  the 0.65 floor for possible-conflict candidates are absolute cosines set with
  bge-small. granite places records much closer together: across the 8,035
  same-table memory pairs of Axil's own project memory, the median cosine is
  0.80 under granite against 0.66 under bge-small, every pair clears 0.65, and
  1.8x as many pairs clear 0.92
  (`benchmarks/results/embedding-model-ab-2026-10-15.json`,
  `same_table_memory_pair_cosines`). Under granite the 0.65 floor stops
  filtering conflict candidates, and the 0.92 threshold treats more pairs as
  near-duplicates; neither has been re-tuned for it.

## Windows + ONNX

The prebuilt archives (via `cargo binstall` or direct download) **bundle a
known-good `onnxruntime.dll` next to `axil.exe`**, so the embedder loads the
sibling runtime first — no manual DLL copy, and no SmartScreen-free guarantee
aside (the binaries are provenance-attested but not Authenticode code-signed, so
an unsigned-publisher SmartScreen prompt is expected on first run).

If you instead built from source with `cargo install axildb`, only `axil.exe`
lands in `~/.cargo/bin` — the `download-binaries`-produced `onnxruntime.dll`
stays in `target/`, and the Windows loader falls back to a stale system copy
whose ORT API version `ort` can't use. The fix is to put a matching runtime next
to the binary:

```powershell
# copy the runtime the build already downloaded next to the installed binary
copy target\release\onnxruntime.dll $env:USERPROFILE\.cargo\bin\
```

Use ONNX Runtime ≥ 1.22 (API 22); `ORT_DYLIB_PATH` does **not** help under the
`download-binaries` feature. The prebuilt archives avoid all of this.

Both `axil doctor` and the embedder itself check for this setup on Windows and
fail with these instructions instead of crashing inside the ONNX runtime.

## Windows + scip-python

`@sourcegraph/scip-python` (the Python SCIP indexer `axil scip refresh` uses)
currently **crashes at startup on Windows** with:

```
SyntaxError: Invalid regular expression: /\/g: \ at end of pattern
    at ...PythonEnvironment.ts:4
```

This is an upstream bug — `new RegExp(path.sep, 'g')` with Windows' `\`
separator is an invalid regex — and it kills the CLI before it indexes
anything. `axil scip refresh` detects this crash signature and points here.

Until the upstream fix ships, patch the two installed bundles (idempotent —
re-run after any `npm update -g`):

```powershell
# %APPDATA%\npm\node_modules\@sourcegraph\scip-python\dist\{scip-python.js, scip-python-test.js}
# replace:  new RegExp(o.sep,"g")
# with:     new RegExp(o.sep==="\\"?"\\\\":o.sep,"g")
$dist = "$env:APPDATA\npm\node_modules\@sourcegraph\scip-python\dist"
foreach ($f in "scip-python.js", "scip-python-test.js") {
  (Get-Content "$dist\$f" -Raw) -replace [regex]::Escape('new RegExp(o.sep,"g")'),
    'new RegExp(o.sep==="\\"?"\\\\":o.sep,"g")' | Set-Content "$dist\$f" -Encoding utf8 -NoNewline
}
scip-python --version   # should print a version instead of a stack trace
```

Also note: npm installs `scip-python`/`scip-typescript` as `.cmd` shims.
`axil scip refresh`/`status` resolve those correctly (preferring a real
`.exe` when present, falling back to the `.cmd` shim).

## Verify installation

```bash
axil --version
axil features --format table   # confirm the components you expect
axil init ./test.axil
axil --db ./test.axil doctor
```

## Set up agent memory

Installing the binary is step one. To give a coding agent persistent memory,
run `axil install` in your project — it detects the agent tooling present and
wires up the auto-boot / auto-capture loop:

```bash
cd your-project
axil install                     # interactive wizard (detects .claude/, .codex/, …)
axil install --claude-code       # or name an agent explicitly
axil install --all               # every detected + supported agent
```

Claude Code, OpenAI Codex, GitHub Copilot CLI, Factory Droid, Google
Antigravity CLI, Qwen Code, and OpenCode are all supported from one shared
database — see [Terminal Agents](../agents/terminal-agents.md).

## As a Rust library

Add to your `Cargo.toml`:

```toml
[dependencies]
axil-core = "0.7"
axil-vector = { version = "0.7", features = ["embed"] }
axil-graph = "0.7"
axil-fts = "0.7"
```

Compile-time features control what *can* run; the builder controls what *does* run for a given database:

```rust
let db = Axil::open("./memory.axil")
    .with_embedder_model(EmbeddingModel::BgeSmall)?
    .with_graph_engine()?
    .with_fts_engine()?
    .build()?;
```
