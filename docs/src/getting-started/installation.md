# Installation

## Prebuilt binary (recommended)

A prebuilt `axil` — no Rust toolchain, no multi-minute compile, and the
**onnxruntime shared library is bundled next to the binary** so vector search
and embeddings work out of the box on every platform (including Windows). The
prebuilt archive ships the **default** feature set — every component in the
table below except `rerank`, `web-docs`, and `otel`.

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

On first use with vector search, Axil downloads the default embedding model (`bge-small-en-v1.5`, ~33MB). Models are cached in `~/.axil/models/`.

Available models:

| Model | Dimensions | Size | Quality |
|-------|-----------|------|---------|
| `bge-small-en-v1.5` | 384 | 33MB | Good (default) |
| `bge-base-en-v1.5` | 768 | 130MB | Better |
| `nomic-embed-text-v1.5` | 768 | 130MB | Best |

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
