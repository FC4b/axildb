# Code Search

Code-aware retrieval over a project that has been indexed with
`axil index`. These commands query `_idx_code_proxies` (structure-aware
proxies for files / symbols / sections) plus the SCIP code-graph when
present.

For a deep dive on the retrieval pipeline behind these, see
[Retrieval Pipeline](../advanced/retrieval-pipeline.md). For when to
re-index and how it scales, see [Indexing & Scale](../advanced/indexing-and-scale.md).

## Generating the SCIP code-graph

The `--trace-graph` / cross-reference features above need a **SCIP index**
(call/reference/implements edges). One command generates *and* ingests it:

```bash
axil scip refresh        # detect language(s) → run the indexer → ingest, in one step
axil scip status         # what's detected, which indexers are on PATH, file age
```

> **One call for both layers:** [`axil reindex`](#refresh-everything-axil-reindex)
> rebuilds the structural proxy index (what `code-search` queries) **and** the
> SCIP graph together — use it when you just want "make my code-knowledge
> current" without remembering two commands.

**Is it automatic?** Mostly, yes:

- `axil install --claude-code --bootstrap` builds the initial index at install time.
- The brain hook then runs `axil scip refresh --if-stale --in-background --quiet` on
  the first tool call each session — opportunistic, non-blocking, and a no-op
  (<50ms) when the index is fresh (≤14 days old).

**The one prerequisite:** `scip refresh` orchestrates an *external* indexer that
must be on your `PATH`. If it's missing, that language is skipped with an install
hint (no error) — so auto-refresh only kicks in after you've installed the
indexer for your language once:

| Language | Indexer binary | Install |
|----------|----------------|---------|
| Rust | `rust-analyzer` | `rustup component add rust-analyzer` |
| TypeScript / JS | `scip-typescript` | `npm install -g @sourcegraph/scip-typescript` |
| Python | `scip-python` | `pipx install scip-python` |
| Go | `scip-go` | `go install github.com/sourcegraph/scip-go/cmd/scip-go@latest` |
| Java | `scip-java` | `brew install sourcegraph/scip/scip-java` |

Run `axil scip status` or `axil doctor` to see which indexers are present and the
exact install command for any that are missing. SCIP is **optional** — `code-search`
and `code-context` work on the `axil index` proxy layer alone; SCIP just adds the
graph edges that power `--trace-graph` and `recall-for-entity`. Full reference:
[Diagnostics → scip](./diagnostics.md#scip).

## Refresh everything: `axil reindex`

Code-knowledge lives in two layers — the **structural proxy index** (`axil index`,
what `code-search` queries) and the **SCIP graph** (`axil scip refresh`, the
call/reference edges). `axil reindex` refreshes both in one call so you don't
have to run — or remember — two commands.

```bash
axil reindex            # proxy index (incremental, foreground) + scip refresh --if-stale (background)
axil reindex --full     # full proxy rebuild + unconditional scip refresh
axil reindex --no-scip  # proxy index only (skip the graph)
axil reindex --wait     # block until the SCIP refresh finishes (foreground)
```

By default the proxy index runs in the **foreground** (fast, incremental) and the
heavier SCIP refresh is **spawned in the background** — so the command returns in
seconds and the graph edges become current a moment later. Pass `--wait` when you
need the graph guaranteed up to date before the command returns (e.g. a deliberate
pre-work refresh).

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `[PATH]` | `.` | Project directory to index |
| `--full` | off | Force a full proxy re-index **and** an unconditional SCIP refresh (skip incremental + staleness detection) |
| `--no-scip` | off | Only rebuild the proxy index; skip the SCIP graph refresh |
| `--wait` | off | Run the SCIP refresh in the foreground and block until it finishes, instead of spawning it in the background |

The output is a combined JSON object — `proxies` (the index result) and `scip`
(the refresh status: `spawned`, `skipped`, or the full ingest report under
`--wait`). SCIP still requires its [external indexer](#generating-the-scip-code-graph)
on `PATH`; when one is missing, `reindex` still rebuilds the proxy layer and the
`scip` block reports the skip.

> When to use which: reach for `axil reindex` for the everyday "refresh my
> code-knowledge" sweep; drop to `axil index` or `axil scip refresh` when you want
> to refresh exactly one layer (e.g. `axil scip refresh --language typescript`).

## `code-search`

Find file/symbol/section proxies matching a query. Compact, token-efficient
output suitable for agent context injection.

```bash
axil code-search "auth timeout" --top-k 5
axil code-search "WriteBuffer" --top-k 3 --trace-graph
axil code-search "RRF fusion" --json
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--top-k <N>` | `5` | Max results to return |
| `--trace-graph` | off | Walk SCIP edges (`calls`, `references`, `implements`, `type_of`, `defined_in`) from matched proxies; each neighbor's `why` field shows whether it came from direct search or graph expansion |
| `--json` | off | Emit raw JSON instead of the compact text table |

### Output format

Default text mode emits one line per hit:

```
crates/axil-core/src/scoring.rs:43 ScoreWeights — matched via vector
crates/axil-core/src/scoring.rs:255 fuse_signals — matched via full-text proxy
docs/src/api/query-builder.md:64 Scoring — matched via full-text proxy
```

Each line includes the file path, line number, breadcrumb/symbol, and
the retrieval source that placed it in the result set. JSON mode adds
the full proxy payload (proxy_id, canonical_id, score, kind, etc.).

## `code-context`

Assemble a coding-task context block — code proxies + relevant memories
+ rules + recent changes — within a token budget. Use this when starting
work on a task and you want a single recall call that returns
everything an agent needs.

```bash
axil code-context --task "add retry logic to the HTTP fetcher" --budget 2000
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--task <STR>` | required | Task description / question |
| `--budget <N>` | `2000` | Token budget for the assembled context |
| `--context-format <FMT>` | `compact` | `compact` = lean pointer lines; `json` = full bundle |

By default the output is a **compact** block — ranked `path:line symbol — why`
pointer lines for the relevant code, graph neighbors (callers/callees), and
pointer-attached memories. This is ~10× smaller than the full JSON and is the
right shape for locating code (see [Context Economics](../advanced/context-economics.md)).

Pass `--context-format json` for the full structured bundle, which also
includes vector-similar context, keyword module matches, active rules,
recent changes, and per-section token accounting (this is what the MCP
`code_context` tool returns).

For pure "where is X" lookups, prefer [`code-search`](#code-search) /
[`fts`](./diagnostics.md) — one compact line per hit. Reserve `code-context`
for assembling a one-shot task brief.

## `explain-code-hit`

Explain *why* a particular proxy matched a query. Useful when a
recall result looks surprising or when tuning the index.

```bash
# By proxy record id
axil explain-code-hit 01KQ58XC15HKS9AZ3JVXZE4S50 --query "WriteBuffer"

# Or by data-field proxy_id
axil explain-code-hit 35231d68c38b6bb49859f205b8617b48
```

### Flags

| Flag | Effect |
|------|--------|
| `--query <STR>` | Original query used at recall time; omitting limits the explanation to per-proxy facts |

Output shows the score breakdown (vector similarity, FTS BM25, keyword
overlap, recency, graph boost), the proxy's breadcrumb, and any
attached `code_refs` from memory records.

## `recall-for-file`

Surface memories (decisions, errors, context) relevant to a specific
file path. Designed for hooks that want to inject context when the
agent opens or edits a file.

```bash
axil recall-for-file crates/axil-core/src/scoring.rs --top-k 5
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--top-k <N>` | `5` | Max results to return |

Searches the `decisions`, `errors`, and `context` tables for mentions
of the file path or filename, plus memories with `code_refs` pointing
to proxies in this file. With SCIP enabled, the result set also
includes memories about symbols defined in the file (via the canonical
entity bridge).

## `recall-for-entity`

Surface memories about a symbol — and, with `--trace-graph`, memories
about its callers, callees, and impls.

```bash
axil recall-for-entity login --top-k 10
axil recall-for-entity WriteBuffer --trace-graph --depth 2
axil recall-for-entity recall --scope cargo:axil-core --scope cargo:axil-cli
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--top-k <N>` | `10` | Max records to return |
| `--depth <N>` | `1` | Max hop depth for graph traversal |
| `--edge-types <CSV>` | `calls,references,implements,type_of,defined_in` | Which SCIP edge types to follow |
| `--scope <STR>` | (repeatable) | Resolve the display name through these scopes — useful for cross-language disambiguation (`cargo:axil-core` vs `npm:axil-ui`) |
| `--trace-graph` | off | Annotate each hop with the layer (`_idx_files` or `_entities`) and confidence |

### Cross-language disambiguation

When the same symbol name exists across languages (e.g. Rust `login` vs
Python `login`), the SCIP index stores a `lang_hint` on each entity.
`--scope` filters by the canonical scope (e.g. `cargo:my-crate`,
`npm:my-package`, `pypi:my-module`) so you get the right hit.

## MCP parity

`code-search` and `code-context` are also surfaced as MCP tools
(`code_search`, `code_context`) for agents speaking the MCP protocol.
The flag set and semantics match the CLI.

## See also

- [Retrieval Pipeline](../advanced/retrieval-pipeline.md) — fusion + scoring underneath
- [Indexing & Scale](../advanced/indexing-and-scale.md) — when to re-index, how it scales
- [Memory Commands](./memory.md) — non-code recall (`axil recall`, `axil boot`)
- [Diagnostics](./diagnostics.md) — `axil doctor`, SCIP status checks
