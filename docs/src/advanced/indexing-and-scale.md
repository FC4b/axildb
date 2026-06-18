# Indexing & Scale

This page covers how Axil keeps its indexes fresh when code changes,
what the measured times look like, where the comfortable scaling ceiling
sits, and how to push past it on large codebases (100k LOC → 1M+ LOC).

Numbers here come from the Phase 13b A/B benchmarks and Phase 8b performance
work, not extrapolation. Where a number is extrapolated, it is labelled.

## Three indexes, three cadences

Axil maintains three independent indexes over your project. Each has its
own drift-detection and refresh strategy:

| Index | Trigger | Granularity | Background? |
|-------|---------|-------------|-------------|
| Source code (`_idx_code_proxies`, FTS, vectors) | `axil index` | Per-file content hash | Manual |
| SCIP code-graph (`_entities`, edges) | `axil scip refresh --if-stale` | `.scip` file mtime/size | Yes (via brain hook) |
| Dep docs (`_dep_docs`) | `axil deps refresh --if-stale` | Per-dep manifest hash | Yes (via brain hook) |

### Source-code index

`axil index` runs **incremental by default**. The pipeline:

1. Walk the project respecting `.gitignore` and `.axilignore`
   ([crates/extensions/axil-indexer/src/scanner.rs:282](../../crates/extensions/axil-indexer/src/scanner.rs#L282))
2. For each file: read bytes, hash, compare against the
   `content_hash` stored in `_idx_files`. Unchanged files do
   **zero** downstream work — no parse, no embed, no FTS update
   ([crates/extensions/axil-indexer/src/indexer.rs:268-335](../../crates/extensions/axil-indexer/src/indexer.rs#L268))
3. For each changed file: re-parse, then run the proxy pipeline through
   `ProxyDedupCache` — a symbol whose `proxy_text` AND nav metadata are
   identical to the stored version skips the (expensive) ONNX embed +
   FTS index calls. Pure line shifts call `db.update` for nav fields
   only ([crates/extensions/axil-indexer/src/indexer.rs:87](../../crates/extensions/axil-indexer/src/indexer.rs#L87))
4. Detect deleted files; prune their proxies + reverse refs

Force a full rebuild with `axil index --full`. Use this after schema
changes, embedding model swaps, or if the hash table itself gets out of
sync.

**There is no file-watcher yet** for source code — you have to invoke
`axil index` (a pre-commit hook or editor save hook is the usual pattern).
The other two indexes have background refresh; source code doesn't.

### SCIP code-graph

`axil scip refresh --if-stale --in-background --quiet` runs from the
PreToolUse brain hook on the agent's first tool call. Mechanics:

- Lockfile at `.axil/scip-refresh.lock` so concurrent agents don't
  collide
- Child runs under `nohup` so it survives the parent shell exiting
- Size+mtime stabilization gate prevents ingesting a half-written
  `.scip` file
- `--max-age-days N` controls staleness threshold (default 1 day)
- Auto-detects language from repo markers (Cargo.toml, package.json,
  pyproject.toml, go.mod, pom.xml) and runs the right indexer
  (`scip-rust`, `scip-typescript`, `scip-python`, `scip-go`,
  `scip-java`)

Net effect: SCIP refresh is opportunistic and never blocks the agent.
First tool call kicks it off; results are available by the second or
third call.

### Dep-doc index

The PostToolUse brain hook fires a detached `axil deps refresh --if-stale`
whenever the agent edits a manifest or lockfile (Cargo.toml,
package.json, pyproject.toml, go.mod, pom.xml, plus their lockfiles).

Drift is gated on a content-hash of the manifest stored in
`_dep_manifests`. Only deps whose pinned version actually changed
re-ingest. A version bump archives the old version's chunks
(`archived: true`) rather than deleting — preserving migration history.

## Measured times

A/B-measured on real hardware (Phase 13b.10 + 13b.12 benchmarks):

| Workload | Time | Notes |
|----------|-----:|-------|
| Full index, Axil dogfood (233 files, 895 symbols, **no embedder**) | **25 s** | 13b.12 |
| Full index, synthetic (1000 files, 10k symbols, no embedder) | **127 s** | 13b.10 |
| Incremental index, synthetic corpus, **no changes** | **507 ms** | 13b.10 |
| Full index, synthetic (~11k proxies, **WITH embedder**, CPU ONNX) | **~20 min** | 13b.12 |
| Single memory `store` (auto-sync proxies + refs) | **67 ms** | 13b.10 |
| `db.recall` p95, 11k proxies, FTS+vector+graph+ts active | **1 ms** | 13b.12, in-process |
| `db.recall` p95, 3.4k proxies + graph walk | **17 ms** | 13b.12 |
| CLI invocation overhead | **+50 ms** | per invocation |

The headline trade-off: **ONNX embedding dominates indexing cost**. The
same 11k-proxy corpus is **25 s without an embedder vs ~20 min with**.

The honest follow-up: for *code* recall specifically, the embedder
didn't add much quality on the Axil dogfood — FTS + SCIP graph alone
gave 20% top-1 hit-rate vs 0% baseline (Phase 13b.12). That's why the
project itself runs without an embedder configured for code.

## The comfortable scaling ceiling

From the measurements, comfortable zone is **~50k proxies / ~500k LOC**:

- Incremental stays sub-second on no-change runs
- Full rebuild under 2 min without embedder, ~30 min with
- Query p95 stays in single-digit ms

Past that, ONNX embedding wall-clock grows super-linearly on CPU. HNSW
*search* stays log-time, but HNSW *build* slows past ~100k vectors.

## Handling 1M LOC

Rough proxy count for 1M LOC: ~150k-300k proxies depending on language
density. Linear extrapolation from the 11k-proxy numbers:

| Path | Estimate (extrapolated) |
|------|------------------------:|
| Full index, **no embedder** | ~10-15 min |
| Full index, CPU embedder, no quantization | ~8-10 hours |
| Full index, int8 + batch embed + write buffer | ~1-2 hours |
| Full index, GPU embedder | tens of minutes |
| Incremental, ~5% files changed | ~30-60 s |
| Query p95 (HNSW log-scale) | ~5-10 ms |

These are not measured — Axil's largest measured corpus is 11k proxies.
Use as planning ballpark, not promise.

### Playbook for large codebases

**1. Skip the embedder for code.** Configure your code DB with
`with_fts()` and `with_graph()` but no `with_vector()`. You lose
semantic-paraphrase matching ("auth bug" → "login failure") but you
keep:

- Symbol-exact lookup (FTS code tokenizer preserves identifiers)
- Structural queries via proxies (file/symbol/section)
- SCIP graph traversal (calls/refs/impls)
- All the multi-signal scoring except the vector signal

This is what Axil itself does. It's the single biggest scale win.

**2. If you do want embeddings, use int8.** `bge-small-en-v1.5-int8` is
~3× faster than fp32 with minimal quality loss (Phase 8b.5). Set in
`axil.toml`:

```toml
[index]
embedding_model = "bge-small-en-v1.5-int8"
```

**3. Enable deferred indexing.** `WriteBuffer` batches up to 1000
records / 10MB before flushing to plugins — 3-5× faster bulk ingestion
(Phase 8b.11). `insert_batch()` uses it by default; single inserts
still index immediately to preserve interactive responsiveness.

**4. Batch embedding.** Multiple texts per ONNX inference call (Phase
8b.3) cuts per-call overhead substantially. Use the batch insert path
for bulk loads.

**5. Aggressive `.axilignore`.** This is the second-biggest win. Most
"1M LOC" repos are 70%+ vendored code (node_modules, target, vendor,
.venv). Same syntax as `.gitignore`:

```
# .axilignore
node_modules/
target/
vendor/
.venv/
**/generated/
**/*.pb.go
fixtures/
```

The walker already respects `.gitignore` automatically, so `.axilignore`
is for things you want to commit but skip indexing.

**6. Mmap vectors.** Already on by default (Phase 8b.12). Zero-copy
access to the HNSW file means startup doesn't read the whole index into
memory.

**7. Run SCIP refresh in the background.** Already the default via the
brain hook. Don't make the agent wait for fresh SCIP.

**8. Don't run gte-modernbert on CPU.** 8192-context attention is O(n²);
Phase 15 measured **12+ hours for 50 questions** on CPU. Either run on
GPU or stick with BGE-small.

**9. Periodic `axil heal` / `axil compact`.** redb fragments over many
writes. The auto-compact threshold defaults to 1000 deletes; raise or
lower in `axil.toml`:

```toml
[healing]
auto_compact_threshold = 1000
```

**10. Memory branching.** Phase 5d added branch primitives. For a
monorepo where cross-workspace recall isn't useful, give each workspace
its own `.axil` file rather than fighting one giant index.

## Gotchas to know

- **HNSW build time** grows super-linearly past ~100k vectors. *Search*
  stays log-time; *adds* are fine; full *rebuild* is what hurts.
  Mitigation: incremental adds, not rebuilds. Schema/model changes
  force a rebuild — plan for the wall-clock.
- **redb file size** at 1M LOC: rough estimate 2-5 GB depending on what
  you embed. Snapshots and backups get proportionally slower.
- **No source-code file-watcher yet.** SCIP and dep-docs have hooks;
  the source indexer doesn't. Wire `axil index` to a pre-commit hook
  or editor save action.
- **`axil index --full` clears the index tables before re-parsing.**
  Don't run it during an agent session expecting concurrent reads.
- **The 20-min/11k-proxy embedder number is on Mac CPU.** A GPU or a
  smaller model collapses it. We just haven't measured those points.

## What's documented vs. measured vs. extrapolated

Honesty matters here.

- **Measured**: everything in the "Measured times" table, plus all
  three-index mechanics. Source: Phase 13b.10 + 13b.12 + Phase 8b
  benchmark sessions.
- **Documented but not benchmarked end-to-end**: 1M LOC numbers, GPU
  embedding times, int8 + batch + write-buffer combined wall-clock.
  These come from linear extrapolation and the per-feature gain claims
  in Phase 8b.
- **Known gap**: no measurement past 11k proxies. If you push Axil to
  100k+ proxies, please share numbers — they should land in the
  [Evaluation Log](./eval-log.md).

## See also

- [Retrieval Pipeline](./retrieval-pipeline.md) — how the indexes are queried
- [Performance](./performance.md) — Phase 8b optimizations in detail
- [Evaluation Log](./eval-log.md) — version-over-version benchmark history
- [Storage Model](../concepts/storage.md) — the `.axil` file and companion-file layout
- [Engines (Storage Plugins)](../concepts/plugins.md) — which engines to enable/disable for code-only workloads
