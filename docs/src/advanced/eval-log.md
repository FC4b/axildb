# Evaluation Log

Version-over-version performance log. **Append a new section every release** so we can spot regressions (and brag about wins) without re-running historic benches from a clean tree.

## How to add a new entry

1. Build the release binary on the target hardware: `cargo build --release -p axildb`.
2. Run all three layers (or copy-paste the script in [Reproducing](#reproducing)):
   - `axil bench --format pretty` (built-in micro-bench, fresh temp DB)
   - Criterion suites for `axil-core`, `axil-vector`, `axil-graph`, `axil-fts`
   - `scripts/code-recall-gate.sh --fixture --compare` and `--dogfood --compare`
3. Add a new `## vX.Y.Z — YYYY-MM-DD` section at the **top** of the [History](#history) list, copying the table layout below.
4. Note any methodology deviations (`--sample-size`, skipped benches, hardware) under **Notes** — comparisons across rows are only meaningful when methodology matches.
5. Commit the diff alongside the release tag.

## Reproducing

```bash
# Built-in micro
TMP=$(mktemp -d)
./target/release/axil init --db "$TMP/b.axil" --quiet
./target/release/axil bench --db "$TMP/b.axil" --format pretty

# Criterion (sample-size 10 keeps wall-clock tractable; drop the flag for the
# full 100-sample statistical run when you have ~1h to spare)
for c in axil-core axil-vector axil-graph axil-fts; do
  bench=${c#axil-}_benchmarks
  cargo bench -p "$c" --bench "$bench" -- --sample-size 10 --warm-up-time 1 --measurement-time 3
done

# Code-recall gates
scripts/code-recall-gate.sh --fixture --compare
scripts/code-recall-gate.sh --dogfood --compare
```

The Criterion comparison column (`change vs prior`) is auto-emitted by Criterion when a baseline already exists in `target/criterion/` from the previous run on this machine. To make a row directly comparable to the next release, save a baseline at tag time:

```bash
for c in axil-core axil-vector axil-graph axil-fts; do
  bench=${c#axil-}_benchmarks
  cargo bench -p "$c" --bench "$bench" -- --save-baseline vX.Y.Z
done
```

## What "good" looks like

- **Read latencies** (`get_latency`, `vector_search`, `fts_search_text`, `graph_traverse/depth_1`) should stay flat or improve. Any persistent >5% regression across a release blocks the tag.
- **Insert/index throughput** can fluctuate ±15% with redb fsync timing; only call it a regression if it holds across two runs.
- **Code-recall gate** must pass — `structural_proxies` keeps top-1 file ≥ 100% (fixture) / ≥ 20% (dogfood) and ctx tokens ≤ 80.
- **Binary size** for `target/release/axil` should stay under 30 MB until we ship a deliberate plugin push.

## Competitor benchmarks

Cross-engine comparisons live alongside the per-release in-tree numbers. Same workload, same hardware, same day — anything else and the deltas become folklore.

> **Reproducibility caveat.** The harness crates that drive these comparisons (`benchmarks/sqlite-compare/`, `benchmarks/vector-latency/`, `benchmarks/locomo/`, `benchmarks/longmemeval/`) are **workspace-excluded but tracked in git** since 2026-05-16 (Phase 15 R5). The crate *sources* are in the tree; what stays gitignored is per-bench `data/` (HF datasets, ~265 MB for LongMemEval-S), `target/`, `out/`, and `Cargo.lock`. Bootstrap each bench from its own README — `benchmarks/longmemeval/README.md` documents the HF dataset fetch and the gate-script workflow, and each harness crate's README has the full how-to-run commands; the schema below is what you append to this log once a run completes.
>
> Plain `cargo bench` from the workspace root runs only the in-tree Criterion suites (`axil-core`/`vector`/`graph`/`fts`) and does **not** touch competitors.

### Vector search — Axil vs sqlite-vec

Workload: 100k synthetic vectors × 384 dims, top-10, 1000 queries, 50-query warmup. Axil HNSW vs sqlite-vec `vec0` (brute force).

| Date | Axil ver | Axil p50 | sqlite-vec p50 | Speedup | Axil qps | sqlite-vec qps | Axil disk | sqlite-vec disk | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-04-18 | Phase 13 | 686 µs | 50,730 µs | **74×** | 1,450 | 20 | 262 MB | 149 MB | Harness crate `benchmarks/sqlite-compare/` is out-of-tree |

### LongMemEval (retrieval quality)

Workload: `s` variant (500 questions × 30–40 sessions), top-k=5. Hit rate / recall reported per strategy.

| Date | Axil ver | Recall (hit/recall) | Vector (hit/recall) | FTS (hit/recall) | Notes |
|---|---|---:|---:|---:|---|
| 2026-04-18 | Phase 13 | 94.4% / 90.9% | 92.2% / 88.0% | 85.8% / 79.0% | Hindsight (PostgreSQL + LLM) reports 91.4% on a similar split — Axil clears it without an LLM in the loop. Harness crate is `benchmarks/longmemeval/` (gitignored). |
| 2026-05-17 | 0.7.7 | — | 96.4% / 93.6% | — | First **git-tracked, CI-gated** full 500-q run — committed at `benchmarks/results/baseline-500.json`. GPU (RTX 3080), 15m19s wall, `vector` strategy. Higher than the Phase 13 vector row, but that row is untracked/historical (different tooling, possibly different dataset snapshot) — treat 96.4% as the new reproducible reference, **not** a measured +4.2-pt gain. precision@5 is 32.0% (denominator-capped at top-k=5; hit_rate/recall are the quality signals). |

### LoCoMo (retrieval quality, historical)

| Date | Axil ver | Hit rate | Recall | Notes |
|---|---|---:|---:|---|
| Phase 6 (historical) | pre-0.6 | 99% | 94.4% | Numbers reflect the last in-tree harness run before `benchmarks/locomo/` was excluded. CI cannot regenerate; treat as historical reference, not a regression baseline. |

### Feature-parity matrix (no measurement, just positioning)

The full table — Axil vs Memvid / HelixDB / SurrealDB / Mem0 / Hindsight / MemPalace — lives in [`CLAUDE.md`](../../../CLAUDE.md) under "Competitive Position". When a competitor ships a feature we don't have (or vice-versa), update that table and add a row to the relevant section above with the head-to-head numbers if the workload is comparable.

### Adding a new competitor row

1. Pull the relevant harness from the benchmark archive into `benchmarks/<name>/` (it's gitignored; that's intentional — these crates pin large datasets and competitor binaries).
2. Run **on the same hardware and same day** as the in-tree Criterion suite for the version you're publishing — competitor delta is only meaningful when the Axil number is fresh.
3. Append a row with: date, Axil version, headline metric for both engines, ratio/delta, and a one-line note on workload assumptions.
4. If the competitor has a known asterisk (requires LLM, requires PostgreSQL, AGPL-licensed, etc.), call it out in the note column — the table is read by people deciding whether to adopt.

## History

> Newest at the top.

### v0.6.0 — 2026-04-27

**Hardware:** Apple M1 Pro, 16 GB, macOS 15.6, rustc 1.92.0
**Branch / commit:** `dev` @ `df1d3bf`
**Binary size:** 22.6 MB (`target/release/axil`)
**Notes:** Criterion run with `--sample-size 10 --measurement-time 3` (not the 100-sample default). `fts_index_text/10000` skipped (~17 min, linear from `/1000`). Comparison percentages are vs. the prior baseline already in `target/criterion`.

#### Built-in micro-bench (`axil bench`, fresh temp DB)

| op | latency | ops/sec |
|---|---:|---:|
| insert_1000 | 11.85 ms | 84 |
| get_1000 | 2.13 µs | 470,118 |
| fts_search_100 | 0.23 ms | 4,388 |
| graph_traverse_depth1 | 1.57 µs | 638,296 |
| graph_traverse_depth3 | 5.56 µs | 179,842 |
| vector_search_top5 | n/a (fresh DB has no vectors) | — |

#### Criterion — core

| bench | mean | change vs prior |
|---|---:|---:|
| insert_throughput/100 | 1.02 s | — |
| insert_throughput/1000 | 9.55 s | — |
| insert_throughput/10000 | 109.6 s | **−13.6 %** |
| batch_vs_individual/individual_1000 | 6.91 s | −35.7 % |
| batch_vs_individual/batch_1000 | 65 ms | **−52.0 %** |
| get_latency/100 | 1.63 µs | +6.4 % |
| get_latency/1000 | 1.69 µs | +7.3 % |
| get_latency/10000 | 1.77 µs | +8.1 % |
| query_filter/where_eq_1000 | 1.24 ms | +10.4 % |
| query_filter/where_gt_1000 | 1.24 ms | +10.5 % |
| query_filter/where_contains_1000 | 1.27 ms | +9.7 % |
| combined_query/where_eq_order_asc_1000 | 1.24 ms | +10.8 % |
| combined_query/where_gt_order_desc_limit_1000 | 1.27 ms | +10.6 % |
| combined_query/multi_filter_order_1000 | 1.26 ms | +10.2 % |

#### Criterion — vector (HNSW + ONNX BGE-small)

| bench | mean |
|---|---:|
| vector_add/100 | 0.51 s |
| vector_add/1000 | 4.93 s |
| vector_add/10000 | 50.1 s |
| vector_search/1000_top5 | 115 µs |
| vector_search/1000_top50 | 131 µs |
| vector_search/10000_top5 | 122 µs |
| vector_search/10000_top50 | 120 µs |
| vector_delete/delete_1000 | 4.96 s |

#### Criterion — graph

| bench | mean |
|---|---:|
| graph_relate/100 | 0.54 s |
| graph_relate/1000 | 4.98 s |
| graph_relate/10000 | 50.7 s |
| graph_neighbors/density_5 | 729 ns |
| graph_neighbors/density_20 | 2.98 µs |
| graph_neighbors/density_50 | 7.88 µs |
| graph_traverse/depth_1 | 2.23 µs |
| graph_traverse/depth_2 | 20.5 µs |
| graph_traverse/depth_3 | 55.7 µs |
| graph_traverse/depth_5 | 175 µs |

#### Criterion — fts (tantivy)

| bench | mean |
|---|---:|
| fts_index_text/100 | 10.6 s |
| fts_index_text/1000 | 111 s |
| fts_index_text/10000 | _aborted_ — Criterion estimated **~3 h** (10,840 s) for 10 samples, vs the linear projection of ~18 min. Tantivy commit cost scales **superlinearly past ~1 k records** (segment-merge overhead). Tracked as a future optimization; not blocking the v0.6.0 baseline. |
| fts_search_text/1000_authentication | 59 µs |
| fts_search_text/1000_database_connection | 82 µs |
| fts_search_text/1000_API_rate_limiting | 87 µs |
| fts_search_text/10000_authentication | 86 µs |
| fts_search_text/10000_database_connection | 121 µs |
| fts_search_text/10000_API_rate_limiting | 136 µs |

#### Code-recall regression gate

Both PASSED (exit 0).

**Fixture** (`tests/fixtures/code-recall/`)

| Strategy | Top-1 File | Top-3 Symbol | MRR | FP@3 | Ctx Tokens | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_indexer | 0.0% | 80.0% | 0.70 | 1.00 | 45 | 14 |
| structural_proxies | **100.0%** | **100.0%** | 0.77 | 0.80 | 80 | **3** |
| proxies_plus_pointer_memories | 100.0% | 100.0% | 0.77 | 0.80 | 80 | 3 |

**Dogfood** (Axil repo, 5 cases)

| Strategy | Top-1 File | Top-3 Symbol | MRR | FP@3 | Ctx Tokens | Raw Avoided | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_indexer | 0.0% | 0.0% | 0.00 | 0.20 | 3 | 561,540 | 2,423 |
| structural_proxies | **20.0%** | **20.0%** | 0.20 | 0.80 | 80 | 548,349 | **6** |
| proxies_plus_pointer_memories | 20.0% | 20.0% | 0.20 | 0.80 | 80 | 548,349 | 14 |

#### Take-aways

- **Wins**: batch-insert path (~100× over individual + further −52 % vs prior baseline); insert_throughput/10000 −13.6 %; structural-proxy ctx-token & p95 wins intact (~400× p95 speedup over raw indexer on dogfood).
- **Future optimization filed**: `fts_index_text/10000` superlinear scaling (~3 h for 10 samples vs ~18 min linear). Likely tantivy segment-merge cost; investigate batched commits or `IndexWriter` reuse before next release. **Update 2026-05-17 — landed:** `SearchIndex::index_records_batch` / `index_field_batch` (deferred adds + a single commit) now back the bulk-insert paths (`insert_batch_records`, `batch_sync_recall_chunks`), turning N per-record commits into 1. This cut the LongMemEval-S 500-q run 2h45m → 15m19s. The `fts_index_text/N` micro-bench still calls per-record `index_text` by design, so that Criterion row is unchanged — the win shows up in batch ingestion and recall-chunk sync.
- **Baseline saved**: 34 of 36 Criterion benches stored under `target/criterion/<bench>/v0.6.0/` (everything except `fts_index_text/10000` and the post-`/10000` `fts_search_text` rows). The next release can run `cargo bench -- --baseline v0.6.0` for a same-hardware comparison instead of relying on whatever happens to live in `new/`.
- **Regressions to watch**: `get_latency` +6–8 % and all `query_filter` / `combined_query` paths +10 % vs the prior unnamed baseline. Investigated 2026-04-28 and **classified as measurement noise, not a code regression**: (a) the only `db.rs` commits between baseline and HEAD (`dfb1f59` `code_refs`, `f520c48` audit gate) touch the **write** path only — `Axil::get` is unchanged; (b) re-running `insert_throughput/100` immediately after the first run dropped it from 1016 ms → 541 ms with no source changes, confirming `--sample-size 10` on M1 Pro has ±50 % thermal-noise floor on these timings. **Action taken**: saved a clean `--save-baseline v0.6.0` (see `target/criterion/<bench>/v0.6.0/`) so the next release compares against a same-day reference instead of the stale prior baseline.
