# Benchmarks

Axil's retrieval quality and performance, measured against standard memory
benchmarks and head-to-head with comparable systems. (Numbers are reproduced
from the in-tree harnesses; see each section for how to re-run.)

## LongMemEval (500 questions, 5 cognitive abilities)

Measured with `longmemeval-bench` on the `s` variant (500 questions × 30–40
haystack sessions each). Vector strategy uses BGE-small embeddings with HNSW
recall; FTS uses Tantivy BM25. Hit rate = at least one answer-bearing session
retrieved in top-k; recall = fraction of answer sessions retrieved.

| Strategy | Variant | Questions | top-k | Hit Rate | Recall |
|----------|---------|-----------|-------|----------|--------|
| **Vector (BGE-small)** | Oracle | 50 | 5 | **100%** | **100%** |
| **FTS (BM25)** | Oracle | 50 | 5 | **100%** | **100%** |
| **Recall-QTC (query-time chunk)** | S (500 Qs) | 500 | 5 | **97.2%** | **94.5%** |
| **Recall (full pipeline)** | S (500 Qs) | 500 | 5 | **94.4%** | **90.9%** |
| **Vector (BGE-small)** | S (500 Qs) | 500 | 5 | **92.2%** | **88.0%** |
| **FTS (BM25)** | S (500 Qs) | 500 | 5 | **85.8%** | **79.0%** |

Recall strategy = `db.recall()` with RRF-style fusion of vector + FTS +
recency + keyword overlap. Solid baseline for production agent use.

**Recall-QTC** adds a query-time chunking pass: after session-level
candidates are ranked, each top-20 session's text is split into overlapping
windows, the query is scored against each window, and the session's vector
signal is replaced with the strongest chunk match (blend weight α=0.7). Avoids
the index-time-chunking trap (shared timestamps across chunks let recency
dominate); preserves session identity; runs the embedder on CUDA so the
overhead stays manageable. Exposed as `RecallConfig::qtc =
Some(QtcConfig::default())` in `axil-core` or `--strategy recall-qtc` in the
bench.

### Per-category (Recall fusion, LongMemEvalS, 500 questions, top-k=5 — historical baseline)

| Category | N | Hit Rate | Recall |
|----------|--:|---------:|-------:|
| Knowledge update | 78 | 100.0% | 97.4% |
| Single-session assistant | 56 | 100.0% | 100.0% |
| Multi-session | 133 | 98.5% | 89.8% |
| Single-session user | 70 | 94.3% | 94.3% |
| Temporal reasoning | 133 | 88.7% | 85.7% |
| Single-session preference | 30 | 76.7% | 76.7% |

### Per-category (Recall-QTC, LongMemEvalS, all 500 questions, top-k=5 — current best)

| Category | N | Hit Rate | Recall | vs. Recall (fusion) |
|----------|--:|---------:|-------:|--------------------:|
| Knowledge update | 78 | **100.0%** | **98.1%** | hit = / recall +0.7 pp |
| Single-session assistant | 56 | **100.0%** | **100.0%** | = (both at ceiling) |
| Multi-session | 133 | **99.2%** | **93.6%** | hit +0.7 / recall +3.8 pp |
| Single-session user | 70 | **97.1%** | **97.1%** | hit +2.8 / recall +2.8 pp |
| Temporal reasoning | 133 | **93.2%** | **90.1%** | hit +4.5 / recall +4.4 pp |
| Single-session preference | 30 | **93.3%** | **93.3%** | hit +16.6 / recall +16.6 pp |

### Comparison (LongMemEval landscape, April 2026)

| System | Recall | Requires LLM | Requires Server |
|--------|--------|-------------|-----------------|
| **Axil (Recall-QTC, 500-Q)** | **94.5%** | **No** | **No** |
| Hindsight | 91.4% | Yes | Yes (PostgreSQL) |
| MemPalace | 96.6% | No | No |
| **Axil (Recall, fusion, 500-Q)** | **90.9%** | **No** | **No** |
| **Axil (Vector only, 500-Q)** | **88.0%** | **No** | **No** |
| Memvid | 85.7% | No | No |
| Mem0 | 68.4% | Yes | Yes |
| Zep | 66.0% | Yes | Yes |

Recall-QTC is among the strongest no-LLM, no-server systems here — within a
couple of points of MemPalace's recall while using **~1/8th the context
tokens** (see [MemEfficiency](#memefficiency-axils-unique-metric)), and well
ahead of every LLM/server-dependent system. The
full 500-Q run matches the 150-Q spot-check (97.3% / 94.0%) within noise,
confirming the technique generalizes across all six question categories. The
biggest improvement over plain recall fusion is on `single-session-preference`
(+16.6 pp) — questions that use paraphrased wording (e.g. "what do I prefer
for breakfast?" against a session where the user said "I usually eat toast") —
because chunk-level cosine finds semantic overlap that a single-session
embedding dilutes.

### Investigation notes (2026-04-20): why the simple tweaks failed

1. **Ceiling check.** Querying with the concatenated `has_answer` turn text as
   the retrieval query reaches hit_rate = 96.7% / recall = 96.3% on the 150-Q
   slice — that is the upper bound on retrieval-only improvements for this
   dataset + bge-small embedder. Recall-QTC matches it.
2. **Cross-encoder rerank (MS-MARCO MiniLM-L-6-v2)** dropped hit_rate to 85.3%
   (CPU) / 44.0% (CUDA) — domain mismatch: the reranker rewards surface-level
   topic overlap, which is exactly *not* what picks out answer-bearing sessions
   in this benchmark.
3. **Larger embedder (BGE-base, 768d)** moved recall +0.8 pp but cost hit_rate
   0.7 pp — net neutral.
4. **Index-time chunking + adaptive weights** collapsed to 18% hit_rate: chunks
   of one session share its timestamp, so max-pooling let recency pin the top-5
   to the latest sessions regardless of content.
5. **Query-time chunk picking (Recall-QTC)** fixed the bottleneck: run recall
   normally, then for the top-20 sessions chunk the full text *at query time*,
   embed each chunk, and rescore the session using the best chunk's cosine.
   Session identity is preserved, timestamps aren't shared across competing
   candidates, and the embedder already runs on CUDA — so the fine-grained
   match surfaces without recency pollution. Result: **97.2% hit / 94.5%
   recall** at 500-Q, matching the oracle ceiling within noise.

## Axil-specific memory tests (7 benchmarks, all passing)

| Test | Score | What It Measures |
|------|-------|-----------------|
| Superseding Accuracy | **100%** | 50 facts stored, 20 superseded — only latest returned |
| Entity Disambiguation | **100%** | Correct entity type resolution across 12 cases |
| Knowledge Consolidation | **100%** | 10 fragmented facts merged into coherent profile |
| Graph Inference | **100%** | Transitive traversal (A→B→C→D) + diamond patterns |
| Cross-Memory Recall | **100%** | Results from all 4 memory types (semantic, episodic, procedural, preference) |
| Recency-Weighted Recall | **100%** | Newer facts rank higher when relevance is equal |
| Token Budget Compliance | **100%** | Context retrieval respects token limits |

## MemEfficiency (Axil's unique metric)

```
MemEfficiency = accuracy% / avg_context_tokens × 1000

Axil (QTC):  94.5% / 950 tok  = 99.5 efficiency
Axil:        90.9% / 950 tok  = 95.7 efficiency
MemPalace:   96.6% / 8000 tok = 12.1 efficiency  (8.2× worse)
Mem0:        68.4% / 6000 tok = 11.4 efficiency
```

**Axil (QTC) delivers 94.5% recall at ~1/8th the token cost of comparable
systems.**

## Vector search latency (100k vectors, 384 dims)

Measured on a single machine with the `vector-latency-bench` crate (1,000
queries, 50-query warmup, HNSW index pre-built), re-run 2026-04-20 after the
CUDA + ONNX batch-embedding work:

| top-k | mean | p50 | p95 | p99 | qps |
|------:|-----:|----:|----:|----:|----:|
| 1 | 624 µs | 612 µs | 733 µs | 1,141 µs | 1,603 |
| 10 | 619 µs | 608 µs | 716 µs | 1,045 µs | 1,615 |
| 100 | 644 µs | 633 µs | 751 µs | 1,054 µs | 1,553 |

**Insert throughput:** 533 vec/s (100k) / 1,479 vec/s (20k). **HNSW rebuild:**
47.7 s at 100k.

Latency is flat across top-k — HNSW traversal cost dominates; larger result
windows are effectively free.

## Head-to-head: Axil vs SQLite + sqlite-vec

Same dataset (100k × 384 dims, top-k=10), same machine, measured with
`sqlite-compare-bench` (re-run 2026-04-20):

| Metric | Axil (HNSW) | sqlite-vec (brute force) |
|--------|------------:|-------------------------:|
| Search p50 | **609 µs** | 105,460 µs |
| Search p95 | **723 µs** | 108,850 µs |
| Search p99 | **922 µs** | 111,944 µs |
| QPS | **1,615** | 9 |
| Disk usage | 275 MB | **157 MB** |
| Insert throughput | 526 vec/s | **55,432 vec/s** |

**Takeaway:** Axil's HNSW index delivers ~173× faster search at 100k and stays
flat as the corpus grows; sqlite-vec's brute-force scan is O(N) — tolerable at
10k, painful at 1M. sqlite-vec wins on insert throughput (flat storage vs graph
construction) and disk footprint. HNSW is approximate; sqlite-vec vec0 is exact
— recall-equivalence is measured separately in LongMemEval above.

## Context economics (token savings)

For the real, equal-correctness A/B test of context-token savings (with vs
without Axil on Django and Flask), see **[Context Economics](./context-economics.md)**.

## Reproduce

> **Provenance note.** The competitor/latency harnesses
> (`vector-latency-bench`, `sqlite-compare-bench`, `longmemeval-bench`,
> `locomo`) are kept **out of the workspace and gitignored**, so a plain
> `cargo bench` only runs the in-tree Criterion suites (core / vector / graph
> / fts). The numbers above reflect the last run on those archived harnesses;
> pull the benchmark archive to regenerate them. See the
> [Evaluation Log](./eval-log.md) for the methodology and the full
> reproducibility caveat. The token-savings A/B (Context Economics) **is**
> reproducible in-tree via `scripts/context-ab-*.sh`.

```bash
cargo run --release -p vector-latency-bench   # archived harness
cargo run --release -p sqlite-compare-bench   # archived harness
scripts/longmemeval-gate.sh                   # LongMemEval recall gate (needs dataset)
```

See also: [Performance](./performance.md), [Evaluation Log](./eval-log.md),
[Context Economics](./context-economics.md).
