# Retrieval Pipeline

This page documents the end-to-end recall pipeline used by `db.recall()`,
`axil recall`, `axil code-search`, `axil recall-for-file`, and
`axil recall-for-entity`. It covers cascaded source execution, Reciprocal
Rank Fusion (RRF), multi-signal scoring, Query-Time Chunking (QTC),
optional cross-encoder rerank, and the code-aware layer added in Phase 13b.

The authoritative implementation lives in
[`crates/axil-core/src/query.rs`](../../crates/axil-core/src/query.rs) and
[`crates/axil-core/src/scoring.rs`](../../crates/axil-core/src/scoring.rs).

## Why hybrid, not vector-only

Axil's production recall is measurably better than naive vector search:

| Path | recall@1 (LongMemEval-S)* |
|------|---------------------------|
| `query().similar_to(...)` (raw vector only) | 0.100 |
| `db.recall(...)` (full hybrid pipeline) | 0.760 |

\* *Historical numbers from the Phase 15 verification run. The
LongMemEval harness is not in-tree (see `.gitignore: /benchmarks/`),
so CI cannot regenerate these. To re-verify, pull the harness from
the project's benchmark archive and re-run. See the
[Evaluation Log](./eval-log.md) for methodology and version history.*

The 7.6× gap is the difference made by FTS + RRF + multi-signal
scoring + QTC. If you call the raw `QueryBuilder::similar_to` path you are
leaving recall on the table.

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1 — Cascaded retrieval (cheapest first)                │
│                                                             │
│   1a. Timeseries filter    (range/changed_since on _idx_ts) │
│   1b. FTS search           (Tantivy BM25, code tokenizer)   │
│   1c. Vector search        (HNSW ANN; SKIPPED if FTS top    │
│                             hit > 0.95 confidence)          │
├─────────────────────────────────────────────────────────────┤
│ Step 2 — RRF fusion        (k=60, adaptive)                 │
├─────────────────────────────────────────────────────────────┤
│ Step 3 — Record resolution + post-filter                    │
│          (table filter, where-clauses, time bounds)         │
├─────────────────────────────────────────────────────────────┤
│ Step 4 — Multi-signal rescoring                             │
│          (vector + recency + graph + keyword + temporal +   │
│           feedback + preference + activation + importance)  │
├─────────────────────────────────────────────────────────────┤
│ Step 4b — Cross-encoder rerank          (optional, off)     │
├─────────────────────────────────────────────────────────────┤
│ Step 4c — Query-Time Chunk rerank       (optional)          │
├─────────────────────────────────────────────────────────────┤
│ Step 5 — Graph traversal                (optional)          │
├─────────────────────────────────────────────────────────────┤
│ Step 6 — Sort + offset + limit                              │
└─────────────────────────────────────────────────────────────┘
```

## Step 1 — Cascaded retrieval

Sources run cheapest-first so later stages see fewer candidates. Each
source produces a ranked list of `(RecordId, score)` for the fusion step.

**Timeseries** uses the `_idx_ts` index to bound the candidate set by
`created_at`. When the query carries `after`/`before` or
`changed_since`, this runs first and drastically narrows what FTS and
vector have to consider.

**FTS** runs Tantivy BM25 against `_idx_fts`. The code tokenizer
preserves identifiers (`auth_timeout`, `Result::Err`) and applies field
boosting. Phase 4 added fuzzy matching, snippets, and field scoping.

**Vector** runs HNSW ANN against the embedding index (`*.axil.vec`).
The query is either a pre-embedded `&[f32]` or text that the configured
embedder converts. Default embedder is BGE-small-en-v1.5 (int8); BGE-base,
nomic, and gte-modernbert are also registered.

**Cascade skip rule**: vector search is skipped when FTS already returned
a high-confidence top hit (score > 0.95). This is the most common reason
a query is "fast" — symbol-exact lookups never pay the HNSW cost.

Per-source `fetch_k` is `(limit + offset) * 4`, with a minimum floor, so
fusion always has enough candidates to reorder.

## Step 2 — Reciprocal Rank Fusion

All ranked lists are merged with classic RRF:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
              i
```

Axil uses `k = 60` (Cormack/Clarke/Buettcher 2009 baseline). RRF is
rank-based, so it is robust to score-scale drift across the three
retrievers (BM25 scores, cosine similarity, and time-rank live on
different scales).

The `adaptive` part: when only one retriever is enabled (vector-only or
FTS-only), RRF degenerates to that retriever's order — no spurious
re-ranking is introduced.

## Step 3 — Record resolution + post-filter

Fused `RecordId`s are resolved through redb. While resolving, Axil
applies:

- `table` filter (only records from a specified table)
- `where` clauses (field comparisons from the QueryBuilder)
- Time bounds (`after`/`before`) for records not perfectly pre-filtered
  at the timeseries stage

Resolution stops at `seed_cap`. Without rerank/traversal,
`seed_cap = limit + offset`. When a reranker is attached, the cap widens
to `rerank_top_k_in + offset` so the reranker actually sees the full
window (otherwise it would only see the prefix that already won by fused
score — a no-op).

## Step 4 — Multi-signal rescoring

This is the layer most other embedded RAG stores skip. After fusion, each
candidate is scored with up to 9 signals and combined:

```
final_score = w_vector     * vector_similarity
            + w_recency    * recency_decay(age)
            + w_graph      * graph_proximity
            + w_keyword    * keyword_overlap
            + w_temporal   * temporal_proximity
            + w_feedback   * feedback_boost
            + w_preference * preference_match
            + w_activation * activation
            + w_importance * importance
```

### Default weights

From `ScoreWeights::default()`:

| Signal       | Weight | Notes |
|--------------|--------|-------|
| `vector`     | 0.40   | Cosine similarity from HNSW |
| `recency`    | 0.15   | Exponential decay, half-life 168h (1 week) |
| `graph`      | 0.15   | PageRank-style hop boost from active context |
| `temporal`   | 0.10   | Boost when record falls near a parsed time target ("yesterday", "last week") |
| `keyword`    | 0.10   | Multiplicative overlap of query keywords vs. record text (MemPalace technique, +1.2%) |
| `feedback`   | 0.05   | Learned boost from past relevance signals |
| `preference` | 0.05   | Matches against `axil prefer` keys |
| `activation` | 0.0    | Access-frequency × decay (Phase 8b.4, off by default) |
| `importance` | 0.0    | Auto-importance score (Phase 10.1, off by default) |
| `rrf`        | 0.0    | Raw RRF score, plumbed for tuning — see note below |

The `rrf` weight is intentionally `0.0`: RRF scores live on a `1/(60+rank)`
scale, which is too small to contribute meaningfully alongside signals
on `[0, 1]` unless rescaled. RRF already governs the candidate order in
Step 2; adaptive weight renormalization handles the final ranking.

### Why these signals

The signals are computed without an LLM — every weight comes from
algorithmic patterns. `recency_decay` is a true half-life exponential
(0.5 at the configured half-life). `keyword_overlap` uses overlapping
1600-char windows with 400-char stride so long records aren't penalized
for keyword sparsity. `graph_proximity` boosts records reachable in few
hops from the active context (Phase 5e).

### Inspecting scores

Every result carries a `ScoreExplanation` listing each signal's raw
value and a human-readable summary. Pass `--explain` (CLI) or call
`.explain()` (API):

```bash
axil recall "auth timeout" --explain
```

## Step 4b — Cross-encoder rerank (optional, off by default)

Phase 15 added a pluggable `Rerank` trait with two ONNX implementations
(MS-MARCO MiniLM, AnswerAI ColBERT-small) in
[`crates/extensions/axil-rerank`](../../crates/extensions/axil-rerank). When attached via
`with_reranker(...)`, the top `rerank_top_k_in` candidates are re-scored
by a cross-encoder and blended with `weight = 0.7`.

**Status: disabled by default.** Phase 15 verification on LongMemEval-S
showed both models *hurt* quality: MS-MARCO MiniLM dropped recall@1
from 0.760 to 0.580 (-24%), and ColBERT-small showed similar regression.
The infrastructure ships so users can attach domain-specific rerankers,
but no model is enabled out of the box. See
[Eval Log](./eval-log.md) for the full Phase 15 numbers.

## Step 4c — Query-Time Chunking (QTC)

When `RecallConfig.qtc = Some(QtcConfig { .. })`, the top-K session-level
candidates are re-embedded at query time in overlapping windows
(default 1200 chars, 900-char stride = 25% overlap). The best chunk's
cosine similarity is blended with the fused score:

```
new_score = alpha * best_chunk_cosine + (1 - alpha) * fused_score
```

Defaults are tuned on LongMemEval-S (top_k=20, alpha=0.7) and hit a
97.3% hit rate / 94.0% recall at the oracle ceiling on long documents.

QTC fixes the long-document problem: when the answer sits beyond the
indexed embedding's effective window, index-time chunking suffers from
shared-timestamp collisions (every chunk has the same `created_at`).
QTC sidesteps this by chunking only the small top-K post-fusion.

QTC is opt-in because it costs additional embedding calls. Enable it
for document-heavy stores; leave it off for short-record workloads.

## Step 5 — Graph traversal (optional)

When the query carries a `traverse(...)` step, results are expanded
through the graph index (`*.axil.graph`) via fan-out traversal. Edge
types include the SCIP set (`defined_in`, `references`, `calls`,
`implements`, `type_of`, `imports`) plus memory-domain edges (`mentions`,
`supersedes`, `documents`, `session_checkpoint_for`).

For code recall specifically, `--trace-graph` on `code-search`,
`recall-for-file`, and `recall-for-entity` walks SCIP edges from the
proxy hits via the `canonical_id` bridge — surfacing callers/callees of
the symbol the agent just retrieved.

## Step 6 — Sort + offset + limit

Final pass applies any explicit `sort_by`, then `skip(offset)` and
`take(limit)`. The default sort is by final fused score, descending.

## Code-aware layer (Phase 13b)

For code queries, an extra layer sits *on top of* the pipeline above:

**Structure-aware proxies.** `axil-indexer` chunks source into proxy
records stored in `_idx_code_proxies` (file / symbol / section, each
with breadcrumb, signature, line range, and optional SCIP
`canonical_id`). Markdown is split by headings; TOML/JSON/YAML by
section.

**Reverse code-ref index.** Memories stored with
`--code-ref <proxy_id | canonical_id | path:line>` are indexed in
`_idx_code_refs`. When proxies hit, attached memories are surfaced in
the same response.

**SCIP bridge.** Provisional `provisional:<sha>` entities from regex
extraction are rewritten to SCIP `canonical_id` on unambiguous match;
ambiguous cases stay provisional (no silent merge). This means
"calls of `login`" works whether the symbol was first seen by regex or
by SCIP.

**Same-file / tests edges.** Graph composition adds `same_file` and
`tests` edges between proxies so traversal expands to siblings and test
coverage.

End-to-end, `axil code-search "auth timeout"` runs the standard hybrid
pipeline against `_idx_code_proxies`, then optionally walks SCIP edges
to widen the answer. P0 quality on the Axil dogfood: top-3 file/symbol
hit-rate 0% → 20%, ~45% context-token reduction, p95 19ms → 15ms.

## Tuning knobs

| Knob | Where | Default | When to change |
|------|-------|---------|----------------|
| `RecallConfig.weights` | `RecallConfig::default()` | see table above | Domain has unusual signal mix (e.g. logs need higher recency, code needs higher keyword) |
| `recency_half_life_hours` | `RecallConfig` | 168 (1 week) | Longer for archival corpora, shorter for incident channels |
| `qtc` | `RecallConfig` | `None` | Long documents where answers sit beyond the embedding window |
| Embedding model | `axil.toml` `[index] embedding_model` | `bge-small-en-v1.5-int8` | `bge-base` for quality, `int8` for speed |
| Cross-encoder rerank | `with_reranker(...)` | unattached | Only attach a domain-tuned reranker — the stock models hurt on LongMemEval |
| Fetch multiplier | `min_fetch = (limit + offset) * 4` | × 4 | Increase if top-K is very small and you want richer fusion input |

## See also

- [Memory Types](../concepts/memory-types.md) — recency-weighted recall per type
- [Engines (Storage Plugins)](../concepts/plugins.md) — vector / graph / FTS engines that feed Step 1
- [Performance](./performance.md) — Phase 8b cascaded filtering, adaptive RRF, deferred indexing, mmap vectors
- [Cognitive Memory](./cognitive.md) — importance, decay, tiered memory (feeds the activation/importance signals)
- [Evaluation Log](./eval-log.md) — Phase 15 measurements behind the recall@1 numbers and the reranker decision
