# Retrieval Pipeline

This page documents the end-to-end recall pipeline used by `db.recall()`,
`axil recall`, `axil code-search`, `axil recall-for-file`, and
`axil recall-for-entity`. It covers cascaded source execution, Reciprocal
Rank Fusion (RRF), multi-signal scoring, Query-Time Chunking (QTC),
optional cross-encoder rerank, and the code-aware layer added in Phase 13b.

The authoritative implementation lives in
[`crates/axil-core/src/query.rs`](../../../crates/axil-core/src/query.rs) and
[`crates/axil-core/src/scoring.rs`](../../../crates/axil-core/src/scoring.rs).

## Why hybrid, not vector-only

No committed run compares raw vector search with the full pipeline on the
same build. The committed 500-question LongMemEval-S runs come from two
builds, so read them as history, not as a head-to-head:

| Path (500 questions, top-5) | avg recall | Build | Source |
|------|------|------|--------|
| `vector` (raw vector only) | 0.936 | initial release | [`baseline-500.json`](../../../benchmarks/results/baseline-500.json) |
| `recall-qtc` (full hybrid pipeline) | 0.935 | 2026-06-27 rerun | [`qtc-500.json`](../../../benchmarks/results/qtc-500.json) |
| `recall` (fusion without query-time chunks) | 0.915 | 2026-06-27 rerun | [`fusion-500.json`](../../../benchmarks/results/fusion-500.json) |

The harness is in-tree at [`benchmarks/longmemeval/`](../../../benchmarks/longmemeval/)
and needs the dataset downloaded separately; see its README.

What the pipeline adds over the raw `QueryBuilder::similar_to` path are
signals a vector alone cannot carry: exact matches on identifiers and error
strings (FTS), how recent a record is, records connected to what the query
names (graph), and what the agent used before (feedback). Their effect on
recall is measured per change with the dogfood eval and LongMemEval, not
claimed here.

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1 — Cascaded retrieval (cheapest first)                │
│                                                             │
│   1a. Timeseries filter    (range/changed_since on _idx_ts) │
│   1b. FTS search           (Tantivy BM25, code tokenizer)   │
│   1c. Vector search        (exact scan or HNSW; SKIPPED if  │
│                             FTS top hit > 0.95 confidence)  │
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
│ Step 4b — Library reranker   (QueryBuilder only, optional)  │
│           (the CLI's --rerank runs after recall returns)    │
├─────────────────────────────────────────────────────────────┤
│ Step 4c — Query-Time Chunk rerank   (Axil::recall only)     │
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

**Vector** searches the embedding index (`*.axil.vec`): an exact scan up to
20k vectors (100k in a one-shot CLI call, which would otherwise rebuild the
graph on every search), and the HNSW graph above that.
The query is either a pre-embedded `&[f32]` or text that the configured
embedder converts. Default embedder is BGE-small-en-v1.5 (int8); BGE-base,
nomic, and gte-modernbert are also registered. Query text is embedded on the
embedder's query side (`TextEmbedder::embed_query`) and stored text on its
passage side, so a model trained with separate prompts for each can use them.

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
| `graph`      | 0.15   | Links to the top vector hits, for a candidate linked to an entity the query names; normalized by degree |
| `temporal`   | 0.10   | Boost when a record falls near a parsed time target ("yesterday", "3 days ago") or inside a named span ("in March", "last 3 days") |
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
for keyword sparsity. `graph_proximity` counts only when the query names an
entity (a backticked or quoted name, a path, a CamelCase or snake_case
identifier) that the candidate is linked to. It then scores the candidate's links to the ten
candidates closest to the query by vector, `links / sqrt(degree × 10)`, so a
hub linked to everything does not outscore a record whose few links all land
on those seeds.

A query that names a span of time (`in March`, `March 2026`, `between
2026-03-01 and 2026-03-15`, `before 2026-04-01`, `since March`, `last 3 days`)
also keeps only the candidates inside it, by when what they describe happened:
a record's `event_time` field (a date or RFC 3339) when it has one, else when it
was stored. When none are, the span is set aside and relevance alone decides. A point in time (`yesterday`,
`3 days ago`, `March 15`) only boosts. `before`/`after` followed by anything but
a date ("after the release") name no span.

### Inspecting scores

Every result carries a `ScoreExplanation` listing each signal's raw
value and a human-readable summary. Pass `--explain` (CLI) or call
`.explain()` (API):

```bash
axil recall "auth timeout" --explain
```

## Step 4b — Cross-encoder rerank (optional, off by default)

Two rerankers share this name and run at different points.

**The CLI's `--rerank`** is not a stage inside `db.recall()`: it runs after
recall has returned, so after query-time chunk rescoring (Step 4c) and the
post-filter. `axil recall --rerank cross-encoder` re-scores a pool of
recall's hits with an ONNX cross-encoder, reorders the pool by that score
(recall's order is replaced unless `--rerank-weight` blends the two), and
prints the first `--top-k` of the new order.

- **What the rerank receives.** A pool of max(4 × `--top-k`, 20) of recall's
  final, QTC-scored hits: 20 at the default `--top-k 5`, the depth the
  LongMemEval harness reranks. Hits ranked below the `--top-k` cut are in the
  pool, so the rerank can promote them. Recall is asked for the pool size, and
  completeness widening (on by default; `--no-widen` turns it off) can grow
  the pool to ⌈1.5 × the pool size⌉; every hit in the pool is scored. With a
  post-filter (`--table`, `--type`, `--after`, `--before`, `--agent`,
  `--min-importance`, `--fresh-only`), recall over-fetches 5 × `--top-k`
  candidates, or the pool size when that is larger, filters them, and keeps
  the first max(4 × `--top-k`, 20) that pass as the pool. When the primary
  recall returns nothing, the CLI cascades through fallbacks in order and the
  pool comes from the first that returns hits: recall again with the filters
  relaxed (and query-time chunks off), recall on the expanded query, then
  direct FTS; `--no-cascade` turns all three off. `--profile` prints the pool
  size to stderr.
- **Blending.** `--rerank-weight W` (cross-encoder only, default 1.0) ranks
  each pooled hit by W × cross-encoder score + (1 − W) × recall score, both
  min-max normalized over the pool. At 1.0 the cross-encoder's order stands
  alone; at 0.0 recall's order comes back.
- **What is printed.** The first `--top-k` of the reranked pool, minus hits
  the `--recall-format` has nothing to show for (`compact`, `oneline` and
  `context-block` skip them; `context-block` also lists code hits before
  memories). The cut to `--top-k` also applies when the rerank is skipped
  (the `--timeout-ms` deadline passed) or fails (no model): the output is
  then the first `--top-k` of the pool in recall's order. Widening never adds
  hits past `--top-k` under `--rerank`. With `--budget`, only the prefix
  that fits is printed, except that most formats print the first hit even
  when it alone is over the budget (`context-block` may print just a
  code-section heading then).
- **Without `--rerank`** (or with `--rerank off`) none of this runs: recall
  keeps `--top-k`, widened as usual, and the output is the same as before the
  pool existed.

The implementation is
[`crates/extensions/axil-indexer/src/rerank.rs`](../../../crates/extensions/axil-indexer/src/rerank.rs),
compiled in by the `rerank` Cargo feature (part of the CLI's `full`
build, not `default`). It loads `models/cross-encoder.onnx`, with
`tokenizer.json` beside it, relative to the working directory; the expected
model is `cross-encoder/ms-marco-MiniLM-L-6-v2` (`onnx/model.onnx` and
`tokenizer.json` from its Hugging Face repository; Axil does not download
it). `--rerank llm` reorders the same pool through the configured
`LlmProvider` instead; the LLM sees the first 20 hits of the pool.

**The library reranker** belongs to the `QueryBuilder` path only. Embedded
library users can bring their own: implement the `Rerank` trait in
`axil-core` and attach it with `QueryBuilder::with_reranker(...)`. It does
not run inside `Axil::recall` (`db.recall()`), and `QueryBuilder` has no
query-time chunk stage: the diagram shows both paths' stages, and Steps 4b
and 4c each apply to one of them. The top `rerank_top_k_in` candidates
(default 50) are re-scored and up to `rerank_top_k_out` (default 10) are
kept.

**Status: off by default.** Measured on 2026-10-14 with
`cross-encoder/ms-marco-MiniLM-L-6-v2`, each comparison on one corpus the
same day:

- **CLI path, dogfood eval** (57 questions about this repo's memory,
  `--fetch 10`;
  [`dogfood-recall-2026-10-14-rerank.json`](../../../benchmarks/results/dogfood-recall-2026-10-14-rerank.json)):
  NDCG@10 is 0.812 without a rerank, 0.833 with the old CLI window (only the
  printed `--top-k` reordered), 0.838 with the pool, and 0.859 with the pool
  blended at `--rerank-weight 0.3`, the best weight of the sweep. At 0.3,
  MRR@10 goes from 0.791 to 0.831 and hit@5 from 0.895 to 0.947, and the hook
  block holds an answer for 0.930 of the questions instead of 0.895.
- **LongMemEval-S, a 60-question stratified sample**, Recall-QTC, top-k 5,
  every variant ordering the same candidates
  ([`longmemeval-s60-rerank-2026-10-14.json`](../../../benchmarks/results/longmemeval-s60-rerank-2026-10-14.json)):
  recall is 0.979 without a rerank, 0.983 with the cross-encoder's order
  alone and 0.979 at weight 0.3; NDCG@10 is 0.946, 0.933 and 0.961.

So the blend at 0.3 ranks better on both and leaves LongMemEval recall where
it was, while the cross-encoder's order alone costs LongMemEval ranking.
The weight was picked on the dogfood eval; the LongMemEval run reports the
same weight. Rerank still stays opt-in: it needs the `rerank` build and a
model Axil does not download, and it adds a cross-encoder pass over the
pool to every recall.

Earlier runs on a 30-question subset reported the cross-encoder lowering
Recall-QTC from 0.933 to 0.667
([investigation](../../../benchmarks/results/lme-rerank-investigation.md)).
Until this re-measurement the harness ranked sessions by score after a
rerank, with cross-encoder logits on the reranked window and fused scores on
the candidates below it, so a low candidate could outrank a reranked one
with a negative logit. It now ranks by reranked position; the older numbers
may not show the reranker's own effect.

## Step 4c — Query-Time Chunking (QTC)

When `RecallConfig.qtc = Some(QtcConfig { .. })`, the `top_k` (default 20)
most relevant candidates in `Axil::recall`'s pool get their best chunk's
cosine similarity against the query blended with their fused score:

```
new_score = alpha * best_chunk_cosine + (1 - alpha) * fused_score
```

"Most relevant" means ranked by fused score with the recency term taken
out. Recency still counts in every final score, but it does not decide who
gets the blend: when the window followed the plain fused order, a fresh
record could crowd out an old one that matched the query best, and the old
one was never rescored. The window itself stays: blending every candidate
lets records whose only strength is a high cosine (file summaries, code
proxies) outrank the answers, which cost the dogfood eval's hit@5 about
three points.

A record longer than 1600 bytes gets a vector per recall unit at insert:
one per line of its text (a dialog turn or a fact), a short line joined
with the lines after it until the unit holds 400 bytes, a unit longer than
1600 bytes cut into overlapping 1600-byte pieces. QTC scores such a record from those stored
vectors; a shorter one is cut into overlapping windows (default 1200 chars,
900-char stride = 25% overlap) and embedded at query time. Either way each
piece is embedded under a one-line header with the record's table and date
and the files and entities it lists (`decisions, 2026-03-12; files:
src/db.rs`), so a piece cut out of a record still says what the record is
about; the stored and FTS-indexed text is the piece alone.

The piece that scored best comes back with the hit
(`ScoreExplanation::best_chunk`, `snippet` in the CLI's JSON), and compact
recall output shows it, cut to 200 characters, in place of the record's
opening. A database written before the splitter changed keeps its old
pieces until `axil heal --rechunk` re-splits them.

`Axil::recall` also draws a fixed candidate pool (80 per source, doubled for
the fetch) instead of one scaled by `top_k`, so a smaller `top_k` returns a
prefix of a larger one.

`top_k=20` and `alpha=0.7` were tuned on LongMemEval-S.

QTC fixes the long-document problem: when the answer sits beyond the
indexed embedding's effective window, ranking index-time chunks as results
suffers from shared-timestamp collisions (every chunk has the same
`created_at`). QTC sidesteps this by keeping records as the ranked unit and
using chunk vectors only to score them.

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
| Cross-encoder rerank | `with_reranker(...)` | unattached | Not measured on this path; for the CLI's cross-encoder results see Step 4b |
| Fetch multiplier | `min_fetch = (limit + offset) * 4` | × 4 | Increase if top-K is very small and you want richer fusion input |

## See also

- [Memory Types](../concepts/memory-types.md) — recency-weighted recall per type
- [Engines (Storage Plugins)](../concepts/plugins.md) — vector / graph / FTS engines that feed Step 1
- [Performance](./performance.md) — Phase 8b cascaded filtering, adaptive RRF, deferred indexing
- [Cognitive Memory](./cognitive.md) — importance, decay, tiered memory (feeds the activation/importance signals)
- [Evaluation Log](./eval-log.md) — Phase 15 measurements behind the recall@1 numbers and the reranker decision
