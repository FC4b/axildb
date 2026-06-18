# LongMemEval-S Rerank Investigation (April 2026)

30-question subset, top-K=5, --jobs 3-4 (CPU-shared between runs).

## Scoreboard

| Strategy | Rerank | Hit | Recall | Δ vs prod |
|---|---|---|---|---|
| recall-qtc | off | 0.933 | 0.933 | (production) |
| recall | off | 0.900 | 0.900 | -3.3 |
| recall | bge-reranker | 0.733 | 0.733 | -20.0 |
| recall | cross-encoder (ms-marco) | 0.667 | 0.667 | -26.7 |
| recall-qtc | cross-encoder (ms-marco) | 0.667 | 0.667 | -26.7 |
| ask (intent-routed) | off | 0.267 | 0.267 | -66.7 |

Raw JSON in this directory: `lme-baseline-30.json`, `lme-recall-no-rerank-30.json`, `lme-rerank-30.json`, `lme-bge-rerank-30.json`, `lme-qtc-rerank-30.json`, `lme-ask-30.json`.

## Findings

1. **Cross-encoder rerank regresses by ~23 points** regardless of base strategy. The identical 66.7% on `recall` and `recall-qtc` (different base orderings, same final hit rate) means rerank is producing a deterministic ordering of top-20 candidates that ignores the upstream signal.

2. **BGE-reranker is +6.6 better than cross-encoder but still -16.7 vs baseline.** Both rerankers hurt; only the magnitude differs.

3. **`ask` strategy at 26.7%** — the intent-routing path in `axil_indexer::ask::ask` is misrouting LongMemEval questions. Root cause not investigated.

## Suspected root cause for rerank regression

`crates/axil-indexer/src/rerank.rs:228` extracts the cross-encoder output as `data.first()` — index 0 of a flat tensor. If the model has 2-label output `[neg_logit, pos_logit]`, we sort on the *negative* class score. Not verified — would need to log the output shape at runtime.

## Why Hindsight's 91.4% LongMemEval with rerank doesn't transfer

- Hindsight uses a reranker fine-tuned on memory-style passages, not ms-marco/BGE general-purpose.
- Hindsight's candidates come from PostgreSQL + LLM-extracted entity boost, not vector + FTS.
- Hindsight uses a different LongMemEval subset.

## Decision

- **Do NOT wire either reranker into `db.recall()`** — would actively harm MCP production retrieval.
- **Do NOT promote `ask` strategy** — broken on this benchmark.
- Keep existing CLI flags (`axil recall --rerank cross-encoder`) for opt-in experimentation; documented as not-recommended.
- Current `recall-qtc` is already optimal among tested configurations.

## Open follow-up (not for this session)

1. Verify cross-encoder model output shape; if 2-label, take `data[1]`.
2. Diagnose `ask` strategy regression (intent classifier or fusion path).
3. Test other lift sources from research: better embedding model (bge-m3 or e5-mistral), algorithmic query expansion, LLM-optional memory consolidation.
4. If pursuing rerank long-term: fine-tune a reranker on memory-style training data.
