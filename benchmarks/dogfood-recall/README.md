# Dogfood recall eval

57 questions about this repo's own Axil memory (`.axil/memory.axil`), each
naming the records that answer it. It measures what LongMemEval can't:

- **The CLI path.** It runs `axil recall`, the command agents and hooks use.
  The LongMemEval harness calls the library directly, so a bug in the CLI's
  recall config (such as weights that don't sum to 1) is invisible there.
- **Coding-agent memory.** Decisions, errors, gotchas, rules and commits, not
  chat sessions.
- **What the hook injects.** The same question is also run exactly as the
  prompt hook runs it (`--recall-format context-block --budget 2000 --top-k 5`):
  was an answer in the block, and how many tokens did the block cost?

```bash
python3 benchmarks/dogfood-recall/run.py                 # uses `axil` on PATH
# before/after a change: both runs read one frozen, healed copy
python3 benchmarks/dogfood-recall/run.py --snapshot benchmarks/dogfood-recall/data/snap --out before.json
python3 benchmarks/dogfood-recall/run.py --snapshot benchmarks/dogfood-recall/data/snap --axil ./target/release/axil --out after.json
python3 benchmarks/dogfood-recall/run.py --axil ./target/release/axil \
    --out benchmarks/results/dogfood-recall-<label>.json
```

The eval copies the database to a temp dir first, so it never writes to the
live memory, and heals the copy (`axil heal --reindex`) so records a past bug
left without an embedding compete like the rest. It needs this repo's
`.axil/memory.axil`: the expected record ids exist only there. A run takes
about 5 minutes with a release build.

First baseline (2026-09-28, `axil` at `041c9e8`):
`benchmarks/results/dogfood-recall-2026-09-28-baseline.json` — hit@1 0.51,
hit@5 0.74, MRR@10 0.61, recall_all@10 0.70, NDCG@10 0.64; the hook's block
held an answer for 86% of questions at ~288 tokens.

## Questions

`questions.jsonl`, one per line:

```json
{"id": "q03", "kind": "decision", "question": "How do we make local builds link onnxruntime statically like the release binaries?",
 "expect": [["<decision id>"], ["<error id>"]]}
```

`expect` is a list of groups. A group is found when any one of its ids is in
the results (duplicates of the same commit, or a decision and the context note
that restates it). A question is fully answered (`recall_all`) when every group
is found.

Only records created before the cutoff in `run.py` count, so memories written
after the questions don't compete. Recall still weighs recency and decay, which
move with the clock, so compare runs made close together (a before/after pair
for one change), not runs weeks apart. Two plain runs an hour apart differed by
up to ±0.03 on every metric, mostly from memories stored in between; runs that
share a `--snapshot` see an identical corpus.

## Metrics

| Metric | Meaning |
|---|---|
| `hit@1`, `hit@5`, `hit@10` | Any expected group in the top 1/5/10 |
| `mrr@10` | 1 / rank of the first expected record |
| `recall_all@10` | Every expected group in the top 10 |
| `ndcg@10` | Ranking quality, one relevant item per group |
| `hook_hit` | An expected record is in the block the prompt hook injects |
| `hook_tokens` | Size of that block, bytes / 4 |
| `hook_hits_per_1k_tokens` | `hook_hit` per 1,000 injected tokens |
