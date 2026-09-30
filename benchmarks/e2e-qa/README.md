# End-to-end QA

Retrieval metrics say whether the right record was found. This asks the
question that matters to an agent: given what recall returned, can a model
answer? A reader model answers each question from the retrieved context alone;
a judge model grades the answer.

## Run

1. Dump contexts from a recall eval:

   ```bash
   # LongMemEval: top-k sessions in compact (first 200 chars each) and full form
   ./benchmarks/longmemeval/target/release/longmemeval-bench --variant s --limit 20 \
       --strategy recall-qtc --top-k 5 --dump-context benchmarks/e2e-qa/data/lme-s20-contexts.jsonl

   # Dogfood: the exact block the prompt hook injects, plus the expected records
   python3 benchmarks/dogfood-recall/run.py --dump-context benchmarks/e2e-qa/data/dogfood-contexts.jsonl
   ```

2. Grade:

   ```bash
   python3 benchmarks/e2e-qa/qa.py benchmarks/e2e-qa/data/lme-s20-contexts.jsonl --context full
   python3 benchmarks/e2e-qa/qa.py benchmarks/e2e-qa/data/lme-s20-contexts.jsonl --context compact
   python3 benchmarks/e2e-qa/qa.py benchmarks/e2e-qa/data/dogfood-contexts.jsonl --context hook
   ```

The report gives accuracy overall and per question type, the context size in
tokens (bytes / 4), and accuracy per 1,000 context tokens: the number that
shows whether injecting less still answers the question.

## Models and transport

The reader (Claude Haiku 4.5 by default) and the judge (Claude Sonnet by
default) run through headless Claude Code, `claude -p`, on the local
subscription rather than a billed API key. Each call runs in an empty temp
directory with no tools, hooks, MCP servers, user settings or project
instructions, and with a replaced system prompt, so the model sees only the
prompt. Calls count against the subscription's usage limits; the report lists
the tokens a run spent.

Because this is not the production API (Claude Code adds its own framing and
the temperature can't be set), use these numbers to compare Axil changes with
each other, not to publish against other systems. Publishable numbers need the
API and, for LongMemEval, its own GPT-4o judge.

The judge's grading rules follow LongMemEval's evaluator per question type
(off-by-one allowed for temporal reasoning, the updated answer for knowledge
updates, a rubric for preferences, "doesn't know" for abstention questions).
Dogfood answers are graded against the text of the expected records.

Responses are cached in `data/cache/` (gitignored) by model and prompt, so
re-running a report costs nothing; a changed context is a new prompt.

## First results (2026-09-28, reader Haiku 4.5, judge Sonnet)

| Contexts | Accuracy | Context tokens / question | Accuracy per 1k tokens |
|---|---|---|---|
| LongMemEval `s`, first 20, top-5 sessions in full | 0.85 | 15,563 | 0.055 |
| The same hits in the CLI's compact form | 0.00 | 297 | 0.000 |
| Dogfood, the block the prompt hook injects | 0.67 | 276 | 2.41 |

Files: `benchmarks/results/e2e-qa-2026-09-28-*.json`. The first 20
LongMemEval questions are all `single-session-user`; use a larger or
stratified subset before reading anything into per-type numbers.

In the dogfood run, error questions scored 0.37 even though the expected error
was in the hook's block 95% of the time: the block shows an error's first line
cut to 240 characters, never its root cause or fix, so "why did X fail / how
was it fixed" can't be answered from it. Decisions lose their `reason` the
same way. (Fixed since: see below.)

## Per-table context lines (2026-09-30)

The block now renders each hit by its table, inside the same 240-byte line
bound: an error as `error → fix (root cause)`, a decision as
`summary — reason`, a commit as its subject, anything else as before. Both
runs read one frozen corpus (a copy of `dogfood-recall/data/snap-prefix`), so
ranking is identical (same hit@k, NDCG and `hook_hit` 0.89); only the lines
differ. Old is `axil` at `65d45ff`.

| Block lines | Accuracy | Errors (19) | Decisions (27) | Context tokens / question |
|---|---|---|---|---|
| One field per hit (`65d45ff`) | 0.74 | 0.53 | 0.78 | 280 |
| Per-table lines | 0.86 | 0.79 | 0.85 | 299 |

Seven questions went from wrong to right and none the other way. Commit,
context and rule questions were already all right. One run each, and the
reader's temperature can't be set, so read single-question differences as
noise. The old row is not comparable with the first results above: that run
used an older snapshot and binary.

Files: `benchmarks/results/e2e-qa-2026-09-30-dogfood-hook-{old,context-lines}.json`
and the matching `dogfood-recall-2026-09-30-snap-prefix-*.json`.

The new side was built from the change before it was committed, so its
dogfood file names `axildb-v2.3.1-25-g65d45ff-dirty`. A rebuild of that same
rendering from a commit, run on a fresh copy of the corpus, gave the same
hit@k, MRR, NDCG and per-question `hook_hit`, and `hook_tokens` 299.285
against the 299.281 here. One detail came after the measurement: an error
whose root cause opens with its own parenthesis now reads `(cause: (1) …)`
rather than `((1) …)`. That changes one line in 4 of the 57 blocks (q27, q35,
q40, q41; q27 is one of the seven gains); the line keeps its length and its
parts share 7 fewer bytes. QA was not re-run for it.
