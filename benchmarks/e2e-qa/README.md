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
same way.
