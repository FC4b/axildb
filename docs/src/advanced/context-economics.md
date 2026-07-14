# Context Economics

Axil's first job is to keep an agent's **working context small**. An agent
that has to *discover* the relevant code by reading files drags a large
amount of text into its context window; an agent that asks Axil gets a
compact, pointer-shaped answer instead. A smaller working context is
cheaper, faster, and — because there is less irrelevant text to
misattribute — measurably less prone to hallucination.

This page explains the mechanisms, shows how to measure the saving on your
own repo, and records the numbers from Axil's own codebase.

## Numbers integrity (policy)

Every savings, compression, speed-up, or reduction figure Axil surfaces —
in the README, in these docs, or in CLI/MCP output — must be one of:

1. **Measured** against a real, named baseline, or
2. **Labeled an estimate**, naming the heuristic that produced it, or
3. **Sourced** to a committed, reproducible benchmark.

Never print a bare multiplier or percentage a reader (or an agent) could
mistake for a measured result. The honest pattern is used throughout this
page: the synthetic `context-savings` figure is always tagged "optimistic
ceiling," the A/B number is "equal-correctness, conditional," and every table
states what it does *not* cover. When in doubt, **show the inputs** so the
figure can be checked (e.g. `~{ratio}:1 (N index tokens vs M source tokens)`),
and mark anything derived from the `~4 chars/token` heuristic with a `~` or
`est.` rather than presenting it as a real count.

This is a contributor *and* agent rule, not a style note: a surfaced number
that can't be traced to (1), (2), or (3) is a bug — fix the number or fix the
label.

## Why smaller context helps

- **Less distractor text.** When the answer is 80 relevant tokens instead
  of 30k tokens with the right line buried inside, the model has less to
  misread. Retrieval that returns *pointers* keeps the signal-to-noise
  ratio high.
- **Headroom.** Staying off the context ceiling avoids mid-task
  compaction/truncation — the point where agents "forget" earlier
  decisions and start contradicting themselves.
- **Cost and latency.** Fewer input tokens per turn is directly fewer
  tokens billed and less to process.

## The five mechanisms

| Mechanism | Command | What it replaces |
|-----------|---------|------------------|
| Indexed summaries (~15:1) | `axil context` | Reading whole files to learn structure |
| Code proxies (structural recall) | `axil code-search`, `axil code-context` | `grep` + reading candidate files |
| Compact recall + near-dup collapse | `axil recall` (compact by default; `--budget N`) | Full-payload JSON dumps |
| Boot context | `axil boot --budget N` | Re-grepping the repo to reconstruct "where were we" |
| Persistence across sessions | any recall, next session | Re-reading everything every session |

The biggest lever is the last one: without memory, *every* session re-pays
the discovery cost. With Axil, session 2+ pays a few hundred tokens for
`axil recall "topic"` instead of tens of thousands re-reading files.

## Two ways to measure — and why they disagree

There is a quick *estimate* and a real *experiment*. They give very
different numbers, and the gap is itself the lesson.

### Quick estimate: `axil context-savings`

`context-savings` runs **real recall** per task, then compares the compact
pointer block Axil returns against an **upper-bound** baseline: the full
source of every file the hits point at — i.e. *if the unaided agent read
those whole files*.

```bash
axil context-savings                       # built-in dogfood tasks
axil context-savings --task "where is auth handled"
axil context-savings --tasks tasks.json --format markdown
```

On Axil's own repo this reports ~99% reduction (~557 vs ~70k tokens). **Treat
that as an optimistic ceiling, not a real-world figure.** Its baseline
assumes whole-file reads; a competent agent greps to a ~15-line range, so
the realistic baseline is far smaller. Token counts use the shared
`~4 chars/token` heuristic.

### Real experiment: same task, two sandboxes, real agents

The honest test clones one repo into two byte-identical sandboxes
(`without/` plain, `withdb/` Axil-indexed), has a **real agent** answer the
same task in each (one restricted to `grep`/read, one to the `axil` CLI),
verifies both answers against ground truth, and recomputes tokens
**mechanically** from what each agent actually consulted — counting only
tasks where *both* answers are correct.

```bash
scripts/context-ab-setup.sh        # clone flask → without/ + withdb/, index withdb
# run the scripts/context-ab.workflow.js workflow (real agents, per task)
scripts/context-ab-score.py --manifest experiments/context-ab/run.json \
  --without-root experiments/context-ab/without/flask \
  --withdb-root  experiments/context-ab/withdb/flask
```

## What the real experiment found

### Current headline — v2.1.1, 2026-07-13 (three repos, tokens *and* steps)

The maintained baseline, re-measured end-to-end on **axil v2.1.1** with
disciplined Opus 4.8 agents on three public repos. Committed data:
[`benchmarks/results/context-ab/`](https://github.com/FC4b/axildb/tree/main/benchmarks/results/context-ab);
task fixtures: `benchmarks/context-ab/`. This run also measures
**steps-to-finish** (consulted tool round-trips ≈ agent turns), the second
axis on the README hero chart:

Figures **pool two same-day runs** per corpus — v2 (index at the old
100 KB default cap) and v3 (index at the fixed 512 KB default) — per-run
splits are in the committed artifact:

| Corpus | task-runs (both-correct) | no Axil tok | w/ Axil tok | Token reduction | Steps |
|--------|--:|--:|--:|--:|:--|
| flask (24 files) | 20 | 22,190 | 10,476 | **52.8%** | 38 → 34 (**10% fewer**) |
| fastapi (mid) | 19 | 87,951 | **5,384** | **93.9%** | 56 → 32 (**43% fewer**) |
| **django (906 files)** | 20 | 32,978 | **8,102** | **75.4%** | 59 → 36 (**39% fewer**) |

**Sensitivity (read before quoting fastapi's 94%):** one v3 task — "where
does FastAPI run sync endpoints in a threadpool?" — cost the unaided agent
**63,311 tokens** (it read `fastapi/routing.py`, 253 KB, *whole*) versus
Axil's **94**. Excluding that single pair, fastapi is **78.5%**. It is
included because it is real behavior: whole-file reads of large files are
exactly the discovery cost a code index removes — but it shows discovery
costs are heavy-tailed and single-run aggregates are fragile.

The aggregates include per-task losses (fastapi's `APIRouter`, django's
`SQLCompiler.as_sql` — cases where a verbose fallback beat a tight grep) —
measured, not cherry-picked. Two cautionary tales on sample size: flask
read **−20%** (Axil worse) on its first 3 tasks, **+50%** at n=7, **+42%**
at n=11, and **+53%** pooled over 20 task-runs; per-run reductions ranged
flask 42–69%, fastapi 75–98%, django 72–78%. Quote the committed n and
pooling alongside any figure.

**Retrieval recall on the same tasks** (the hero chart's third panel): a
mechanical replay of every Axil lookup the agents actually ran, testing
whether the ground-truth answer file surfaced — flask **91%**, fastapi
**83%**, django **77%** (n = 11/12/13, index cap 512 KB;
`benchmarks/results/context-ab/code-recall-agent-queries-512k-2026-07-13.json`).
Two instructive companion artifacts are committed alongside: the
**default-config run** (fastapi drops to **58%** because the indexer's
old 100 KB default **silently skipped** `fastapi/routing.py`, 253 KB —
found by this measurement, fixed to 512 KB + a loud skip warning), and a
**verbatim-question diagnostic** (25–38%: pasting the full natural-language
question as the query underperforms badly vs the short symbol/keyphrase
queries agents actually issue — query *like an agent*).

**Post-v2.1.1 fixes re-measured (2026-07-14,
`code-recall-agent-queries-methodproxies-2026-07-14.json`).** Two fixes in
the working tree — method-level Python proxies (`Class.method` symbols) and
the ignore-boundary fix (a nested project no longer inherits its *parent*
repo's `.gitignore`, which had silently excluded Django's entire
`db/models/` subtree from every earlier run) — move the same replay to
flask **91%**, fastapi **75%**, django **100%** (aggregate 83%→89%).
Django's perfect score is both fixes working; the fastapi/flask dips are
**ranking dilution**: ~4× more symbol proxies now compete, so broad queries
can crowd the file-level answer out of the top-5 (`fa1`, `fl10`). Result
diversification across proxy kinds is the tracked follow-up.

**SCIP does not move these numbers (yet).** Ingesting scip-python indexes
(django: 43,902 entities, 152k call edges) leaves the replay recall
identical — plain `code-search`/`fts` operate over the structural proxy
table, which gains no method-level entries from SCIP (`proxy_backfill`
upgrades only *provisional* regex entities, absent in a fresh index). The
remaining misses (`SQLCompiler.as_sql`, `QuerySet._fetch_all`) are a
proxy-granularity gap; minting method-level proxies from SCIP definitions
is the tracked next step
(`code-recall-agent-queries-scip-2026-07-13.json`).

### Earlier runs — 2026-06, flask + django only (the discipline lesson)

The original three runs, each counting only tasks both agents answered
correctly:

| Run | Corpus | Agent | no Axil | w/ Axil | Result |
|-----|--------|-------|--:|--:|:--|
| 1 | flask (24 files) | naive | 3,018 | 7,525 | Axil **2.5× worse** |
| 2 | flask (24 files) | disciplined | 4,695 | 5,519 | ~parity |
| 3 | **django (906 files)** | disciplined | 11,218 | **3,006** | **Axil 3.7× cheaper (73%)** |

"disciplined" = the agent leads with cheap `code-search`/`fts` and uses the
heavy `code-context` bundle at most once.

**The saving is real but conditional:**

1. **Axil wins big on large codebases + semantic questions** ("where does
   the ORM compile a QuerySet to SQL?"). The unaided agent has no obvious
   symbol to grep, so it greps several guesses and reads multi-hundred-line
   ranges across the tree (Django URL resolver: 2,250 tok); Axil answers
   from two `code-search` calls (193 tok). Per-task savings on Django ran
   35–95%.
2. **On a tiny idiomatic repo it only ties.** When the agent knows to
   `rg "def jsonify"` and read 15 lines, grep is near-free — Axil's compact
   lookup matches it but can't beat it, and a single `code-context`
   fallback tips a couple of tasks negative.
3. **Usage discipline matters** (run 1 → 2: −149% → parity). `code-search`/
   `fts` cost ~80 tok/hit; `code-context` is a 0.5–2.2k-token JSON
   *task-brief assembler*, not a symbol locator. Lead with the former.

> **The synthetic `context-savings` ~99% is an optimistic ceiling** — its
> baseline assumes reading whole files. The defensible, equal-correctness
> numbers (v2.1.1, 2026-07-13, 39 pooled task-runs) are **~75% on a large
> repo, ~94% mid-size (~78% excluding one whole-file-read outlier), ~53%
> on a tiny one** — smallest on tiny repos where a tight `grep`+range-read
> is already cheap, largest wherever unaided discovery hits big files.

## Fixing the conditional: lean `code-context`

Every Axil *loss* above was the same line item — a `code-context` call,
which returned a fat JSON bundle (`relevant_code` + `graph_neighbors` +
`similar_context` + `recent_changes` + scores/ids/nulls) at 0.5–2.2k tokens.
As a *symbol locator* that is mostly noise. `code-context` now defaults to a
**compact** output — just the ranked `path:line symbol — why` pointer lines
(use `--context-format json` for the full bundle). On the same query that
cost 900 tokens of JSON, the compact form is **145 tokens (−84%)**.

Re-scoring the same agent trajectories with the lean output flips the
result:

| Run | before | after lean `code-context` |
|-----|--:|--:|
| flask (small) disciplined | −18% (Axil worse) | **+14% (Axil wins)** |
| django (large) disciplined | +73% | **+80%** |

So the conditional is largely closed: with compact output Axil is
net-cheaper on **both** small and large repos. The remaining per-task
negatives are recall *misses* — the agent pays for extra queries when the
first lookup doesn't surface the answer.

The 2026-07-13 v2.1.1 live runs (fresh agents, not re-scored
trajectories — see the current headline above) confirm the direction of
this prediction and land higher: flask **+53%**, django **+75%** (pooled).

## Recall output discipline: compact default + near-dup collapse

The `code-context` fix above made *one* command compact by default. The same
discipline now applies to `axil recall` itself:

- **Compact is the default.** `axil recall` returns `{id, score, table,
  summary}` per hit instead of the full record JSON. The dropped detail is
  one call away — every hit carries its `id`, and `axil get <id>` (or
  `--recall-format full`) expands it. Lossy on the wire, lossless on demand.
- **Near-duplicate collapse.** Before truncating to `top_k`, recall collapses
  near-identical, **same-table** hits (a lexical 64-bit SimHash, Hamming ≤ 3)
  into the highest-scored representative, so the scarce slots aren't spent on
  near-exact restatements. It is deliberately conservative — only near-*exact*
  redundancy collapses, never distinct content — and it is scoped to a single
  table so a downstream `--table` filter can't silently lose a record. Needs no
  vector index. Disable with `--no-dedup`.

**Measured on the same Django + Flask `context-ab` corpora**, routing the 15
ground-truth questions through `axil recall` and comparing the old default
(full + no-dedup) against the new default (compact + dedup), same binary and
DB, back-to-back. Ranking and ids are identical across the two, so correctness
is held equal *by construction*:

| Corpus | recall, old (full) | recall, new (compact+dedup) | Reduction |
|--------|-------------------:|----------------------------:|----------:|
| **Django** (8 questions) | 9,192 tok | **1,720 tok** | **81.3%** |
| **Flask** (7 questions) | 8,780 tok | **1,530 tok** | **82.6%** |
| **Both** | 17,972 tok | **3,250 tok** | **81.9%** |

Attribution from a three-way split (full → compact → compact+dedup): essentially
**all** of the win is the compact projection. On these freshly-indexed code
repos same-table near-duplicate collapse contributes **~0%** — their apparent
near-dups are *cross-table* (a file proxy and its symbol proxy, in different
`_idx_*` tables), which collapse deliberately does **not** merge (doing so would
let a `--table` filter silently drop a record). Same-table near-exact dups are
rare in a fresh index, so dedup stays a quiet safety net here.

> **Scope — read this before quoting the number.** This measures the **`axil
> recall` surface specifically**, *not* the grep-vs-Axil A/B above. The
> headline 73–80% figures came from `code-search`/`fts`, which were already
> pointer-shaped and are unchanged by this work. Quote 81.9% as "how much
> compact-default recall saves over the old full-JSON recall," not as a new
> grep-vs-Axil number.

One honest detail: same-table near-dup collapse found **0** duplicates on Axil's
own maintained `.axil` (insert-time supersede already keeps it dup-light) and
**~0** on these fresh external repos (their redundancy was cross-table, which it
correctly leaves alone). It is a zero-cost safety net that earns its keep only
when a single table accumulates genuine near-exact restatements — common on an
un-curated store written via raw `store` without embedding.

## Recall quality: why misses happen, and the fix

Measuring recall@k on the ground-truth tasks (`scripts/recall-quality.py`)
exposed the real gap. On the large corpus, semantic queries had **recall@1
= 0%** — and the cause was *not* a reranker. Three root causes, fixed in
order of leverage:

1. **Text-sparse symbol proxies.** A class with no docstring embedded as
   *just its breadcrumb* (`URLResolver` → 13 tokens), so a conceptual query
   had nothing to match. **Fix:** the parser adds the class's base classes
   and a **method-name digest** to the proxy text — the method names
   (`resolve`, `compile`, `as_sql`) *are* the concept terms.
2. **CamelCase-blind, single-token name boost.** `"url resolver"` couldn't
   boost `URLResolver` because the boost matched the whole query as one
   token and never split CamelCase. **Fix:** a per-term, CamelCase-aware
   **identity boost** rewards a proxy for each query content-word in its
   symbol/path/breadcrumb, over a larger candidate pool.
3. **Raw-similarity-polluted fusion (the big one).** The RRF fusion seeded
   each entry with the backend's *raw* score (vector cosine ~0.7 vs BM25
   ~0.9–15, different scales) and only *added* the rank term + boost on top
   — so raw magnitudes dominated and both RRF and the boost were rounding
   errors. **Fix:** fusion is now **pure RRF** (entries start at 0), so rank
   + boost actually decide. Plus **query-side expansion** — the query's
   content words concatenated (`"url resolver"` → `"urlresolver"`, which the
   embedder matches to `URLResolver`) and a small synonym/abbrev bridge.

Result — recall@k jumped:

| Metric | before | after |
|--------|-------:|------:|
| django file recall@3 | 38% | **62%** |
| django file recall@5 | 50% | **75%** |
| django clean queries hitting top-5 | 1/6 | **5/6** |
| flask file recall@1 | 29% | **71%** |
| dogfood gate top-3 symbol | 20% | **40%** |

No regression (fixture 100/100, dogfood improved). `"url resolver"`,
`"migration autodetector"`, and `"WSGI handler"` now rank **0**. Remaining
hard case: short queries with many same-named classes (`"SQL compiler"` —
django ships a base plus per-backend compilers). Memory recall is a separate
path (`axil-core fuse_signals`) and is unaffected.

> **Takeaway for agents:** for "where is X" use `axil code-search` / `axil
> fts` (now boosted on symbol-name matches); `axil code-context` is lean by
> default and safe to use as a one-shot task brief.

Full write-up, per-task traces, and caveats live in
`experiments/context-ab/report.md` after a run.

See also: [Performance](./performance.md), [Retrieval Pipeline](./retrieval-pipeline.md),
[Indexing & Scale](./indexing-and-scale.md).
