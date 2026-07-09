# Session Compaction & Token Budgets

A long agent session accumulates a large, low-density history: tool
outputs, intermediate reasoning, dead ends. Axil's job is to turn that
sprawling working history into a small amount of *durable, retrievable*
memory — and then to hand only a bounded slice of it back to the next
session. This page covers both halves: the **compaction pipeline** that
distills a session down to episodes and a checkpoint, and the **token
budgets** that cap how much of the store any single recall or boot can
inject.

Nothing here requires an LLM. The agent (or the CLI hooks) writes the
structured records; Axil does the summarizing, superseding, decaying,
and budget-shaping algorithmically.

## The compaction pipeline

A session's transient state is compacted into durable memory in four
stages, each narrower and longer-lived than the last:

```
working log  ─►  episode  ─►  checkpoint  ─►  boot replay
(this session)   (past run)   (resume state)  (next session)
```

### 1. Working log — the live session

A session is a first-class record. Start one, log records against it,
and every logged record is linked back to the session via a graph edge
(`session_contains`) and stamped with its `_session_id`:

```bash
axil session start
axil session log <SESSION_ID> decisions '{"summary":"chose redb over sled"}'
axil session list --active
axil session history <SESSION_ID>
```

Working-memory records (`_working`) hold the current session's scratch
context — active tasks, tool outputs, open files. They are deliberately
short-lived: ending the session clears them.

The implementation lives in
[`crates/extensions/axil-memory/src/session.rs`](../../../crates/extensions/axil-memory/src/session.rs).

### 2. Episode — the durable summary

Ending a session transitions its working memory into an **episode** — a
compact, permanent record of what happened, with an outcome. The raw
working records are cleared; the episode remains:

```bash
axil session end <SESSION_ID> --summary "Fixed auth timeout; bumped pool size"
```

`session end` records the summary, outcome, `decisions_made`,
`files_touched`, duration, and linked-record count, then creates an
`_episodes` row from the session and clears the session's working
records. When a vector index is configured, the summary is embedded so
the episode is semantically recallable later. Episodes carry a default
90-day TTL (configurable); see [Memory Types](../concepts/memory-types.md).

This is the first big compression: an entire session's tool traffic
collapses to one summarized, embedded episode.

### 3. Checkpoint — the resume state

An episode says *what happened*. A **checkpoint** says *what the next
agent needs to pick this up*. It is a structured record — not free text —
with fields tuned for resumption:

| Field | Purpose |
|-------|---------|
| `goal` | The north-star intent (the most-often-lost field) |
| `state` | One or two sentences on where things stand |
| `next_steps[]` | Ordered, the single highest-value resume field |
| `open_questions[]` | Blockers and unresolved uncertainty |
| `references[]` | Typed pointers (`record`/`file`/commit/PR) — *not* copies |
| `summary` | Optional one-line headline, embedded for recall |

```bash
axil checkpoint '{"goal":"ship token-budget docs","state":"draft written","next_steps":["cross-link cognitive.md"],"references":[{"kind":"file","ref":"docs/src/advanced/session-compaction.md"}]}'
echo '{...}' | axil checkpoint -          # read payload from stdin
axil checkpoint --final '{...}'           # mark as the session's final checkpoint
axil checkpoint show                       # print the current checkpoint (stored or derived)
```

By default `axil checkpoint` writes a mid-session **snapshot** without
ending the owning session (auto-creating an active session if none
exists); `--final` stamps it as the session's closing checkpoint. The
key discipline is **reference, don't duplicate** — a
`{"kind":"record","ref":"<id>"}` pointer at a stored decision keeps the
checkpoint tiny and always current, because the reference is resolved
live at boot time (a superseded decision shows its current state, not a
stale copy).

Checkpoints live in `_checkpoint_records`. The full write/derive/render
logic is in
[`crates/extensions/axil-checkpoint/src/lib.rs`](../../../crates/extensions/axil-checkpoint/src/lib.rs).

### 4. Boot replay — resume instead of re-discover

The next session calls `axil boot`, which renders the current checkpoint
as a **"Resume Here"** block at the top of the wake-up context, before
rules, decisions, and errors:

```bash
axil boot                          # full boot context
axil boot --boot-format narrative  # plain-text narrative (Resume Here first)
```

If no explicit checkpoint was written, boot **derives** one from the
latest session's `summary`, `status`, `files_touched`, and unresolved
errors, and labels it as derived so the agent knows the difference.
There is also a freshness rule: a stored checkpoint is treated as stale
once a *newer* session exists that never wrote its own checkpoint, so
boot falls back to deriving from the newer session rather than replaying
an unrelated one.

End to end, a multi-hour session's history reaches the next session as a
handful of structured lines instead of a re-grep of the whole repo. See
[Context Economics](./context-economics.md) for why that matters.

## Token-budgeted recall

Compaction shrinks what is *stored*. Token budgets bound what is
*injected* into a prompt on any given call — so a chat-heavy database
can't dump hundreds of decisions into the context window.

### Bounding a recall

`axil recall` accepts a `--budget` that truncates the result set to fit
within an approximate token ceiling (estimated at ~4 bytes per token):

```bash
axil recall "auth timeout" --top-k 20 --budget 800
```

Recall is already compact by default (`{id, score, table, summary}` per
hit, with `axil get <id>` to expand), and near-duplicate hits are
collapsed so scarce slots aren't spent on restatements. `--budget`
layers a hard token cap on top of that: results are emitted in ranked
order and dropped once the running total would exceed the ceiling. The
library `remember()` API exposes the same cap via
`RecallOptions.max_tokens`
([`crates/extensions/axil-memory/src/recall.rs`](../../../crates/extensions/axil-memory/src/recall.rs)).

### Bounding a boot context

`axil boot` is budget-shaped too, with a priority-ordered drop policy:

```bash
axil boot --budget 1500
```

Boot assembles fixed, ordered sections (current scope → constraints →
recent decisions → active failures → open threads → preferences →
confidence notes). When the estimated total exceeds the budget, it drops
sections in **reverse priority order** — lowest-value first — and never
drops the four load-bearing sections (scope, constraints, decisions,
failures). Anything dropped is reported in a `dropped_sections` list so
the caller can see "we omitted X to stay in budget." The default budget
is picked to fit comfortably in a small prompt window. The contract
lives in [`crates/axil-core/src/boot.rs`](../../../crates/axil-core/src/boot.rs).

Token estimation is a pluggable seam
([`crates/axil-core/src/token.rs`](../../../crates/axil-core/src/token.rs)):
the default is the dependency-free `~4 bytes/token` heuristic, with an
optional tokenizer-backed estimator behind the `real-tokenizer` feature
for callers that need exact counts. All budget figures are estimates —
see the [Numbers integrity](./context-economics.md#numbers-integrity-policy)
policy for why they are always labeled as such.

## Keeping the store bounded

Compaction and budgets shape reads and writes; a third set of mechanisms
keeps the store itself from growing without limit as facts change and
age.

### Superseding — new facts retire old ones

When a new fact is stored and closely matches an existing same-table
record (vector similarity at or above the default `0.92` threshold), the
old record is marked superseded and a `supersedes` graph edge is created
from new to old. Superseded records are excluded from recall by default
but retained for history and migration questions. This requires a vector
index; without embeddings, no superseding happens. See
[`crates/extensions/axil-memory/src/supersede.rs`](../../../crates/extensions/axil-memory/src/supersede.rs).

### Consolidation — merge a fact timeline

Multiple facts about the same entity can be collapsed into a single
merged summary:

```bash
axil consolidate <ENTITY>
```

Without an LLM this is a template-based merge (latest value plus a count
of priors); with one configured it produces a prose summary noting what
changed and when. Either way the output is one consolidated record. See
[Optional LLM Grounding](./llm-grounding.md).

### Decay & pressure — age out the cold tail

Every record carries an importance score that decays over time
(exponential, default 90-day half-life; access reinforces it). Records
are sorted into Hot / Warm / Cold / Archived tiers by effective
importance:

```bash
axil decay --dry-run        # preview importance decay + below-threshold records
axil decay                  # apply decay
axil memory-pressure        # show tier distribution + archive candidates
axil memory-pressure --archive   # auto-archive records below the threshold
```

Pin the records that must never age out or be compacted away:

```bash
axil pin <ID>       # protect from decay / archive / deletion
axil unpin <ID>     # re-enable decay
```

Decay, consolidation, and inference also run as background worker tasks
(`axil worker run`), typically fired from a session Stop hook. For the
full importance/decay/belief model see
[Cognitive Memory](./cognitive.md).

### Hard reclamation

Decay and archiving are reversible states; `axil compact` is where space
is actually reclaimed — it hard-deletes expired and long-superseded
records and cleans orphaned edges, vectors, and FTS entries. High-
importance and pinned records are protected. Auto-compact fires when the
delete count crosses a configurable threshold. See
[Memory Hygiene](./memory-hygiene.md) for the full maintenance toolkit.

One current caveat: checkpoint records (`_checkpoint_records`) are **not
yet swept** by the decay/TTL machinery — they accumulate slowly (one row
per snapshot) and are pruned manually with `axil delete <ID>` until a
cross-extension retention hook lands.

## See also

- [Cognitive Memory](./cognitive.md) — importance scoring, decay, tiers, beliefs
- [Memory Types](../concepts/memory-types.md) — working → episodic transition, per-type recency
- [Memory Hygiene](./memory-hygiene.md) — compact, heal, maintain, and cadence
- [Context Economics](./context-economics.md) — why a smaller working context is cheaper and safer
- [Retrieval Pipeline](./retrieval-pipeline.md) — how recall scores and fuses candidates
- [Optional LLM Grounding](./llm-grounding.md) — what an LLM sharpens (and what runs without one)
- [Memory Commands](../cli/memory.md) — `session`, `boot`, `believe`, and friends
