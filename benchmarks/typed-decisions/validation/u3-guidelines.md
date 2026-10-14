# U3 capture-classification labeling guidelines

These rules define the gold labels for `sets/u3_texts.jsonl`. Each item is a
piece of text an AI coding agent produced or saw while working on a Rust
project named Axil (an embedded memory database for AI agents):

- the output of a shell command (some failed, some did not), often cut down,
  with `[... N lines cut ...]` where lines were removed;
- a paragraph the agent wrote: narration, a status update, a finding, a report;
- a paragraph of a task brief one agent wrote for another;
- a git commit subject line.

Items were shortened, so one can end mid-sentence or mid-line. Every
annotator, model or human, labels from these rules alone, using only the
item's own text; items are independent. Placeholders such as `$SCRATCH/`, `~/`
and `[REDACTED]` stand for paths and secrets that were removed.

## 1. What the label is for

Axil has an **auto-capture** step: it reads text like this and decides whether
to store it as a memory record, and in which table. The agent reads those
records back in later sessions: open errors and recent decisions are shown at
the start of every session, and every record can come back in search.

So each item gets one **class**, the table the text belongs in, or `none`:

> If this text were stored as a memory record, would a future session on this
> project be better off, and which kind of record would it be?

A stored junk record costs more than a missed one: a junk *error* is shown as
an open problem at the start of every session until someone resolves it, and
junk crowds real memories out of search. The agent also stores memories by
hand, so a missed capture is often recovered. **When you are unsure whether an
item is worth a record at all, choose `none`.**

Each item that is not `none` also gets an **importance**: `low`, `medium` or
`high` (section 4).

## 2. Output format

Return only a JSON array, no prose and no markdown fence, one object per item
in the order given:

```json
[{"id": "x-001", "class": "error", "importance": "medium", "rule": "E1"},
 {"id": "x-002", "class": "none", "importance": null, "rule": "N1"}]
```

- `class` is one of `decision`, `error`, `context`, `rule`, `none`.
- `importance` is `low`, `medium` or `high`, and `null` exactly when `class`
  is `none`.
- `rule` is the id of the guideline rule that decided the class (for example
  `E2`, `N3`, `P1`), so disagreements can be traced. One id is enough.

## 3. Classes

Decide in this order: first check the `none` tests (3.1). If none of them
applies, pick the class whose test the item's **main point** passes (3.2 to
3.5). If two classes fit equally, use the precedence in 3.6.

### 3.1 `none`: nothing worth a record

The item is `none` when any of these holds:

- **N1 Narration.** The agent says what it is about to do, is doing, or just
  did, with no finding: "Let me check the config first.", "Now the tests.",
  "Committing.", "While that builds, I'll write the harness.", "Retrying with
  a narrower pattern:". A reason attached to a next step does not change this
  ("Checking the batch size, since that is the likely cause:").
- **N2 Normal output.** Tool output that shows no failure: build progress
  (`Compiling ...`, `Finished ...`), passing test runs, file listings, source
  code or diffs printed for reading, search hits, JSON rows, counts, version
  strings, background-job notices. Code that *contains* error-handling words
  (`map_err`, `eprintln!("... failed ...")`, `Err(e)`) is still just code.
- **N3 Exit status as the answer.** A non-zero exit that is the command's
  normal way of answering, not a failure: `grep`/`rg` found nothing (often
  shown as `0 matches for ...` or an empty result), `diff`/`cmp` found
  differences, `test`/`[` was false, `git diff --quiet` saw changes, a
  formatter check printed the diff it would apply. When the only failure in a
  shell item is the last command of a pipeline answering this way, the item is
  `none`.
- **N4 Command-line mistake.** The only thing that failed is the command line
  itself, and nothing is learned about the project or its tools: a shell
  syntax or quoting error, a typo'd command or flag, a glob that matched
  nothing (`no matches found: ...`), zsh treating `===` as a command
  (`=== not found`), a regex the agent wrote that does not parse, a missing
  required argument, a path that does not exist because the agent guessed or
  mistyped it, a `cd` into a directory that is not there.
- **N5 Fragment.** No claim can be read from the text: a heading alone, a
  table or list cut off before any finding, raw numbers or ids without
  meaning, a code fence that only holds a command line.
- **N6 Task mechanics.** One-off instructions for the current task only: the
  report format to return, the commands to run for this check, which worktree
  or branch this one task uses, "read file X first". (A constraint that keeps
  applying to later work is a `rule`, see 3.5.)
- **N7 Unsettled.** Questions, options under consideration, plans that are not
  committed, conditional plans ("if X passes, I'd merge it"), with no finding.
- **N8 Harness noise.** Messages from the agent's environment, not about the
  project: usage-limit notices, permission prompts, "the session ended".

A shell item that shows a real failure (section 3.3) somewhere in it is not
`none` just because other parts are normal output.

### 3.2 `decision`: a settled choice

**D1.** The text commits to one option over others for the project's design,
tooling, process or scope, as done or decided, not proposed: "Use X instead
of Y", "switched to", "chose", "dropped Y", "keep X off", "X stays", "not
doing Y because Z".
- The alternatives may be implied ("parse lazily instead of building the
  index up front" rejects building it up front).
- A momentary working step is not a decision (N1): "Using the debug binary
  for this quick check" is narration.
- A choice reported second-hand counts ("the user chose to keep X").
- **D2 Commit subjects.** A commit subject is a `decision` when it names the
  choice it makes, usually with what it replaces or rules out: `perf(fts):
  stream postings instead of loading the segment`, `refactor: replace the
  hand-rolled LRU with the lru crate`, `build: keep LTO off for dev builds`.

### 3.3 `error`: something failed or is wrong

**E1 Failure output.** Shell output showing that the project's code, build,
tests, scripts or tools failed: a compiler error, a failing test or
assertion, a panic, a traceback from a script the agent ran, a crash or kill
(`exit 137`), a network or permission failure, a tool reporting an error
about real input. A traceback from a throwaway one-liner the agent typed is
still E1 (the script failed), unless it is only a typo (N4).

**E2 Failure report.** Prose reporting a specific failure or defect that
happened or exists: what broke, what is wrong, a bug found, a release that
failed, a test that is flaky, with or without its cause and fix ("The nightly
job failed because the runner image dropped Python 3.10.", "Root cause: ...",
"Search has ignored the date filter since the last refactor").

**E3 Fix commit.** A commit subject that names the wrong behavior it fixes,
so the defect can be read from it: `fix(graph): stop double-counting edges on
re-insert`, `fix(cli): exit non-zero when the export fails`. A `fix:` subject
that names no defect ("address feedback", "misc fixes", "tidy up") is `none`
(N5).

Not errors: a passing run that mentions `0 failed` (N2); a failure that is
only a command-line mistake (N4) or an exit status as the answer (N3); a plan
to fix something not yet described; a report that a previous error is now
**resolved** with no new failure in it, which is `context` (C1) when it
records how it was fixed or verified.

### 3.4 `context`: how things work, or a finding

**C1.** A durable statement about how the project, its code, its tools or its
environment work, or a finding that stays useful after this session:
architecture, behavior, a gotcha phrased as a general fact, a how-to, a
measured result with its meaning, the state of a release or a branch that
later work depends on, a verified outcome ("200 parallel runs finished with no
hang, so the fix works").
- Test: would the statement still be true and useful to someone starting a
  new session tomorrow? Progress of the current task ("3 of 5 agents are
  done") is not (N1).
- A results table or list counts only if it states what was measured and the
  outcome can be read from it; otherwise N5.
- **C2 Commit subjects.** A commit subject is `context` only when it states
  how something works or a measured fact beyond the change itself. A bare
  change announcement (add, implement, support, bump, update, merge, docs,
  style, chore) with no choice, failure or how-it-works fact is `none` (N5):
  the change itself is already in git.

### 3.5 `rule`: a standing constraint

**R1.** A normative statement that constrains later work on the project,
beyond the current step: "never", "always", "must", "do not", "only",
"every X must Y", conventions to follow. Test: would it still apply to the
next task in a new session? "Never edit generated fixtures by hand", "Every
new internal table name starts with an underscore", "Run the parity test after
any tokenizer change" are rules. "Work only in worktree X", "report in 500 words",
"run these three checks" are bound to one task (N6).
- An item that mixes standing rules with one-off task instructions is a
  `rule` if at least one standing rule is stated in it.

### 3.6 Precedence

When two classes fit the main point equally: **error > decision > rule >
context**. Examples: a report that names a failure and the choice made in
response is `error` when the failure is the headline, `decision` when the
choice is; "X broke, so the rule is now: never do Y" is `rule` only if the
rule is the main point.

**P1 Mixed items.** An item listing several findings is labeled by its first
substantive finding (the headline), not by the majority of its lines.

## 4. Importance (only when the class is not `none`)

Judge how much a future session would lose without this record.

- **high** when any of these holds:
  - **H1** it constrains or redirects later work across the project: a
    project-wide rule, or a decision about architecture, storage format,
    release, process, or what ships (a rule or decision scoped to one module
    is `medium`);
  - **H2** it reports or fixes a failure that loses or corrupts data, blocks a
    release, install or CI, is a security problem, or makes the product give
    wrong results silently;
  - **H3** it records a verdict or measurement other work depends on (a
    GO/NO-GO, a benchmark baseline, "the fix holds on all 200 runs").
- **low** when none of the high tests holds and any of these does:
  - **L1** transient failure output with no cause or lesson: one compile error
    mid-edit, a failing test run with no diagnosis, a traceback from a
    throwaway script;
  - **L2** trivial or cosmetic: a typo, wording, formatting, a lint;
  - **L3** a narrow fact unlikely to be needed again.
- **medium** otherwise: a specific bug with its cause, a scoped design choice,
  a tool gotcha, a how-to, a finding about one module.

## 5. Worked examples

These are invented, in the style of the items.

1. `Exit code 101` followed by `error[E0308]: mismatched types --> src/db.rs:42`
   and `error: could not compile axil-core` → `error`, `low` (E1, L1: one
   compile error mid-edit, no cause).
2. `Exit code 1` followed by forty lines of source code and
   `0 matches for 'fn open_store'` → `none` (N3: the grep at the end found
   nothing; the code is N2).
3. `Exit code 1` and `zsh: no matches found: docs/**/*.toml` → `none` (N4).
4. `Exit code 1`, a Python traceback ending in
   `KeyError: 'rows'` from a `python3 -c` one-liner → `error`, `low` (E1, L1).
5. `test result: FAILED. 87 passed; 3 failed` with the three test names →
   `error`, `low` (E1, L1: no diagnosis). With a line explaining the shared
   cause → `medium`.
6. `test result: ok. 219 passed; 0 failed` → `none` (N2).
7. "Let me rerun the suite with one thread to rule out a race:" → `none` (N1).
8. "The flaky test is a race: two tests share one temp dir. Giving each its own
   dir fixed it (20 runs green)." → `error`, `medium` (E2: the defect and its
   fix).
9. "Import silently skipped every record with a non-ASCII key, so the restored
   database was missing 4% of its rows and reported success." → `error`,
   `high` (E2, H2: silent data loss).
10. "Decided to keep the int8 model off by default: it loses 2 points of recall
    for 30% less RAM." → `decision`, `high` (D1, H1).
11. "Using the debug binary for this quick check." → `none` (N1).
12. "Hooks run in a fresh process each time, so anything a hook loads (the
    model, the index) is paid on every call." → `context`, `medium` (C1).
13. "Never commit the downloaded model files; the build fetches them by
    revision." → `rule`, `high` (R1, H1).
14. "Report back in under 300 words with the commit sha." → `none` (N6).
15. "If the benchmark holds, I'd merge it into dev; otherwise drop the branch."
    → `none` (N7).
16. `feat(cli): add --dry-run to prune` → `none` (C2/N5: bare change).
17. `fix(hook): stop storing every commit twice` → `error`, `medium` (E3).
18. `refactor(cache): drop the disk tier, keep entries in memory only` →
    `decision`, `medium` (D2).
19. `Merge branch 'feat/x' into dev` and `chore(release): bump to 2.4.0` →
    `none` (C2).
20. "**Deadlock: resolved.** 200 parallel runs finished with no hang after the
    lock-order fix." → `context`, `high` (C1: a verified outcome; H3).
21. "The user denied permission for this command." → `none` (N8).
22. "The CI cache key ignores the lockfile, so a dependency bump reuses stale
    builds until the cache expires." → `context`, `medium` (C1).

## 6. Checklist before you answer

- Did you check the `none` tests first, and choose `none` when unsure whether
  the item is worth a record?
- For a failed shell command: did something in the project or its tools fail
  (E1), or did the command only answer (N3) or get typed wrong (N4)?
- Is code that merely contains the word "error" or "failed" labeled `none`?
- Is `importance` `null` exactly for `none` items?
- One object per item, in order, ids copied exactly, valid JSON only.
