## Axil Agent Brain

This project uses **Axil** as your persistent memory. The DB auto-detects at `.axil/memory.axil` — no flags needed.

**Background hooks** auto-track file changes. Your job is the high-level knowledge: decisions, architecture, gotchas.

### Axil is NOT read-only — you write to it via Bash

You will see `<context source="axil">` blocks injected into your conversation. **Those are read-only signals from the harness.** They are NOT the whole interface.

**To write to Axil, run shell commands via the Bash tool:**

```bash
axil store decisions '{"summary":"…","reason":"…"}'
axil store errors    '{"error":"…","root_cause":"…","fix":"…"}'
axil store context   '{"type":"…","summary":"…"}'
axil recall "<query>" --top-k 5
axil boot
```

If `axil` isn't on PATH, fall back to `./target/release/axil` or the absolute path. The CLI is the write interface — there is no separate tool, just `Bash(axil ...)`.

### CRITICAL: DB is your primary source of truth

**NEVER read task files, grep for project status, or manually explore for context before consulting the DB.**

**Axil-first target:** almost every repo-context lookup should start in Axil, then use normal file reads/search only to verify current code. Do not treat Axil as an optional afterthought.

**Search/query gate:** before any repo-discovery command (`rg`, `grep`, `git grep`, `find`, `fd`, `ls`, `tree`) or any "where/how/what changed" question, run the most relevant Axil command first.

When asked "what's next", "what happened", "what did we decide", or any project status question:
1. Run `axil boot` first (or `./target/release/axil boot` if not on PATH)
2. Run `axil recall "<topic>" --top-k 5` for specific queries
3. Only read files if the DB doesn't have what you need

When you need code or docs context:
1. For symbols, modules, APIs, or "where is X?", run `axil code-search "<query>" --top-k 5` before `rg`/`grep`
2. For exact docs/config text, run `axil fts "<term>" --limit 5` before broad text scans
3. For a known file, run `axil recall-for-file "<path>" --top-k 5` before editing it
4. Open/read the specific files Axil returns and verify against current code

Bypass Axil only for a user-named exact file/line, command output from the current turn, or a tiny local edit that needs no project context.

The SessionStart hook runs `axil boot` automatically when the session starts, injecting recent decisions, errors, and session history into your context. It fires again after a compaction (and after `/clear`), so the boot and the latest checkpoint come back without you re-running anything; recall anything specific the compaction summary dropped.

```bash
# For deeper context on a specific topic:
axil recall "<specific topic>" --top-k 5
```

### MANDATORY: Store after every completed unit of work

This is NOT optional. After each of these events, store IMMEDIATELY before moving on:

**After completing a task or feature — write a structured checkpoint (Phase 18):**
```bash
axil checkpoint '{"goal":"<north-star intent>","state":"<where things stand>","next_steps":["<what to do next>"],"open_questions":["<unresolved>"],"references":[{"kind":"file","ref":"<path>"}]}'
```

This replaces the old `axil store context '{"type":"session_summary",…}'` pattern. The checkpoint is what `axil boot` replays as the "Resume Here" block on the next session — so write the smallest payload that lets the next agent pick up without re-discovery. `next_steps` and `references[]` are the highest-value fields. Use `{"kind":"record","ref":"<id>"}` to point at a stored decision/error/context rather than copying it. See the `axil-checkpoint` skill for the full template.

**After making a design decision (choosing approach A over B):**
```bash
axil store decisions '{"summary":"<what>","reason":"<why>","files":["<affected>"]}'
```

**After hitting a bug, gotcha, or fixing a review finding:**
```bash
axil store errors '{"error":"<what broke>","root_cause":"<why>","fix":"<how>"}'
```

**When a stored error is fixed** (so `axil boot` stops listing it as open):
```bash
axil resolve <error-id> --by <id-of-the-record-that-fixed-it>
```

**After learning how the codebase works:**
```bash
axil store context '{"type":"architecture","summary":"<what you learned>","files":["<key files>"]}'
```

### How to categorize: by FUNCTION, not TOPIC

The **table** is the record's *kind* — the question it answers / when you reach
for it: `decisions` (a choice + why), `errors` (a failure + fix), `rules`
(constraints to obey), `context` (durable how-it-works knowledge). Pick the
table by function.

**Never encode TOPIC as a category.** "The auth feature", a module or file
name, an area — those are *topics*, already found by recall's semantic search +
auto-extracted entities. Putting them in a table name or `type` just duplicates
the embedding. Let entities capture them, or pass `--scope`.

For `context` records, an optional `type` facet scopes later retrieval —
recommended values: `architecture`, `gotcha`, `howto`, `reference`. Filter on it
with `axil recall "<q>" --type architecture`. The vocabulary is a recommendation,
not enforced (any string is accepted). `decisions`/`errors` don't need a `type` —
their field shape already encodes their function.

### Rules

1. **Store immediately, not in batches.** Don't wait until the end of a session to dump 6 records. Store each decision/error/summary right after it happens.
2. **Store before responding to the user.** When you finish a task, store the summary THEN tell the user it's done.
3. **If you're about to say "done" or "all tasks complete", you MUST store first.** No exceptions.
4. **When you need context you don't have**, recall it:
   ```bash
   axil recall "<what you need>" --top-k 5
   ```

### What to store vs. not store

| Store | Skip |
|-------|------|
| Design decisions with rationale | File contents (git has those) |
| Gotchas and non-obvious behavior | Trivial one-line fixes |
| Architecture: how modules connect | Info already in CLAUDE.md |
| Build/test errors and their fixes | Temporary debug output |
| Facts tied to a file or symbol (`--code-ref <path:line>`) | Raw code snippets |
| Implementation plans for multi-step work | The user's preferences and feedback on how you work (auto-memory's job) |

### Axil and Claude Code auto-memory

Claude Code keeps its own auto-memory: notes indexed by a `MEMORY.md` that is loaded into every conversation. The two hold different things, so nothing is written twice:

- **Axil owns the project:** architecture, gotchas, facts anchored to a file or symbol, errors with their root cause and fix, decisions with their reason, and the code graph (`axil code-search`, SCIP edges).
- **Auto-memory owns the user:** how they like to work, their corrections of your behavior, standing preferences. Leave those notes there and don't mirror them into Axil; don't copy Axil records into auto-memory either.

A correction about the code ("that module is deprecated", "this test is flaky because…") is project knowledge: store it in Axil.
