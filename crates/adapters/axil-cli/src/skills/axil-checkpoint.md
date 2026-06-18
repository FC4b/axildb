---
name: axil-checkpoint
description: "Write a structured checkpoint so the next agent (or your next session) resumes instead of re-discovering. Replaces the free-text session_summary pattern."
trigger: when user says "wrap up", "checkpoint", "/axil-checkpoint", "we're switching sessions", or context is getting full and you want to snapshot a resume point
---

# Axil Checkpoint — Structured Resume State

Use this skill when work needs to carry across to another session — yours or another agent's. `axil checkpoint` writes a structured record that the *next* `axil boot` automatically surfaces as the "Resume Here" block at the top of context.

This is the Axil-native form of the conversational `/handoff` pattern. Same principles (compact, reference don't duplicate, redact, tailor to focus), but the delivery is persistent and queryable instead of a one-shot temp file.

## When to Trigger

- **Context is getting full** — snapshot before compaction so nothing is lost.
- **Mid-session breakpoint** — pausing work, want a clean resume point.
- **Session end** — pair with `axil store context` for the wrap-up summary; add `--final` to mark this as the closing checkpoint.
- **Handing off to a different agent** — Claude → Codex / Cursor / etc.
- The user says "/axil-checkpoint", "wrap up", "let's checkpoint", or similar.

## The Five Fields

Write the smallest checkpoint that lets the next session resume:

| Field | What goes here | Cost of skipping |
|-------|----------------|------------------|
| `goal` | The user's north-star intent — *why* this work matters | Next session re-asks "what are we trying to do?" |
| `state` | 1–2 sentences on where things stand right now | Next session re-discovers progress |
| `next_steps[]` | Ordered, actionable — what you'd do next | **The single highest-value field.** Without it, every resume is a re-plan. |
| `open_questions[]` | Unresolved blockers, uncertainty | Hidden landmines become surprises |
| `references[]` | Typed pointers (not copies) — `{kind, ref, note?}` | Verbose checkpoint + stale snapshots |

`summary` is optional and used as the rendered headline.

## How to Run

```bash
# Common case — JSON via positional arg
axil checkpoint '{
  "goal": "ship axil-checkpoint Tier-2 extension",
  "state": "scaffold landed, tests green, boot wiring in flight",
  "next_steps": [
    "wire CLAUDE.md guidance for new store pattern",
    "run full workspace test + commit"
  ],
  "open_questions": [
    "should /axil-learn skill be updated in the same commit?"
  ],
  "references": [
    {"kind": "file", "ref": "crates/extensions/axil-checkpoint/src/lib.rs", "note": "crate root"},
    {"kind": "record", "ref": "01KSENA7S0QPRGM94RCQADGSQH", "note": "in-flight decision"}
  ]
}'

# Or via stdin for multi-line shells
cat <<'EOF' | axil checkpoint -
{ "goal": "…", "next_steps": ["…"] }
EOF

# Read it back (uses stored checkpoint, else derives from latest session)
axil checkpoint show

# Final checkpoint for the session (does NOT end the session by itself)
axil checkpoint --final '{"summary":"phase complete","next_steps":["open PR"]}'
```

## Reference, Don't Duplicate

`references[]` accepts any `kind`. The boot replay treats `record` specially — it resolves to the *current* row by id, so a superseded decision shows its live state, not a stale copy.

| Reference target | Use |
|------------------|-----|
| A decision/error/context already in Axil | `{"kind":"record","ref":"<record-id>"}` — boot resolves to live state |
| A file you touched | `{"kind":"file","ref":"<path>"}` |
| A commit / PR / plan doc | `{"kind":"commit","ref":"<sha>"}` / `pr` / `plan` |
| An external issue or ticket | `{"kind":"issue","ref":"<url>"}` |

Don't restate things already stored as `decisions` / `errors` / `context` — point at them. The checkpoint stays small; boot rehydrates them live.

## Redact

Strip API keys, passwords, tokens, and PII before writing. The checkpoint lands in `_checkpoint_records` and surfaces at the top of every subsequent boot — assume it will be read by future you and possibly another agent.

## Tailor to Focus

If the user named what the next session should focus on, bias `next_steps` toward that. The checkpoint is a contract with the *next* turn, not a journal of the current one.

## Snapshot vs Final

- Default: `kind: "snapshot"` — the owning session keeps running. Use freely; multiple snapshots replace nothing, they accumulate. Boot uses the most recent.
- `--final`: stamps `kind: "final"` — signals the session is closing. Does not actually end the session (use the existing session-lifecycle tooling for that).

## What This Replaces

If you used to write `axil store context '{"type":"session_summary",…}'` at session end — stop. Use `axil checkpoint` instead. The session_summary pattern wrote free text into the `context` table; boot surfaced it as a flat bullet you couldn't resume from. The structured checkpoint is the upgrade path and `axil boot` actively promotes it.

The other store patterns (`axil store decisions`, `axil store errors`, `axil store context type:architecture`) are unchanged — checkpoints reference those records via `references[]`.

## Why This Matters

Without a checkpoint, every new session does the same discovery loop: read recent commits, grep for context, ask "what was I doing?". With one stored, `axil boot` opens the next session with a "Resume Here" block carrying goal, state, next_steps, open_questions, and live references — and the agent picks up where the last one stopped.
