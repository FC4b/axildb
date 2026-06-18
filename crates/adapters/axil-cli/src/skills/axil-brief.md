---
name: axil-brief
description: "Daily brief — morning summary of recent memory, open threads, and what to work on next"
trigger: when user says "good morning", "/axil-brief", "what's the state", or starts a new session after an absence
---

# Axil Brief — Daily Briefing

Produce a two-minute read summarizing where the project stands right now. Designed to replace manual meeting-prep / standup-prep.

## When to Trigger

- Start of a new session after >8 hours absence
- User explicitly invokes `/axil-brief`
- User asks "what's the state?" / "where did we leave off?" / "what should I work on?"
- Scheduled (via `axil schedule install daily-brief`) — runs once per morning

## Procedure

```bash
axil brief --window 24h --format markdown
```

The command synthesizes:

1. **Since yesterday** — sessions logged, decisions made, errors hit
2. **Open threads** — in-progress tasks, un-fixed errors, decisions without follow-through
3. **Patterns surfacing** — recurring issues from the last 7 days
4. **Top-of-mind** — highest-importance records recently accessed

## Custom Windows

```bash
axil brief --window 3d       # over a long weekend
axil brief --window 7d       # weekly brief
axil brief --after 2026-04-10 # since a specific date
```

## Output Format

- `--format markdown` (default, human-readable)
- `--format json` (for downstream agents)
- `--budget 500` (token cap for inline hook use)

## Why This Matters

The goal is that every work session starts with *context*, not with "what was I doing?". A good brief answers that in 30 seconds.
