---
name: axil-retro
description: "Monthly or weekly retrospective — review patterns, revise beliefs, and update summaries from stored memory"
trigger: when user says "run a retro", "/axil-retro", "monthly review", or asks for a retrospective
---

# Axil Retro — Periodic Retrospective

Synthesize a longer-horizon review from Axil memory. Unlike `/axil-learn` (per-session), this is weekly or monthly and produces a durable summary record.

## When to Trigger

- User explicitly invokes `/axil-retro`
- Natural break point (end of sprint, end of month, end of project phase)
- User asks "how's the project going?" or "what patterns are you seeing?"

## Procedure

1. **Scope the window.** Default is the last 30 days. Honor the user's range if specified.

   ```bash
   axil retro --window 30d --format markdown > .axil/reports/retro-$(date +%Y-%m).md
   # or interactively:
   axil retro --window 30d --interactive
   ```

2. **Answer the four questions.** The retro command produces a draft. Review and fill gaps:
   - **Goals:** What were we trying to do?
   - **Execution:** What actually happened? Which decisions held up; which didn't?
   - **Patterns:** Recurring errors, recurring asks, recurring friction.
   - **Changes:** What should we do differently next period?

3. **Store the retrospective.**

   ```bash
   axil store context '{"type":"retrospective","window":"30d","summary":"<narrative>","lessons":["<...>"],"changes":["<...>"]}'
   ```

4. **Optional: revise beliefs.** If the retro surfaces evidence that contradicts a stored belief, run:

   ```bash
   axil doubt <belief-id> --reason "<why it no longer holds>"
   ```

## Output Location

Retros land in `.axil/reports/retro-YYYY-MM.md` by default. Keep these checked into git if the project is private; gitignore them if sensitive.

## Why This Matters

Per-session learning captures facts. Retros find *patterns* — and the belief system (`axil beliefs`) is where patterns graduate into durable agent knowledge.
