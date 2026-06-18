---
name: axil-learn
description: "End-of-session learning loop — extract decisions, gotchas, and corrections from recent work and store them in Axil"
trigger: when user says "learn from this session", "capture what we did", "/axil-learn", or session is ending with un-stored work
---

# Axil Learn — Session Reflection

Use this skill at the end of a unit of work (feature complete, bug fixed, decision made, or session wrapping up). It captures what's worth remembering so future sessions boot with that knowledge.

## When to Trigger

- After completing a task or feature
- After the user says "we're done" / "that's it for today"
- When about to respond "all done" — stop and run this first
- When you notice un-stored decisions / gotchas / corrections accumulating

## Procedure

1. **Review the session.** Skim the recent conversation for:
   - Design decisions (chose A over B, with rationale)
   - Errors hit and how they were fixed
   - User corrections ("no, do it this way")
   - Non-obvious gotchas discovered

2. **Store each atomic learning** in the right table — one record per discrete item:

```bash
# Design decision
axil store decisions '{"summary":"<what was chosen>","reason":"<why>","files":["<affected files>"]}'

# Error + fix
axil store errors '{"error":"<what broke>","root_cause":"<why>","fix":"<how resolved>"}'

# Architecture / code-behavior learning
axil store context '{"type":"architecture","summary":"<what you learned>","files":["<key files>"]}'
```

3. **Then write the structured checkpoint** — the *resume contract* with the next session. Point at the records you just stored instead of restating them:

```bash
axil checkpoint '{
  "goal":          "<the north-star intent that drove this work>",
  "state":         "<1–2 sentences on where things stand>",
  "next_steps":    ["<ordered, actionable>"],
  "open_questions":["<unresolved blockers or uncertainty>"],
  "references": [
    {"kind":"record","ref":"<decision-id you just stored>","note":"why it matters"},
    {"kind":"file",  "ref":"<key path>"}
  ]
}'
```

`axil boot` automatically replays the latest checkpoint as "## Resume Here" at the top of the next session's context — `references[]` of `kind:"record"` resolve live to the current row, so superseded decisions show their current state.

> This checkpoint **replaces** the old `axil store context '{"type":"session_summary",…}'` pattern. The session_summary blob landed in the `context` table as flat prose you couldn't resume from; the structured checkpoint is what `boot` actively promotes. See the dedicated `axil-checkpoint` skill for the full template.

4. **One record per discrete learning.** Don't batch 6 decisions into one summary — they become unrecallable. The checkpoint is the one place where you summarise; everywhere else, write atomic records.

5. **Confirm with the user** before marking work complete. Never say "done" before storing.

6. **Optional — distill recurring failures into corrective rules.** When the same kind of error has bitten more than once (and you've recorded its fix), run the write-back loop so the next session is *warned before* it repeats the mistake:

```bash
axil rule distill --dry-run   # preview; drop --dry-run to apply
```

This reads the `errors` table, groups near-identical failures, and for any seen ≥2× with a recorded fix writes a *"Last N times you hit X, the fix was Y"* directive into a managed `<!-- axil:learned:… -->` block in CLAUDE.md (idempotent, never clobbers human content) and pins it so `axil boot` echoes it. It's the counterpart to `axil rule extract` (which reads rules *from* CLAUDE.md) — `rule distill` writes hard-won corrections *back out*. Distinct from `axil learn <name> <desc>`, which stores a procedural pattern.

## What NOT to Store

- File contents (git already has those)
- Trivial one-line fixes
- Info already in CLAUDE.md
- Raw debug output
- Anything the checkpoint `references[]` can point to instead

## Why This Matters

Future sessions call `axil boot` at startup. Without atomic records, the searchable memory layer is empty. Without the checkpoint, every new session does the same discovery loop ("what was I doing?"). The pair — atomic stores + a structured checkpoint — is what makes resume feel instant instead of cold-start.
