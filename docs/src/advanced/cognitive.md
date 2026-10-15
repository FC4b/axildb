# Cognitive Memory

Axil's cognitive memory features make the database behave more like a brain than a storage engine.

## Auto-importance scoring

Every record gets an importance score on insert based on:
- Entity density (more entities = more important)
- Structural markers (decisions, errors, architecture)
- Text complexity

```bash
axil --db ./db store context '{"summary": "Critical: auth bypass in production"}'
# Auto-scored with high importance due to "Critical" marker
```

## Memory decay

Records lose effective importance over time using exponential decay:

```
effective_importance = base_importance × 2^(-age_days / half_life)
```

Default half-life: 90 days. Access reinforces importance.

```bash
axil --db ./db memory-pressure           # Show tier distribution
axil --db ./db memory-pressure --archive # Auto-archive cold records
```

## Belief system

High-importance facts auto-generate beliefs — the agent's high-level understanding:

```bash
axil --db ./db believe "Auth module uses JWT with 1h expiry"
axil --db ./db beliefs
axil --db ./db doubt <belief-id>
```

## Context-aware push

`axil boot` proactively surfaces relevant memories based on context:

```bash
axil boot --files src/auth.rs      # Memories related to auth files
axil boot --entities auth-module   # Memories about auth-module
axil boot --error "timeout"        # Memories about timeout errors
```

## Auto-capture

`axil auto-capture` reads a piece of text, decides whether it holds a record
worth keeping, and stores it in the matching table. It uses patterns, not a
model:

```bash
cargo test 2>&1 | axil auto-capture - --source bash --min-confidence 0.8
axil auto-capture "Root cause: the hook held the writer lock past its timeout." --dry-run
```

What it looks for depends on `--source`:

- **`bash`** is command output. The brain hook sends the output of every
  failed shell command this way, at `--min-confidence 0.8`. Only failures
  are read from it: compiler errors, panics, tracebacks, failing tests (even
  next to suites that passed). Some failures are skipped because nothing in
  the project failed: zsh or bash complaining about the command line
  (`=== not found`, a glob that matched nothing), a regex rg could not
  parse, git refusing a command as typed, a usage error from clap or cargo,
  and source or diff lines the command printed.
- **Any other source** (the default is `auto`) is text the agent wrote. It
  can also yield:
  - **errors**: a reported failure ("failed because", "Root cause: ..."),
    or a `fix:` commit subject that names its defect;
  - **rules**: a `CONVENTIONS:` list, or a sentence opening with "Never",
    "Always", "Do not" or "Don't", or "every X must";
  - **context**: a labeled finding (`Gotcha:`, `Lesson learned:`, `TIL:`),
    or "it turns out ...";
  - **decisions**: `Decision:`, "we settled on" or "moved from" pass the
    default `--min-confidence 0.7`; "decided to", "switching to" and
    similar phrases score 0.6 and are only shown.

Narration ("Let me find the root cause") and conditions ("if it turns out
...") are not captured.

A captured record is stored with an importance that depends on its kind:
0.3 for a failure printed by a command; 0.8 for a rule, or for a failure
that loses data, goes silent, panics or is broken; 0.5 for anything else.

## Pattern detection

The worker detects recurring patterns:

```bash
axil --db ./db worker run
axil --db ./db patterns          # List detected patterns
axil --db ./db patterns --active # Only active (non-dismissed) patterns
```
