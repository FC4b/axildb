# Claude Code Integration

Axil is designed as the memory backend for Claude Code agents.

## Installation

```bash
cd your-project
axil install
```

Run bare on a terminal, this opens an interactive wizard: it detects the
agent tooling already in the project (`.claude/`, `.cursor/`, …),
pre-selects those integrations, and offers bootstrap (index the codebase
now) and repo-local skills as toggles. In scripts/CI — or whenever any
flag is passed — there is no prompt; use flags explicitly:

```bash
axil install --claude-code --bootstrap
```

This creates:
- `.axil/memory.axil` — the database
- Hook wiring in `.claude/settings.json` — every lifecycle event runs
  `axil hook run --dialect claude` (the brain lives in the binary; no
  bash/jq needed, works natively on Windows)
- `AGENTS.md` managed block — the cross-tool memory contract (also read
  by Codex, OpenCode, Qwen Code, Droid, …); skip with `--no-agents-md`

## How it works

1. **Boot**: `axil boot` runs automatically when the session starts (and again after a compaction), injecting recent decisions, errors, and session history
2. **Auto-capture**: Hooks detect file changes and store them automatically
3. **Manual store**: The agent stores decisions, errors, and summaries via `axil store`
4. **Recall**: `axil recall` retrieves relevant context using vector + graph + recency scoring

## Compaction

Claude Code sends `SessionStart` again after a compaction (source `compact`,
auto or manual) and after `/clear`. Axil registers `SessionStart` with no
matcher, so every source reaches the hook, which re-injects `axil boot`,
including the latest checkpoint's "Resume Here" block, as
`additionalContext`. `PostCompact` is not registered: Claude Code documents
no `additionalContext` for it, so it has no way to put the boot in front of
the model.

## Coexisting with auto-memory

Claude Code keeps its own auto-memory: notes indexed by a `MEMORY.md` that is
loaded into every conversation. The installed `CLAUDE.md` section and the
skills split the work so nothing is written twice:

| Axil | Claude Code auto-memory |
|------|------------------------|
| Architecture: how modules connect | How the user likes to work |
| Gotchas and facts tied to a file or symbol (`--code-ref`) | Corrections of the agent's behavior |
| Errors with root cause and fix; decisions with their reason | Standing personal preferences |
| The code graph (`code-search`, SCIP edges) | |

Preference and feedback notes stay in auto-memory and are not mirrored into
Axil. `axil install` writes one note there itself
(`feedback_axil_proactive.md`, indexed from `MEMORY.md`): a pointer that says
to use Axil first and store as you go, not a copy of any Axil record.

## Agent workflow

```
Session starts → axil boot (auto via hook)
Working...     → axil store decisions/errors/context
Need context?  → axil recall "topic" --top-k 5
Session ends   → session record + maintenance (auto via hook)
```

## Skill integration

Axil provides Claude Code skills for structured workflows:

```
/axil-store "summary of what happened"
/axil-report  # Generate a field report
```

## Configuration

Axil auto-detects the database at `.axil/memory.axil`. Override with:

```bash
export AXIL_DB="/path/to/memory.axil"
```
