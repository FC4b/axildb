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
- Hook scripts for automatic memory capture
- Claude Code settings integration

## How it works

1. **Boot**: `axil boot` runs automatically on first tool call, injecting recent decisions, errors, and session history
2. **Auto-capture**: Hooks detect file changes and store them automatically
3. **Manual store**: The agent stores decisions, errors, and summaries via `axil store`
4. **Recall**: `axil recall` retrieves relevant context using vector + graph + recency scoring

## Agent workflow

```
Session starts → axil boot (auto via hook)
Working...     → axil store decisions/errors/context
Need context?  → axil recall "topic" --top-k 5
Session ends   → axil store context (session summary)
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
