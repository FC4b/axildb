---
name: generic-agent
description: "Portable agent memory instructions — works with any AI agent that can run shell commands"
---

# Axil Agent Memory — Portable Instructions

These instructions work with any AI agent that can execute shell commands (Claude Code, Codex, Copilot CLI, Cursor, OpenCode, Qwen Code, Droid, Aider, etc.).

## Setup

```bash
# Install Axil (if not already available)
cargo install axildb --features full

# Or build from source
cd /path/to/axildb && cargo build --release -p axildb --features full
export PATH="$PATH:/path/to/axildb/target/release"

# Initialize a memory database in your project
axil init ./memory.axil
export AXIL_DB="./memory.axil"
```

## Core Operations

All commands output JSON. Parse with `jq` or your language's JSON parser.

### Store a Memory

```bash
axil store <table> '<json_data>'

# Examples:
axil store decisions '{"summary":"Use PostgreSQL for user data","reason":"Need ACID guarantees"}'
axil store errors '{"error":"Connection timeout","component":"auth","fix":"Added retry logic"}'
axil store patterns '{"pattern":"Repository pattern for data access","files":["src/repo.rs"]}'
```

### Recall Memories (Semantic Search)

```bash
# Recency-weighted semantic search (recommended)
axil recall "<query>" --top-k 5

# Examples:
axil recall "authentication issues" --top-k 3
axil recall "database connection patterns" --top-k 5 --alpha 0.5
```

### Full-Text Search

```bash
axil fts "<query>" --limit 10
```

### Link Related Memories

```bash
axil link <from_id> "<relationship>" <to_id>

# Examples:
axil link $FIX_ID "fixes" $ERROR_ID
axil link $NEW_DECISION "supersedes" $OLD_DECISION
```

### Session Tracking

```bash
# Start session
SESSION=$(axil session start --meta '{"project":"my-app"}' | jq -r '.session_id')

# Log to session
axil session log $SESSION decisions '{"summary":"..."}'

# End session
axil session end $SESSION --summary "Completed auth refactor"
```

### Time Queries

```bash
axil since 7d                    # Last 7 days
axil since 1h --table errors     # Recent errors
axil timeline --limit 10         # Latest records
axil activity --days 7           # Daily counts
```

## Recommended Tables

| Table | Store Here |
|-------|-----------|
| `decisions` | Architecture choices, tradeoffs made |
| `errors` | Bugs encountered, stack traces |
| `fixes` | How bugs were resolved |
| `patterns` | Code patterns, conventions discovered |
| `context` | Project context, environment notes |

## Agent Behavior Guidelines

1. **At session start**: run `axil recall` for the current topic
2. **After key decisions**: store with rationale
3. **After fixing bugs**: store error + fix, link them
4. **At session end**: summarize what was done
5. **When decisions change**: store new decision, link as "supersedes" old one

## Generating Problem Reports

If Axil itself has issues (slow, errors, crashes), generate a report:

```bash
axil report generate
# Outputs to .axil-reports/report-<date>.json
```

Then manually add observed problems to the report's `problems` array.

## Configuration

Create `axil.toml` in your project root:

```bash
axil config init
```

Key settings:
```toml
[database]
path = "./memory.axil"

[debug]
slow_query_threshold_ms = 100
log_level = "warn"
```
