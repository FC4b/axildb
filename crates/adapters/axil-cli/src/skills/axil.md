---
name: axil
description: "AI agent memory — store, recall, link, and search context using the Axil database"
trigger: when user mentions memory, recall, remember, store context, agent memory, session tracking
---

# Axil — AI Agent Memory

You have access to **Axil**, a local embedded database for agent memory. Use it to persist decisions, patterns, errors, and context across sessions. All commands output JSON.

## Quick Reference

```bash
# Database location (set once per project)
export AXIL_DB="./memory.axil"

# Or use --db flag on every command
axil --db ./memory.axil <command>
```

## When to Use Axil

### Auto-Recall (do this automatically)
- **Session start**: recall recent context for the current project
- **Topic switch**: recall past work related to the new topic
- **Before decisions**: check if similar decisions were made before
- **Error investigation**: recall past errors and fixes

### Auto-Store (do this automatically)
- **Key decisions**: architecture choices, tradeoffs, rationale
- **Bug fixes**: what broke, why, how it was fixed
- **Patterns discovered**: recurring code patterns, anti-patterns found
- **Error resolutions**: error → root cause → fix mapping
- **Context summaries**: end-of-session summaries

### Auto-Link (do this automatically)
- **Files to sessions**: link modified files to the current session
- **Decisions to context**: link a decision to the evidence that informed it
- **Bugs to fixes**: link error records to fix records
- **Superseding**: when a decision changes, store new record + link to old

### Categorize by FUNCTION, not TOPIC
- The **table** is the record's *kind* — the question it answers: `decisions`
  (choice + why), `errors` (failure + fix), `rules` (constraints), `context`
  (how-it-works knowledge). Pick the table by function.
- **Topic ≠ category.** "The auth feature", a module/file name, an area — those
  are topics, already found by semantic recall + auto-extracted entities. Don't
  encode them as a table name or `type`; let entities capture them, or use `--scope`.
- `context` records may carry an optional `type` facet (recommended:
  `architecture`, `gotcha`, `howto`, `reference`) to scope retrieval — filter with
  `axil recall "<q>" --type architecture`. Recommended, not enforced.

### What NOT to Store
- File contents already in git (use git, not Axil)
- Ephemeral chatter or trivial questions
- Information that will be stale in minutes
- Exact code snippets (store descriptions instead)

## Session Lifecycle

### Starting Work

```bash
# Start a session (returns session_id)
axil session start --meta '{"project":"my-app","goal":"fix auth bug"}'

# Recall recent relevant context
axil recall "auth timeout bug" --top-k 5

# Check recent activity
axil since 7d --table decisions
```

### During Work

```bash
# Store a decision
axil store decisions '{"summary":"Switch from JWT to session cookies","reason":"JWT refresh was causing timeout","impact":"auth,security"}'

# Store with auto-embedding for semantic search
axil store decisions '{"summary":"..."}' --embed summary

# Log to current session (auto-links via graph)
axil session log <SESSION_ID> decisions '{"summary":"..."}'

# Link related records
axil link <DECISION_ID> "informed_by" <ERROR_ID>
axil link <FIX_ID> "supersedes" <OLD_DECISION_ID>

# Search for related past work
axil recall "authentication timeout" --top-k 3
axil fts "JWT refresh token"
```

### Ending Work

```bash
# End session with summary
axil session end <SESSION_ID> --summary "Fixed auth timeout by switching to session cookies. Updated middleware and tests."
```

## Command Reference

### Database Management

| Command | Description |
|---------|-------------|
| `axil init <path>` | Create a new database |
| `axil info` | Show database stats |
| `axil tables` | List tables with counts |
| `axil config init` | Create `axil.toml` config |
| `axil config show` | Show resolved config |

### Storing Data

```bash
# Basic store
axil store <table> '<json>'

# Store with auto-embed
axil store <table> '<json>' --embed <field1>,<field2>

# Store from stdin (for large payloads)
echo '{"summary":"..."}' | axil store decisions -

# Update existing record
axil update <id> '{"summary":"updated"}'
```

### Searching & Recall

```bash
# Semantic search with recency weighting (primary command)
axil recall "<query>" --top-k 5

# Adjust recency weight (0=pure recency, 1=pure similarity)
axil recall "<query>" --top-k 5 --alpha 0.5

# Time-bounded recall
axil recall "<query>" --after 2026-03-01 --before 2026-04-01

# Pure vector similarity (no recency weighting)
axil search "<query>" --top-k 10

# Full-text search
axil fts "<query>" --limit 10
```

### Graph Operations

```bash
# Link two records
axil link <from_id> "<edge_type>" <to_id>
axil link <from_id> "depends_on" <to_id> --props '{"weight":0.8}'

# Find neighbors
axil neighbors <id> --type "depends_on" --direction out

# Traverse graph paths
axil traverse <id> "->depends_on->->uses->"

# List edges
axil edges <id> --direction both
```

### Session Management

```bash
axil session start --meta '{"project":"..."}'
axil session log <sid> <table> '<json>'
axil session end <sid> --summary "..."
axil session list --active
axil session history <sid>
```

### Time-Series Queries

```bash
axil since 3d                    # Records from last 3 days
axil since 1h --table errors     # Errors from last hour
axil timeline --limit 20         # Recent records
axil diff --since 1d             # What changed today
axil activity --days 7           # Daily counts
```

### Querying

```bash
# Filtered query
axil query <table> --where "status=active" --order-by created_at --direction desc --limit 10

# List all records in a table
axil list <table> --limit 50
```

## Output Parsing (jq)

All commands output JSON. Common jq patterns:

```bash
# Get just the IDs from a recall
axil recall "auth" --top-k 5 | jq '.[].id'

# Get summaries with scores
axil recall "auth" --top-k 5 | jq '.[] | {summary: .data.summary, score}'

# Count records per table
axil tables | jq '.[] | "\(.name): \(.count)"'

# Get the ID from a store operation
axil store decisions '{"summary":"..."}' | jq -r '.id'
```

## Suggested Table Names

| Table | Use For |
|-------|---------|
| `decisions` | Architecture choices, tradeoffs |
| `errors` | Bugs found, stack traces, root causes |
| `fixes` | How bugs were resolved |
| `patterns` | Recurring code patterns |
| `context` | Project context, environment notes |
| `todos` | Deferred work items |

## Memory Patterns

### Superseding (when a decision changes)
```bash
NEW_ID=$(axil store decisions '{"summary":"Use Redis instead of memcached","reason":"Need pub/sub"}' | jq -r '.id')
axil link $NEW_ID "supersedes" $OLD_DECISION_ID
```

### Error → Fix chain
```bash
ERR_ID=$(axil store errors '{"error":"TimeoutError in auth middleware","stacktrace":"..."}' | jq -r '.id')
# ... fix the bug ...
FIX_ID=$(axil store fixes '{"summary":"Added retry logic to auth middleware","files":["auth.rs"]}' | jq -r '.id')
axil link $FIX_ID "fixes" $ERR_ID
```

### Context-aware recall
```bash
# Start of session: recall project-specific context
axil recall "project setup and conventions" --top-k 3
axil query context --where "project=my-app" --limit 5
```
