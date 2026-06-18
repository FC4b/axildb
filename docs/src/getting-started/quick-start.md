# Quick Start

## Create a database

```bash
# Create the database file
axil init ./memory.axil

# Store a first record
axil --db ./memory.axil store notes '{"text": "Hello, Axil!"}'
```

This creates `memory.axil` (core storage) and companion files as needed.

## Store and retrieve

```bash
# Store records
axil --db ./memory.axil store sessions '{"summary": "Fixed auth bug", "project": "my-app"}'
axil --db ./memory.axil store decisions '{"choice": "Use JWT", "reason": "Simpler than OAuth"}'

# List all records in a table
axil --db ./memory.axil list sessions

# Get a specific record by ID
axil --db ./memory.axil get <record-id>
```

## Semantic search

```bash
# Embed a record's text field for vector search
axil --db ./memory.axil embed <record-id> summary

# Find similar records
axil --db ./memory.axil recall "authentication issues" --top-k 5
```

## Knowledge graph

```bash
# Create relationships between records
axil --db ./memory.axil link <from-id> modified <to-id>

# Find connected records
axil --db ./memory.axil neighbors <record-id>

# Traverse the graph
axil --db ./memory.axil traverse <start-id> "->modified->file"
```

## Full-text search

```bash
# Search
axil --db ./memory.axil fts "timeout error"
```

## Agent memory (with `--features memory`)

```bash
# Store facts about entities
axil --db ./memory.axil know auth-module "Uses JWT tokens with 1h expiry"
axil --db ./memory.axil know auth-module "Supports refresh token rotation"

# Query what you know
axil --db ./memory.axil know-about auth-module

# Session lifecycle
axil --db ./memory.axil session start
axil --db ./memory.axil session log <session-id> context '{"tool": "grep", "result": "found"}'
axil --db ./memory.axil session end <session-id> --summary "Fixed the auth timeout"
```

## Diagnostics

```bash
# Health check
axil --db ./memory.axil doctor

# Database statistics
axil --db ./memory.axil stats

# Run maintenance
axil --db ./memory.axil worker run
```

## AxilQL

```bash
# Interactive query console
axil --db ./memory.axil ql -i

# One-shot queries
axil --db ./memory.axil ql 'RECALL "auth error" TOP 5'
axil --db ./memory.axil ql 'FIND "timeout" WHERE table = "sessions"'
axil --db ./memory.axil ql 'COUNT FROM sessions'
```
