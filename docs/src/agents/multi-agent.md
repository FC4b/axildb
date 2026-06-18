# Multi-Agent Memory

Axil supports multiple agents sharing a single database with isolated sessions.

## Agent tagging

Tag records with an agent name using `--agent`:

```bash
axil --db ./db store sessions '{"summary": "fixed bug"}' --agent alice
```

## Per-agent sessions

Each agent gets its own session history:

```rust
let mem = AgentMemory::for_agent(&db, "alice");
let session = mem.working().start_session(None)?;
// Alice's session — isolated from other agents

let bob_mem = AgentMemory::for_agent(&db, "bob");
let bob_sessions = bob_mem.working().list_sessions(false)?;
// Bob only sees his own sessions
```

## Shared vs isolated memory

| Memory Type | Scope | Rationale |
|-------------|-------|-----------|
| Working | Per-agent | Each agent has its own active context |
| Episodic | Per-agent | Session history is agent-specific |
| Semantic | Shared | Knowledge graph benefits all agents |
| Procedural | Shared | Learned patterns apply universally |
| Preference | Shared | User preferences are global |

## Querying across agents

Unscoped `AgentMemory` sees all sessions:

```rust
let mem = AgentMemory::new(&db);
let all_sessions = mem.working().list_sessions(false)?;
// Returns sessions from all agents
```

## Best practices

1. Use consistent agent names across sessions
2. Store shared knowledge via semantic memory (not working memory)
3. Use `axil recall` without `--agent` for cross-agent search
4. Tag important decisions with the agent name for traceability
