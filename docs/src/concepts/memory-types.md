# Memory Types

Axil provides five distinct memory types purpose-built for AI agents:

## Working Memory

Current session context — active tasks, tool outputs, open files. Auto-cleared when a session ends.

- **Table**: `_working`
- **Recency weight**: 0.3 (heavily weights recency)
- **TTL**: None (cleared on session end)

## Semantic Memory

Facts, entities, and relationships — the agent's knowledge graph.

- **Table**: `_entities`
- **Recency weight**: 0.8 (heavily weights relevance)
- **TTL**: None (facts persist)
- **Features**: Entity extraction, aliases, consolidation, auto-linking

```bash
axil --db ./db know auth-module "Uses JWT tokens with 1h expiry"
axil --db ./db know-about auth-module
```

## Episodic Memory

Past sessions and interactions with outcomes. Created automatically when sessions end.

- **Table**: `_episodes`
- **Recency weight**: 0.5 (balanced)
- **TTL**: 90 days (configurable)

## Procedural Memory

Learned patterns, strategies, and tool usage sequences.

- **Table**: `_procedures`
- **Recency weight**: 0.7 (relevance matters more)
- **TTL**: None (patterns persist)

## Preference Memory

User preferences, feedback, rules, and conventions.

- **Table**: `_preferences`
- **Recency weight**: 0.9 (almost pure relevance)
- **TTL**: None (rules persist)

## Cross-Memory Recall

The `remember()` function searches across all memory types simultaneously:

```bash
axil --db ./db recall "authentication" --top-k 10
```

Results are ranked using Reciprocal Rank Fusion (RRF), combining vector similarity, graph connectivity, recency, and keyword matches.

## Multi-Agent Memory Model

When using per-agent sessions:

| Memory Type | Scope |
|-------------|-------|
| Working | Per-agent (isolated) |
| Episodic | Per-agent (isolated) |
| Semantic | Shared (all agents) |
| Procedural | Shared (all agents) |
| Preference | Shared (all agents) |
