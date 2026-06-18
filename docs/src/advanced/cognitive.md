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

Axil detects errors and decisions in text and stores them automatically:

```bash
axil --db ./db store context '{"summary": "Error: connection pool exhausted after 50 concurrent requests"}'
# Auto-detected as an error, stored with appropriate metadata
```

## Pattern detection

The worker detects recurring patterns:

```bash
axil --db ./db worker run
axil --db ./db patterns          # List detected patterns
axil --db ./db patterns --active # Only active (non-dismissed) patterns
```
