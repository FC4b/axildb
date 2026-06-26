# Memory Taxonomy

How is a stored memory categorized, and how does that categorization help an
agent retrieve the right thing fast? This page covers the categorization model
for the agent-facing `axil store` / `axil recall` path.

> This is distinct from [Memory Types](./memory-types.md), which documents the
> formal five-way `MemoryType` enum used by the Rust structured-memory API
> (`know`/`learn`/session lifecycle). The model below is what the CLI and agent
> hooks actually write through.

## The guiding principle: categorize by *function*, not *topic*

A category can answer one of two questions, and only one of them is worth
spending a category on:

- **"What is this *about*?"** → `auth`, the login flow, the vector crate. This is
  **topic** — and it is already solved by vector embeddings, auto-extracted
  [entities](./architecture.md), and `_scope`. A category that encodes topic just
  duplicates the embedding; you would never filter on it, you would search.
- **"What *role* does this play / when do I reach for it?"** → a decision, a
  bug-fix, a rule to obey. This is **function** — and it is the one thing
  semantic search *cannot* express as a filter.

So: **categorize by function.** Topic belongs in entities + scope (both
automatic). This is why putting a feature/module/area name into a table or a
`type` is an anti-pattern.

## The four axes

| Axis | What it is | How it's set | Who reads it |
|------|-----------|--------------|--------------|
| **table** (kind) | the record's function: `decisions`, `errors`, `rules`, `context`, … | you choose it at `store` time | boot section routing, recall, scoring |
| **`type`** (facet) | an optional sub-facet inside a kind (mainly `context`) | a `type` field in the JSON payload | the `--type` recall filter |
| **scope** | `session` / `agent` / `project` / `user` / `global` | `_scope` field / `--scope` | recall scoping |
| **cognitive** | `_importance`, decay, tiers, `_entities` | computed automatically on every non-`_` table | recall ranking, archival, boot order |

The **table** is canonical: it is what boot groups by and what scoring sees.
Choose it by function — `decisions` (a choice + its rationale), `errors` (a
failure + its fix), `rules` (constraints to obey), `context` (durable
how-it-works knowledge).

## The `type` facet

`context` is the one schemaless, grab-bag kind, so it earns an optional `type`
sub-facet to scope retrieval. Recommended (but **not enforced** — any string is
accepted) values:

| `type` | use it for |
|--------|-----------|
| `architecture` | how a module/subsystem works |
| `gotcha` | non-obvious behavior, a footgun |
| `howto` | a procedure / recipe |
| `reference` | a pointer or lookup |

```bash
axil store context '{"type":"architecture","summary":"recall fuses vector+FTS+graph via RRF","files":["crates/axil-core/src/scoring.rs"]}'

# Retrieve only architecture context:
axil recall "how does recall rank results" --type architecture
```

`decisions` and `errors` deliberately have **no** recommended `type` — their
field shapes (`{summary, reason, files}` / `{error, root_cause, fix}`) already
encode their function, and a sub-`type` there would just be topic in disguise.

### `--type` semantics

- Matches `data.type` **case-insensitively** (lowercased + trimmed on both
  sides), exact — `--type Architecture` == `--type architecture`.
- **No alias expansion** — `--type architecture` returns exactly `architecture`,
  nothing fuzzy. Predictability is the point.
- Records **without** a `type` field are **excluded** when `--type` is set.
- It is a plain post-retrieval filter: no index, no change to scoring. On the
  `query` command the same effect is available via `--where type=<value>`.
- MCP parity: the `recall` tool accepts an optional `type` parameter with the
  same semantics.

## What is intentionally *not* here

Some things look like they belong on this axis but don't:

- **Topic** (feature/module/component/area) → entities + `_scope`, never a `type`.
- **Lifecycle** (current/superseded/resolved) → already carried by supersede
  edges (`→supersedes→`), decay tiers (Hot/Warm/Cold/Archived), and stored
  fixes — not a separate field.
- **A closed/enforced vocabulary** → the `store` write path stays instant and
  never rejects; the vocabulary is a recommendation, surfaced in `axil store
  --help` and the agent guidance, not validated at write time.
