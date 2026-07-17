# MCP Server

Axil ships a [Model Context Protocol](https://modelcontextprotocol.io) server so any
MCP-compatible agent (Claude Code, Cursor, Windsurf, custom hosts) can read and write
the same cognitive memory the CLI uses. The server speaks JSON-RPC over stdio.

> The CLI is the primary, fully-featured surface. The MCP server exposes the
> agent-facing subset of that surface as tools — same database, same records.
> Anything not yet exposed as a tool is reachable by shelling out to `axil …`.

## Starting the server

The database is selected with the global `--db` flag (or `AXIL_DB`), **not**
positionally:

```bash
axil --db ./.axil/memory.axil mcp
# or, with auto-detection of ./.axil/memory.axil:
axil mcp
```

The process reads newline-delimited JSON-RPC requests from stdin and writes
responses to stdout.

## Configuration

For Claude Code, the one-liner registers the server in a single step (note the
`--` separator — everything after it is the server command, so the global
`--db` flag lands on `axil`, not on `claude`):

```bash
claude mcp add axil -- axil --db ./.axil/memory.axil mcp
```

For any other MCP client, add the equivalent block to its configuration. Keep
the `--db` value and the literal `mcp` subcommand as the last argument:

```json
{
  "mcpServers": {
    "axil": {
      "command": "axil",
      "args": ["--db", "/path/to/memory.axil", "mcp"]
    }
  }
}
```

## Smoke test (no client required)

Before wiring up a client, confirm the server speaks the protocol by piping
three newline-delimited JSON-RPC frames — `initialize`, `tools/list`, and a
`tools/call` of `recall` — straight into its stdin. No MCP client, SDK, or
network is involved; the server reads each line and writes one response line.

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"query":"auth timeout","top_k":3}}}' \
  | axil --db ./.axil/memory.axil mcp
```

You get three response lines. The first confirms the handshake — look for
`serverInfo.name == "axil-mcp"`:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"axil-mcp","version":"…"},"instructions":"…"}}
```

The second lists every registered tool (`recall`, `store`, `boot`, …). The
third is the `recall` result, wrapped as MCP tool-call content — its `text`
field is a JSON array of hits (`[]` against a fresh, empty database):

```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"[]"}]}}
```

A non-empty array (each hit carrying `id`, `table`, `data`, `score`) means the
server is reading real memory. If the handshake frame is missing or
`serverInfo.name` is anything else, the wrong binary is on `PATH`.

## The tool surface

The assembled surface is the built-in tools plus the tools of every enabled
**Extension** (`deps`, `checkpoint`, and `cache` ship by default). The groups below describe
the full default surface; a trimmed build that disables an Extension drops that
Extension's tools.

A Rust drift test (`crates/adapters/axil-mcp/src/lib.rs`) asserts this document
stays in sync with the assembled runtime set — adding or removing a tool without
updating this table fails the build.

### Boot — start every session here

| Tool | Params | When to use |
|------|--------|-------------|
| `boot` | `budget?` int, `topic?` string, `scope?` string[] | **First call of a session.** Returns a stable `BootContext` (schema v1): current scope, constraints, recent decisions, active failures, open threads, preferences, confidence notes. Token-budget aware — lower-priority sections drop when over budget. |
| `inspect` | (none) | Read-only overview of what kinds of memory this brain holds and whether it is healthy. Returns a per-record-type census (e.g. `decisions`, `errors`, `sessions`; all internal bookkeeping tables collapse into one `_internal` bucket) plus a light health verdict (`ok`/`warning`/`error`) drawn from the same checks as `axil doctor`. Performs zero writes — the MCP-only equivalent of glancing at `axil tables` + `axil doctor`. |

### Intent-native writes — store cognition, not rows

Prefer these over raw `store` when the thing you are recording *is* a decision,
an error, or a preference: they auto-embed, auto-supersede, and dedupe.

| Tool | Params | When to use |
|------|--------|-------------|
| `remember_decision` | `summary` string (req), `reason?`, `files?` string[], `agent_id?`, `external_id?`, `force_new?` bool | After choosing approach A over B. Auto-embeds, auto-supersedes prior decisions, and dedupes by `(agent_id, external_id)` or a 5-minute content hash. |
| `remember_error` | `error` string (req), `root_cause?`, `fix?`, `files?` string[], `agent_id?`, `external_id?`, `force_new?` bool | After hitting a bug or gotcha. Same idempotency rules as `remember_decision`. |
| `set_preference` | `key` string (req), `value` any (req) | Record a user preference. Overwrites by key; the previous value is kept on the new record as `_previous_value` for a lightweight audit trail. |
| `close_session` | `id` string (req), `summary?` string | Mark a session closed with an optional summary. Idempotent by `id`. |

### Code — structural recall without dumping source

| Tool | Params | When to use |
|------|--------|-------------|
| `code_context` | `task` string (req), `budget?` int | Assemble a coding-task context block within a token budget. Groups results into `relevant_code`, `related_memories`, `relevant_modules`, `similar_context`, `active_rules`, `recent_changes`. Auto-sizes the budget by indexed repo size when omitted. The highest-leverage tool before an edit. |
| `code_search` | `query` string (req), `top_k?` int (default 5) | Search structural code proxies; returns compact pointers (path, line, symbol, breadcrumb, canonical_id). Smaller and more actionable than `recall` for code-shaped queries because raw source is never returned. |

### CRUD — general memory read/write

| Tool | Params | When to use |
|------|--------|-------------|
| `recall` | `query` string (req), `top_k?` int (default 5), `table?`, `type?`, `across?` string[], `strict_consent?` bool | Semantic + graph + time-based recall. Ranks by vector similarity blended with recency. `across` fans out to sibling workspace DBs with per-sibling read consent and provenance tags. |
| `store` | `table` string (req), `data` object (req), `embed_fields?` string[] | Insert an arbitrary record, optionally auto-embedding named fields. Use the intent-native writes above when the record is a decision/error/preference. |
| `search` | `query` string (req), `limit?` int (default 10) | Full-text search across all indexed fields. Use when you want lexical matches, not semantic similarity. |
| `query_history` | `after?` ISO-8601, `before?` ISO-8601, `table?`, `limit?` int (default 50) | Time-based query of past records by date range and table. |
| `get` | `id` string (req) | Fetch a single record by ID. |
| `list` | `table` string (req), `limit?` int (default 50) | List records in a table. |
| `query` | `table` string (req), `where?` string, `order_by?`, `direction?` (`asc`/`desc`), `limit?` int, `offset?` int | Query a table with typed field predicates, ordering, and pagination. The `where` string mirrors the CLI `--where` syntax: conditions joined by `AND` (case-insensitive), operators `= != > < >= <= contains`, quoted values forced to strings, unquoted numbers compared numerically. Flat top-level fields only — no OR, parentheses, or nested dot-paths. |
| `aggregate` (`ql` feature) | `table` string (req), `metrics?` string[] (`count`/`avg(f)`/`min(f)`/`max(f)`/`sum(f)`), `group_by?`, `where?` string, `include_archived?` bool | Aggregate a table into per-group `count`/`avg`/`min`/`max`/`sum`. Non-numeric or missing values are skipped for the numeric metrics and reported per group as `skipped`. Returns `{table, group_by, groups:[{group, count, …}], total_rows}`. |
| `delete` | `id` string (req) | Delete a record by ID. |
| `link` | `from` string (req), `edge_type` string (req), `to` string (req), `props?` object | Create a graph edge between two records. |

### Extension tools — dependency docs (`deps` feature)

| Tool | Params | When to use |
|------|--------|-------------|
| `dep_docs` | `query` string (req), `top_k?` int, `dep?` string, `include_superseded?` bool | Scoped query over version-pinned dependency-doc memory. Returns docs for the exact versions the project resolves to — zero network calls. |
| `deps_status` | (none) | List the dependencies whose docs are in memory: name, resolved version, ecosystem, and stored doc-chunk count. |

### Extension tools — session checkpoints (`checkpoint` feature)

| Tool | Params | When to use |
|------|--------|-------------|
| `checkpoint` | `goal?`, `state?`, `next_steps?` string[], `open_questions?` string[], `references?` object[], `summary?`, `session?`, `final?` bool | Write a structured checkpoint so a fresh agent can resume. `references[]` are typed pointers (`{kind, ref, note?}`), not copies; `record` kinds resolve live at boot. Replaces the old free-text session summary. |
| `checkpoint_show` | (none) | Return the current checkpoint — the stored one if present, otherwise one derived from the latest session. |

### Extension tools — semantic answer cache (`cache` feature)

| Tool | Params | When to use |
|------|--------|-------------|
| `cache_put` | `question` string (req), `answer` string (req), `code_refs?` string[], `ttl?` int, `valid_until?` string | Cache a question/answer pair so a future semantically similar question returns the stored answer instead of re-deriving it. `code_refs` (proxy_id \| canonical_id \| path[:line]) pin the answer to code; the entry is invalidated when that code changes. |
| `cache_get` | `question` string (req), `threshold?` number (default 0.92), `top_k?` int (default 1) | Return a cached answer for a semantically similar question, or a miss. A hit re-checks TTL and code-ref fingerprints first, so a returned answer is neither expired nor invalidated by a code change. Miss reasons distinguish `no_match` / `below_threshold` / `stale_code` / `expired`. |
| `cache_stats` | (none) | Cumulative semantic-cache statistics: live entry count, lifetime hits/misses, hit rate, and how many entries were evicted for stale code or expiry. |

### Cross-agent delta — `recall_delta` (`event-log` feature, off by default)

When the server is built with the `event-log` feature, it exposes one more
tool: **`recall_delta`**. It is **off by default** — the feature drives a
write-amplifying durable event log, so it is opt-in and the default tool surface
is unchanged without it.

`recall_delta(since_cursor?, exclude_agent?, limit?)` pulls committed *semantic
events* that happened after a cursor, oldest first — a belief revised, a
decision superseded, an error fixed, a checkpoint written. Each event carries a
monotonic `cursor`; pass the last one back in as `since_cursor` to resume. Use
`exclude_agent` to drop a given agent's own writes. It is the pull-based "what
changed since I last looked" feed for a second agent sharing the brain.

Isolation: `recall_delta` returns **committed facts only** — a notification that
a record changed, never another agent's private record body. It does not relax
cross-agent session isolation; resolve a returned `record_id` through the normal
access path.

## HTTP API alternative

For non-stdio environments, use the HTTP API:

```bash
axil serve <DB> --host 0.0.0.0 --port 8080
```

Endpoints: `/api/health`, `/api/records`, `/api/recall`, `/api/search`, `/api/schema`, etc.

## See also

- [CLI overview](../cli/overview.md) — the full command surface the MCP tools draw from.
- [Extending Axil](../extending/overview.md) — how Extensions contribute their own MCP tools.
