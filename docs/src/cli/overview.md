# CLI Overview

The `axil` CLI is the primary interface for agents and developers.

## Usage

```
axil [OPTIONS] <COMMAND> [ARGS]
```

## Global options

| Option | Description |
|--------|-------------|
| `--db <PATH>` | Database path (or set `AXIL_DB`) |
| `--format <FMT>` | Output format: `json` (default), `pretty`, `table` |

## Command groups

| Group | Commands | Description |
|-------|----------|-------------|
| **Data** | `insert`, `get`, `list`, `delete`, `update` | CRUD operations |
| **Search** | `recall`, `search`, `embed` | Vector + FTS search |
| **Graph** | `relate`, `neighbors`, `traverse`, `edges` | Graph operations |
| **Memory** | `know`, `about`, `session`, `believe`, `checkpoint` | Agent memory |
| **Rules** | `rule set/get/list/delete`, `rule extract`, `rule distill` | Agent directives & conventions. `extract` reads them *from* `CLAUDE.md` into memory; `distill` distills recurring failures *back out* into corrective directives (`rules` is an accepted alias) |
| **Diagnostics** (read-only) | `doctor`, `health-report`, `snapshot`, `trends`, `detect`, `stats` | *Watch* the DB — never change memories |
| **Memory lifecycle** | `compact`, `heal`, `worker`, `decay` | *Change or forget* memories (reclaim / consolidate / downsample) |
| **Automation** | `maintain` | Run the additive diagnostics on a cadence |
| **Branch** | `branch create/list/delete/diff/switch/merge` | Database branching |
| **Query** | `ql` | AxilQL queries |
| **Server** | `mcp`, `serve` | MCP + HTTP servers |
| **Setup** | `install`, `sync`, `features` | Project install (bare `install` on a TTY opens an interactive wizard); `sync` updates an existing install; `features` inspects binary components (+ rebuild wizard) |

## Diagnostics vs. lifecycle vs. maintenance — when to use which

These three families overlap by name, which is the usual source of confusion. The rule:

- **Diagnostics** *watch* the DB and never modify your memories.
- **Lifecycle** commands *change or forget* memories.
- **`maintain`** just automates the (additive, non-destructive) diagnostics so you never have to schedule them.

| I want to… | Command |
|---|---|
| Check if anything's wrong right now | `axil doctor` |
| Get a scored health report + fixes | `axil health-report` |
| Chart metrics over time | `axil snapshot` (records) → `axil trends` (charts) |
| Run the deep / expensive detectors | `axil detect` |
| Reclaim space / drop dead rows | `axil compact` |
| Rebuild a broken or empty index | `axil heal --reindex` |
| Roll up very old records (**destructive**) | `axil heal` |
| Let recurring upkeep happen automatically | *(nothing — the brain hook runs `axil maintain`)* |

> ⚠️ **`snapshot` is a *metrics* snapshot, not a data backup** — for a data copy use `axil branch create`. And **`heal`/`compact` change data; the diagnostics don't** — run `axil doctor` first to see if you even need them.

## JSON output

All commands output JSON by default, making them easy to parse programmatically. Use `--format pretty` for human-readable output.
