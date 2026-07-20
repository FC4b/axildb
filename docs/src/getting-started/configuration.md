# Configuration

Axil is configured via `axil.toml` files. Configuration is searched in order:

1. The nearest `axil.toml`, walking up from the database's directory
   (CLI/library opens) or the current directory
2. `~/.config/axil/config.toml` (user config)
3. Built-in defaults

Because the *nearest* file wins, a database can carry its own policy: drop an
`axil.toml` in the same directory as the `.axil` file and it overrides the
project-root config for that database only.

## Example `axil.toml`

```toml
[database]
path = "./memory.axil"

[index]
embedding_model = "bge-small-en-v1.5"
embedding_dimensions = 384

[runtime]
max_results = 50

[fts]
default_limit = 10

[healing]
auto_compact = true                      # false = compaction is manual-only
supersede_similarity_threshold = 0.92    # auto-supersede above this similarity

[metrics]
enabled = true

[llm]
endpoint = "https://api.openai.com/v1/chat/completions"
model = "gpt-4o-mini"
# api_key = "..." or set AXIL_LLM_API_KEY env var
```

## Configuration keys

### Database

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `database.path` | string | `./memory.axil` | Default database path |

### Index

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `index.embedding_model` | string | `bge-small-en-v1.5` | ONNX embedding model |
| `index.embedding_dimensions` | int | `384` | Vector dimensions |

### Runtime

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `runtime.max_results` | int | `50` | Default max results |

### Healing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `healing.auto_compact` | bool | `true` | Allow automatic healing (`axil heal`, session-close auto-heal) to purge expired/superseded records. `false` makes compaction manual-only (`axil compact` / `axil heal --compact` still work). |
| `healing.supersede_similarity_threshold` | float | `0.92` | Auto-supersede fires when a new record's vector similarity to a same-table record exceeds this. Values above `1.0` disable auto-supersede globally. |

### Per-table lifecycle policy

For tables where similar-sounding records are **distinct events, not
revisions** — experiment logs, trade autopsies, audit trails — opt out of the
memory-hygiene machinery per table:

```toml
[lifecycle.tables.autopsies]
supersede = false   # new similar records never demote existing ones
decay = false       # importance never decays
compact = "never"   # compaction never purges this table (append-only)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lifecycle.tables.<t>.supersede` | bool | `true` | Allow auto-supersede to mark older similar records in `<t>` as superseded (and the brain pipeline to dedupe near-identical observations). |
| `lifecycle.tables.<t>.decay` | bool | `true` | Allow importance decay for `<t>` (false = infinite half-life). |
| `lifecycle.tables.<t>.compact` | string | `"auto"` | `"never"` = `compact()`/`heal` never delete records from `<t>`, and its records are excluded from "pending cleanup" diagnostics. |

The policy is enforced in the core (`AxilBuilder::build` reads the nearest
`axil.toml`), so CLI, MCP server, and embedded library use all honor it.
To scope a policy to one database in a multi-DB project, place the
`axil.toml` next to that `.axil` file — nearest config wins.

### LLM (optional)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.endpoint` | string | none | OpenAI-compatible endpoint |
| `llm.model` | string | none | Model name |
| `llm.api_key` | string | none | API key (or `AXIL_LLM_API_KEY` env) |

## Environment variables

| Variable | Description |
|----------|-------------|
| `AXIL_DB` | Override the database path |
| `AXIL_LLM_API_KEY` | LLM API key |
| `AXIL_LOG` | Log level (`error`, `warn`, `info`, `debug`, `trace`) |

## CLI configuration

```bash
# View current config
axil config show

# Set a value
axil config set index.embedding_model bge-base-en-v1.5

# Get a value
axil config get index.embedding_model
```
