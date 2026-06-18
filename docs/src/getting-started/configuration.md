# Configuration

Axil is configured via `axil.toml` files. Configuration is searched in order:

1. `./axil.toml` (current directory)
2. `~/.config/axil/axil.toml` (user config)
3. Built-in defaults

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
auto_compact_threshold = 1000

[metrics]
enabled = true

[llm]
endpoint = "https://api.openai.com/v1"
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
