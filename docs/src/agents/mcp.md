# MCP Server

Axil includes a Model Context Protocol (MCP) server for integration with any MCP-compatible agent.

## Starting the server

```bash
axil mcp <DB>
```

The server communicates over stdio using JSON-RPC.

## Available tools

| Tool | Description |
|------|-------------|
| `recall` | Semantic search + graph + time-based recall |
| `store` | Store a record with auto-embedding |
| `link` | Create a graph relationship |
| `search` | Full-text search |
| `query_history` | Time-based query of past records |
| `get` | Fetch a record by ID |
| `list` | List records in a table |
| `delete` | Delete a record |

## Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "axil": {
      "command": "axil",
      "args": ["mcp", "/path/to/memory.axil"]
    }
  }
}
```

## HTTP API alternative

For non-stdio environments, use the HTTP API:

```bash
axil serve <DB> --host 0.0.0.0 --port 8080
```

Endpoints: `/api/health`, `/api/records`, `/api/recall`, `/api/search`, `/api/schema`, etc.
