# Graph Commands

Records relate to each other through typed, directed edges carrying optional
JSON properties. Edges live in the `<db>.axil.graph` companion file.

## link / unlink

```bash
axil --db <DB> link <FROM_ID> <EDGE_TYPE> <TO_ID> --props '{"action": "bugfix"}'
axil --db <DB> unlink <EDGE_ID>
```

## neighbors / edges

```bash
axil --db <DB> neighbors <ID> --type modified --direction out   # out|in|both
axil --db <DB> edges <ID>                                       # full edge structs incl. props
```

## traverse

Fixed-length path traversal — follows exactly the steps in the path
expression and returns the endpoint records:

```bash
axil --db <DB> traverse <ID> "->modified->file"
```

## lineage

Walk a derivation chain and keep the *path*: each hop carries the record
fields you select plus the numeric delta against its parent hop, so you can
read how a metric drifted across a chain of mutations.

```bash
# Record that candidate B was derived from A, with the mutation as edge props
axil --db <DB> link <B_ID> derived_from <A_ID> --props '{"mutation": "widened stop"}'

# The mutation path that led to B, with per-hop Sharpe/drawdown deltas
axil --db <DB> lineage <B_ID> --fields oos_sharpe,max_dd
```

Options: `--direction ancestors|descendants|both` (default `ancestors` —
root-first: what each node was derived from), `--edge-type <type>` (default
`derived_from`), `--max-depth <N>` (default 20), `--fields a,b,c` (which
`data` keys appear per hop; all keys when omitted).

The walk is breadth-first and cycle-safe (each node is emitted at most once).
In a branching tree, each hop's `delta` is measured against its *parent* —
the node on the other end of the discovering edge — never against a sibling.
A hop whose record was deleted is reported as `"missing": true` rather than
failing the walk. Envelope:

```json
{"root": "<id>", "direction": "ancestors", "edge_type": "derived_from",
 "hops": [
   {"depth": 0, "id": "<B>", "table": "strategies",
    "fields": {"oos_sharpe": 0.42}, "edge": null, "delta": {}},
   {"depth": 1, "id": "<A>", "table": "strategies",
    "fields": {"oos_sharpe": 0.35},
    "edge": {"edge_id": "…", "props": {"mutation": "widened stop"}},
    "delta": {"oos_sharpe": -0.07}}
 ]}
```

The `lineage` MCP tool returns the identical envelope.
