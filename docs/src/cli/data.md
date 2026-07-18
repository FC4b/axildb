# Data Commands

## store

Store a record in a table.

```bash
axil --db <DB> store <TABLE> '<JSON>'
axil --db ./db store sessions '{"summary": "Fixed auth bug"}'
```

Options: `--agent <name>`, `--entities '<json array>'`, `--llm`,
`--embed <fields>` (auto-embed text fields), `--code-ref <ref>`,
`--vector '<json floats>'` + `--space <name>` (attach a raw vector in one
shot — see [Vectors & Similarity](./vectors.md); mutually exclusive with
`--embed`)

## query

Filter a table by field predicates — range and equality filters over the
record's top-level JSON fields.

```bash
axil --db <DB> query <TABLE> --where "oos_sharpe > 0.3 AND family = 'meanrev'"
axil --db <DB> query autopsies --where "trades < 30" --order-by created_at --direction desc
axil --db <DB> query t --where "note contains 'timeout'" --limit 20
```

The `--where` grammar:

- One string may hold several conditions joined by `AND` (case-insensitive);
  the split is quote-aware. Repeating `--where` composes the same way —
  everything ANDs. `OR`, parentheses, and nested dot-paths are not supported.
- Operators: `=`, `!=`, `>`, `<`, `>=`, `<=`, and the word operator
  `contains` (substring for strings, membership for arrays).
- Unquoted values are typed by JSON rules (`0.3` compares numerically,
  `true`/`false`/`null` as themselves); single- or double-quoted values are
  always strings. Numbers compare numerically, never lexicographically.
- Malformed input errors instead of silently matching nothing: an
  unterminated quote is rejected, and an unquoted value like `5 oops` (a
  scalar followed by trailing text — usually a missing `AND`) asks you to
  quote it or split the conditions.

Also accepted by `list` and `explain`.

## agg

Aggregate over a table: `count`, `avg`, `min`, `max`, `sum`, with optional
grouping and the same `--where` grammar as `query`.

```bash
# Kill-reason histogram
axil --db <DB> agg autopsies --count --group-by kill_reason

# Average IS→OOS Sharpe decay per strategy family, survivors only
axil --db <DB> agg autopsies --avg sharpe_decay --group-by family \
    --where "oos_sharpe > 0"

# Exact trial accounting (deflated-Sharpe style): count everything, even archived
axil --db <DB> agg autopsies --count --include-archived
```

Metric flags repeat (`--avg a --avg b --min c`). Output is a stable envelope:

```json
{"table": "autopsies", "group_by": "kill_reason",
 "groups": [{"group": "drawdown", "count": 3, "avg_sharpe_decay": -0.42, "skipped": 0}],
 "total_rows": 3}
```

Rows whose field is missing or non-numeric are skipped for that metric and
counted per group in `skipped`; a group with no numeric samples renders the
metric as `null`. Records archived by memory pressure are excluded unless
`--include-archived` is set. The same aggregation is available in AxilQL as
the [`AGG` statement](./axilql.md) and over MCP as the `aggregate` tool.

## get

Retrieve a record by ID.

```bash
axil --db <DB> get <ID>
```

## list

List all records in a table.

```bash
axil --db <DB> list <TABLE>
axil --db <DB> list <TABLE> --limit 10
```

## delete

Delete a record by ID.

```bash
axil --db <DB> delete <ID>
```

## update

Update a record's data.

```bash
axil --db <DB> update <ID> '<JSON>'
```

## tables

List all tables with record counts.

```bash
axil --db <DB> tables
```

## legacy alias

`axil insert` remains available as a compatibility alias for `axil store`.

```bash
axil --db ./db insert context '{"type": "architecture", "summary": "..."}'
```
