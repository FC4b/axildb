# AxilQL

AxilQL is a verb-first query language that compiles to the QueryBuilder API.

## Interactive mode

```bash
axil ql -i <DB>
```

## Syntax

### RECALL — semantic vector search

```sql
RECALL "auth timeout bug" TOP 10
RECALL "error handling" TOP 5 FROM sessions
RECALL "auth" TOP 5 BOOST recency 0.8
```

### FIND — full-text search

```sql
FIND "timeout" LIMIT 20
FIND "authentication" IN summary
FIND "error" WHERE table = "sessions"
```

### TRAVERSE — graph traversal

```sql
TRAVERSE ->modified->file FROM <record-id>
TRAVERSE <-mentions WHERE table = "_entities"
```

**Knowledge-time cutoff** — restrict traversal to edges recorded at or before a
timestamp, answering "what did the graph look like as of then?":

```sql
TRAVERSE ->depends_on FROM <record-id> KNOWN AT '2026-01-01T00:00:00Z'
```

`KNOWN AT` takes an RFC 3339 timestamp and requires the record-seeded form
(`FROM <record-id>`). It filters on when each edge was *recorded*; the
event-time axis (edge validity windows) is available programmatically via
`GraphEngine::traverse_ids_bitemporal`.

### GET — fetch by ID

```sql
GET <record-id>
```

### COUNT — count records

```sql
COUNT FROM sessions
COUNT FROM autopsies WHERE family = "meanrev"
COUNT
```

Without `FROM`, all tables (including internal `_`-prefixed ones) are
counted; adding `WHERE` filters each table's records before counting.

### AGG — aggregations

```sql
AGG count FROM autopsies GROUP BY kill_reason
AGG avg(sharpe_decay), count FROM autopsies WHERE oos_sharpe > 0 GROUP BY family
AGG min(fees), max(fees), sum(fees) FROM trades
```

Metric functions: `count`, `avg(field)`, `min(field)`, `max(field)`,
`sum(field)` (names case-insensitive). Returns one row per group with
`count`, one `<func>_<field>` key per metric, and a `skipped` counter for
rows whose field was missing or non-numeric. Every matching row is folded —
there is no result cap. The CLI `axil agg` command and the MCP `aggregate`
tool run the same executor.

### EXPLAIN — show query plan

```sql
EXPLAIN RECALL "auth error" TOP 5
```

## Clauses

| Clause | Description | Example |
|--------|-------------|---------|
| `WHERE` | Filter conditions | `WHERE table = "sessions"` |
| `AND` | Additional conditions | `AND created_at > "2026-01-01"` |
| `FROM` | Table filter | `FROM sessions` |
| `TOP` | Limit for RECALL | `TOP 10` |
| `LIMIT` | Result limit | `LIMIT 20` |
| `OFFSET` | Skip results | `OFFSET 10` |
| `ORDER BY` | Sort results | `ORDER BY created_at DESC` |
| `BOOST` | Adjust scoring | `BOOST recency 0.8` |
| `PROFILE` | Include timing | `RECALL "x" TOP 5 PROFILE` |
| `TRAVERSE` | Chain traversal | `TRAVERSE ->edge` |

## Contextual keywords

`AGG`, `GROUP`, `KNOWN`, and `AT` are keywords only where the grammar expects
them (a leading `AGG`, `GROUP BY`, `KNOWN AT`). Anywhere else they are
ordinary identifiers, so fields and tables with those names need no quoting:

```sql
COUNT FROM events WHERE at > "2026-01-01"
AGG count FROM group GROUP BY known
```

## Operators

`=`, `!=`, `>`, `<`, `>=`, `<=`, `CONTAINS`

## Comments

```sql
-- This is a comment
RECALL "test" TOP 5  -- inline comment
```
