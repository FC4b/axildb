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

### GET — fetch by ID

```sql
GET <record-id>
```

### COUNT — count records

```sql
COUNT FROM sessions
COUNT
```

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

## Operators

`=`, `!=`, `>`, `<`, `>=`, `<=`, `CONTAINS`

## Comments

```sql
-- This is a comment
RECALL "test" TOP 5  -- inline comment
```
