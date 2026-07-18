# Vectors & Similarity

Beyond text embeddings, Axil stores **user-supplied raw vectors** — any
`Vec<f32>` your application computes (a strategy's daily-returns fingerprint,
an image embedding, a sensor signature) — and answers nearest-neighbor
queries over them with cosine similarity (higher = more similar, ~1.0 = near
duplicate). Vectors are persisted as exact f32 values; nothing is quantized
or transformed.

## Named vector spaces

A database has one *default* vector space (`<db>.axil.vec`) whose dimension
is bound to the text-embedding model (384 for the bundled bge-small). Raw
vectors of any other dimension go in a **named space** — an independent
companion file `<db>.axil.vec.<name>` with its own dimension, created lazily
on first write and never consulted by text recall:

```bash
# First write to a named space binds its dimension (here: 5)
axil --db <DB> add-vector <ID> "[0.1, -0.2, 0.4, 0.0, 0.9]" --space fingerprints
```

Space names match `[a-z0-9_-]{1,32}` (they become file-name suffixes).
Named spaces are not yet integrated with `heal`/`branch`/`snapshot`/`doctor`.

## store --vector

Insert a record and attach its vector in one shot:

```bash
axil --db <DB> store strategies '{"name": "meanrev-v3", "oos_sharpe": 0.42}' \
    --vector "[0.03, -0.11, 0.07, ...]" --space fingerprints
```

Without `--space` the vector goes to the default space and must match its
dimension. `--vector` and `--embed` are mutually exclusive.

## similar

Nearest-neighbor search by raw vector or by an existing record's stored
vector:

```bash
# By raw vector
axil --db <DB> similar --vector "[0.03, -0.11, ...]" --space fingerprints --top-k 5

# By record id — the record itself is excluded from results
axil --db <DB> similar --id <ID> --space fingerprints --threshold 0.95
```

`--threshold <F>` keeps only results with cosine similarity ≥ F — the
near-duplicate detector: two "different" strategies whose fingerprints score
0.97 are the same trade wearing different clothes.

## add-vector / search-vector

`add-vector <ID> '<json floats>'` attaches a vector to an existing record
(the record must exist first). `search-vector` is a deprecated alias for
`similar --vector`.

## MCP parity

The `add_vector` and `similar` MCP tools mirror the CLI, including `space`
and `threshold` — see [MCP Server](../agents/mcp.md).
