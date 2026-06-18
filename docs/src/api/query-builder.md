# Query Builder

The `QueryBuilder` API provides a fluent interface for constructing queries.

## Basic queries

```rust
// Vector similarity search
let results = db.query()
    .similar_to("auth error", 5)
    .exec()?;

// Filter by table
let results = db.query()
    .table("sessions")
    .exec()?;

// Filter by field
let results = db.query()
    .where_field("project", Op::Eq, json!("my-app"))
    .exec()?;
```

## Combined queries

```rust
let results = db.query()
    .similar_to("authentication", 10)
    .table("sessions")
    .where_field("created_at", Op::Gt, json!("2026-01-01"))
    .traverse("->modified")
    .boost_recency(0.3)
    .limit(5)
    .exec()?;
```

## Operators

| Op | Description |
|----|-------------|
| `Op::Eq` | Equal |
| `Op::Ne` | Not equal |
| `Op::Gt` | Greater than |
| `Op::Lt` | Less than |
| `Op::Gte` | Greater or equal |
| `Op::Lte` | Less or equal |
| `Op::Contains` | String contains |

## Query explanation

```rust
let plan = db.query()
    .similar_to("test", 5)
    .explain()?;
// Returns QueryPlan with estimated costs and steps

let results = db.query()
    .similar_to("test", 5)
    .profile()
    .exec()?;
// Returns results with per-step timing
```

## Scoring

Results are scored using Reciprocal Rank Fusion (RRF):

- Vector similarity (cosine distance)
- Graph connectivity (PageRank boost)
- Recency (time decay)
- Keyword match (BM25)
- Feedback (relevance feedback loop)
