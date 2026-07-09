# Semantic Answer Cache

Axil can cache a **question → answer pair** so that when a *semantically
similar* question recurs, the stored answer is returned instead of being
re-derived. Unlike a plain semantic cache guarded only by a similarity
threshold and a TTL, an Axil cache entry can pin itself to the **code it
talks about** — and is invalidated when that code changes.

> **Scope:** the feature lives behind the `cache` Cargo feature, which is
> enabled in the default `axil` binary and excluded only from minimal
> `--features core` builds.

## How it works

```
cache put  ──►  embed question ──►  _cache_entries  (+ code-ref fingerprints)
                                          │
cache get "<question>"  ──►  similarity ≥ threshold?
                                          │  yes
                          re-check TTL + code-ref fingerprints
                             │ fresh                    │ changed / expired
                             ▼                          ▼
                        return answer              evict + report miss
```

1. **Put** — `cache put` stores the question (embedded + full-text indexed)
   and answer. Any `code_refs` are resolved and each gets a content
   fingerprint captured at put time.
2. **Get** — `cache get` ranks entries by question similarity. The best
   entry at or above the threshold is returned **only if** it is neither
   expired nor code-stale; otherwise it is evicted and the read reports why.

### Code-aware invalidation

Each code ref captures two independent fingerprints when available:

- a **file hash** of the referenced file's on-disk content — recomputed
  straight from disk on every read, so a raw edit invalidates the entry even
  before the indexer re-runs;
- a **proxy hash** of the matching `_idx_code_proxies` row's structural text
  — which also goes absent when the symbol is removed and re-indexed.

On read, both are recomputed and compared to the values stored at put time.
Any mismatch (a changed hash, or a hash that was captured but is now gone)
marks the entry stale: it is dropped and the read returns a miss with reason
`stale_code`. Staleness is checked lazily on read, so nothing hooks into
file-change events.

## Commands

### `axil cache put`

Store a question/answer pair. Accepts a JSON object as a positional
argument, or `-` to read it from stdin.

```
axil cache put '{"question": "how does auth token refresh work?", "answer": "the refresh worker rotates the token in auth.rs", "code_refs": ["src/auth.rs"]}'

echo '{"question":"…","answer":"…"}' | axil cache put -
```

Fields:

- `question` (required) — embedded for semantic recall.
- `answer` (required) — returned on a future similar question.
- `code_refs` (optional) — an array of code-ref specs to invalidate against.
  Each is a `proxy_id`, a SCIP `canonical_id`, or a `path` / `path:line`
  (the same forms `axil store --code-ref` accepts).
- `ttl` (optional) — time-to-live in seconds from now.
- `valid_until` (optional) — an explicit RFC 3339 expiry; overrides `ttl`.

### `axil cache get`

Look up a cached answer for a question.

```
axil cache get "how does the token refresh flow work?" [--threshold 0.92] [--top-k 1]
```

- `--threshold` — minimum cosine similarity for a hit. Default `0.92`, the
  same threshold Axil uses for memory superseding.
- `--top-k` — maximum number of hits to return. Default `1`.

A **hit** returns the answer(s), each with its similarity `score` and
`hit_count`:

```json
{
  "result": "hit",
  "count": 1,
  "hits": [
    { "id": "01K…", "question": "how does auth token refresh work?",
      "answer": "the refresh worker rotates the token in auth.rs",
      "score": 0.97, "hit_count": 2 }
  ]
}
```

A **miss** names its reason so the agent knows whether to re-derive:

| `reason` | Meaning |
|----------|---------|
| `no_match` | Nothing similar is cached. |
| `below_threshold` | The closest entry scored under the threshold (`best_score` is reported). |
| `stale_code` | The closest entry referenced code that changed; it was evicted (`detail` names the ref). |
| `expired` | The closest entry had passed its TTL; it was evicted. |

When no vector index is configured, `cache get` falls back to an exact
question-text match (scored `1.0`) so the cache still works, without
semantic matching.

### `axil cache stats`

Cumulative counters over the cache's lifetime.

```json
{
  "entries": 12,
  "total_hits": 40,
  "total_misses": 8,
  "hit_rate": 0.833,
  "stale_evictions": 3,
  "expired_evictions": 1
}
```

`hit_rate` is `null` until at least one read has happened.

### `axil cache clear`

Remove cached entries.

```
axil cache clear             # removes only expired entries (safe default)
axil cache clear --expired   # same as above, explicit
axil cache clear --all       # removes every entry
```

Cumulative counters in `cache stats` are historical and are not reset by
`clear`.

## MCP tools

The same surface is available over MCP as `cache_put` and `cache_get`, with
identical parameters and result shapes.

## Storage

- `_cache_entries` — one row per cached pair (`question`, `answer`,
  `created_at`, optional `valid_until`, `hit_count`, `last_hit_at`,
  `code_refs[]`). The `question` field is embedded and full-text indexed.
- `_cache_meta` — a single row of cumulative counters backing `cache stats`.

Both tables are internal (leading underscore), so cache entries never appear
in ordinary `axil recall`.
