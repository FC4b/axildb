# axil-client

A thin, **pure-standard-library** Python client for [Axil](https://github.com/FC4b/axildb).

It shells out to the `axil` binary for every operation — no PyO3, no FFI, no
network layer. Each method builds an argument list, runs the binary with
`--format json --quiet`, and parses the single line of JSON the CLI prints to
stdout. Diagnostic chatter (model notices, index warnings) goes to stderr, so
stdout stays clean JSON. A non-zero exit raises `AxilError` carrying the exit
code and stderr.

## Requirements

- Python 3.9+
- The `axil` binary on `PATH` (or pass `binary=`/absolute path). Build it from
  the repo with `cargo build --release -p axildb` → `target/release/axil`.

## Install

```bash
pip install axil-client          # once published
# or, from a checkout:
pip install ./clients/python
```

The package has **zero runtime dependencies**; you can also just drop
`axil_client/` on your `PYTHONPATH`.

## Quick start

```python
from axil_client import Axil, AxilError

# `axil init ./memory.axil` once first — that creates the graph + vector stores
# that link/lineage and default-space vectors need. (Named vector spaces are
# created lazily, so the fingerprint example below needs no init.)
db = Axil("./memory.axil")

rec = db.store("autopsies", {"strategy": "mr-1", "oos_sharpe": 0.42})
print(rec["id"])
```

## The strategy-R&D loop

The client mirrors the R&D-loop workflow end to end: store an autopsy, tally the
kill-reason histogram, check a new fingerprint for near-duplicates, and trace a
strategy's mutation lineage.

```python
from axil_client import Axil

db = Axil("./research.axil")   # run `axil init ./research.axil` first

# 1. Store the post-mortem of a discarded trial.
autopsy = db.store("autopsies", {
    "strategy":    "meanrev-v3",
    "family":      "meanrev",
    "regime":      "chop",
    "oos_sharpe":  0.18,
    "max_dd":      0.27,
    "trades":      41,
    "fees":        0.004,
    "kill_reason": "drawdown",
})

# 2. Kill-reason histogram across every trial (including archived/discarded
#    ones — the deflated-Sharpe accounting needs exact counts).
hist = db.agg("autopsies", ["count"], group_by="kill_reason", include_archived=True)
for g in hist["groups"]:
    print(f"{g['group']:>12}: {g['count']}")
# ->     drawdown: 12
# ->         fees: 4
# ->    overfit  : 7

# Per-family average out-of-sample Sharpe, filtered to real contenders.
decay = db.agg(
    "autopsies",
    ["count", "avg(oos_sharpe)"],
    group_by="family",
    where="trades >= 30 AND oos_sharpe > 0.0",
)

# 3. Near-duplicate check: before spending compute on a "new" idea, compare its
#    fingerprint against everything we've already tried. A named vector space
#    keeps these N-dim fingerprints off the text-embedding index.
candidate = [0.81, 0.02, 0.44, 0.10, 0.33, 0.09, 0.55, 0.07]
db.store("fingerprints", {"strategy": "meanrev-v4"},
         vector=candidate, space="fingerprints")

new_id = db.store("fingerprints", {"strategy": "meanrev-v5"},
                  vector=[0.80, 0.03, 0.45, 0.11, 0.31, 0.08, 0.56, 0.06],
                  space="fingerprints")["id"]

dupes = db.similar(id=new_id, space="fingerprints", threshold=0.95)
if dupes:
    print("near-duplicate of:", dupes[0]["data"]["strategy"], dupes[0]["score"])

# 4. Lineage: how did this strategy mutate, and how did the metrics drift?
#    Record the ancestry at store time:
#      db.link(child_id, "derived_from", parent_id,
#              props={"mutation": "widened stop"})
chain = db.lineage(new_id, direction="ancestors", fields=["oos_sharpe", "max_dd"])
for hop in chain["hops"]:
    mut = (hop.get("edge") or {}).get("props", {}).get("mutation", "(root)")
    print(hop["depth"], mut, hop.get("delta"))
```

## Methods

| Method | Wraps | Returns |
|--------|-------|---------|
| `store(table, data, embed=None, vector=None, space=None)` | `axil store` | `{"id", "table", "created_at"[, "vector_dims", "space"]}` |
| `get(id)` | `axil get` | record object (raises on missing) |
| `delete(id)` | `axil delete` | `{"deleted", "id"}` (raises on missing) |
| `recall(query, top_k=5)` | `axil recall` | list of `{"id", "score", "summary", "table"}` |
| `query(table, where=None, order_by=None, limit=None)` | `axil query` | list of record objects |
| `agg(table, metrics, group_by=None, where=None, include_archived=False)` | `axil agg` | `{"table", "group_by", "groups", "total_rows"}` |
| `add_vector(id, vector, space=None)` | `axil add-vector` | `{"added", "id", "dimensions"[, "space"]}` |
| `similar(vector=None, id=None, space=None, top_k=5, threshold=None)` | `axil similar` | list of `{"id", "score", "data", "table", "created_at"}` |
| `link(from_id, edge_type, to_id, props=None)` | `axil link` | `{"edge_id", "from", "to", "edge_type"}` |
| `lineage(id, direction="ancestors", edge_type="derived_from", max_depth=20, fields=None)` | `axil lineage` | `{"root", "direction", "edge_type", "hops"}` |

### `agg` metric specs

`metrics` is a list. Each item is `"count"` or a function form
`"avg(field)"`, `"min(field)"`, `"max(field)"`, `"sum(field)"` (the shorthand
`"avg:field"` also works). Metrics combine; grouped results carry a
`skipped` count of non-numeric/missing values per group.

### `where` predicates

A single string; several conditions may be joined by `AND` (case-insensitive),
with single- or double-quoted string values. Operators: `= != > < >= <=` and
`contains`. Numbers compare numerically. `OR`, parentheses, and nested
dot-paths are **not** supported (matching Axil's core WHERE semantics).

## Errors

Every non-zero exit raises `AxilError`:

```python
from axil_client import Axil, AxilError

db = Axil("./research.axil")
try:
    db.get("01ARZ3NDEKTSV4RRFFQ69G5FAV")     # not found
except AxilError as e:
    print(e.exit_code)   # 2  (0=ok, 1=error, 2=not-found)
    print(e.stderr)      # {"error": "not found", "id": "..."}
```

## Tests

```bash
cd clients/python
python -m pytest                 # locates target/release/axil, else PATH; skips if absent
AXIL_BIN=/path/to/axil python -m pytest    # pin a specific binary
```

The suite round-trips every method against a temporary database. Recall tests
additionally skip when no embedding model is present on the host.

## License

PolyForm Noncommercial 1.0.0 — same as Axil. Commercial use requires a
commercial license.
