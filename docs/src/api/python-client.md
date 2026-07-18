# Python Client

`axil-client` (in [`clients/python/`](https://github.com/FC4b/axildb/tree/dev/clients/python))
is a thin, pure-stdlib Python wrapper over the `axil` CLI — every call shells
out with `--format json --quiet` and returns parsed JSON. No PyO3, no
compiled extension, no dependencies; it needs an `axil` binary on `PATH` (or
pass `binary=` explicitly).

```bash
pip install ./clients/python        # from a checkout
```

```python
from axil_client import Axil, AxilError

db = Axil("./memory.axil")

# Structured store + predicate query + aggregation
db.store("autopsies", {"family": "meanrev", "oos_sharpe": 0.42, "trades": 18,
                       "kill_reason": None})
survivors = db.query("autopsies", where="oos_sharpe > 0.3 AND trades < 30")
histogram = db.agg("autopsies", ["count"], group_by="kill_reason")

# Raw-vector fingerprints + near-duplicate detection
a = db.store("strategies", {"name": "meanrev-v3"},
             vector=daily_returns_fp, space="fp")
twins = db.similar(id=a["id"], space="fp", threshold=0.95)

# Lineage with per-hop metric deltas
db.link(b_id, "derived_from", a["id"], props={"mutation": "widened stop"})
path = db.lineage(b_id, fields=["oos_sharpe"])
```

Methods: `store` (with `embed=`/`vector=`/`space=`), `get`, `delete`,
`recall`, `query`, `agg`, `add_vector`, `similar`, `link`, `lineage`.
Non-zero exits raise `AxilError` carrying the exit code and stderr
(`get`/`delete` of a missing id exit 2 → raise). Aggregation metrics use the
same spelling as every other surface: `"count"`, `"avg(field)"`,
`"min(field)"`, `"max(field)"`, `"sum(field)"`.

The package's [README](https://github.com/FC4b/axildb/blob/dev/clients/python/README.md)
walks through the full trading-R&D loop example, and its pytest suite
round-trips every method against a real binary.
