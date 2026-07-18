"""Round-trip tests for :class:`axil_client.Axil` against a real ``axil`` binary.

The binary is located via, in order: the ``AXIL_BIN`` env var, the repo's
``target/release/axil(.exe)``, then ``PATH``. If none resolves, the whole
module is skipped politely (so a checkout without a build still runs green).

Every public method of the client is exercised against a temporary database.
Recall/embedding tests additionally skip if no embedding model is present on
the host (a missing model surfaces as an :class:`AxilError`).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from axil_client import Axil, AxilError


# ── binary discovery ────────────────────────────────────────────────────────
def _find_binary():
    """Return a path/name for the axil binary, or ``None`` if unavailable."""
    env = os.environ.get("AXIL_BIN")
    if env and (Path(env).exists() or shutil.which(env)):
        return env

    # Walk up from this file to a repo root that has target/release/axil(.exe).
    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in ("axil.exe", "axil"):
            cand = parent / "target" / "release" / name
            if cand.exists():
                return str(cand)

    found = shutil.which("axil")
    if found:
        return found
    return None


AXIL_BIN = _find_binary()
pytestmark = pytest.mark.skipif(
    AXIL_BIN is None,
    reason="axil binary not found (set AXIL_BIN, build target/release/axil, or add to PATH)",
)


def _embeddings_available(bin_path, db_path):
    """True if the host has a working embedding model (needed for recall)."""
    try:
        subprocess.run(
            [bin_path, "--db", str(db_path), "--format", "json", "--quiet",
             "store", "_probe", '{"summary": "probe"}', "--embed", "summary"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            stdin=subprocess.DEVNULL, check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture()
def db(tmp_path):
    """A fresh, fully-initialized database + client bound to it.

    ``axil init`` creates the graph and default vector stores that
    ``link``/``lineage`` and default-space vectors require.
    """
    db_path = tmp_path / "mem.axil"
    subprocess.run(
        [AXIL_BIN, "init", str(db_path), "--format", "json", "--quiet"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        stdin=subprocess.DEVNULL, check=True,
    )
    return Axil(db_path, binary=AXIL_BIN)


# ── record CRUD ─────────────────────────────────────────────────────────────
def test_store_and_get_roundtrip(db):
    rec = db.store("autopsies", {"strategy": "mr-1", "oos_sharpe": 0.42})
    assert rec["table"] == "autopsies"
    rid = rec["id"]
    assert rid

    got = db.get(rid)
    assert got["id"] == rid
    assert got["data"]["strategy"] == "mr-1"
    assert got["data"]["oos_sharpe"] == 0.42


def test_get_missing_raises_not_found(db):
    # A well-formed but non-existent id → exit code 2.
    with pytest.raises(AxilError) as exc:
        db.get("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    assert exc.value.exit_code != 0


def test_delete_roundtrip(db):
    rec = db.store("autopsies", {"strategy": "gone"})
    out = db.delete(rec["id"])
    assert out["deleted"] is True
    assert out["id"] == rec["id"]
    # Second delete of the same id fails.
    with pytest.raises(AxilError):
        db.delete(rec["id"])


# ── query ───────────────────────────────────────────────────────────────────
def test_query_where_and_numeric_and_string(db):
    ids = {}
    ids["a"] = db.store("autopsies",
                        {"strategy": "mr-1", "family": "meanrev", "oos_sharpe": 0.42, "trades": 12})["id"]
    ids["b"] = db.store("autopsies",
                        {"strategy": "mr-2", "family": "meanrev", "oos_sharpe": 0.55, "trades": 100})["id"]
    db.store("autopsies",
             {"strategy": "mom-1", "family": "momentum", "oos_sharpe": 0.20, "trades": 5})

    rows = db.query("autopsies", where="oos_sharpe > 0.3 AND family = 'meanrev'")
    got = {r["id"] for r in rows}
    assert got == {ids["a"], ids["b"]}

    # Numeric (not lexicographic): trades < 30 matches 12 but not 100.
    rows = db.query("autopsies", where="trades < 30")
    strategies = {r["data"]["strategy"] for r in rows}
    assert "mr-1" in strategies and "mom-1" in strategies
    assert "mr-2" not in strategies  # trades=100


def test_query_order_by_and_limit(db):
    for s in (0.1, 0.9, 0.5):
        db.store("t", {"x": s})
    rows = db.query("t", order_by="x", limit=2)
    assert len(rows) == 2
    assert rows[0]["data"]["x"] == 0.1


# ── aggregation ─────────────────────────────────────────────────────────────
def test_agg_count_group_by(db):
    db.store("autopsies", {"kill_reason": "drawdown"})
    db.store("autopsies", {"kill_reason": "drawdown"})
    db.store("autopsies", {"kill_reason": "fees"})

    out = db.agg("autopsies", ["count"], group_by="kill_reason")
    assert out["group_by"] == "kill_reason"
    assert out["total_rows"] == 3
    by_group = {g["group"]: g["count"] for g in out["groups"]}
    assert by_group == {"drawdown": 2, "fees": 1}


def test_agg_avg_and_where(db):
    db.store("autopsies", {"family": "meanrev", "oos_sharpe": 0.40})
    db.store("autopsies", {"family": "meanrev", "oos_sharpe": 0.60})
    db.store("autopsies", {"family": "momentum", "oos_sharpe": 0.20})

    out = db.agg("autopsies", ["count", "avg(oos_sharpe)"], group_by="family")
    by_group = {g["group"]: g for g in out["groups"]}
    assert by_group["meanrev"]["avg_oos_sharpe"] == pytest.approx(0.50)
    assert by_group["momentum"]["avg_oos_sharpe"] == pytest.approx(0.20)

    # WHERE narrows the fold.
    out = db.agg("autopsies", ["count"], where="oos_sharpe >= 0.4")
    assert out["groups"][0]["count"] == 2


def test_agg_include_archived_changes_count(db):
    db.store("autopsies", {"strategy": "live"})
    db.store("autopsies", {"strategy": "dead", "_archived": True})

    default = db.agg("autopsies", ["count"])
    with_arch = db.agg("autopsies", ["count"], include_archived=True)
    assert default["total_rows"] == 1
    assert with_arch["total_rows"] == 2


def test_agg_metric_spec_forms(db):
    db.store("t", {"v": 2})
    db.store("t", {"v": 4})
    out = db.agg("t", ["min(v)", "max(v)", "sum(v)"])
    g = out["groups"][0]
    assert g["min_v"] == 2
    assert g["max_v"] == 4
    assert g["sum_v"] == 6
    with pytest.raises(ValueError):
        db.agg("t", ["max:v"])


# ── raw vectors (named space, lazily created) ───────────────────────────────
def test_store_vector_and_similar_by_id(db):
    a = db.store("fp", {"strategy": "A"},
                 vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], space="fp")
    assert a["vector_dims"] == 8
    assert a["space"] == "fp"
    b = db.store("fp", {"strategy": "B"},
                 vector=[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], space="fp")
    db.store("fp", {"strategy": "O"},
             vector=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], space="fp")

    # Near-dup detection: A's nearest above 0.9 is exactly B (self excluded).
    hits = db.similar(id=a["id"], space="fp", threshold=0.9, top_k=5)
    assert [h["id"] for h in hits] == [b["id"]]
    assert hits[0]["score"] >= 0.9


def test_similar_by_vector(db):
    a = db.store("fp", {"strategy": "A"},
                 vector=[1.0, 0.0, 0.0, 0.0], space="fp2")
    db.store("fp", {"strategy": "O"},
             vector=[0.0, 1.0, 0.0, 0.0], space="fp2")
    hits = db.similar(vector=[0.99, 0.01, 0.0, 0.0], space="fp2", top_k=1)
    assert hits[0]["id"] == a["id"]


def test_add_vector_to_named_space(db):
    rec = db.store("fp", {"strategy": "X"})
    out = db.add_vector(rec["id"], [1.0, 0.0, 0.0, 0.0], space="fp3")
    assert out["added"] is True
    assert out["dimensions"] == 4
    assert out["space"] == "fp3"
    hits = db.similar(id=rec["id"], space="fp3", top_k=5)
    assert isinstance(hits, list)  # only member of the space → no other hits


def test_similar_requires_exactly_one_of_vector_or_id(db):
    with pytest.raises(ValueError):
        db.similar()
    with pytest.raises(ValueError):
        db.similar(vector=[1.0], id="x")


def test_store_embed_and_vector_are_exclusive(db):
    with pytest.raises(ValueError):
        db.store("t", {"summary": "x"}, embed="summary", vector=[1.0])


# ── graph: link + lineage ───────────────────────────────────────────────────
def test_link_and_lineage_chain_with_deltas(db):
    a = db.store("trials", {"n": "A", "sharpe": 1.0, "trades": 10})["id"]
    b = db.store("trials", {"n": "B", "sharpe": 1.5, "trades": 20})["id"]
    c = db.store("trials", {"n": "C", "sharpe": 1.2, "trades": 35})["id"]

    e1 = db.link(a, "derived_from", b, props={"mutation": "widened stop"})
    assert e1["edge_type"] == "derived_from"
    assert e1["from"] == a and e1["to"] == b
    db.link(b, "derived_from", c, props={"mutation": "tighter entry"})

    out = db.lineage(a, fields=["sharpe", "trades"])
    assert out["root"] == a
    assert out["direction"] == "ancestors"
    hops = out["hops"]
    assert [h["id"] for h in hops] == [a, b, c]  # root-first
    assert hops[0]["delta"] == {}
    assert hops[1]["delta"]["sharpe"] == pytest.approx(0.5)
    assert hops[1]["delta"]["trades"] == pytest.approx(10.0)
    assert hops[2]["delta"]["sharpe"] == pytest.approx(-0.3)
    assert hops[1]["edge"]["props"]["mutation"] == "widened stop"


def test_lineage_descendants_returns_children(db):
    root = db.store("trials", {"n": "root"})["id"]
    x = db.store("trials", {"n": "x"})["id"]
    y = db.store("trials", {"n": "y"})["id"]
    db.link(x, "derived_from", root)
    db.link(y, "derived_from", root)

    out = db.lineage(root, direction="descendants")
    ids = {h["id"] for h in out["hops"]}
    assert {x, y}.issubset(ids)


# ── recall (needs an embedding model) ───────────────────────────────────────
def test_recall_roundtrip(db, tmp_path):
    if not _embeddings_available(AXIL_BIN, tmp_path / "probe.axil"):
        pytest.skip("no embedding model available on host")
    db.store("notes", {"summary": "auth timeout bug fixed in login flow"}, embed="summary")
    hits = db.recall("authentication timeout", top_k=3)
    assert isinstance(hits, list)
    assert any("auth" in (h.get("summary") or "") for h in hits)
