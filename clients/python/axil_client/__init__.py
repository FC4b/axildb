"""axil-client — a thin, pure-stdlib Python wrapper around the ``axil`` CLI.

This package shells out to the ``axil`` binary for every operation. There is no
FFI, no PyO3, and no network layer: each method builds an argument list, runs
the binary with ``--format json --quiet``, and parses the single line of JSON
the CLI prints to stdout. All diagnostic chatter (model-provider notices, index
warnings, the recall "compact view" hint) is written by the CLI to stderr, so
stdout stays clean JSON.

Example
-------
>>> from axil_client import Axil
>>> db = Axil("./memory.axil")                    # doctest: +SKIP
>>> rec = db.store("autopsies", {"strategy": "mr-1", "oos_sharpe": 0.42})
>>> db.recall("mean reversion drawdown", top_k=3)  # doctest: +SKIP

See ``README.md`` for the full trading-R&D-loop walkthrough.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["Axil", "AxilError"]
__version__ = "0.1.0"

JsonValue = Any
PathLike = Union[str, "os.PathLike[str]"]


class AxilError(RuntimeError):
    """Raised when the ``axil`` binary exits with a non-zero status.

    Attributes
    ----------
    exit_code:
        The process exit code. The CLI uses ``0`` for success, ``1`` for a
        generic error, and ``2`` for "not found" (e.g. ``get``/``delete`` of a
        missing id).
    stderr:
        The full text the binary wrote to stderr, stripped of trailing
        whitespace. The CLI reports failures as a JSON object on stderr, e.g.
        ``{"error": "not found", "id": "..."}``.
    command:
        The exact argument list that was executed (for debugging).
    """

    def __init__(self, exit_code: int, stderr: str, command: Sequence[str]):
        self.exit_code = exit_code
        self.stderr = (stderr or "").strip()
        self.command = list(command)
        super().__init__(
            f"axil exited with code {exit_code}: {self.stderr or '<no stderr>'}"
        )


# count() takes no field; the rest require exactly one field.
_METRIC_RE = re.compile(r"^\s*(count|avg|min|max|sum)\s*(?:\(\s*([^)]*?)\s*\))?\s*$", re.I)


class Axil:
    """A subprocess-backed client for a single ``.axil`` database.

    Parameters
    ----------
    db_path:
        Path to the ``.axil`` database file. Passed to the CLI as ``--db``.
        The parent database is created lazily by write commands; graph-backed
        commands (``link``/``lineage``) and default-space vectors require the
        stores to exist first — run ``axil init <path>`` once, or attach a
        vector via a named ``space`` (which is created lazily).
    binary:
        Name or path of the ``axil`` executable. Defaults to ``"axil"`` and is
        resolved via ``PATH`` (Windows appends ``.exe`` automatically). Pass an
        absolute path to pin a specific build.
    """

    def __init__(self, db_path: PathLike, binary: str = "axil") -> None:
        self.db_path = str(db_path)
        self.binary = binary

    # ── internal plumbing ────────────────────────────────────────────────
    def _run(self, args: Sequence[str]) -> JsonValue:
        """Execute a subcommand and return the parsed stdout JSON.

        Global flags (``--db``, ``--format json``, ``--quiet``) are prepended
        automatically. Raises :class:`AxilError` on a non-zero exit.
        """
        cmd = [
            self.binary,
            "--db",
            self.db_path,
            "--format",
            "json",
            "--quiet",
            *args,
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,  # never invoke a shell: args are passed verbatim
        )
        if proc.returncode != 0:
            raise AxilError(proc.returncode, proc.stderr, cmd)
        out = (proc.stdout or "").strip()
        if not out:
            return None
        return json.loads(out)

    # ── record CRUD ──────────────────────────────────────────────────────
    def store(
        self,
        table: str,
        data: Dict[str, Any],
        embed: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        space: Optional[str] = None,
    ) -> JsonValue:
        """Insert a record into ``table``.

        Parameters
        ----------
        table:
            Destination table name.
        data:
            The record body; serialized to JSON and passed as the positional
            argument.
        embed:
            Comma-separated field names to auto-embed after insert (requires an
            embedding model on the host). Mutually exclusive with ``vector``.
        vector:
            A pre-computed raw vector (list of floats) to attach in one shot.
            Mutually exclusive with ``embed``.
        space:
            Named vector space for ``vector`` (``[a-z0-9_-]{1,32}``). Omit for
            the default text-embedding space. Named spaces are created lazily
            at the supplied vector's dimension.

        Returns
        -------
        dict
            ``{"id", "table", "created_at"[, "vector_dims", "space"]}``.
        """
        if embed is not None and vector is not None:
            raise ValueError("`embed` and `vector` are mutually exclusive")
        args: List[str] = ["store", table, json.dumps(data)]
        if embed is not None:
            args += ["--embed", embed]
        if vector is not None:
            args += ["--vector", json.dumps(list(vector))]
        if space is not None:
            args += ["--space", space]
        return self._run(args)

    def get(self, id: str) -> JsonValue:
        """Fetch a record by id.

        Returns the record object ``{"id", "table", "data", "created_at",
        "updated_at"}``. Raises :class:`AxilError` (exit code 2) if no record
        with that id exists.
        """
        return self._run(["get", id])

    def delete(self, id: str) -> JsonValue:
        """Delete a record by id.

        Returns ``{"deleted": True, "id": ...}``. Raises :class:`AxilError`
        (exit code 2) if no record with that id exists.
        """
        return self._run(["delete", id])

    # ── recall & query ───────────────────────────────────────────────────
    def recall(self, query: str, top_k: int = 5) -> List[JsonValue]:
        """Recency-weighted semantic search (requires an embedding model).

        Returns a list of compact hits ``{"id", "score", "summary", "table"}``.
        Flags go before the ``--`` separator so free-text queries that start
        with ``-`` (e.g. ``"-1.5% drawdown"``) are not parsed as flags.
        """
        return self._run(["recall", "--top-k", str(top_k), "--", query])

    def query(
        self,
        table: str,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[JsonValue]:
        """Filter records in ``table``.

        Parameters
        ----------
        where:
            A single predicate string; several conditions may be joined by
            ``AND`` (case-insensitive), with quoted string values, e.g.
            ``"oos_sharpe > 0.3 AND family = 'meanrev'"``. Operators:
            ``= != > < >= <=`` and ``contains``. Numbers compare numerically.
        order_by:
            Field to sort by (ascending).
        limit:
            Maximum number of results.

        Returns a list of record objects.
        """
        args: List[str] = ["query", table]
        if where is not None:
            args += ["--where", where]
        if order_by is not None:
            args += ["--order-by", order_by]
        if limit is not None:
            args += ["--limit", str(limit)]
        return self._run(args)

    def agg(
        self,
        table: str,
        metrics: Sequence[str],
        group_by: Optional[str] = None,
        where: Optional[str] = None,
        include_archived: bool = False,
    ) -> JsonValue:
        """Aggregate ``table`` with count/avg/min/max/sum, optionally grouped.

        Parameters
        ----------
        metrics:
            A sequence of metric specs. Each is either ``"count"`` or a
            function form ``"avg(field)"``, ``"min(field)"``, ``"max(field)"``,
            ``"sum(field)"``. Repeated
            metrics combine.
        group_by:
            Field whose value keys the groups (missing field → ``null`` group).
        where:
            Predicate string using the same syntax as :meth:`query`.
        include_archived:
            Count archived/discarded records too (excluded by default).

        Returns
        -------
        dict
            ``{"table", "group_by", "groups": [{"group", "count",
            "avg_<field>", ..., "skipped"}], "total_rows"}``.
        """
        args: List[str] = ["agg", table]
        for spec in metrics:
            args += _metric_to_flags(spec)
        if group_by is not None:
            args += ["--group-by", group_by]
        if where is not None:
            args += ["--where", where]
        if include_archived:
            args += ["--include-archived"]
        return self._run(args)

    # ── raw vectors ──────────────────────────────────────────────────────
    def add_vector(
        self,
        id: str,
        vector: Sequence[float],
        space: Optional[str] = None,
    ) -> JsonValue:
        """Attach a raw vector to an existing record.

        The default space must already exist at a matching dimension (create it
        with ``axil init``); a named ``space`` is created lazily at the
        supplied vector's dimension. Returns ``{"added": True, "id",
        "dimensions"[, "space"]}``.
        """
        args: List[str] = ["add-vector", id, json.dumps(list(vector))]
        if space is not None:
            args += ["--space", space]
        return self._run(args)

    def similar(
        self,
        vector: Optional[Sequence[float]] = None,
        id: Optional[str] = None,
        space: Optional[str] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[JsonValue]:
        """Find records with vectors similar to a query vector or record.

        Provide exactly one of ``vector`` (a raw query vector) or ``id`` (whose
        stored vector is used, excluding the record itself). ``threshold``
        filters to results scoring ``>=`` the given cosine similarity — use e.g.
        ``0.95`` for near-duplicate detection. ``space`` targets a named space.

        Returns a list of ``{"id", "score", "data", "table", "created_at"}``.
        """
        if (vector is None) == (id is None):
            raise ValueError("provide exactly one of `vector` or `id`")
        args: List[str] = ["similar"]
        if vector is not None:
            args += ["--vector", json.dumps(list(vector))]
        if id is not None:
            args += ["--id", id]
        if space is not None:
            args += ["--space", space]
        args += ["--top-k", str(top_k)]
        if threshold is not None:
            args += ["--threshold", str(threshold)]
        return self._run(args)

    # ── graph ────────────────────────────────────────────────────────────
    def link(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> JsonValue:
        """Create a graph edge ``from_id --edge_type--> to_id``.

        ``props`` is an optional JSON object stored on the edge. Requires a
        graph store (create it with ``axil init``). Returns ``{"edge_id",
        "from", "to", "edge_type"}``.
        """
        args: List[str] = ["link", from_id, edge_type, to_id]
        if props is not None:
            args += ["--props", json.dumps(props)]
        return self._run(args)

    def lineage(
        self,
        id: str,
        direction: str = "ancestors",
        edge_type: str = "derived_from",
        max_depth: int = 20,
        fields: Optional[Sequence[str]] = None,
    ) -> JsonValue:
        """Walk a derivation chain over ``edge_type`` edges from ``id``.

        Parameters
        ----------
        direction:
            ``"ancestors"`` (follow OUT edges: what each node derived from,
            root-first), ``"descendants"`` (follow IN edges: what derived from
            the node), or ``"both"``.
        edge_type:
            Edge label to follow (default ``"derived_from"``).
        max_depth:
            Maximum hops from the root.
        fields:
            Record-data keys to include per hop (and diff for numeric deltas).
            Omit to include all fields.

        Returns
        -------
        dict
            ``{"root", "direction", "edge_type", "hops": [{"depth", "id",
            "table", "fields", "edge", "delta"}]}``. A missing edge endpoint is
            reported as a ``{"missing": true}`` hop rather than an error.

        Notes
        -----
        Create lineage at store time with::

            axil link <child> derived_from <parent> --props '{"mutation": "..."}'
        """
        args: List[str] = [
            "lineage",
            id,
            "--direction",
            direction,
            "--edge-type",
            edge_type,
            "--max-depth",
            str(max_depth),
        ]
        if fields:
            args += ["--fields", ",".join(fields)]
        return self._run(args)


def _metric_to_flags(spec: str) -> List[str]:
    """Translate one aggregation metric spec into CLI flags.

    Accepts ``"count"`` and ``"avg(field)"`` / ``"min(field)"`` /
    ``"max(field)"`` / ``"sum(field)"`` — the same spelling every other Axil
    surface (CLI, MCP, AxilQL) uses.
    """
    m = _METRIC_RE.match(spec)
    if not m:
        raise ValueError(f"invalid metric spec: {spec!r}")
    func = m.group(1).lower()
    field = m.group(2)
    if func == "count":
        return ["--count"]
    if not field:
        raise ValueError(f"metric {func!r} requires a field, e.g. '{func}(oos_sharpe)'")
    return [f"--{func}", field]
