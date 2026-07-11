#!/usr/bin/env python3
"""LongMemEval retrieval driver for redis/agent-memory-server ("AMS").

Head-to-head counterpart to the in-tree Axil harness
(``benchmarks/longmemeval``). It measures the SAME metric on the SAME dataset:
retrieval recall@5 = "is every answer-bearing session in the top-5 retrieved
sessions?", averaged over the 500 LongMemEval-S questions.

Per question it:
  1. ingests each haystack session as ONE long-term memory (session-level
     granularity — the same unit the Axil harness indexes), isolated in a
     per-question namespace;
  2. waits for AMS's background indexer to finish;
  3. searches with the question text, collapses hits to unique session tags,
     takes the top-k;
  4. scores recall@k / precision@k against the answer-bearing sessions, using
     the identical definition the Axil harness uses (turns flagged
     ``has_answer`` -> session tag ``session_<index>``).

Endpoint shapes are verified against the AMS source; see README.md
"Verified sources". Default mode is extraction OFF (no generative LLM), which
matches Axil's no-LLM retrieval condition. Note that AMS still requires an
embedding provider at ingest/search time regardless of mode (see README).

NO NUMBERS FROM THIS SCRIPT MAY BE CITED until a run is committed to
``benchmarks/results/`` with environment details. See README "NO RESULTS YET".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - dependency guard
    sys.stderr.write(
        "error: the `requests` package is required. Install it with:\n"
        "    python -m pip install -r requirements.txt\n"
    )
    raise SystemExit(1)


# ── Dataset path convention (mirrors benchmarks/longmemeval/src/main.rs) ──────

VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

# The upstream download hint, quoted so a skipped run is self-explanatory.
DATASET_HINT = (
    "The LongMemEval-S dataset is not checked into git (~265 MB). Pull it once:\n"
    "    mkdir -p benchmarks/longmemeval/data\n"
    "    curl -L -o benchmarks/longmemeval/data/longmemeval_s_cleaned.json \\\n"
    "      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json\n"
    "This driver defaults to the SAME file the Axil harness reads, so the two "
    "sides run on identical bytes."
)


# ── AMS REST client (verified endpoints) ─────────────────────────────────────


class AmsClient:
    """Thin wrapper over the AMS long-term-memory REST endpoints.

    Every path here is verified against agent_memory_server/api.py + models.py
    (see README "Verified sources"). No endpoint is invented.
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> bool:
        # GET /v1/health -> {"now": <int>} (also the compose healthcheck path).
        resp = self.session.get(f"{self.base_url}/v1/health", timeout=self.timeout)
        return resp.status_code == 200

    def create_memories(self, memories: list[dict[str, Any]], deduplicate: bool) -> None:
        # POST /v1/long-term-memory/  body: CreateMemoryRecordRequest
        #   {"memories": [ExtractedMemoryRecord, ...], "deduplicate": bool}
        # Indexing is enqueued as a background task (Docket) -> returns an ack
        # immediately; the records are NOT searchable until the task-worker runs.
        payload = {"memories": memories, "deduplicate": deduplicate}
        resp = self.session.post(
            f"{self.base_url}/v1/long-term-memory/",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        # POST /v1/long-term-memory/search  body: SearchRequest
        # Response: MemoryRecordResultsResponse
        #   {"memories": [{..., "session_id", "namespace", "dist"}], "total": int}
        resp = self.session.post(
            f"{self.base_url}/v1/long-term-memory/search",
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_memories(self, ids: list[str]) -> None:
        # DELETE /v1/long-term-memory?memory_ids=a&memory_ids=b (repeated query param)
        for batch in _batched(ids, 100):
            resp = self.session.delete(
                f"{self.base_url}/v1/long-term-memory",
                params=[("memory_ids", mid) for mid in batch],
                timeout=self.timeout,
            )
            resp.raise_for_status()


# ── Dataset helpers ──────────────────────────────────────────────────────────


def session_text(turns: list[dict[str, Any]]) -> str:
    """Join a session's turns into one blob.

    Byte-for-byte the same rendering as the Axil harness's ``session_text``:
    ``"{role}: {content}"`` per turn, newline-joined. Keeping this identical is
    what makes the two systems index the same unit of the same text.
    """
    return "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)


def answer_session_tags(question: dict[str, Any]) -> list[str]:
    """Answer-bearing session tags, computed exactly as the Axil harness does.

    A session is answer-bearing iff any of its turns is flagged
    ``has_answer: true``. Tag = ``session_<0-based-index>``. We deliberately do
    NOT use the dataset's ``answer_session_ids`` field (a different naming
    scheme) — matching the Axil harness so both sides score identically.
    """
    tags: list[str] = []
    for idx, sess in enumerate(question.get("haystack_sessions", [])):
        if any(turn.get("has_answer") for turn in sess):
            tags.append(f"session_{idx}")
    return tags


def parse_haystack_date(raw: str) -> str | None:
    """Parse '2023/05/30 (Tue) 23:40' -> ISO 8601, else None.

    Same format string the Axil harness parses (``%Y/%m/%d (%a) %H:%M``). Only
    used to populate ``event_date`` so an optional recency-boosted run has the
    timestamps; irrelevant when ``--recency-boost`` is off (the default).
    """
    try:
        dt = datetime.strptime(raw, "%Y/%m/%d (%a) %H:%M")
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat()


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── Scoring (mirrors benchmarks/longmemeval/src/main.rs) ─────────────────────


@dataclass
class QuestionResult:
    question_id: str
    category: str
    question_date: str
    recall: float
    precision: float
    retrieved_sessions: list[str]
    answer_sessions: list[str]

    @property
    def is_hit(self) -> bool:
        return self.recall > 0.0

    @property
    def is_miss(self) -> bool:
        # A question with no answer-bearing sessions is scored recall=1.0 (see
        # score_question) and is never a miss — same as the Axil harness.
        return self.recall <= 0.0 and bool(self.answer_sessions)


def collapse_to_sessions(
    ranked_session_tags: list[str], top_k: int
) -> list[str]:
    """First-occurrence dedup of session tags in rank order, capped at top_k.

    AMS returns memories already ranked (ascending ``dist``). Because we ingest
    one memory per session, this collapse is usually 1:1, but we keep it so the
    top-k unit is a unique *session* — identical to the Axil harness's
    ``collapse_to_sessions``.
    """
    seen: list[str] = []
    for tag in ranked_session_tags:
        if tag not in seen:
            seen.append(tag)
        if len(seen) >= top_k:
            break
    return seen


def score_question(
    question: dict[str, Any],
    retrieved_sessions: list[str],
) -> QuestionResult:
    answers = answer_session_tags(question)
    qid = question.get("question_id", "")
    category = question.get("question_type", "")
    qdate = question.get("question_date", "")

    if not answers:
        # No answer-bearing session in the haystack: the Axil harness treats
        # this as recall=1.0, precision=0.0 (nothing to retrieve). Mirror it so
        # aggregate numbers line up.
        return QuestionResult(qid, category, qdate, 1.0, 0.0, retrieved_sessions, answers)

    hits = sum(1 for a in answers if a in retrieved_sessions)
    recall = hits / len(answers)
    precision = (
        sum(1 for r in retrieved_sessions if r in answers) / len(retrieved_sessions)
        if retrieved_sessions
        else 0.0
    )
    return QuestionResult(qid, category, qdate, recall, precision, retrieved_sessions, answers)


# ── Per-question run ─────────────────────────────────────────────────────────


@dataclass
class Config:
    server_url: str
    variant: str
    limit: int
    top_k: int
    search_mode: str
    hybrid_alpha: float
    recency_boost: bool
    extraction: str
    namespace_prefix: str
    index_timeout: float
    poll_interval: float
    batch_size: int
    cleanup: bool
    request_timeout: float
    distance_threshold: float | None
    verbose: bool


def ingest_question(
    client: AmsClient, cfg: Config, question: dict[str, Any], namespace: str
) -> tuple[list[str], int]:
    """Create one long-term memory per haystack session. Returns (ids, count)."""
    memories: list[dict[str, Any]] = []
    ids: list[str] = []
    dates = question.get("haystack_dates", [])
    for idx, turns in enumerate(question.get("haystack_sessions", [])):
        text = session_text(turns)
        if not text.strip():
            # AMS rejects empty text; skip rather than fail the whole question.
            continue
        mem_id = f"{namespace}__session_{idx}"
        record: dict[str, Any] = {
            "id": mem_id,
            "text": text,
            "session_id": f"session_{idx}",  # the tag we score against
            "namespace": namespace,
            "memory_type": "semantic",
            # Mark as already-extracted so the ingest path stores the text
            # verbatim and does not attempt any further LLM extraction.
            "discrete_memory_extracted": "t",
        }
        event_date = parse_haystack_date(dates[idx]) if idx < len(dates) else None
        if event_date is not None:
            record["event_date"] = event_date
        memories.append(record)
        ids.append(mem_id)

    for batch in _batched(memories, cfg.batch_size):
        # deduplicate=False: preserve a 1:1 session->memory mapping. Dedup could
        # collapse near-identical sessions and break session-level scoring.
        client.create_memories(batch, deduplicate=False)

    return ids, len(memories)


def wait_until_indexed(
    client: AmsClient, cfg: Config, namespace: str, expected: int
) -> int:
    """Poll a filter-only search until >= expected memories are indexed.

    AMS indexes long-term memories on a background queue, so a search issued
    immediately after create would see a partial index. We poll the namespace's
    total until it reaches the expected count or the timeout elapses. Returns
    the last observed indexed count.
    """
    deadline = time.monotonic() + cfg.index_timeout
    last = 0
    while time.monotonic() < deadline:
        try:
            # No `text` and no `search_mode` -> a pure filter-only query: AMS
            # returns memories matching the namespace filter and a `total`
            # count, with no query embedding computed (nothing to embed). This
            # is the compatibility-safe way to count what's indexed so far.
            body = {"namespace": {"eq": namespace}, "limit": 1}
            resp = client.search(body)
            last = int(resp.get("total", 0))
        except requests.RequestException:
            last = 0
        if last >= expected:
            return last
        time.sleep(cfg.poll_interval)
    return last


def search_question(
    client: AmsClient, cfg: Config, question: dict[str, Any], namespace: str
) -> list[str]:
    # Fetch more than top_k, then collapse to unique sessions — same "over-fetch
    # then collapse" the Axil harness uses (fetch_k = top_k*8, capped to AMS's
    # max limit of 100).
    fetch_k = min(max(cfg.top_k * 8, 40), 100)
    body: dict[str, Any] = {
        "text": question.get("question", ""),
        "namespace": {"eq": namespace},
        "search_mode": cfg.search_mode,
        "limit": fetch_k,
    }
    if cfg.search_mode == "hybrid":
        body["hybrid_alpha"] = cfg.hybrid_alpha
    # Recency re-ranking off by default => pure similarity/lexical ranking, the
    # closest analog to Axil's `vector`/`similar_to` strategy. Turning it on
    # approximates Axil's recall-qtc (recency-blended) condition instead.
    body["recency_boost"] = cfg.recency_boost
    if cfg.distance_threshold is not None:
        body["distance_threshold"] = cfg.distance_threshold

    resp = client.search(body)
    ranked_tags: list[str] = []
    for mem in resp.get("memories", []):
        tag = mem.get("session_id")
        if isinstance(tag, str):
            ranked_tags.append(tag)
    return collapse_to_sessions(ranked_tags, cfg.top_k)


# ── Aggregation + report ─────────────────────────────────────────────────────


@dataclass
class CategoryStats:
    total: int = 0
    hits: int = 0
    recall_sum: float = 0.0
    precision_sum: float = 0.0


@dataclass
class Report:
    results: list[QuestionResult] = field(default_factory=list)

    def to_json(self, cfg: Config, embedding_note: str) -> dict[str, Any]:
        n = len(self.results)
        by_cat: dict[str, CategoryStats] = {}
        total_recall = total_precision = 0.0
        total_hits = 0
        misses: list[dict[str, Any]] = []
        for r in self.results:
            total_recall += r.recall
            total_precision += r.precision
            if r.is_hit:
                total_hits += 1
            cat = by_cat.setdefault(r.category, CategoryStats())
            cat.total += 1
            cat.hits += 1 if r.is_hit else 0
            cat.recall_sum += r.recall
            cat.precision_sum += r.precision
            if r.is_miss:
                misses.append(
                    {
                        "question_id": r.question_id,
                        "question_type": r.category,
                        "question_date": r.question_date,
                        "recall": r.recall,
                        "precision": r.precision,
                        "retrieved_sessions": r.retrieved_sessions,
                        "answer_sessions": r.answer_sessions,
                    }
                )
        return {
            # Same top-level keys as benchmarks/results/qtc-500.json so the two
            # reports diff cleanly. `strategy` names the AMS condition.
            "benchmark": "LongMemEval",
            "variant": cfg.variant,
            "strategy": f"ams-{cfg.search_mode}",
            "rerank": "off",
            "top_k": cfg.top_k,
            "total_questions": n,
            "overall": {
                "hit_rate": (total_hits / n) if n else 0.0,
                "avg_recall": (total_recall / n) if n else 0.0,
                "avg_precision": (total_precision / n) if n else 0.0,
            },
            "by_category": {
                cat: {
                    "total": s.total,
                    "hits": s.hits,
                    "hit_rate": (s.hits / s.total) if s.total else 0.0,
                    "avg_recall": (s.recall_sum / s.total) if s.total else 0.0,
                    "avg_precision": (s.precision_sum / s.total) if s.total else 0.0,
                }
                for cat, s in by_cat.items()
            },
            "misses": misses,
            # AMS-specific run metadata (not in the Axil report). Records the
            # exact conditions so a committed result is reproducible + auditable.
            "run_meta": {
                "system": "redis/agent-memory-server",
                "server_url": cfg.server_url,
                "search_mode": cfg.search_mode,
                "hybrid_alpha": cfg.hybrid_alpha if cfg.search_mode == "hybrid" else None,
                "recency_boost": cfg.recency_boost,
                "extraction": cfg.extraction,
                "deduplicate": False,
                "distance_threshold": cfg.distance_threshold,
                "ingest_granularity": "one long-term memory per haystack session",
                "embedding_note": embedding_note,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LongMemEval retrieval driver for redis/agent-memory-server."
    )
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/data"),
        help="Defaults to the SAME dir the Axil harness reads (identical bytes).",
    )
    p.add_argument("--variant", choices=sorted(VARIANT_FILES), default="s")
    p.add_argument("--limit", type=int, default=0, help="Max questions (0 = all).")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--search-mode",
        choices=["semantic", "keyword", "hybrid"],
        default="semantic",
        help="semantic = vector (closest analog to Axil vector recall; needs an "
        "embedding provider). keyword = Redis full-text (lexical; different "
        "condition). hybrid = both.",
    )
    p.add_argument("--hybrid-alpha", type=float, default=0.7)
    p.add_argument(
        "--recency-boost",
        action="store_true",
        help="Enable AMS recency re-ranking (default off = pure similarity).",
    )
    p.add_argument(
        "--extraction",
        choices=["off", "on"],
        default="off",
        help="Records/labels the condition. 'off' matches Axil's no-LLM "
        "condition and requires the server to be started with extraction "
        "flags False (see docker-compose.yml). This flag does NOT toggle the "
        "server; it asserts intent and stamps the report.",
    )
    p.add_argument("--distance-threshold", type=float, default=None)
    p.add_argument("--namespace-prefix", default="lme")
    p.add_argument("--index-timeout", type=float, default=120.0)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep ingested memories after each question (default deletes them).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results JSON path. Default: out/ams-<variant>-<mode>-<N>.json "
        "(out/ is gitignored; promote to benchmarks/results/ manually per policy).",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def skip_loud(data_file: Path) -> None:
    """Loud, CI-friendly skip when the dataset is absent (exit 0 = skip).

    Mirrors the in-tree dataset-gated harnesses: a missing dataset is a LOUD
    skip, never a silent pass. A green run here means nothing was verified.
    """
    bar = "=" * 72
    sys.stderr.write(
        f"\n{bar}\n"
        "::warning::LongMemEval dataset NOT FOUND -- AMS comparison SKIPPED "
        "(nothing was measured).\n"
        f"{bar}\n"
        f"Expected dataset file: {data_file}\n\n"
        f"{DATASET_HINT}\n"
        f"{bar}\n\n"
    )
    raise SystemExit(0)


def wait_for_server(client: AmsClient, attempts: int = 30, delay: float = 2.0) -> None:
    for _ in range(attempts):
        try:
            if client.health():
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    sys.stderr.write(
        "error: AMS server did not become healthy at "
        f"{client.base_url}/v1/health. Is the stack up? "
        "(`docker compose up -d` then `docker compose logs -f api`)\n"
    )
    raise SystemExit(1)


def embedding_condition_note(cfg: Config) -> str:
    if cfg.search_mode == "keyword":
        return (
            "keyword (Redis full-text) search: query is NOT embedded. NOTE: AMS "
            "still embeds each memory at INGEST to populate its vector index, so "
            "an embedding provider is required to ingest regardless of mode."
        )
    return (
        "semantic/hybrid search embeds both memories (at ingest) and the query "
        "via the server's configured EMBEDDING_MODEL (default OpenAI "
        "text-embedding-3-small, 1536-dim). This is a HOSTED provider requiring "
        "OPENAI_API_KEY -- an inherent asymmetry vs Axil's in-process bge-small."
    )


def main() -> None:
    args = build_parser().parse_args()

    data_file = args.data_dir / VARIANT_FILES[args.variant]
    if not data_file.is_file():
        skip_loud(data_file)

    cfg = Config(
        server_url=args.server_url,
        variant=args.variant,
        limit=args.limit,
        top_k=args.top_k,
        search_mode=args.search_mode,
        hybrid_alpha=args.hybrid_alpha,
        recency_boost=args.recency_boost,
        extraction=args.extraction,
        namespace_prefix=args.namespace_prefix,
        index_timeout=args.index_timeout,
        poll_interval=args.poll_interval,
        batch_size=args.batch_size,
        cleanup=not args.no_cleanup,
        request_timeout=args.request_timeout,
        distance_threshold=args.distance_threshold,
        verbose=args.verbose,
    )

    sys.stderr.write(
        "LongMemEval x redis/agent-memory-server\n"
        f"  variant={cfg.variant} search_mode={cfg.search_mode} "
        f"recency_boost={cfg.recency_boost} extraction={cfg.extraction} "
        f"top_k={cfg.top_k}\n"
        f"  server={cfg.server_url} data={data_file}\n"
    )
    if cfg.extraction == "on":
        sys.stderr.write(
            "  note: extraction=on measures an LLM-ASSISTED condition, NOT the "
            "no-LLM retrieval baseline. Start the server with the extraction "
            "flags True and label the result accordingly.\n"
        )

    with data_file.open("r", encoding="utf-8") as fh:
        questions: list[dict[str, Any]] = json.load(fh)
    total = len(questions) if cfg.limit == 0 else min(len(questions), cfg.limit)
    sys.stderr.write(f"  {len(questions)} questions loaded, evaluating {total}\n\n")

    client = AmsClient(cfg.server_url, timeout=cfg.request_timeout)
    wait_for_server(client)

    report = Report()
    for i, question in enumerate(questions[:total]):
        namespace = f"{cfg.namespace_prefix}_{cfg.variant}_{question.get('question_id', i)}"
        ids: list[str] = []
        try:
            ids, expected = ingest_question(client, cfg, question, namespace)
            if expected == 0:
                retrieved: list[str] = []
            else:
                indexed = wait_until_indexed(client, cfg, namespace, expected)
                if indexed < expected:
                    sys.stderr.write(
                        f"\n  warning: {namespace} indexed {indexed}/{expected} "
                        "before timeout; scoring on the partial index.\n"
                    )
                retrieved = search_question(client, cfg, question, namespace)
        except requests.RequestException as exc:
            sys.stderr.write(f"\n  request error on {namespace}: {exc}\n")
            retrieved = []
        finally:
            if cfg.cleanup and ids:
                try:
                    client.delete_memories(ids)
                except requests.RequestException as exc:
                    sys.stderr.write(f"\n  cleanup error on {namespace}: {exc}\n")

        result = score_question(question, retrieved)
        report.results.append(result)

        done = i + 1
        if cfg.verbose:
            sys.stderr.write(
                f"  {result.question_id} [{result.category}] "
                f"recall={result.recall:.2f} precision={result.precision:.2f} "
                f"{'HIT' if result.is_hit else 'MISS'}\n"
            )
        elif done == 1 or done % 10 == 0 or done == total:
            sys.stderr.write(f"\r  {done}/{total}")
            sys.stderr.flush()

    sys.stderr.write(f"\r  {total}/{total} done.\n\n")

    doc = report.to_json(cfg, embedding_condition_note(cfg))

    out_path = args.out
    if out_path is None:
        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ams-{cfg.variant}-{cfg.search_mode}-{total}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    overall = doc["overall"]
    sys.stderr.write(
        f"overall: hit_rate={overall['hit_rate']:.4f} "
        f"avg_recall={overall['avg_recall']:.4f} "
        f"avg_precision={overall['avg_precision']:.4f}\n"
        f"results written to {out_path}\n"
        "\nREMINDER: this number is NOT citable until the JSON is committed to "
        "benchmarks/results/ with environment details (see README).\n"
    )


if __name__ == "__main__":
    main()
