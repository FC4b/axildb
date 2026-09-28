#!/usr/bin/env python3
"""Dogfood recall eval: questions about this repo's own Axil memory.

Each question in questions.jsonl names the records that answer it, as groups
of equivalent ids (a group is found when any one of its ids is). The eval runs
the real `axil recall` CLI, the path agents and hooks use, against a copy of
the database, so it never writes to the live memory. The copy is healed
(`axil heal --reindex`) first, so a record a past bug left without an
embedding competes like the rest. Hits created after the cutoff are dropped,
so records written since the questions don't compete.

Two views per question:
  ranked  `axil recall --recall-format full` (or `axil ask` with --via ask),
          for hit@k, MRR, recall_all, NDCG.
  hook    `axil recall --recall-format context-block --budget 2000 --top-k 5`,
          exactly what the prompt hook injects: was an answer in it, and how
          many tokens (bytes / 4) did it cost.

Recall weighs recency, so compare runs made close together in time. For a
before/after pair, pass the same --snapshot directory to both runs: the first
creates it (a healed copy of the database), and both then read an identical
corpus, which removes the drift from memories stored between the runs.

With --dump-context, each question's hook block (what the agent actually gets)
and the text of its expected records are written as JSONL for end-to-end QA
(benchmarks/e2e-qa/qa.py).

Usage:
  python3 benchmarks/dogfood-recall/run.py [--db .axil/memory.axil]
      [--axil axil] [--top-k 10] [--out results.json] [--dump-context qa.jsonl]
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUTOFF = "2026-09-28T02:00:00Z"
FETCH = 25  # over-fetch so dropping post-cutoff hits still leaves top-k


def copy_db(db: Path, dest: Path) -> Path:
    """Copy the core file and every companion (`memory.axil.*`)."""
    for src in db.parent.glob(db.name + "*"):
        target = dest / src.name
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    return dest / db.name


def run_axil(axil: str, db: Path, args: list) -> str:
    env = dict(os.environ, AXIL_SLOW_QUERY_LOG="0")
    out = subprocess.run(
        [axil, "--db", str(db), *args],
        capture_output=True, text=True, env=env, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"axil {' '.join(args[:2])} failed: {out.stderr.strip()[-300:]}")
    return out.stdout


def ranked_ids(axil: str, db: Path, question: str, top_k: int, via: str) -> list:
    if via == "ask":
        hits = json.loads(run_axil(axil, db, ["ask", question, "--top-k", str(FETCH)]) or "{}")
        hits = hits.get("results", [])
    else:
        hits = json.loads(run_axil(axil, db, [
            "recall", question, "--top-k", str(FETCH), "--recall-format", "full",
        ]) or "[]")
    kept = [h["id"] for h in hits if (h.get("created_at") or "") <= CUTOFF]
    return kept[:top_k]


def hook_block(axil: str, db: Path, question: str) -> str:
    return run_axil(axil, db, [
        "recall", question, "--recall-format", "context-block",
        "--budget", "2000", "--top-k", "5",
    ])


def score(expect: list, ranked: list, block: str, top_k: int) -> dict:
    def first_rank(group):
        ranks = [ranked.index(i) for i in group if i in ranked]
        return min(ranks) if ranks else None

    ranks = [first_rank(g) for g in expect]
    found = [r for r in ranks if r is not None]
    best = min(found) if found else None
    dcg = sum(1.0 / math.log2(r + 2) for r in found)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(expect), top_k)))
    block_ids = set(re.findall(r"id=([0-9A-Z]{26})", block))
    return {
        "first_rank": None if best is None else best + 1,
        "hit@1": float(best is not None and best < 1),
        "hit@5": float(best is not None and best < 5),
        "hit@10": float(best is not None and best < 10),
        "mrr@10": 0.0 if best is None else 1.0 / (best + 1),
        "recall_all@10": float(len(found) == len(expect)),
        "ndcg@10": dcg / ideal if ideal else 0.0,
        "hook_hit": float(any(i in block_ids for g in expect for i in g)),
        "hook_tokens": len(block.encode()) / 4,
    }


def reference_text(axil: str, db: Path, record_id: str) -> str:
    """An expected record as the judge sees it: its main text plus the
    fields that explain it (why, root cause, fix)."""
    data = json.loads(run_axil(axil, db, ["get", record_id]))["data"]
    parts = [data.get(k) for k in ("summary", "error", "rule", "subject", "reason", "root_cause", "fix", "resolution")]
    return " | ".join(p for p in parts if isinstance(p, str) and p.strip())[:1200]


METRICS = ["hit@1", "hit@5", "hit@10", "mrr@10", "recall_all@10", "ndcg@10", "hook_hit", "hook_tokens"]


def summarize(rows: list) -> dict:
    n = len(rows)
    out = {m: sum(r[m] for r in rows) / n for m in METRICS}
    tokens = out["hook_tokens"]
    out["hook_hits_per_1k_tokens"] = out["hook_hit"] / tokens * 1000 if tokens else 0.0
    out["n"] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=".axil/memory.axil")
    ap.add_argument("--axil", default="axil")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--dump-context")
    ap.add_argument("--snapshot", help="reuse (or create) a frozen, healed copy of the DB here")
    ap.add_argument("--via", choices=["recall", "ask"], default="recall",
                    help="command for the ranked view (the hook view is always recall)")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(HERE / "questions.jsonl") if l.strip()]
    version = subprocess.run([args.axil, "--version"], capture_output=True, text=True).stdout.strip()

    rows = []
    contexts = []
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(args.db).resolve()
        if args.snapshot:
            snap = Path(args.snapshot)
            if not (snap / source.name).exists():
                snap.mkdir(parents=True, exist_ok=True)
                run_axil(args.axil, copy_db(source, snap), ["heal", "--reindex"])
            db = copy_db(snap / source.name, Path(tmp))
        else:
            db = copy_db(source, Path(tmp))
            run_axil(args.axil, db, ["heal", "--reindex"])
        for i, q in enumerate(questions, 1):
            ranked = ranked_ids(args.axil, db, q["question"], args.top_k, args.via)
            block = hook_block(args.axil, db, q["question"])
            row = {"id": q["id"], "kind": q["kind"], **score(q["expect"], ranked, block, args.top_k)}
            rows.append(row)
            if args.dump_context:
                contexts.append({
                    "question_id": q["id"],
                    "question_type": q["kind"],
                    "question": q["question"],
                    "reference": [reference_text(args.axil, db, g[0]) for g in q["expect"]],
                    "hook": block,
                })
            print(f"\r  {i}/{len(questions)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    report = {
        "benchmark": "dogfood-recall",
        "via": args.via,
        "axil": version,
        "cutoff": CUTOFF,
        "top_k": args.top_k,
        "overall": summarize(rows),
        "by_kind": {k: summarize(v) for k, v in sorted(by_kind.items())},
        "per_question": [{k: r[k] for k in ("id", "first_rank", "hook_hit")} for r in rows],
    }
    if args.dump_context:
        Path(args.dump_context).write_text("".join(json.dumps(c) + "\n" for c in contexts))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
