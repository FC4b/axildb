#!/usr/bin/env python3
"""code-recall-measure.py — mechanical retrieval recall@5 on the context-ab tasks.

For every task in the committed context-ab fixtures, run
`axil code-search "<task text>" --top-k 5 --json` in that corpus's indexed
sandbox and check whether the ground-truth answer FILE appears among the
returned pointer paths. No agent in the loop: the query is the task text
verbatim, the hit test is a normalized path comparison against the fixture's
primary ground-truth file (alternates in prose notes are NOT credited — the
figure is conservative).

Also records an `fts@5` diagnostic per task (same protocol via `axil fts`),
since disciplined agents use both cheap lookups.

Usage:
  python scripts/code-recall-measure.py \
      [--exp-root experiments/context-ab] [--out benchmarks/results/context-ab/code-recall-<date>.json]
"""
import argparse
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXIL = os.environ.get("AXIL_BIN", os.path.join(ROOT, "target", "release", "axil"))
CORPORA = ["flask", "fastapi", "django"]


import re


def norm(p: str) -> str:
    # JSON-escaped Windows paths (`fastapi\\routing.py`) must collapse to a
    # single separator or substring matching silently fails.
    p = p.replace("\\", "/")
    p = re.sub(r"/+", "/", p)
    return p.strip().lstrip("./").lower()


def paths_from_code_search(cwd: str, query: str, top_k: int = 5):
    out = subprocess.run(
        [AXIL, "code-search", query, "--top-k", str(top_k), "--json"],
        cwd=cwd, capture_output=True, text=True, timeout=180,
    )
    try:
        hits = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return [norm(h.get("path", "")) for h in hits if isinstance(h, dict)]


def paths_from_fts(cwd: str, query: str, limit: int = 5):
    out = subprocess.run(
        [AXIL, "fts", query, "--limit", str(limit)],
        cwd=cwd, capture_output=True, text=True, timeout=180,
    )
    try:
        hits = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    paths = []
    for h in hits:
        data = h.get("data", {}) if isinstance(h, dict) else {}
        p = data.get("path", "")
        if p:
            paths.append(norm(p))
    return paths


def replay_agent_queries(exp_root, manifests, top_k):
    """Recall via the queries real agents issued (recorded in the A/B manifests).

    For each task, re-run every `axil` command the withdb agent consulted and
    check whether the ground-truth answer file appears in any command's stdout.
    Tasks where the agent recorded no lookups (agent failure) are excluded and
    reported separately — they are agent failures, not retrieval misses.
    """
    import shlex
    per_corpus = []
    for corpus, manifest_path in manifests.items():
        sandbox = os.path.join(exp_root, "withdb", corpus)
        tasks = json.load(open(manifest_path))["tasks"]
        rows, skipped = [], 0
        for t in tasks:
            gt = norm(t["ground_truth"]["file"])
            cmds = [c["ref"] for c in t["withdb"].get("consulted", [])
                    if c.get("type") == "command"]
            if not cmds:
                skipped += 1
                continue
            hit = False
            for cmd in cmds:
                try:
                    parts = shlex.split(cmd)
                except ValueError:
                    continue
                if not parts or "axil" not in os.path.basename(parts[0]).lower():
                    continue
                parts[0] = AXIL  # normalize binary path
                try:
                    out = subprocess.run(parts, cwd=sandbox, capture_output=True,
                                         text=True, timeout=180)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if gt in norm(out.stdout):
                    hit = True
                    break
            rows.append({"id": t["id"], "task": t["task"],
                         "gt_file": t["ground_truth"]["file"], "hit": hit,
                         "queries": cmds})
        n = len(rows)
        hits = sum(r["hit"] for r in rows)
        per_corpus.append({"corpus": corpus, "tasks_scored": n,
                           "agent_failures_excluded": skipped,
                           "hits": hits, "recall": round(hits / n, 3) if n else 0.0,
                           "rows": rows})
        print(f"{corpus:8} n={n} (excl {skipped} agent-failures)  "
              f"agent-query recall: {hits}/{n} ({hits/n:.0%})")
        for r in rows:
            print(f"   {'✓' if r['hit'] else '✗'} {r['id']:6} {r['task'][:60]}")
    return per_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=os.path.join(ROOT, "experiments", "context-ab"))
    ap.add_argument("--out")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--from-manifests", action="store_true",
                    help="replay the withdb agents' recorded axil queries instead of verbatim task text")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated subset of corpora to measure")
    args = ap.parse_args()
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    if args.from_manifests:
        manifests = {c: os.path.join(args.exp_root, f"manifest-{c}-v2.json") for c in corpora}
        per_corpus = replay_agent_queries(args.exp_root, manifests, args.top_k)
        report = {
            "benchmark": "context-ab-code-recall-agent-queries",
            "metric": "answer-file surfaced: ground-truth PRIMARY file appears in the stdout of at "
                      "least one axil lookup the withdb agent actually ran (recorded in the A/B "
                      "manifests); agent-failure tasks (no lookups recorded) excluded",
            "query_protocol": "replay of agent-issued commands (code-search / fts / code-context), "
                              "binary path normalized to the current build",
            "per_corpus": [{k: v for k, v in c.items() if k != "rows"} for c in per_corpus],
            "detail": per_corpus,
        }
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            print("wrote", args.out)
        return

    per_corpus = []
    for corpus in corpora:
        fixture = os.path.join(ROOT, "benchmarks", "context-ab", f"tasks-{corpus}.json")
        sandbox = os.path.join(args.exp_root, "withdb", corpus)
        tasks = json.load(open(fixture))["tasks"]
        rows = []
        for t in tasks:
            gt = norm(t["ground_truth"]["file"])
            cs_paths = paths_from_code_search(sandbox, t["task"], args.top_k)
            fts_paths = paths_from_fts(sandbox, t["task"], args.top_k)
            rows.append({
                "id": t["id"], "kind": t.get("kind", ""), "task": t["task"],
                "gt_file": t["ground_truth"]["file"],
                "code_search_hit": gt in cs_paths,
                "fts_hit": gt in fts_paths,
                "either_hit": gt in cs_paths or gt in fts_paths,
                "code_search_paths": cs_paths,
            })
        n = len(rows)
        cs = sum(r["code_search_hit"] for r in rows)
        ft = sum(r["fts_hit"] for r in rows)
        either = sum(r["either_hit"] for r in rows)
        per_corpus.append({
            "corpus": corpus, "tasks": n,
            "code_search_at5": cs, "code_search_recall": round(cs / n, 3),
            "fts_at5": ft, "fts_recall": round(ft / n, 3),
            "either_at5": either, "either_recall": round(either / n, 3),
            "rows": rows,
        })
        print(f"{corpus:8} n={n}  code-search@5: {cs}/{n} ({cs/n:.0%})  "
              f"fts@5: {ft}/{n} ({ft/n:.0%})  either: {either}/{n} ({either/n:.0%})")
        for r in rows:
            mark = "✓" if r["code_search_hit"] else ("~" if r["either_hit"] else "✗")
            print(f"   {mark} {r['id']:6} {r['task'][:60]}")

    report = {
        "benchmark": "context-ab-code-recall",
        "metric": f"answer-file@{args.top_k}: ground-truth PRIMARY file among top-{args.top_k} "
                  "pointers; query = task text verbatim; alternates not credited (conservative)",
        "query_protocol": "axil code-search '<task>' --top-k 5 --json (fts@5 recorded as diagnostic)",
        "per_corpus": [{k: v for k, v in c.items() if k != "rows"} for c in per_corpus],
        "detail": per_corpus,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
