#!/usr/bin/env python3
"""recall-quality.py — measure Axil's retrieval miss-rate on a ground-truth task set.

For each task, runs the agent's FIRST cheap queries (the verbatim task text
through `code-search` and `fts`) against the indexed corpus and checks
whether the ground-truth file/symbol lands in the top-k. Reports recall@1/3/5
at file and symbol level, per strategy and for the union (agent tries both),
plus the explicit miss list.

This isolates retrieval QUALITY (does the right hit rank high?) from output
SIZE (already handled by lean code-context). No LLM — deterministic.

Usage:
  recall-quality.py --tasks experiments/context-ab/tasks-django.json \
    --withdb-root experiments/context-ab/withdb/django \
    --axil-bin target/release/axil
"""
import argparse
import json
import subprocess
import sys


def run_json(bin_, args, cwd):
    try:
        out = subprocess.run([bin_, *args], cwd=cwd, capture_output=True,
                             text=True, timeout=60).stdout
        return json.loads(out) if out.strip() else []
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return []


def code_search_hits(bin_, query, cwd, k):
    items = run_json(bin_, ["code-search", query, "--top-k", str(k), "--json"], cwd)
    return [(i.get("path", ""), i.get("symbol") or "") for i in items if isinstance(i, dict)]


def fts_hits(bin_, query, cwd, k):
    items = run_json(bin_, ["fts", query, "--limit", str(k)], cwd)
    out = []
    for i in items:
        d = i.get("data", i) if isinstance(i, dict) else {}
        out.append((d.get("path", ""), d.get("symbol") or ""))
    return out


def first_rank(hits, gt_file, gt_symbol, symbol_level):
    """0-based rank of the first matching hit, or None."""
    gsym = (gt_symbol or "").lower()
    for idx, (path, sym) in enumerate(hits):
        if path != gt_file:
            continue
        if not symbol_level:
            return idx
        s = (sym or "").lower()
        if s == gsym or gsym in s or s in gsym and s:
            return idx
    return None


def union_rank(r1, r2):
    rs = [r for r in (r1, r2) if r is not None]
    return min(rs) if rs else None


def at_k(rank, k):
    return rank is not None and rank < k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--withdb-root", required=True)
    ap.add_argument("--axil-bin", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    rows = []
    for t in tasks:
        q = t["task"]
        gt = t["ground_truth"]
        cs = code_search_hits(args.axil_bin, q, args.withdb_root, args.top_k)
        ft = fts_hits(args.axil_bin, q, args.withdb_root, args.top_k)
        # file-level ranks
        cs_f = first_rank(cs, gt["file"], gt.get("symbol"), False)
        ft_f = first_rank(ft, gt["file"], gt.get("symbol"), False)
        un_f = union_rank(cs_f, ft_f)
        # symbol-level ranks
        cs_s = first_rank(cs, gt["file"], gt.get("symbol"), True)
        ft_s = first_rank(ft, gt["file"], gt.get("symbol"), True)
        un_s = union_rank(cs_s, ft_s)
        rows.append({
            "task": q, "gt": f"{gt['file']}::{gt.get('symbol','')}",
            "cs_file_rank": cs_f, "fts_file_rank": ft_f, "union_file_rank": un_f,
            "union_symbol_rank": un_s,
            "miss_at1": not at_k(un_f, 1),  # right FILE not even rank-1 on either query
            "hit_file_3": at_k(un_f, 3),
            "hit_sym_3": at_k(un_s, 3),
        })

    n = len(rows)
    def rate(pred):
        return sum(1 for r in rows if pred(r)) / n * 100 if n else 0.0

    label = args.label or args.tasks
    print(f"\n### Recall quality — {label}  (n={n}, top_k={args.top_k})")
    print("Query = verbatim task text. 'union' = best of code-search OR fts.\n")
    print("| metric | rate |")
    print("|---|---:|")
    print(f"| file recall@1 (union) | {rate(lambda r: at_k(r['union_file_rank'],1)):.0f}% |")
    print(f"| file recall@3 (union) | {rate(lambda r: r['hit_file_3']):.0f}% |")
    print(f"| file recall@5 (union) | {rate(lambda r: at_k(r['union_file_rank'],5)):.0f}% |")
    print(f"| symbol recall@3 (union) | {rate(lambda r: r['hit_sym_3']):.0f}% |")
    print(f"| code-search file recall@3 | {rate(lambda r: at_k(r['cs_file_rank'],3)):.0f}% |")
    print(f"| fts file recall@3 | {rate(lambda r: at_k(r['fts_file_rank'],3)):.0f}% |")

    misses = [r for r in rows if not r["hit_file_3"]]
    print(f"\n**Misses (right file not in top-3 of either query): {len(misses)}/{n}**")
    for r in misses:
        print(f"- {r['task'][:60]}  → want {r['gt']}; "
              f"cs_rank={r['cs_file_rank']} fts_rank={r['fts_file_rank']}")
    return rows


if __name__ == "__main__":
    main()
