#!/usr/bin/env python3
"""context-ab-score.py — deterministic token scorer for the real A/B test.

Reads the manifest the A/B workflow emits (one entry per task, with each
regime's answer, correctness verdict, and the exact files/commands the
agent consulted) and recomputes the context-token cost MECHANICALLY:

  * file entries   -> read the file (or the cited line range) from disk
  * command entries-> re-run the command and read its stdout

Token count uses the same ~4-chars/token heuristic Axil uses internally,
so figures line up with `axil context-savings`. Only tasks where BOTH
regimes produced a verdict-correct answer are counted in the totals — we
compare context cost at EQUAL task success, not raw output volume.

Re-running commands is restricted to a read-only allow-list (search /
read / axil); anything else is skipped and logged, so the scorer cannot
be steered into running arbitrary commands from the manifest.

Usage:
  context-ab-score.py --manifest run.json \
      --without-root experiments/context-ab/without/flask \
      --withdb-root  experiments/context-ab/withdb/flask \
      --out-md experiments/context-ab/report.md \
      --out-json experiments/context-ab/report.json
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys

ALLOW = {
    "rg", "grep", "egrep", "fgrep", "find", "ls", "tree", "sed", "cat",
    "head", "tail", "wc", "awk", "axil",
}


def est_tokens(text: str) -> int:
    n = len(text)
    return 0 if n == 0 else math.ceil(n / 4)


def file_tokens(root: str, ref: str, line_start, line_end):
    """Tokens for a file (or a 1-based inclusive line range)."""
    path = ref if os.path.isabs(ref) else os.path.join(root, ref)
    if not os.path.isfile(path):
        return 0, f"MISSING:{ref}"
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        return 0, f"ERR:{ref}:{e}"
    if line_start:
        s = max(1, int(line_start))
        e = int(line_end) if line_end else len(lines)
        text = "".join(lines[s - 1:e])
        return est_tokens(text), f"{ref}:{s}-{e}"
    return est_tokens("".join(lines)), f"{ref}:full"


def command_tokens(root: str, cmd: str):
    """Tokens for a command's stdout, re-run in `root`. Allow-list gated."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return 0, f"UNPARSEABLE:{cmd[:40]}"
    if not parts:
        return 0, "EMPTY"
    head = os.path.basename(parts[0])
    if head not in ALLOW:
        return 0, f"SKIPPED(not-allowed):{head}"
    try:
        out = subprocess.run(
            parts, cwd=root, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 0, f"ERR:{head}:{e}"
    return est_tokens(out.stdout), f"{head}(stdout {len(out.stdout)}B)"


def score_regime(entries, root):
    total = 0
    trace = []
    for it in entries or []:
        kind = it.get("type")
        if kind == "file":
            t, note = file_tokens(root, it.get("ref", ""),
                                  it.get("line_start"), it.get("line_end"))
        elif kind == "command":
            t, note = command_tokens(root, it.get("ref", ""))
        else:
            t, note = 0, f"UNKNOWN-TYPE:{kind}"
        total += t
        trace.append({"tokens": t, "note": note})
    return total, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--without-root", required=True)
    ap.add_argument("--withdb-root", required=True)
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        data = json.load(fh)

    rows = []
    for task in data.get("tasks", []):
        wo = task.get("without", {})
        wd = task.get("withdb", {})
        wo_correct = bool(wo.get("verdict", {}).get("correct"))
        wd_correct = bool(wd.get("verdict", {}).get("correct"))
        wo_tok, wo_trace = score_regime(wo.get("consulted"), args.without_root)
        wd_tok, wd_trace = score_regime(wd.get("consulted"), args.withdb_root)
        rows.append({
            "task": task.get("task", ""),
            "ground_truth": task.get("ground_truth"),
            "without_tokens": wo_tok,
            "withdb_tokens": wd_tok,
            "without_correct": wo_correct,
            "withdb_correct": wd_correct,
            "both_correct": wo_correct and wd_correct,
            "without_answer": wo.get("answer", ""),
            "withdb_answer": wd.get("answer", ""),
            "without_trace": wo_trace,
            "withdb_trace": wd_trace,
        })

    counted = [r for r in rows if r["both_correct"]]
    tot_wo = sum(r["without_tokens"] for r in counted)
    tot_wd = sum(r["withdb_tokens"] for r in counted)
    reduction = (1 - tot_wd / tot_wo) * 100 if tot_wo else 0.0
    ratio = (tot_wo / tot_wd) if tot_wd else 0.0

    summary = {
        "tasks_total": len(rows),
        "tasks_counted": len(counted),
        "without_correct": sum(r["without_correct"] for r in rows),
        "withdb_correct": sum(r["withdb_correct"] for r in rows),
        "total_without_tokens": tot_wo,
        "total_withdb_tokens": tot_wd,
        "reduction_pct": round(reduction, 1),
        "compression_ratio": round(ratio, 1),
    }
    report = {"summary": summary, "tasks": rows}

    # ---- stdout + markdown ----
    def fmt(n):
        return f"{n:,}"

    md = []
    md.append("# Real A/B: context cost with vs without Axil (flask corpus)\n")
    md.append(f"- Tasks run: **{summary['tasks_total']}**, "
              f"counted (both answers correct): **{summary['tasks_counted']}**")
    md.append(f"- Correct answers — without Axil: {summary['without_correct']}/"
              f"{summary['tasks_total']}, with Axil: {summary['withdb_correct']}/"
              f"{summary['tasks_total']}\n")
    md.append("| Task | Both ok | no Axil | w/ Axil | Saved |")
    md.append("|---|:--:|--:|--:|--:|")
    for r in rows:
        ok = "✅" if r["both_correct"] else "—"
        saved = (f"{(1 - r['withdb_tokens']/r['without_tokens'])*100:.0f}%"
                 if r["without_tokens"] else "n/a")
        md.append(f"| {r['task'][:60]} | {ok} | {fmt(r['without_tokens'])} | "
                  f"{fmt(r['withdb_tokens'])} | {saved} |")
    md.append(f"| **Total (counted)** | | **{fmt(tot_wo)}** | "
              f"**{fmt(tot_wd)}** | **{summary['reduction_pct']}%** |\n")
    md.append(f"**{summary['reduction_pct']}% fewer context tokens "
              f"({summary['compression_ratio']}:1)** to correctly answer the "
              f"same tasks through Axil vs reading the codebase directly.\n")
    md_text = "\n".join(md)

    print(md_text)
    if args.out_md:
        with open(args.out_md, "w") as fh:
            fh.write(md_text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"\n[scorer] summary: {json.dumps(summary)}", file=sys.stderr)


if __name__ == "__main__":
    main()
