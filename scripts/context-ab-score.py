#!/usr/bin/env python3
"""context-ab-score.py — deterministic token + effort scorer for the real A/B test.

Reads the manifest the A/B workflow emits (one entry per task, with each
regime's answer, correctness verdict, and the exact files/commands the
agent consulted) and recomputes, MECHANICALLY:

  * context-token cost — file entries -> read the file (or cited range);
    command entries -> re-run the command and read its stdout.
  * effort-to-finish  — the number of discovery STEPS each regime took
    (one consulted artifact = one tool roundtrip = ~one agent turn).

Token count uses the same ~4-chars/token heuristic Axil uses internally,
so figures line up with `axil context-savings`. Only tasks where BOTH
regimes produced a verdict-correct answer are counted in the totals — we
compare context cost AND step count at EQUAL task success.

Re-running commands is restricted to a read-only allow-list (search /
read / axil); anything else is skipped and logged, so the scorer cannot
be steered into running arbitrary commands from the manifest.

A note on `gather_ms`: the scorer re-executes each consulted command in a
COLD process. For `axil` that reloads the ONNX embedding model on every
call — a fixed cost a warm agent session never pays repeatedly — so
`gather_ms` is captured for diagnostics only and is NOT a publishable
speed figure. The honest speed metric is `steps` (agent roundtrips).

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
import time

# Windows consoles default to cp1252 and choke on the ✅ / arrows in the report.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ALLOW = {
    "rg", "grep", "egrep", "fgrep", "find", "ls", "tree", "sed", "cat",
    "head", "tail", "wc", "awk", "axil",
}


def est_tokens(text: str) -> int:
    n = len(text)
    return 0 if n == 0 else math.ceil(n / 4)


def file_tokens(root: str, ref: str, line_start, line_end):
    """Tokens + read time (ms) for a file (or a 1-based inclusive line range)."""
    path = ref if os.path.isabs(ref) else os.path.join(root, ref)
    t0 = time.perf_counter()
    if not os.path.isfile(path):
        return 0, f"MISSING:{ref}", 0.0
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        return 0, f"ERR:{ref}:{e}", 0.0
    ms = (time.perf_counter() - t0) * 1000.0
    if line_start:
        s = max(1, int(line_start))
        e = int(line_end) if line_end else len(lines)
        text = "".join(lines[s - 1:e])
        return est_tokens(text), f"{ref}:{s}-{e}", ms
    return est_tokens("".join(lines)), f"{ref}:full", ms


def command_tokens(root: str, cmd: str):
    """Tokens + exec time (ms) for a command's stdout, re-run in `root`. Allow-list gated."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return 0, f"UNPARSEABLE:{cmd[:40]}", 0.0
    if not parts:
        return 0, "EMPTY", 0.0
    head = os.path.basename(parts[0])
    if head not in ALLOW:
        return 0, f"SKIPPED(not-allowed):{head}", 0.0
    t0 = time.perf_counter()
    try:
        out = subprocess.run(
            parts, cwd=root, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 0, f"ERR:{head}:{e}", 0.0
    ms = (time.perf_counter() - t0) * 1000.0
    return est_tokens(out.stdout), f"{head}(stdout {len(out.stdout)}B)", ms


def score_regime(entries, root):
    """Return (total_tokens, steps, gather_ms, trace) for one regime's consulted list.

    steps = number of consulted artifacts = tool roundtrips (~agent turns).
    gather_ms = summed cold re-execution time (DIAGNOSTIC ONLY — see module docstring).
    """
    total = 0
    steps = 0
    gather_ms = 0.0
    trace = []
    for it in entries or []:
        kind = it.get("type")
        if kind == "file":
            t, note, ms = file_tokens(root, it.get("ref", ""),
                                      it.get("line_start"), it.get("line_end"))
        elif kind == "command":
            t, note, ms = command_tokens(root, it.get("ref", ""))
        else:
            t, note, ms = 0, f"UNKNOWN-TYPE:{kind}", 0.0
        total += t
        steps += 1
        gather_ms += ms
        trace.append({"tokens": t, "ms": round(ms, 2), "note": note})
    return total, steps, round(gather_ms, 2), trace


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

    corpus = data.get("corpus", "unknown")

    rows = []
    for task in data.get("tasks", []):
        wo = task.get("without", {})
        wd = task.get("withdb", {})
        wo_correct = bool(wo.get("verdict", {}).get("correct"))
        wd_correct = bool(wd.get("verdict", {}).get("correct"))
        wo_tok, wo_steps, wo_ms, wo_trace = score_regime(wo.get("consulted"), args.without_root)
        wd_tok, wd_steps, wd_ms, wd_trace = score_regime(wd.get("consulted"), args.withdb_root)
        rows.append({
            "task": task.get("task", ""),
            "ground_truth": task.get("ground_truth"),
            "without_tokens": wo_tok,
            "withdb_tokens": wd_tok,
            "without_steps": wo_steps,
            "withdb_steps": wd_steps,
            "without_gather_ms": wo_ms,
            "withdb_gather_ms": wd_ms,
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

    # Effort-to-finish (steps) at equal correctness — the honest speed metric.
    steps_wo = sum(r["without_steps"] for r in counted)
    steps_wd = sum(r["withdb_steps"] for r in counted)
    step_reduction = (1 - steps_wd / steps_wo) * 100 if steps_wo else 0.0
    step_ratio = (steps_wo / steps_wd) if steps_wd else 0.0
    n = len(counted)

    summary = {
        "corpus": corpus,
        "tasks_total": len(rows),
        "tasks_counted": n,
        "without_correct": sum(r["without_correct"] for r in rows),
        "withdb_correct": sum(r["withdb_correct"] for r in rows),
        # tokens
        "total_without_tokens": tot_wo,
        "total_withdb_tokens": tot_wd,
        "reduction_pct": round(reduction, 1),
        "compression_ratio": round(ratio, 1),
        # steps (effort-to-finish)
        "total_without_steps": steps_wo,
        "total_withdb_steps": steps_wd,
        "avg_without_steps": round(steps_wo / n, 2) if n else 0.0,
        "avg_withdb_steps": round(steps_wd / n, 2) if n else 0.0,
        "step_reduction_pct": round(step_reduction, 1),
        "step_ratio": round(step_ratio, 1),
        # gather_ms — DIAGNOSTIC ONLY, cold-process ONNX reload (not publishable)
        "total_without_gather_ms": round(sum(r["without_gather_ms"] for r in counted), 1),
        "total_withdb_gather_ms": round(sum(r["withdb_gather_ms"] for r in counted), 1),
        "_gather_ms_note": "cold-process re-exec; axil reloads ONNX each call — diagnostic only, do not publish",
    }
    report = {"summary": summary, "tasks": rows}

    # ---- stdout + markdown ----
    def fmt(n):
        return f"{n:,}"

    md = []
    md.append(f"# Real A/B: context cost + effort with vs without Axil ({corpus} corpus)\n")
    md.append(f"- Tasks run: **{summary['tasks_total']}**, "
              f"counted (both answers correct): **{summary['tasks_counted']}**")
    md.append(f"- Correct answers — without Axil: {summary['without_correct']}/"
              f"{summary['tasks_total']}, with Axil: {summary['withdb_correct']}/"
              f"{summary['tasks_total']}\n")
    md.append("| Task | Both ok | no Axil tok | w/ Axil tok | Tok saved | no Axil steps | w/ Axil steps |")
    md.append("|---|:--:|--:|--:|--:|--:|--:|")
    for r in rows:
        ok = "✅" if r["both_correct"] else "—"
        saved = (f"{(1 - r['withdb_tokens']/r['without_tokens'])*100:.0f}%"
                 if r["without_tokens"] else "n/a")
        md.append(f"| {r['task'][:56]} | {ok} | {fmt(r['without_tokens'])} | "
                  f"{fmt(r['withdb_tokens'])} | {saved} | {r['without_steps']} | {r['withdb_steps']} |")
    md.append(f"| **Total (counted)** | | **{fmt(tot_wo)}** | "
              f"**{fmt(tot_wd)}** | **{summary['reduction_pct']}%** | "
              f"**{steps_wo}** | **{steps_wd}** |\n")
    md.append(f"**Tokens:** {summary['reduction_pct']}% fewer context tokens "
              f"({summary['compression_ratio']}:1) to correctly answer the same tasks "
              f"through Axil vs reading the codebase directly.")
    md.append(f"**Effort:** {summary['step_reduction_pct']}% fewer steps "
              f"({summary['avg_without_steps']} vs {summary['avg_withdb_steps']} tool "
              f"roundtrips per task on average) — each step is ~one agent turn.\n")
    md_text = "\n".join(md)

    print(md_text)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md_text)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(f"\n[scorer] summary: {json.dumps(summary)}", file=sys.stderr)


if __name__ == "__main__":
    main()
