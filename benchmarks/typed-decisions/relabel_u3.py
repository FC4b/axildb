#!/usr/bin/env python3
"""Blind relabel of the U3 texts by an external model.

Each annotator sees only validation/u3-guidelines.md and the item texts,
under neutral ids (b-001 ...) in its own shuffled order, in batches of
BATCH_SIZE; no labels, sources, splits or other annotators' answers. The
answer must be a strict JSON array; it is parsed defensively (code fences and
surrounding prose are tolerated, every object is validated), and a batch
with a missing or invalid answer is retried, up to MAX_ATTEMPTS. Every
attempt's raw output and outcome is kept.

Annotators:
  glm     zai-coding-plan/glm-5.3 through OpenCode
          (`opencode run --dir <worktree> -m ... "<prompt>"`)
  gemini  gemini-3.8-flash-high through the Antigravity CLI
          (`agy -p "<prompt>" --model ... --print-timeout <t>`), run from an
          empty directory
  gptoss  gpt-oss-120b-medium through the same CLI, the same way; the
          stand-in when the Gemini quota is exhausted

Outputs, under validation/u3/:
  <name>-map.json         neutral id -> u3 id, with the shuffle seed
  <name>-raw/             batch_NN.attemptK.out, batch_NN.attempts.json, run.log
  <name>.jsonl            one row per item, ids mapped back:
                          {"id", "class", "importance", "rule"}
  <name>-raw/postprocess-summary.json

Usage:
  python relabel_u3.py --annotator glm --lanes 2
  python relabel_u3.py --annotator gemini --lanes 2 --timeout 300s
  python relabel_u3.py --annotator glm --collect     # re-parse saved raw outputs only
"""

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
TEXTS = HERE / "sets" / "u3_texts.jsonl"
GUIDELINES = HERE / "validation" / "u3-guidelines.md"
V = HERE / "validation" / "u3"
SCRATCH = WORKTREE / "target" / "agent-scratch" / "u3-relabel"

BATCH_SIZE = 20
MAX_ATTEMPTS = 3
CLASSES = {"decision", "error", "context", "rule", "none"}
IMPORTANCE = {"low", "medium", "high"}
ANNOTATORS = {
    "glm": {"model": "zai-coding-plan/glm-5.3", "seed": 20261014},
    "gemini": {"model": "gemini-3.8-flash-high", "seed": 20261015},
    "gptoss": {"model": "gpt-oss-120b-medium", "seed": 20261016},
}

PROMPT = """You are labeling data for a benchmark, as one of several independent annotators.
Answer from this message alone. Do not read, open, search or write any files, and do not run
any commands or tools: everything you need is below.

Follow these labeling guidelines exactly:

<guidelines>
{guidelines}
</guidelines>

Label each of the {n} items below. Each item's text is between its <item> tags; ids look like
b-NNN. Return ONLY a JSON array with exactly {n} objects, one per item, in the order given, of
the form {{"id": "b-NNN", "class": "...", "importance": "low|medium|high" or null, "rule": "..."}}.
No prose before or after it, no markdown fence.

{items}
"""


def log(path: Path, msg: str, lock=threading.Lock()):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    with lock:
        with path.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


def load_texts():
    return [json.loads(line) for line in TEXTS.read_text().splitlines() if line.strip()]


def neutral_map(rows, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return {f"b-{k + 1:03d}": rows[i]["id"] for k, i in enumerate(order)}


def batches(blind_ids):
    ids = sorted(blind_ids)
    return {f"{k // BATCH_SIZE + 1:02d}": ids[k:k + BATCH_SIZE] for k in range(0, len(ids), BATCH_SIZE)}


def render(batch_ids, text_of, guidelines):
    items = "\n\n".join(f'<item id="{b}">\n{text_of[b]}\n</item>' for b in batch_ids)
    return PROMPT.format(guidelines=guidelines, n=len(batch_ids), items=items)


# ── Parsing ─────────────────────────────────────────────────────────

OBJ = re.compile(r'\{[^{}]*"id"\s*:\s*"b-\d{3}"[^{}]*\}')


def parse(raw: str, expected: list):
    """(answers by blind id, problems) from one raw output."""
    problems, objs = [], None
    # The CLI's stderr is appended to the saved output; the answer is on stdout.
    text = re.sub(r"```(?:json)?", "", raw.split("\n[stderr]\n")[0])
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            objs = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            objs = None
    if not isinstance(objs, list):
        found = []
        for m in OBJ.finditer(text):
            try:
                found.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
        if found:
            problems.append("array did not parse; recovered objects one by one")
            objs = found
        else:
            return {}, ["no JSON array found"]
    answers = {}
    for o in objs:
        if not isinstance(o, dict):
            problems.append("non-object entry")
            continue
        bid = o.get("id")
        cls = o.get("class")
        imp = o.get("importance")
        if isinstance(cls, str):
            cls = cls.strip().lower()
        if isinstance(imp, str):
            imp = imp.strip().lower()
            imp = None if imp in ("null", "none", "") else imp
        if bid not in expected:
            problems.append(f"unexpected id {bid}")
            continue
        if bid in answers:
            problems.append(f"duplicate id {bid}")
            continue
        if cls not in CLASSES:
            problems.append(f"{bid}: bad class {cls!r}")
            continue
        if cls == "none" and imp is not None:
            problems.append(f"{bid}: importance {imp!r} on a none item, set to null")
            imp = None
        if cls != "none" and imp not in IMPORTANCE:
            problems.append(f"{bid}: bad importance {imp!r}")
            continue
        answers[bid] = {"class": cls, "importance": imp, "rule": str(o.get("rule") or "")}
    missing = [b for b in expected if b not in answers]
    if missing:
        problems.append(f"missing {len(missing)}: {missing[:5]}")
    return answers, problems


# ── Running ─────────────────────────────────────────────────────────


def command(name, args, prompt_text):
    if name == "glm":
        return ["opencode", "run", "--dir", str(WORKTREE), "-m", ANNOTATORS[name]["model"], prompt_text], None
    empty = SCRATCH / "agy-empty-cwd"
    empty.mkdir(parents=True, exist_ok=True)
    return (["agy", "-p", prompt_text, "--model", ANNOTATORS[name]["model"],
             "--print-timeout", args.timeout], str(empty))


def run_batch(name, args, key, batch_ids, text_of, guidelines, raw_dir, run_log):
    prompt_text = render(batch_ids, text_of, guidelines)
    (SCRATCH / name).mkdir(parents=True, exist_ok=True)
    (SCRATCH / name / f"batch_{key}.prompt.txt").write_text(prompt_text)
    attempts_path = raw_dir / f"batch_{key}.attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    if any(a["ok_ids"] == len(batch_ids) for a in attempts):
        log(run_log, f"batch {key}: already complete, skipped")
        return
    for k in range(len(attempts) + 1, MAX_ATTEMPTS + 1):
        log(run_log, f"batch {key} attempt {k}: start")
        cmd, cwd = command(name, args, prompt_text)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=args.hard_timeout)
            out, rc = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else ""), proc.returncode
        except subprocess.TimeoutExpired as e:
            out, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
        secs = round(time.time() - t0, 1)
        out_name = f"batch_{key}.attempt{k}.out"
        (raw_dir / out_name).write_text(out)
        answers, problems = parse(out, batch_ids)
        attempts.append({"attempt": k, "rc": rc, "secs": secs, "ok_ids": len(answers),
                         "problems": problems, "out": out_name})
        attempts_path.write_text(json.dumps(attempts, indent=1))
        log(run_log, f"batch {key} attempt {k}: rc={rc} {secs}s ok_ids={len(answers)}/{len(batch_ids)} "
                     f"problems={problems[:3]}")
        if len(answers) == len(batch_ids):
            log(run_log, f"batch {key} done after {k} attempt(s)")
            return
        log(run_log, f"batch {key}: RETRY because {problems[:3]}")
    log(run_log, f"batch {key}: gave up after {MAX_ATTEMPTS} attempts")


def collect(name, rows, blind_to_id, groups, raw_dir):
    """Best attempt per batch (most valid ids, earliest on ties) -> <name>.jsonl."""
    out, failures, used, all_problems = {}, [], {}, Counter()
    for key, batch_ids in groups.items():
        attempts_path = raw_dir / f"batch_{key}.attempts.json"
        if not attempts_path.exists():
            failures.append({"batch": key, "reason": "never run"})
            continue
        attempts = json.loads(attempts_path.read_text())
        best, best_answers = None, {}
        for a in attempts:
            answers, problems = parse((raw_dir / a["out"]).read_text(), batch_ids)
            for p in problems:
                all_problems[re.sub(r"b-\d{3}", "b-NNN", p.split(":")[0])] += 1
            if len(answers) > len(best_answers):
                best, best_answers = a, answers
        used[key] = {"attempts": len(attempts), "used_attempt": best["attempt"] if best else None}
        for b in batch_ids:
            if b in best_answers:
                out[blind_to_id[b]] = best_answers[b]
            else:
                failures.append({"batch": key, "blind_id": b, "id": blind_to_id[b]})
    with (V / f"{name}.jsonl").open("w") as fh:
        for r in rows:
            if r["id"] in out:
                fh.write(json.dumps({"id": r["id"], **out[r["id"]]}) + "\n")
    summary = {
        "annotator": name,
        "model": ANNOTATORS[name]["model"],
        "items": len(rows),
        "items_labeled": len(out),
        "failures": failures,
        "by_class": dict(Counter(v["class"] for v in out.values())),
        "by_importance": dict(Counter(v["importance"] for v in out.values() if v["importance"])),
        "batches": used,
        "parse_problems_seen": dict(all_problems),
    }
    (raw_dir / "postprocess-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("items_labeled", "by_class", "by_importance")}))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotator", choices=list(ANNOTATORS), required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--timeout", default="300s", help="agy --print-timeout (gemini)")
    ap.add_argument("--hard-timeout", type=int, default=3600, help="seconds before a call is killed")
    ap.add_argument("--only", nargs="*", help="batch keys to run (default: all)")
    ap.add_argument("--collect", action="store_true", help="only re-parse the saved raw outputs")
    args = ap.parse_args()

    name = args.annotator
    rows = load_texts()
    blind_to_id = neutral_map(rows, ANNOTATORS[name]["seed"])
    text_of = {b: next(r["text"] for r in rows if r["id"] == i) for b, i in blind_to_id.items()}
    groups = batches(blind_to_id)
    raw_dir = V / f"{name}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (V / f"{name}-map.json").write_text(json.dumps({
        "annotator": name, "model": ANNOTATORS[name]["model"],
        "source": "benchmarks/typed-decisions/sets/u3_texts.jsonl",
        "shuffle": f"random.Random({ANNOTATORS[name]['seed']}).shuffle(range(n))",
        "seed": ANNOTATORS[name]["seed"], "batch_size": BATCH_SIZE,
        "blind_to_id": blind_to_id}, indent=1) + "\n")
    if not args.collect:
        guidelines = GUIDELINES.read_text()
        run_log = raw_dir / "run.log"
        keys = args.only or list(groups)
        with ThreadPoolExecutor(max_workers=args.lanes) as pool:
            futures = [pool.submit(run_batch, name, args, k, groups[k], text_of, guidelines, raw_dir, run_log)
                       for k in keys]
            for f in futures:
                f.result()
        log(run_log, "DONE")
    collect(name, rows, blind_to_id, groups, raw_dir)


if __name__ == "__main__":
    main()
