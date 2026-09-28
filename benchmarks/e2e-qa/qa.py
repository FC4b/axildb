#!/usr/bin/env python3
"""End-to-end QA: does recall give a reader model what it needs to answer?

Reads the per-question contexts a recall eval dumped (`--dump-context` on the
LongMemEval harness or on benchmarks/dogfood-recall/run.py), has a reader
model answer each question from that context alone, and has a judge model
grade the answer: against the gold answer (LongMemEval) or against the
expected records (dogfood).

Models run through headless Claude Code (`claude -p`) on the local
subscription, with no tools, hooks, MCP servers, settings or project
instructions, and a replaced system prompt. That is a different transport
from the API, so these numbers are for comparing Axil changes with each
other, not for publishing against other systems.

Responses are cached under data/cache/ by (model, prompt), so re-running a
report spends nothing.

Usage:
  python3 benchmarks/e2e-qa/qa.py CONTEXTS.jsonl --context full|compact|hook
      [--reader haiku] [--judge sonnet] [--limit N] [--jobs 4] [--out FILE]
"""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"

READER_SYSTEM = (
    "You answer a user's question using only the memory excerpts you are given. "
    "Answer briefly and directly. If the excerpts do not contain the answer, reply "
    "exactly: I don't know."
)
JUDGE_SYSTEM = "You grade answers to questions. Reply with exactly one word: yes or no."

# Grading rules modeled on LongMemEval's evaluator, per question type.
LME_RULES = {
    "default": (
        "Answer yes if the response contains the correct answer, or all the steps "
        "needed to reach it. Answer no if it gives a different answer or only part "
        "of the required information."
    ),
    "temporal-reasoning": (
        "Answer yes if the response contains the correct answer. Do not penalize "
        "off-by-one errors in a number of days, weeks or months. Otherwise answer no."
    ),
    "knowledge-update": (
        "Answer yes if the response contains the correct, updated answer, even if it "
        "also mentions earlier information. Otherwise answer no."
    ),
    "single-session-preference": (
        "The correct answer is a rubric describing what the user would want. Answer "
        "yes if the response satisfies the rubric. Otherwise answer no."
    ),
    "abstention": (
        "The question cannot be answered from the history. Answer yes if the response "
        "says it cannot answer or does not know. Otherwise answer no."
    ),
}
DOGFOOD_RULE = (
    "Answer yes if the response gives the key information from the reference records "
    "that answers the question, even if worded differently. Answer no if it says it "
    "does not know, contradicts the references, or misses the point."
)


def claude(model: str, system: str, prompt: str) -> dict:
    """One headless call, isolated from this machine's Claude Code setup."""
    key = hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return {**json.loads(cached.read_text()), "cached": True}
    cmd = [
        "claude", "-p", prompt, "--model", model, "--system-prompt", system,
        "--tools", "", "--strict-mcp-config", "--setting-sources", "project",
        "--settings", '{"disableAllHooks": true}', "--no-session-persistence",
        "--output-format", "json",
    ]
    last_error = ""
    for _ in range(2):
        with tempfile.TemporaryDirectory() as empty_dir:
            out = subprocess.run(cmd, cwd=empty_dir, capture_output=True, text=True, timeout=300)
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError:
            last_error = (out.stderr or out.stdout)[-300:]
            continue
        if data.get("is_error"):
            last_error = str(data.get("result"))[:300]
            continue
        usage = data.get("usage") or {}
        result = {
            "text": (data.get("result") or "").strip(),
            "input_tokens": usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result))
        return {**result, "cached": False}
    raise RuntimeError(f"claude -p failed: {last_error}")


def context_of(item: dict, mode: str) -> str:
    if mode not in item:
        raise SystemExit(f"context '{mode}' not in this file (have: {sorted(k for k in item if k in ('full', 'compact', 'hook'))})")
    return item[mode] or "(nothing was retrieved)"


def reader_prompt(item: dict, context: str) -> str:
    date = f"Today is {item['question_date']}.\n\n" if item.get("question_date") else ""
    return f"{date}Memory excerpts:\n{context}\n\nQuestion: {item['question']}"


def judge_prompt(item: dict, response: str) -> str:
    if "reference" in item:
        refs = "\n".join(f"- {r}" for r in item["reference"])
        return (
            f"{DOGFOOD_RULE}\n\nQuestion: {item['question']}\n\n"
            f"Reference records (ground truth):\n{refs}\n\nResponse: {response}\n\nyes or no?"
        )
    kind = "abstention" if item["question_id"].endswith("_abs") else item["question_type"]
    rule = LME_RULES.get(kind, LME_RULES["default"])
    return (
        f"{rule}\n\nQuestion: {item['question']}\n\nCorrect answer: {item['answer']}\n\n"
        f"Response: {response}\n\nyes or no?"
    )


def grade(item: dict, mode: str, reader: str, judge: str) -> dict:
    context = context_of(item, mode)
    answer = claude(reader, READER_SYSTEM, reader_prompt(item, context))
    verdict = claude(judge, JUDGE_SYSTEM, judge_prompt(item, answer["text"]))
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "correct": verdict["text"].lower().startswith("yes"),
        "answer": answer["text"][:300],
        "context_tokens": len(context.encode()) / 4,
        "model_input_tokens": answer["input_tokens"] + verdict["input_tokens"],
        "model_output_tokens": answer["output_tokens"] + verdict["output_tokens"],
        "cached": answer["cached"] and verdict["cached"],
    }


def summarize(rows: list) -> dict:
    n = len(rows)
    accuracy = sum(r["correct"] for r in rows) / n
    tokens = sum(r["context_tokens"] for r in rows) / n
    return {
        "n": n,
        "accuracy": accuracy,
        "context_tokens": tokens,
        "accuracy_per_1k_context_tokens": accuracy / tokens * 1000 if tokens else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("contexts")
    ap.add_argument("--context", required=True, choices=["full", "compact", "hook"])
    ap.add_argument("--reader", default="haiku")
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.contexts) if line.strip()]
    if args.limit:
        items = items[: args.limit]

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(grade, item, args.context, args.reader, args.judge) for item in items]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(future.result())
            print(f"\r  {i}/{len(items)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    rows.sort(key=lambda r: r["question_id"])

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    fresh = [r for r in rows if not r["cached"]]
    report = {
        "benchmark": "e2e-qa",
        "contexts": Path(args.contexts).name,
        "context": args.context,
        "reader": args.reader,
        "judge": args.judge,
        "transport": "claude -p (Claude Code headless, subscription)",
        "overall": summarize(rows),
        "by_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "model_tokens_this_run": {
            "calls": 2 * len(fresh),
            "input": sum(r["model_input_tokens"] for r in fresh),
            "output": sum(r["model_output_tokens"] for r in fresh),
        },
        "per_question": [
            {k: r[k] for k in ("question_id", "correct", "answer")} for r in rows
        ],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
