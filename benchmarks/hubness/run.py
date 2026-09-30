#!/usr/bin/env python3
"""Are records embedded through searchable_text's fallback hubs?

A record with none of searchable_text's key fields (full_text, content, text,
description, message, summary, fact, error, statement) is embedded from every
string value joined, ids and timestamps included. The worry is that such a
vector sits close to everything and turns up in top-k lists for unrelated
queries (hubness). Two measurements, on copies of one frozen snapshot:

  vector  crates/engines/axil-vector/examples/hubness.rs: exact cosine search
          over the default vector index with the CLI's embedder. N_k (how many
          queries have a record in their top k) for the dogfood questions and
          for every record's own vector, split by fallback / keyed / internal,
          plus a what-if that re-embeds every commit as if it had no key field.
  recall  the real `axil recall` CLI (the dogfood ranked view): how often a
          fallback record takes a top-k slot for a question whose answers
          all have a key field.

Usage:
  python3 benchmarks/hubness/run.py \\
      --snapshot benchmarks/dogfood-recall/data/snap-prefix \\
      [--axil axil] [--example-bin target/release/examples/hubness] \\
      [--label <snapshot name>] [--k 10] [--out benchmarks/results/hubness-<label>.json]

Without --example-bin the example is built with cargo first.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOGFOOD = ROOT / "benchmarks" / "dogfood-recall"


def load_dogfood():
    spec = importlib.util.spec_from_file_location("dogfood_run", DOGFOOD / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_example() -> Path:
    env = dict(os.environ)
    env.setdefault("CARGO_BUILD_JOBS", "2")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "axil-vector", "--features", "embed",
         "--example", "hubness"],
        cwd=ROOT, env=env, check=True,
    )
    return ROOT / "target" / "release" / "examples" / "hubness"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recall_view(dogfood, axil: str, db: Path, questions: list, fallback: set,
                k: int, fetch: int) -> dict:
    """The dogfood ranked view, with each top-k hit put in a class."""
    per_question = []
    slots = {"keyed": 0, "fallback": 0, "internal": 0}
    keyed_q = with_fallback = fallback_slots = fallback_above = 0
    for q in questions:
        hits = json.loads(dogfood.run_axil(axil, db, [
            "recall", q["question"], "--top-k", str(fetch), "--recall-format", "full",
        ]) or "[]")
        top = [h for h in hits if (h.get("created_at") or "") <= dogfood.CUTOFF][:k]
        classes = [
            "fallback" if h["id"] in fallback
            else "internal" if h["table"].startswith("_")
            else "keyed"
            for h in top
        ]
        for c in classes:
            slots[c] += 1
        expected = {i for g in q["expect"] for i in g}
        answers_keyed = not (expected & fallback)
        first_expected = next((r + 1 for r, h in enumerate(top) if h["id"] in expected), None)
        fallback_hits = [
            {"rank": r + 1, "id": h["id"]}
            for r, (h, c) in enumerate(zip(top, classes)) if c == "fallback"
        ]
        if answers_keyed:
            keyed_q += 1
            with_fallback += bool(fallback_hits)
            fallback_slots += len(fallback_hits)
            if fallback_hits and (first_expected is None or fallback_hits[0]["rank"] < first_expected):
                fallback_above += 1
        per_question.append({
            "id": q["id"],
            "kind": q["kind"],
            "answers_all_keyed": answers_keyed,
            "first_expected_rank": first_expected,
            "fallback_hits": fallback_hits,
        })
        print(f"\r  recall {len(per_question)}/{len(questions)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    total = sum(slots.values()) or 1
    return {
        "k": k,
        "fetch": fetch,
        "cutoff": dogfood.CUTOFF,
        "slots_by_class": slots,
        "share_of_slots_by_class": {c: round(n / total, 4) for c, n in slots.items()},
        "keyed_answer_questions": {
            "questions": keyed_q,
            "with_fallback_in_top_k": with_fallback,
            "fallback_slots": fallback_slots,
            "fallback_above_first_expected": fallback_above,
        },
        "per_question": per_question,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True,
                    help="directory holding a frozen, healed memory.axil (see dogfood-recall)")
    ap.add_argument("--axil", default="axil", help="CLI for the recall view")
    ap.add_argument("--example-bin", help="prebuilt hubness example (default: build it)")
    ap.add_argument("--label", help="name recorded for the snapshot (default: --snapshot)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args()

    dogfood = load_dogfood()
    snapshot = Path(args.snapshot)
    source = snapshot / "memory.axil"
    if not source.exists():
        sys.exit(f"{source} not found")
    example = Path(args.example_bin) if args.example_bin else build_example()
    questions_path = DOGFOOD / "questions.jsonl"
    questions = [json.loads(l) for l in open(questions_path) if l.strip()]
    version = subprocess.run([args.axil, "--version"], capture_output=True, text=True).stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        # Separate copies: the example's open may migrate its copy.
        vector_dir = Path(tmp) / "vector"
        recall_dir = Path(tmp) / "recall"
        vector_dir.mkdir()
        recall_dir.mkdir()
        out = subprocess.run(
            [str(example), "--db", str(dogfood.copy_db(source, vector_dir)),
             "--questions", str(questions_path), "--cutoff", dogfood.CUTOFF,
             "--k", str(args.k)],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            sys.exit(f"hubness example failed: {out.stderr.strip()[-500:]}")
        vector = json.loads(out.stdout)
        fallback = {r["id"] for r in vector["fallback_records"]}
        recall = recall_view(dogfood, args.axil, dogfood.copy_db(source, recall_dir),
                             questions, fallback, args.k, dogfood.FETCH)

    report = {
        "benchmark": "hubness",
        "snapshot": args.label or str(snapshot),
        "snapshot_files": {
            p.name: p.stat().st_size for p in sorted(snapshot.iterdir()) if p.is_file()
        },
        "memory_axil_sha256": sha256(source),
        "questions": str(questions_path.relative_to(ROOT)),
        "axil": version,
        "method": {
            "classes": "keyed: searchable_text read a key field; fallback: it joined every "
                       "string value; internal: an _-prefixed table embedded by an extension",
            "vector": "exact cosine over every vector in the default index, owners scored by "
                      "their closest vector (a recall chunk counts for its source record); "
                      "questions embedded with embed_query, records query with their stored "
                      "vector and exclude themselves; records created after the cutoff are "
                      "left out of every population",
            "hub": vector["hub_rule"],
            "what_if": "every keyed commit re-embedded from its data without any key field "
                       "(sha, author, date, subject, body joined: the shape commits had before "
                       "the hook stored content/summary), replacing its vectors",
            "recall": "`axil recall --recall-format full`, the dogfood ranked view: fetch "
                      f"{dogfood.FETCH}, drop hits after the cutoff, keep the top k",
        },
        "vector": vector,
        "recall": recall,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
