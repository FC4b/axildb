#!/usr/bin/env python3
"""Full-path numbers for the "possible contradiction" review edge.

Runs `heuristic --full-path <set> --judge` on the U1 sets: every old text is
stored in a real database (default bge-small embedder), then each new text
goes through `Axil::supersede_candidates` (cosine >= 0.65, top 5) and both
judges:

* before: `judge_conflict`, the rule judge `axil detect-conflicts` uses,
  which supersedes;
* after: the U1 model through `Axil::judge_possible_contradictions`, the
  review pass's judgement, which only flags.

Reports, per set and split: how often the true predecessor was a candidate,
updates caught (label 1), still-valid records hit (label 0), and other
candidates hit. Labels and splits come from the sets.

    cargo build --release --manifest-path benchmarks/typed-decisions/heuristic/Cargo.toml
    python3 benchmarks/typed-decisions/review_edge_full_path.py \\
        --bin benchmarks/typed-decisions/heuristic/target/release/typed-decisions-heuristic \\
        --out benchmarks/results/typed-decisions-u1-review-edge-full-path-DATE.json
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rate(rows, key):
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else None


def summarize(rows):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    return {
        "pairs": {"updates": len(pos), "non_updates": len(neg)},
        "candidate_recall_updates": rate(pos, "candidate"),
        "before_rule_judge_supersedes": {
            "updates_caught": rate(pos, "supersedes_true"),
            "still_valid_retired": rate(neg, "supersedes_true"),
            "other_candidates_retired": sum(r["supersedes_other"] for r in rows),
        },
        "after_u1_review_edge_flags": {
            "updates_flagged": rate(pos, "flagged_true"),
            "still_valid_flagged": rate(neg, "flagged_true"),
            "other_candidates_flagged": sum(r["flagged_other"] for r in rows),
            "other_candidates_flagged_per_pair": round(
                sum(r["flagged_other"] for r in rows) / len(rows), 4
            ),
            "records_retired": 0,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True, help="the typed-decisions-heuristic binary")
    ap.add_argument("--sets", default="u1_repo.jsonl,u1_lme.jsonl")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {
        "benchmark": "typed-decisions-u1-review-edge-full-path",
        "date": time.strftime("%Y-%m-%d"),
        "what": __doc__.split("\n\n")[1].replace("\n", " "),
        "judge": "nli-deberta-v3-xsmall-qint8, calibration u1@2026-10-04 (threshold 0.695294)",
        "candidates": "Axil::supersede_candidates, cosine >= 0.65, top 5, default bge-small embedder",
        "note": args.note,
        "sets": {},
    }
    for name in args.sets.split(","):
        path = HERE / "sets" / name
        labels = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row
        proc = subprocess.run(
            [args.bin, "--full-path", str(path), "--judge"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows, superseded = [], None
        for line in proc.stdout.splitlines():
            out = json.loads(line)
            if "superseded" in out:
                superseded = out["superseded"]
                continue
            out["label"] = labels[out["id"]]["label"]
            out["split"] = labels[out["id"]]["split"]
            rows.append(out)
        report["sets"][path.stem] = {
            "all": summarize(rows),
            "test": summarize([r for r in rows if r["split"] == "test"]),
            "superseded_at_end": superseded,
            "superseded_note": "records the database marked superseded after the run; the judge path "
            "has no supersede call, so these come from insert-time near-duplicate handling "
            "(cosine >= 0.92) while the old texts were stored",
            "flagged_scores": sorted(
                round(r["p_true"], 4) for r in rows if r["flagged_true"]
            ),
        }
    text = json.dumps(report, indent=2)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
