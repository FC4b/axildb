#!/usr/bin/env python3
"""Merge the raw bake-off outputs in out/ into the committed summary JSON.

Reads out/bakeoff-<date>.json (NLI arm), out/bakeoff-laya-*-<date>.json
(Jev-style arm) and out/heuristic-u1.jsonl (production-heuristic arm), and
writes benchmarks/results/typed-decisions-bakeoff-<date>.json.

Usage: python summarize.py 2026-09-27
"""

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP = ["n_train", "n_test", "auroc_test", "auroc_all", "threshold_from_train", "at_threshold_test",
        "at_argmax_test", "ece_test_raw", "ece_test_platt", "latency_ms_p50", "latency_ms_p95",
        "accuracy_by_kind_test", "auroc_test_by_source"]


def pick(r):
    return {k: r[k] for k in KEEP if k in r}


def count(path):
    return sum(1 for line in open(path) if line.strip())


def main():
    date = sys.argv[1]
    onnx = json.load(open(HERE / "out" / f"bakeoff-{date}.json"))
    arms = {}
    for name, e in onnx["results"].items():
        arms[name] = {"family": "nli-cross-encoder",
                      "runtime": f"onnxruntime {onnx['onnxruntime']} CPU, {onnx['threads']} threads",
                      "size_mb": e["size_mb"], "load_ms": e["load_ms"],
                      "U1": pick(e["U1"]), "U1+sym": pick(e["U1+sym"]), "U2": pick(e["U2"])}
    for f in sorted(glob.glob(str(HERE / "out" / f"bakeoff-laya-*-{date}.json"))):
        e = json.load(open(f))
        arms[e["model"]] = {"family": "jev-style (laya 0.3.20)", "runtime": f"torch CPU, {e['threads']} threads",
                            "load_ms": e["load_ms"], "U1": pick(e["U1"]), "U2": pick(e["U2"]),
                            "questions": {"U1": e["U1"]["question"], "U2": e["U2"]["question"]}}
    h = [json.loads(line) for line in open(HERE / "out" / "heuristic-u1.jsonl")]
    arms["axil-check_conflict (production heuristic)"] = {
        "family": "heuristic",
        "runtime": "axil-core check_conflict + bge-small cosine (heuristic/ crate)",
        "U1": {"pairs": len(h),
               "verdicts": {v: sum(1 for x in h if x["verdict"] == v) for v in ["novel", "supersedes", "contradicts"]},
               "tpr": 0.0 if all(x["verdict"] == "novel" for x in h) else None,
               "share_similarity_ge_0.92": round(sum(x["similarity"] >= 0.92 for x in h) / len(h), 3),
               "note": "gate requires bge-small cosine >= 0.92; the consolidation.rs doc example (deploys 5pm -> 6pm) scores 0.914"},
        "U2": {"note": "no heuristic exists; errors have no resolved status"}}

    sets = HERE / "sets"
    out = {
        "benchmark": "typed-decisions-bakeoff",
        "phase": "29 increment 1: U1 supersession + U2 error-resolved",
        "date": date,
        "machine": onnx["machine"],
        "sets": {
            "U1": f"u1_lme.jsonl ({count(sets / 'u1_lme.jsonl')}, LongMemEval oracle, MIT) + "
                  f"u1_repo.jsonl ({count(sets / 'u1_repo.jsonl')}, dogfood)",
            "U2": f"u2_repo.jsonl ({count(sets / 'u2_repo.jsonl')}, dogfood) + "
                  f"u2_github.jsonl ({count(sets / 'u2_github.jsonl')}, GitHub issue -> PR links)",
            "labels": "LongMemEval pairs mechanical; repo pairs authored by Claude and blind-validated by "
                      "Codex gpt-6-sol + GLM glm-5.3-flash (validation/2026-09-27-labels.json)",
            "split": "40% train (threshold + Platt fit) / 60% test, sha256(id) based",
        },
        "metric_notes": {
            "at_threshold_test": f"highest-recall threshold with TRAIN FPR <= {onnx['target_fpr']}, applied to test",
            "U1_score": "P(contradiction | old, new) for NLI; noul for Laya",
            "U2_score": f"P(entailment | later, '{onnx['u2_template']}') for NLI; noul for Laya",
            "caveat": "U2 is small: directional only. Latency is Python-side, batch 1, model loaded.",
        },
        "arms": arms,
    }
    dest = HERE.parent / "results" / f"typed-decisions-bakeoff-{date}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
