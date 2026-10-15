#!/usr/bin/env python3
"""Paired, clustered comparison of U2 arms from bake-off reports with per-row scores.

Reads one or more out/bakeoff-*.json files written by run_bakeoff.py or
run_laya.py (both store `scores` per row id) and reports, on the U2 test split:
  * AUROC per arm with a cluster-bootstrap 95% interval (run_bakeoff.clusters_of);
  * the paired AUROC difference against a reference arm, with an interval from
    resampling the same clusters for both arms;
  * AUROC of fixed pairs against each kind of negative separately: topic
    negatives (a fix for a different bug) and status negatives (the same bug,
    not fixed: still broken, planned, not shipped, a workaround, a mention);
  * AUROC per source repo, and mean / sd over arms that share a prefix (seeds).

Usage: python analyze_u2_finetune.py --ref nli-deberta-v3-xsmall --seeds v2-s out/bakeoff-a.json [...] [--out FILE]
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import run_bakeoff as rb

TOPIC_NEGATIVES = {"other_fix", "other_fix_same_area", "unrelated_same_area"}
N_BOOT = 2000


def load_arms(paths):
    arms = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        entries = d["results"].items() if "results" in d else [(d["model"], d)]
        for name, entry in entries:
            if "U2" in entry and "scores" in entry["U2"]:
                arms[name] = entry["U2"]["scores"]
    return arms


def repo_of(row):
    src = row["source"]
    return src.split(":", 1)[1].split("#")[0] if src.startswith("github:") else src.split(":", 1)[0]


def subset_auroc(rows, scores, keep):
    idx = [i for i, r in enumerate(rows) if keep(r)]
    return rb.auroc([scores[i] for i in idx], [rows[i]["label"] for i in idx]), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--ref", required=True, help="reference arm for paired differences")
    ap.add_argument("--seeds", action="append", default=[], help="arm-name prefix to summarize as seeds")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = [r for r in rb.load_set("u2_repo.jsonl") + rb.load_set("u2_github.jsonl") if r["split"] == "test"]
    labels = [r["label"] for r in rows]
    arms = load_arms(args.reports)
    if args.ref not in arms:
        raise SystemExit(f"reference arm {args.ref!r} not in {sorted(arms)}")
    scores = {name: [s[r["id"]] for r in rows] for name, s in arms.items()}

    clusters = rb.clusters_of(rows)
    groups = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    keys = list(groups)
    rng = random.Random(0)
    resamples = [[i for _ in keys for i in groups[keys[rng.randrange(len(keys))]]] for _ in range(N_BOOT)]

    def ci(values):
        values = sorted(v for v in values if v is not None)
        return [round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values)) - 1], 4)]

    def boot(name):
        return [rb.auroc([scores[name][i] for i in pick], [labels[i] for i in pick]) for pick in resamples]

    ref_boot = boot(args.ref)
    ref_auc = rb.auroc(scores[args.ref], labels)
    report = {
        "set": "U2 test split (sets/u2_repo.jsonl + sets/u2_github.jsonl)",
        "n_test": len(rows), "n_test_pos": sum(labels), "n_clusters": len(keys),
        "bootstrap": f"{N_BOOT} cluster resamples, shared by every arm (paired)",
        "reference_arm": args.ref, "arms": {},
    }
    for name in sorted(scores):
        s = scores[name]
        b = ref_boot if name == args.ref else boot(name)
        diffs = [x - y for x, y in zip(b, ref_boot) if x is not None and y is not None]
        topic, n_topic = subset_auroc(rows, s, lambda r: r["label"] == 1 or r["kind"] in TOPIC_NEGATIVES)
        status, n_status = subset_auroc(rows, s, lambda r: r["label"] == 1 or r["kind"] not in TOPIC_NEGATIVES)
        by_repo = {}
        for repo in sorted({repo_of(r) for r in rows}):
            a, n = subset_auroc(rows, s, lambda r, repo=repo: repo_of(r) == repo)
            by_repo[repo] = {"auroc": None if a is None else round(a, 4), "n": n}
        report["arms"][name] = {
            "auroc": round(rb.auroc(s, labels), 4),
            "auroc_ci95_cluster": ci(b),
            "diff_vs_ref": None if name == args.ref else round(rb.auroc(s, labels) - ref_auc, 4),
            "diff_vs_ref_ci95": None if name == args.ref else ci(diffs),
            "auroc_fixed_vs_topic_negatives": {"auroc": round(topic, 4), "n": n_topic},
            "auroc_fixed_vs_status_negatives": {"auroc": round(status, 4), "n": n_status},
            "auroc_by_repo": by_repo,
        }
    for prefix in args.seeds:
        vals = [a["auroc"] for n, a in report["arms"].items() if n.startswith(prefix) and n.split(prefix, 1)[1].isdigit()]
        if vals:
            report.setdefault("seed_summary", {})[prefix] = {
                "arms": len(vals), "auroc_mean": round(statistics.mean(vals), 4),
                "auroc_sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None}

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
