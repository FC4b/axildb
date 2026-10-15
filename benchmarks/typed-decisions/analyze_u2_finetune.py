#!/usr/bin/env python3
"""Paired, clustered comparison of U2 arms from bake-off reports with per-row scores.

Reads one or more out/bakeoff-*.json files written by run_bakeoff.py or
run_laya.py (both store `scores` per row id) and reports, on the U2 test split:
  * AUROC per arm with a cluster-bootstrap 95% interval (run_bakeoff.clusters_of)
    and whether that interval's lower bound clears --bar;
  * the paired AUROC difference against a reference arm, with an interval from
    resampling the same clusters for both arms;
  * AUROC of the fixed pairs against each kind of negative separately:
      topic   - a fix for a different bug, picked for word overlap;
      mention - a PR that cross-references the issue without fixing it (same
                repo or downstream); often off-topic for the bug itself;
      state   - the same bug, not fixed: still broken, planned, merged but not
                shipped, a lexical trap. Only a handful exist in the sets;
  * AUROC per source repo, and mean / sd over arms that share a prefix (seeds).

The bootstrap treats clusters as independent; rows from one repo are still
correlated beyond that, so intervals are, if anything, too narrow.

Usage: python analyze_u2_finetune.py --ref nli-deberta-v3-xsmall --seeds v2-s out/bakeoff-a.json [...] [--out FILE]
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import run_bakeoff as rb

NEGATIVE_GROUPS = {
    "topic": {"other_fix", "other_fix_same_area", "unrelated_same_area"},
    "mention": {"mentioned_not_fix", "downstream_workaround"},
    "state": {"still_broken", "planned", "merged_not_shipped", "lexical_trap"},
}
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
    a = rb.auroc([scores[i] for i in idx], [rows[i]["label"] for i in idx])
    return {"auroc": None if a is None else round(a, 4), "n": len(idx),
            "n_neg": sum(1 for i in idx if rows[i]["label"] == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--ref", required=True, help="reference arm for paired differences")
    ap.add_argument("--seeds", action="append", default=[], help="arm-name prefix to summarize as seeds")
    ap.add_argument("--bar", type=float, default=0.70, help="AUROC the interval's lower bound is checked against")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = [r for r in rb.load_set("u2_repo.jsonl") + rb.load_set("u2_github.jsonl") if r["split"] == "test"]
    labels = [r["label"] for r in rows]
    grouped = set().union(*NEGATIVE_GROUPS.values())
    ungrouped = sorted({r["kind"] for r in rows if r["label"] == 0 and r["kind"] not in grouped})
    if ungrouped:
        raise SystemExit(f"negative kinds with no group: {ungrouped}")
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
        "negatives_by_group": {g: sum(1 for r in rows if r["label"] == 0 and r["kind"] in kinds)
                               for g, kinds in NEGATIVE_GROUPS.items()},
        "bootstrap": f"{N_BOOT} cluster resamples, shared by every arm (paired); repo-level correlation not modelled",
        "bar": args.bar, "reference_arm": args.ref, "arms": {},
    }
    for name in sorted(scores):
        s = scores[name]
        b = ref_boot if name == args.ref else boot(name)
        diffs = [x - y for x, y in zip(b, ref_boot) if x is not None and y is not None]
        interval = ci(b)
        entry = {
            "auroc": round(rb.auroc(s, labels), 4),
            "auroc_ci95_cluster": interval,
            "lower_bound_clears_bar": interval[0] > args.bar,
            "diff_vs_ref": None if name == args.ref else round(rb.auroc(s, labels) - ref_auc, 4),
            "diff_vs_ref_ci95": None if name == args.ref else ci(diffs),
            "fixed_vs_negative_group": {
                g: subset_auroc(rows, s, lambda r, kinds=kinds: r["label"] == 1 or r["kind"] in kinds)
                for g, kinds in NEGATIVE_GROUPS.items()},
            "auroc_by_repo": {repo: subset_auroc(rows, s, lambda r, repo=repo: repo_of(r) == repo)
                              for repo in sorted({repo_of(r) for r in rows})},
        }
        report["arms"][name] = entry
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
