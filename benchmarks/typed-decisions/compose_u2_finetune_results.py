#!/usr/bin/env python3
"""Assemble the committed U2 fine-tune result from the run outputs.

Inputs (all gitignored, produced by the commands in README.md "U2 fine-tune"):
  out/bakeoff-<date>.json, out/bakeoff-laya-<...>.json   per-row scores
  out/u2-analysis-ref-<arm>.json                         analyze_u2_finetune.py, one per reference
  data/u2_train/harvest-meta.json                        harvest parameters and data hash
  data/u2_finetune/<arm>/finetune-meta.json, model.onnx  each training run
Every figure in the verdict is computed from those files here.

Usage: python compose_u2_finetune_results.py --date 2026-10-15 --machine "..." [--load-note "..."]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
BASE = "nli-deberta-v3-xsmall"
LAYA = "convaiinnovations/laya/typed-decisions"
SEEDS = ["v2-s13", "v2-s29", "v2-s47"]
ABLATION = "v2-s13-nosib"
PRIOR = HERE.parent / "results" / "typed-decisions-bakeoff-2026-09-27.json"


def load(rel):
    return json.loads((HERE / rel).read_text(encoding="utf-8"))


def strip_scores(entry):
    return {k: ({kk: vv for kk, vv in v.items() if kk != "scores"} if isinstance(v, dict) else v)
            for k, v in entry.items()}


def fmt_range(vals, digits=3):
    return f"{min(vals):.{digits}f} to {max(vals):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--machine", required=True, help="CPU / RAM / GPU description of the run machine")
    ap.add_argument("--load-note", default="")
    args = ap.parse_args()

    bake = load(f"out/bakeoff-{args.date}.json")
    laya = load(f"out/bakeoff-laya-{LAYA.replace('/', '_')}-{args.date}.json")
    vs_base = load(f"out/u2-analysis-ref-{BASE}.json")
    vs_laya = load(f"out/u2-analysis-ref-{LAYA.replace('/', '_')}.json")
    harvest = load("data/u2_train/harvest-meta.json")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))["arms"]

    entries = dict(bake["results"], **{LAYA: laya})
    test_ids = [r["id"] for f in ("u2_repo.jsonl", "u2_github.jsonl")
                for r in map(json.loads, (HERE / "sets" / f).read_text(encoding="utf-8").splitlines())
                if r["split"] == "test"]
    scores = {name: {i: e["U2"]["scores"][i] for i in test_ids} for name, e in entries.items()}

    runs = {}
    for arm in SEEDS + [ABLATION]:
        m = load(f"data/u2_finetune/{arm}/finetune-meta.json")
        runs[arm] = {k: m[k] for k in ("base_revision", "args", "n_train", "n_train_pos", "n_dev", "dev_auroc_before",
                                       "exported_dev_auroc", "exported_step", "train_seconds", "device", "gpu",
                                       "torch", "truncated_pairs", "onnx_max_abs_prob_diff_vs_torch",
                                       "pairs_sha256", "history")}
        runs[arm]["model_onnx_sha256"] = hashlib.sha256(
            (HERE / f"data/u2_finetune/{arm}/model.onnx").read_bytes()).hexdigest()

    a_base, a_laya = vs_base["arms"], vs_laya["arms"]
    lows = {s: a_base[s]["auroc_ci95_cluster"][0] for s in SEEDS}
    gain_base = {s: (a_base[s]["diff_vs_ref"], a_base[s]["diff_vs_ref_ci95"]) for s in SEEDS}
    gain_laya = {s: (a_laya[s]["diff_vs_ref"], a_laya[s]["diff_vs_ref_ci95"]) for s in SEEDS}
    laya_excl0 = sum(1 for _, ci in gain_laya.values() if ci[0] > 0)
    seed_sum = vs_base["seed_summary"]["v2-s"]
    axil = {n: a_base[n]["auroc_by_repo"]["axil-dogfood"] for n in [BASE, LAYA] + SEEDS}
    groups = {n: a_base[n]["fixed_vs_negative_group"] for n in [BASE, LAYA] + SEEDS}
    negs = vs_base["negatives_by_group"]
    u1 = {n: entries[n]["U1"]["auroc_test"] for n in [BASE] + SEEDS}
    fpr = {s: bake["results"][s]["U2"]["at_threshold_test"]["fpr"] for s in SEEDS}

    verdict = {
        "summary": (
            "The tuned model clearly beats the untuned base. It does not clear the 0.70 bar or the best "
            "prior arm (Laya typed-decisions) at 95%, and on the Axil-sourced slice Laya ranks first."),
        "bar_0_70": (
            f"not met at 95%: per-seed cluster lower bounds {', '.join(f'{v:.3f}' for v in lows.values())}"
            if any(v <= 0.70 for v in lows.values()) else
            f"met at 95%: per-seed cluster lower bounds {', '.join(f'{v:.3f}' for v in lows.values())}"),
        "u2_auroc_three_seeds": {"mean": seed_sum["auroc_mean"], "sd": seed_sum["auroc_sd"]},
        "vs_base": (f"paired gain {fmt_range([d for d, _ in gain_base.values()])}; lower bounds "
                    f"{fmt_range([ci[0] for _, ci in gain_base.values()])}, all above 0"
                    if all(ci[0] > 0 for _, ci in gain_base.values()) else
                    f"paired gain {fmt_range([d for d, _ in gain_base.values()])}; not every interval excludes 0"),
        "vs_laya": (f"paired gain {fmt_range([d for d, _ in gain_laya.values()])}; 95% intervals from "
                    f"{min(ci[0] for _, ci in gain_laya.values()):+.3f} to {max(ci[1] for _, ci in gain_laya.values()):+.3f}; "
                    f"{laya_excl0} of {len(SEEDS)} intervals exclude 0"),
        "axil_subset": {"note": f"{axil[BASE]['n']} test rows ({axil[BASE]['n_neg']} negatives): too few to "
                                "decide, but it is the only slice from Axil's own records",
                        "auroc": {n: v["auroc"] for n, v in axil.items()}},
        "by_negative_group": {"negatives": negs,
                              "auroc_fixed_vs_group": {n: {g: v["auroc"] for g, v in gs.items()} for n, gs in groups.items()},
                              "note": "state negatives (the same bug, not fixed) are the ones a deployed judge "
                                      f"meets after Axil's topic filter; there are {negs['state']}, no evidence either way"},
        "sibling_repo_ablation": (f"one seed: {a_base[ABLATION]['auroc']:.3f} without the test repos' sibling projects "
                                  f"against {a_base['v2-s13']['auroc']:.3f} with them; within seed noise "
                                  f"(sd {seed_sum['auroc_sd']:.3f})"),
        "u1_side_effect": (f"U1 AUROC {u1[BASE]:.3f} untuned, {fmt_range([u1[s] for s in SEEDS])} tuned: the tuned "
                           "model is U2-only, never a U1 judge"),
        "operating_point": (f"test FPR at the train-fit threshold {fmt_range(list(fpr.values()))}, above the 0.05 "
                            "target, and optimistic: test rows share issues with the rows the threshold is fit on. "
                            "A top-3 suggestion list per error would rely on per-error ranking, which pooled AUROC "
                            "does not measure; it is unmeasured"),
        "limits": (f"{vs_base['n_test']} test rows in {vs_base['n_clusters']} dependent clusters; "
                   f"{negs['topic']} of {sum(negs.values())} test negatives are fixes for another bug; repo-level "
                   "correlation is not modelled. Not for README, docs or CLI output (see the typed-decisions README "
                   "numbers policy)."),
    }

    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {
        "benchmark": "typed-decisions-u2-finetune",
        "date": args.date,
        "question": "Does fine-tuning nli-deberta-v3-xsmall on GitHub issue->PR pairs from held-out repos lift "
                    "U2 (error resolved?) above the best prior arm?",
        "code_commit": commit,
        "machine": {"platform": bake["machine"], "description": args.machine,
                    "onnxruntime_scoring": bake["onnxruntime"]},
        "machine_load_note": args.load_note,
        "base_model": {"repo": "cross-encoder/nli-deberta-v3-xsmall", "revision": runs[SEEDS[0]]["base_revision"]},
        "u2_template": bake["u2_template"],
        "training_data": {k: harvest[k] for k in ("args", "query_sha256_12", "positives_per_repo",
                                                  "dev_share_percent_of_repos", "pairs", "positives",
                                                  "by_split_kind", "dropped", "failed_repos", "pairs_sha256")},
        "training_repos": sorted(harvest["per_repo"]),
        "ablation_dropped_repos": runs[ABLATION]["args"]["drop_repos"],
        "runs": runs,
        "verdict": verdict,
        "reproduced_prior": {
            "source": str(PRIOR.relative_to(HERE.parent.parent)),
            "nli-deberta-v3-xsmall U2 auroc_test": [prior[BASE]["U2"]["auroc_test"], entries[BASE]["U2"]["auroc_test"]],
            "laya typed-decisions U2 auroc_test": [prior[LAYA]["U2"]["auroc_test"], laya["U2"]["auroc_test"]],
        },
        "bakeoff": {name: strip_scores(e) for name, e in entries.items()},
        "u2_test_paired_vs_base": vs_base,
        "u2_test_paired_vs_laya": vs_laya,
        "u2_test_scores": scores,
    }
    out = RESULTS / f"typed-decisions-u2-finetune-{args.date}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(HERE.parent.parent)}")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
