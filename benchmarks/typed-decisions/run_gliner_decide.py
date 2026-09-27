#!/usr/bin/env python3
"""Second Jev-style arm of the Phase 29 bake-off: Fastino GLiNER2.5-Decide.

GLiNER2.5-Decide is an open-weight (Apache-2.0) classifier that scores labels
passed at call time in one forward pass, the same "typed decision" idea as
TypeSafe Jev and Laya. Each pair becomes one text plus a yes/no head, and
P(yes) is the score. Evaluated with the metrics in run_bakeoff.py (imported).

Usage: python run_gliner_decide.py [--model fastino/GLiNER2.5-Decide]
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import torch
from gliner2 import AutoExtractor

import run_bakeoff as rb

HEADS = {
    "U1": ("old_note_outdated", lambda r: f"Old note: {r['old']}\nNew note: {r['new']}"),
    "U2": ("error_fixed", lambda r: f"Error: {r['error']}\nLater change: {r['later']}"),
}


def p_yes(result, head):
    ans = result[head]
    conf = float(ans["confidence"])
    return conf if ans["label"] == "yes" else 1.0 - conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fastino/GLiNER2.5-Decide")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    model = AutoExtractor.from_pretrained(args.model)
    load_ms = (time.perf_counter() - t0) * 1000
    for _ in range(3):
        model.classify_text("Old note: warm\nNew note: up", {"old_note_outdated": ["yes", "no"]},
                            include_confidence=True)

    sets = {
        "U1": rb.load_set("u1_lme.jsonl") + rb.load_set("u1_repo.jsonl"),
        "U2": rb.load_set("u2_repo.jsonl") + rb.load_set("u2_github.jsonl"),
    }
    entry = {"model": args.model, "load_ms": round(load_ms, 1), "device": "cpu", "threads": args.threads}
    for dec, rows in sets.items():
        head, render = HEADS[dec]
        scores, lat = [], []
        for r in rows:
            t = time.perf_counter()
            res = model.classify_text(render(r), {head: ["yes", "no"]}, include_confidence=True)
            lat.append((time.perf_counter() - t) * 1000)
            scores.append(p_yes(res, head))
        res = rb.evaluate(scores, rows, [s >= 0.5 for s in scores])
        res["latency_ms_p50"] = round(statistics.median(lat), 2)
        res["latency_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2)
        res["question"] = f"head '{head}' in ['yes', 'no']"
        entry[dec] = res

    rb.OUT.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = rb.OUT / f"bakeoff-jevstyle-{args.model.replace('/', '_')}-{date}.json"
    path.write_text(json.dumps(entry, indent=2))
    for dec in ("U1", "U2"):
        r = entry[dec]
        t, a = r["at_threshold_test"], r["at_argmax_test"]
        print(f"{args.model:<34}{dec:<4} AUROC test {r['auroc_test']:.3f} all {r['auroc_all']:.3f} "
              f"by-source {({k: round(v, 3) for k, v in r['auroc_test_by_source'].items()})} | "
              f"thr TPR {t['tpr']:.3f} FPR {t['fpr']:.3f} | argmax TPR {a['tpr']:.3f} FPR {a['fpr']:.3f} | "
              f"p50 {r['latency_ms_p50']} ms")
    print(f"load {load_ms:.0f} ms; wrote {path.relative_to(rb.HERE)}")


if __name__ == "__main__":
    main()
