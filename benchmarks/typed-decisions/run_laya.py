#!/usr/bin/env python3
"""Jev-style arm of the Phase 29 bake-off: Laya `noul` questions on U1/U2.

Scores each pair with one yes/no question over a JSON state and evaluates with
the same metrics as run_bakeoff.py (imported), so the numbers line up.
Runs on CPU (torch) to match the deployment target; Laya's published CPU
figure is 193-464 ms per call.

Usage: python run_laya.py [--model convaiinnovations/laya] [--subfolder multilingual]
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import laya
import torch

import run_bakeoff as rb

QUESTIONS = {
    "U1": {"supersedes": {
        "type": "noul",
        "instructions": "Does the NEW statement replace, update, or contradict the OLD statement, "
                        "so that the OLD statement is no longer current?",
    }},
    "U2": {"fixed": {
        "type": "noul",
        "instructions": "Does the LATER note show that the ERROR has actually been fixed?",
    }},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    kwargs = {"subfolder": args.subfolder} if args.subfolder else {}
    try:
        agent = laya.load(args.model, device="cpu", **kwargs)
    except TypeError:
        agent = laya.load(args.model, **kwargs)
    load_ms = (time.perf_counter() - t0) * 1000
    for _ in range(3):
        agent.predict({"old": "warm", "new": "up"}, QUESTIONS["U1"])

    sets = {
        "U1": rb.load_set("u1_lme.jsonl") + rb.load_set("u1_repo.jsonl"),
        "U2": rb.load_set("u2_repo.jsonl") + rb.load_set("u2_github.jsonl"),
    }
    name = args.model + (f"/{args.subfolder}" if args.subfolder else "")
    entry = {"model": name, "load_ms": round(load_ms, 1), "device": "cpu", "threads": args.threads}
    for dec, rows in sets.items():
        (qid, q), = QUESTIONS[dec].items()
        scores, lat = [], []
        for r in rows:
            state = {"old": r["old"], "new": r["new"]} if dec == "U1" else {"error": r["error"], "later": r["later"]}
            t = time.perf_counter()
            ans = agent.predict(state, {qid: q})["answers"][qid]
            lat.append((time.perf_counter() - t) * 1000)
            scores.append(float(ans["noul"]))
        res = rb.evaluate(scores, rows, [s >= 0.5 for s in scores])
        res["latency_ms_p50"] = round(statistics.median(lat), 2)
        res["latency_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2)
        res["question"] = q["instructions"]
        entry[dec] = res

    rb.OUT.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = name.replace("/", "_")
    path = rb.OUT / f"bakeoff-laya-{slug}-{date}.json"
    path.write_text(json.dumps(entry, indent=2))
    for dec in ("U1", "U2"):
        r = entry[dec]
        t = r["at_threshold_test"]
        a = r["at_argmax_test"]
        print(f"{name:<44}{dec:<4} AUROC test {r['auroc_test']:.3f} all {r['auroc_all']:.3f} | "
              f"thr TPR {t['tpr']:.3f} FPR {t['fpr']:.3f} | argmax TPR {a['tpr']:.3f} FPR {a['fpr']:.3f} | "
              f"ECE {r['ece_test_raw']:.3f}->{r['ece_test_platt']:.3f} | p50 {r['latency_ms_p50']} ms")
    print(f"load {load_ms:.0f} ms; wrote {path.relative_to(rb.HERE)}")


if __name__ == "__main__":
    main()
