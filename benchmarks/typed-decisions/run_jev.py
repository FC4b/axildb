#!/usr/bin/env python3
"""Hosted Jev arm of the Phase 29 bake-off: TypeSafe Jev via its cloud API.

Jev is closed and API-only, so Axil can never ship it; this arm exists only to
measure how the original System One model compares with the open ones on the
same pairs. It asks exactly the questions run_laya.py asks (Laya copies Jev's
API), and evaluates with the metrics in run_bakeoff.py (imported).

This SENDS EVERY PAIR TO TYPESAFE'S SERVERS, including the texts drawn from
this repo's own memory (sets/u1_repo.jsonl, sets/u2_repo.jsonl). It refuses
to run without an API key and an explicit --confirm-external.

Usage:
  pip install typesafe-sdk
  export TYPESAFE_API_KEY=...
  python run_jev.py --confirm-external [--model jev-latest] [--concurrency 8]
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import run_bakeoff as rb
from run_laya import QUESTIONS


async def score_all(client, model, rows, dec, concurrency):
    (qid, q), = QUESTIONS[dec].items()
    from typesafe_sdk import Noul

    sem = asyncio.Semaphore(concurrency)
    scores, lat = [None] * len(rows), [None] * len(rows)

    async def one(i, r):
        state = {"old": r["old"], "new": r["new"]} if dec == "U1" else {"error": r["error"], "later": r["later"]}
        async with sem:
            t = time.perf_counter()
            res = await client.system_one(state, {qid: Noul(instructions=q["instructions"])}, model=model)
            lat[i] = (time.perf_counter() - t) * 1000
            scores[i] = float(res.nouls[qid].noul)

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return scores, lat, q["instructions"]


async def main_async(args):
    from typesafe_sdk import AsyncTypeSafeClient

    sets = {
        "U1": rb.load_set("u1_lme.jsonl") + rb.load_set("u1_repo.jsonl"),
        "U2": rb.load_set("u2_repo.jsonl") + rb.load_set("u2_github.jsonl"),
    }
    entry = {"model": f"typesafe/{args.model}", "device": "hosted API", "concurrency": args.concurrency}
    async with AsyncTypeSafeClient() as client:
        for dec, rows in sets.items():
            scores, lat, question = await score_all(client, args.model, rows, dec, args.concurrency)
            res = rb.evaluate(scores, rows, [s >= 0.5 for s in scores])
            res["latency_ms_p50"] = round(statistics.median(lat), 2)
            res["latency_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2)
            res["latency_note"] = "end-to-end HTTPS round trip incl. network, not comparable to local CPU"
            res["question"] = question
            entry[dec] = res
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--confirm-external", action="store_true",
                    help="acknowledge that every pair is sent to TypeSafe's API")
    args = ap.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY is not set (Jev is early-access: console.typesafe.ai)")
    if not args.confirm_external:
        sys.exit("refusing to send data to an external API without --confirm-external")

    entry = asyncio.run(main_async(args))
    rb.OUT.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = rb.OUT / f"bakeoff-jev-{args.model}-{date}.json"
    path.write_text(json.dumps(entry, indent=2))
    for dec in ("U1", "U2"):
        r = entry[dec]
        t = r["at_threshold_test"]
        print(f"{entry['model']:<24}{dec:<4} AUROC test {r['auroc_test']:.3f} all {r['auroc_all']:.3f} "
              f"by-source {({k: round(v, 3) for k, v in r['auroc_test_by_source'].items()})} | "
              f"thr TPR {t['tpr']:.3f} FPR {t['fpr']:.3f} | p50 {r['latency_ms_p50']} ms (network)")
    print(f"wrote {path.relative_to(rb.HERE)}")


if __name__ == "__main__":
    main()
