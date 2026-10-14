#!/usr/bin/env python3
"""Compare the importance two builds give the same records.

Each input is the output of `typed-decisions-regex-arm --records <export>`
from one build: one JSON line per record with the importance insert would
compute, the entities auto-linking would keep, and the entities the
importance scorer counts. The report says how far importance moved, how
many records crossed the 0.7 / 0.8 / 0.9 lines, and how the entity counts
changed.

usage: compare_importance.py <base.jsonl> <new.jsonl> [<new-with-gazetteer.jsonl>]
                             [--out <report.json>]
"""

import argparse
import json
import statistics as st
from collections import Counter


def load(path):
    with open(path) as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {r["id"]: r for r in rows}


def mean(values, places):
    return round(st.mean(values), places)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("new_gazetteer", nargs="?")
    ap.add_argument("--out")
    args = ap.parse_args()

    base, new = load(args.base), load(args.new)
    ids = [i for i in base if i in new]
    delta = [new[i]["importance"] - base[i]["importance"] for i in ids]
    tables = sorted({base[i]["table"] for i in ids})
    report = {
        "records": len(ids),
        "importance_mean_base": mean((base[i]["importance"] for i in ids), 4),
        "importance_mean_new": mean((new[i]["importance"] for i in ids), 4),
        "importance_mean_delta": mean(delta, 4),
        "importance_changed": sum(1 for d in delta if abs(d) > 1e-6),
        "importance_delta_histogram": {
            str(k): v for k, v in sorted(Counter(round(d, 2) for d in delta).items())
        },
        "importance_text_entities_mean_base": mean((base[i]["pure_entities"] for i in ids), 2),
        "importance_text_entities_mean_new": mean((new[i]["pure_entities"] for i in ids), 2),
        "auto_link_entities_mean_base": mean((base[i]["entities"] for i in ids), 2),
        "auto_link_entities_mean_new_patterns": mean((new[i]["entities"] for i in ids), 2),
        # compute_importance adds 0.05 per entity up to 0.2, so 4 entities is the cap.
        "at_entity_bonus_cap_base": sum(base[i]["pure_entities"] >= 4 for i in ids),
        "at_entity_bonus_cap_new": sum(new[i]["pure_entities"] >= 4 for i in ids),
        "per_table_mean_delta": {
            t: mean(
                (new[i]["importance"] - base[i]["importance"] for i in ids if base[i]["table"] == t),
                4,
            )
            for t in tables
        },
        "crossing": {
            str(t): {
                "up": sum(1 for i in ids if base[i]["importance"] < t <= new[i]["importance"]),
                "down": sum(1 for i in ids if new[i]["importance"] < t <= base[i]["importance"]),
            }
            for t in (0.7, 0.8, 0.9)
        },
    }
    if args.new_gazetteer:
        gaz = load(args.new_gazetteer)
        report["auto_link_entities_mean_new_with_gazetteer"] = mean(
            (gaz[i]["entities"] for i in ids if i in gaz), 2
        )
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
