#!/usr/bin/env python3
"""Score the production pattern arms alone, for pattern development.

Runs regex-arm/ (the axil-core pattern code without the model crates) on the
U5 entity set and the U3 capture set and scores it with the bake-off scripts'
own code (`run_u5_bakeoff`, `run_u3_bakeoff`), so a number here is the number
the bake-off would report for the same build.

Develop on train; score test once per finished pattern family:

  python run_regex_arm.py --build                       # both sets, train split
  python run_regex_arm.py u5 --errors                   # train misses and wrong mentions
  python run_regex_arm.py u5 --project-dir ../..        # with the project gazetteer
  python run_regex_arm.py --split test --out <file>     # the once-per-family test score

`--errors` is refused on test. Raw output goes to out/regex-arm/ (gitignored).
"""

import argparse
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = HERE / "regex-arm"
ARM_BIN = ARM / "target" / "debug" / "typed-decisions-regex-arm"
OUT = HERE / "out" / "regex-arm"

sys.path.insert(0, str(HERE))


def load_u3_module():
    # numpy is only needed by the LR heads, not by the scoring used here.
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.modules["numpy"] = types.ModuleType("numpy")
    return importlib.import_module("run_u3_bakeoff")


def build():
    subprocess.run(["cargo", "build", "--manifest-path", str(ARM / "Cargo.toml")], check=True)


def run_arm(mode, set_path, project_dir=None):
    cmd = [str(ARM_BIN), mode, str(set_path)]
    if project_dir:
        cmd += ["--project-dir", str(project_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def u5(args):
    m = importlib.import_module("run_u5_bakeoff")
    rows = m.load_rows()
    gold = m.gold_forms(rows)
    ids = [r["id"] for r in rows if r["split"] == args.split]
    raw = run_arm("--entities", m.SET, args.project_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"u5-{args.split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    preds = {r["id"]: m.to_forms([[e["text"], m.production_type(e["axil_type"], e["text"]), 1.0]
                                  for e in r["entities"]]) for r in raw}
    junk_words = m.load_junk_list()
    wrong = [(i, f) for i in ids for f in preds.get(i, {}) if f not in gold[i]]
    junk = [(i, f) for i, f in wrong if m.is_junk(f, junk_words)]
    n_pred = sum(len(preds.get(i, {})) for i in ids)
    result = {
        "split": args.split,
        "texts": len(ids),
        "span": m.score(preds, gold, ids),
        "span_and_type": m.score(preds, gold, ids, typed=True),
        "per_type": {t: m.score(preds, gold, ids, type_filter=t) for t in m.TYPES},
        "junk": {"junk_mentions": len(junk), "wrong_mentions": len(wrong), "predicted_mentions": n_pred,
                 "junk_rate": round(len(junk) / n_pred, 4) if n_pred else 0.0,
                 "junk_forms": sorted(f for _, f in junk)},
    }
    s = result["span"]
    print(f"u5 {args.split}: P {s['precision']:.4f} R {s['recall']:.4f} F1 {s['f1']:.4f} "
          f"(tp {s['tp']} pred {s['pred']} gold {s['gold']}) junk {result['junk']['junk_rate']:.4f} "
          f"{result['junk']['junk_forms']}")
    if args.errors:
        if args.split != "train":
            sys.exit("--errors is for the train split only")
        text = {r["id"]: r["text"] for r in rows}
        for i in ids:
            miss = [f"{f}:{t}" for f, t in gold[i].items() if f not in preds.get(i, {})]
            extra = [f for f in preds.get(i, {}) if f not in gold[i]]
            if miss or extra:
                print(f"\n== {i}: {text[i][:args.width]!r}")
                if miss:
                    print("  MISS ", miss)
                if extra:
                    print("  EXTRA", extra)
    return result


def u3(args):
    m = load_u3_module()
    rows = m.load_rows()
    gold = {r["id"]: r["class"] for r in rows}
    gold_rows = {r["id"]: r for r in rows}
    train = m.split_ids(rows, "train")
    ids = m.split_ids(rows, args.split)
    raw = {r["id"]: r for r in run_arm("--capture", m.SET)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"u3-{args.split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw.values()))
    pred, score, _ = m.production_view(raw, m.HOOK_MIN_CONFIDENCE)
    cuts = m.fit_cuts(score, gold_rows, train)
    bucket = {i: m.bucket_of(score[i], cuts["cuts"]) for i in score}
    cm = m.class_metrics(pred, gold, ids)
    result = {
        "split": args.split,
        "min_confidence": m.HOOK_MIN_CONFIDENCE,
        **{k: cm[k] for k in ("items", "accuracy", "macro_f1", "false_capture_rate", "false_captures",
                              "gold_none", "wrong_table", "missed", "stored")},
        "per_class": cm["per_class"],
        "importance": m.importance_metrics(bucket, gold_rows, ids),
        "importance_cuts": cuts,
        "importance_majority_baseline": m.importance_baseline(gold_rows, train, ids),
    }
    print(f"u3 {args.split}: acc {cm['accuracy']} macro-F1 {cm['macro_f1']} FC {cm['false_capture_rate']} "
          f"({cm['false_captures']}/{cm['gold_none']}) stored {cm['stored']} missed {cm['missed']} "
          f"wrong-table {cm['wrong_table']} importance acc {result['importance']['accuracy']}")
    for c in m.CLASSES:
        pc = cm["per_class"][c]
        print(f"   {c:9s} P {pc['precision']} R {pc['recall']} (tp {pc['tp']} pred {pc['pred']} gold {pc['gold']})")
    if args.errors:
        if args.split != "train":
            sys.exit("--errors is for the train split only")
        for i in ids:
            if pred[i] != gold[i]:
                caps = [(c["type"], c["confidence"], c["summary"][:80]) for c in raw[i]["captures"]]
                print(f"\n== {i} [{gold_rows[i]['source']}] gold {gold[i]} pred {pred[i]} caps {caps}")
                print("  ", gold_rows[i]["text"][:args.width].replace("\n", "\n   "))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", choices=["u5", "u3"], default=[])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--build", action="store_true", help="cargo build regex-arm/ first")
    ap.add_argument("--errors", action="store_true", help="print train misses and wrong outputs")
    ap.add_argument("--project-dir", help="U5: extract with the gazetteer of this project directory")
    ap.add_argument("--width", type=int, default=400, help="text characters shown by --errors")
    ap.add_argument("--out", help="write the metrics JSON here")
    args = ap.parse_args()
    if args.build:
        build()
    results = {}
    for name in args.sets or ["u5", "u3"]:
        results[name] = u5(args) if name == "u5" else u3(args)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
