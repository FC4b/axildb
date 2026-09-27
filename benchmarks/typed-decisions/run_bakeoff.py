#!/usr/bin/env python3
"""Phase 29 bake-off: score ONNX pair classifiers on the U1/U2 decision sets.

For each candidate and set this reports, on the held-out `test` split:
  * AUROC (threshold-free ranking quality),
  * TPR / FPR / precision at the highest-recall threshold whose TRAIN FPR is
    <= 5%  (false positives are the costly error: a false supersede hides a
    correct memory; a false "fixed" hides an open error),
  * the same at the model's own argmax label,
  * ECE before and after Platt scaling fit on the train split,
  * per-kind accuracy at the fitted threshold (which negatives fool it),
  * CPU latency p50/p95 (batch 1, after warm-up), load time, file size.

Scores:
  U1  P(contradiction | premise=OLD, hypothesis=NEW); the `+sym` variant takes
      the max over both directions (two forward passes).
  U2  P(entailment | premise=LATER, hypothesis=U2_TEMPLATE.format(error)).

Usage: python run_bakeoff.py [--threads 4] [--only NAME ...]
Writes out/bakeoff-<utc-date>.json (gitignored) and prints a summary table.
"""

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
OUT = HERE / "out"

NLI3 = ["contradiction", "entailment", "neutral"]

CANDIDATES = {
    "nli-deberta-v3-xsmall": dict(repo="cross-encoder/nli-deberta-v3-xsmall", onnx="onnx/model.onnx", labels=NLI3),
    "nli-deberta-v3-xsmall-qint8": dict(repo="cross-encoder/nli-deberta-v3-xsmall", onnx="onnx/model_qint8_arm64.onnx", labels=NLI3),
    "ettinx-nli-xs": dict(repo="onnx-community/EttinX-nli-xs-ONNX", onnx="onnx/model.onnx", labels=NLI3),
    "ettinx-nli-xs-int8": dict(repo="onnx-community/EttinX-nli-xs-ONNX", onnx="onnx/model_int8.onnx", labels=NLI3),
}

U2_TEMPLATE = "This problem has been fixed: {}"
MAX_LEN = 512
TARGET_FPR = 0.05


def load_set(name):
    return [json.loads(line) for line in (SETS / name).read_text().splitlines() if line.strip()]


class PairClassifier:
    def __init__(self, repo, onnx_path, labels, threads):
        t0 = time.perf_counter()
        model_file = hf_hub_download(repo, onnx_path)
        tok_file = hf_hub_download(repo, "tokenizer.json")
        self.tok = Tokenizer.from_file(tok_file)
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.labels = labels
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.size_mb = os.path.getsize(model_file) / 1e6

    def probs(self, premise, hypothesis):
        enc = self.tok.encode(premise, hypothesis)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        logits = self.sess.run(None, feed)[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        return dict(zip(self.labels, e / e.sum()))


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rates(scores, labels, thr):
    tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
    p = sum(labels)
    n = len(labels) - p
    return {
        "tpr": tp / p if p else None,
        "fpr": fp / n if n else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
    }


def fit_threshold(scores, labels):
    """Lowest threshold (max recall) whose FPR on these pairs stays <= TARGET_FPR."""
    best = 1.0 + 1e-9
    for thr in sorted(set(scores), reverse=True):
        if rates(scores, labels, thr)["fpr"] <= TARGET_FPR:
            best = thr
        else:
            break
    return best


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_platt(scores, labels, iters=500, lr=0.1):
    """Two-parameter logistic calibration on logit(score), plain gradient descent."""
    a, b = 1.0, 0.0
    xs = [logit(s) for s in scores]
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, labels):
            q = 1 / (1 + math.exp(-(a * x + b)))
            ga += (q - y) * x
            gb += q - y
        a -= lr * ga / len(xs)
        b -= lr * gb / len(xs)
    return a, b


def ece(probs, labels, bins=10):
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(probs) if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[j] for j in idx) / len(idx)
            acc = sum(labels[j] for j in idx) / len(idx)
            total += len(idx) / len(probs) * abs(conf - acc)
    return total


def score_rows(clf, rows, decision, sym):
    scores, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        if decision == "U1":
            s = clf.probs(r["old"], r["new"])["contradiction"]
            if sym:
                s = max(s, clf.probs(r["new"], r["old"])["contradiction"])
        else:
            s = clf.probs(r["later"], U2_TEMPLATE.format(r["error"]))["entailment"]
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(s)
    return scores, lat


def evaluate(scores, rows, argmax_hits):
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    y = [r["label"] for r in rows]
    tr_s, tr_y = [scores[i] for i in train], [y[i] for i in train]
    te_s, te_y = [scores[i] for i in test], [y[i] for i in test]

    thr = fit_threshold(tr_s, tr_y)
    a, b = fit_platt(tr_s, tr_y)
    cal = [1 / (1 + math.exp(-(a * logit(s) + b))) for s in te_s]

    by_kind = {}
    for i in test:
        k = rows[i]["kind"]
        pred = 1 if scores[i] >= thr else 0
        hit = by_kind.setdefault(k, [0, 0])
        hit[0] += int(pred == y[i])
        hit[1] += 1

    te_argmax = [argmax_hits[i] for i in test]
    by_source = {}
    for i in test:
        by_source.setdefault(rows[i]["source"].split(":")[0], []).append(i)
    return {
        "n_train": len(train), "n_test": len(test),
        "auroc_test": auroc(te_s, te_y),
        "auroc_all": auroc(scores, y),
        "threshold_from_train": thr,
        "at_threshold_test": rates(te_s, te_y, thr),
        "at_argmax_test": rates([1.0 if h else 0.0 for h in te_argmax], te_y, 0.5),
        "ece_test_raw": ece(te_s, te_y),
        "ece_test_platt": ece(cal, te_y),
        "accuracy_by_kind_test": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())},
        "auroc_test_by_source": {src: auroc([scores[i] for i in idx], [y[i] for i in idx])
                                 for src, idx in sorted(by_source.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    sets = {
        "U1": load_set("u1_lme.jsonl") + load_set("u1_repo.jsonl"),
        "U2": load_set("u2_repo.jsonl") + load_set("u2_github.jsonl"),
    }
    report = {
        "benchmark": "typed-decisions-bakeoff",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": f"{platform.machine()} {platform.platform()}",
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "u2_template": U2_TEMPLATE,
        "target_fpr": TARGET_FPR,
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "results": {},
    }

    for name, spec in CANDIDATES.items():
        if args.only and name not in args.only:
            continue
        clf = PairClassifier(spec["repo"], spec["onnx"], spec["labels"], args.threads)
        for _ in range(5):
            clf.probs("warm up premise", "warm up hypothesis")
        entry = {"repo": spec["repo"], "onnx": spec["onnx"], "load_ms": round(clf.load_ms, 1),
                 "size_mb": round(clf.size_mb, 1)}
        for decision, rows in sets.items():
            variants = [False, True] if decision == "U1" else [False]
            for sym in variants:
                scores, lat = score_rows(clf, rows, decision, sym)
                target = "contradiction" if decision == "U1" else "entailment"
                if decision == "U1":
                    argmax_hits = [max(clf.probs(r["old"], r["new"]).items(), key=lambda kv: kv[1])[0] == target
                                   for r in rows]
                else:
                    argmax_hits = [max(clf.probs(r["later"], U2_TEMPLATE.format(r["error"])).items(),
                                       key=lambda kv: kv[1])[0] == target for r in rows]
                res = evaluate(scores, rows, argmax_hits)
                res["latency_ms_p50"] = round(statistics.median(lat), 2)
                res["latency_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2)
                entry[decision + ("+sym" if sym else "")] = res
        report["results"][name] = entry

    OUT.mkdir(exist_ok=True)
    path = OUT / f"bakeoff-{report['date_utc'][:10]}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"{'candidate':<30}{'dec':<8}{'AUROC':>7}{'TPR@thr':>9}{'FPR@thr':>9}{'ECEraw':>8}{'ECEcal':>8}{'p50ms':>8}")
    for name, entry in report["results"].items():
        for dec in ("U1", "U1+sym", "U2"):
            r = entry.get(dec)
            if not r:
                continue
            t = r["at_threshold_test"]
            fmt = lambda v: "  -" if v is None else f"{v:.3f}"
            print(f"{name:<30}{dec:<8}{fmt(r['auroc_test']):>7}{fmt(t['tpr']):>9}{fmt(t['fpr']):>9}"
                  f"{r['ece_test_raw']:>8.3f}{r['ece_test_platt']:>8.3f}{r['latency_ms_p50']:>8}")
    print(f"\nwrote {path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
