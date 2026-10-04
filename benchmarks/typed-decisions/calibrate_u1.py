#!/usr/bin/env python3
"""Calibrate the U1 judge: one temperature and a low-FPR threshold.

The U1 judge scores P(contradiction | premise=OLD, hypothesis=NEW) with
cross-encoder/nli-deberta-v3-xsmall (dynamic int8 ONNX), one pair per run.
Raw NLI probabilities are over-confident, so the runtime divides the
one-vs-rest logit z_c - logsumexp(z_other) by a temperature T before the
sigmoid, and flags a pair when that calibrated probability reaches a threshold.
At T = 1 this is exactly the softmax probability, and for any T it ranks pairs
the same way. Dividing all three logits by T before the softmax does not: it
moves mass between entailment and neutral and reorders pairs, so it is
reported as a comparison only, like Platt scaling.

Both are fitted on the sets' `train` split, used here as the validation split:
  * T minimises the binary log-loss on validation;
  * the threshold is the lowest (highest-recall) one whose validation FPR has
    a one-sided 95% Clopper-Pearson upper bound <= 5%. The bake-off's rule
    (validation FPR <= 5%) is reported next to it.

The report labels each group of numbers with its split (`splits` in the JSON):
  * test only: the quality numbers (AUROC, ECE, NLL, Brier, TPR/FPR at each
    threshold, accuracy by kind, AUROC by source);
  * validation only: what the fit saw (`validation`, and each threshold rule's
    `validation` rates);
  * all 249 pairs (validation + test): checks that compare two ways of
    scoring the same pairs, not quality (onnxruntime drift, session settings,
    batching), and the Python latency.

The model and tokenizer are pinned by revision and sha256. Score with the same
onnxruntime version and graph optimisation level as the Rust runtime (ort
2.0.0-rc.10 bundles 1.22.0; Level3 = ORT_ENABLE_ALL): this model's logits move
between onnxruntime versions (`onnxruntime_drift` in the report, with
--compare) and between optimisation levels (`session_settings_check`), so a
calibration is only valid for the scores the runtime will produce.

Usage:
  python calibrate_u1.py --scores-only            # dump out/u1-scores-ort<ver>.jsonl
  python calibrate_u1.py --compare out/u1-scores-ort1.30.0.jsonl
Writes out/u1-scores-ort<ver>.jsonl (gitignored),
benchmarks/results/typed-decisions-u1-calibration-<date>.json, the runtime file
crates/extensions/axil-decide/calibration/u1-nli-deberta-v3-xsmall-qint8.json
and the Rust parity fixture crates/extensions/axil-decide/tests/fixtures/u1_parity.json.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SETS = HERE / "sets"
OUT = HERE / "out"
CRATE = ROOT / "crates" / "extensions" / "axil-decide"

MODEL = {
    "name": "nli-deberta-v3-xsmall-qint8",
    "repo": "cross-encoder/nli-deberta-v3-xsmall",
    "revision": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
    "onnx": "onnx/model_qint8_arm64.onnx",
    "onnx_sha256": "ef41fe8474f6070373f97fbe8362461d55108f387723280fcdb64f8015133db2",
    "tokenizer_sha256": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "labels": ["contradiction", "entailment", "neutral"],
    "max_len": 512,
}
LABEL = "contradiction"
TARGET_FPR = 0.05
CONFIDENCE = 0.95
THREADS = 4


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows():
    rows = []
    for name in ("u1_lme.jsonl", "u1_repo.jsonl"):
        rows += [json.loads(line) for line in (SETS / name).read_text().splitlines() if line.strip()]
    return rows


def pinned(path_in_repo, expected):
    path = hf_hub_download(MODEL["repo"], path_in_repo, revision=MODEL["revision"])
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{path_in_repo}: sha256 {actual} != pinned {expected}")
    return path


def score_all(rows, threads=THREADS, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    """Raw logits per pair, batch 1, no padding.

    Batching changes this model's output: dynamic int8 quantization takes one
    activation scale over the whole batch tensor, padding included.
    """
    tok = Tokenizer.from_file(pinned("tokenizer.json", MODEL["tokenizer_sha256"]))
    tok.enable_truncation(max_length=MODEL["max_len"])
    tok.no_padding()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = level
    model_path = pinned(MODEL["onnx"], MODEL["onnx_sha256"])
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    names = {i.name for i in sess.get_inputs()}
    out, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        enc = tok.encode(r["old"], r["new"])
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        logits = sess.run(None, feed)[0][0]
        lat.append((time.perf_counter() - t0) * 1000)
        out.append({"id": r["id"], "tokens": len(enc.ids), "logits": [float(x) for x in logits]})
    return out, load_ms, lat


def score_batched(rows, batch=8):
    """Raw logits per pair from padded batches of `batch`, shortest pairs
    first (how a runtime would batch a batch-invariant model)."""
    tok = Tokenizer.from_file(pinned("tokenizer.json", MODEL["tokenizer_sha256"]))
    tok.enable_truncation(max_length=MODEL["max_len"])
    tok.no_padding()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(pinned(MODEL["onnx"], MODEL["onnx_sha256"]), opts,
                                providers=["CPUExecutionProvider"])
    encs = [tok.encode(r["old"], r["new"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(encs[i].ids))
    out = {}
    for start in range(0, len(order), batch):
        group = order[start:start + batch]
        width = max(len(encs[i].ids) for i in group)
        ids = np.zeros((len(group), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, i in enumerate(group):
            ids[row, :len(encs[i].ids)] = encs[i].ids
            mask[row, :len(encs[i].ids)] = 1
        for i, logits in zip(group, sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]):
            out[rows[i]["id"]] = [float(x) for x in logits]
    return out


def ovr_logit(logits):
    """z_label - logsumexp(other logits): the log-odds of the label."""
    z = np.asarray(logits, dtype=np.float64)
    i = MODEL["labels"].index(LABEL)
    rest = np.delete(z, i)
    m = rest.max()
    return float(z[i] - (m + math.log(np.exp(rest - m).sum())))


def sigmoid(u):
    return 1 / (1 + math.exp(-u)) if u >= 0 else math.exp(u) / (1 + math.exp(u))


def prob(logits, temperature=1.0):
    """Shipped calibration: sigmoid(one-vs-rest logit / T)."""
    return sigmoid(ovr_logit(logits) / temperature)


def prob_softmax_t(logits, temperature):
    """Comparison only: softmax(logits / T)[label]."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return float(e[MODEL["labels"].index(LABEL)] / e.sum())


def nll(probs, labels):
    eps = 1e-12
    return -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                for p, y in zip(probs, labels)) / len(probs)


def brier(probs, labels):
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def ece(probs, labels, bins=10):
    """Same binning as run_bakeoff.py: 10 equal-width bins."""
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(probs) if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[j] for j in idx) / len(idx)
            acc = sum(labels[j] for j in idx) / len(idx)
            total += len(idx) / len(probs) * abs(conf - acc)
    return total


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fit_temperature(logits, labels, score=prob):
    """T minimising binary log-loss: a log-spaced grid, then golden-section."""
    def loss(log_t):
        return nll([score(z, math.exp(log_t)) for z in logits], labels)

    grid = [-3 + 6 * i / 600 for i in range(601)]
    best = min(grid, key=loss)
    lo, hi = best - 0.01, best + 0.01
    g = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if loss(a) < loss(b):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_platt(probs, labels):
    """sigmoid(a * logit(p) + b) by damped Newton steps on the log-loss."""
    xs = np.array([logit(p) for p in probs])
    y = np.array(labels, dtype=np.float64)

    def loss(a, b):
        u = a * xs + b
        return float(np.mean(np.logaddexp(0, u) - y * u))

    a, b = 1.0, 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-np.clip(a * xs + b, -500, 500)))
        g = np.array([np.sum((q - y) * xs), np.sum(q - y)])
        w = q * (1 - q)
        h = np.array([[np.sum(w * xs * xs), np.sum(w * xs)], [np.sum(w * xs), np.sum(w)]])
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        scale = 1.0
        while loss(a - scale * step[0], b - scale * step[1]) > loss(a, b) and scale > 1e-6:
            scale /= 2
        a, b = a - scale * step[0], b - scale * step[1]
        if np.abs(scale * step).max() < 1e-10:
            break
    return float(a), float(b)


def platt(p, a, b):
    return sigmoid(a * logit(p) + b)


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k, n, confidence=CONFIDENCE):
    """One-sided Clopper-Pearson upper bound on a rate with k hits in n."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rates(scores, labels, thr):
    tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
    pos = sum(labels)
    neg = len(labels) - pos
    return {"tp": tp, "fp": fp, "positives": pos, "negatives": neg,
            "tpr": tp / pos, "tpr_wilson95": wilson(tp, pos),
            "fpr": fp / neg, "fpr_wilson95": wilson(fp, neg),
            "precision": tp / (tp + fp) if tp + fp else None}


def fit_threshold(scores, labels, accept):
    """Lowest threshold whose validation (fp, negatives) satisfies `accept`.

    Walks the distinct scores downward and stops at the first violation, then
    returns the midpoint to the next lower score, so a test score a hair away
    from a validation score cannot flip the decision.
    """
    neg = len(labels) - sum(labels)
    distinct = sorted(set(scores), reverse=True)
    chosen = None
    for i, thr in enumerate(distinct):
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
        if not accept(fp, neg):
            break
        chosen = i
    if chosen is None:
        return 1.0
    nxt = distinct[chosen + 1] if chosen + 1 < len(distinct) else 0.0
    return (distinct[chosen] + nxt) / 2


def bootstrap_ece(probs_by_method, labels, n=2000, seed=29):
    rng = random.Random(seed)
    idx = list(range(len(labels)))
    draws = {m: [] for m in probs_by_method}
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        ys = [labels[i] for i in sample]
        for m, ps in probs_by_method.items():
            draws[m].append(ece([ps[i] for i in sample], ys))
    out = {}
    for m, xs in draws.items():
        xs.sort()
        out[m] = [round(xs[int(0.025 * n)], 4), round(xs[int(0.975 * n) - 1], 4)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-only", action="store_true", help="only dump raw logits")
    ap.add_argument("--compare", help="raw logits from another onnxruntime version, for drift")
    args = ap.parse_args()

    rows = load_rows()
    scored, load_ms, lat = score_all(rows)
    OUT.mkdir(exist_ok=True)
    raw_path = OUT / f"u1-scores-ort{ort.__version__}.jsonl"
    raw_path.write_text("".join(json.dumps(s) + "\n" for s in scored))
    print(f"wrote {raw_path.relative_to(HERE)}")
    if args.scores_only:
        return

    by_id = {s["id"]: s for s in scored}
    val = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    z_val = [by_id[r["id"]]["logits"] for r in val]
    z_test = [by_id[r["id"]]["logits"] for r in test]
    y_val = [r["label"] for r in val]
    y_test = [r["label"] for r in test]

    t = fit_temperature(z_val, y_val)
    raw_val = [prob(z) for z in z_val]
    raw_test = [prob(z) for z in z_test]
    cal_val = [prob(z, t) for z in z_val]
    cal_test = [prob(z, t) for z in z_test]
    pa, pb = fit_platt(raw_val, y_val)
    platt_test = [platt(p, pa, pb) for p in raw_test]
    t_soft = fit_temperature(z_val, y_val, score=prob_softmax_t)
    soft_test = [prob_softmax_t(z, t_soft) for z in z_test]

    rules = {
        "val_fpr_cp95_upper_le_0.05": lambda fp, n: cp_upper(fp, n) <= TARGET_FPR,
        "val_fpr_le_0.05": lambda fp, n: fp / n <= TARGET_FPR,
    }
    chosen_rule = "val_fpr_cp95_upper_le_0.05"
    operating = {}
    for name, accept in rules.items():
        thr = fit_threshold(cal_val, y_val, accept)
        operating[name] = {"threshold": thr, "validation": rates(cal_val, y_val, thr),
                           "test": rates(cal_test, y_test, thr)}
    threshold = operating[chosen_rule]["threshold"]

    by_kind, by_source = {}, {}
    for r, p in zip(test, cal_test):
        hit = by_kind.setdefault(r["kind"], [0, 0])
        hit[0] += int((p >= threshold) == (r["label"] == 1))
        hit[1] += 1
        by_source.setdefault(r["source"].split(":")[0], []).append((p, r["label"]))

    drift = None
    if args.compare:
        other = {json.loads(line)["id"]: json.loads(line) for line in Path(args.compare).read_text().splitlines()}
        dl = max(abs(a - b) for s in scored for a, b in zip(s["logits"], other[s["id"]]["logits"]))
        dp = max(abs(prob(s["logits"], t) - prob(other[s["id"]]["logits"], t)) for s in scored)
        flips = sum((prob(s["logits"], t) >= threshold) != (prob(other[s["id"]]["logits"], t) >= threshold)
                    for s in scored)
        drift = {"against": Path(args.compare).name, "split": "all", "pairs": len(scored),
                 "max_abs_logit_diff": dl, "max_abs_calibrated_p_diff": dp, "threshold_flips": flips}

    # How much the session settings move the scores, against the run above
    # (ORT_ENABLE_ALL, THREADS threads): why the runtime pins Level3 and why
    # its thread count is free.
    level = ort.GraphOptimizationLevel
    variants = {f"optimization {lv.name}": dict(level=lv)
                for lv in (level.ORT_DISABLE_ALL, level.ORT_ENABLE_BASIC, level.ORT_ENABLE_EXTENDED)}
    variants.update({f"threads {n}": dict(threads=n) for n in (1, 2, 8)})
    session_check = {"against": f"ORT_ENABLE_ALL, {THREADS} threads", "split": "all", "pairs": len(scored)}
    for name, kwargs in variants.items():
        other = {o["id"]: o for o in score_all(rows, **kwargs)[0]}
        session_check[name] = {
            "max_abs_logit_diff": max(abs(a - b) for s_ in scored
                                      for a, b in zip(s_["logits"], other[s_["id"]]["logits"])),
            "threshold_flips": sum((prob(s_["logits"], t) >= threshold)
                                   != (prob(other[s_["id"]]["logits"], t) >= threshold) for s_ in scored),
        }

    # Dynamic int8 quantization takes one activation scale per input tensor,
    # so batching changes every pair's score: why the runtime scores this
    # model one pair per call.
    batched = score_batched(rows)
    batch_check = {
        "what": "padded batches of 8, shortest pairs first, against one pair per run",
        "split": "all",
        "pairs": len(scored),
        "max_abs_logit_diff": max(abs(a - b) for s_ in scored for a, b in zip(s_["logits"], batched[s_["id"]])),
        "max_abs_calibrated_p_diff": max(abs(prob(s_["logits"], t) - prob(batched[s_["id"]], t)) for s_ in scored),
        "threshold_flips": sum((prob(s_["logits"], t) >= threshold) != (prob(batched[s_["id"]], t) >= threshold)
                               for s_ in scored),
    }

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    runtime = {
        "decision": "U1",
        "signal": "possible contradiction for review: P(contradiction | premise=OLD, hypothesis=NEW); "
                  "not an update detector, it flags few plain updates",
        "model": MODEL["name"],
        "repo": MODEL["repo"],
        "revision": MODEL["revision"],
        "premise": "old",
        "hypothesis": "new",
        "label": LABEL,
        "calibration": "sigmoid((z_label - logsumexp(z_other)) / temperature)",
        "temperature": round(t, 6),
        "threshold": round(threshold, 6),
        "threshold_rule": f"lowest threshold whose validation FPR has a one-sided {CONFIDENCE:.0%} "
                          f"Clopper-Pearson upper bound <= {TARGET_FPR}",
        "fitted_on": f"U1 train split ({len(val)} pairs)",
        # What the scores were produced with. The runtime refuses this
        # calibration for a model or session that differs in any of them.
        "onnx_sha256": MODEL["onnx_sha256"],
        "tokenizer_sha256": MODEL["tokenizer_sha256"],
        "max_len": MODEL["max_len"],
        "onnxruntime": ort.__version__,
        "optimization_level": "ORT_ENABLE_ALL",
        "one_pair_per_call": True,
        "source": f"benchmarks/results/typed-decisions-u1-calibration-{date}.json",
    }
    # The runtime compares against the rounded values; re-derive the test
    # numbers from them so the report matches what ships.
    t, threshold = runtime["temperature"], runtime["threshold"]
    cal_test = [prob(z, t) for z in z_test]
    shipped = rates(cal_test, y_test, threshold)

    report = {
        "benchmark": "typed-decisions-u1-calibration",
        "date": date,
        "splits": {
            "test": "test, threshold.at_shipped_threshold_test, threshold.rules.*.test, "
                    "accuracy_by_kind_test, auroc_test_by_source",
            "validation": "validation, threshold.rules.*.validation (the fit's own data)",
            "all_249_pairs": "onnxruntime_drift, session_settings_check, batch_check, python_latency "
                             "(agreement between two ways of scoring the same pairs, not quality)",
        },
        "machine": f"{platform.machine()} {platform.platform()}",
        "onnxruntime": ort.__version__,
        "threads": THREADS,
        "model": {k: MODEL[k] for k in ("name", "repo", "revision", "onnx", "onnx_sha256", "tokenizer_sha256",
                                         "labels", "max_len")},
        "score": "sigmoid((z_contradiction - logsumexp(z_entailment, z_neutral)) / T), premise=old, "
                 "hypothesis=new, batch 1, no padding, ORT_ENABLE_ALL",
        "sets": "u1_lme.jsonl + u1_repo.jsonl; split field: train = validation (fit), test = report",
        "n_validation": {"pairs": len(val), "positives": sum(y_val), "negatives": len(y_val) - sum(y_val)},
        "n_test": {"pairs": len(test), "positives": sum(y_test), "negatives": len(y_test) - sum(y_test)},
        "temperature": runtime["temperature"],
        "comparisons": {
            "platt": {"a": pa, "b": pb, "note": "sigmoid(a * logit(p_raw) + b); not shipped"},
            "softmax_temperature": {"temperature": t_soft,
                                    "note": "softmax(logits / T)[contradiction]; not shipped: it reorders pairs"},
        },
        "test": {
            "auroc_raw": auroc(raw_test, y_test),
            "auroc_calibrated": auroc(cal_test, y_test),
            "auroc_softmax_temperature": auroc(soft_test, y_test),
            "ece_raw": ece(raw_test, y_test),
            "ece_temperature": ece(cal_test, y_test),
            "ece_platt": ece(platt_test, y_test),
            "ece_softmax_temperature": ece(soft_test, y_test),
            "ece_bootstrap95": bootstrap_ece({"raw": raw_test, "temperature": cal_test, "platt": platt_test,
                                              "softmax_temperature": soft_test}, y_test),
            "nll": {"raw": nll(raw_test, y_test), "temperature": nll(cal_test, y_test),
                    "platt": nll(platt_test, y_test), "softmax_temperature": nll(soft_test, y_test)},
            "brier": {"raw": brier(raw_test, y_test), "temperature": brier(cal_test, y_test),
                      "platt": brier(platt_test, y_test), "softmax_temperature": brier(soft_test, y_test)},
            "ece_note": "10 equal-width bins (same as the bake-off); 147 pairs, so ECE is noisy: see the "
                        "bootstrap interval",
        },
        "validation": {"ece_raw": ece(raw_val, y_val), "ece_temperature": ece(cal_val, y_val),
                       "nll_raw": nll(raw_val, y_val), "nll_temperature": nll(cal_val, y_val)},
        "threshold": {
            "shipped": runtime["threshold"],
            "rule": chosen_rule,
            "at_shipped_threshold_test": shipped,
            "rules": operating,
            "note": "Thresholds are on the temperature-calibrated probability and chosen on validation "
                    "only. val_fpr_le_0.05 is the bake-off's rule, shown for comparison.",
        },
        "accuracy_by_kind_test": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())},
        "auroc_test_by_source": {s: auroc([p for p, _ in v], [y for _, y in v]) for s, v in sorted(by_source.items())},
        "python_latency": {"load_ms": round(load_ms, 1), "p50_ms": round(statistics.median(lat), 2),
                           "p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2),
                           "split": "all",
                           "note": "tokenize + run, batch 1, one run on this machine"},
        "onnxruntime_drift": drift,
        "session_settings_check": session_check,
        "batch_check": batch_check,
        "notes": [
            "Test ECE after temperature scaling misses the 0.05 bar of the bake-off's go rules. It is not "
            "applied to this model: it only proposes a 'possible contradiction' review edge, never retires a "
            "record (Axil decision 01M42C1K7XJK3NNG0JM9CYGDW0).",
            "Latency is one run on a machine shared with other jobs; compare it only with numbers from the "
            "same day.",
        ],
        "runtime_file": "crates/extensions/axil-decide/calibration/u1-nli-deberta-v3-xsmall-qint8.json",
    }
    dest = ROOT / "benchmarks" / "results" / f"typed-decisions-u1-calibration-{date}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")

    cal_dir = CRATE / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / f"u1-{MODEL['name']}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"wrote {(cal_dir / ('u1-' + MODEL['name'] + '.json')).relative_to(ROOT)}")

    fixtures = CRATE / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    fixture = {
        "note": f"Python onnxruntime {ort.__version__} logits for the U1 test pairs (texts: "
                f"benchmarks/typed-decisions/sets). Written by benchmarks/typed-decisions/calibrate_u1.py.",
        "onnxruntime": ort.__version__,
        "model": MODEL["name"],
        "revision": MODEL["revision"],
        "temperature": t,
        "threshold": threshold,
        "pairs": [{"id": r["id"], "tokens": by_id[r["id"]]["tokens"], "logits": by_id[r["id"]]["logits"],
                   "p": prob(by_id[r["id"]]["logits"], t),
                   "flag": prob(by_id[r["id"]]["logits"], t) >= threshold} for r in test],
    }
    (fixtures / "u1_parity.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {(fixtures / 'u1_parity.json').relative_to(ROOT)}")

    print(f"\nT = {t:.4f}   Platt a = {pa:.3f} b = {pb:.3f}   softmax T = {t_soft:.4f}")
    tt = report["test"]
    print(f"test ECE raw {tt['ece_raw']:.3f}  T {tt['ece_temperature']:.3f}  Platt {tt['ece_platt']:.3f}"
          f"  softmax-T {tt['ece_softmax_temperature']:.3f}"
          f"   AUROC raw {tt['auroc_raw']:.4f} cal {tt['auroc_calibrated']:.4f}")
    for name, op in operating.items():
        v, te = op["validation"], op["test"]
        print(f"{name:<30} thr {op['threshold']:.4f}  val FPR {v['fpr']:.3f} TPR {v['tpr']:.3f}"
              f"  | test FPR {te['fpr']:.3f} ({te['fp']}/{te['negatives']}) TPR {te['tpr']:.3f}"
              f" ({te['tp']}/{te['positives']})")
    print(f"batched vs single: max |logit diff| {batch_check['max_abs_logit_diff']:.3g}, "
          f"max |p diff| {batch_check['max_abs_calibrated_p_diff']:.3g}, flips {batch_check['threshold_flips']}")
    for name, v in session_check.items():
        if isinstance(v, dict):
            print(f"{name:<36} max |logit diff| {v['max_abs_logit_diff']:.3g}  flips {v['threshold_flips']}")
    if drift:
        print(f"drift vs {drift['against']}: logits {drift['max_abs_logit_diff']:.2e}, "
              f"p {drift['max_abs_calibrated_p_diff']:.2e}, flips {drift['threshold_flips']}")


if __name__ == "__main__":
    main()
