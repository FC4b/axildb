#!/usr/bin/env python3
"""U3 bake-off: capture classification on sets/u3_repo.jsonl.

Each text gets one class (decision / error / context / rule / none) and, when
it is not none, an importance bucket (low / medium / high); see
validation/u3-guidelines.md. Every metric is on the `test` split.

Arms:
  production       auto_capture::analyze through the Rust crate in heuristic/
                   (`--capture`): a text is stored as its first capture whose
                   confidence passes the caller's --min-confidence. The
                   control is the hook's call, 0.8 (`production`); the CLI
                   default, 0.7, is reported as `production@0.7`. Importance
                   is what `Axil::insert` computes for the stored record
                   (`compute_importance`), or for a plain {summary: text}
                   record when nothing is captured, bucketed at cut points
                   chosen on train.
  lr-weak          a multinomial logistic-regression head on the bge-small
                   vector Axil stores for a record (heuristic/ `--embed`),
                   trained on the live DB's own records with their table as a
                   weak label (decisions / errors / context / rules); none is
                   predicted when the top probability is below a threshold.
                   C, class weighting and the threshold are chosen on train.
  lr-weak+train    the same head trained on the weak records plus the u3 train
                   split (which has gold none items, so five classes);
                   hyperparameters by 5-fold cross-validation inside train.
  gliclass-edge    knowledgator/gliclass-edge-v3.0, zero-shot (gliclass)
  gliclass-base    knowledgator/gliclass-base-v3.0
  gliner2.5-small  fastino/gliner2.5-small-v1, classify_text (gliner2)
  gliner2.5-base   fastino/gliner2.5-base-v1
Both LR arms share one importance head: logistic regression on the same
vector, trained on the gold importance of the train split's non-none items.
The zero-shot arms score three importance labels.

Selection, fixed before looking at test: for each model arm, the setting
(phrasing and none threshold for the zero-shot arms; C, class weighting and
threshold for the heads) with the best train macro-F1 among those whose train
false-capture rate is at most the control's; if none qualifies, the best
train macro-F1, flagged. For zero-shot arms the scores of every train and
test text are computed once per phrasing and thresholds are swept offline;
the chosen setting then runs for real on test (batch 1, model loaded), which
gives the latency.

Verdict (the go/no-go rule the earlier bake-offs use): a model wins only if it beats the control's test
macro-F1 at an equal or lower test false-capture rate, fits a CPU call site
(FIT_BARS; the hook stays model-free, so a model runs in the queued
`axil auto-capture` job, the worker or the daemon), and is the smallest model
within 1 macro-F1 point of the best model. Otherwise NO-GO.

Usage:
  python run_u3_bakeoff.py --prepare                # weak records + vectors (needs --db, heuristic/)
  python run_u3_bakeoff.py --arm all                # every arm, one process each, then summarize
  python run_u3_bakeoff.py --arm gliclass-edge      # one arm (raw output to out/u3/)
  python run_u3_bakeoff.py --summarize              # rebuild the results JSON from out/u3/

Raw outputs go to out/u3/ (gitignored); the summary goes to
benchmarks/results/typed-decisions-u3-bakeoff-<date>.json and the two LR
heads (weights, bias, decision rule, parity fixtures) to
benchmarks/results/typed-decisions-u3-heads-<date>.json.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from repo_sets_u3 import CUTOFF

HERE = Path(__file__).resolve().parent
SET = HERE / "sets" / "u3_repo.jsonl"
OUT = HERE / "out" / "u3"
RESULTS = HERE.parent / "results"
HEURISTIC = HERE / "heuristic"
HEURISTIC_BIN = HEURISTIC / "target" / "release" / "typed-decisions-heuristic"

CLASSES = ["decision", "error", "context", "rule", "none"]
STORED = CLASSES[:4]
BUCKETS = ["low", "medium", "high"]
SOURCES = ["bash_failure", "bash_output", "agent_text", "agent_brief", "commit_subject"]
HOOK_MIN_CONFIDENCE = 0.8
CLI_MIN_CONFIDENCE = 0.7
CAPTURE_CLASS = {"error": "error", "decision": "decision", "context": "context"}

WEAK_TABLES = {"decisions": "decision", "errors": "error", "context": "context", "rules": "rule"}
BGE_SMALL = {"repo": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
             "files": {"onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
                       "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"},
             "params": 33_360_000, "size_on_disk_bytes": 133_093_490 + 711_396,
             "note": "the embedder Axil already loads to store a record (axil-vector download.rs)"}

GLINER25_FILES = ["config.json", "encoder_config/config.json", "model.safetensors",
                  "tokenizer.json", "tokenizer_config.json"]
MODELS = {
    "gliclass-edge": dict(lib="gliclass", repo="knowledgator/gliclass-edge-v3.0",
                          revision="df03993a2ed98e5e4a0d2dd7efbbd105abe874cf",
                          files=["config.json", "model.safetensors", "special_tokens_map.json",
                                 "tokenizer.json", "tokenizer_config.json"]),
    "gliclass-base": dict(lib="gliclass", repo="knowledgator/gliclass-base-v3.0",
                          revision="77a70e6cd52e602ed18184ef37d18bdd3741e3d5",
                          files=["added_tokens.json", "config.json", "model.safetensors",
                                 "special_tokens_map.json", "spm.model", "tokenizer.json",
                                 "tokenizer_config.json"]),
    "gliner2.5-small": dict(lib="gliner2", repo="fastino/gliner2.5-small-v1",
                            revision="df5910e44bc4ffdb0d95399a83b0ca4516349aa5", files=GLINER25_FILES),
    "gliner2.5-base": dict(lib="gliner2", repo="fastino/gliner2.5-base-v1",
                           revision="54785d51df8d86a0d8f2eb212fec15871aad2c4d", files=GLINER25_FILES),
}
HEADS = ["lr-weak", "lr-weak+train"]
ARMS = ["production", *HEADS, *MODELS]

# Two label phrasings per zero-shot model, picked on train.
PHRASINGS = {
    "short": {
        "decision": "decision",
        "error": "error or failure",
        "context": "technical knowledge",
        "rule": "rule or instruction",
        "none": "chatter or routine output",
    },
    "descriptive": {
        "decision": "a decision choosing one option over another",
        "error": "a report of an error, bug or failure",
        "context": "an explanation of how something works",
        "rule": "a rule that must always be followed",
        "none": "routine tool output, progress update or small talk",
    },
}
IMPORTANCE_LABELS = {"low": "minor detail", "medium": "useful project knowledge",
                     "high": "critical project knowledge"}
# 0.00 .. 0.95, then finer near 1: the zero-shot scores crowd near the top, so
# a threshold that keeps junk out has to sit above 0.95 (the first run stopped
# at 0.95 and no zero-shot setting met the train false-capture bar there).
TAU_GRID = [round(0.05 * k, 2) for k in range(20)] + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
FIRST_RUN_TAU_MAX = 0.95
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
WEIGHTINGS = [None, "balanced"]
CV_FOLDS = 5
SEED = 20261014

# Where a model would run: never the hook (it must stay model-free and has a
# kill timeout). The hook queues `axil auto-capture` as a detached job; a
# model there loads per job, so load time is reported next to these bars, and
# the worker or the daemon loads it once. Same bars as U5.
FIT_BARS = {"peak_rss_mb_max": 1500.0, "p95_ms_max": 1000.0}


# ── Data and scoring ────────────────────────────────────────────────


def load_rows():
    return [json.loads(line) for line in SET.read_text().splitlines() if line.strip()]


def split_ids(rows, split):
    return [r["id"] for r in rows if r["split"] == split]


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def class_metrics(pred, gold, ids):
    """Accuracy, macro-F1, per-class P/R/F1 and the capture error rates over `ids`."""
    per = {}
    for c in CLASSES:
        tp = sum(1 for i in ids if pred[i] == c and gold[i] == c)
        n_pred = sum(1 for i in ids if pred[i] == c)
        n_gold = sum(1 for i in ids if gold[i] == c)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"precision": rnd(p), "recall": rnd(r), "f1": rnd(f), "tp": tp, "pred": n_pred, "gold": n_gold}
    present = [c for c in CLASSES if per[c]["gold"] or per[c]["pred"]]
    gold_none = [i for i in ids if gold[i] == "none"]
    gold_store = [i for i in ids if gold[i] != "none"]
    stored = [i for i in ids if pred[i] != "none"]
    both = [i for i in gold_store if pred[i] != "none"]
    false_capture = [i for i in gold_none if pred[i] != "none"]
    wrong_table = [i for i in both if pred[i] != gold[i]]
    missed = [i for i in gold_store if pred[i] == "none"]
    return {
        "items": len(ids),
        "accuracy": rnd(sum(pred[i] == gold[i] for i in ids) / len(ids)) if ids else None,
        "macro_f1": rnd(statistics.mean(per[c]["f1"] for c in present)) if present else None,
        "macro_f1_classes": present,
        "per_class": per,
        "false_capture_rate": rnd(len(false_capture) / len(gold_none)) if gold_none else None,
        "false_captures": len(false_capture),
        "gold_none": len(gold_none),
        "wrong_table_rate": rnd(len(wrong_table) / len(both)) if both else None,
        "wrong_table": len(wrong_table),
        "stored_and_should_be": len(both),
        "miss_rate": rnd(len(missed) / len(gold_store)) if gold_store else None,
        "missed": len(missed),
        "junk_share_of_stored": rnd(len(false_capture) / len(stored)) if stored else None,
        "stored": len(stored),
        "confusion": {g: dict(Counter(pred[i] for i in ids if gold[i] == g)) for g in CLASSES},
    }


def importance_metrics(pred_bucket, gold_rows, ids):
    ids = [i for i in ids if gold_rows[i]["class"] != "none"]
    if not ids:
        return None
    hits = sum(pred_bucket[i] == gold_rows[i]["importance"] for i in ids)
    off = [abs(BUCKETS.index(pred_bucket[i]) - BUCKETS.index(gold_rows[i]["importance"])) for i in ids]
    return {"items": len(ids), "accuracy": rnd(hits / len(ids)), "within_one": rnd(sum(o <= 1 for o in off) / len(ids)),
            "mean_abs_bucket_error": rnd(statistics.mean(off)),
            "confusion": {g: dict(Counter(pred_bucket[i] for i in ids if gold_rows[i]["importance"] == g))
                          for g in BUCKETS}}


def macro_f1(pred, gold, ids):
    return class_metrics(pred, gold, ids)["macro_f1"]


def false_capture(pred, gold, ids):
    none = [i for i in ids if gold[i] == "none"]
    return sum(pred[i] != "none" for i in none) / len(none) if none else 0.0


def select(candidates, ctrl_fc):
    """Best train macro-F1 among candidates with train false capture <= the control's."""
    feasible = [c for c in candidates if c["train_false_capture"] <= ctrl_fc + 1e-12]
    pool, ok = (feasible, True) if feasible else (candidates, False)
    best = max(pool, key=lambda c: (c["train_macro_f1"], -c["train_false_capture"]))
    free = max(candidates, key=lambda c: (c["train_macro_f1"], -c["train_false_capture"]))
    return {**best, "false_capture_constraint_met_on_train": ok}, free


def bucket_of(score, cuts):
    lo, hi = cuts
    return "low" if score < lo else "medium" if score < hi else "high"


def fit_cuts(scores, gold_rows, ids):
    """Two cut points on a continuous importance score with the best train accuracy."""
    ids = [i for i in ids if gold_rows[i]["class"] != "none"]
    grid = [round(0.05 * k, 2) for k in range(1, 21)]
    best = None
    for lo in grid:
        for hi in grid:
            if hi <= lo:
                continue
            acc = sum(bucket_of(scores[i], (lo, hi)) == gold_rows[i]["importance"] for i in ids) / len(ids)
            if best is None or acc > best[0]:
                best = (acc, (lo, hi))
    return {"cuts": list(best[1]), "train_accuracy": rnd(best[0]), "train_items": len(ids)}


# ── Environment ─────────────────────────────────────────────────────


def loadavg():
    return round(os.getloadavg()[0], 2)


def pct(values, q):
    s = sorted(values)
    return round(s[int(q * (len(s) - 1))], 3)


def machine():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for key, name in [("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "memory_bytes")]:
            try:
                info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
            except OSError:
                pass
    return info


def timed(cmd, stdout_path=None):
    """Run `cmd` under /usr/bin/time -l; returns (stdout, stderr, footprint)."""
    full = (["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]) + cmd
    lb = loadavg()
    t0 = time.perf_counter()
    proc = subprocess.run(full, capture_output=True, text=True)
    wall = (time.perf_counter() - t0) * 1000
    if proc.returncode:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    if stdout_path:
        Path(stdout_path).write_text(proc.stdout)
    return proc.stdout, proc.stderr, {**time_stats(proc.stderr), "wall_ms": round(wall, 1),
                                      "loadavg_1m_before": lb, "loadavg_1m_after": loadavg()}


def time_stats(stderr):
    """Peak RSS and peak memory footprint from /usr/bin/time -l (macOS) or -v (Linux)."""
    out = {}
    for line in stderr.splitlines():
        low = line.lower().strip()
        m = re.match(r"(\d+)\s+(.+)$", low)
        if m and m.group(2) == "maximum resident set size":
            out["peak_rss_mb"] = round(int(m.group(1)) / (1024 * 1024), 1)
        elif m and m.group(2) == "peak memory footprint":
            out["peak_footprint_mb"] = round(int(m.group(1)) / (1024 * 1024), 1)
        elif low.startswith("maximum resident set size (kbytes):"):
            out["peak_rss_mb"] = round(int(low.split(":")[1]) / 1024, 1)
    return out


def write_arm(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{out['arm']}.json").write_text(json.dumps(out, indent=1))
    lat = out.get("latency_ms", {})
    print(f"{out['arm']:<16} p50 {lat.get('p50')} ms p95 {lat.get('p95')} ms (load avg "
          f"{lat.get('loadavg_1m_before')}->{lat.get('loadavg_1m_after')}), load {out.get('load_ms')} ms",
          flush=True)


# ── Prepare: weak records and vectors ───────────────────────────────


def prepare(args):
    if not HEURISTIC_BIN.exists():
        sys.exit(f"build it first: cargo build --release --manifest-path {HEURISTIC}/Cargo.toml")
    db = args.db or os.environ.get("AXIL_DB")
    if not db:
        sys.exit("pass --db <memory.axil> (read-only: only `axil export` runs against it)")
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [args.axil, "export", "--db", db, "--tables", ",".join(WEAK_TABLES), "--quiet"]
    lines = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    kept, skipped = [], Counter()
    for line in lines:
        r = json.loads(line) if line.strip() else {}
        if r.get("kind") != "record" or r.get("table") not in WEAK_TABLES:
            continue
        d = r["data"]
        if (r.get("created_at") or "").replace("+00:00", "Z") >= CUTOFF:
            skipped["after_cutoff"] += 1
        elif d.get("_superseded"):
            skipped["superseded"] += 1
        elif d.get("_origin") == "fallback_capture" or d.get("type") == "fallback_capture":
            skipped["fallback_capture"] += 1
        else:
            kept.append({"id": r["id"], "table": r["table"], "label": WEAK_TABLES[r["table"]], "data": d})
    with (OUT / "weak_records.jsonl").open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    stats = {"weak_records": dict(Counter(r["label"] for r in kept)), "skipped": dict(skipped),
             "cutoff": CUTOFF}
    for name, src in [("weak", OUT / "weak_records.jsonl"), ("u3", SET)]:
        _, err, fp = timed([str(HEURISTIC_BIN), "--embed", str(src)], OUT / f"emb_{name}.jsonl")
        meta = next(json.loads(l) for l in err.splitlines() if l.startswith("{"))
        stats[f"embed_{name}"] = {**fp, **meta}
    (OUT / "prepare.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def load_vectors(name):
    out = {}
    for line in (OUT / f"emb_{name}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = (np.asarray(r["vector"], dtype=np.float32), r["embed_us"])
    return out


# ── Production arm ──────────────────────────────────────────────────


def run_production(args):
    if not HEURISTIC_BIN.exists():
        sys.exit(f"build it first: cargo build --release --manifest-path {HEURISTIC}/Cargo.toml")
    stdout, err, fp = timed([str(HEURISTIC_BIN), "--capture", str(SET)])
    meta = next(json.loads(l) for l in err.splitlines() if l.startswith("{"))
    raw = {r["id"]: r for r in (json.loads(l) for l in stdout.splitlines() if l.strip())}
    rows = load_rows()
    test = split_ids(rows, "test")
    lat = [raw[i]["analyze_us"] / 1000 for i in test]
    out = {
        "arm": "production",
        "what": "axil_core::auto_capture::analyze(text, \"bash\"), then capture_to_record and "
                "compute_importance on each capture, as `axil auto-capture` stores them",
        "binary": str(HEURISTIC_BIN.relative_to(HERE)),
        "raw": raw,
        "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
                       "loadavg_1m_before": fp["loadavg_1m_before"], "loadavg_1m_after": fp["loadavg_1m_after"],
                       "measured_in": "Rust, Instant around analyze() per test text"},
        "load_ms": round(meta["first_call_us"] / 1000, 3),
        "load_note": "first analyze call (the patterns are plain substring lists; nothing to load)",
        **{k: fp[k] for k in ("peak_rss_mb", "peak_footprint_mb") if k in fp},
        "rss_note": "whole heuristic process under /usr/bin/time -l (it links axil-core and "
                    "axil-vector; no model is loaded in --capture mode)",
        "params": 0,
        "size_on_disk_bytes": 0,
        "size_note": "no model: the patterns are compiled into axil-core",
    }
    write_arm(out)
    return out


def production_view(raw, threshold):
    """Class and importance score per id when captures >= threshold are stored."""
    pred, score, multi = {}, {}, 0
    for i, r in raw.items():
        stored = [c for c in r["captures"] if c["confidence"] >= threshold - 1e-6]
        multi += len(stored) > 1
        if stored:
            pred[i] = CAPTURE_CLASS[stored[0]["type"]]
            score[i] = stored[0]["importance"]
        else:
            pred[i] = "none"
            score[i] = r["plain_importance"]
    return pred, score, multi


# ── LR heads ────────────────────────────────────────────────────────


def fit_lr(X, y, C, weighting, labels):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, class_weight=weighting, max_iter=5000, random_state=SEED)
    clf.fit(X, y)
    order = [list(clf.classes_).index(c) for c in labels if c in clf.classes_]
    return clf, [clf.classes_[k] for k in order], order


def head_predict(probs, classes, tau):
    """Top class, or none when the top stored class's probability is below tau."""
    out = []
    for p in probs:
        k = int(np.argmax(p))
        c = classes[k]
        out.append("none" if c == "none" or p[k] < tau else c)
    return out


def ctrl_train_fc(rows, gold):
    raw = json.loads((OUT / "production.json").read_text())["raw"]
    pred, _, _ = production_view(raw, HOOK_MIN_CONFIDENCE)
    return false_capture(pred, gold, split_ids(rows, "train"))


def importance_head(vec, rows, gold_rows):
    """LR on the vector for the gold importance of train's non-none items; C by leave-one-out."""
    from sklearn.linear_model import LogisticRegression

    train = [i for i in split_ids(rows, "train") if gold_rows[i]["class"] != "none"]
    X = np.stack([vec[i][0] for i in train])
    y = np.array([gold_rows[i]["importance"] for i in train])
    curve = []
    for C in C_GRID:
        hits = 0
        for k in range(len(train)):
            mask = np.arange(len(train)) != k
            if len(set(y[mask])) < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED).fit(X[mask], y[mask])
            hits += clf.predict(X[k:k + 1])[0] == y[k]
        curve.append({"C": C, "loo_accuracy": rnd(hits / len(train))})
    best = max(curve, key=lambda c: (c["loo_accuracy"], -c["C"]))
    clf = LogisticRegression(C=best["C"], max_iter=5000, random_state=SEED).fit(X, y)
    pred = {r["id"]: clf.predict(vec[r["id"]][0][None, :])[0] for r in rows}
    return pred, {"train_items": len(train), "train_buckets": dict(Counter(y.tolist())), "curve": curve,
                  "C": best["C"], "classes": list(clf.classes_), "weights": clf.coef_.round(6).tolist(),
                  "bias": clf.intercept_.round(6).tolist()}


def run_head(name, args):
    rows = load_rows()
    gold = {r["id"]: r["class"] for r in rows}
    gold_rows = {r["id"]: r for r in rows}
    train, test = split_ids(rows, "train"), split_ids(rows, "test")
    vec = load_vectors("u3")
    weak_vec = load_vectors("weak")
    weak = [json.loads(l) for l in (OUT / "weak_records.jsonl").read_text().splitlines() if l.strip()]
    Xw = np.stack([weak_vec[r["id"]][0] for r in weak])
    yw = np.array([r["label"] for r in weak])
    ctrl_fc = ctrl_train_fc(rows, gold)
    labels = STORED if name == "lr-weak" else CLASSES

    candidates = []
    if name == "lr-weak":
        Xt = np.stack([vec[i][0] for i in train])
        for C in C_GRID:
            for w in WEIGHTINGS:
                clf, classes, order = fit_lr(Xw, yw, C, w, labels)
                probs = clf.predict_proba(Xt)[:, order]
                for tau in TAU_GRID:
                    pred = dict(zip(train, head_predict(probs, classes, tau)))
                    candidates.append({"C": C, "class_weight": w, "tau": tau,
                                       "train_macro_f1": macro_f1(pred, gold, train),
                                       "train_false_capture": rnd(false_capture(pred, gold, train))})
    else:
        rng = np.random.default_rng(SEED)
        folds = {}
        for c in CLASSES:  # stratified by gold class
            ids = [i for i in train if gold[i] == c]
            rng.shuffle(ids)
            for k, i in enumerate(ids):
                folds[i] = k % CV_FOLDS
        for C in C_GRID:
            for w in WEIGHTINGS:
                oof = {}
                for f in range(CV_FOLDS):
                    fit_ids = [i for i in train if folds[i] != f]
                    held = [i for i in train if folds[i] == f]
                    X = np.concatenate([Xw, np.stack([vec[i][0] for i in fit_ids])])
                    y = np.concatenate([yw, np.array([gold[i] for i in fit_ids])])
                    clf, classes, order = fit_lr(X, y, C, w, labels)
                    oof.update({i: p for i, p in zip(held, clf.predict_proba(np.stack([vec[i][0] for i in held]))[:, order])})
                probs = np.stack([oof[i] for i in train])
                for tau in TAU_GRID:
                    pred = dict(zip(train, head_predict(probs, classes, tau)))
                    candidates.append({"C": C, "class_weight": w, "tau": tau,
                                       "train_macro_f1": macro_f1(pred, gold, train),
                                       "train_false_capture": rnd(false_capture(pred, gold, train))})
    chosen, free = select(candidates, ctrl_fc)

    # Final fit and test predictions; head latency measured per test text.
    if name == "lr-weak":
        X, y = Xw, yw
    else:
        X = np.concatenate([Xw, np.stack([vec[i][0] for i in train])])
        y = np.concatenate([yw, np.array([gold[i] for i in train])])
    clf, classes, order = fit_lr(X, y, chosen["C"], chosen["class_weight"], labels)
    W = clf.coef_[order].astype(np.float32)
    b = clf.intercept_[order].astype(np.float32)
    all_ids = [r["id"] for r in rows]
    probs = {i: clf.predict_proba(vec[i][0][None, :])[0, order] for i in all_ids}
    pred = {i: head_predict([probs[i]], classes, chosen["tau"])[0] for i in all_ids}
    head_lat, total_lat = [], []
    lb = loadavg()
    for i in test:
        t = time.perf_counter()
        z = W @ vec[i][0] + b
        p = np.exp(z - z.max())
        p /= p.sum()
        _ = head_predict([p], classes, chosen["tau"])
        dt = (time.perf_counter() - t) * 1000
        head_lat.append(dt)
        total_lat.append(dt + vec[i][1] / 1000)
    la = loadavg()
    free_pred = None
    if (free["C"], free["class_weight"], free["tau"]) != (chosen["C"], chosen["class_weight"], chosen["tau"]):
        fclf, fclasses, forder = fit_lr(X, y, free["C"], free["class_weight"], labels)
        free_pred = {i: head_predict(fclf.predict_proba(vec[i][0][None, :])[:, forder], fclasses, free["tau"])[0]
                     for i in all_ids}

    imp_pred, imp_head = importance_head(vec, rows, gold_rows)
    prep = json.loads((OUT / "prepare.json").read_text())
    fixtures = [{"id": i, "probabilities": [rnd(x, 6) for x in probs[i]], "prediction": pred[i]} for i in test[:8]]
    out = {
        "arm": name,
        "what": "multinomial logistic regression on the bge-small passage vector "
                "(heuristic/ --embed: Embedder::embed of the text)",
        "training": {"weak_records": dict(Counter(yw.tolist())),
                     "u3_train_items": 0 if name == "lr-weak" else len(train),
                     "weak_label_source": "live DB tables decisions/errors/context/rules, records created "
                                          f"before {CUTOFF}, superseded and fallback-capture rows left out"},
        "control_train_false_capture": rnd(ctrl_fc),
        "selection": {"chosen": chosen, "max_f1_train": free,
                      "method": "train macro-F1 on the u3 train split" if name == "lr-weak"
                                else f"{CV_FOLDS}-fold stratified cross-validation inside the u3 train split "
                                     "(weak records in every fold's training data)"},
        "selection_curve": candidates,
        "head": {"classes": classes, "weights": W.round(6).tolist(), "bias": b.round(6).tolist(),
                 "tau": chosen["tau"], "rule": "softmax(W v + b); top class, or none when the top class "
                                               "is not none and its probability < tau",
                 "parity_fixtures": fixtures},
        "predictions": pred,
        "probabilities": {i: [rnd(x, 6) for x in probs[i]] for i in all_ids},
        "free_predictions": free_pred,
        "importance": imp_pred,
        "importance_head": imp_head,
        "latency_ms": {"p50": pct(total_lat, 0.5), "p95": pct(total_lat, 0.95),
                       "head_only_p50": pct(head_lat, 0.5), "head_only_p95": pct(head_lat, 0.95),
                       "embed_loadavg_1m": [prep["embed_u3"]["loadavg_1m_before"], prep["embed_u3"]["loadavg_1m_after"]],
                       "loadavg_1m_before": lb, "loadavg_1m_after": la,
                       "measured_in": "embed: Rust Instant around Embedder::embed per test text (heuristic "
                                      "--embed); head: numpy W v + b and softmax per test text"},
        "load_ms": round(prep["embed_u3"]["load_ms"], 1),
        "load_note": "bge-small load in the --embed process; Axil loads it anyway to store a record",
        "peak_rss_mb": prep["embed_u3"].get("peak_rss_mb"),
        "peak_footprint_mb": prep["embed_u3"].get("peak_footprint_mb"),
        "rss_note": "the Rust --embed process (bge-small loaded) under /usr/bin/time -l; the head adds "
                    "its weights only",
        "params": int(W.size + b.size),
        "head_bytes_f32": int((W.size + b.size) * 4),
        "embedder": BGE_SMALL,
        "size_on_disk_bytes": int((W.size + b.size) * 4),
        "size_note": "head only (f32); the bge-small files are already on disk for every Axil install "
                     "that embeds",
    }
    write_arm(out)
    return out


# ── Zero-shot arms ──────────────────────────────────────────────────


def snapshot(repo, revision, files):
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, revision=revision, allow_patterns=files)
    return path, sum(os.path.getsize(os.path.realpath(os.path.join(path, f))) for f in files)


class GliclassArm:
    def __init__(self, path):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        self.model = GLiClassModel.from_pretrained(path)
        self.model.eval()
        tok = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
        self.pipe = ZeroShotClassificationPipeline(self.model, tok, classification_type="multi-label",
                                                   device="cpu", progress_bar=False)

    def scores(self, text, labels):
        """{key: score} for every label (multi-label sigmoid scores)."""
        names = list(labels.values())
        back = {v: k for k, v in labels.items()}
        res = self.pipe(text, names, threshold=0.0)[0]
        return {back[r["label"]]: float(r["score"]) for r in res}


class Gliner2Arm:
    def __init__(self, path):
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(path, map_location="cpu")
        self.model.eval()

    def scores(self, text, labels):
        """{key: confidence} for the chosen label only (single-label classification)."""
        names = list(labels.values())
        back = {v: k for k, v in labels.items()}
        res = self.model.classify_text(text, {"label": names}, include_confidence=True)["label"]
        return {back[res["label"]]: float(res["confidence"])}


def param_count(arm):
    return sum(p.numel() for p in arm.model.parameters())


def zs_predict(scores, tau):
    """Top label; a stored class whose score is below tau becomes none."""
    top = max(scores, key=scores.get)
    return "none" if top == "none" or scores[top] < tau else top


def run_model(name, args):
    import torch

    torch.set_num_threads(args.threads)
    spec = MODELS[name]
    rows = load_rows()
    gold = {r["id"]: r["class"] for r in rows}
    train, test = split_ids(rows, "train"), split_ids(rows, "test")
    path, size = snapshot(spec["repo"], spec["revision"], spec["files"])
    lb = loadavg()
    t0 = time.perf_counter()
    arm = (GliclassArm if spec["lib"] == "gliclass" else Gliner2Arm)(path)
    load_ms = (time.perf_counter() - t0) * 1000
    load_la = [lb, loadavg()]
    for _ in range(3):
        arm.scores("Warm up: error[E0308] in src/db.rs", PHRASINGS["short"])
    ctrl_fc = ctrl_train_fc(rows, gold)

    sweeps, candidates = {}, []
    for pname, labels in PHRASINGS.items():
        with torch.inference_mode():
            sc = {r["id"]: arm.scores(r["text"], labels) for r in rows}
        sweeps[pname] = sc
        for tau in TAU_GRID:
            pred = {i: zs_predict(sc[i], tau) for i in train}
            candidates.append({"phrasing": pname, "tau": tau, "train_macro_f1": macro_f1(pred, gold, train),
                               "train_false_capture": rnd(false_capture(pred, gold, train))})
    chosen, free = select(candidates, ctrl_fc)
    labels = PHRASINGS[chosen["phrasing"]]

    # Real run on test at the chosen setting: predictions and latency.
    preds, lat, mismatch = {}, [], 0
    lb = loadavg()
    for i in test:
        text = next(r["text"] for r in rows if r["id"] == i)
        t = time.perf_counter()
        with torch.inference_mode():
            s = arm.scores(text, labels)
        lat.append((time.perf_counter() - t) * 1000)
        preds[i] = zs_predict(s, chosen["tau"])
        if preds[i] != zs_predict(sweeps[chosen["phrasing"]][i], chosen["tau"]):
            mismatch += 1
    la = loadavg()
    for i in train:
        preds[i] = zs_predict(sweeps[chosen["phrasing"]][i], chosen["tau"])
    free_pred = {i: zs_predict(sweeps[free["phrasing"]][i], free["tau"]) for i in train + test}

    # Importance: one more zero-shot pass with three bucket labels.
    imp, imp_lat = {}, []
    for r in rows:
        t = time.perf_counter()
        with torch.inference_mode():
            s = arm.scores(r["text"], IMPORTANCE_LABELS)
        if r["split"] == "test":
            imp_lat.append((time.perf_counter() - t) * 1000)
        imp[r["id"]] = max(s, key=s.get)

    out = {
        "arm": name, "repo": spec["repo"], "revision": spec["revision"], "library": spec["lib"],
        "files": spec["files"], "params": param_count(arm), "size_on_disk_bytes": size,
        "load_ms": round(load_ms, 1), "load_loadavg_1m": load_la, "threads": args.threads,
        "control_train_false_capture": rnd(ctrl_fc),
        "selection": {"chosen": chosen, "max_f1_train": free}, "selection_curve": candidates,
        "labels": labels, "importance_labels": IMPORTANCE_LABELS,
        "predictions": preds, "free_predictions": free_pred, "importance": imp,
        "scores": {p: s for p, s in sweeps.items()},
        "sweep_mismatch_texts": mismatch,
        "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "mean": round(statistics.mean(lat), 3),
                       "with_importance_pass_p50": pct([a + b for a, b in zip(lat, imp_lat)], 0.5),
                       "with_importance_pass_p95": pct([a + b for a, b in zip(lat, imp_lat)], 0.95),
                       "loadavg_1m_before": lb, "loadavg_1m_after": la,
                       "measured_in": "Python, perf_counter around one classification call per test text, "
                                      "batch 1, model loaded and warmed"},
    }
    write_arm(out)
    return out


def reselect(name):
    """Redo a zero-shot arm's train-only selection from its saved scores.

    Scores do not depend on the threshold, so a new threshold grid needs no
    new inference. Latency stays as measured (batch 1, at the phrasing the
    first selection chose); the first selection is kept for the record.
    """
    path = OUT / f"{name}.json"
    out = json.loads(path.read_text())
    rows = load_rows()
    gold = {r["id"]: r["class"] for r in rows}
    train, test = split_ids(rows, "train"), split_ids(rows, "test")
    ctrl_fc = ctrl_train_fc(rows, gold)
    candidates = []
    for pname, sc in out["scores"].items():
        for tau in TAU_GRID:
            pred = {i: zs_predict(sc[i], tau) for i in train}
            candidates.append({"phrasing": pname, "tau": tau, "train_macro_f1": macro_f1(pred, gold, train),
                               "train_false_capture": rnd(false_capture(pred, gold, train))})
    chosen, free = select(candidates, ctrl_fc)
    out.setdefault("first_selection", {"tau_max": FIRST_RUN_TAU_MAX, **out["selection"]})
    out["selection"] = {"chosen": chosen, "max_f1_train": free}
    out["selection_curve"] = candidates
    out["labels"] = PHRASINGS[chosen["phrasing"]]
    sc = out["scores"][chosen["phrasing"]]
    out["predictions"] = {i: zs_predict(sc[i], chosen["tau"]) for i in train + test}
    fsc = out["scores"][free["phrasing"]]
    out["free_predictions"] = {i: zs_predict(fsc[i], free["tau"]) for i in train + test}
    out["latency_ms"]["phrasing_timed"] = out["first_selection"]["chosen"]["phrasing"]
    path.write_text(json.dumps(out, indent=1))
    print(f"{name}: {out['first_selection']['chosen']} -> {chosen}")


# ── Summary and verdict ─────────────────────────────────────────────


def versions():
    out = {}
    from importlib.metadata import version

    for mod in ("torch", "transformers", "tokenizers", "gliclass", "gliner2", "scikit-learn", "numpy",
                "huggingface_hub"):
        try:
            out[mod] = version(mod)
        except Exception:  # noqa: BLE001 - a missing package just isn't listed
            pass
    return out


def per_source(pred, gold, rows, ids):
    out = {}
    for s in SOURCES:
        sids = [i for i in ids if next(r for r in rows if r["id"] == i)["source"] == s]
        if not sids:
            continue
        m = class_metrics(pred, gold, sids)
        out[s] = {k: m[k] for k in ("items", "accuracy", "false_capture_rate", "false_captures", "gold_none",
                                    "miss_rate", "missed", "wrong_table")}
    return out


def evaluate(pred, imp_bucket, rows, gold, gold_rows, test):
    m = class_metrics(pred, gold, test)
    m["importance"] = importance_metrics(imp_bucket, gold_rows, test)
    m["per_source"] = per_source(pred, gold, rows, test)
    return m


def summarize(args):
    rows = load_rows()
    gold = {r["id"]: r["class"] for r in rows}
    gold_rows = {r["id"]: r for r in rows}
    train, test = split_ids(rows, "train"), split_ids(rows, "test")
    arms = {}
    raw = {}
    for name in ARMS:
        path = OUT / f"{name}.json"
        if path.exists():
            raw[name] = json.loads(path.read_text())
        else:
            print(f"missing {path}; run --arm {name}", file=sys.stderr)
    footprints = {}
    for name in ARMS:
        fp = OUT / f"{name}.time.json"
        if fp.exists():
            footprints[name] = json.loads(fp.read_text())

    heads = {}
    if "production" in raw:
        p = raw["production"]
        for label, thr in [("production", HOOK_MIN_CONFIDENCE), ("production@0.7", CLI_MIN_CONFIDENCE)]:
            pred, score, multi = production_view(p["raw"], thr)
            cuts = fit_cuts(score, gold_rows, train)
            bucket = {i: bucket_of(score[i], cuts["cuts"]) for i in score}
            entry = {k: v for k, v in p.items() if k not in ("raw", "arm")}
            entry["min_confidence"] = thr
            entry["texts_with_two_stored_captures"] = sum(1 for i in test if
                                                          len([c for c in p["raw"][i]["captures"] if c["confidence"] >= thr - 1e-6]) > 1)
            entry["importance_cuts"] = cuts
            entry["train"] = {k: v for k, v in class_metrics(pred, gold, train).items()
                              if k in ("macro_f1", "accuracy", "false_capture_rate")}
            entry["test"] = evaluate(pred, bucket, rows, gold, gold_rows, test)
            entry["predictions_test"] = {i: pred[i] for i in test}
            arms[label] = entry
    for name in HEADS + list(MODELS):
        if name not in raw:
            continue
        r = raw[name]
        entry = {k: v for k, v in r.items() if k not in ("predictions", "free_predictions", "importance",
                                                        "scores", "probabilities", "selection_curve", "arm",
                                                        "head", "importance_head")}
        if name in HEADS:
            entry["head_summary"] = {"classes": r["head"]["classes"], "tau": r["head"]["tau"],
                                     "weights_in": f"typed-decisions-u3-heads-<date>.json"}
            entry["importance_head_summary"] = {k: r["importance_head"][k]
                                                for k in ("train_items", "train_buckets", "C", "curve")}
        if name in footprints:
            entry.update({k: footprints[name][k] for k in ("peak_rss_mb", "peak_footprint_mb") if k in footprints[name]})
            entry["footprint_process"] = "the whole arm process under /usr/bin/time -l"
            if name in HEADS:
                entry["python_process_peak_rss_mb"] = entry.pop("peak_rss_mb", None)
                entry["python_process_peak_footprint_mb"] = entry.pop("peak_footprint_mb", None)
                entry["peak_rss_mb"] = r.get("peak_rss_mb")
                entry["peak_footprint_mb"] = r.get("peak_footprint_mb")
                entry["footprint_process"] = ("the Rust --embed process (bge-small loaded); the Python "
                                              "training process is listed as python_process_*")
        entry["train"] = {k: v for k, v in class_metrics(r["predictions"], gold, train).items()
                          if k in ("macro_f1", "accuracy", "false_capture_rate")}
        entry["test"] = evaluate(r["predictions"], r["importance"], rows, gold, gold_rows, test)
        if r.get("free_predictions"):
            m = class_metrics(r["free_predictions"], gold, test)
            entry["test_at_max_f1_train_setting"] = {
                "setting": r["selection"]["max_f1_train"],
                "note": "the unconstrained best-train-macro-F1 setting; for reference only, the verdict uses "
                        "the selected setting",
                **{k: m[k] for k in ("accuracy", "macro_f1", "false_capture_rate", "wrong_table_rate", "miss_rate")}}
        entry["predictions_test"] = {i: r["predictions"][i] for i in test}
        if name in HEADS:
            heads[name] = {**r["head"], "importance_head": r["importance_head"], "embedder": r["embedder"],
                           "training": r["training"], "selection": r["selection"]["chosen"]}
        arms[name] = entry

    verdict = decide(arms)
    if "production" in arms:
        verdict["control_errors_test"] = control_errors(raw["production"]["raw"], rows, gold)
        if verdict["verdict"] == "NO-GO":
            verdict["regex_refinement_needs"] = refinement_needs(verdict["control_errors_test"], arms)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prep = json.loads((OUT / "prepare.json").read_text()) if (OUT / "prepare.json").exists() else {}
    result = {
        "date": date,
        "decision": "U3: should auto-captured text become a memory record, in which table, at what importance",
        "set": "benchmarks/typed-decisions/sets/u3_repo.jsonl",
        "labels": "benchmarks/typed-decisions/validation/2026-10-14-u3-labels.json",
        "texts": {"train": len(train), "test": len(test)},
        "gold_test": dict(Counter(gold[i] for i in test)),
        "gold_test_by_source": {s: dict(Counter(gold[i] for i in test if gold_rows[i]["source"] == s)) for s in SOURCES},
        "importance_majority_baseline": importance_baseline(gold_rows, train, test),
        "metric": "single-label class over decision/error/context/rule/none on the test split. macro_f1 = mean "
                  "F1 over the classes present in gold or predictions. false_capture_rate = gold none "
                  "predicted as anything else / gold none (a junk record). wrong_table_rate = stored in the "
                  "wrong table / (gold not none and stored). miss_rate = gold not none predicted none / gold "
                  "not none. Importance accuracy is over gold not-none test items, whatever the arm's class.",
        "control": "production at the hook's --min-confidence 0.8 (the only automatic caller); "
                   "production@0.7 is the CLI default, for reference",
        "selection": "see the script docstring; every model setting was chosen on train only",
        "sampling_note": "half of each source's quota went to texts containing a production pattern, so "
                         "the regex fires more often here than on all hook traffic; rates are for this set",
        "weak_training_data": prep.get("weak_records"),
        "settings": {"threads": args.threads, "tau_grid": TAU_GRID,
                     "tau_grid_note": f"the first run's grid stopped at {FIRST_RUN_TAU_MAX}; no zero-shot "
                                      "setting met the train false-capture bar there, so the grid was extended "
                                      "toward 1 and the zero-shot arms re-selected on train from their saved "
                                      "scores (--reselect); each arm keeps its first_selection",
                     "c_grid": C_GRID,
                     "class_weightings": [str(w) for w in WEIGHTINGS], "cv_folds": CV_FOLDS,
                     "phrasings": PHRASINGS, "importance_labels": IMPORTANCE_LABELS, "device": "cpu",
                     "fit_bars": FIT_BARS},
        "machine": machine(),
        "versions": versions(),
        "prepare": prep,
        "arms": arms,
        "verdict": verdict,
        "numbers_policy": "Candidate measurements for a feature that is not built; not for README, docs or "
                          "CLI output. Latency was measured on a shared, heavily loaded machine; read it with "
                          "the load average recorded next to it.",
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"typed-decisions-u3-bakeoff-{date}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    if heads:
        hp = RESULTS / f"typed-decisions-u3-heads-{date}.json"
        hp.write_text(json.dumps({
            "date": date,
            "what": "the U3 logistic-regression heads on Axil's bge-small vector, as a matrix plus bias",
            "input": "Embedder::embed(text) with bge-small-en-v1.5 (the passage vector Axil stores for a "
                     "record; for a stored record the text is util::searchable_text of its data)",
            "rule": "z = W v + b; p = softmax(z); predict classes[argmax p], or none when that class is not "
                    "none and its probability < tau",
            "parity": "parity_fixtures give the Python probabilities for test texts of sets/u3_repo.jsonl; "
                      "a Rust port must reproduce them from its own embedding of the same text",
            "verdict": verdict["verdict"],
            "heads": heads}, indent=1) + "\n")
        print(f"wrote {hp.relative_to(HERE.parent.parent)}")
    print_table(arms)
    print(json.dumps({k: v for k, v in verdict.items() if k not in ("control_errors_test",)}, indent=1))
    print(f"wrote {path.relative_to(HERE.parent.parent)}")


def importance_baseline(gold_rows, train, test):
    """Accuracy of always answering train's most common bucket, on gold not-none test items."""
    top = Counter(gold_rows[i]["importance"] for i in train if gold_rows[i]["class"] != "none").most_common(1)[0][0]
    ids = [i for i in test if gold_rows[i]["class"] != "none"]
    return {"bucket": top, "accuracy": rnd(sum(gold_rows[i]["importance"] == top for i in ids) / len(ids)),
            "items": len(ids)}


def fits(entry):
    lat = entry["latency_ms"]
    rss = entry.get("peak_rss_mb") or 0.0
    checks = {"peak_rss_mb": (rss, rss <= FIT_BARS["peak_rss_mb_max"]),
              "p95_ms": (lat["p95"], lat["p95"] <= FIT_BARS["p95_ms_max"])}
    detail = {k: {"value": v, "ok": ok} for k, (v, ok) in checks.items()}
    detail["load_s"] = {"value": round(entry["load_ms"] / 1000, 2), "ok": None,
                        "note": "reported, not gated: paid once per worker or daemon process, but per job if "
                                "the model ran inside the queued `axil auto-capture` call"}
    return all(ok for _, ok in checks.values()), detail


def call_site(name, entry):
    if name.startswith("lr-"):
        return ("the queued `axil auto-capture` job (never the hook): one bge-small embed of the candidate, "
                "which the store path computes anyway, plus a 5x384 dot product")
    load_s = entry["load_ms"] / 1000
    return (f"the worker or the daemon only (never the hook): loading it per queued `axil auto-capture` job "
            f"would add {load_s:.1f} s to each failed command's capture")


def decide(arms):
    ctrl = arms.get("production")
    models = {n: a for n, a in arms.items() if n in HEADS or n in MODELS}
    if not ctrl or not models:
        return {"verdict": "incomplete", "reason": "control or model arms missing"}
    c = ctrl["test"]
    # The size preference only ranks candidates: a model already ruled out by
    # its false-capture rate does not set the bar the others must come near.
    eligible = [a["test"]["macro_f1"] for a in models.values()
                if a["test"]["false_capture_rate"] <= c["false_capture_rate"]]
    best = max(eligible) if eligible else max(a["test"]["macro_f1"] for a in models.values())
    best_all = max(a["test"]["macro_f1"] for a in models.values())
    rows = {}
    for n, a in models.items():
        t = a["test"]
        ok, detail = fits(a)
        fc_ok = t["false_capture_rate"] <= c["false_capture_rate"]
        rows[n] = {"macro_f1": t["macro_f1"], "accuracy": t["accuracy"],
                   "false_capture_rate": t["false_capture_rate"], "params": a["params"],
                   "beats_control_macro_f1": t["macro_f1"] > c["macro_f1"],
                   "false_capture_at_most_control": fc_ok,
                   "fits_cpu_call_site": ok, "fit_detail": detail, "call_site": call_site(n, a),
                   "within_1_point_of_best": fc_ok and best - t["macro_f1"] <= 0.01}
    within = [n for n, r in rows.items() if r["within_1_point_of_best"]]
    smallest = min(within, key=lambda n: models[n]["params"]) if within else None
    for n, r in rows.items():
        r["smallest_within_1_point"] = n == smallest
        r["wins"] = (r["beats_control_macro_f1"] and r["false_capture_at_most_control"]
                     and r["fits_cpu_call_site"] and r["smallest_within_1_point"])
    winners = [n for n, r in rows.items() if r["wins"]]
    alt = arms.get("production@0.7", {}).get("test", {})
    return {
        "verdict": f"GO {winners[0]}" if winners else "NO-GO",
        "rule": "a model wins only if (1) it beats the control's test macro-F1 at an equal or lower test "
                "false-capture rate, (2) it fits a CPU call site off the hook (FIT_BARS), and (3) it is the "
                "smallest model within 1 macro-F1 point of the best model that meets (1)'s false-capture "
                "condition; params count only what the arm adds (the LR heads reuse the bge-small vector "
                "Axil already computes)",
        "control": {k: c[k] for k in ("macro_f1", "accuracy", "false_capture_rate", "wrong_table_rate", "miss_rate")},
        "control_at_cli_default_0.7": {k: alt.get(k) for k in ("macro_f1", "accuracy", "false_capture_rate",
                                                             "wrong_table_rate", "miss_rate")},
        "best_model_macro_f1": best,
        "best_model_macro_f1_any_false_capture": best_all,
        "arms": rows,
        "caveat": "Test has 125 texts but few of some classes (decision 3, rule 3, context 10), so a macro-F1 "
                  "gap of a few points can rest on one or two texts; latency was measured at a 1-minute load "
                  "average of about 140-210 on a 10-core machine.",
    }


def control_errors(raw, rows, gold):
    """What the regex misses and what it stores wrongly on test, at the hook's threshold."""
    test = split_ids(rows, "test")
    pred, _, _ = production_view(raw, HOOK_MIN_CONFIDENCE)
    any_pred, _, _ = production_view(raw, 0.0)
    src = {r["id"]: r["source"] for r in rows}
    text = {r["id"]: r["text"] for r in rows}
    misses = Counter((gold[i], src[i]) for i in test if gold[i] != "none" and pred[i] == "none")
    fired_below = Counter(gold[i] for i in test if gold[i] != "none" and pred[i] == "none" and any_pred[i] != "none")
    false_caps = [{"id": i, "source": src[i], "stored_as": pred[i],
                   "summary": next(c["summary"] for c in raw[i]["captures"] if c["confidence"] >= HOOK_MIN_CONFIDENCE - 1e-6)[:120]}
                  for i in test if gold[i] == "none" and pred[i] != "none"]
    low_conf_false = [{"id": i, "source": src[i], "would_store_as": any_pred[i],
                       "confidence": max(c["confidence"] for c in raw[i]["captures"]),
                       "summary": raw[i]["captures"][0]["summary"][:120]}
                      for i in test if gold[i] == "none" and any_pred[i] != "none" and pred[i] == "none"]
    wrong = [{"id": i, "gold": gold[i], "stored_as": pred[i], "text": text[i][:120]}
             for i in test if gold[i] != "none" and pred[i] != "none" and pred[i] != gold[i]]
    return {
        "missed_by_gold_class_and_source": [{"gold": g, "source": s, "count": n}
                                            for (g, s), n in sorted(misses.items(), key=lambda kv: -kv[1])],
        "missed_but_fired_below_threshold": dict(fired_below),
        "false_captures": false_caps,
        "false_captures_below_threshold": low_conf_false,
        "wrong_table": wrong,
        "classes_the_regex_cannot_emit": ["context", "rule"],
    }


def refinement_needs(errors, arms):
    missed = Counter()
    missed_src = Counter()
    for e in errors["missed_by_gold_class_and_source"]:
        missed[e["gold"]] += e["count"]
        if e["gold"] == "error":
            missed_src[e["source"]] += e["count"]
    below = errors["missed_but_fired_below_threshold"]
    at_cli = [e for e in errors["false_captures_below_threshold"] if e["confidence"] >= CLI_MIN_CONFIDENCE - 1e-6]
    shapes = Counter()
    for e in at_cli:
        t = e["summary"].lower()
        shapes["zsh `=...` or glob error" if "(eval)" in t or "no matches found" in t
               else "regex parse error from the agent's own search" if "regex parse error" in t
               else "usage error (missing argument)" if "required arguments" in t
               else "git refusing the command" if t.startswith("fatal:")
               else "a source line printed by a search" if re.search(r"map_err|eprintln|errln|\.context\(", t)
               else "other"] += 1
    ctrl = arms["production"]["test"]
    return [
        f"Context and rule records: {missed['context']} context and {missed['rule']} rule test texts missed; "
        "analyze() has no pattern for either (CaptureType::Context exists but is never emitted, and there is "
        "no rule type), so the regex can only ever store errors and decisions.",
        f"Errors: {missed['error']} of {ctrl['per_class']['error']['gold']} error test texts missed "
        f"(by source: {dict(missed_src)}), {below.get('error', 0)} of them matched a pattern below the hook's "
        "0.8 bar. In raw output only rustc error codes, panics and tracebacks reach 0.8 (0.6 base, +0.1 for "
        "the 'exit code' line every hook text starts with, +0.1 or +0.2 for the specific format); an agent's "
        "prose report of what broke carries none of the output-shaped patterns.",
        f"Decisions: {missed['decision']} decision test texts missed: commit subjects and agent prose state "
        "the choice directly (\"X stays\", \"scan exactly instead of ...\") without the decision phrases. "
        "The one decision the hook bar stores is a false capture: a \"decision:\" list header inside a "
        "conditional plan, so the marker needs a guard for headers and conditionals (if, I'd).",
        f"Junk before the bar can move: at the CLI default 0.7, {len(at_cli)} gold-none test texts would also "
        f"be stored as errors ({dict(shapes)}): command-line mistakes (guideline N4) and printed code (N2) "
        "that reach 0.7 only through the 'exit code' bonus. A guard list for these shapes, and for a non-zero exit that is the "
        "command's answer (grep found nothing, diff, test), has to come before any lower threshold.",
        f"Importance: compute_importance ignores the class and the failure's severity; on gold not-none test "
        f"texts its bucket accuracy is {ctrl['importance']['accuracy']} with cut points fit on train, against "
        "the always-the-most-common-bucket baseline in importance_majority_baseline.",
    ]


def print_table(arms):
    print(f"{'arm':<16}{'acc':>7}{'macF1':>7}{'FC':>7}{'wrongT':>8}{'miss':>7}{'impAcc':>8}"
          f"{'p50ms':>9}{'p95ms':>9}{'load':>8}{'RSS MB':>8}")
    for n, a in arms.items():
        t = a["test"]
        lat = a["latency_ms"]
        imp = (t.get("importance") or {}).get("accuracy")
        print(f"{n:<16}{t['accuracy']:>7.3f}{t['macro_f1']:>7.3f}{t['false_capture_rate']:>7.3f}"
              f"{(t['wrong_table_rate'] or 0):>8.3f}{t['miss_rate']:>7.3f}{(imp or 0):>8.3f}"
              f"{lat['p50']:>9.2f}{lat['p95']:>9.2f}{a['load_ms']:>8.0f}{(a.get('peak_rss_mb') or 0):>8.0f}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[*ARMS, "all", "models"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=1, help="zero-shot arm processes at once (at most 2)")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reselect", action="store_true",
                    help="redo the zero-shot arms' train-only selection from their saved scores")
    ap.add_argument("--db", help="live Axil DB, read with `axil export` only (--prepare)")
    ap.add_argument("--axil", default="axil")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))

    if args.prepare:
        prepare(args)
    if args.arm in ("all", "models"):
        from concurrent.futures import ThreadPoolExecutor

        def one(name):
            cmd = [sys.executable, __file__, "--arm", name, "--threads", str(args.threads)]
            out, err, fp = timed(cmd)
            sys.stdout.write(out)
            (OUT / f"{name}.time.json").write_text(json.dumps(fp, indent=1))

        names = ARMS if args.arm == "all" else list(MODELS)
        light = [n for n in names if n not in MODELS]
        for name in light:  # the zero-shot arms read the control's output
            one(name)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(one, [n for n in names if n in MODELS]))
        args.summarize = True
    elif args.arm == "production":
        run_production(args)
    elif args.arm in HEADS:
        run_head(args.arm, args)
    elif args.arm:
        run_model(args.arm, args)
    if args.reselect:
        for name in MODELS:
            if (OUT / f"{name}.json").exists():
                reselect(name)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
