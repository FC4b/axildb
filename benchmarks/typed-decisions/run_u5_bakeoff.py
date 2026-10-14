#!/usr/bin/env python3
"""U5 bake-off: entity extraction on sets/u5_repo.jsonl.

Arms:
  production          Axil's own extractor, through the Rust crate in heuristic/
                      (`--entities`): what `Axil::auto_link` links on insert.
  gliner2.5-small     fastino/gliner2.5-small-v1 (gliner2 AutoExtractor)
  gliner_small-v2.1   urchade/gliner_small-v2.1  (gliner GLiNER)
  gliner2.5-base      fastino/gliner2.5-base-v1  (gliner2 AutoExtractor)

Mentions are compared as normalized surface forms (`adjudicate_u5.norm`:
case-insensitive, outer backticks/quotes and trailing punctuation stripped),
one form per text. Every metric is on the `test` split; `train` is used only
to pick each GLiNER arm's label phrasing and span threshold:

  * each phrasing runs once over all texts at FLOOR_THRESHOLD, keeping every
    span's score; filtering those spans at t gives the output of a run at t
    (both libraries decode greedily by descending score), so thresholds are
    swept offline;
  * selection, fixed before looking at test: the threshold with the highest
    train span F1 among those whose train precision is at least the
    production extractor's train precision (precision plays the
    false-positive role); if none reaches it, the highest train F1. The
    phrasing with the better selected train F1 wins;
  * the chosen setting then runs for real on test, batch 1, model loaded,
    which gives the reported predictions and latency (and checks the sweep
    shortcut: `sweep_mismatch_texts`).

Usage:
  python run_u5_bakeoff.py --arm all              # every arm, one process each, then summarize
  python run_u5_bakeoff.py --arm gliner2.5-small  # one arm (raw output to out/u5/)
  python run_u5_bakeoff.py --summarize            # rebuild the results JSON from out/u5/
  python run_u5_bakeoff.py --write-junk-list      # regenerate u5_junk_words.txt

Raw outputs go to out/u5/ (gitignored); the summary goes to
benchmarks/results/typed-decisions-u5-bakeoff-<date>.json.
"""

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from adjudicate_u5 import norm

HERE = Path(__file__).resolve().parent
SET = HERE / "sets" / "u5_repo.jsonl"
TEXTS = HERE / "sets" / "u5_texts.jsonl"
OUT = HERE / "out" / "u5"
RESULTS = HERE.parent / "results"
JUNK_LIST = HERE / "u5_junk_words.txt"
HEURISTIC = HERE / "heuristic"
HEURISTIC_BIN = HEURISTIC / "target" / "release" / "typed-decisions-heuristic"

TYPES = ["file", "code", "project", "concept"]
STRATA = ["code_heavy", "mixed", "junk_trap", "markdown", "prose"]

GLINER25_FILES = ["config.json", "encoder_config/config.json", "model.safetensors",
                  "tokenizer.json", "tokenizer_config.json"]
MODELS = {
    "gliner2.5-small": dict(lib="gliner2", repo="fastino/gliner2.5-small-v1",
                            revision="df5910e44bc4ffdb0d95399a83b0ca4516349aa5",
                            files=GLINER25_FILES),
    "gliner_small-v2.1": dict(lib="gliner", repo="urchade/gliner_small-v2.1",
                              revision="4e091416cf7c3481db542c2a3d26156916f3a47f",
                              files=["gliner_config.json", "pytorch_model.bin"],
                              # The checkpoint names its encoder by HF id; gliner reads that
                              # repo's config and tokenizer at load time (the encoder weights
                              # come from the checkpoint, so the backbone's are not fetched).
                              backbone=dict(repo="microsoft/deberta-v3-small",
                                            revision="a36c739020e01763fe789b4b85e2df55d6180012",
                                            files=["config.json", "spm.model",
                                                   "tokenizer_config.json"])),
    "gliner2.5-base": dict(lib="gliner2", repo="fastino/gliner2.5-base-v1",
                           revision="54785d51df8d86a0d8f2eb212fec15871aad2c4d",
                           files=GLINER25_FILES),
}
ARMS = ["production", *MODELS]

# At most two label phrasings per model, natural-language names for the
# guideline types; chosen per model on train.
PHRASINGS = {
    "short": {
        "file": "file path",
        "code": "code identifier",
        "project": "software product",
        "concept": "technical concept",
    },
    "descriptive": {
        "file": "file or directory path",
        "code": "function, type, command or flag name",
        "project": "library, tool, programming language or model name",
        "concept": "named algorithm or technique",
    },
}
FLOOR_THRESHOLD = 0.05
GRID = [round(0.05 + 0.01 * k, 2) for k in range(91)]  # 0.05 .. 0.95

# Axil EntityType -> guideline type. `reference` is a quoted string of three
# or more words; it is judged by its content: a path is a file, anything else
# a multi-word phrase, the closest guideline type to which is concept.
AXIL_TYPE_MAP = {"file": "file", "code": "code", "code_symbol": "code", "project": "project"}

# A write-time call site off the hook path (the worker pass or the daemon)
# loads the model once and runs it per stored record, so load time is reported
# but does not gate (it matters in a hook, which is ruled out anyway). These
# are the bars for "fits": generous for a background pass, but they rule out a
# model that would dominate the daemon's memory or fall behind an agent's
# write rate.
FIT_BARS = {"peak_rss_mb_max": 1500.0, "p95_ms_max": 1000.0}


# ── Data and scoring ────────────────────────────────────────────────


def load_rows():
    return [json.loads(line) for line in SET.read_text().splitlines() if line.strip()]


def gold_forms(rows):
    return {r["id"]: {norm(e["text"]): e["type"] for e in r["entities"]} for r in rows}


def to_forms(mentions):
    """[(text, type, score, ...)] -> {norm: type}, keeping the best-scored type per form."""
    best = {}
    for text, typ, score, *_ in mentions:
        n = norm(text)
        if n and (n not in best or score > best[n][1]):
            best[n] = (typ, score)
    return {k: v[0] for k, v in best.items()}


def prf(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * tp / (n_pred + n_ref) if (n_pred + n_ref) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "pred": n_pred, "gold": n_ref}


def score(pred, gold, ids, type_filter=None, typed=False):
    tp = n_p = n_g = 0
    for i in ids:
        p, g = pred.get(i, {}), gold[i]
        if type_filter:
            p = {k: v for k, v in p.items() if v == type_filter}
            g = {k: v for k, v in g.items() if v == type_filter}
        n_p += len(p)
        n_g += len(g)
        tp += sum(1 for k, v in p.items() if k in g and (not (typed or type_filter) or g[k] == v))
    return prf(tp, n_p, n_g)


def load_junk_list():
    words = set()
    for line in JUNK_LIST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line)
    return words


MARKDOWN = re.compile(
    r"^(?:#{1,6}|[-*+]|\d+[.)]|- \[[ xX]\]|\[[ xX]\]|\*\*.*\*\*:?|```.*|>|---+|"
    r"(?:feat|fix|docs|chore|bench|refactor|test|perf|ci|build|style|wip)(?:\([^)]*\))?!?:)$"
)


def is_junk(form, junk_words):
    """A wrong mention that is markdown syntax, or plain words that are all in the list.

    Every whitespace-separated token must be a plain word (letters, inner
    hyphens or apostrophes allowed) and each of its parts listed; a token with
    code syntax (`--flag`, `a::b`, `x.rs`, `snake_case`, digits) is never junk.
    """
    if not re.search(r"[A-Za-z0-9]", form) or MARKDOWN.match(form):
        return True
    for token in form.split():
        if not re.fullmatch(r"[a-z]+(?:[-'][a-z]+)*", token):
            return False
        if not all(part in junk_words for part in token.split("-")):
            return False
    return True


def metrics(pred, gold, rows, junk_words):
    test = [r["id"] for r in rows if r["split"] == "test"]
    by_stratum = {s: [r["id"] for r in rows if r["split"] == "test" and r["stratum"] == s]
                  for s in STRATA}
    wrong = [(i, f) for i in test for f in pred.get(i, {}) if f not in gold[i]]
    junk = [(i, f) for i, f in wrong if is_junk(f, junk_words)]
    n_pred = sum(len(pred.get(i, {})) for i in test)
    counts = [len(pred.get(i, {})) for i in test]
    junk_counts = {}
    for _, f in junk:
        junk_counts[f] = junk_counts.get(f, 0) + 1
    return {
        "span": score(pred, gold, test),
        "span_and_type": score(pred, gold, test, typed=True),
        "per_type": {t: score(pred, gold, test, type_filter=t) for t in TYPES},
        "per_stratum": {s: {"texts": len(ids), "span": score(pred, gold, ids),
                            "span_and_type": score(pred, gold, ids, typed=True)}
                        for s, ids in by_stratum.items()},
        "junk": {"junk_mentions": len(junk), "wrong_mentions": len(wrong), "predicted_mentions": n_pred,
                 "junk_rate": round(len(junk) / n_pred, 4) if n_pred else 0.0,
                 "junk_share_of_wrong": round(len(junk) / len(wrong), 4) if wrong else 0.0,
                 "top": sorted(junk_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]},
        "entities_per_text": {"mean": round(statistics.mean(counts), 2),
                              "median": statistics.median(counts),
                              "gold_mean": round(statistics.mean(len(gold[i]) for i in test), 2)},
    }


# ── Environment ─────────────────────────────────────────────────────


def loadavg():
    return round(os.getloadavg()[0], 2)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def current_rss_mb():
    import psutil

    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def pct(values, q):
    s = sorted(values)
    return round(s[int(q * (len(s) - 1))], 3)


def machine():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for key, name in [("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "memory_bytes")]:
            try:
                info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                            text=True).stdout.strip()
            except OSError:
                pass
    return info


# ── Production arm ──────────────────────────────────────────────────


def production_type(axil_type, text):
    if axil_type in AXIL_TYPE_MAP:
        return AXIL_TYPE_MAP[axil_type]
    return "file" if re.search(r"[/\\]|^\.\w|\.\w{1,10}$", text) and " " not in text else "concept"


def run_production(args):
    if not HEURISTIC_BIN.exists():
        sys.exit(f"build it first: cargo build --release --manifest-path {HEURISTIC}/Cargo.toml")
    if sys.platform == "darwin":
        cmd = ["/usr/bin/time", "-l", str(HEURISTIC_BIN), "--entities", str(SET)]
    else:
        cmd = ["/usr/bin/time", "-v", str(HEURISTIC_BIN), "--entities", str(SET)]
    load_before = loadavg()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_ms = (time.perf_counter() - t0) * 1000
    load_after = loadavg()
    rss_line = next(line for line in proc.stderr.splitlines()
                    if "maximum resident set size" in line.lower())
    rss = int(re.search(r"(\d+)", rss_line).group(1))
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    meta = next(json.loads(line) for line in proc.stderr.splitlines() if line.startswith("{"))
    rows = {r["id"]: r for r in load_rows()}
    raw = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    preds, lat = {}, {}
    for r in raw:
        preds[r["id"]] = [[e["text"], production_type(e["axil_type"], e["text"]), 1.0, e["axil_type"]]
                          for e in r["entities"]]
        lat[r["id"]] = r["extract_us"] / 1000
    test_lat = [lat[i] for i in lat if rows[i]["split"] == "test"]
    out = {
        "arm": "production",
        "what": "axil_core::extract_entities over util::record_text of a record {summary: text}, "
                "truncated to 50: the no-LLM path Axil::auto_link runs on insert",
        "binary": str(HEURISTIC_BIN.relative_to(HERE)),
        "type_map": {**AXIL_TYPE_MAP, "reference": "file if it looks like a path, else concept"},
        "surface_form": "Entity.source_text (Entity.name lowercases and snake-cases CamelCase)",
        "predictions": preds,
        "latency_ms": {"p50": pct(test_lat, 0.5), "p95": pct(test_lat, 0.95),
                       "loadavg_1m_before": load_before, "loadavg_1m_after": load_after,
                       "measured_in": "Rust, Instant around record_text + extract_entities, per test text"},
        "load_ms": round(meta["first_call_us"] / 1000, 3),
        "load_note": "first call, which compiles the regex patterns (LazyLock)",
        "peak_rss_mb": round(rss_mb, 1),
        "rss_note": "whole heuristic process (it links axil-core and axil-vector; no model is loaded "
                    "in --entities mode)",
        "size_on_disk_bytes": 0,
        "size_note": "no model file: the patterns are compiled into axil-core",
        "wall_ms": round(wall_ms, 1),
    }
    write_arm(out)
    return out


# ── GLiNER arms ─────────────────────────────────────────────────────


def snapshot(repo, revision, files):
    """Download only `files` of `repo` at `revision`; returns the local dir and their bytes."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, revision=revision, allow_patterns=files)
    return path, sum(os.path.getsize(os.path.realpath(os.path.join(path, f))) for f in files)


class Gliner2Arm:
    def __init__(self, path):
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(path, map_location="cpu")
        self.model.eval()

    def predict(self, text, labels, threshold):
        names = list(labels.values())
        back = {v: k for k, v in labels.items()}
        res = self.model.extract_entities(text, names, threshold=threshold,
                                          include_confidence=True, include_spans=True)
        out = []
        for name, items in res.get("entities", {}).items():
            for it in items:
                out.append([it["text"], back[name], float(it["confidence"])])
        return out


class GlinerArm:
    def __init__(self, path):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(path, map_location="cpu")
        self.model.eval()

    def predict(self, text, labels, threshold):
        names = list(labels.values())
        back = {v: k for k, v in labels.items()}
        res = self.model.predict_entities(text, names, threshold=threshold)
        return [[it["text"], back[it["label"]], float(it["score"])] for it in res]


def param_count(arm):
    return sum(p.numel() for p in arm.model.parameters())


def filtered(spans, t):
    return [s for s in spans if s[2] >= t]


def run_model(name, args):
    import torch

    torch.set_num_threads(args.threads)
    spec = MODELS[name]
    rows = load_rows()
    gold = gold_forms(rows)
    train = [r["id"] for r in rows if r["split"] == "train"]
    rss_after_import = current_rss_mb()

    path, files_bytes = snapshot(spec["repo"], spec["revision"], spec["files"])
    backbone = None
    if "backbone" in spec:
        b = spec["backbone"]
        _, b_bytes = snapshot(b["repo"], b["revision"], b["files"])
        backbone = {**b, "bytes": b_bytes}
        files_bytes += b_bytes

    load_before = loadavg()
    t0 = time.perf_counter()
    arm = (Gliner2Arm if spec["lib"] == "gliner2" else GlinerArm)(path)
    load_ms = (time.perf_counter() - t0) * 1000
    load_after = loadavg()
    rss_after_load = current_rss_mb()
    for _ in range(3):
        arm.predict("Warm up crates/axil-core/src/db.rs and Axil::open in redb.", PHRASINGS["short"], 0.5)

    control = json.loads((OUT / "production.json").read_text())
    ctrl_forms = {i: to_forms(m) for i, m in control["predictions"].items()}
    ctrl_train_p = score(ctrl_forms, gold, train)["precision"]

    sweeps = {}
    for pname, labels in PHRASINGS.items():
        spans = {}
        for r in rows:
            with torch.inference_mode():
                spans[r["id"]] = arm.predict(r["text"], labels, FLOOR_THRESHOLD)
        curve = []
        for t in GRID:
            forms = {i: to_forms(filtered(s, t)) for i, s in spans.items()}
            m = score(forms, gold, train)
            curve.append({"threshold": t, **{k: m[k] for k in ("precision", "recall", "f1")}})
        feasible = [c for c in curve if c["precision"] >= ctrl_train_p]
        pool, constrained = (feasible, True) if feasible else (curve, False)
        best = max(pool, key=lambda c: (c["f1"], c["threshold"]))
        best_free = max(curve, key=lambda c: (c["f1"], c["threshold"]))
        sweeps[pname] = {"labels": labels, "spans": spans, "curve": curve,
                         "selected": {**best, "precision_constraint_met_on_train": constrained},
                         "max_f1_train": best_free}
    pick = max(sweeps, key=lambda p: (sweeps[p]["selected"]["precision_constraint_met_on_train"],
                                      sweeps[p]["selected"]["f1"]))
    thr = sweeps[pick]["selected"]["threshold"]
    labels = PHRASINGS[pick]

    # Real run on test at the chosen setting: predictions and latency.
    test_rows = [r for r in rows if r["split"] == "test"]
    preds, lat, mismatch = {}, [], 0
    lb = loadavg()
    for r in test_rows:
        t = time.perf_counter()
        with torch.inference_mode():
            p = arm.predict(r["text"], labels, thr)
        lat.append((time.perf_counter() - t) * 1000)
        preds[r["id"]] = p
        if to_forms(p) != to_forms(filtered(sweeps[pick]["spans"][r["id"]], thr)):
            mismatch += 1
    la = loadavg()
    # The unconstrained max-F1 point, also on test, from the sweep (for reference).
    free_thr = sweeps[pick]["max_f1_train"]["threshold"]
    free_preds = {r["id"]: filtered(sweeps[pick]["spans"][r["id"]], free_thr) for r in test_rows}

    out = {
        "arm": name,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "library": spec["lib"],
        "backbone": backbone,
        "params": param_count(arm),
        "size_on_disk_bytes": files_bytes,
        "load_ms": round(load_ms, 1),
        "load_loadavg_1m": [load_before, load_after],
        "rss_mb": {"after_import": rss_after_import, "after_load": rss_after_load},
        "peak_rss_mb": peak_rss_mb(),
        "threads": args.threads,
        "control_train_precision": ctrl_train_p,
        "phrasing": pick,
        "labels": labels,
        "threshold": thr,
        "selection": {p: {"selected": s["selected"], "max_f1_train": s["max_f1_train"]}
                      for p, s in sweeps.items()},
        "curves_train": {p: s["curve"] for p, s in sweeps.items()},
        "predictions": preds,
        "max_f1_point": {"threshold": free_thr, "predictions": free_preds},
        "sweep_mismatch_texts": mismatch,
        "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "mean": round(statistics.mean(lat), 3),
                       "loadavg_1m_before": lb, "loadavg_1m_after": la,
                       "measured_in": "Python, perf_counter around one predict call per test text, "
                                      "batch 1, model loaded and warmed"},
    }
    write_arm(out)
    return out


def write_arm(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{out['arm']}.json").write_text(json.dumps(out, indent=1))
    lat = out["latency_ms"]
    print(f"{out['arm']:<18} p50 {lat['p50']} ms p95 {lat['p95']} ms (load avg {lat['loadavg_1m_before']}"
          f"->{lat['loadavg_1m_after']}), load {out['load_ms']} ms, peak RSS {out['peak_rss_mb']} MB",
          flush=True)


# ── Summary and verdict ─────────────────────────────────────────────


def versions():
    out = {}
    for mod in ("torch", "transformers", "tokenizers", "gliner", "gliner2", "peft", "sentencepiece",
                "huggingface_hub"):
        try:
            from importlib.metadata import version

            out[mod] = version(mod)
        except Exception:  # noqa: BLE001 - a missing package just isn't listed
            pass
    return out


def summarize(args):
    rows = load_rows()
    gold = gold_forms(rows)
    junk_words = load_junk_list()
    arms = {}
    for name in ARMS:
        path = OUT / f"{name}.json"
        if not path.exists():
            print(f"missing {path}; run --arm {name}", file=sys.stderr)
            continue
        raw = json.loads(path.read_text())
        pred = {i: to_forms(m) for i, m in raw["predictions"].items()}
        entry = {k: v for k, v in raw.items()
                 if k not in ("predictions", "max_f1_point", "curves_train")}
        entry["test"] = metrics(pred, gold, rows, junk_words)
        if "max_f1_point" in raw:
            free = {i: to_forms(m) for i, m in raw["max_f1_point"]["predictions"].items()}
            m = metrics(free, gold, rows, junk_words)
            entry["test_at_max_f1_train_threshold"] = {
                "threshold": raw["max_f1_point"]["threshold"],
                "note": "the unconstrained best-train-F1 threshold, from the sweep; for "
                        "reference only, the verdict uses the selected setting",
                **{k: m[k] for k in ("span", "span_and_type", "per_type", "entities_per_text")},
                "junk_rate": m["junk"]["junk_rate"]}
        arms[name] = entry

    verdict = decide(arms)
    if "production" in arms:
        errors = control_errors(rows)
        verdict["control_errors_test"] = errors
        if verdict["verdict"] == "NO-GO":
            verdict["regex_refinement_needs"] = refinement_needs(errors)
    result = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "decision": "U5: which strings in a stored note are entities worth linking",
        "set": "benchmarks/typed-decisions/sets/u5_repo.jsonl",
        "labels": "benchmarks/typed-decisions/validation/2026-10-14-u5-labels.json",
        "texts": {"train": sum(r["split"] == "train" for r in rows),
                  "test": sum(r["split"] == "test" for r in rows)},
        "gold_mentions_test": sum(len(gold[r["id"]]) for r in rows if r["split"] == "test"),
        "metric": "mention-level micro precision/recall/F1 on the test split over normalized "
                  "surface forms (adjudicate_u5.norm), one form per text; span = form match, "
                  "span_and_type = form and type match. The verdict uses span F1 and precision.",
        "junk_definition": "a predicted test mention that is not gold and is markdown syntax or "
                           "made only of words in u5_junk_words.txt (stopwords, generic nouns, "
                           "language keywords, dictionary words); junk_rate = junk / predicted",
        "selection": "per GLiNER arm, on train only: phrasing and threshold with the best train "
                     "span F1 at train precision >= the production extractor's; see the script "
                     "docstring",
        "settings": {"threads": args.threads, "floor_threshold": FLOOR_THRESHOLD,
                     "grid": [GRID[0], GRID[-1], 0.01], "phrasings": PHRASINGS, "device": "cpu",
                     "fit_bars": FIT_BARS},
        "machine": machine(),
        "versions": versions(),
        "arms": arms,
        "verdict": verdict,
        "numbers_policy": "Candidate measurements for a feature that is not built; not for README, "
                          "docs or CLI output. Latency was measured on a shared machine; read it "
                          "with the load average recorded next to it.",
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"typed-decisions-u5-bakeoff-{result['date']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print_table(arms)
    print(json.dumps(verdict, indent=1))
    print(f"wrote {path.relative_to(HERE.parent.parent)}")


def fits(entry):
    lat = entry["latency_ms"]
    checks = {
        "peak_rss_mb": (entry["peak_rss_mb"], entry["peak_rss_mb"] <= FIT_BARS["peak_rss_mb_max"]),
        "p95_ms": (lat["p95"], lat["p95"] <= FIT_BARS["p95_ms_max"]),
    }
    detail = {k: {"value": v, "ok": ok} for k, (v, ok) in checks.items()}
    detail["load_s"] = {"value": round(entry["load_ms"] / 1000, 2), "ok": None,
                        "note": "reported, not gated: loaded once per worker or daemon process"}
    return all(ok for _, ok in checks.values()), detail


def decide(arms):
    ctrl = arms.get("production")
    if not ctrl:
        return {"verdict": "incomplete", "reason": "production arm missing"}
    c = ctrl["test"]["span"]
    models = {n: a for n, a in arms.items() if n != "production"}
    if not models:
        return {"verdict": "incomplete", "reason": "no model arm"}
    best_f1 = max(a["test"]["span"]["f1"] for a in models.values())
    rows = {}
    for n, a in models.items():
        s = a["test"]["span"]
        fit_ok, fit_detail = fits(a)
        within = best_f1 - s["f1"] <= 0.01
        rows[n] = {
            "f1": s["f1"], "precision": s["precision"], "params": a["params"],
            "beats_control_f1": s["f1"] > c["f1"],
            "precision_at_least_control": s["precision"] >= c["precision"],
            "fits_write_time_cpu": fit_ok, "fit_detail": fit_detail,
            "within_1_f1_point_of_best": within,
        }
    candidates = [n for n, r in rows.items() if r["within_1_f1_point_of_best"]]
    smallest = min(candidates, key=lambda n: models[n]["params"])
    for n, r in rows.items():
        r["smallest_within_1_point"] = n == smallest
        r["wins"] = (r["beats_control_f1"] and r["precision_at_least_control"]
                     and r["fits_write_time_cpu"] and r["smallest_within_1_point"])
    winners = [n for n, r in rows.items() if r["wins"]]
    return {
        "verdict": f"GO {winners[0]}" if winners else "NO-GO",
        "rule": "a model wins only if (1) it beats the production extractor's test span F1 at equal "
                "or better test precision, (2) it fits a write-time CPU call site (worker or daemon, "
                "never a hook; FIT_BARS), and (3) it is the smallest model within 1 F1 point of the "
                "best model",
        "control": {"f1": c["f1"], "precision": c["precision"], "recall": c["recall"]},
        "best_model_f1": best_f1,
        "arms": rows,
    }


def shape(text):
    """Coarse surface shape of a mention, for the error breakdown."""
    if text.startswith("--"):
        return "long flag"
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", text):
        return "ALL_CAPS word"
    if "::" in text:
        return "qualified path"
    if " " in text:
        return "multi-word"
    if "/" in text:
        return "slash path"
    if re.fullmatch(r"\.?[\w.-]+\.\w{1,6}", text):
        return "bare file name"
    if "-" in text:
        return "hyphenated"
    if "_" in text:
        return "snake_case"
    if re.fullmatch(r"[A-Z][a-z0-9]+", text):
        return "Capitalized word"
    if re.search(r"[a-z][A-Z]|[A-Z]{2}", text):
        return "mixed case"
    if re.fullmatch(r"[a-z0-9.]+", text):
        return "lowercase word"
    return "other"


def control_errors(rows):
    """Where the production extractor misses and what it gets wrong, on test."""
    gold = gold_forms(rows)
    verbatim = {r["id"]: {norm(e["text"]): e["text"] for e in r["entities"]} for r in rows}
    raw = json.loads((OUT / "production.json").read_text())["predictions"]
    test = [r["id"] for r in rows if r["split"] == "test"]
    misses, wrong = {}, {}
    for i in test:
        pred = to_forms(raw[i])
        for f, typ in gold[i].items():
            if f not in pred:
                key = f"{typ} / {shape(verbatim[i][f])}"
                misses.setdefault(key, []).append(verbatim[i][f])
        seen = set()
        for text, _, _, axil_type in raw[i]:
            f = norm(text)
            if f in gold[i] or f in seen:
                continue
            seen.add(f)
            overlap = any(f in g or g in f for g in gold[i])
            key = f"{axil_type} / {'part or superset of a gold name' if overlap else 'no gold overlap'}"
            wrong.setdefault(key, []).append(text[:80])

    def table(d):
        return [{"kind": k, "count": len(v), "examples": v[:6]}
                for k, v in sorted(d.items(), key=lambda kv: -len(kv[1]))]

    return {"missed_gold_by_type_and_shape": table(misses),
            "wrong_by_axil_type": table(wrong)}


def refinement_needs(errors):
    """What the production patterns still miss or mint, in test-split counts."""
    missed = {e["kind"]: e["count"] for e in errors["missed_gold_by_type_and_shape"]}
    wrong = {e["kind"]: e["count"] for e in errors["wrong_by_axil_type"]}

    def m(*kinds):
        return sum(missed.get(k, 0) for k in kinds)

    project = sum(v for k, v in missed.items() if k.startswith("project /"))
    return [
        f"Project names: {project} missed gold mentions; there is no project pattern at all. "
        "Shape cannot separate redb, ring or Windows from English, so this needs a gazetteer "
        "(dependency names from Cargo.lock / _deps, the workspace crates, a curated list of "
        "tools, languages, platforms and models), not another regex.",
        f"Long flags (--name): {m('code / long flag')} missed; a plain pattern.",
        f"File names without a slash and companion files (db.rs, Cargo.lock, .axil.vec, README): "
        f"{m('file / bare file name', 'file / lowercase word', 'file / ALL_CAPS word')} missed; "
        "a name.ext pattern over known extensions plus the .axil companion suffixes. Slash paths "
        f"without an extension or ending in / ({m('file / slash path')} missed) and git branches "
        f"({m('code / slash path')} missed) need their own rules; the path trimmer also strips "
        "a leading dot (.claude/CLAUDE.md becomes claude/CLAUDE.md).",
        f"Commands (axil heal, cargo check): {m('code / multi-word')} missed; program plus "
        "subcommand words up to the first flag, for known programs.",
        f"Hyphenated crate and tool names: {m('code / hyphenated')} code and "
        f"{missed.get('project / hyphenated', 0)} project mentions missed; the workspace crate "
        "list covers the code ones.",
        f"ALL_CAPS constants and env vars: {m('code / ALL_CAPS word')} missed; "
        "[A-Z][A-Z0-9]*_[A-Z0-9_]+ (an underscore keeps acronyms out).",
        f"Precision: {wrong.get('code / part or superset of a gold name', 0)} wrong code mentions "
        "are parts of a qualified path emitted next to it (axil_vector beside "
        "axil_vector::read_stored_dimensions) or a whole backticked command line kept as one "
        "entity; quoted strings of three or more words gave "
        f"{sum(v for k, v in wrong.items() if k.startswith('reference /'))} wrong mentions and no "
        "correct one.",
    ]


def print_table(arms):
    print(f"{'arm':<18}{'P':>7}{'R':>7}{'F1':>7}{'typedF1':>9}{'junk':>7}{'ents/txt':>9}"
          f"{'p50ms':>9}{'p95ms':>9}{'load':>9}{'RSS MB':>8}")
    for n, a in arms.items():
        t = a["test"]
        lat = a["latency_ms"]
        print(f"{n:<18}{t['span']['precision']:>7.3f}{t['span']['recall']:>7.3f}{t['span']['f1']:>7.3f}"
              f"{t['span_and_type']['f1']:>9.3f}{t['junk']['junk_rate']:>7.3f}"
              f"{t['entities_per_text']['mean']:>9.2f}{lat['p50']:>9.2f}{lat['p95']:>9.2f}"
              f"{a['load_ms']:>9.0f}{a['peak_rss_mb']:>8.0f}")


# ── Junk list ───────────────────────────────────────────────────────

STOPWORDS = """a about above after again against all also am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did didn't do does
doesn't doing don't down during each either else even ever every few for from further had hadn't
has hasn't have haven't having he her here hers herself him himself his how however i if in into
is isn't it it's its itself just least less let's like may me might more most much must mustn't
my myself neither never no nor not now of off often on once only or other ought our ours
ourselves out over own per rather same shall shan't she should shouldn't so some such than that
that's the their theirs them themselves then there there's these they this those though through
thus to too under until up upon us very via was wasn't we were weren't what what's when where
whether which while who whom whose why will with within without won't would wouldn't yet you your
yours yourself yourselves""".split()

# Guidelines section 5.1 (generic technical nouns) and 6 (plain-word examples),
# plus generic words of this domain that name no specific thing.
GENERIC = """api abi agent agents branch cache ci cli config cpu csv daemon database db embedding
embeddings fts gpu graph hook hooks http index json llm lock model models nli oom pr query recall
regex release rss sdk search store test tests toml tty user vector yaml core rules summary resolved
heal decide deps file files code record records table tables field fields error errors fix fixes
path paths command commands feature features default build run open new get set main parse result
results value values type types name names list data text line lines check commit commits merge
session sessions context decision decisions memory memories entity entities engine engines plugin
plugins extension extensions adapter adapters snapshot eval benchmark benchmarks score scores rank
ranking window gate budget token tokens chunk chunks candidate candidates threshold step steps
phase task tasks item items review fixture fixtures crate crates module modules function functions
method methods struct trait key keys flag flags option options binary output input format schema
version versions sha id ids note notes doc docs readme""".split()

# Guidelines 5.5 (keywords and basic types) and 5.7 (generic method names).
KEYWORDS = """fn impl struct trait enum pub dyn async mod def class func function let self true false
none some ok err bool str string usize u8 u16 u32 u64 i32 i64 f32 f64 option result vec box arc t e
new open get set run build parse default main""".split()


def write_junk_list():
    rows = [json.loads(line) for line in SET.read_text().splitlines() if line.strip()]
    gold_project = {norm(e["text"]) for r in rows for e in r["entities"] if e["type"] == "project"}
    dict_path = Path("/usr/share/dict/words")
    dictionary = {w.strip().lower() for w in dict_path.read_text().splitlines() if w.strip()}
    tokens = {t.lower() for r in rows for t in re.findall(r"[A-Za-z]+(?:'[a-z]+)?", r["text"])}
    in_dict = sorted((tokens & dictionary) - gold_project - set(STOPWORDS) - set(GENERIC) - set(KEYWORDS))
    lines = [
        "# Words that make a wrong U5 mention junk (run_u5_bakeoff.py --write-junk-list).",
        "# A wrong mention is junk when it is markdown syntax or every word of it is listed here.",
        "# Matching is on the normalized (lowercased) form.",
        "",
        "## stopwords",
        *sorted(set(STOPWORDS)),
        "",
        "## generic nouns (guidelines 5.1 and 6, plus generic words of this domain)",
        *sorted(set(GENERIC) - set(STOPWORDS)),
        "",
        "## language keywords, basic types, generic method names (guidelines 5.5, 5.7)",
        *sorted(set(KEYWORDS) - set(STOPWORDS) - set(GENERIC)),
        "",
        "## dictionary words: alphabetic tokens of sets/u5_repo.jsonl found in /usr/share/dict/words",
        "## (Webster's Second International), minus forms the gold set labels as project anywhere",
        "## (cargo, ring, rust, axil, ...: tool and product names that are also English words)",
        *in_dict,
    ]
    JUNK_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {JUNK_LIST.name}: {len(in_dict)} dictionary words; skipped project forms "
          f"{sorted(gold_project & tokens & dictionary)}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[*ARMS, "all"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--write-junk-list", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))

    if args.write_junk_list:
        write_junk_list()
        return
    if args.arm == "all":
        for name in ARMS:
            subprocess.run([sys.executable, __file__, "--arm", name, "--threads", str(args.threads)],
                           check=True)
        args.summarize = True
    elif args.arm == "production":
        run_production(args)
    elif args.arm:
        run_model(args.arm, args)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
