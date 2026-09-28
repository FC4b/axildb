#!/usr/bin/env bash
#
# scripts/longmemeval-gate.sh — Phase 15 P0.1 CI gate.
#
# Runs the LongMemEval recall harness on the `s` split (small,
# single-session-user questions) and fails on >2% relative regression
# in overall avg_recall or ndcg_at_10 vs the baseline recorded for the same
# configuration (benchmarks/longmemeval/baseline.jsonl for the default
# 500-question recall-qtc run, benchmarks/longmemeval/baselines/<config>.json
# for the rest).
#
# The bench binary writes a JSON BenchmarkReport to stdout — this script
# captures it, optionally promotes it to the baseline (--save), and
# otherwise compares it against the existing baseline via a small
# python helper (python3 is the only non-cargo dependency).
#
# Usage:
#   scripts/longmemeval-gate.sh                       # compare vs baseline
#   scripts/longmemeval-gate.sh --save                # overwrite baseline
#   scripts/longmemeval-gate.sh --rerank              # measure reranker delta (needs --features rerank)
#   scripts/longmemeval-gate.sh --questions 20        # smoke-test mode
#   scripts/longmemeval-gate.sh --strategy recall     # default: recall-qtc
#   scripts/longmemeval-gate.sh --tolerance 0.05      # relax to 5%
#
# Exit codes:
#   0  pass (within tolerance, or skipped because dataset is missing)
#   1  usage / setup error (e.g. bench binary failed)
#   2  no baseline for this configuration (re-run with --save), or the
#      baseline was recorded under a different configuration
#   3  regression beyond tolerance
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/benchmarks/longmemeval"
BASELINE="${HARNESS}/baseline.jsonl"
CANDIDATE_DIR="${HARNESS}/out"
mkdir -p "${CANDIDATE_DIR}"

VARIANT="s"
# Default to the full 500-question Recall-QTC run that backs the README's
# 94.5% figure — the committed baseline (benchmarks/longmemeval/baseline.jsonl)
# is that run, so a real (dataset-present) gate compares apples-to-apples.
# Pass --questions 20 --strategy vector for a quick smoke instead.
QUESTIONS=500
TOP_K=5
MODEL="bge-small"
STRATEGY="recall-qtc"
RERANK="off"
EXTRA_FEATURES=""
SAVE=0
TOLERANCE="0.02"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save)        SAVE=1 ; shift ;;
    --questions)   QUESTIONS="$2" ; shift 2 ;;
    --top-k)       TOP_K="$2" ; shift 2 ;;
    --model)       MODEL="$2" ; shift 2 ;;
    --variant|--split) VARIANT="$2" ; shift 2 ;;
    --strategy)    STRATEGY="$2" ; shift 2 ;;
    --tolerance)   TOLERANCE="$2" ; shift 2 ;;
    --rerank)
      RERANK="cross-encoder"
      EXTRA_FEATURES="--features rerank"
      shift ;;
    -h|--help)
      sed -n '2,28p' "${BASH_SOURCE[0]}" ; exit 0 ;;
    *) echo "unknown flag: $1" ; exit 1 ;;
  esac
done

# The requested split's own file: an `m` run must not proceed because the `s`
# or oracle file happens to be present.
if [[ "${VARIANT}" == "oracle" ]]; then
  DATASET="${HARNESS}/data/longmemeval_oracle.json"
else
  DATASET="${HARNESS}/data/longmemeval_${VARIANT}_cleaned.json"
fi
if [[ ! -f "${DATASET}" ]]; then
  # Degrade honestly: a skip is non-fatal, but it must be loud so a green run is
  # never mistaken for a verified one ("green can mean never ran").
  msg="longmemeval-gate SKIPPED — ${VARIANT} dataset missing at ${DATASET} (see ${HARNESS}/README.md). This gate did NOT run; recall was NOT verified by it."
  echo "⚠️  ${msg}" >&2
  [[ -n "${CI:-}" ]] && echo "::warning title=longmemeval-gate skipped::${msg}"
  exit 0
fi

# One baseline per configuration: a 20-question `recall` smoke compared with
# the 500-question `recall-qtc` run measures the config change, not the code.
# That run keeps its original path, which the docs cite.
N_LABEL="${QUESTIONS}"
[[ "${QUESTIONS}" == "0" ]] && N_LABEL="all"
CONFIG="${VARIANT}-${STRATEGY}-k${TOP_K}-n${N_LABEL}"
[[ "${RERANK}" != "off" ]] && CONFIG="${CONFIG}-rerank-${RERANK}"
if [[ "${CONFIG}" == "s-recall-qtc-k5-n500" ]]; then
  BASELINE="${HARNESS}/baseline.jsonl"
else
  BASELINE="${HARNESS}/baselines/${CONFIG}.json"
fi

CANDIDATE="${CANDIDATE_DIR}/candidate.json"
echo "[gate] variant=${VARIANT} questions=${QUESTIONS} top_k=${TOP_K} model=${MODEL} strategy=${STRATEGY} rerank=${RERANK}"

CARGO_ARGS=(run --release --manifest-path "${HARNESS}/Cargo.toml")
if [[ -n "${EXTRA_FEATURES}" ]]; then
  # shellcheck disable=SC2206
  CARGO_ARGS+=( ${EXTRA_FEATURES} )
fi
CARGO_ARGS+=(
  --
  --variant   "${VARIANT}"
  --limit     "${QUESTIONS}"
  --top-k     "${TOP_K}"
  --model     "${MODEL}"
  --strategy  "${STRATEGY}"
  --rerank    "${RERANK}"
)

# Bench writes a pretty-printed JSON report to stdout; everything else
# is human-progress on stderr.
cargo "${CARGO_ARGS[@]}" > "${CANDIDATE}"

if [[ "${SAVE}" -eq 1 ]]; then
  echo "[gate] saving new baseline → ${BASELINE}"
  mkdir -p "$(dirname "${BASELINE}")"
  cp "${CANDIDATE}" "${BASELINE}"
  exit 0
fi

if [[ ! -f "${BASELINE}" ]]; then
  echo "[gate] no baseline for config ${CONFIG} at ${BASELINE} — re-run with --save to seed one"
  exit 2
fi

python3 - "${BASELINE}" "${CANDIDATE}" "${TOLERANCE}" <<'PY'
import json, sys
base_path, cand_path, tol_arg = sys.argv[1:4]
tolerance = float(tol_arg)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

base = load(base_path)
cand = load(cand_path)

# A baseline recorded under another configuration can't judge this run.
config_keys = ("variant", "strategy", "rerank", "top_k", "total_questions")
mismatch = [k for k in config_keys if base.get(k) != cand.get(k)]
if mismatch:
    for k in mismatch:
        print(f"[gate] config mismatch: {k} baseline={base.get(k)!r} candidate={cand.get(k)!r}", file=sys.stderr)
    print("[gate] FAIL: the baseline was recorded under a different configuration", file=sys.stderr)
    sys.exit(2)

# Gated: session recall, and NDCG@10 (ranking quality) once the baseline
# has it. The rest is printed so a change's effect is visible.
gated = ("avg_recall", "ndcg_at_10")
metrics = (
    "avg_recall", "ndcg_at_10", "hit_rate", "avg_precision", "recall_all",
    "turn_recall_compact", "turn_recall_full", "tokens_compact", "tokens_full",
)
ok = True
print(f"[gate] {'metric':<20} {'baseline':>10} {'candidate':>10} {'delta':>10}")
for m in metrics:
    b = base["overall"].get(m)
    c = cand["overall"].get(m)
    if b is None or c is None:
        shown = "n/a" if b is None else f"{float(b):.4f}"
        print(f"[gate] {m:<20} {shown:>10} {'' if c is None else f'{float(c):.4f}':>10}  (not in both reports)")
        continue
    b, c = float(b), float(c)
    delta = c - b
    rel = (delta / b) if b > 0 else 0.0
    flag = ""
    if m in gated and rel < -tolerance:
        flag = f"  REGRESSION (rel={rel:+.3%} > -{tolerance:.0%})"
        ok = False
    print(f"[gate] {m:<20} {b:>10.4f} {c:>10.4f} {delta:>+10.4f}{flag}")

if not ok:
    print(f"[gate] FAIL: a gated metric regressed beyond {tolerance:.0%} tolerance", file=sys.stderr)
    sys.exit(3)
print("[gate] PASS")
PY
