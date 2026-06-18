#!/usr/bin/env bash
#
# scripts/longmemeval-gate.sh — Phase 15 P0.1 CI gate.
#
# Runs the LongMemEval recall harness on the `s` split (small,
# single-session-user questions) and fails on >2% relative regression
# in overall avg_recall vs the baseline checked in at
# benchmarks/longmemeval/baseline.jsonl.
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
#   scripts/longmemeval-gate.sh --strategy recall     # default: vector
#   scripts/longmemeval-gate.sh --tolerance 0.05      # relax to 5%
#
# Exit codes:
#   0  pass (within tolerance, or skipped because dataset is missing)
#   1  usage / setup error (e.g. bench binary failed)
#   2  no baseline on disk and --save was not passed
#   3  regression beyond tolerance
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/benchmarks/longmemeval"
BASELINE="${HARNESS}/baseline.jsonl"
CANDIDATE_DIR="${HARNESS}/out"
mkdir -p "${CANDIDATE_DIR}"

VARIANT="s"
QUESTIONS=20
TOP_K=5
MODEL="bge-small"
STRATEGY="vector"
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

DATASET_S="${HARNESS}/data/longmemeval_${VARIANT}_cleaned.json"
DATASET_ORACLE="${HARNESS}/data/longmemeval_oracle.json"
if [[ ! -f "${DATASET_S}" && ! -f "${DATASET_ORACLE}" ]]; then
  # Degrade honestly: a skip is non-fatal, but it must be loud so a green run is
  # never mistaken for a verified one ("green can mean never ran").
  msg="longmemeval-gate SKIPPED — dataset missing at ${DATASET_S} (see ${HARNESS}/README.md). This gate did NOT run; recall was NOT verified by it."
  echo "⚠️  ${msg}" >&2
  [[ -n "${CI:-}" ]] && echo "::warning title=longmemeval-gate skipped::${msg}"
  exit 0
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
  cp "${CANDIDATE}" "${BASELINE}"
  exit 0
fi

if [[ ! -f "${BASELINE}" ]]; then
  echo "[gate] no baseline at ${BASELINE} — re-run with --save to seed one"
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

def get(report, key):
    return float(report["overall"][key])

metrics = ("avg_recall", "hit_rate", "avg_precision")
ok = True
print(f"[gate] {'metric':<14} {'baseline':>10} {'candidate':>10} {'delta':>10}")
for m in metrics:
    b = get(base, m)
    c = get(cand, m)
    delta = c - b
    rel = (delta / b) if b > 0 else 0.0
    flag = ""
    # Only avg_recall is the gated metric — others are informational.
    if m == "avg_recall" and rel < -tolerance:
        flag = f"  REGRESSION (rel={rel:+.3%} > -{tolerance:.0%})"
        ok = False
    print(f"[gate] {m:<14} {b:>10.4f} {c:>10.4f} {delta:>+10.4f}{flag}")

if not ok:
    print(f"[gate] FAIL: avg_recall regressed beyond {tolerance:.0%} tolerance", file=sys.stderr)
    sys.exit(3)
print("[gate] PASS")
PY
