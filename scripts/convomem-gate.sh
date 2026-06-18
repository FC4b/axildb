#!/usr/bin/env bash
#
# scripts/convomem-gate.sh — Phase 15 P0.4 CI gate.
#
# Runs the ConvoMem recall@K harness on the two categories Axil
# advertises strongest support for — Changing Facts (Phase 5
# superseding) and Abstention (Phase 10 beliefs) — and fails on >2%
# relative regression vs the baseline at
# benchmarks/convomem/baseline-<category>.jsonl.
#
# Skips silently when the dataset isn't present (so CI doesn't break
# on a fresh checkout). Bootstrap instructions in
# benchmarks/convomem/README.md.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/benchmarks/convomem"
mkdir -p "${HARNESS}/out"

DATASET="${HARNESS}/data/convomem.json"
if [[ ! -f "${DATASET}" ]]; then
  # Degrade honestly: a skip is non-fatal, but it must be loud so a green run is
  # never mistaken for a verified one ("green can mean never ran").
  msg="convomem-gate SKIPPED — dataset missing at ${DATASET} (see README). This gate did NOT run; recall was NOT verified by it."
  echo "⚠️  ${msg}" >&2
  [[ -n "${CI:-}" ]] && echo "::warning title=convomem-gate skipped::${msg}"
  exit 0
fi

QUESTIONS=50
TOP_K=5
TOLERANCE="0.02"
SAVE=0
RERANK_FLAGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save)        SAVE=1 ; shift ;;
    --questions)   QUESTIONS="$2" ; shift 2 ;;
    --top-k)       TOP_K="$2" ; shift 2 ;;
    --tolerance)   TOLERANCE="$2" ; shift 2 ;;
    --rerank)      RERANK_FLAGS=(--rerank) ; shift ;;
    *) echo "unknown flag: $1" ; exit 1 ;;
  esac
done

EXTRA_CARGO=()
if [[ "${#RERANK_FLAGS[@]}" -gt 0 ]]; then
  EXTRA_CARGO+=(--features rerank-onnx)
fi

run_category() {
  local CAT="$1"
  local SLUG="$2"
  local CANDIDATE="${HARNESS}/out/candidate-${SLUG}.jsonl"
  local BASELINE="${HARNESS}/baseline-${SLUG}.jsonl"

  cargo run --release --manifest-path "${HARNESS}/Cargo.toml" "${EXTRA_CARGO[@]}" -- \
    --category "${CAT}" \
    --max-questions "${QUESTIONS}" \
    --top-k "${TOP_K}" \
    --out "${CANDIDATE}" \
    ${RERANK_FLAGS[@]:-}

  if [[ "${SAVE}" -eq 1 ]]; then
    cp "${CANDIDATE}" "${BASELINE}"
    echo "[convomem-gate] saved baseline ${BASELINE}"
    return 0
  fi
  if [[ ! -f "${BASELINE}" ]]; then
    echo "[convomem-gate] no baseline for ${CAT} — re-run with --save"
    return 1
  fi
  cargo run --release --manifest-path "${HARNESS}/Cargo.toml" -- compare \
    --baseline "${BASELINE}" \
    --candidate "${CANDIDATE}" \
    --max-regression "${TOLERANCE}"
}

run_category "Changing Facts" "changing-facts"
run_category "Abstention"     "abstention"
