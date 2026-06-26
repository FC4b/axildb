#!/usr/bin/env bash
#
# scripts/sqlite-compare-gate.sh — Phase 25 P0.2 CI gate.
#
# Runs the in-tree sqlite-compare harness (Axil HNSW vs SQLite + sqlite-vec
# brute force) on SYNTHETIC vectors — no dataset, no model, fully
# deterministic — and asserts a FLOOR on the vector-search speedup:
#
#     sqlite-vec p50  >=  MIN_SPEEDUP * axil p50
#
# This keeps the README "~173× faster vector search" claim honest by gating
# the *direction* (Axil is dramatically faster) on every PR, without trying to
# reproduce the exact 100k-vector number in CI (that run is minutes-long and
# memory-heavy; it stays a labeled local figure). The gate runs at a reduced n
# for speed and asserts a generous floor so HNSW/runner variance never flakes.
#
# The harness writes a JSON Report to stdout (`--format json`); this script
# captures it, asserts the floor, and optionally promotes it to a committed
# reference baseline (--save) for informational regression context.
#
# Usage:
#   scripts/sqlite-compare-gate.sh                 # gate at n=5000, floor 10x
#   scripts/sqlite-compare-gate.sh --n 10000       # larger n
#   scripts/sqlite-compare-gate.sh --min-speedup 8 # relax the floor
#   scripts/sqlite-compare-gate.sh --save          # write reference baseline
#
# Exit codes:
#   0  pass (speedup floor met)
#   1  usage / setup error (harness failed to build/run)
#   3  speedup below the floor (regression)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/benchmarks/sqlite-compare"
BASELINE="${HARNESS}/baseline.json"
OUT_DIR="${HARNESS}/out"
mkdir -p "${OUT_DIR}"

# n=10000 is the CI sweet spot: the HNSW-vs-brute-force gap is n-dependent
# (~3x at 5k, ~6x at 10k, ~16x at 20k, ~173x at 100k local), so we run a
# moderate n that finishes in ~1 min and assert a 3x floor — comfortably below
# the observed ~6x, leaving headroom for runner variance. The 100k/173x figure
# stays a labeled local run.
N=10000
DIMS=384
QUERIES=500
TOP_K=10
MIN_SPEEDUP=3
SAVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)           N="$2" ; shift 2 ;;
    --dims)        DIMS="$2" ; shift 2 ;;
    --queries)     QUERIES="$2" ; shift 2 ;;
    --top-k)       TOP_K="$2" ; shift 2 ;;
    --min-speedup) MIN_SPEEDUP="$2" ; shift 2 ;;
    --save)        SAVE=1 ; shift ;;
    -h|--help)     sed -n '2,30p' "${BASH_SOURCE[0]}" ; exit 0 ;;
    *) echo "unknown flag: $1" ; exit 1 ;;
  esac
done

CANDIDATE="${OUT_DIR}/candidate.json"
echo "[gate] sqlite-compare n=${N} dims=${DIMS} queries=${QUERIES} top_k=${TOP_K} floor=${MIN_SPEEDUP}x"

# Build + run the excluded harness via its own manifest (rusqlite `bundled`
# compiles SQLite from source; sqlite-vec is a pure-cargo dep — no install).
cargo run --release --manifest-path "${HARNESS}/Cargo.toml" -- \
  --n "${N}" --dims "${DIMS}" --queries "${QUERIES}" --top-k "${TOP_K}" \
  --format json > "${CANDIDATE}"

if [[ "${SAVE}" -eq 1 ]]; then
  echo "[gate] saving reference baseline → ${BASELINE}"
  cp "${CANDIDATE}" "${BASELINE}"
fi

python3 - "${CANDIDATE}" "${MIN_SPEEDUP}" <<'PY'
import json, sys
cand_path, floor_arg = sys.argv[1:3]
floor = float(floor_arg)
with open(cand_path, "r", encoding="utf-8") as f:
    r = json.load(f)

axil_p50 = float(r["axil"]["p50_us"])
sqlite_p50 = float(r["sqlite"]["p50_us"])
if axil_p50 <= 0:
    print("[gate] FAIL: axil p50 is non-positive — harness produced no timing", file=sys.stderr)
    sys.exit(1)

speedup = sqlite_p50 / axil_p50
print(f"[gate] {'metric':<16} {'axil':>12} {'sqlite-vec':>12}")
print(f"[gate] {'search p50 (us)':<16} {axil_p50:>12.1f} {sqlite_p50:>12.1f}")
print(f"[gate] speedup = {speedup:.1f}x  (floor = {floor:.1f}x)")

if speedup < floor:
    print(f"[gate] FAIL: vector-search speedup {speedup:.1f}x is below the {floor:.1f}x floor", file=sys.stderr)
    sys.exit(3)
print(f"[gate] PASS: speedup {speedup:.1f}x >= {floor:.1f}x floor")
PY
