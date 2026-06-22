#!/usr/bin/env bash
# scripts/needle-recall-gate.sh — recall-regression gate (local).
#
# Runs `axil brain-eval`, which now includes a synthetic, dataset-free
# needle-retention retrieval eval: records carrying planted UUIDs / error
# codes / anomalies are inserted among distractors, then recalled by
# natural-language queries. The gate FAILS if any planted needle is not
# returned within top-k, or if a recalled needle token did not survive intact.
#
# No network, no API key, no embedding model (FTS-only) — deterministic and
# offline. Run it manually or as a pre-commit hook. CI wiring is intentionally
# out of scope (the GitHub Actions piece is deferred); this gate is
# the local stand-in that protects the recall path.
#
# Usage:  scripts/needle-recall-gate.sh
# Exit:   0  pass
#         1  setup error (binary missing 'retrieval' section, etc.)
#         3  recall regression (a needle missed, or retention < 90%)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${AXIL_BIN:-}" ]; then
    # Explicit binary — trust it as-is (a prebuilt release, or a test stub).
    axil_bin="$AXIL_BIN"
    if [ ! -x "$axil_bin" ]; then
        echo "[needle-gate] ERROR: AXIL_BIN=$axil_bin is not executable" >&2
        exit 1
    fi
else
    # Default: rebuild so the gate always tests the CURRENT checkout, never a
    # stale prior build (cargo is incremental — a no-op when nothing changed).
    echo "[needle-gate] building axil (release) to test the current checkout…" >&2
    (cd "$ROOT" && cargo build --release -p axildb --quiet)
    axil_bin="$ROOT/target/release/axil"
fi

report="$("$axil_bin" brain-eval)"

REPORT="$report" python3 <<'PY'
import json, os, sys

report = json.loads(os.environ["REPORT"])
r = report.get("retrieval")
if r is None:
    print("[needle-gate] ERROR: brain-eval output has no 'retrieval' section "
          "(binary built without the `fts` feature?)", file=sys.stderr)
    sys.exit(1)

total, recalled, retained = r["total"], r["recalled"], r["retained"]
rk, rr, k = r["recall_at_k"], r["retention_rate"], r["top_k"]
misses = [x for x in r["results"] if not x["recalled"]]
mangled = [x for x in r["results"] if x["recalled"] and not x["retained"]]

print(f"[needle-gate] needles={total}  recall@{k}={rk:.3f} ({recalled}/{total})  "
      f"retention={rr:.3f} ({retained}/{recalled})")
for m in misses:
    print(f"  MISS    {m['name']}: query={m['query']!r} needle={m['needle']!r} "
          f"not in top-{k}", file=sys.stderr)
for m in mangled:
    print(f"  MANGLED {m['name']}: needle {m['needle']!r} lost from recalled record",
          file=sys.stderr)

if rk < 1.0:
    print(f"[needle-gate] FAIL: recall@{k} regressed to {rk:.3f} (expected 1.000) "
          f"— a known record stopped surfacing.", file=sys.stderr)
    sys.exit(3)
if rr < 0.90:
    print(f"[needle-gate] FAIL: needle retention {rr:.3f} (< 0.900).", file=sys.stderr)
    sys.exit(3)

print("[needle-gate] OK — every planted needle recalled intact.")
PY
