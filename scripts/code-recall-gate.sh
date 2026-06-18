#!/usr/bin/env bash
# code-recall-gate.sh — Phase 13b.9 / 13b.12 regression gate.
#
# Indexes a corpus, runs the bench harness, then either saves the run as
# a baseline or compares against a saved baseline. Two corpus modes:
#
#   fixture  — checked-in `tests/fixtures/code-recall/` (deterministic)
#   dogfood  — the Axil repo itself (real-world signal)
#
# Usage:
#   scripts/code-recall-gate.sh                          # compare fixture (default)
#   scripts/code-recall-gate.sh --save                   # save fixture baseline
#   scripts/code-recall-gate.sh --compare                # compare fixture
#   scripts/code-recall-gate.sh --dogfood --save         # save dogfood baseline
#   scripts/code-recall-gate.sh --dogfood --compare      # compare dogfood (CI)
#   scripts/code-recall-gate.sh --all --compare          # both corpora; fail if either does
#
# Regression conditions (enforced by `axil code-recall-bench --regression-gate`):
#   - top-3 symbol/section hit rate decreased for any strategy
#   - mean context tokens for `structural_proxies` grew >10%
#
# Exit codes:
#   0  passed
#   1  setup/usage error
#   3  regression detected (matches `EXIT_BENCH_REGRESSION` in axil-cli)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_SRC="$ROOT/tests/fixtures/code-recall"
FIXTURE_BASELINE="$FIXTURE_SRC/baseline.json"
FIXTURE_CASES="$FIXTURE_SRC/cases.json"
DOGFOOD_BASELINE="$FIXTURE_SRC/dogfood-baseline.json"

# axil_dogfood_cases() is the built-in fixture inside the binary, so
# `--cases` is omitted for the dogfood run.

mode=""
corpus="fixture"
for arg in "$@"; do
    case "$arg" in
        --save|--compare) mode="$arg" ;;
        --fixture)        corpus="fixture" ;;
        --dogfood)        corpus="dogfood" ;;
        --all)            corpus="all" ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *) echo "ERROR: unknown arg $arg" >&2; exit 1 ;;
    esac
done

# Default mode picks based on whether the relevant baseline already exists.
if [ -z "$mode" ]; then
    case "$corpus" in
        fixture)
            mode=$( [ -f "$FIXTURE_BASELINE" ] && echo "--compare" || echo "--save" )
            ;;
        dogfood)
            mode=$( [ -f "$DOGFOOD_BASELINE" ] && echo "--compare" || echo "--save" )
            ;;
        all)
            mode="--compare"
            ;;
    esac
fi

export AXIL_BENCH_COMMIT="${AXIL_BENCH_COMMIT:-$(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)}"

axil_bin="${AXIL_BIN:-$ROOT/target/release/axil}"
if [ ! -x "$axil_bin" ]; then
    echo "Building axil release binary…"
    (cd "$ROOT" && cargo build --release -p axil-cli --quiet)
fi

run_fixture() {
    local m="$1"
    local work
    work=$(mktemp -d)
    trap 'rm -rf "$work"' RETURN

    cp -R "$FIXTURE_SRC"/. "$work/"
    rm -f "$work/baseline.json" "$work/dogfood-baseline.json"

    (
        cd "$work"
        "$axil_bin" install --quiet >/dev/null 2>&1 || true
        "$axil_bin" index . --quiet >/dev/null
    )

    case "$m" in
        --save)
            echo "[fixture] saving baseline to $FIXTURE_BASELINE"
            (cd "$work" && "$axil_bin" code-recall-bench \
                --cases "$FIXTURE_CASES" \
                --bench-format json \
                --save "$FIXTURE_BASELINE" \
                > /dev/null)
            echo "[fixture] OK"
            ;;
        --compare)
            if [ ! -f "$FIXTURE_BASELINE" ]; then
                echo "[fixture] ERROR: no baseline at $FIXTURE_BASELINE — run --fixture --save" >&2
                return 1
            fi
            (cd "$work" && "$axil_bin" code-recall-bench \
                --cases "$FIXTURE_CASES" \
                --bench-format markdown \
                --regression-gate "$FIXTURE_BASELINE")
            ;;
    esac
}

run_dogfood() {
    local m="$1"
    # Index the Axil repo to a scratch DB so the user's
    # `.axil/memory.axil` is left alone. `axil install` is run inside a
    # temp cwd because it ignores `--db` and always writes to ./.axil/ —
    # this both isolates the bench DB and creates the companion FTS /
    # graph / timeseries stores so the gate exercises the plugin path
    # the dogfood numbers were measured against.
    local scratch_dir
    scratch_dir=$(mktemp -d -t axil-dogfood-bench.XXXXXX)
    trap 'rm -rf "$scratch_dir"' RETURN

    (cd "$scratch_dir" && "$axil_bin" install --quiet >/dev/null 2>&1 || true)
    local tmp_db="$scratch_dir/.axil/memory.axil"
    if [ ! -f "$tmp_db" ]; then
        echo "[dogfood] ERROR: install did not create $tmp_db" >&2
        return 1
    fi
    (cd "$ROOT" && "$axil_bin" --db "$tmp_db" index . --quiet >/dev/null)

    case "$m" in
        --save)
            echo "[dogfood] saving baseline to $DOGFOOD_BASELINE"
            "$axil_bin" --db "$tmp_db" code-recall-bench \
                --bench-format json \
                --save "$DOGFOOD_BASELINE" \
                > /dev/null
            echo "[dogfood] OK"
            ;;
        --compare)
            if [ ! -f "$DOGFOOD_BASELINE" ]; then
                echo "[dogfood] ERROR: no baseline at $DOGFOOD_BASELINE — run --dogfood --save" >&2
                return 1
            fi
            "$axil_bin" --db "$tmp_db" code-recall-bench \
                --bench-format markdown \
                --regression-gate "$DOGFOOD_BASELINE"
            ;;
    esac
}

case "$corpus" in
    fixture) run_fixture "$mode" ;;
    dogfood) run_dogfood "$mode" ;;
    all)
        rc=0
        run_fixture "$mode" || rc=$?
        run_dogfood "$mode" || { dr=$?; [ "$dr" -gt "$rc" ] && rc=$dr; }
        exit "$rc"
        ;;
esac
