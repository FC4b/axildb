#!/usr/bin/env bash
# bench-check.sh — Run Criterion benchmarks and detect >5% regressions.
#
# Usage:
#   ./scripts/bench-check.sh              # Run all benchmarks
#   ./scripts/bench-check.sh --save       # Save baseline
#   ./scripts/bench-check.sh --compare    # Compare against saved baseline (fails on a regression)
#   ./scripts/bench-check.sh --watch      # Compare and save in one run (warns on a regression)
#
# Outputs JSON summary for CI integration.
set -euo pipefail

BASELINE_DIR="${BENCH_BASELINE_DIR:-target/criterion-baseline}"
THRESHOLD="${BENCH_THRESHOLD:-5}" # percent

# Map bench target -> package so cargo can find them from the workspace root.
# These are the per-crate benches under crates/*/benches/ (workspace members).
# The combined `criterion-suite` crate under benchmarks/ is tracked but
# `exclude`d from the workspace — run it via
# `--manifest-path benchmarks/criterion-suite/Cargo.toml` when desired.
declare -A BENCH_PKGS=(
    [core_benchmarks]=axil-core
    [vector_benchmarks]=axil-vector
    [graph_benchmarks]=axil-graph
    [fts_benchmarks]=axil-fts
)

# Criterion's output on stdin; prints "<benchmark> +<pct>%" for each benchmark
# Criterion calls regressed whose middle change estimate is above THRESHOLD.
# Criterion prints the estimate on a `change: [lo mid hi]` line (or, with a
# throughput, on the `time:` line under a bare `change:`) and its verdict,
# "Performance has regressed.", on a line of its own after it.
regressions_in() {
    awk -v threshold="$THRESHOLD" '
        /^(Benchmarking|Found|Warning|Gnuplot|WARNING)/ { next }
        /^[^ \t]/ { name = $1 }
        /change:/ { pending = 1 }
        pending && /\[.*%.*\]/ {
            inner = substr($0, index($0, "[") + 1)
            inner = substr(inner, 1, index(inner, "]") - 1)
            split(inner, est, " ")
            mid = est[2]
            sub(/%$/, "", mid)
            pending = 0
        }
        /Performance has regressed/ {
            if (mid + 0 > threshold + 0) printf "%s +%s%%\n", name, mid + 0
        }
    '
}

save_baseline() {
    echo "Saving benchmark baseline to $BASELINE_DIR..."
    rm -rf "$BASELINE_DIR"
    for bench in "${!BENCH_PKGS[@]}"; do
        pkg="${BENCH_PKGS[$bench]}"
        cargo bench -p "$pkg" --bench "$bench" -- --save-baseline main 2>&1 | tail -5
    done
    echo '{"status": "saved", "baseline_dir": "'"$BASELINE_DIR"'"}'
}

# Run every suite with the Criterion arguments in "$@" and report the
# regressions above THRESHOLD. With `warn`, a regression is a warning and the
# run still passes; otherwise it fails. A suite that fails to build or run
# always fails the run.
run_and_report() {
    local mode=$1
    shift
    local regressions=0
    local failures=0

    for bench in "${!BENCH_PKGS[@]}"; do
        pkg="${BENCH_PKGS[$bench]}"
        echo "--- $bench ($pkg) ---"

        if ! output=$(cargo bench -p "$pkg" --bench "$bench" -- "$@" 2>&1); then
            echo "$output" | tail -20
            echo "ERROR: $bench failed to build or run"
            failures=$((failures + 1))
            continue
        fi
        echo "$output" | tail -20

        while IFS= read -r found; do
            [ -n "$found" ] || continue
            regressions=$((regressions + 1))
            if [ "$mode" = warn ]; then
                echo "::warning::$bench: $found (above ${THRESHOLD}%)"
            else
                echo "REGRESSION: $found"
            fi
        done <<< "$(regressions_in <<< "$output")"
    done

    if [ "$failures" -gt 0 ]; then
        echo ""
        echo "FAIL: $failures benchmark suite(s) failed to build or run"
        echo '{"status": "error", "failures": '"$failures"', "regressions": '"$regressions"'}'
        exit 1
    elif [ "$regressions" -gt 0 ] && [ "$mode" != warn ]; then
        echo ""
        echo "FAIL: $regressions benchmark(s) regressed by more than ${THRESHOLD}%"
        echo '{"status": "fail", "regressions": '"$regressions"', "threshold_pct": '"$THRESHOLD"'}'
        exit 1
    elif [ "$regressions" -gt 0 ]; then
        echo ""
        echo "WARN: $regressions benchmark(s) regressed by more than ${THRESHOLD}%"
        echo '{"status": "warn", "regressions": '"$regressions"', "threshold_pct": '"$THRESHOLD"'}'
        exit 0
    else
        echo ""
        echo "OK: No regressions above ${THRESHOLD}%"
        echo '{"status": "pass", "regressions": 0, "threshold_pct": '"$THRESHOLD"'}'
        exit 0
    fi
}

compare_baseline() {
    echo "Comparing against baseline (threshold: ${THRESHOLD}%)..."
    run_and_report fail --baseline main
}

# One run that does both: `--save-baseline main` compares each benchmark with
# the `main` baseline already saved (when there is one) before replacing it,
# so the comparison costs no second run of the suites.
watch_baseline() {
    echo "Comparing against baseline and saving this run (threshold: ${THRESHOLD}%)..."
    run_and_report warn --save-baseline main
}

run_all() {
    echo "Running all Criterion benchmarks..."
    for bench in "${!BENCH_PKGS[@]}"; do
        pkg="${BENCH_PKGS[$bench]}"
        cargo bench -p "$pkg" --bench "$bench" 2>&1
    done
    echo '{"status": "complete"}'
}

case "${1:-}" in
    --save)     save_baseline ;;
    --compare)  compare_baseline ;;
    --watch)    watch_baseline ;;
    *)          run_all ;;
esac
