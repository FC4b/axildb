#!/usr/bin/env bash
# bench-check.sh — Run Criterion benchmarks and detect >5% regressions.
#
# Usage:
#   ./scripts/bench-check.sh              # Run all benchmarks
#   ./scripts/bench-check.sh --save       # Save baseline
#   ./scripts/bench-check.sh --compare    # Compare against saved baseline
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

save_baseline() {
    echo "Saving benchmark baseline to $BASELINE_DIR..."
    rm -rf "$BASELINE_DIR"
    for bench in "${!BENCH_PKGS[@]}"; do
        pkg="${BENCH_PKGS[$bench]}"
        cargo bench -p "$pkg" --bench "$bench" -- --save-baseline main 2>&1 | tail -5
    done
    echo '{"status": "saved", "baseline_dir": "'"$BASELINE_DIR"'"}'
}

compare_baseline() {
    echo "Comparing against baseline (threshold: ${THRESHOLD}%)..."
    local regressions=0
    local failures=0

    for bench in "${!BENCH_PKGS[@]}"; do
        pkg="${BENCH_PKGS[$bench]}"
        echo "--- $bench ($pkg) ---"

        if ! output=$(cargo bench -p "$pkg" --bench "$bench" -- --baseline main 2>&1); then
            echo "ERROR: $bench failed to build or run"
            failures=$((failures + 1))
            continue
        fi
        echo "$output" | tail -20

        # Count lines with "regressed" that exceed threshold
        while IFS= read -r line; do
            if echo "$line" | grep -q "regressed"; then
                pct=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%')
                if [ -n "$pct" ]; then
                    above=$(echo "$pct > $THRESHOLD" | bc -l 2>/dev/null || echo "0")
                    if [ "$above" = "1" ]; then
                        regressions=$((regressions + 1))
                        echo "REGRESSION: $line"
                    fi
                fi
            fi
        done <<< "$output"
    done

    if [ "$failures" -gt 0 ]; then
        echo ""
        echo "FAIL: $failures benchmark suite(s) failed to build or run"
        echo '{"status": "error", "failures": '"$failures"', "regressions": '"$regressions"'}'
        exit 1
    elif [ "$regressions" -gt 0 ]; then
        echo ""
        echo "FAIL: $regressions benchmark(s) regressed by more than ${THRESHOLD}%"
        echo '{"status": "fail", "regressions": '"$regressions"', "threshold_pct": '"$THRESHOLD"'}'
        exit 1
    else
        echo ""
        echo "OK: No regressions above ${THRESHOLD}%"
        echo '{"status": "pass", "regressions": 0, "threshold_pct": '"$THRESHOLD"'}'
        exit 0
    fi
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
    *)          run_all ;;
esac
