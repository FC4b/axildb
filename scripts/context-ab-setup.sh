#!/usr/bin/env bash
# context-ab-setup.sh — deterministic setup for the real context A/B test.
#
# Clones one public repo ONCE and lays it down identically in two sandboxes:
#
#   experiments/context-ab/without/<name>   — plain checkout, no Axil
#   experiments/context-ab/withdb/<name>    — same checkout, indexed by Axil
#
# Both hold byte-identical source (verified by checksum). Multiple corpora
# coexist (flask, django, …); each setup only resets its own <name> dirs.
#
# Usage:
#   scripts/context-ab-setup.sh                                  # flask (default)
#   NAME=django REPO_URL=https://github.com/django/django.git \
#     KEEP_SUBDIR=django scripts/context-ab-setup.sh             # trim to django/ pkg
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/experiments/context-ab"
NAME="${NAME:-flask}"
REPO_URL="${REPO_URL:-https://github.com/pallets/flask.git}"
REPO_REF="${REPO_REF:-main}"
KEEP_SUBDIR="${KEEP_SUBDIR:-}"        # if set, keep ONLY this top-level dir in each sandbox
axil_bin="${AXIL_BIN:-$ROOT/target/release/axil}"

if [ ! -x "$axil_bin" ]; then
    echo "Building axil release binary…" >&2
    (cd "$ROOT" && cargo build --release -p axil-cli --quiet)
fi

mkdir -p "$EXP/without" "$EXP/withdb" "$EXP/.cache"
echo "Resetting $NAME sandboxes" >&2
rm -rf "$EXP/without/$NAME" "$EXP/withdb/$NAME" "$EXP/.cache/$NAME"

echo "Cloning $REPO_URL@$REPO_REF (shallow)…" >&2
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$EXP/.cache/$NAME" >/dev/null 2>&1 \
    || git clone --depth 1 "$REPO_URL" "$EXP/.cache/$NAME" >/dev/null 2>&1
CLONE_SHA="$(git -C "$EXP/.cache/$NAME" rev-parse HEAD)"

# Lay down byte-identical copies WITHOUT the .git dir.
for side in without withdb; do
    mkdir -p "$EXP/$side/$NAME"
    (cd "$EXP/.cache/$NAME" && tar --exclude=.git -cf - .) | (cd "$EXP/$side/$NAME" && tar -xf -)
    # Optionally trim to a single top-level dir (e.g. the framework package)
    # so the corpus is source-only and indexing stays bounded.
    if [ -n "$KEEP_SUBDIR" ]; then
        find "$EXP/$side/$NAME" -mindepth 1 -maxdepth 1 -not -name "$KEEP_SUBDIR" -exec rm -rf {} +
    fi
done

sum_a="$(cd "$EXP/without/$NAME" && find . -type f -not -path './.axil/*' | sort | xargs shasum 2>/dev/null | shasum | awk '{print $1}')"
sum_b="$(cd "$EXP/withdb/$NAME"  && find . -type f -not -path './.axil/*' | sort | xargs shasum 2>/dev/null | shasum | awk '{print $1}')"
if [ "$sum_a" != "$sum_b" ]; then
    echo "ERROR: sandboxes differ ($sum_a != $sum_b)" >&2
    exit 1
fi
echo "Sandboxes identical (checksum $sum_a)" >&2

echo "Indexing withdb/$NAME with Axil…" >&2
(cd "$EXP/withdb/$NAME" && "$axil_bin" install --quiet >/dev/null 2>&1 || true)
(cd "$EXP/withdb/$NAME" && "$axil_bin" index . --quiet >/dev/null)

py_files="$(find "$EXP/without/$NAME" -name '*.py' 2>/dev/null | wc -l | tr -d ' ')"
echo >&2
echo "Setup complete:" >&2
echo "  repo:    $NAME @ $CLONE_SHA" >&2
echo "  without: $EXP/without/$NAME  (no index)" >&2
echo "  withdb:  $EXP/withdb/$NAME   (indexed)" >&2
echo "  .py files: $py_files" >&2
echo "$CLONE_SHA" > "$EXP/.clone-sha-$NAME"
