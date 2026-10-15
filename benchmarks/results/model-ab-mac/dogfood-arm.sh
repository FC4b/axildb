#!/usr/bin/env bash
# Build one model's dogfood snapshot from snap-prefix and run the eval on it.
# usage: dogfood-arm.sh <model-name-for-axil.toml> <label>
set -euo pipefail
MODEL="$1"; LABEL="$2"
# Paths as run on the Mac: WT is the worktree, SRC the main checkout's snapshot.
WT=${WT:?the worktree root}
SRC=${SRC:?<main checkout>/benchmarks/dogfood-recall/data/snap-prefix}
W=$WT/target/agent-scratch/dogfood
AXIL=$WT/target/release/axil
SWAP=$WT/target/release/examples/swap_model
mkdir -p "$W"
FROM="$W/from-$LABEL"; SNAP="$W/snap-$LABEL"
rm -rf "$FROM" "$SNAP"
cp -cR "$SRC" "$FROM"
mkdir -p "$SNAP"
cp -c "$SRC/memory.axil" "$SRC/memory.axil.graph" "$SRC/memory.axil.ts" "$SNAP/"
cp -cR "$SRC/memory.axil.fts" "$SNAP/"
printf '[database]\nembedding_model = "%s"\n' "$MODEL" > "$SNAP/axil.toml"
echo "[$(date +%T)] $LABEL: swap ($(uptime | sed 's/.*load averages: //'))" >&2
"$SWAP" --from "$FROM/memory.axil" --to "$SNAP/memory.axil" --model "$MODEL" > "$W/swap-$LABEL.json"
echo "[$(date +%T)] $LABEL: eval ($(uptime | sed 's/.*load averages: //'))" >&2
python3 "$WT/benchmarks/dogfood-recall/run.py" --axil "$AXIL" --snapshot "$SNAP" --fetch 10 --out "$W/dogfood-$LABEL.json" > /dev/null
echo "[$(date +%T)] $LABEL: done ($(uptime | sed 's/.*load averages: //'))" >&2
