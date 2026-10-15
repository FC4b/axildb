#!/usr/bin/env bash
# Embedding-model A/B on LongMemEval: the m5 and s60 arms (see README.md).
#
#   run.sh fetch              download the datasets and the three models, verified
#   run.sh build              build the harness against the pinned Cargo.lock
#   run.sh check              verify data, models, lockfile and pins by checksum
#   run.sh run [--with-gte-m5] check, then run the arms, at most two at once
#
# Written for bash on Linux (Ubuntu under WSL2); macOS works too. Outputs go
# to benchmarks/results/model-ab-pc/.
set -euo pipefail

# Everything runs from the repo root with repo-relative paths, so no home
# directory ends up in a result file. LME_DATA and MODEL_AB_OUT, when set,
# should be repo-relative too.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"
HERE=benchmarks/longmemeval/model-ab
LME_DIR=benchmarks/longmemeval
BIN=$LME_DIR/target/release/longmemeval-bench
DATA=${LME_DATA:-$LME_DIR/data}
OUT=${MODEL_AB_OUT:-benchmarks/results/model-ab-pc}
MODELS=${HOME}/.axil/models
TOOLCHAIN=${MODEL_AB_TOOLCHAIN:-1.92.0}

# LongMemEval cleaned splits, pinned to one dataset revision.
DATASET_REV=98d7416c24c778c2fee6e6f3006e7a073259d48f
DATASET_URL=https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/$DATASET_REV
DATA_FILES=(
  "longmemeval_s_cleaned.json 277383467 d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
  "longmemeval_m_cleaned.json 2737100077 9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f"
)

# The pins in crates/engines/axil-vector/src/download.rs, one line per file:
# model dir, repo, revision, remote path, local name, size, sha256.
# `check` confirms every sha256 here is still in download.rs.
MODEL_FILES=(
  "bge-small-en-v1.5 BAAI/bge-small-en-v1.5 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a onnx/model.onnx model.onnx 133093490 828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
  "bge-small-en-v1.5 BAAI/bge-small-en-v1.5 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a tokenizer.json tokenizer.json 711396 d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
  "granite-embedding-small-english-r2 onnx-community/granite-embedding-small-english-r2-ONNX 1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8 onnx/model.onnx model.onnx 182504 cddb145cd1147ec24a3908b2ca2602b98b20a3d198365cff270b7cb26c98179e"
  "granite-embedding-small-english-r2 onnx-community/granite-embedding-small-english-r2-ONNX 1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8 onnx/model.onnx_data model.onnx_data 194844160 86a3a705d4598615894d89540ea71a3d9bbdb17a315e79edcd5dfc737222834b"
  "granite-embedding-small-english-r2 onnx-community/granite-embedding-small-english-r2-ONNX 1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8 tokenizer.json tokenizer.json 2128614 feeb83348dcb033bc6b9d2e1f7906ca9eb2d122845000c9416d894d7c2927149"
  "gte-modernbert-base Alibaba-NLP/gte-modernbert-base e7f32e3c00f91d699e8c43b53106206bcc72bb22 onnx/model.onnx model.onnx 596392315 947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78"
  "gte-modernbert-base Alibaba-NLP/gte-modernbert-base e7f32e3c00f91d699e8c43b53106206bcc72bb22 tokenizer.json tokenizer.json 3583228 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
)

# label | --model | --variant | --limit | extra harness args
ARM_M5_BGE="m5-bge|bge-small|m|5|"
ARM_M5_GRANITE="m5-granite|granite-small|m|5|"
ARM_M5_GTE="m5-gte|gte-modernbert-base|m|5|"
ARM_S60_BGE="s60-bge|bge-small|s|0|--question-ids $HERE/s60-question-ids.json"
ARM_S60_GRANITE="s60-granite|granite-small|s|0|--question-ids $HERE/s60-question-ids.json"

die() { echo "run.sh: $*" >&2; exit 1; }

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

file_size() { wc -c < "$1" | tr -d ' '; }

# 1-minute load average.
load1() {
  if [ -r /proc/loadavg ]; then cut -d' ' -f1 /proc/loadavg
  else sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}'; fi
}

machine() {
  local cpu cores mem os kernel wsl=no
  if [ -r /proc/cpuinfo ]; then
    cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
    cores=$(nproc)
    mem=$(awk '/MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)
    os=$( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null || uname -s)
    grep -qi microsoft /proc/version && wsl=yes
  else
    cpu=$(sysctl -n machdep.cpu.brand_string)
    cores=$(sysctl -n hw.ncpu)
    mem=$(sysctl -n hw.memsize | awk '{printf "%.1f GiB", $1 / 1073741824}')
    os="macOS $(sw_vers -productVersion)"
  fi
  kernel=$(uname -srm)
  printf 'cpu: %s\ncores: %s\nmemory: %s\nos: %s\nkernel: %s\nwsl: %s\n' \
    "$cpu" "$cores" "$mem" "$os" "$kernel" "$wsl"
}

# Verify one file against its size and sha256; prints why on a mismatch.
verify() {
  local path=$1 size=$2 sum=$3
  [ -f "$path" ] || { echo "missing: $path"; return 1; }
  [ "$(file_size "$path")" = "$size" ] || { echo "size: $path is $(file_size "$path") bytes, pinned $size"; return 1; }
  [ "$(sha256 "$path")" = "$sum" ] || { echo "sha256: $path does not match $sum"; return 1; }
}

fetch_one() {
  local url=$1 dest=$2 size=$3 sum=$4
  if verify "$dest" "$size" "$sum" >/dev/null; then echo "  [ok] $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "  [download] $url"
  curl -fL --retry 3 -o "$dest.part" "$url"
  verify "$dest.part" "$size" "$sum" || { rm -f "$dest.part"; die "download failed verification"; }
  mv "$dest.part" "$dest"
}

cmd_fetch() {
  local line name size sum dir repo rev remote local_name
  for line in "${DATA_FILES[@]}"; do
    read -r name size sum <<< "$line"
    fetch_one "$DATASET_URL/$name" "$DATA/$name" "$size" "$sum"
  done
  for line in "${MODEL_FILES[@]}"; do
    read -r dir repo rev remote local_name size sum <<< "$line"
    fetch_one "https://huggingface.co/$repo/resolve/$rev/$remote" "$MODELS/$dir/$local_name" "$size" "$sum"
  done
}

cmd_build() {
  cp "$HERE/Cargo.lock.pinned" "$LME_DIR/Cargo.lock"
  CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4} cargo "+$TOOLCHAIN" build --release --locked \
    --manifest-path "$LME_DIR/Cargo.toml"
  echo "built $BIN"
  echo "sha256 $(sha256 "$BIN")"
}

cmd_check() {
  local bad=0 line name size sum dir repo rev remote local_name
  echo "commit: $(git rev-parse HEAD)"
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tracked files differ from the commit:"; git status --short --untracked-files=no; bad=1
  fi
  for line in "${DATA_FILES[@]}"; do
    read -r name size sum <<< "$line"
    verify "$DATA/$name" "$size" "$sum" && echo "  [ok] data $name" || bad=1
  done
  for line in "${MODEL_FILES[@]}"; do
    read -r dir repo rev remote local_name size sum <<< "$line"
    verify "$MODELS/$dir/$local_name" "$size" "$sum" && echo "  [ok] model $dir/$local_name" || bad=1
    grep -q "$sum" crates/engines/axil-vector/src/download.rs \
      || { echo "pin drift: $sum is not in download.rs"; bad=1; }
  done
  cmp -s "$HERE/Cargo.lock.pinned" "$LME_DIR/Cargo.lock" \
    && echo "  [ok] Cargo.lock is the pinned one" \
    || { echo "benchmarks/longmemeval/Cargo.lock differs from model-ab/Cargo.lock.pinned (run build)"; bad=1; }
  [ -x "$BIN" ] && echo "  [ok] harness $(sha256 "$BIN")" || { echo "no harness at $BIN (run build)"; bad=1; }
  rustc "+$TOOLCHAIN" --version >/dev/null 2>&1 || echo "warning: toolchain $TOOLCHAIN not installed"
  [ $bad -eq 0 ] || die "check failed"
  echo "check passed"
}

# Run one arm: <label>|<model>|<variant>|<limit>|<extra args>
run_arm() {
  local label model variant limit extra pid rc time_cmd=()
  IFS='|' read -r label model variant limit extra <<< "$1"
  mkdir -p "$OUT"
  [ -s "$OUT/$label.json" ] && { echo "$label: $OUT/$label.json exists, skipped"; return 0; }
  if /usr/bin/time -v true >/dev/null 2>&1; then time_cmd=(/usr/bin/time -v)
  elif /usr/bin/time -l true >/dev/null 2>&1; then time_cmd=(/usr/bin/time -l); fi
  # shellcheck disable=SC2206
  local args=(--variant "$variant" --limit "$limit" --strategy recall-qtc --top-k 5
              --model "$model" --data-dir "$DATA" --jobs 1 $extra)
  {
    echo "label: $label"
    echo "commit: $(git rev-parse HEAD)"
    echo "harness_sha256: $(sha256 "$BIN")"
    echo "lockfile_sha256: $(sha256 "$LME_DIR/Cargo.lock")"
    echo "rustc: $(rustc "+$TOOLCHAIN" --version 2>/dev/null || rustc --version)"
    echo "command: longmemeval-bench ${args[*]}"
    machine
    echo "start: $(date -u +%FT%TZ) load1 $(load1)"
  } > "$OUT/$label.meta"
  : > "$OUT/$label.load"
  echo "$(date +%T) $label: start (load $(load1))"
  "${time_cmd[@]}" "$BIN" "${args[@]}" > "$OUT/$label.json" 2> "$OUT/$label.err" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date +%T) $(load1)" >> "$OUT/$label.load"
    sleep 30
  done
  rc=0; wait "$pid" || rc=$?
  echo "end: $(date -u +%FT%TZ) load1 $(load1) exit $rc" >> "$OUT/$label.meta"
  echo "$(date +%T) $label: exit $rc (load $(load1))"
  [ $rc -eq 0 ] || mv "$OUT/$label.json" "$OUT/$label.failed.json"
}

cmd_run() {
  local with_gte=no
  [ "${1:-}" = "--with-gte-m5" ] && with_gte=yes
  cmd_check
  # Two lanes. The m arms share one lane: each holds the 2.7 GB m file in
  # memory twice over while it loads, so two at once need a lot of RAM.
  (
    run_arm "$ARM_M5_BGE"
    run_arm "$ARM_M5_GRANITE"
    [ $with_gte = yes ] && run_arm "$ARM_M5_GTE"
    true
  ) &
  local lane1=$!
  (
    run_arm "$ARM_S60_BGE"
    run_arm "$ARM_S60_GRANITE"
  ) &
  local lane2=$!
  wait $lane1; wait $lane2
  echo "done; results in $OUT"
}

case "${1:-}" in
  fetch) cmd_fetch ;;
  build) cmd_build ;;
  check) cmd_check ;;
  run) shift; cmd_run "$@" ;;
  *) sed -n '2,10p' "$HERE/run.sh"; exit 2 ;;
esac
