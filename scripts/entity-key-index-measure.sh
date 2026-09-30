#!/usr/bin/env bash
# Measure what the `_entities` key index costs on a real store: the core
# `.axil` file's size, and the CPU cost of the fingerprint every lookup
# computes. Sizes and instruction counts do not depend on machine load, so
# this can run on a busy machine; it reports no wall times.
#
#   scripts/entity-key-index-measure.sh <snapshot-dir> <old-axil> <new-axil> [cycles]
#
# <snapshot-dir> holds memory.axil and its companions (e.g.
# benchmarks/dogfood-recall/data/snap-prefix); it is copied, never written.
# <old-axil> is a binary that predates the index, <new-axil> one that has it.
#
# Sizes: one copy is written only by OLD (the control). The other gets a NEW
# lookup (which must not build the index), a NEW store (which builds it),
# then <cycles> rounds of an OLD store that adds an `_entities` row behind the
# index followed by a NEW store that finds the index stale and repairs it.
# "apparent" is the file length, "allocated_kb" the blocks the filesystem
# holds for it.
#
# Fingerprint: instructions retired (macOS `/usr/bin/time -l`) for 1000
# fingerprints of the copy's raw `table_index["_entities"]` value, minus a
# 0-iteration run, for CRC-32 (what the index uses) against SHA-256 and a
# JSON parse of the same bytes.
set -euo pipefail
SNAP=${1:?snapshot dir}
OLD=${2:?old axil binary}
NEW=${3:?new axil binary}
CYCLES=${4:-5}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/entity-key-index.XXXXXX")
export AXIL_SLOW_QUERY_LOG=0

size() {
    printf '%-44s apparent=%-10s allocated_kb=%s\n' "$1" \
        "$(stat -f %z "$2" 2>/dev/null || stat -c %s "$2")" "$(du -k "$2" | cut -f1)"
}
quiet() { "$@" >/dev/null 2>&1; }

# `cp -c` clones on APFS; elsewhere fall back to a plain copy.
for side in old new; do cp -c -R "$SNAP" "$WORK/$side" 2>/dev/null || cp -R "$SNAP" "$WORK/$side"; done
O=$WORK/old/memory.axil
N=$WORK/new/memory.axil
size "snapshot" "$O"

quiet "$OLD" --db "$O" store decisions '{"summary":"We chose `redis_cache` for SessionCache"}'
for i in $(seq 1 "$CYCLES"); do
    quiet "$OLD" --db "$O" store decisions "{\"summary\":\"\`probe_entity_$i\` carries AuditEvents\"}"
    quiet "$OLD" --db "$O" store decisions "{\"summary\":\"\`probe_entity_$i\` partitions raised\"}"
done
size "old only, after $((2 * CYCLES + 1)) stores" "$O"

quiet "$NEW" --db "$N" fts "connection pool" --limit 3
size "new, after a lookup" "$N"
quiet "$NEW" --db "$N" store decisions '{"summary":"We chose `redis_cache` for SessionCache"}'
size "new, after the first store (builds)" "$N"
for i in $(seq 1 "$CYCLES"); do
    quiet "$OLD" --db "$N" store decisions "{\"summary\":\"\`probe_entity_$i\` carries AuditEvents\"}"
    quiet "$NEW" --db "$N" store decisions "{\"summary\":\"\`probe_entity_$i\` partitions raised\"}"
    size "new, after round $i (old store, new repair)" "$N"
done
for side in old new; do
    dupes=$("$OLD" --db "$WORK/$side/memory.axil" list _entities --limit 10000000 2>/dev/null |
        jq '[.[] | (.data.canonical_id // .data.name) | tostring | select(startswith("probe_entity"))]
            | group_by(.) | map(length) | max')
    echo "$side: most rows sharing one probe entity key = $dupes"
done

if [ "$(uname)" = Darwin ]; then
    mkdir -p "$WORK/probe/src"
    cat >"$WORK/probe/Cargo.toml" <<'EOF'
[package]
name = "fingerprint-probe"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
redb = "=3.1.3"
sha2 = "0.10"
crc32fast = "1"
serde_json = "1"
EOF
    cat >"$WORK/probe/src/main.rs" <<'EOF'
use redb::{ReadableDatabase, TableDefinition};
use sha2::{Digest, Sha256};
const TABLE_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("table_index");
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let db = redb::ReadOnlyDatabase::open(&a[1]).unwrap();
    let n: usize = a[3].parse().unwrap();
    let txn = db.begin_read().unwrap();
    let idx = txn.open_table(TABLE_INDEX).unwrap();
    let bytes = idx.get("_entities").unwrap().unwrap().value().to_vec();
    let mut sink = 0u64;
    for _ in 0..n {
        let b = std::hint::black_box(&bytes);
        sink = sink.wrapping_add(match a[2].as_str() {
            "crc32" => crc32fast::hash(b) as u64,
            "sha256" => Sha256::digest(b)[0] as u64,
            "json" => serde_json::from_slice::<Vec<String>>(b).unwrap().len() as u64,
            _ => panic!("mode"),
        });
    }
    println!("entry_bytes={} sink={sink}", bytes.len());
}
EOF
    (cd "$WORK/probe" && cargo build --release --offline -q)
    for mode in crc32 sha256 json; do
        per=()
        for n in 0 1000; do
            per+=("$(/usr/bin/time -l "$WORK/probe/target/release/fingerprint-probe" "$O" "$mode" "$n" 2>&1 |
                awk '/instructions retired/ {print $1}')")
        done
        echo "$mode: $(( (per[1] - per[0]) / 1000 )) instructions per fingerprint"
    done
    "$WORK/probe/target/release/fingerprint-probe" "$O" crc32 0
fi
echo "work dir: $WORK"
