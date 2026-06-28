#!/usr/bin/env bash
# Rebuild the conformance-guest WASM component fixture from source.
# Requires: cargo-component (it provisions the wasm32-wasip1 target itself).
set -euo pipefail
cd "$(dirname "$0")"
cargo component build --release
cp target/wasm32-wasip1/release/axil_conformance_guest.wasm \
    ../tests/fixtures/conformance-guest.component.wasm
echo "fixture updated: crates/adapters/axil-runtime/tests/fixtures/conformance-guest.component.wasm"
