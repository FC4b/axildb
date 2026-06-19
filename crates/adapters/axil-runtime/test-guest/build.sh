#!/usr/bin/env bash
# Rebuild the hello-guest WASM component fixture from source.
# Requires: cargo-component + the wasm32-wasip2 target.
set -euo pipefail
cd "$(dirname "$0")"
cargo component build --release
cp target/wasm32-wasip1/release/axil_hello_guest.wasm ../tests/fixtures/hello-guest.component.wasm
echo "fixture updated: crates/adapters/axil-runtime/tests/fixtures/hello-guest.component.wasm"
