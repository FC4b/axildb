# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/FC4b/axildb/releases/tag/axil-runtime-v1.0.0) - 2026-06-22

### Added

- *(runtime)* ergonomic guest authoring layer for WASM plugins (22.7)
- *(runtime)* per-call wall-clock timeout for WASM plugins (22.5)
- *(runtime)* load-time ABI-version negotiation for WASM plugins (22.8)
- *(runtime)* compiled-module cache for WASM plugins (22.9)
- deny-by-default capability policy for WASM plugins (Phase 22.5 capability layer)
- runtime WASM plugin discovery + axil ext commands (Phase 22.6 complete)
- *(runtime)* WasmExtension shim — a loaded .wasm is just another dyn Extension (Phase 22.4 complete)
- *(runtime)* end-to-end WASM round-trip — load + instantiate + call a real guest (Phase 22.4)
- *(runtime)* host instantiation wiring — Linker + add_to_linker (Phase 22.4 host side)
- *(runtime)* host ABI — implement the WIT host imports against Axil (Phase 22.3)
- *(runtime)* generate host/guest bindings from the WIT via bindgen (Phase 22.3 foundation)
- *(runtime)* embed Wasmtime behind wasm-host — WASM component host (Phase 22.2)

### Fixed

- *(wasm)* clear remaining review issues + cleanup
- *(wasm)* resolve all 15 code-review findings on the plugin runtime

### Other

- *(release)* prepare workspace for crates.io publishing
- strip phase/task-ID bookkeeping from code comments
- *(runtime)* host-ABI conformance suite for WASM plugins (22.10)
