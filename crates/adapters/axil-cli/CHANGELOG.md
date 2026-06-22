# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/FC4b/axildb/releases/tag/axildb-v1.0.0) - 2026-06-22

### Added

- *(adapters)* migrate in-tree cli/mcp/ql to impl Adapter (21.4)
- *(runtime)* load-time ABI-version negotiation for WASM plugins (22.8)
- *(runtime)* compiled-module cache for WASM plugins (22.9)
- deny-by-default capability policy for WASM plugins (Phase 22.5 capability layer)
- runtime WASM plugin discovery + axil ext commands (Phase 22.6 complete)
- *(core)* post-build Extension registration — enables runtime WASM plugin loading (Phase 22.6 foundation)
- *(cli)* generic Extension CLI dispatch — zero per-command code (Phase 21.3, core path)
- compact --drop-engine — clean a removed Engine's companion file (Phase 21.6, partial)
- *(cli)* axil extensions list|enable|disable — runtime toggle (Phase 21.5)
- central axil-bundle registry — single Extension registration site (Phase 21.2)

### Fixed

- *(cli)* ExtCommand was double-gated on deps + wasm-host
- *(wasm)* clear remaining review issues + cleanup
- *(wasm)* resolve all 15 code-review findings on the plugin runtime

### Other

- *(release)* prepare workspace for crates.io publishing
- restructure README and rename CLI crate to axildb
- strip phase/task-ID bookkeeping from code comments
- rename Tier-1 storage trait Plugin -> Engine
- Initial public release
