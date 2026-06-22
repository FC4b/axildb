# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/FC4b/axildb/releases/tag/axil-mcp-v1.0.0) - 2026-06-22

### Added

- *(adapters)* migrate in-tree cli/mcp/ql to impl Adapter (21.4)
- *(core)* post-build Extension registration — enables runtime WASM plugin loading (Phase 22.6 foundation)
- central axil-bundle registry — single Extension registration site (Phase 21.2)

### Fixed

- *(wasm)* clear remaining review issues + cleanup
- *(wasm)* resolve all 15 code-review findings on the plugin runtime

### Other

- *(release)* prepare workspace for crates.io publishing
- strip phase/task-ID bookkeeping from code comments
- rename Tier-1 storage trait Plugin -> Engine
- Initial public release
