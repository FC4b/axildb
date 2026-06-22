# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0](https://github.com/FC4b/axildb/releases/tag/axil-core-v1.0.0) - 2026-06-22

### Added

- *(adapters)* migrate in-tree cli/mcp/ql to impl Adapter (21.4)
- deny-by-default capability policy for WASM plugins (Phase 22.5 capability layer)
- *(core)* post-build Extension registration — enables runtime WASM plugin loading (Phase 22.6 foundation)
- *(runtime)* host ABI — implement the WIT host imports against Axil (Phase 22.3)
- compact --drop-engine — clean a removed Engine's companion file (Phase 21.6, partial)
- *(cli)* axil extensions list|enable|disable — runtime toggle (Phase 21.5)
- central axil-bundle registry — single Extension registration site (Phase 21.2)
- *(core)* lock 1.0 extensibility SPI (Phase 21.1 + 22.0 foundation)

### Fixed

- *(wasm)* clear remaining review issues + cleanup
- *(wasm)* resolve all 15 code-review findings on the plugin runtime

### Other

- strip phase/task-ID bookkeeping from code comments
- axil-core manifest description Plugin traits -> Engine traits
- rename Tier-1 storage trait Plugin -> Engine
- Initial public release
