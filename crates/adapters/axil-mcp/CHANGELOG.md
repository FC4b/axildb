# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0](https://github.com/FC4b/axildb/compare/axil-mcp-v2.1.2...axil-mcp-v2.2.0) - 2026-07-18

### Added

- *(graph,cli,mcp)* lineage chains with per-hop metric deltas
- *(vector,cli,mcp)* user-supplied vectors — store --vector, similar, named spaces
- *(ql,cli,mcp)* AGG aggregations with group-by
- *(cli,mcp)* where-clause AND expressions + contains + query tool

### Fixed

- *(vector,ql,ci)* keep the R&D-loop release a minor bump
- *(core,cli,mcp,clients)* second review round — Opencode + Codex findings
- *(cli,mcp,ql,core)* review fixes — quote guard, parent deltas, typed group keys
- *(mcp)* don't drop pipelined JSON-RPC requests in the stdio serve loop

### Other

- *(cli,mcp,ql,core,vector,clients)* simplify pass over the R&D-loop features

## [2.1.2](https://github.com/FC4b/axildb/compare/axil-mcp-v2.1.1...axil-mcp-v2.1.2) - 2026-07-14

### Other

- updated the following local packages: axil-core, axil-vector, axil-indexer, axil-cache, axil-checkpoint, axil-docs, axil-bundle, axil-graph, axil-fts, axil-timeseries, axil-workspace

## [2.1.1](https://github.com/FC4b/axildb/compare/axil-mcp-v2.1.0...axil-mcp-v2.1.1) - 2026-07-12

### Other

- *(release)* independent per-crate versioning

## [2.1.0](https://github.com/FC4b/axildb/compare/axil-mcp-v2.0.1...axil-mcp-v2.1.0) - 2026-07-11

### Added

- *(cache)* semantic answer cache extension with code-aware invalidation

### Other

- Merge remote-tracking branch 'origin/main' into dev

## [2.0.0](https://github.com/FC4b/axildb/compare/axil-mcp-v1.2.0...axil-mcp-v2.0.0) - 2026-06-30

### Added

- *(core)* pull-based recall_delta on a durable semantic event log (Phase 26 T14)
- *(mcp)* read-only `inspect` tool — record-type census + light health (Phase 26 T8)
- *(recall)* --type taxonomy filter + function-not-topic guidance

### Other

- *(mcp)* claude mcp add one-liner + zero-client JSON-RPC smoke test (Phase 26 T9)
- *(mcp)* document full assembled MCP tool surface + drift guard (Phase 26 T5)

## [1.1.1](https://github.com/FC4b/axildb/compare/axil-mcp-v1.1.0...axil-mcp-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axil-mcp-v1.0.0...axil-mcp-v1.1.0) - 2026-06-23

### Added

- *(code-intel)* Phase 23 CodeGraph gap triage — MCP instructions, boot SCIP nudge, adaptive code-context budget

### Fixed

- *(code-intel)* address /octo:review findings on Phase 23

### Other

- *(release)* centralize internal deps in [workspace.dependencies]; bump 1.0.0 → 1.1.0
- *(code-intel)* route code-graph hint through extension_blocks; dedup budget glue
