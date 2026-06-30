# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0](https://github.com/FC4b/axildb/compare/axildb-v1.2.0...axildb-v2.0.0) - 2026-06-30

### Added

- *(encryption)* wire encryption-at-rest end-to-end via process-wide cipher (T17)
- *(core)* pull-based recall_delta on a durable semantic event log (Phase 26 T14)
- *(cli)* axil ext new scaffold + single-source guest SDK (Phase 26 T10)
- *(core)* single-writer / read-only-reader concurrency contract (Phase 26 T11)
- *(recall)* --type taxonomy filter + function-not-topic guidance

### Fixed

- *(review)* address Phase 26 validation review findings
- *(branch)* point-in-time-consistent branch create from live handle (Phase 26 T15)
- *(cli)* surface the read-only/degraded fallback under writer contention (Phase 26 review #6)
- *(core)* reverse-orphan detector + reembed heal path (Phase 26 T2)

### Other

- *(release)* honest install instructions + kill cargo-dist double-workflow trap (T1)
- *(core)* compute importance decay lazily at read time (Phase 26 T16)
- *(dist)* prebuilt binaries via cargo-dist + cargo-binstall, bundle ONNX, wire CI (Phase 26 T1)

## [1.1.1](https://github.com/FC4b/axildb/compare/axildb-v1.1.0...axildb-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axildb-v1.0.0...axildb-v1.1.0) - 2026-06-23

### Added

- *(code-intel)* Phase 23 CodeGraph gap triage — MCP instructions, boot SCIP nudge, adaptive code-context budget

### Other

- *(release)* centralize internal deps in [workspace.dependencies]; bump 1.0.0 → 1.1.0
- *(code-intel)* route code-graph hint through extension_blocks; dedup budget glue
- recall-quality CI gate, numbers-integrity policy, AGENTS.md drift guard
