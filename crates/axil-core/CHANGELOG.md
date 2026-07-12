# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.1](https://github.com/FC4b/axildb/compare/axil-core-v2.1.0...axil-core-v2.1.1) - 2026-07-12

### Other

- *(release)* independent per-crate versioning

## [2.1.0](https://github.com/FC4b/axildb/compare/axil-core-v2.0.1...axil-core-v2.1.0) - 2026-07-11

### Added

- *(cli)* self-verifying import report for embeddings
- *(cli)* portable memory export/import (JSONL, dedup)

### Fixed

- *(core)* honest import merge semantics

### Other

- Merge remote-tracking branch 'origin/main' into dev

## [2.0.0](https://github.com/FC4b/axildb/compare/axil-core-v1.2.0...axil-core-v2.0.0) - 2026-06-30

### Added

- *(phase-26)* complete deferred T17/T18/T15 follow-ups
- *(encryption)* wire encryption-at-rest end-to-end via process-wide cipher (T17)
- *(core)* opt-in encryption-at-rest for record bodies (Phase 26 T17)
- *(core)* pull-based recall_delta on a durable semantic event log (Phase 26 T14)
- *(core)* durable opt-in _changelog CDC tape (Phase 26 T13)
- *(core)* single-writer / read-only-reader concurrency contract (Phase 26 T11)

### Fixed

- *(review)* address Phase 26 validation review findings
- *(cdc)* monotonic _changelog cursor (Phase 26 P2 review)
- *(branch)* point-in-time-consistent branch create from live handle (Phase 26 T15)
- *(core)* error on short embed batch; clamp boot age; fix compaction doc (Phase 26 review #8/#13/#10)
- *(core)* FTS reverse-orphan detector must mirror the text-presence gate (Phase 26 review #2)
- *(core)* reverse-orphan detector + reembed heal path (Phase 26 T2)
- *(query)* deterministic RRF tie-break + ranking-stability proptest (Phase 26 T3)

### Other

- *(semver)* mark public error enums #[non_exhaustive] at the 2.0 boundary
- *(encryption)* measure encryption-at-rest overhead + document results
- *(boot)* load decay config once per boot, not per table (Phase 26 11)
- *(branch)* document the branch-create consistency model (Phase 26 T15)
- *(core)* compute importance decay lazily at read time (Phase 26 T16)
- *(ingest)* batch vector + dep-docs ingest to cut per-chunk fsync (Phase 26 T12)
- *(vector)* incremental HNSW backend, stop full-rebuild on store-then-recall (Phase 26 T7)

## [1.1.1](https://github.com/FC4b/axildb/compare/axil-core-v1.1.0...axil-core-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axil-core-v1.0.0...axil-core-v1.1.0) - 2026-06-23

### Added

- *(code-intel)* Phase 23 CodeGraph gap triage — MCP instructions, boot SCIP nudge, adaptive code-context budget

### Fixed

- *(code-intel)* address /octo:review findings on Phase 23

### Other

- *(code-intel)* route code-graph hint through extension_blocks; dedup budget glue
- *(brain)* de-flake pipeline_overhead_under_5ms — best-of-N timing
