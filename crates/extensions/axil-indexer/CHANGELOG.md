# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0](https://github.com/FC4b/axildb/compare/axil-indexer-v1.2.0...axil-indexer-v2.0.0) - 2026-06-30

### Fixed

- *(query)* deterministic RRF tie-break + ranking-stability proptest (Phase 26 T3)

## [1.1.1](https://github.com/FC4b/axildb/compare/axil-indexer-v1.1.0...axil-indexer-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axil-indexer-v1.0.0...axil-indexer-v1.1.0) - 2026-06-23

### Added

- *(code-intel)* Phase 23 CodeGraph gap triage — MCP instructions, boot SCIP nudge, adaptive code-context budget

### Fixed

- *(code-intel)* address /octo:review findings on Phase 23

### Other

- *(release)* centralize internal deps in [workspace.dependencies]; bump 1.0.0 → 1.1.0
- *(numbers-integrity)* correct unsourced/contradictory benchmark claims
- *(code-intel)* route code-graph hint through extension_blocks; dedup budget glue
- recall-quality CI gate, numbers-integrity policy, AGENTS.md drift guard
