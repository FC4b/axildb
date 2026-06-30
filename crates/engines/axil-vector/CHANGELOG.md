# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0](https://github.com/FC4b/axildb/compare/axil-vector-v1.2.0...axil-vector-v2.0.0) - 2026-06-30

### Fixed

- *(vector)* deterministic id tie-break in brute_force search (Phase 26 review #3)
- *(vector)* count re-add tombstones toward compaction (Phase 26 review #1)

### Other

- *(axil-vector)* de-flake incremental HNSW recall parity (Phase 26 T7 follow-up)
- *(ingest)* batch vector + dep-docs ingest to cut per-chunk fsync (Phase 26 T12)
- *(vector)* incremental HNSW backend, stop full-rebuild on store-then-recall (Phase 26 T7)
- *(axil-vector)* brute-force recall oracle for HNSW/int8/binary (Phase 26 T4)

## [1.1.1](https://github.com/FC4b/axildb/compare/axil-vector-v1.1.0...axil-vector-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axil-vector-v1.0.0...axil-vector-v1.1.0) - 2026-06-23

### Other

- *(release)* centralize internal deps in [workspace.dependencies]; bump 1.0.0 → 1.1.0
