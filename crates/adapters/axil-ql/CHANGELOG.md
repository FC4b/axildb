# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0](https://github.com/FC4b/axildb/compare/axil-ql-v2.2.0...axil-ql-v2.3.0) - 2026-09-24

### Added

- *(core,ql,graph)* surface the two ABI-gated phase-27 follow-ups

### Fixed

- *(ql)* keep Query::Traverse's published shape; honour KNOWN AT on table seeds
- *(ql)* make AGG, GROUP, KNOWN and AT contextual keywords
- *(ql)* fold every matching row in AGG and filtered COUNT

## [2.2.0](https://github.com/FC4b/axildb/compare/axil-ql-v2.1.2...axil-ql-v2.2.0) - 2026-07-18

### Added

- *(ql,cli,mcp)* AGG aggregations with group-by

### Fixed

- *(vector,ql,ci)* keep the R&D-loop release a minor bump
- *(cli,mcp,ql,core)* review fixes — quote guard, parent deltas, typed group keys

### Other

- *(cli,mcp,ql,core,vector,clients)* simplify pass over the R&D-loop features

## [2.1.2](https://github.com/FC4b/axildb/compare/axil-ql-v2.1.1...axil-ql-v2.1.2) - 2026-07-14

### Other

- updated the following local packages: axil-core

## [2.1.1](https://github.com/FC4b/axildb/compare/axil-ql-v2.1.0...axil-ql-v2.1.1) - 2026-07-12

### Other

- *(release)* independent per-crate versioning

## [1.1.1](https://github.com/FC4b/axildb/compare/axil-ql-v1.1.0...axil-ql-v1.1.1) - 2026-06-23

### Other

- release v1.1.0

## [1.1.0](https://github.com/FC4b/axildb/compare/axil-ql-v1.0.0...axil-ql-v1.1.0) - 2026-06-23

### Other

- *(release)* centralize internal deps in [workspace.dependencies]; bump 1.0.0 → 1.1.0
