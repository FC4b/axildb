# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0](https://github.com/FC4b/axildb/compare/axildb-v2.1.2...axildb-v2.2.0) - 2026-07-18

### Added

- *(graph,cli,mcp)* lineage chains with per-hop metric deltas
- *(vector,cli,mcp)* user-supplied vectors — store --vector, similar, named spaces
- *(ql,cli,mcp)* AGG aggregations with group-by
- *(cli,mcp)* where-clause AND expressions + contains + query tool
- *(cli)* store --embed auto-creates a missing vector store

### Fixed

- *(vector,ql,ci)* keep the R&D-loop release a minor bump
- *(core,cli,mcp,clients)* second review round — Opencode + Codex findings
- *(cli,mcp,ql,core)* review fixes — quote guard, parent deltas, typed group keys
- *(cli)* probe vector store before attaching engines in open_with_embedder

### Other

- *(cli,mcp,ql,core,vector,clients)* simplify pass over the R&D-loop features

## [2.1.2](https://github.com/FC4b/axildb/compare/axildb-v2.1.1...axildb-v2.1.2) - 2026-07-14

### Fixed

- *(windows)* survive npm .cmd shims, ONNX DLL mismatch, and scip-python crash

## [2.1.1](https://github.com/FC4b/axildb/compare/axildb-v2.1.0...axildb-v2.1.1) - 2026-07-12

### Other

- *(release)* independent per-crate versioning

## [2.1.0](https://github.com/FC4b/axildb/compare/axildb-v2.0.1...axildb-v2.1.0) - 2026-07-11

### Added

- *(cli)* self-verifying import report for embeddings
- *(cli)* typed cache subcommand so it appears in axil --help
- *(cache)* semantic answer cache extension with code-aware invalidation
- *(cli)* portable memory export/import (JSONL, dedup)
- *(cli)* Wave 3 — OpenCode integration via a local plugin (no npm)
- *(cli)* Wave 2 — Antigravity CLI and Qwen Code hook dialects
- *(cli)* Wave 1 — Codex, Copilot CLI, and Droid hook dialects
- *(cli)* axil mcp install + installer hygiene for the terminal-agent waves
- *(cli)* move the brain hook into the binary + AGENTS.md by default

### Fixed

- *(cli)* hook binary handshake, Antigravity injection wiring, import report surface
- *(cli)* register cache feature in the features CATALOG
- *(cli)* Antigravity works — plugin mechanism + stdin/pipe hang fixes
- *(cli)* codex apply_patch edit capture — patch body is at tool_input.command
- *(cli)* address review of the hook-capture probe
- *(cli)* resolve 15 code-review findings across the terminal-agent waves

### Other

- Merge remote-tracking branch 'origin/main' into dev
- *(cli)* golden-fixture gate for hook dialects + run axildb tests in CI
- *(agents)* terminal-agent integrations + axil hook capture probe

## [2.0.1](https://github.com/FC4b/axildb/compare/axildb-v2.0.0...axildb-v2.0.1) - 2026-06-30

### Fixed

- *(vector)* adaptive HNSW search `ef` so recall@10 holds at scale — the nightly
  large-N oracle (N=20000) caught recall dropping to ~0.78 under a fixed `ef`;
  `ef` now widens with the graph population, restoring recall@10 ≥ 0.90 at 20k
- *(release)* prebuilt archives now ship the self-contained binary (onnxruntime
  is statically linked — there is no sidecar lib to bundle); aarch64-linux builds
  on a native ARM runner; a `workflow_dispatch` tag input can (re)build a tag's
  archives when a release-plz tag push doesn't trigger the workflow

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
