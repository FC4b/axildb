# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0](https://github.com/FC4b/axildb/compare/axil-cache-v2.0.1...axil-cache-v2.1.0) - 2026-07-11

### Fixed

- *(cache)* keep TTL overflow handling semver-additive
- *(cache)* overflow-safe TTL, index-root code-ref resolution, MCP parity

### Added

- Semantic answer cache Extension: `cache put` / `cache get` / `cache stats` /
  `cache clear`, plus `cache_put` / `cache_get` MCP tools. Caches
  question → answer pairs and returns a stored answer when a semantically
  similar question recurs. Code-aware invalidation drops an entry on read
  when a referenced code proxy or file has changed; TTL expiry is honored the
  same way.
