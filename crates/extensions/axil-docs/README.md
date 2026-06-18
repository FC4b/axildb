# axil-docs

Dependency documentation memory for Axil — **Phase 16**.

> **This crate is the reference [Extension](../../docs/src/extending/extensions.md) (Tier 2) in Axil's three-tier extensibility model.** If you're authoring a new Extension, read [`docs/src/extending/extensions.md`](../../docs/src/extending/extensions.md) for the contract this crate implements, then use `axil-docs`'s modules (`manifest.rs`, `resolve.rs`, `local.rs`, `ingest.rs`, `refresh.rs`, `query.rs`) as the canonical pattern.

Pre-loads version-pinned documentation for a project's dependencies into
Axil memory, so an agent can recall library docs without re-reading
`node_modules` / crate source or round-tripping the web. Context7's
idea, Axil's substrate: the docs are pinned to the *lockfile* version
and re-ingest when that version changes.

**Scope:** five ecosystems — Cargo, npm, Python, Go and Java.

This crate is wired behind the `deps` Cargo feature of `axil-cli`
(in `default` and `full`; absent from minimal `core` builds).

## Pipeline

```
manifest + lockfile          installed deps on disk          web (optional)
Cargo / npm / Python /       registry caches, node_modules,  registry.npmjs.org
Go / Java manifests          site-packages, module cache, ~/.m2
       │                            │                              │
   manifest.rs ── resolve.rs ──► local.rs ──────────► web.rs (Path B, `web-docs`)
   exact versions                  │                              │
       │                           └──────────┬───────────────────┘
       │                                      ▼
       │                                 ingest.rs
       │                split_doc_sections → embed → FTS → _dep_docs rows
       └────────────► _dep_manifests (drift)          query.rs ──► axil dep-docs
```

## Ecosystems

| Ecosystem | Manifests | Version pinned from |
|---|---|---|
| Cargo | `Cargo.toml` | `Cargo.lock` |
| npm | `package.json` | `package-lock.json`, `yarn.lock` (v1 + Berry), `pnpm-lock.yaml` |
| Python | `requirements.txt`, `pyproject.toml`, `Pipfile` | `uv.lock`, `poetry.lock`, `Pipfile.lock`, pinned `==` |
| Go | `go.mod` | exact versions inline in `go.mod` |
| Java | `pom.xml` | resolved versions inline in `pom.xml` |

## CLI

```
axil deps list   [--path <dir>] [--dev]    # resolved dependency set
axil deps sync   [--path <dir>]            # extract local docs → memory
axil deps refresh [--path <dir>] [--if-stale]   # re-ingest changed deps
axil deps ingest --dep <name@version> [--from-web]   # Path A / Path B
axil deps status [--path <dir>]            # synced deps + manifest drift
axil dep-docs    "<library question>" [--dep <name>] [--top-k N] [--include-superseded]
```

MCP parity: the `dep_docs` and `deps_status` tools.

## Tables

| Table | One row per | Key fields |
|---|---|---|
| `_dep_manifests` | detected manifest | `path`, `ecosystem`, `manifest_hash`, `lockfile_hash` |
| `_deps` | resolved dependency | `name`, `version`, `ecosystem`, `kind`, `doc_chunks` |
| `_dep_docs` | doc chunk | `dep_name`, `dep_version`, `section_path`, `content`, embedding |

## Modules

- `manifest.rs` — detect + parse manifests for all five ecosystems.
- `resolve.rs` — pin each dependency to its exact lockfile version.
- `local.rs` — extract docs from the dependency copy on disk.
- `ingest.rs` — chunk → embed → FTS → `_dep_docs` (idempotent).
- `refresh.rs` — content-hash drift detection.
- `query.rs` — scoped recall over `_dep_docs`.
- `web.rs` — Path B HTTP fetcher (behind the default-off `web-docs`
  feature; offline-first is the default posture).
```
