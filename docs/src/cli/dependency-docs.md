# Dependency Docs

Axil can pre-load **version-pinned documentation for your project's
dependencies** into memory, so an agent recalls library docs without
re-reading `node_modules` / crate source or round-tripping the web.

The docs are pinned to the version in your **lockfile** — not "latest" —
and re-ingest when that version changes.

> **Scope:** five ecosystems — Cargo, npm, Python, Go and Java. The
> feature lives behind the `deps` Cargo feature, which is enabled in the
> default `axil` binary and excluded only from minimal `--features core`
> builds.

## How it works

```
manifest + lockfile  ──►  installed deps on disk  ──►  chunk ─► embed ─► FTS
  detect + resolve            extract docs                       │
                                                                 ▼
                                          axil dep-docs "<question>"  (recall)
```

1. **Detect & resolve** — every supported manifest under the project is
   found, and each dependency is pinned to its exact lockfile version.
2. **Extract** — docs are read from the dependency copy already on disk
   (the Cargo registry cache, `node_modules/`, the Python environment,
   the Go module cache, the local Maven repository).
3. **Ingest** — docs are split by section, embedded and full-text
   indexed into the `_dep_docs` table.
4. **Recall** — `axil dep-docs "<question>"` returns the matching,
   version-pinned chunks. No network call at recall time.

## Ecosystems

| Ecosystem | Manifests | Version pinned from | Docs extracted from |
|-----------|-----------|---------------------|---------------------|
| Cargo | `Cargo.toml` | `Cargo.lock` | `~/.cargo/registry/src/` — README + crate `//!` docs |
| npm | `package.json` | `package-lock.json`, `yarn.lock` (v1 + Berry), `pnpm-lock.yaml` | `node_modules/` — README / `package.json` description |
| Python | `requirements.txt`, `pyproject.toml`, `Pipfile` | `uv.lock`, `poetry.lock`, `Pipfile.lock`, pinned `==` | site-packages `*.dist-info/METADATA` |
| Go | `go.mod` | exact versions inline in `go.mod` | the Go module cache (`GOPATH/pkg/mod`) |
| Java | `pom.xml` | resolved versions inline in `pom.xml` | the local Maven repo (`~/.m2`) — `.pom` description |

When a manifest has no lockfile, its dependencies fall back to the
manifest range and are recorded as **unpinned** — surfaced in
`axil deps status` so the agent knows the caveat.

## Commands

### `axil deps list`

List the resolved dependency set — proves resolution before any sync.

```
axil deps list [--path <dir>] [--dev]
```

- `--path` — project root to scan (default: current directory).
- `--dev` — include dev/build dependencies, not just direct ones.

### `axil deps sync`

Extract each dependency's docs from its on-disk copy and ingest them.

```
axil --db <path> deps sync [--path <dir>] [--transitive]
```

Reports `ingested`, `chunks`, `needs_web_fallback` (docs too sparse to
be useful) and `not_installed` (dependency not on disk — run
`cargo build` / `npm install` / the ecosystem's install step first).
`--transitive` additionally ingests the transitive dependencies your
own source imports (see [Transitive dependencies](#transitive-dependencies)).

### `axil deps refresh`

Re-ingest docs for dependencies whose manifest or lockfile changed.

```
axil --db <path> deps refresh [--path <dir>] [--if-stale] [--transitive]
```

`--if-stale` is a fast no-op when nothing changed — cheap enough to run
routinely. The brain hook fires `deps refresh --if-stale` in the
background whenever you edit a manifest or lockfile, so docs stay fresh
without an explicit call.

### `axil deps ingest`

Ingest docs for one dependency from text you supply (Path A) or fetch
them over HTTP (Path B).

```
# Path A — pipe in docs the agent fetched itself
cat README.md | axil --db <path> deps ingest --dep tokio@1.40.0 --ecosystem cargo

# Path B — fetch over HTTP (requires a build with --features web-docs)
axil --db <path> deps ingest --dep react@19.2.0 --ecosystem npm --from-web
```

`--ecosystem` accepts `cargo`, `npm`, `python`, `go` or `java`.

### `axil deps status`

Show the dep-doc memory state: synced dependencies and per-manifest
drift.

```
axil --db <path> deps status [--path <dir>]
```

### `axil dep-docs`

Query the dependency documentation memory.

```
axil --db <path> dep-docs "<library question>" [--dep <name>] [--top-k N] [--include-superseded]
```

Returns doc chunks with the dependency name, exact version, section
breadcrumb and content — ranked by relevance. Docs for superseded or
removed versions are excluded unless `--include-superseded` is passed
(each hit then carries a `superseded` flag).

## Transitive dependencies

By default only **direct** dependencies — the ones your manifest
declares — are ingested. A lockfile's full transitive closure is far
too large to index wholesale.

`deps sync --transitive` / `deps refresh --transitive` add a *smart*
slice of it: a transitive dependency is ingested only when your
project's own source actually imports it. Axil scans your `.rs` /
`.js`-family source for `use` / `import` statements and intersects
that with the lockfile closure — so a transitive dep you reach into
directly gets docs, while the hundreds you never touch do not. Such
deps are stored with `kind: "transitive"`, and the run reports a
`transitive` count.

This is supported for Cargo and npm — the ecosystems whose import
token maps cleanly to a package name.

## Version history

`deps sync` / `deps refresh` track each dependency across version bumps
instead of overwriting it:

- **Bumped** — when a dependency's locked version changes, the new
  version's docs are ingested and the *old* version's chunks are kept,
  flagged `archived`. The old `_deps` row is marked `superseded` and
  linked to its replacement, so an agent can still answer "what changed
  when we bumped X" from memory.
- **Removed** — a dependency dropped from every manifest is marked
  `removed` and its chunks archived (kept, not deleted).
- **Changelog memory** — on a version bump, the dependency's
  `CHANGELOG.md` is read from its on-disk copy and stored as
  `migration`-tagged chunks, so the agent can recall *"what changed
  when we bumped X"*. Each such hit carries `doc_kind: "migration"`.
  Available for Cargo, npm and Go — Python wheels and Maven artifacts
  rarely bundle a changelog.
- **Doc diff** — on a bump, Axil also compares the old and new docs
  section by section and stores a `doc_kind: "doc_diff"` chunk listing
  which sections were added, removed or changed. This is the *observed*
  delta — it catches changes the dependency's authors never wrote into
  their changelog.

Archived chunks are excluded from `dep-docs` by default; pass
`--include-superseded` to see them. Each `deps sync` / `deps refresh`
run reports the dependencies it swept (`removed`), the changelogs it
captured (`migrations`) and how many doc diffs it recorded
(`doc_diffs`).

## Tables

| Table | Holds |
|-------|-------|
| `_dep_manifests` | One row per detected manifest; content hashes for drift detection. |
| `_deps` | One row per resolved dependency — name, version, ecosystem, chunk count. |
| `_dep_docs` | One row per doc chunk — embedded and full-text indexed. |

## Web fallback

Local extraction is the default and needs no network. When a
dependency's on-disk docs are missing or too sparse:

- **Path A** (always available) — the agent fetches docs with its own
  tools and pipes them into `axil deps ingest`. This keeps Axil
  decoupled from any one doc provider.
- **Path B** (opt-in) — the built-in HTTP fetcher, behind the
  default-off `web-docs` Cargo feature. Offline-first is the default
  posture.

## MCP

Non-CLI agents reach the same memory through the MCP tools `dep_docs`
(scoped query) and `deps_status`.
