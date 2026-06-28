# Turso-Informed Hardening — Task Plan

> **Branch:** `analysis/turso-comparison` — local research artifact, **not pushed**.
> **Companion to:** [analysis/turso-vs-axil.md](turso-vs-axil.md) (the gap analysis this executes).
> **Grounding:** every task below was verified against Axil's **actual current source** (12 agents
> read the cited files and corrected the analysis report's line numbers where they had drifted).
> File:line references are from that pass — re-confirm before editing, but they were accurate at
> time of writing.
> **Candidate phase:** Phase 26 (Turso-informed hardening). Move this file to `tasks/` if you want
> it tracked as a private phase doc; it's kept under `analysis/` so it's reviewable on the branch.

---

## How to read this

Each task is one PR-sized unit: **objective → verified current state → target → ordered subtasks
(file + concrete change) → tests → acceptance → corrections/risks**. The P0 set is do-now and
mostly low/medium effort. P1 is the next wave. P2 is the strategic backlog (milestone-level only —
detailed specs deferred until they're scheduled).

### Sequencing

```
WAVE 1  (P0 — all independent, parallelizable)
  T1  cargo-dist + ONNX-on-Windows fix      [medium]  no deps
  T2  reverse-orphan detector + heal        [medium]  no deps  (land before T6)
  T3  deterministic RRF tie-break + proptest [low]    no deps
  T4  HNSW-vs-brute-force recall oracle      [medium]  no deps  (land before T7)
  T5  document full MCP surface + drift test [low]     no deps

WAVE 2  (P1)
  T6  seeded fault-injection heal test       [low]     pairs with T2 (test-only, no hard dep)
  T7  incremental HNSW                        [high]    after T4 (oracle is the safety net)
  T8  inspect MCP tool                         [low]     soft-couple T5
  T9  MCP setup ergonomics                     [low]     soft-couple T5
  T10 guest SDK + `axil ext new` scaffold      [medium]  no deps
  T11 single-writer / lock-free-reader contract [medium] no deps
  T12 batch vector + dep-docs ingest           [medium]  no deps

WAVE 3  (P2 — strategic backlog, see §P2)
  T13 durable _changelog CDC tape (Atlas keystone)   [high]
  T14 recall_delta on an upgraded audit log           [medium]
  T15 point-in-time-consistent branch create          [low]
  T16 lazy read-time importance decay                  [low]
  T17 opt-in encryption-at-rest (record bodies)        [high]
  T18 AxilQL cargo-fuzz target + contributor guides    [medium]
```

**Recommended first PR:** T3 (lowest effort, pure correctness win) or T5 (lowest effort, pure docs)
to warm up; **highest leverage:** T1 (fixes the broken-on-install flagship feature).

### Global definition of done (every task)

- `cargo check --workspace` clean; relevant `cargo test -p …` green.
- Any perf/quality number surfaced anywhere is measured against a named baseline or a committed
  benchmark (CLAUDE.md numbers-integrity policy).
- No `Phase 26:`/task-id prefixes in code comments (CLAUDE.md convention) — rationale only.
- CLI/MCP parity preserved where a task touches a shared surface.
- A `axil store decisions '{…}'` entry recorded for any A-vs-B choice made during implementation.

---

# WAVE 1 — P0

## T1 · Prebuilt binaries via cargo-dist + cargo-binstall (and fix the ONNX-on-Windows break)

**Effort:** medium · **Deps:** none · **Impact:** high (fixes broken-on-install flagship feature)

**Objective.** Replace the ~3-min from-source `cargo install axildb` (which on Windows yields an
`axil.exe` whose vector/embed engine fails ONNX init) with one-line prebuilt-binary installers that
bundle a known-good onnxruntime, plus `cargo binstall` support and per-PR Windows/macOS CI.

**Verified current state.**
- No `dist-workspace.toml`; no `release.yml`. Release is `release-plz.yml` only → it publishes
  crates and creates **per-crate** tags like `axildb-v1.2.0` (confirmed via `git tag`).
- The publishable CLI crate is `axildb` (binary `axil`) at `crates/adapters/axil-cli/Cargo.toml`;
  its `default` feature list (line 96) **includes `embed`** → `cargo install axildb` builds ONNX by
  default, so the broken path **is** the default. No `[package.metadata.binstall]`.
- **Root cause** (from committed `MEMORY.md` — more precise than the analysis report):
  `crates/engines/axil-vector/Cargo.toml:20` builds `ort = { features = ["download-binaries","std"] }`.
  `download-binaries` drops `onnxruntime.dll` next to the compiled `.exe` in `target/`, but
  `cargo install` copies only the `.exe` to `~/.cargo/bin` — **not** the sibling DLL. The Windows
  loader then finds a stale System32 `onnxruntime.dll` (v1.10, API 10); `ort 2.0.0-rc.10` needs ONNX
  Runtime 1.22 (API 22) → `Failed to initialize ORT API … version [22] is not supported`.
  `ORT_DYLIB_PATH` does **not** help under `download-binaries` (only under `load-dynamic`).
- `cuda`/`directml` features (axil-vector/Cargo.toml:45,49) **already** use `ort/load-dynamic` — so
  load-dynamic is a proven in-codebase pattern.
- A first-run download pattern already exists: `axil-vector/src/download.rs:120` (`download_file`,
  ureq + atomic tmp-rename + SHA256 sidecar) → `~/.axil/models/`.
- `ci.yml` is **all `ubuntu-latest`** — zero Windows/macOS coverage. README:43-48 and
  `docs/src/getting-started/installation.md:6` lead with `cargo install axildb`.

**Target.** `cargo dist` generates a tag-triggered `release.yml` building prebuilt `axil` archives
for x86_64/aarch64 × {pc-windows-msvc, apple-darwin, unknown-linux-gnu}, each **bundling a pinned
onnxruntime shared lib next to the binary**; shell+powershell installers to `~/.axil/bin`;
`cargo binstall axildb` resolves the same archive; `ci.yml` gains one windows + one macOS job that
builds default features and runs a real embed smoke test; README/docs lead with the one-liner.

**Subtasks.**
1. **Decision gate — ONNX fix strategy** (`crates/engines/axil-vector/Cargo.toml`). **Strategy A
   (recommended, low risk):** keep `download-binaries`, have cargo-dist bundle the produced
   `onnxruntime.{dll,dylib,so}` into each archive next to `axil` (loader finds the sibling first) —
   no Cargo change. **Strategy B (cleaner runtime, more code):** switch default `embed`/`rerank` to
   `ort/load-dynamic` + fetch-and-cache a pinned DLL on first run (mirror `download.rs`). Record the
   choice as an `axil store decisions` entry. This spec implements **A** as primary.
2. **`dist-workspace.toml`** (repo root, new). `[workspace] members=["cargo:."]`, `[dist]` with
   `cargo-dist-version` (pin latest stable, e.g. `0.31.0`), `ci=["github"]`,
   `installers=["shell","powershell"]`, the 6 targets, `install-path="~/.axil/bin"` (matches the
   `~/.axil/models/` convention), `install-updater=true`, `github-attestations=true`,
   `precise-builds=true`. Scope the build to the `axildb` package only.
3. **Pin the tag pattern to `axildb-v*`** (`dist-workspace.toml`). release-plz emits per-crate tags;
   configure cargo-dist's tag trigger to `axildb-v*` — **not** bare `v*` (legacy `v0.7.11` won't
   recur). Getting this wrong = the dist job never fires.
4. **Generate `release.yml`** (`.github/workflows/release.yml`). `cargo dist init` + `generate-ci`;
   commit the output. Tag-triggered on `axildb-v*` so it runs **after** release-plz pushes the tag;
   coexists with (does not replace) release-plz.yml. Verify the two don't both create the GitHub
   Release (configure one to create, the other to upload assets).
5. **Bundle onnxruntime into each archive** (`dist-workspace.toml`). Use cargo-dist
   `[dist] include`/post-build hook to copy the `download-binaries`-produced lib from
   `target/<triple>/release/` into the archive root next to `axil`. Confirm exact filename per target
   in the first dry-run. **Load-bearing step** that fixes Windows.
6. *(Strategy B alt)* (`axil-vector/Cargo.toml` **and** `crates/extensions/axil-indexer/Cargo.toml:21`
   — the report **missed** that axil-indexer rerank also uses `download-binaries`). Switch both to
   `load-dynamic`, add `download_onnxruntime()` in `download.rs` → `~/.axil/runtime/`, set
   `ORT_DYLIB_PATH` before `Embedder::new`. Defer unless A proves insufficient.
7. **binstall metadata** (`crates/adapters/axil-cli/Cargo.toml`). Add `[package.metadata.binstall]`
   (or let `cargo dist init` manage it so URLs match the archive layout). Verify
   `cargo binstall axildb --dry-run`.
8. **Per-PR Windows+macOS CI** (`.github/workflows/ci.yml`). New `embed-smoke` job, matrix
   `windows-latest` + `macos-latest`: build `-p axildb` (default features), download a tiny model,
   run a real embed/recall forcing `Embedder::new` → `build_session`; assert exit 0 and **absence of
   `Failed to initialize ORT API`**. This is the regression gate that the broken platform now works.
9. **README** (`README.md:43-48`). Lead with the shell/powershell one-liner + `cargo binstall axildb`;
   demote `cargo install axildb` into a `<details>` from-source subsection noting it needs a C
   toolchain and that prebuilt archives bundle ONNX.
10. **Docs parity** (`docs/src/getting-started/installation.md`). Mirror README; add a "Windows +
    ONNX" note explaining prebuilt archives bundle onnxruntime (no manual DLL copy).

**Tests / gates.** `embed-smoke` CI job (the only Windows test anywhere); `cargo dist plan` PR step
enumerating all 6 triples; post-first-release `cargo binstall --dry-run` + clean-Windows
`axil recall` works with no manual DLL.

**Acceptance.** `dist-workspace.toml` scoped to `axildb`, 6 triples, install-path `~/.axil/bin`,
attestations on · `release.yml` tag-triggered on `axildb-v*`, coexists with release-plz · each
archive contains the onnxruntime lib next to `axil` · `cargo binstall axildb` works · ci.yml has a
windows + macOS embed job that fails on ORT init failure · README/docs lead with the installer · a
fresh Windows install needs no manual DLL copy.

**Corrections / risks.** (1) Report's Touches missed `axil-indexer/Cargo.toml:21` — Strategy B must
change both or rerank stays broken. (2) It's an ORT **API-version init failure**, install-mechanism
specific — not a `build_session` code bug. (3) Tag is `axildb-v*`, not `v*`. (4) `aarch64-pc-windows-msvc`
may lack a prebuilt ort `download-binaries` artifact — validate in dry-run, drop that single triple
if unavailable. (5) Code signing (Turso's Azure step) is **out of scope** for v1 — attestations only;
unsigned Windows binaries show a SmartScreen warning (document it).

---

## T2 · Reverse-orphan detector + heal path (a torn insert must not silently lose a memory)

**Effort:** medium · **Deps:** none (land **before** T6) · **Impact:** high

**Objective.** Detect & auto-heal records that committed to core `.axil` but never got their `.vec`
embedding (and/or `.fts` doc), so a torn/failed insert fan-out can't leave a stored memory
permanently invisible to recall.

**Verified current state.** Axil commits the core record first, then fans out — and **swallows embed
errors silently**. `insert_record` (`db.rs:526`) commits at `:547`, then the auto-embed block
(`:553-571`) only embeds when `!table.starts_with('_')` && `searchable_text` non-empty && `len>5`,
dropping failures (`if let Ok(vec)`). FTS is added in `run_insert_hooks` (`db.rs:889`, gated
`!internal` at `:895`). The doc comment at `:601-613` even states "On hook failure the record IS
persisted." Detection is **one-directional only**: `clean_orphaned_vectors` (`:2208`) /
`clean_orphaned_fts` (`:2224`) sweep index→missing-record; **nothing** sweeps record→missing-index.
`detect_problems_with` (`:2297`) only emits `index_size_mismatch` when the vector/record ratio falls
outside `0.5..=2.0` (`:2337-2351`) — a few torn inserts never trip it. `axil heal --reindex`
(`main.rs:8129-8140`) calls `vector_rebuild()` which **only compacts tombstones** (`:2257`) — it does
**not** re-embed, so doctor recommends a command that can't fix the problem. `doctor()` vector_index
check (`:3099-3110`) and fts check (`:3153-3161`) **always return `Ok`**. Building blocks exist:
`Axil::embed_text` (`:1204`), `VectorIndex::all_ids` (plugin.rs:105 → VectorEngine lib.rs:246),
`FtsEngine::all_indexed_ids` (fts/lib.rs:187), `scan_all_records` (storage.rs:542).

**Target.** `detect_problems` does a real record→index reconciliation (embeddable IDs minus
`all_ids()` → auto-fixable `missing_embeddings`; same for `missing_fts`). New `Axil::reembed_missing()`
regenerates them, wired into `heal_all` **and** `Command::Heal --reindex`. `doctor()` reports a
Warning with fix `axil heal --reindex` when records lack their index entry. `SessionHeal` closes the
gap opportunistically at session start. Zero schema change (embeddings are derived).

**Subtasks** (all in `crates/axil-core/src/db.rs` unless noted).
1. `is_embeddable(&Record)` mirroring the insert gate **exactly** (`!_`-prefix + `searchable_text`
   non-empty + `len>5`) and `is_fts_indexable(&Record)` mirroring the `!internal` gate
   `run_insert_hooks` applies. Keep next to the insert auto-embed block so drift is obvious.
2. `count_missing_embeddings(&[Record])` / `count_missing_fts(&[Record])` taking the
   already-scanned slice (share one scan). HashSet from `all_ids()`; count embeddable records not in
   it. **Only count missing embeddings when `has_vector_index() && embedder.is_some()`** (a manual
   `add_vector` user with no embedder must not be flagged).
3. Emit `missing_embeddings`/`missing_fts` ProblemDetections in `detect_problems_with` (after the
   `index_size_mismatch` block ~`:2351`), `auto_fixable: true` only when an embedder is present.
   Thread the single existing scan through (report() already scans once at `:2442`).
4. `pub fn reembed_missing(&self) -> Result<(usize,usize)>`: scan once, re-embed each missing
   embeddable record via `embed_text` (`:1204`), re-index missing FTS via `fi.on_record_insert`,
   best-effort per record, `audit_heal_action("reembed_missing", …)`.
5. Wire `reembed_missing` into `heal_all` (`:2592`) — emit a `reembed_missing` HealAction when counts>0.
6. (`crates/adapters/axil-cli/src/main.rs`) `Command::Heal --reindex` (`:8129-8140`): call
   `reembed_missing` after `vector_rebuild`; add `"missing_embeddings"|"missing_fts" => reindex` to
   the dry-run dominated match (`:8081-8088`).
7. Fix `doctor()` vector_index (`:3099-3110`) **and** fts (`:3153-3161`) checks to reconcile: Warning
   + `fix=Some("axil heal --reindex")` when records lack their index entry; reuse one scan.
8. (`crates/axil-tests/tests/self_healing.rs`) Add a `FrontWindowEmbedder` mock-vector(+fts) harness
   (copy `intelligent_db.rs:34-72`) + tests: `detect_problems_finds_missing_embedding`,
   `reembed_missing_restores_recall`, `heal_all_reembeds_missing`, `doctor_flags_missing_embedding`.
9. (same file) Library-level regression: producing a missing embedding then exercising the
   `--reindex` code path (vector_rebuild **then** reembed_missing) leaves zero missing.

**Tests.** Extend the `detect_problems_*`/`heal_all_*` cluster in `self_healing.rs:198-399`; runs
under the existing `cargo test` CI gate.

**Acceptance.** `detect_problems()` yields `missing_embeddings`/`missing_fts` for torn fixtures ·
`reembed_missing()` exists, drives counts to 0, restores recall · `heal_all` emits the action ·
`axil heal --reindex` actually fixes it · doctor flips to Warning with the fix · **no** flagging when
no embedder · `cargo test -p axil-tests --test self_healing` green.

**Corrections / risks.** Loss site is the swallowed-error block (`:553-571`), not the commit. CLI is
`crates/adapters/axil-cli/` (not `crates/axil-cli/`). Predicate must match the insert gate **exactly**
(incl. FTS `!internal`) or it false-positives and spams doctor. `all_ids()` clones IDs — heal/doctor
cold paths only, never the insert hot path. Fully additive, no schema/feature change, no MCP parity
concern (no MCP heal tool).

---

## T3 · Deterministic RRF tie-break + ranking-stability property test

**Effort:** low · **Deps:** none · **Impact:** high (recall reproducibility)

**Objective.** Make RRF fusion + final-sort rankings byte-stable by adding a `RecordId` secondary
sort key everywhere ties occur; lock it with proptest.

**Verified current state.** `reciprocal_rank_fusion` (`query.rs:1488`) accumulates into
`HashMap<RecordId,f32>` (`:1493`), collects to a Vec (`:1505` — randomized order), and sorts at
`:1508` with **no secondary key** → equal-RRF records (common with disjoint lists; see existing
`rrf_three_lists_disjoint` at `:1876` where three records all score 1/61) come out
non-deterministically. `apply_sort` (`:1410`) has `order_by` (`:1412`) and `time_sort` (`:1421`)
branches, both stable but tie-breaking on the already-unstable fused order. `RecordId(pub String)`
(`record.rs:8`) derives `Ord`. A **second** `reciprocal_rank_fusion` with the identical bug lives at
`crates/extensions/axil-indexer/src/ask.rs:764` (the report missed it).

**Subtasks.**
1. (`query.rs:1508`) `…partial_cmp(&a.1)…` → append `.then_with(|| a.0.cmp(&b.0))` (ascending
   RecordId on ties). Update the doc comment (`:1477-1487`).
2. (`query.rs:1412`, order_by) chain `.then_with(|| a.id.cmp(&b.id))` — note: closure compares
   `Record`, so use `a.id` **not** `a.0` (the report's literal `a.0` snippet won't compile here).
3. (`query.rs:1421`, time_sort) same `.then_with(|| a.id.cmp(&b.id))`.
4. (`query.rs` test mod, after `:1876`) `rrf_ties_break_by_record_id_ascending` — insert ids in
   descending order, fuse, assert ascending RecordId out.
5. (root `Cargo.toml` `[workspace.dependencies]` ~`:95`) add `proptest = "1"`.
6. (`crates/axil-core/Cargo.toml` `[dev-dependencies]` ~`:32`) `proptest = { workspace = true }`;
   add inline `rrf_ranking_is_deterministic` proptest (private fn → must live in `query.rs`): build a
   tie-heavy corpus from a seed, fuse twice, `prop_assert_eq!` identical orderings.
7. (`crates/axil-tests/Cargo.toml`) add proptest dev-dep.
8. (`crates/axil-tests/tests/ranking_stability.rs`, new) count-consistency proptest over arbitrary
   insert/update/delete/upsert: assert `sum(tables_with_counts) == total_records` and
   `detect_problems()` has no orphan-class problem.
9. (`crates/extensions/axil-indexer/src/ask.rs:784`) fix the duplicate RRF: tie-break by the item's
   string `id`; extend the ask.rs test mod (~`:1319`).

**Tests.** `query.rs` inline (`rrf_ties_break_…`, `rrf_ranking_is_deterministic`);
`ranking_stability.rs` (count consistency); `ask.rs` tie-break unit test.

**Acceptance.** `grep then_with query.rs` shows all three comparators · indexer RRF tie-broken ·
proptests green and non-flaky · default build still compiles (proptest dev-only).

**Corrections / risks.** Two RRF sites, not one. `apply_sort` needs `a.id` not `a.0`.
`scoring.rs:446` (the report's third "ranking site") is a **display-only** top-3 signal-name sort —
does not affect recall order; excluded to avoid scope creep. Tie-break only reorders already-equal
records → recall-quality baselines (LongMemEval/QTC/needle) unaffected; re-baseline only if some
snapshot pins an exact tie order (none observed).

---

## T4 · HNSW-vs-brute-force recall oracle (vector correctness)

**Effort:** medium · **Deps:** none (land **before** T7) · **Impact:** high

**Objective.** Prove the vector index returns the *correct* neighbors (not just fast ones): assert
HNSW/quantized/binary top-k overlaps a brute-force exact oracle above a recall floor, across variant
paths and the deletes-since-rebuild boundary; gate in CI.

**Verified current state.** axil-vector has **no** correctness oracle — every `hnsw.rs` test (mod at
`:283`) checks tiny 2-3 vector cases or top-1 identity, never approximate-recall overlap at scale.
`cosine_sim` is a private module fn (`hnsw.rs:10`, reachable via `use super::*`); `vectors()`
(`:254`), `search()` (`:128`), `search_clean()` (`:138`), `deletes_since_rebuild()` (`:115`),
`rebuild()` (`:258-280`) all exist. Production path = `VectorEngine::search` (`lib.rs:220`) → only
ever `search_clean` after `rebuild_if_needed`. `sqlite-compare` diffs latency only; the needle gate
is **FTS-only** (`axil brain-eval`, no model/HNSW). `ci.yml` `test` job (`:36`) runs
core/graph/fts/timeseries/ql/scip but **not axil-vector**.

**Subtasks.**
1. (`hnsw.rs` test mod, `:283`) inline splitmix64/xorshift PRNG (no `rand` dep — none in workspace),
   `make_vectors(n,dims,seed)` with `RecordId(format!("v{i:08}"))`, `brute_force_topk` via
   `super::cosine_sim` with deterministic tie-break, `recall_overlap`.
2. `hnsw_recall_matches_brute_force`: N=2000, dims=64, add all, `rebuild_if_needed`, M=50 queries,
   mean recall@10 `>= 0.90` (`const RECALL_FLOOR_K10`). Never assert exact equality.
3. `recall_correct_after_deletes_without_rebuild`: index N=1500, force rebuild, `remove` ~20%
   without manual rebuild, `search` (auto-rebuilds): assert (1) no removed id ever appears, (2)
   survivor recall ≥ floor vs post-delete brute force.
4. (`quantize.rs` test mod, `:146`) `two_phase_int8_recall_matches_brute_force` vs
   `axil_core::util::cosine_similarity`, mean recall@k ≥ ~0.80; comment that int8 `two_phase_search`
   is a standalone path **not wired into** `VectorEngine::search`.
5. (`binary.rs` test mod, `:88`) `binary_two_phase_recall_matches_brute_force`, floor ~0.65 (tune to
   measured); comment binary is the lossiest variant.
6. (`.github/workflows/ci.yml` test job `:46-54`) add `-p axil-vector` (default features, no
   embed/ort, model-free, sub-second at N=2000).
7. (`hnsw.rs`) env-scale N/M/floor via `AXIL_ORACLE_N` (default 2000) so the same test scales nightly.
8. (`.github/workflows/nightly-bench.yml`) run `AXIL_ORACLE_N=20000 cargo test -p axil-vector --release recall`.
9. (`scripts/needle-recall-gate.sh`) optional: correct the stale "CI out of scope" header; do **not**
   repurpose it for the vector oracle (it's FTS-only).

**Acceptance.** `cargo test -p axil-vector` (default features) green incl. 4 oracle tests · every
floor a named const/env value with justification · delete test fails if a removed id reappears or
survivor recall drops · fully seeded/deterministic, no model/network · ci.yml runs `-p axil-vector`
per-PR; a forced regression (truncate `search_clean` to top-1) fails the gate · nightly runs N=20000.

**Corrections / risks.** `search_mrl`/`two_phase_search`/`binary_two_phase_search` are
standalone, **unwired** helpers (zero production consumers) — the oracle is a real HNSW oracle on the
live path **plus** overlap tests on dead-but-shipped helpers; `search_mrl` is lowest priority. The
needle gate **cannot** exercise HNSW — wire via the `test` job, which omits axil-vector today.
`cosine_sim` is private (fine in-module; cross-module tests use `axil_core::util::cosine_similarity`).
Pin floors slightly below first observed (HNSW at N~2k/dims=64 is typically ~0.95+). All `#[cfg(test)]`,
no public API or dep change.

---

## T5 · Document the full ~19-tool MCP surface + doc-vs-runtime drift test

**Effort:** low · **Deps:** none (soft-couples T8/T9) · **Impact:** high (the surface *is* the product)

**Objective.** Rewrite `docs/src/agents/mcp.md` to cover all ~19 tools the assembled server actually
exposes (grouped, with params + one-line "when to use"), and add a Rust drift test asserting the doc's
tool list stays in sync with the assembled runtime surface (built-ins + enabled Extension tools).

**Verified current state.** Docs list only the 8 original CRUD tools; the runtime exposes ~19 incl.
the agent-native differentiators `boot`, `code_context`, `remember_decision`, plus extension tools
(`checkpoint`, `dep_docs`). `tool_definitions()` omits the extension tools (those come from each
extension's `mcp_surface()` via `register_builtin_extensions`), so the drift test must enumerate the
**full assembled** set, not just `tool_definitions()`. An existing `SERVER_INSTRUCTIONS` drift guard
(`axil-mcp/src/lib.rs` ~`:481`) is the pattern to mirror.

**Subtasks.**
1. (`docs/src/agents/mcp.md`) rewrite the tool table: full assembled surface grouped by purpose
   (CRUD / intent-native writes / code / boot / extension tools), each with params + a one-line
   "when to use", prioritizing `boot`, `code_context`, `remember_decision`.
2. (`crates/adapters/axil-mcp/src/lib.rs`) add a drift test mirroring the SERVER_INSTRUCTIONS guard:
   assert the doc's tool list == the full assembled set (enumerate via `register_builtin_extensions`
   + each extension's `mcp_surface()`, **not** just `tool_definitions()`).
3. README delegates to mcp.md (single source of truth; no duplication).

**Tests / acceptance.** Drift test fails if a tool is added/removed without a doc update · mcp.md
covers every assembled tool with params + when-to-use · `cargo test -p axil-mcp` green.

**Corrections / risks.** `tool_definitions()` is **not** the full surface — it omits the 4 extension
tools; the drift test must assemble the runtime set or it'll be wrong in the opposite direction.

---

# WAVE 2 — P1

## T6 · Seeded fault-injection test proving heal recovers the fan-out

**Effort:** low · **Deps:** pairs with T2 (test-only, no hard dep) · **Impact:** high

**Objective.** A seeded, deterministic test that synthesizes genuine torn-write orphans
(record gone, index entry remains) and asserts the existing repair path
(`detect_problems → compact/heal_all`) drives the DB back to consistent — Turso's DST idea scoped to
Axil's real durability seam.

**Key shape** (new `crates/axil-tests/tests/fault_injection.rs`): in-test xorshift PRNG (no `rand`
dep); a `full_db()` with vector+graph+fts; per-class orphan tests —
`orphaned_vector_is_detected_and_healed` (`add_vector` then `storage().delete(id)`),
`orphaned_fts_is_detected_and_healed` (the weakest link — `index_text` then torn delete),
`orphaned_edge_is_detected_and_healed` (`relate` then delete target); a `seeded_fanout_workload_heals_clean`
(~200 PRNG ops incl. torn deletes → `heal_all`/`compact` → `detect_problems` empty); a pinned-seed
regression case (minimal "bugbase").

**Acceptance.** `cargo test -p axil-tests --test fault_injection` green (5 fns) · each orphan test
fails if its `clean_orphaned_*` call (`db.rs:2191/2208/2224`) is removed · seed printed on failure ·
**no new Cargo dep** · `detect_problems()` flags `orphaned_edges` (auto_fixable) · recovery idempotent.

**Corrections / risks.** The report's "test-only failpoint in `db.rs` between storage and engine
hooks" is **unnecessary** — `db.storage().delete(id)` (`db.rs:4442` → `storage.rs:159`) removes only
the core record (cascade lives in `db.delete`, not `storage.delete`), so it synthesizes a genuine
orphan with **zero production-code change**. Don't add failpoints to `db.rs`.

---

## T7 · Incremental HNSW — stop full-rebuild on store-then-recall

**Effort:** high · **Deps:** **after T4** (oracle is the safety net) · **Impact:** high

**Objective.** Replace the immutable `instant-distance` `HnswMap` (any add/remove sets `dirty` →
full O(n) clone+rebuild on the next search, under a write lock) with an HNSW supporting incremental
insert + lazy tombstone delete, so the agent's mandated store-then-recall loop pays cost proportional
to what changed.

**Key shape.**
1. **Benchmark FIRST** (`benchmarks/vector-latency/src/main.rs`) — add
   `cold_recall_after_single_store_us` (populate N, rebuild once, loop {add 1; time 1 search}),
   commit a before/after JSON so the win is a committed number (numbers-integrity).
2. (`axil-vector/Cargo.toml`) swap `instant-distance` → `hnsw_rs` (pure Rust, no C++/cmake — the
   onnxruntime.dll Windows lesson; **do not** add usearch). Verify no C build deps via `cargo tree`.
3. (`hnsw.rs`) reimplement `HnswIndex` over `hnsw_rs Hnsw<f32,DistCosine>` (or DistDot on normalized
   vectors); keep `vectors: HashMap` as source of truth; `add` links O(log n) with **no** dirty flag;
   `remove` tombstones.
4. (`hnsw.rs`) `search_clean` over-fetches `top_k + tombstone_count`, maps usize→RecordId, skips
   tombstones, converts distance→similarity as today (`:163-170`).
5. (`hnsw.rs`) `from_vectors` (`:73-84`) inserts each loaded vector into the live graph on open
   (load already O(n)).
6. (`lib.rs:220-232`) read-lock fast path now covers the common case; write-lock only for compaction.
7. (`crates/axil-core/src/worker.rs:87`) add background `compact_vector_index_if_needed` gated by
   `deletes_since_rebuild` ratio vs `vector_rebuild_threshold` — **off the write path**.
8-9. Unit tests: `add_does_not_dirty`, `tombstone_excluded_from_search`, `over_fetch_returns_full_topk`;
   engine-layer **incremental-vs-rebuild top-1 parity** test (correct, not just fast).

**Acceptance.** `needs_rebuild()` false after `add` and search returns the new vector with no rebuild
· removed id never appears pre-compaction · `search(top_k)` returns top_k live results despite
tombstones · incremental top-1 == rebuild top-1 on a fixed corpus · `cargo test -p axil-vector` green
· vector-latency cold-recall p95 at 20k materially below the committed baseline · worker reports
`vectors_compacted` only when ratio exceeds threshold · trait signatures + all callers compile.

**Corrections / risks.** The report implies the worker already does vector compaction — it does
**not** (`worker.rs:87-124` never touches the vector index; `rebuild()` is only `db.rs:2640`). T4's
oracle is the load-bearing safety net for this swap — sequence it first.

---

## T8 · Read-only `inspect` MCP tool (record-type census + light health)

**Effort:** low · **Deps:** soft-couple T5 · **Impact:** medium

**Objective.** One read-only MCP tool answering "what kinds of memory does this brain hold, and is it
healthy?" — per-type counts (`tables_with_counts()`, `_`-prefixed tables rolled into one `_internal`
bucket) + a light health summary reusing `db.doctor()`'s read-only checks — for MCP-only clients that
can't shell out to `axil tables`/`doctor`.

**Key shape** (`crates/adapters/axil-mcp/src/tools.rs`): add the `ToolDefinition` (after `boot`,
~`:294`), `"inspect" => handle_inspect` in `dispatch()` (~`:333`), `handle_inspect` building the
census + health verdict; framed as the memory model not SQL columns; unit test in the `:1049` test
mod; document in mcp.md (overlaps T5).

**Acceptance.** `cargo test -p axil-mcp` green · `dispatch(&db,"inspect",{})` returns non-error JSON
with `record_types` + health · appears in `tools/list` · no `_`-prefixed name leaks · zero writes ·
documented in mcp.md.

**Corrections / risks.** "last write 2d ago" is **not** cheaply available — the only `last_write_at`
is a process-local in-memory metric (`metrics.rs:131/178`, reset every open) → a fresh MCP process
reports `None`. **Omit "last write" from v1** (or derive by scanning `created_at`, extra cost).

---

## T9 · MCP setup ergonomics (one-liner + JSON-RPC smoke test + reconcile client docs)

**Effort:** low · **Deps:** soft-couple T5 · **Impact:** high (first-run friction suppresses adoption)

**Key shape.** Fix the **broken** invocation in `docs/src/agents/mcp.md:8` and the config block
(`:30-39`): the DB is passed via the global `--db` flag, **not** positionally. Add a
`claude mcp add axil -- axil --db ./.axil/memory.axil mcp` one-liner; add a zero-client JSON-RPC smoke
test piping real `initialize` + `tools/list` + `tools/call recall` frames into `axil --db <DB> mcp`
with expected output. Reconcile the README over-promise (`:73`): route Cursor/Windsurf/Codex to the
rules-file path `axil install --cursor/--windsurf/--codex` that actually exists. Make `handle_request`
(`axil-mcp/src/lib.rs:284`) test-seam-able and add a gate driving the exact documented frames.

**Acceptance.** No `axil mcp <DB>` positional form anywhere in docs · `claude mcp add` one-liner
present · copy-paste JSON-RPC smoke test present and works (`serverInfo.name == "axil-…"`) · README no
longer implies per-client MCP snippets · `cargo test -p axil-mcp` green.

**Corrections / risks.** **Load-bearing:** the report's `axil mcp ./.axil/memory.axil` one-liner is
**invalid** — `Command::Mcp` (`main.rs:2354`) has no positional path; DB resolves via the global
`--db` (`require_db`, `main.rs:12545`). The existing docs are also wrong. Every one-liner must be
`axil --db <path> mcp` / `AXIL_DB=<path> axil mcp` / bare `axil mcp`.

---

## T10 · Guest SDK module + `axil ext new` scaffold for WASM plugin authoring

**Effort:** medium · **Deps:** none · **Impact:** high (plugin adoption)

**Key shape.** Make `test-guest/src/sdk.rs` the **one** canonical source of the `Plugin` trait +
`export_plugin!` macro; convert `conformance-guest/src/lib.rs` to consume it (stop the copy
divergence — it currently hand-rolls all 10 raw `Guest` methods). Add `axil ext new <name>
[--caps recall,records.write]` (`ExtCommand::New` behind `#[cfg(feature="wasm-host")]`, `main.rs:2782`;
handler in `run_ext` `:14090` **before** `require_db` so it needs no DB) emitting a buildable detached
guest crate (cdylib, own `[workspace]`, component target pinned to the bundled WIT, a `lib.rs` stub
overriding one high-value hook). **Fix the wasip2→wasip1 lie**: `docs/src/extending/wasm-plugins.md:69-75`
and both guests' `build.sh:3` say wasip2 but the artifact is `wasm32-wasip1`.

**Acceptance.** `axil ext new myplug --caps recall` (under `--features wasm-host`) creates a crate
that builds via `cargo component build --release` and loads via `axil ext install` · conformance-guest
uses `sdk::Plugin` + `export_plugin!` (no direct `bindings::export!`) · exactly ONE copy of the trait
+ macro · `grep wasip2` finds no build instruction pointing at a wasip1 artifact · `cargo check
--workspace` + `--features wasm-host` both pass.

**Corrections / risks.** The report's `new crates/sdk/axil-plugin-sdk` workspace crate is a **trap** —
guests build as **detached** workspaces via cargo-component; a workspace-member SDK crate breaks those
builds. A real published crate is explicitly out of scope (defer). Keep one physical `sdk.rs` and have
conformance-guest reference it.

---

## T11 · Single-writer / lock-free-reader concurrency contract (no shared-WAL coordinator)

**Effort:** medium · **Deps:** none · **Impact:** medium

**Key shape.** Add typed `AxilError::Busy` (`error.rs`, after `Plugin`) + `is_busy()`; map
`redb::DatabaseError::DatabaseAlreadyOpen` to it in the `From` impl (`:57-61`), everything else stays.
Add `Storage::open_read_only` (`storage.rs`, next to `open` `:31`) via `redb::ReadOnlyDatabase::open`
(proven probe at `axil-vector lib.rs:348`); thread a `read_only` flag through `AxilBuilder`
(`db.rs:142`/`open` `:425`/`build` `:308`). Route hot read CLI commands (boot/recall/code-search/fts/
get/list) through a read-only open helper next to `open_with_all_detected` (`main.rs:4682`). Add
bounded retry (3×, ~50-200ms) on the foreground writer. New `crates/axil-tests/tests/concurrency.rs`:
writer-vs-writer returns `is_busy()`; reader-while-writer succeeds. Fix `docs/src/concepts/storage.md:90`
("Single-writer, multiple-reader" → accurate one-RW-process + read-only-readers statement; explicitly
no shared-WAL coordinator).

**Acceptance.** `cargo test -p axil-tests --test concurrency` green (both tests) · `Busy` + `is_busy()`
exist and map correctly · `open_read_only` answers get/list against another process's committed record
· a read CLI command works while a writer holds the lock · storage.md corrected · workspace stays green.

**Corrections / risks.** Reuse the existing `LockGuard`/`O_CREAT|O_EXCL` pattern (scip-refresh.lock) —
**no** new `fs2`/`fd-lock` dep. Explicitly **drop** any `.tshm`-style coordinator. The proven
read-only open lives at `crates/engines/axil-vector/src/lib.rs:348`.

---

## T12 · Batch the vector + dep-docs ingest path (cut per-chunk fsync)

**Effort:** medium · **Deps:** none · **Impact:** medium (boot/scip/deps-refresh latency)

**Key shape.** Add `VectorIndex::add_batch(&[(RecordId,&[f32])])` with a **default loop impl**
(`plugin.rs`, after `add` `:87` — no breaking change), overridden in `VectorEngine` (`lib.rs`, after
`add` `:209-218`) as one `.vec` begin_write/commit (`persist_vectors_batch`) then per-vector in-mem
add. Route the four `db.rs` per-record `vi.add` loops through it: `insert_batch_records` (`:828-832`),
`insert_batch_raw` (`:746-750`), `batch_sync_recall_chunks` (`:5198-5202`),
`sync_recall_chunks_for_record` (`:5241`). Add public `embed_fields_batch` + `index_text_batch`
helpers (near `embed_field` `:1178`). Route dep-docs `ingest_dep_docs` (`:254-275`) and
`ingest_migration_note` (`:458-478`) through them (insert loop → collect (id,content) → one
embed_batch + one index_text_batch). Fix the stale `insert_batch` doc comment (`:631-635`).

**Acceptance.** `cargo test -p axil-vector` green incl. add_batch parity/persistence/empty tests ·
`cargo test -p axil-docs` confirms every chunk is vector- + FTS-retrievable after batched ingest ·
`cargo test -p axil-core` unchanged behavior · `cargo check --workspace` clean (default-impl = no
broken implementors) · each of the 4 loops issues exactly one `add_batch` · bench shows add_batch
beats the per-record loop.

**Corrections / risks.** Crate is `crates/engines/axil-vector/` (CLAUDE.md's `crates/axil-vector/` is
stale). `_dep_docs` is `_`-prefixed so its core insert skips FTS/vector hooks — the dep-docs batching
must call the embed/index helpers explicitly (which the current per-chunk path already does). Only
production `VectorIndex` impl is `VectorEngine` — grep `impl VectorIndex` to confirm before finishing.

---

# WAVE 3 — P2 (strategic backlog, milestone-level)

Detailed grounding deferred until scheduled; see the analysis report §3 P2 for full WHY/WHAT.

- **T13 · Durable opt-in `_changelog` CDC tape** *(high)* — the keystone the closed **Atlas** sync
  product needs and the foundation for cheaper `branch_merge`. Feature-gated, written **inside** the
  redb write txn in `storage.rs` (not post-commit `run_insert_hooks`), keyed by a fresh ULID cursor;
  id-only capture default, before/after opt-in; `changes_since(cursor)`; prove value now by replaying
  the tape in `branch_merge`; self-prune via tiering. Reserve a versioned `_sync_meta` shape so Atlas
  needs no later migration. **No** replication/LSN plumbing (no consumer in-tree).
- **T14 · Pull-based `recall_delta` on an upgraded audit log** *(medium)* — extend `audit_log` into a
  durable, opt-in semantic event log (stop skipping `_`-prefixed tables for a curated allowlist:
  belief-revised, decision-superseded, error-fixed, checkpoint-written), ULID cursor, `agent_id` tag;
  surface as MCP `recall_delta(since_cursor, exclude_agent)` + a `recent_changes` boot block. Does
  **not** relax cross-agent session isolation. Drop the in-process push channel (Atlas owns cross-process).
- **T15 · Point-in-time-consistent `branch create`** *(low — claims-integrity)* — the documented
  "atomic copy" does a blind sequential `fs::copy` while writes may be in flight. Retarget to the live
  `&Axil`, hold a redb read txn across the core copy, quiesce each engine. Stop calling it "atomic"
  until it is. Delete or fix the divergent dead `create_snapshot`/`restore_snapshot`.
- **T16 · Lazy read-time importance decay** *(low)* — compute `effective_importance` at read time in
  `scoring.rs` (mirroring `recency_decay`) instead of the worker's full-DB `run_decay` write sweep;
  deletes one of three worker loops. **Reject** DBSP/incremental-view machinery (no relational views);
  defer the O(n²) consolidate/strengthen dirty-set rewrite until entity counts hit the thousands.
- **T17 · Opt-in encryption-at-rest (record bodies)** *(high)* — off-by-default `encryption` feature,
  value-level AEAD (`chacha20poly1305`, per-record nonce, AAD bound to id+table) around
  `record.to_bytes()`. **Key management is the real work, not the cipher**: ship key-from-env
  (`AXIL_ENC_KEY`) + keyfile first, Argon2-from-passphrase for laptop-unlock. Honest scope: v1
  encrypts **record bodies only** — `.vec` embeddings + `.fts` tokens stay cleartext, so the pitch is
  "encrypted record bodies", not "encrypted memory" (numbers-integrity).
- **T18 · AxilQL cargo-fuzz target + contributor agent-guides** *(medium)* — one cargo-fuzz target for
  `axil_ql::parse` (the one genuinely untrusted byte surface), seeding from `comprehensive.rs`
  adversarial inputs; warn-only nightly. Plus a thin `docs/agent-guides/` that points at existing
  material and **moves** per-task contributor mechanics out of the ~6.3k-token always-loaded root
  `CLAUDE.md` to cut Axil's own dogfooded-agent context cost.

---

## Appendix — provenance

Produced on the `analysis/turso-comparison` branch from a depth-1 clone of `tursodatabase/turso`.
Two multi-agent passes: (1) an 8-dimension gap analysis with adversarial per-gap verification
(→ [turso-vs-axil.md](turso-vs-axil.md)); (2) a 12-item code-grounding pass that read Axil's actual
source per item and corrected the report's line numbers. No Axil source was modified — this is a plan,
not an implementation.
