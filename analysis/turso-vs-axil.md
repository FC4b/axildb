# Turso → Axil: Gap Analysis & Improvement Opportunities

> **Branch:** `analysis/turso-comparison` — local research artifact, **not pushed to remote**.
> **Date:** 2026-06-27.
> **Source:** depth-1 clone of [`tursodatabase/turso`](https://github.com/tursodatabase/turso),
> analyzed across 8 dimensions by a multi-agent workflow (35 agents, ~2.4M tokens). Every claimed
> gap was **adversarially re-verified against Axil's actual source** (`axil recall` / `code-search`
> + file reads) so this report does **not** recommend things Axil already has.

---

## TL;DR

Turso is a mature, MIT-licensed, SQLite-**compatible** SQL database rewrite in Rust. Axil is a
cognitive **memory layer for AI agents**. The missions differ, so most of Turso's machinery
(SQL/VDBE, MVCC `BEGIN CONCURRENT`, DBSP incremental views, virtual tables, 9 language bindings,
TLA+/Jepsen linearizability) **does not transfer**. But a narrow, high-value subset does, and Turso
executes it at a level Axil can learn from:

1. **Distribution (P0, highest leverage).** Turso ships prebuilt binaries via **`cargo-dist`** for 6
   target triples with shell/PowerShell installers, a self-updater, GitHub build attestations, and
   code signing. This is the exact fix for Axil's known #1 pain: `cargo install axildb` forces a
   ~3-min compile **and** produces a Windows binary whose vector engine panics (no `onnxruntime.dll`).
2. **Reliability (P0/P1).** Turso's crown jewel is **deterministic simulation testing** with fault
   injection, oracles, and shrinking. Axil has zero fault-injection coverage of its **multi-engine
   write fan-out** — the one place `redb` does *not* protect it (a crash after the core commit but
   before `.vec`/`.fts` persist = a memory the agent stored but can never recall).
3. **Recall correctness (P0).** Axil proves vector search is *fast* but never proves it's *correct*
   (no HNSW-vs-brute-force oracle), and RRF fusion has a non-deterministic tie-break — for a product
   whose entire output **is** the ranking, both are real bugs.
4. **Agent surface (P0/P1, cheap).** Axil's MCP server exposes ~19 cognition-shaped tools but the
   docs list only 8 — the agent-native differentiators (`boot`, `code_context`, `remember_decision`)
   are invisible to anyone wiring Axil up.

The expensive strategic bets (durable `_changelog`/CDC tape for Atlas, incremental HNSW,
encryption-at-rest) are real but high-effort; sequence them after the cheap wins.

**Verdict ledger:** 26 candidate gaps → **13 confirmed real, 13 partial, 0 already-covered/not-applicable.**

---

## 1. Why the comparison is asymmetric

| | **Turso** | **Axil** |
|---|---|---|
| Mission | SQLite-compatible SQL database | Cognitive memory for AI agents |
| Maturity | BETA, production-leaning, funded | Feature-complete core, niche |
| License | MIT | PolyForm Noncommercial |
| Storage | From-scratch pager / WAL / MVCC | `redb` (ACID) core + companion files per engine |
| Correctness target | "match SQLite's bytes" | "did the right memory get recalled" |
| Distribution | 8+ language bindings, prebuilt binaries | CLI + MCP + embedded Rust lib + WASM plugins |
| Reliability story | DST + Antithesis + TLA+ + differential + fuzz | example-based unit/integration tests + recall-quality gates |
| LLM required | No | No |

**The transfer surface is narrow but high-value:** reliability of a *single-file embedded store*,
vector recall *correctness & speed*, *install/distribution UX*, *agent-facing surface*, a *sync
foundation*, and *plugin ergonomics*. Everything Turso built for SQL compatibility is correctly
absent from Axil and should stay that way.

---

## 2. Where Axil already leads — do **not** chase these

Verified during the analysis; listed so this report doesn't push Axil toward non-transferable
SQL-database work:

1. **Extension safety.** Axil's WASM plugin path (wasmtime Component Model, deny-by-default
   capability grants, table-prefix write jail, fuel + epoch budgets, trap-poisoning, secret
   redaction, ABI negotiation) is *decisively safer* than Turso's **unsandboxed native `cdylib`**
   extensions running in full process trust. Copying Turso's loadable-native-extension model would
   be a **safety regression**. Don't port virtual tables — Axil's deny-all sandbox means a guest
   can't even read an external file, so the vtab "external data source" pattern doesn't apply.
2. **Agent-native MCP surface.** Axil exposes ~19 cognition-shaped tools (`recall`,
   `remember_decision`, `boot`, `code_context`, `checkpoint`) with auto-embed, auto-supersede,
   idempotent `(agent_id, external_id)` dedup, token budgeting, a 16 MB DoS cap, and test-enforced
   CLI/MCP parity. Turso exposes raw SQL CRUD with none of this. **The gap is documentation, not
   capability.**
3. **Domain-specific quality gating.** Needle-retention recall gate, committed 500-Q
   LongMemEval/QTC baselines, A/B framework, self-healing (`compact`/`heal`/`detect_problems` with
   tests), and a `doctor` surface — Turso has no analog.
4. **Storage simplicity.** Leaning on `redb`'s battle-tested COW B-tree per companion is *more
   robust* than Turso's from-scratch pager/WAL/MVCC (still BETA, with ignored FIXME concurrency
   tests). Axil's "index is a derived, rebuildable artifact" design is a genuine durability strength.
5. **Heavy-asset distribution.** Lazy, cached model download to `~/.axil/models/` is the correct
   pattern; Axil already solved heavy-blob distribution that Turso never faces.

**Explicitly ignore from Turso:** the DBSP/Z-set incremental-view engine (no relational views in
Axil), MVCC `BEGIN CONCURRENT`, the `.tshm` shared-WAL coordinator (agent memory has one logical
writer), most of the 9 language bindings (they serve SQL drivers Axil will never have), TLA+/Elle
linearizability (wrong altitude for short-lived CLI processes), and SQL-shaped differential oracles
like SQLancer/SQLRight.

---

## 3. Prioritized recommendations

Impact = value to AI-agent users. Effort is rough. "Touches" lists the Axil crates/files involved.

### P0 — do now (high leverage, mostly low/medium effort)

| # | Recommendation | Theme | Impact | Effort |
|---|---|---|---|---|
| 1 | Prebuilt binaries via cargo-dist + binstall (+ fix ONNX-on-Windows) | Distribution | High | Medium |
| 2 | Reverse-orphan detector + heal path (torn insert must not lose a memory) | Reliability | High | Medium |
| 3 | Deterministic RRF tie-break + ranking-stability property test | Recall correctness | High | Low |
| 4 | HNSW-vs-brute-force recall oracle | Vector correctness | High | Medium |
| 5 | Document the full ~19-tool MCP surface (only 8 documented) | Agent surface | High | Low |

**1. Ship prebuilt CLI binaries via cargo-dist + cargo-binstall (and fix the ONNX-on-Windows break).**
Today every user pays a ~3-min C-toolchain compile, and on Windows `cargo install axildb` yields an
`axil.exe` whose vector engine **panics at runtime** (no usable `onnxruntime.dll`) — the flagship
"no LLM, local embeddings" value prop is broken-on-install on the most common dev platform.
*What:* add `dist-workspace.toml` (cargo-dist) targeting x86_64/aarch64 for `pc-windows-msvc`,
`apple-darwin`, `unknown-linux-gnu`, with shell+powershell installers and github-attestations,
tag-triggered after the existing release-plz tag. Fix ONNX in the same release job — either place the
correct onnxruntime shared lib next to the binary in each archive, **or** (cleaner) switch the
default `embed` feature to `ort`/`load-dynamic` and fetch+cache a pinned DLL on first run (mirroring
the existing model download). Add `[package.metadata.binstall]`. Add one windows-msvc + one macOS job
to `ci.yml` so the broken platform is exercised per-PR. Demote `cargo install` to a from-source
subsection in the README; lead with the one-liner installer.
*Touches:* new `dist-workspace.toml`; `.github/workflows/` (release-plz integration, new win/mac
CI jobs); `crates/engines/axil-vector/Cargo.toml` (ort features) + `src/download.rs` (DLL fetch);
`crates/adapters/axil-cli/Cargo.toml` (binstall metadata); `README.md`.

**2. Reverse-orphan detector + heal path.** Axil commits the core `.axil` record **first**, then
fans out to `.vec`/`.fts`/`.graph` as separate transactions. A crash in between leaves a stored
memory permanently invisible to semantic recall — the worst failure mode for a memory system. Axil
only sweeps the *other* direction (index entries whose record is gone).
*What:* in `detect_problems_with`, after one `scan_all_records`, compute *embeddable record IDs minus
`vector_index.all_ids()`* → emit auto-fixable `missing_embeddings`; same vs FTS for `missing_fts`.
Wire a `reembed_missing()` path into `heal_all` and `Command::Heal --reindex` (today `--reindex` only
compacts tombstones and would **not** fix this — a bug, since `doctor` recommends exactly that
command). Fix the `vector_index` doctor check to do real record-vs-vector reconciliation instead of
always reporting `Ok`. Add an opportunistic check at boot/session-start. Embeddings are
deterministically regenerable, so the detector closes the silent-loss hole for ~zero cost.
*Touches:* `crates/axil-core/src/db.rs` (`detect_problems`, `clean_orphaned_*`, `heal_all`, vector
doctor check ~3099, insert fan-out ~547); `crates/engines/axil-vector/src/lib.rs`; CLI `Heal`.

**3. Deterministic RRF tie-break + property test.** `reciprocal_rank_fusion` collects scores into a
`HashMap` (randomized iteration order) and sorts with no secondary key, so on **tied** RRF scores
(common with disjoint result lists) the final ranking is **not byte-stable across runs**. For an
agent-memory product whose output *is* the ranking, this undermines reproducibility (Turso's
`--doublecheck` principle).
*What:* chain `.then_with(|| a.0.cmp(&b.0))` after the score comparison in `reciprocal_rank_fusion`
and the final sort comparators so ties order by `RecordId`. Add `proptest` as a dev-dep and write two
properties: (a) recall/RRF **determinism** — run fusion twice on a fixed tie-heavy corpus, assert
byte-identical ranking; (b) **count consistency** — arbitrary insert/update/delete/upsert sequences,
assert per-table counts sum to `total_records` with no orphaned index entries.
*Touches:* `crates/axil-core/src/query.rs` (~1488/1508, `apply_sort` ~1412); `scoring.rs` (~446);
`axil-core/Cargo.toml` + `axil-tests` (proptest).

**4. HNSW-vs-brute-force recall oracle.** The `sqlite-compare` harness proves vector search is *fast*
but never proves it's *correct* — it diffs latency, not result sets. An HNSW bug (bad rebuild,
quantization/Matryoshka/binary path drift, deletes-since-rebuild staleness) that silently drops
correct neighbors passes every current gate. Silent recall loss = the agent invisibly forgets things.
*What:* add a `#[cfg(test)]` oracle in axil-vector (primitives exist: `cosine_sim`, `vectors()`).
Generate N deterministic vectors; for M random queries compute exact top-k by brute force and assert
HNSW `search()` top-k overlap ≥ a recall floor (~0.90 at N~2k; **never** assert exact equality — HNSW
is approximate). Run it across the variant paths that silently regress: post-rebuild, `search_mrl`
truncation, int8/binary quantization, and after deletes-since-rebuild. Wire small-N into CI by
extending the needle-recall-gate job; large-N nightly.
*Touches:* `crates/engines/axil-vector/src/hnsw.rs` + tests; `scripts/needle-recall-gate.sh`/`ci.yml`.

**5. Document the full ~19-tool MCP surface.** `docs/src/agents/mcp.md` lists only the 8 original
CRUD tools; the server actually exposes ~19 including the agent-native differentiators `boot`,
`code_context`, `remember_decision`, plus extension tools (`checkpoint`, `dep_docs`). Anyone wiring
Axil via MCP reads the docs, never learns these exist, and falls back to generic `store`/`recall` —
defeating the features that make Axil agent-native.
*What:* rewrite the table to cover the full assembled runtime surface, grouped by purpose (CRUD /
intent-native writes / code / boot / extension tools), each with params + a one-line "when to use".
Add a CI drift test (mirroring the existing `SERVER_INSTRUCTIONS` guard) asserting the doc's tool
list against the full assembled set.
*Touches:* `docs/src/agents/mcp.md`; `crates/adapters/axil-mcp/src/tools.rs` + `lib.rs` (drift test).

### P1 — next (high value, medium effort)

| # | Recommendation | Theme | Impact | Effort |
|---|---|---|---|---|
| 6 | Seeded fault-injection test proving heal recovers the fan-out | Reliability | High | Medium |
| 7 | Incremental HNSW (stop full-rebuild on store-then-recall) | Vector perf | High | High |
| 8 | `inspect` MCP introspection tool (record-type census + health) | Agent surface | Medium | Low |
| 9 | MCP setup ergonomics: `claude mcp add` one-liner + JSON-RPC smoke test | DX | High | Low |
| 10 | Guest SDK + `axil ext new` scaffold for WASM plugin authoring | Plugin DX | High | Medium |
| 11 | Document + harden single-writer/lock-free-reader concurrency contract | Multi-agent | Medium | Medium |
| 12 | Batch the vector + dep-docs ingest path (cut per-chunk fsync) | Ingest perf | Medium | Medium |

**6. Seeded fault-injection test for the multi-engine fan-out.** The torn-write boundary from #2 has
zero fault-injection coverage, and nothing proves the existing repair fns actually recover a *genuine*
inconsistency (`self_healing.rs` only tests clean state). This is Turso's DST idea scoped to Axil's
real durability boundary.
*What:* a ~150-line `#[cfg(test)]` sim: seeded `StdRng` drives a random
insert/embed/relate/index/delete workload against a real Axil; inject a "crash" via a **test-only
failpoint** between `storage.insert/delete` and the engine `on_record_*` calls (not a full VFS
newtype — `redb`/`tantivy` already give per-write atomicity). Reopen, assert
`detect_problems()`/`count_orphaned_*` is non-empty, then assert `heal_all()`/`compact()` drives it
back to empty. Start with delete (dangling FTS is the most likely real corruption — tantivy is the
weakest link). Persist failing seeds to a bugbase for regression replay.
*Touches:* new `crates/axil-sim` or `axil-tests` module; `axil-core/src/db.rs` failpoints;
`axil-tests/tests/self_healing.rs`.

**7. Incremental HNSW.** `instant-distance`'s `HnswMap` is immutable, so **any** add/remove sets
`dirty` and forces a **full O(n log n) rebuild** (cloning all vectors) on the next search, under a
write lock blocking readers. The store-then-recall loop Axil's own brain hooks *mandate* thus pays a
cold rebuild scaling with total memory count, not with what changed.
*What:* replace `instant-distance` with an HNSW supporting incremental insert + lazy tombstone delete.
**Prefer `hnsw_rs` (pure Rust) over `usearch`** — the onnxruntime.dll Windows pain is a direct
warning against adding C++/cmake build surface to a "one binary, easy install" mission. On add: link
the new node in O(log n). On remove: tombstone + over-fetch at query time. Reuse the existing
`deletes_since_rebuild` counters to gate periodic **background** compaction (off the write path via
`axil worker`). Add a cold-recall-after-single-store benchmark at 1k/5k/20k first so the win is a
committed number (numbers-integrity policy).
*Touches:* `crates/engines/axil-vector/src/hnsw.rs`, `lib.rs`, `Cargo.toml` (dep swap);
`axil-core/src/worker.rs`; `benchmarks/vector-latency`.

**8. `inspect` MCP introspection tool.** MCP `list` requires a table name you must already know;
there's no MCP path to answer "what kinds of memory does this brain hold?" or "is it healthy?".
MCP-only clients (Claude Desktop, Cursor) can't shell out to `axil tables/doctor`.
*What:* one read-only `inspect` tool returning per-record-type counts (`db.tables_with_counts()`,
rolling `_`-prefixed engine tables into one `_internal` bucket) plus a light health summary reusing
`db.doctor()`'s read-only checks. Frame output as the memory model ("decisions: 42, errors: 13 (3
unresolved), last write 2d ago, vector index: healthy"), not SQL columns.
*Touches:* `crates/adapters/axil-mcp/src/tools.rs`; `axil-core/src/db.rs` (`tables_with_counts`,
`doctor`).

**9. MCP setup ergonomics.** Add a `claude mcp add axil -- axil mcp ./.axil/memory.axil` one-liner
and a copy-paste JSON-RPC smoke test using **real** methods (`printf initialize → tools/list → recall
| axil mcp <DB>`) so a dev can verify the server with zero client. Reconcile the README over-promise:
point Cursor/Windsurf/Codex users at `axil install --cursor/--windsurf` (the rules-file path that
actually exists) rather than implying per-client MCP snippets exist.
*Touches:* `docs/src/agents/mcp.md`; `README.md` (client matrix); CLI `install_wizard.rs`.

**10. Guest SDK + `axil ext new` scaffold.** The `Plugin` trait + `export_plugin!` macro that make
WASM-guest authoring ergonomic live only inside `test-guest/src/sdk.rs`; docs literally tell authors
to copy it (and `conformance-guest` hand-rolls all 10 raw methods — the copy-paste tax is already
visible).
*What:* lead with the **scaffold** (higher value than a crates.io publish for a PolyForm-NC niche
project): `axil ext new <name> [--caps recall,records.write]` emitting a buildable guest crate
(cdylib, detached workspace, component target pinned to the bundled WIT, a `lib.rs` stub overriding
one high-value hook like `boot_block`). Promote `sdk.rs` into a shared in-tree path so `test-guest`
and `conformance-guest` stop diverging. Make build instructions match reality (verify the actual
wasm target, not the assumed `wasip2`). Defer the crates.io publish until external demand.
*Touches:* new `crates/sdk/axil-plugin-sdk`; `axil-runtime/test-guest/src/sdk.rs`; CLI
`wasm_plugins.rs`; `docs/src/extending/wasm-plugins.md`.

**11. Document + harden the single-writer / lock-free-reader contract** (do **not** build a shared-WAL
coordinator). Axil advertises multi-agent shared memory, but a second long-lived process gets `redb`'s
opaque "Cannot acquire lock", and brain hooks already spawn detached scip/maintain/index subprocesses
that race the foreground agent for companion-file locks. There is **not a single** concurrency
correctness test.
*What:* (1) lock-free readers for hot subprocess paths via the already-present
`ReadOnlyDatabase::open` (proven in `axil-vector lib.rs:348`) for read-only ops (boot, recall,
code-search). (2) Map `redb`'s `DatabaseAlreadyOpen` to a typed `AxilError::Busy` with a short bounded
retry on the foreground writer. (3) Reuse the existing `LockGuard`/`O_CREAT|O_EXCL` pattern
(scip-refresh.lock), not a new dep. (4) Add one multi-process integration test. (5) Document the
topology. **Explicitly drop** any `.tshm`-style coordinator.
*Touches:* `axil-core/src/storage.rs`, `error.rs`; engines (`ReadOnlyDatabase::open`); CLI read paths;
`axil-tests`; `docs/src/concepts/storage.md`.

**12. Batch the vector + dep-docs ingest path.** FTS already batches, but the vector engine has **no
batch API** — even `insert_batch_records` loops `vi.add()` per record = N `.vec` fsyncs. dep-docs
ingest is worse (per-chunk `insert` + `embed_field` + `index_text`), so a dep with hundreds of chunks
pays hundreds of fsyncs across all three files. This is exactly the ingest latency the agent waits on
during boot/scip-refresh/deps-refresh.
*What:* add `VectorIndex::add_batch(&[(RecordId, &[f32])])` with a default loop impl (no breaking
change), override in `VectorEngine` with a single `.vec` begin_write/commit (one fsync), and route
`insert_batch_records` + dep-docs ingest through it — mirroring the existing FTS
`index_records_batch`. Fix the stale doc comment claiming all hooks are per-record post-batch.
*Touches:* `crates/engines/axil-vector/src/lib.rs`; `axil-core/src/db.rs` (`insert_batch_records`);
`crates/extensions/axil-docs/src/ingest.rs`.

### P2 — strategic / opportunistic

| # | Recommendation | Theme | Impact | Effort |
|---|---|---|---|---|
| 13 | Durable opt-in `_changelog` CDC tape (Atlas keystone + cheaper merge) | Sync foundation | High | High |
| 14 | Pull-based `recall_delta` + boot block on an upgraded audit log | Live context | Medium | Medium |
| 15 | Make `branch create` (the backup path) point-in-time consistent | Reliability | Medium | Low |
| 16 | Lazy importance decay at read time (delete the worker's full-DB sweep) | Incremental compute | Medium | Low |
| 17 | Opt-in encryption-at-rest for record bodies (honestly scoped) | Security | High | High |
| 18 | One cargo-fuzz target for AxilQL parser + contributor agent-guides | Robustness / DX | Low | Medium |

**13. Durable opt-in `_changelog` CDC tape.** Axil has the write-time fan-out seam
(`run_insert_hooks`) but persists no event, so "what changed since cursor X" requires whole-table
diffs — exactly what `branch_merge` is forced to do today (O(all-records)). A durable,
cursor-addressable tape is the keystone Turso's entire sync stack rests on, and the foundation the
closed **Atlas** product will need. No in-tree consumer yet beyond merge, so ship lean and **off by
default**. *What:* feature-gated `_changelog` written **inside** the existing redb write txn in
`storage.rs` (not in the post-commit `run_insert_hooks`), keyed by a fresh ULID `change_id` used
directly as the cursor. Default capture = id-only; before/after opt-in. Expose `changes_since(cursor)`
and **prove value now** by rewriting `branch_merge` to replay the tape. Self-prune via the existing
tiering/decay path. Reserve a versioned `_sync_meta` shape so Atlas doesn't force a later migration.
Do **not** build replication/LSN/reader-isolation plumbing — no consumer in this repo.
*Touches:* `axil-core/src/storage.rs` + new changelog module; `branch.rs` (merge); `tiering.rs`;
`record.rs`.

**14. Pull-based `recall_delta`.** In multi-agent setups, agent B should learn what A committed
without re-polling `recall`. Axil's `audit_log` already answers "what changed since T" but is off by
default, a lossy ring buffer, and **skips all `_`-prefixed tables** — so belief revisions,
supersede/consolidation, and checkpoints (the exact semantic events worth streaming) are never
recorded. *What:* drop the in-process push channel; extend `audit_log` into a durable, opt-in semantic
event log — stop skipping `_`-prefixed tables for a curated allowlist (belief-revised,
decision-superseded, error-fixed, checkpoint-written), tag entries with `agent_id`, use a monotonic
ULID cursor. Surface as an MCP `recall_delta(since_cursor, exclude_agent)` tool and a
`recent_changes` block in `axil boot`. Note explicitly that it does **not** relax cross-agent session
isolation — it surfaces committed facts only.
*Touches:* `axil-core/src/db.rs` (audit_log, `_`-table skip), `worker.rs`, `boot.rs`; MCP `tools.rs`.

**15. Point-in-time-consistent `branch create`.** The shipped, documented "atomic copy of the database
files" command does a blind sequential `fs::copy` of core + companions while another process may be
mid-write — so a backup can capture core and `.vec` at different logical points. The word "atomic" is
false (a claims-integrity issue). *What:* retarget `branch_create` to take the live `&Axil` handle,
hold a redb read transaction open across the core copy, and quiesce each engine so all companions
reflect one logical point. Stop calling it "atomic" until it is. Delete or wire+fix the divergent dead
`create_snapshot`/`restore_snapshot` so there aren't two "snapshot" paths.
*Touches:* `axil-core/src/branch.rs`; `snapshot.rs`; CLI help; `docs/src/advanced/branching.md`.

**16. Lazy importance decay at read time.** The worker's `run_decay` walks every record in every table
re-evaluating `effective_importance` each cycle. Recency is *already* computed lazily from
`created_at` at query time, and `effective_importance` is a pure function — so the whole-DB write
sweep is avoidable. *What:* compute effective importance at **read** time in `scoring.rs` (mirroring
`recency_decay`) instead of reading the stored `_effective_importance` field. This deletes `run_decay`
entirely (one of three worker loops gone) with near-zero new infra. **Reject** any
DBSP/incremental-view machinery — Axil has no relational view layer; defer the dirty-set rewrite of
the O(n²) consolidate/strengthen loops until entity counts hit the thousands.
*Touches:* `axil-core/src/scoring.rs`; `worker.rs` (remove `run_decay`); `importance.rs`.

**17. Opt-in encryption-at-rest for record bodies (honestly scoped).** Axil stores recalled code,
decisions, error traces, and anything the agent saw (possibly secrets/PII) as **plaintext**. For a
memory product that "follows a developer across machines", a synced or lost plaintext file is a real
enterprise/regulated-user adoption blocker. *What:* off-by-default `encryption` Cargo feature doing
value-level AEAD — encrypt `record.to_bytes()` before insert, decrypt in `from_bytes`/`get`. Prefer
`chacha20poly1305` (pure-Rust) over `aes-gcm`; per-record random nonce; bind AEAD additional-data to
record id + table. **Key management is the real design problem, not the cipher**: agents recall
non-interactively, so ship key-from-env (`AXIL_ENC_KEY`) + keyfile first, Argon2-from-passphrase for
the laptop-unlock case. Be explicit in docs that v1 encrypts **record bodies only** — `.vec`
embeddings (a reconstruction channel) and `.fts` tokens stay cleartext — so the honest pitch is
"encrypted record bodies", not "encrypted memory" (numbers/claims-integrity).
*Touches:* `axil-core/src/record.rs` (`to_bytes`/`from_bytes`), `storage.rs`, `config.rs`, new crypto
module + feature; docs.

**18. One cargo-fuzz target for AxilQL + contributor agent-guides.** Two cheap items merged.
(a) AxilQL query text is the one genuinely agent/user-supplied **untrusted** byte surface — add one
cargo-fuzz target for `axil_ql::parse` (assert no panic/OOM), seeding its corpus from the existing
`comprehensive.rs` adversarial inputs; wire as a warn-only nightly batch. Skip SCIP/lockfile/JSON
targets (developer-controlled, serde-hardened). (b) Add a thin `docs/agent-guides/` that mostly
*points* at existing material (extending taxonomy, gate scripts, numbers-integrity policy, MCP parity
tests) + a task→guide table in `AGENTS.md`, and **move** per-task contributor mechanics out of the
~6.3k-token always-loaded root `CLAUDE.md` to cut per-task context cost for Axil's own dogfooded agent.
*Touches:* new `fuzz/` crate; `axil-ql/tests/comprehensive.rs`; `nightly-bench.yml`; new
`docs/agent-guides/*`; `AGENTS.md`; `CLAUDE.md`.

---

## 4. Verification ledger (8 dimensions, 26 gaps)

Every candidate gap was re-checked against Axil's source by an independent adversarial agent.

| Dimension | Gap | Verdict |
|---|---|---|
| Reliability & Testing | Seeded DST for multi-engine fan-out + crash recovery | **real_gap** |
| | Property-based testing of storage + scoring invariants (proptest) | **real_gap** |
| | Differential/oracle correctness vs brute-force reference | **real_gap** |
| | libFuzzer targets for parser-facing surfaces | partial |
| | Concurrency / multi-process correctness tests | partial |
| Vector search | Incremental HNSW (no full rebuild on add/remove) | **real_gap** |
| Storage/concurrency | Cross-file crash consistency (torn insert loses embedding) | **real_gap** |
| | Encryption at rest | **real_gap** |
| | Single-process only — no safe concurrent access | **real_gap** |
| | Snapshots not point-in-time-consistent / not incremental | **real_gap** |
| | Synchronous blocking I/O on critical path (no batched fsync) | partial |
| Sync / CDC | Durable `_changelog` CDC tape | **real_gap** |
| | Live CDC stream feeding agent context | partial |
| | Change-log-driven branch merge + incremental snapshot | partial |
| | Per-client cursor metadata + invertible changes (Atlas) | **real_gap** |
| Bindings / distribution | Prebuilt CLI binaries via cargo-dist | **real_gap** |
| Incremental compute | Delta-driven worker (replace full re-scan with dirty-set) | partial |
| Extensions / plugins | Publish guest SDK crate + scaffold command | **real_gap** |
| | Streaming host import for virtual-table-style data sources | partial |
| | Conformance suite + golden fixtures for plugin authors | partial |
| | Compile-time-static vs loadable dual-build pattern | partial |
| MCP / agent DX | MCP docs list 8 of ~19 tools | **real_gap** |
| | No schema/introspection MCP tool | partial |
| | README MCP setup ergonomics lag | partial |
| | No MCP resources/prompts (boot/skills as native primitives) | partial |
| | No contributor-facing agent guides | partial |

**Tally: 13 real_gap · 13 partial · 0 already-have/not-applicable.**

---

## 5. Concrete Turso artifacts worth copying (citations)

**Distribution — `dist-workspace.toml`** (the template for P0 #1):
```toml
[dist]
cargo-dist-version = "0.31.0"
ci = "github"
installers = ["shell", "powershell"]
targets = ["aarch64-apple-darwin", "aarch64-pc-windows-msvc", "aarch64-unknown-linux-gnu",
           "x86_64-apple-darwin", "x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"]
install-path = "~/.turso"
install-updater = true
github-attestations = true
precise-builds = true
```
Turso also hand-edits the generated `release.yml` to add Azure Windows **code signing** (`allow-dirty
= ["ci"]`), and installs land in `~/.turso` — the same pattern Axil could use for `~/.axil/bin`.

**Reliability — `testing/simulator/` (`limbo_sim`)** (the model for P0 #2 / P1 #6): a seeded
(`--seed`) randomized simulator with a **simulated IO layer** (`runner/memory/io.rs`, virtual clock
in `runner/clock.rs`) that injects read/write/sync faults and latency from `profiles/io.rs`
(`FaultProfile{read,write,sync}`, `LatencyProfile{...}`). `--doublecheck` runs a plan twice and
asserts identical output (determinism). Failing seeds are **shrunk** (`shrink/plan.rs`) to a minimal
repro and persisted to a **bugbase** (`runner/bugbase.rs`). The differential oracle
(`testing/differential-oracle/fuzzer/oracle.rs`) runs every statement against both Turso and SQLite
and — crucially — does **post-mutation hidden-state verification** (`verify_table_snapshots`),
catching corruption a query result alone would mask. Axil doesn't need the SQL parts; it needs the
*seeded-fault-injection-then-assert-invariant* skeleton scoped to its companion-file boundary.

**Bindings — thinness** (context for "don't chase 9 bindings"): each binding is a thin wrapper over
`core` (e.g. Python = PyO3 in `bindings/python/src/`, with sync + async variants, a SQLAlchemy
dialect, typed stubs) generated off a single C-ABI `sdk-kit` (`sdk-kit/turso.h`). Axil's equivalent
investment is better spent on the C-ABI/WASM SDK ergonomics (P1 #10) than on per-language drivers it
has no SQL protocol to serve.

---

## 6. Methodology

1. Depth-1 clone of `tursodatabase/turso` into a scratchpad; structural scout of `core/`,
   `testing/`, `bindings/`, `sync/`, `extensions/`, `dist-workspace.toml`, README.
2. A multi-agent workflow analyzed **8 dimensions** in parallel — Reliability & Testing, Vector
   search, Storage/concurrency/encryption, Sync & CDC, Bindings & distribution, Incremental compute,
   Extensions/plugins, MCP & agent DX. Each analyst read the relevant Turso code **and** verified
   Axil's actual current state via `axil recall`/`code-search`/`fts` + source reads.
3. **Every** claimed gap was handed to an independent adversarial verifier instructed to default to
   skepticism and to check whether Axil already has the feature — producing the
   real_gap/partial/already-have/not-applicable verdicts above.
4. A lead-architect synthesis pass deduped across dimensions and prioritized into P0/P1/P2.

**Caveat:** this is a *desk analysis* of a depth-1 snapshot. File line numbers in "Touches" are from
the verification agents' reads and should be confirmed before editing. No Axil code was changed.
