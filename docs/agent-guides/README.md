# Agent Guides — Contributor Mechanics, by Task

This directory is a **thin index for agents (and humans) working in the Axil
repo**. It does not re-document anything; it points at the canonical source for
each kind of contribution so a contributor can find the one file that already
answers their question instead of re-deriving it.

> **Why a separate index?** The root `CLAUDE.md` / `AGENTS.md` are loaded into
> *every* agent turn, so they must stay lean. Per-task mechanics that only
> matter while you are doing that specific task live here and are read on
> demand — keeping the always-loaded context small (see
> [Context Economics](../src/advanced/context-economics.md)).

This is a **pointer index, not a tutorial.** Every row links to the
authoritative file. If a pointer and its target ever disagree, the target wins —
fix the pointer.

## Task → where to look

| If you are… | Read | Then |
|---|---|---|
| Adding a new storage index (vectors, edges, FTS, …) | [Engines (Tier 1)](../src/extending/engines.md) · [Three-tier overview](../src/extending/overview.md) | Implement `Engine` + an index trait; own a `*.axil.<suffix>` companion file |
| Adding a capability on top of existing engines (new tables, CLI/MCP/hooks) | [Extensions (Tier 2)](../src/extending/extensions.md) | Own `_`-prefixed tables; register CLI/MCP via the `Extension` trait |
| Putting Axil behind a new protocol (HTTP, a query language, …) | [Adapters (Tier 3)](../src/extending/adapters.md) | Translate the protocol to/from `Axil::query()`; store nothing |
| Authoring a WASM runtime plugin | [WASM Plugins](../src/extending/wasm-plugins.md) | Guests build **detached** via `cargo component` — not as workspace members |
| Adding or changing an MCP tool | [MCP server reference](../src/agents/mcp.md) | Keep CLI/MCP parity — see the parity test below |
| Surfacing any savings / speed-up / % figure | [Numbers integrity policy](../src/advanced/context-economics.md#numbers-integrity-policy) | Every number must trace to a baseline, an estimate (named heuristic), or a committed benchmark |
| Touching recall ranking / quality | [Retrieval pipeline](../src/advanced/retrieval-pipeline.md) · the recall gates below | Re-run the relevant gate; re-baseline only if a snapshot pins exact order |
| Fuzzing the untrusted parse surface | [`fuzz/`](#fuzzing) | `cargo +nightly fuzz run ql_parse` |

## Verification gates (run the one your change touches)

These are the committed, reproducible gates. CI runs them; run them locally
before you push.

| Gate script | Guards | When it runs |
|---|---|---|
| [`scripts/needle-recall-gate.sh`](../../scripts/needle-recall-gate.sh) | FTS needle retention (planted UUIDs/errors still surface in top-k) | per-PR |
| [`scripts/sqlite-compare-gate.sh`](../../scripts/sqlite-compare-gate.sh) | Vector-search speedup floor (Axil HNSW vs sqlite-vec, reduced n) | per-PR |
| [`scripts/code-recall-gate.sh`](../../scripts/code-recall-gate.sh) | Code-recall regression (proxy/file/symbol surfacing) | local / on demand |
| [`scripts/bench-check.sh`](../../scripts/bench-check.sh) | Criterion latency regression (>5%, `core`/`vector`/`graph`/`fts`) | nightly (informational) |
| [`scripts/longmemeval-gate.sh`](../../scripts/longmemeval-gate.sh) | LongMemEval recall vs committed 500-Q baseline | dataset-gated (skip-loud) |

> The dataset-gated gates (LongMemEval / LoCoMo) emit a loud `::warning` skip
> when their out-of-tree dataset is absent — a green CI run never means they
> verified anything. Committed baselines live in `benchmarks/results/`.

> **Recall core is human-review-only.** These gates are threshold-based, not
> exact oracles — they bound recall regressions, they do not define
> correctness. Changes to `query.rs` / `scoring.rs` / the `brain.rs` recall
> paths need human review, not autonomous/multi-agent runs. See the
> **Oracle-scoped autonomy** convention in the root [`CLAUDE.md`](../../CLAUDE.md).

## Parity & drift tests

Some surfaces must not drift apart silently. These tests fail the build if they
do:

- **CLI ↔ MCP parity** — [`crates/adapters/axil-mcp/tests/parity.rs`](../../crates/adapters/axil-mcp/tests/parity.rs).
  If you add an MCP tool, mirror its CLI counterpart (and vice versa).
- **MCP doc ↔ runtime drift** — the drift guard in
  [`crates/adapters/axil-mcp/src/lib.rs`](../../crates/adapters/axil-mcp/src/lib.rs)
  asserts the documented tool list matches the assembled runtime surface.
- **Extension audit** — new `Extension`s are picked up automatically by the
  extension-audit test; a missing registration fails it.

## Fuzzing

`axil_ql::parse` is the one genuinely untrusted byte surface in Axil — an
AxilQL query string can arrive verbatim from an external client. The
[`fuzz/`](../../fuzz/) crate holds a [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
target that asserts the parser never panics, aborts, OOMs, or hangs on
arbitrary bytes.

```sh
# Requires nightly + libFuzzer (cargo-fuzz). The fuzz crate is detached from the
# workspace, so build it via its own manifest:
cargo install cargo-fuzz
cargo +nightly fuzz run ql_parse
```

The seed corpus under `fuzz/corpus/ql_parse/` mirrors the adversarial inputs
already asserted panic-free in
[`crates/adapters/axil-ql/tests/comprehensive.rs`](../../crates/adapters/axil-ql/tests/comprehensive.rs)
(the `f0*` fuzz-safety tests: garbage bytes, a 100K string, null bytes,
unicode). A nightly, warn-only CI step (`nightly-bench.yml` → `ql-fuzz`) runs a
short bounded smoke; findings land as reproducible artifacts under
`fuzz/artifacts/` and are surfaced as a warning, never a blocking failure.

## Code conventions (the short list)

The full conventions live in the root [`CLAUDE.md`](../../CLAUDE.md). The two
most easily missed:

- **No task/phase tags in code comments.** Comments explain *why the code is the
  way it is*; which phase shipped it is git-history noise. Phase numbers belong
  in commit messages and Axil memory, not the source.
- `thiserror` in library crates, `anyhow` in binary crates (CLI, MCP server).
  Public APIs get doc comments.
