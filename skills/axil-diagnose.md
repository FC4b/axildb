---
name: axil-diagnose
description: "Read field reports from working projects, fix Axil Rust source code, run tests"
trigger: when user mentions axil diagnose, fix axil, axil field reports, diagnose axil issues
---

# Axil Diagnose — Fix Reported Issues

Read field reports from working projects that use Axil, correlate problems with source code, fix bugs, and verify with tests. **This skill is for use in the Axil source repository only.**

## Prerequisites

- You are working in the Axil source repo (`axildb/`)
- Field reports exist in `reports/incoming/` or in sibling project directories

## Workflow

### 1. Find Reports

```bash
# Check the incoming directory
ls reports/incoming/*.json 2>/dev/null

# Check sibling project directories (configured in axil.toml)
axil config get diagnose.watch_projects

# Import from a specific project
axil report import --from ../my-agent-app
```

Also scan for reports at:
- `../*/axil-reports/*.json` (sibling project directories)
- Path set in `AXIL_REPORTS_DIR` environment variable
- `reports/incoming/` in this repo

### 2. Parse and Prioritize

Read each report's `problems` array. Prioritize by:

1. **Severity**: `error` > `warning` > `info`
2. **Frequency**: same component across multiple reports = higher priority
3. **Type**: `crash` > `error` > `data` > `performance`

### 3. Correlate with Source Code

Map each problem's `component` to source files:

| Component | Source Location |
|-----------|----------------|
| `storage` | `crates/axil-core/src/storage.rs`, `crates/axil-core/src/db.rs` |
| `vector_search` | `crates/axil-vector/src/hnsw.rs` |
| `graph` | `crates/axil-graph/src/edge.rs`, `crates/axil-graph/src/traverse.rs` |
| `fts` | `crates/axil-fts/src/index.rs` |
| `timeseries` | `crates/axil-timeseries/src/` |
| `cli` | `crates/axil-cli/src/main.rs` |
| `config` | `crates/axil-core/src/config.rs` |

### 4. Common Fix Patterns

**Panics / unwrap errors:**
- Search for `unwrap()` in the reported component
- Replace with proper error handling (`?`, `.ok_or_else()`, `.map_err()`)
- Add context with `anyhow::Context`

**Performance issues:**
- Check for allocations in hot loops (`Vec::new()` inside iterations)
- Look for unnecessary `.clone()` — use references
- Check for missing `with_capacity()` pre-allocation
- Profile with `--release` builds

**Data issues:**
- Check serialization/deserialization round-trips
- Verify query filter logic in `crates/axil-core/src/query.rs`
- Check edge cases in graph traversal

**Concurrency issues:**
- Check lock ordering (prevent deadlocks)
- Verify `Send + Sync` bounds
- Look for `RwLock` contention in vector operations

### 5. Fix and Verify

```bash
# After making fixes, run the full test suite
cargo test --workspace

# Run clippy
cargo clippy --workspace --all-features -- -D warnings

# Run specific crate tests
cargo test -p axil-core
cargo test -p axil-vector
cargo test -p axil-graph
cargo test -p axil-fts
cargo test -p axil-timeseries
cargo test -p axil-tests
```

### 6. Generate Fix Summary

After fixing, document what was done:
- Which report(s) were addressed
- What was changed and why
- Which tests were added
- What needs further discussion

Move processed reports from `reports/incoming/` to `reports/addressed/`.

## Example Flow

```
Found 3 field reports:
  reports/incoming/report-2026-04-01.json (2 problems)
  ../my-agent-app/.axil-reports/report-2026-03-30.json (1 problem)

Analyzing...

Report #1 (report-2026-04-01.json):
  [error] Panic on concurrent axil store — unwrap() in storage.rs:142
    → Fixed: replaced unwrap() with proper error propagation.

  [warning] Recall takes 230ms with 10k vectors
    → Fixed: pre-allocate Vec with capacity in hnsw.rs search path.

Report #2 (report-2026-03-30.json):
  [warning] FTS returns 0 results for queries with brackets
    → Fixed: sanitize query input before passing to tantivy.

Ran: cargo test --workspace — all pass
Ran: cargo clippy --workspace — clean

Fixed 3 issues across 3 files.
Working projects using path dependency will get fixes on next build.
```
