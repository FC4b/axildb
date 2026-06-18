# Diagnostics, Lifecycle & Maintenance

These health/maintenance commands fall into **three families**, and telling
them apart is the usual source of confusion:

- **Diagnostics** only *read* the DB — they never change your memories.
- **Lifecycle** commands *change or forget* memories.
- **Maintenance** (`maintain`) just *automates* the additive diagnostics.

See the [CLI overview cheat-sheet](./overview.md#diagnostics-vs-lifecycle-vs-maintenance--when-to-use-which)
for a quick "when to use which".

## Diagnostics (read-only — never change memories)

### doctor

Quick read-only health check — "is anything wrong right now?". For a scored
report with fix recommendations, use `health-report`.

```bash
axil doctor
```

Checks: companion-file presence/consistency, vector dimension mismatch, SCIP
freshness, indexer staleness, orphaned edges/vectors/FTS, recent
error/healing history.

### health-report

Scored health assessment (0–100) + fix recommendations — the deeper sibling
of `doctor`. `--save`/`--compare` track the score over time.

```bash
axil health-report            # full report
axil health-report --brief    # one-line summary
axil health-report --save     # store to _health_reports
axil health-report --compare  # diff against the last saved report
```

### snapshot

Record DB metrics (record/vector/edge counts, latencies) for trend charts —
**not** a data backup. (For a data copy, use `axil branch create`.) Aliased
`metrics-snapshot`; feeds `trends`.

```bash
axil snapshot                 # one-shot capture
axil metrics-snapshot         # same command, clearer name
```

### trends

Chart the metric history that `snapshot` records over time.

```bash
axil trends --days 30
```

### detect

Deep / expensive problem scan — the detectors `doctor` skips.

```bash
axil detect
```

Detectors: stale sessions (>24h active), slow queries, storage growth rate,
embedding drift.

### stats

Show database statistics (table counts, sizes).

```bash
axil stats
```

## Memory lifecycle (these CHANGE or forget memories)

### compact

Hard-delete expired/superseded records and clean orphaned edges/vectors/FTS,
reclaiming space. Does **not** downsample. Runs automatically when the delete
count crosses `auto_compact_threshold` (default 1000).

```bash
axil compact
```

### heal

Rebuild drifted indexes or roll up old data. Run deliberately — check
`axil doctor` first.

```bash
axil heal --compact     # same as `axil compact`
axil heal --reindex     # rebuild all indexes from the canonical records (slow)
axil heal --orphans     # clean orphaned companion entries only
axil heal --dry-run     # print what would be fixed, change nothing
axil heal               # also DOWNSAMPLES (see warning below)
```

> ⚠️ A bare `axil heal` **downsamples**: it deletes records older than
> `full_retention_days` (default 90d), replacing each day's rows with a single
> count summary. This is irreversible. `axil maintain` deliberately never runs
> it. Don't run `heal --reindex` during an active agent session — it
> clears/rebuilds index tables and queries can return incomplete results mid-rebuild.

### worker

Run background cognitive maintenance: importance **decay**, consolidation, and
connection inference. Fired automatically by the Stop hook each session.

```bash
axil worker run                                    # single run
axil worker status                                 # last run report
axil worker daemon --interval 300 --duration 3600  # background loop
```

## Automation

### maintain

Opportunistic, time-gated maintenance — runs the *additive* diagnostics
(`snapshot` + `health-report --save`) only when their cadence
(`[maintenance]` in `axil.toml`) has elapsed. Fired by the brain hook on each
session start, non-blocking. Never runs the destructive downsample/reindex.

```bash
axil maintain --dry-run    # show what would run, change nothing
axil maintain --if-stale   # run only what's due (the hook's mode)
axil maintain              # run every eligible task now
```

See [Memory Hygiene](../advanced/memory-hygiene.md) for the full maintenance
toolkit and recommended cadence.

## scip

Manage the SCIP code-graph index. Closes the loop between `doctor` (which warns when SCIP is missing or stale) and `ingest-scip` (which only consumes a pre-existing file).

```bash
axil scip status                              # (language, project dir) pairs, indexer presence, file age
axil scip refresh                             # detect every project → run each indexer → ingest all
axil scip refresh --language typescript       # restrict the sweep to one language's projects
axil scip refresh --if-stale                  # skip in <50ms when every index is fresh (≤14d)
axil scip refresh --if-stale --in-background  # spawn detached refresh, return instantly
axil scip refresh --root path/to/project      # scan this tree instead of the DB-derived root
```

`--root` overrides where projects are detected (output `.scip` files still land
next to the database). It defaults to the repo root derived from the DB location
(`<db>/../..`); set it when the database lives outside the project being indexed.
`axil reindex <path>` passes the indexed path through automatically so its proxy
and SCIP layers always cover the same tree.

`refresh` orchestrates an **external** language indexer that must be on your `PATH`
— `rust-analyzer`, `scip-typescript`, `scip-python`, `scip-go`, or `scip-java`. Run
`axil scip status` to see which are installed and the exact install command for any
that are missing; see the [indexer prerequisites table](./code-search.md#generating-the-scip-code-graph)
for the full list.

Polyglot repos are swept in one run. Detection walks subfolders (depth ≤ 4; `node_modules`, `target`, dot-dirs, and gitignored dirs skipped), so a monorepo with `frontend/package.json` and `backend/pyproject.toml` — even with no marker at the root — gets each indexer run from its own project directory. Single-project repos keep writing `.axil/index.scip`; polyglot repos write one `.axil/index-<lang>-<dir>-<hash>.scip` per project (the short hash keeps lossy-slug siblings like `web-ui` / `web.ui` from sharing a file), and a leftover single-file `index.scip` is retired automatically once a full sweep covers every project. A missing indexer binary skips that project with an install hint instead of failing the sweep; an explicit `--language` keeps the old hard-error contract. `--if-stale` is checked per project, and a project whose indexer isn't installed counts as non-actionable so it can't defeat the fast path. A failed ingest removes its `.scip` so the next refresh retries instead of being masked by freshness.

The brain hook (`.claude/hooks/axil-brain.sh`) calls `axil scip refresh --if-stale --in-background --quiet` on first PreToolUse, so SCIP stays fresh transparently — in polyglot repos this now keeps *every* language's index fresh, not just the first detected one. Lock file at `.axil/scip-refresh.lock` prevents concurrent spawns; child runs under `nohup` to survive parent shell exit. Staleness threshold (14 days by default) matches `axil doctor`'s SCIP warning.

> To refresh the structural proxy index (`axil index`) **and** the SCIP graph in a single call, use [`axil reindex`](./code-search.md#refresh-everything-axil-reindex) — the proxy layer rebuilds in the foreground and the SCIP refresh is spawned in the background, the same machinery described here.
