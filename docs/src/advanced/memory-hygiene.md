# Memory Hygiene

Long-running Axil databases drift: records get superseded, vector
deletes leave orphans in the HNSW graph, FTS schema changes leave the
old index behind, redb fragments after many writes. This page
documents the maintenance toolkit and when to use each piece.

## The toolkit at a glance

| Command | What it does | Run when |
|---------|--------------|----------|
| `axil doctor` | Read-only health check | Anytime — fast, no writes |
| `axil compact` | Purge expired/superseded records, clean orphans | DB feels bloated, after large deletes |
| `axil heal [--reindex] [--orphans] [--dry-run]` | Compact + rebuild drifted companions | After crash, schema change, or `doctor` flags issues |
| `axil maintain [--if-stale] [--dry-run]` | Opportunistic, time-gated snapshot + health-report (additive only) | Auto-fired by the brain hook; never needs cron |
| `axil health-report [--brief] [--save] [--compare]` | Full health snapshot + trend tracking | Weekly / before releases |
| `axil snapshot` | Capture current metrics for trend tracking | Cron'd hourly/daily for trends |
| `axil trends [--days N]` | Show metric history | When investigating regressions |
| `axil detect` | Run deferred problem detectors | Anytime — surfaces issues `doctor` doesn't |
| `axil session-heal` | End-of-session: replay captured failures, auto-fix | Stop hook, after every session |
| `axil branch create <name>` | Atomic point-in-time copy | Before risky operations |

## `axil doctor` — read-only health check

Runs a battery of checks and prints a structured report. Doesn't
modify the DB. Covers:

- Companion file presence + consistency
- Vector dimension mismatch (embedder vs index)
- SCIP index freshness (if a code repo)
- Indexer staleness (files changed since last index)
- Orphaned edges / vectors / FTS entries
- Recent error/healing history

Use `axil doctor` as the first step when something feels off. It tells
you which heavier tool to reach for next.

## `axil compact` — purge + orphan cleanup

`db.compact()` does three things:

1. Purges records with expired `valid_until` timestamps
2. Hard-deletes records marked `superseded`
3. Cleans orphaned edges, vectors, and FTS entries (those pointing at
   record IDs that no longer exist)

Skipped: pinned records, records with importance ≥ 0.8, and every record
in a table configured `compact = "never"` (see below). Returns a
`CompactReport` showing counts cleaned. Cheap to run on a healthy DB —
most cleanup paths short-circuit when there's nothing to do.

Compaction also runs *automatically*: the end-of-session heal pass
(`axil session-heal`, run by the Stop hook) and bare `axil heal` both
compact whenever any expired or superseded records exist. Set
`[healing] auto_compact = false` to make compaction strictly manual —
automatic healing then reports what is pending instead of purging, and
only an explicit `axil compact` / `axil heal --compact` deletes.
Orphan cleanup (dangling edges/vectors/FTS entries) still runs either
way — it repairs referential integrity and never deletes records.

## Append-only tables — `[lifecycle.tables.<t>]`

Auto-supersede assumes similar text means *a newer revision of the same
fact*. That is wrong for experiment logs, trade autopsies, and audit
trails, where hundreds of similar-sounding records are **distinct events
that must all survive** (counting every trial, keeping lineage parents
resolvable). Give those tables append-only semantics:

```toml
[lifecycle.tables.autopsies]
supersede = false   # never demote records in this table
decay = false       # importance never decays
compact = "never"   # compact()/heal never delete from this table
```

`compact = "never"` also covers time-series downsampling (the bare
`axil heal` retention purge): protected tables are neither summarized
nor deleted when records age past `full_retention_days`. Protected
records are likewise excluded from the "pending cleanup" counts, so
`doctor` and `session-heal` don't nag (or auto-heal) about records that
are kept by design. The policy is enforced inside the core
insert/compact/downsample paths, so CLI, MCP, and embedded use all
honor it. To scope it to a single database in a multi-DB project, put
the `axil.toml` next to that `.axil` file — the nearest config wins.

## `axil heal` — compact + rebuild

The heavier sibling of `compact`. Used when companion files drift from
their canonical records (after crashes, schema migrations, or manual
file mucking).

| Flag | Effect |
|------|--------|
| `--compact` | Just compact (same as `axil compact`) |
| `--reindex` | Rebuild all indexes from the canonical records |
| `--orphans` | Clean orphaned companion entries only |
| `--dry-run` | Print what would be fixed, change nothing |

`axil heal --reindex` is the recovery command — re-embeds every
indexed record, rebuilds FTS from record text, regenerates graph
edges. Slow (it's effectively a full re-index), so run it deliberately
after `doctor` confirms a real problem.

> **Concurrent reads warning** — like `axil index --full`, `heal
> --reindex` clears and rebuilds index tables. Do not run it during an
> active agent session: queries against the partially-rebuilt indexes
> can return incomplete results until the rebuild finishes. Quiesce
> writers and pause read clients, or schedule it between sessions.

## `axil health-report` — comprehensive health + trends

Produces a richer report than `doctor`:

```bash
axil health-report                    # full JSON report
axil health-report --brief            # one-line summary
axil health-report --save             # store the report as _health_reports
axil health-report --compare          # diff against last saved report
```

The `--save` / `--compare` pattern is the trend mechanism: save a
report weekly, compare to see if scoring is drifting up or down over
time.

## `axil snapshot` — metrics snapshot for trends

**This is a metrics snapshot, not a data snapshot.** It records the
current values of internal counters and latency histograms into a
trend-tracking table so `axil trends` can chart drift over time.

```bash
axil snapshot                  # one-shot capture
axil trends --days 30          # chart the last 30 days
```

For a *data* snapshot (atomic copy of the database files), use
`axil branch create <name>` instead.

## `axil detect` — deferred problem detectors

Detectors that are too expensive to run inside `doctor` live here.
Currently:

- Stale sessions (sessions that opened but never closed)
- Slow query log analysis
- Storage growth anomalies
- Embedding drift (when the model changed under the index)

Run periodically; usually surfaces issues that auto-fix on the next
heal.

## `axil session-heal` — end-of-session auto-fix loop

The Stop hook captures axil command failures and empty-result misses
to a per-session JSONL file. `axil session-heal` reads that file,
runs `detect_problems()`, applies auto-fixable repairs (compact /
reindex / orphans), classifies misses (e.g. empty `code-search` →
suggests reindex), and writes a `_heal_log` row so the next session
sees what was fixed.

This is the closed-loop: the agent's own failures become the input to
the next session's repair pass.

## Backup & restore

There is no `axil dump` command, and **Axil has no atomic backup
mechanism today**. All three options below require that you quiesce
writers first — otherwise the backup may capture a mixed-time view
across `memory.axil` and its companions.

1. **Quiesce + `axil branch create backup-YYYY-MM-DD`** — sequentially
   copies the main `.axil` and each companion file via `fs::copy`. No
   internal locking or snapshot, so concurrent writers can interleave
   between copies and produce an inconsistent branch. Safe when nothing
   else is writing.
2. **External tools** (rsync, restic, etc.) — copy the explicit file
   list (`memory.axil` + each companion) while no writer is active.
   See [Storage Model](../concepts/storage.md#portability--what-to-copy)
   for the file list.
3. **Snapshot filesystems** (ZFS, Btrfs) — take a filesystem snapshot
   covering the DB directory. This is the only option that's truly
   atomic without quiescing writers, because the filesystem layer
   provides the snapshot boundary Axil doesn't.

A real online-snapshot command is on the roadmap; until then prefer
option 3 if your filesystem supports it.

## `axil maintain` — opportunistic, time-gated maintenance

The daily/weekly cadence below does **not** require a wall-clock cron.
`axil maintain --if-stale` runs each task only when its cadence has
elapsed since the last run (tracked in the `_maintenance_runs` table),
so it's cheap to fire on every session start:

| Task | Cadence key (in `[maintenance]`) | Default |
|------|----------------------------------|---------|
| `snapshot` (trend metrics) | `snapshot_every` | `24h` |
| `health-report --save` | `health_report_every` | `7d` |

The brain hook fires `axil maintain --if-stale --in-background --quiet`
on the first tool call of a session, so the cadence is **automated for
agent use with no cron**. `--in-background` re-execs a detached child —
the lock at `.axil/maintain.lock` is claimed atomically (`O_CREAT|O_EXCL`)
so two concurrent fires can't double-spawn — and it never blocks the
agent; `--dry-run` prints what would run without doing it. An explicit
`axil maintain` (no `--if-stale`) runs every eligible task immediately.

**Only safe, additive tasks auto-run.** Destructive maintenance is
deliberately excluded: downsampling **purges** records past the
retention window, and `heal --reindex` clears/rebuilds indexes — both
stay explicit via `axil heal`, never fired by the hook. Disable the
opportunistic trigger entirely with `[maintenance] auto = false` (then
`--if-stale` is a no-op; explicit `axil maintain` still works).

## A recommended cadence

For a working agent memory DB:

- **Every session end**: `axil session-heal` (wire into the Stop hook)
- **Opportunistic (automatic via the brain hook)**: `axil maintain
  --if-stale` covers the daily `snapshot` and weekly `health-report
  --save` below — no cron needed when the hook is installed
- **Daily** (if you're *not* using the hook): `axil snapshot` (cron) for trend data
- **Weekly** (if you're *not* using the hook): `axil health-report --save --compare`
- **On demand**: `axil doctor` whenever something feels off; `axil heal
  --reindex` only when doctor flags drift

For a heavy-churn DB (lots of inserts/deletes):
- Lower `[healing] compact_expired_threshold` / `compact_superseded_threshold`
  so `doctor` flags cleanup pressure earlier
- Add `axil compact` to a daily cron
- Snapshot to a branch before bulk imports

For an append-only DB (experiment logs, audit trails):
- Set `[lifecycle.tables.<t>] supersede = false` + `compact = "never"`
  for the append-only tables, or `[healing] auto_compact = false` to make
  all compaction manual

## See also

- [Storage Model](../concepts/storage.md) — what each file holds
- [Indexing & Scale](./indexing-and-scale.md) — when to re-index source code
- [Branching](./branching.md) — full branch lifecycle
- [Diagnostics CLI Reference](../cli/diagnostics.md) — command flags in detail
