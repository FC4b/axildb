# Export & Import (Portable Memory)

`axil export` and `axil import` move memory between databases as **mergeable
JSONL** — one JSON object per line, safe to commit to git or drop in a shared
folder. This is the portable path for taking one developer's memory to another
machine, or sharing curated context with a teammate.

> A dedicated team-sync product is coming separately. Export/import is the
> local-first, no-server way to share memory today.

## How this differs from `branch` / `snapshot`

They solve different problems, and complement each other:

| | `axil export` / `import` | `axil branch` (and legacy `snapshot`) |
|---|---|---|
| Unit | Individual records + edges (JSONL) | The whole database, binary |
| On the destination | **Merged into** an existing DB | **Restored over** a DB (a clone) |
| Embeddings | Excluded; rebuilt on import | Copied verbatim (`.vec` file) |
| Cross-machine | Yes — text, diffs in git | Copy is machine-specific |
| Use it for | Sharing/combining memory | Point-in-time backup, rollback |

Use a **branch** when you want a faithful, restorable copy of everything. Use
**export/import** when you want to *merge* records from one memory into another.

## export

```bash
axil export                          # JSONL to stdout
axil export --out memory.jsonl       # ...or to a file
axil export --tables decisions,errors
axil export --since 2026-01-01T00:00:00Z
axil export --include-system         # include `_`-prefixed tables
```

The first line is a header describing the format version and record/edge counts.
Record lines follow, then edge lines for graph edges **between exported
records**. A summary (`exported N record(s), M edge(s)…`) is printed to stderr,
so piping the JSONL to a file stays a clean stream.

**What travels:** user memory tables (`decisions`, `errors`, `context`,
`sessions`, …) and the graph edges among them.

**What does not travel:**

- **Embedding vectors** — they are machine-local ONNX artifacts and are
  regenerated on import. Full-text and code-reference indexes are likewise
  rebuilt when each record is re-inserted.
- **Rebuilt index tables** (`_idx_*`) — derived data, always excluded.

By default only user tables are exported. `--include-system` opts in the
`_`-prefixed system tables (entities, checkpoints, dependency docs, …) where
re-import is safe; `_idx_*` index tables are excluded even then.

### Line format

```json
{"kind":"header","format":"axil-export-jsonl","format_version":1,"axil_version":"…","record_count":3,"edge_count":1}
{"kind":"record","table":"decisions","id":"01J…","data":{…},"created_at":"…","updated_at":"…"}
{"kind":"edge","from":"01J…","edge_type":"modified","to":"01J…","props":{}}
```

Output is deterministic — records and edges are emitted in id order — so a
re-export of an unchanged database produces a byte-identical file that diffs
cleanly in git.

## import

```bash
axil import memory.jsonl
axil import memory.jsonl --dedup      # skip records that already exist
axil import memory.jsonl --dry-run    # report without writing
cat memory.jsonl | axil import -      # read from stdin
```

Records are recreated through the **normal insert path**, so every engine fires:
each record is re-embedded (if the destination has an embedder), FTS-indexed, and
its code references re-linked. **Original ids are preserved**, so checkpoint
`references[]` and code-reference pointers keep resolving after the trip. Edges
are recreated between records whose endpoints resolved; an edge with a missing
endpoint is skipped rather than left dangling.

Without `--dedup`, importing a record whose id already exists **overwrites that
record in place** (storage is keyed by id) — the engines re-fire on the imported
content. Each overwrite is counted as `overwritten` in the report (never hidden
inside `imported`) and echoed as a warning. Use `--dedup` whenever the
destination may already contain any of the exported records; overwrite-by-id is
only the right tool when you intend the import to win.

An imported record can also **supersede** an existing near-duplicate (same
auto-supersede rule as a normal insert), but only when the incoming record's
`created_at` is at least as new — an older export can never demote fresher
local memory. The report counts these as `superseded`.

`import` prints a JSON report:

```json
{"dry_run":false,"interrupted":false,"imported":12,"overwritten":0,"skipped_id":0,"skipped_dup":0,
 "superseded":0,"edges_created":4,"edges_skipped":0,"edges_remapped":0,"id_remapped":0,
 "embeddings":{"status":"verified","expected":12,"indexed":12,"missing":0}}
```

The export header must be the first line of the stream — a truncated or
headerless file is rejected **before anything is written**. Past the header,
import is fail-fast with partial state: if it stops mid-stream (malformed line,
insert error), everything before the failure is already committed, and the
report is still printed with `"interrupted":true` so the counts show exactly
what landed.

### Self-verifying embeddings

Auto-embedding is best-effort by design — an embedder failure mid-import must
never lose the record — so an import *can* finish with records stored but not
semantically searchable. Instead of leaving that gap silent, the report's
`embeddings` block verifies the index after the fact:

- `"status":"verified"` with `"missing":0` — every embeddable record has a
  vector; nothing to do.
- `"missing" > 0` — the embedder was attached but failed for those records
  (e.g. a broken ONNX runtime). They are stored and full-text-searchable, but
  invisible to semantic recall until you run `axil heal --reindex`.
- `"status":"engine_unavailable"` — no vector engine or embedder was attached
  at import time; run `axil heal --reindex` once embeddings are available.

The CLI prints a warning with that exact fix whenever the check finds a gap,
so no post-import ritual is needed: import, and act only if it tells you to.
(`axil doctor` performs the same drift detection database-wide.)

### `--dedup`

Skips a record when **either**:

- its id already exists in the destination (`skipped_id`), or
- a record with the same content already exists in the same table
  (`skipped_dup`) — matched by a hash of the record's data with volatile
  internal fields (e.g. importance score) removed, so the same memory matches
  even when two machines scored it differently.

Edges that referenced a content-deduped duplicate are reattached to the
surviving record (`edges_remapped`) instead of being dropped, so the imported
copy's graph context lands on the record that won. Duplicate edges (same
`from`/type/`to`) are always skipped — with or without `--dedup` — so
re-importing the same file never doubles an edge.

## Team workflow

```bash
# On your machine: capture the memory worth sharing and commit it.
axil export --tables decisions,errors --out team-memory.jsonl
git add team-memory.jsonl && git commit -m "share auth-migration memory"

# On a teammate's machine, after pulling:
axil import team-memory.jsonl --dedup
```

`--dedup` makes the import idempotent: a teammate can import repeatedly (or
import files that overlap) and only genuinely new records land. Because ids are
preserved, everyone's copy of a given decision shares the same id, so graph edges
and checkpoint references line up across machines.
