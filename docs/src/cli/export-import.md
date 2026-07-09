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

`import` prints a JSON report:

```json
{"dry_run":false,"imported":12,"skipped_id":0,"skipped_dup":0,"edges_created":4,"edges_skipped":0,"id_remapped":0}
```

### `--dedup`

Skips a record when **either**:

- its id already exists in the destination (`skipped_id`), or
- a record with the same content already exists in the same table
  (`skipped_dup`) — matched by a hash of the record's data with volatile
  internal fields (e.g. importance score) removed, so the same memory matches
  even when two machines scored it differently.

Duplicate edges (same `from`/type/`to`) are skipped too, so re-importing the same
file is a no-op.

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
