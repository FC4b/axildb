# Branching

Axil supports database branching for experimentation and rollback.

## Create a branch

```bash
axil branch create <DB> <NAME>
```

Creates a full copy of the database and all companion files at `<DB>.branch.<NAME>`.

> **Consistency:** a branch is a sequential file copy of the core database and its
> companion files (`.vec`/`.graph`/`.fts`), not a coordinated snapshot. It is
> point-in-time consistent only when the database is **quiescent** — Axil's
> single-writer model means a copy taken while another process is mid-write can
> capture the core and companion files at slightly different logical points. For a
> reliable backup, branch when no writer is active.

## List branches

```bash
axil branch list <DB>
```

## Compare changes

```bash
axil branch diff <DB> <NAME>
```

Shows per-table record count differences between main and branch.

## Switch to a branch

```bash
axil branch switch <DB> <NAME>
# Outputs: export AXIL_DB="/path/to/db.branch.name"
```

## Merge a branch

```bash
axil branch merge <DB> <NAME>
axil branch merge <DB> <NAME> --strategy main-wins
axil branch merge <DB> <NAME> --strategy keep-both
axil branch merge <DB> <NAME> --delete  # Delete branch after merge
```

### Merge strategies

| Strategy | Behavior |
|----------|----------|
| `branch-wins` | Branch record overwrites main (default) |
| `main-wins` | Main record kept, branch changes skipped |
| `keep-both` | Conflicting branch records inserted as new records |

## Delete a branch

```bash
axil branch delete <DB> <NAME>
```

## Architecture notes

- Branches are independent file copies (not git-style refs)
- All companion files (vector, graph, FTS) are copied
- Merge operates at the record level, not file level
- After merge, run `axil doctor` to verify index consistency
