---
name: axil-autoagent
description: "Auto-use Axil DB for every session — recall context on start, store decisions during work, summarize on end"
---

# Axil Auto-Agent Memory

You MUST use the Axil database (`.axil/memory.axil`) as your working memory for this project. The DB auto-detects — no `--db` flag needed.

## Session Start (do this FIRST, before any work)

```bash
# Recall context relevant to the user's request
axil recall "<topic from user's message>" --top-k 5

# Check what happened recently
axil since 3d --limit 10
```

Read the results. Use them to avoid re-learning what you already know — file locations, design decisions, gotchas, past bugs.

## Axil-First Repo Workflow

Use Axil before broad repo exploration. The target behavior is that nearly every project-context lookup starts with Axil, then you open the specific files it points to.

Search/query gate: before any repo-discovery command (`rg`, `grep`, `git grep`, `find`, `fd`, `ls`, `tree`) or any "where/how/what changed" question, run the most relevant Axil command first.

| Need | Use Axil first | Use repo search after |
|------|----------------|----------------------|
| Project status, recent work, decisions | `axil boot`; `axil since 7d --limit 20`; `axil recall "<topic>" --top-k 5` | Only if memory has no answer |
| Symbol/module/API location | `axil code-search "<query>" --top-k 5` | `rg`/`grep` to verify current text |
| Exact docs/config text | `axil fts "<term>" --limit 5` | Broad scan only if FTS misses |
| Editing a known file | `axil recall-for-file "<path>" --top-k 5` | Read/edit the file normally |

Bypass Axil only for a user-named exact file/line, command output from the current turn, or a tiny local edit that needs no project context.

## During Work (do these automatically, don't ask)

### After every key decision:
```bash
axil store decisions '{"summary":"<what you decided>","reason":"<why>","files":["<affected files>"]}'
```

### After discovering a gotcha or bug:
```bash
axil store errors '{"error":"<what went wrong>","root_cause":"<why>","fix":"<how you fixed it>"}'
```

### After learning codebase structure:
```bash
axil store context '{"type":"architecture","summary":"<what you learned>","files":["<relevant files>"],"line_numbers":"<key lines>"}'
```

### After fixing a build/test error:
```bash
ERR_ID=$(axil store errors '{"error":"<error>","fix":"<fix>"}' | jq -r '.id')
```

### When you need context you don't have:
```bash
axil recall "<what you need to know>" --top-k 5
axil fts "<exact term>" --limit 5
```

## What to Store

| Store | Example |
|-------|---------|
| Architecture knowledge | File paths, key structs, how modules connect |
| Design decisions | Why you chose approach A over B |
| Gotchas | Things that broke unexpectedly, non-obvious patterns |
| Build/test fixes | Compilation errors and their solutions |
| Implementation plans | Steps you plan to take for a multi-step task |

## What NOT to Store

- File contents (that's what git is for)
- Trivial one-line changes
- Information already in CLAUDE.md
- Temporary debugging output

## Rules

1. **Always recall before starting work** — even if you think you know the codebase
2. **Store as you go** — don't batch everything at the end
3. **Be specific** — include file paths, line numbers, function names
4. **Use JSON** — all data must be valid JSON objects
5. **Keep summaries short** — under 200 chars, optimize for future recall
6. **Link related records** when a new decision supersedes an old one:
   ```bash
   axil link <new_id> "supersedes" <old_id>
   ```
