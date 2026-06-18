# Memory Commands

Requires `--features memory`.

## know

Store a fact about an entity.

```bash
axil --db <DB> know <ENTITY> "<FACT>" [--source <SOURCE>]
axil --db ./db know auth-module "Uses JWT with 1h expiry"
```

## know-about

Get everything known about an entity — facts, related entities, consolidated summary.

```bash
axil --db <DB> know-about <ENTITY>
```

## entity-alias

Register an alias for an entity.

```bash
axil --db <DB> entity-alias <ENTITY> <ALIAS>
axil --db ./db entity-alias "Sarah" "VP of Engineering"
```

## entity-resolve

Resolve a name to its canonical entity.

```bash
axil --db <DB> entity-resolve <NAME>
axil --db <DB> entity-resolve <NAME> --fuzzy
axil --db <DB> entity-resolve <NAME> --fuzzy --strategy frequency
axil --db <DB> entity-resolve <NAME> --fuzzy --strategy context --context "JWT,login"
```

Strategies: `default`, `frequency`, `session`, `context`.

## entity-merge

Merge two entities (move facts, transfer aliases).

```bash
axil --db <DB> entity-merge <TARGET> <SOURCE>
```

## session

Manage agent sessions.

```bash
axil --db <DB> session start
axil --db <DB> session log <SESSION_ID> <TABLE> '<JSON>'
axil --db <DB> session end <SESSION_ID> --summary "What happened"
axil --db <DB> session list [--active]
```

## believe / doubt / beliefs

Manage the agent's belief system.

```bash
axil --db <DB> believe "Auth module uses JWT tokens"
axil --db <DB> doubt <BELIEF_ID>
axil --db <DB> beliefs
```

## boot

Get startup context for an agent session.

```bash
axil boot
axil boot --files src/auth.rs
axil boot --entities auth-module
```

## rule / rules

Manage agent directives and conventions, and distill recurring failures into
corrective directives. `rules` is an accepted alias for `rule`.

> Requires `--features indexer`. Note `axil rule distill` (this family) is
> unrelated to `axil learn <name> <description>` (Memory, `--features memory`),
> which stores a single *procedural* pattern with a confidence score — see the
> note at the end.

```bash
axil --db <DB> rule set <KEY> "<RULE TEXT>"   # add or update a directive
axil --db <DB> rule get <KEY>
axil --db <DB> rule list
axil --db <DB> rule delete <KEY>
axil --db <DB> rule extract [PATH]            # read conventions FROM CLAUDE.md into the DB
axil --db <DB> rule distill [OPTIONS]         # distill recurring failures INTO CLAUDE.md
```

### rule distill — failure → corrective-rule write-back

`rule extract` reads conventions *from* `CLAUDE.md` *into* memory. `rule distill`
is the opposite direction: it distills the failures you've already recorded
*back out* into corrective directives, so the next session is warned before it
repeats a mistake. (`rule learn` is kept as a hidden alias.)

**What it does**

1. Reads the `errors` table — everything written by `axil store errors` and the
   auto-capture hook.
2. Groups near-identical failures with a lexical SimHash, so case / whitespace /
   line-number variants of the same error collapse into one cluster (no model,
   no embedding).
3. For any cluster seen **≥ `--min-evidence`** times (default 2) **with a
   recorded `fix`**, synthesizes the directive *"Last N times you hit X, the fix
   was: Y."* (Y is the most recent fix.) Clusters with no recorded fix are
   skipped.
4. Ranks directives by impact (`frequency × recency`, 30-day half-life) and
   keeps the top `--max` (default 10).
5. Writes them to **two places**:
   - **`CLAUDE.md`** — inside an idempotent
     `<!-- axil:learned:start -->` … `<!-- axil:learned:end -->` block.
     Re-running replaces the block in place; anything outside the markers is
     never touched; when a failure stops recurring its directive is removed.
     Target is `--file` (default `CLAUDE.md`); delete the markers to disable.
   - **The pinned `rules` table** — so `axil boot` echoes the corrections under
     "## Rules (pinned — always apply)" even without reading the CLAUDE.md edit.

**Options**

| Flag | Default | Meaning |
|------|---------|---------|
| `--dry-run` | off | Preview the directives + block; write nothing (no file, no DB change) |
| `--file <PATH>` | `CLAUDE.md` | Target file for the managed block |
| `--min-evidence <N>` | `2` | Minimum occurrences before a failure earns a directive |
| `--max <N>` | `10` | Cap on how many directives are emitted |

```bash
axil --db <DB> rule distill --dry-run               # preview only
axil --db <DB> rule distill                          # apply to ./CLAUDE.md + pin in boot
axil --db <DB> rule distill --min-evidence 3 --max 5 # stricter, smaller block
```

**When to run it** — deliberately and periodically, once the same class of
error has bitten more than once and you've recorded the fixes. Preview with
`--dry-run` first.

**It is not automatic.** Nothing runs `rule distill` on your behalf — no hook, no
background worker. It only touches `CLAUDE.md` when you invoke it explicitly.
(Auto-editing a tracked `CLAUDE.md` on every tool call would be invasive, so it
is off by design. Wiring the PostToolUse brain hook to fire a `--dry-run` — or a
quiet apply when the `errors` table grows — is a possible future opt-in.)

> **Don't confuse with `axil learn`.** This command was renamed from `rule
> learn` to `rule distill` precisely because the old name echoed the unrelated
> top-level `axil learn <name> <description>` (Memory, `memory` feature), which
> stores one *procedural pattern* with a confidence score. `rule distill`
> distills failures into *rules*; `axil learn` records a how-to. (`rule learn`
> still works as a hidden alias for `rule distill`.)
