# Terminal Agents

Axil's Claude Code integration — the auto-boot / auto-capture / stop-guard
memory loop — is not Claude-specific. The same loop runs under **six other
terminal coding agents**, all served by one binary.

| Agent | Vendor | `axil install` flag |
|-------|--------|---------------------|
| Claude Code | Anthropic | `--claude-code` |
| OpenAI Codex | OpenAI | `--codex` |
| GitHub Copilot CLI | GitHub | `--copilot` |
| Factory Droid | Factory | `--droid` |
| Google Antigravity CLI | Google | `--antigravity` |
| Qwen Code | Alibaba | `--qwen` |
| OpenCode | Anomaly | `--opencode` |

## One brain, six dialects

Every integration wires the agent's lifecycle hooks to a single command —
`axil hook run --dialect <agent>`. The brain lives *in the binary* (no bash,
no jq, works natively on Windows); each `--dialect` is a thin field mapping
between that agent's hook JSON and one shared cognitive loop:

```
session start  → axil boot injected as context
before search  → axil-first gate (code-search / fts before rg/grep)
before edit    → recall-for-file surfaces past memories
after edit     → file manifest captured
after shell    → heartbeat, git-commit capture, failed-command capture
stop           → narrative guard, then session close (worker, beliefs, heal)
```

Because it's one loop, a memory stored under Codex is recalled under Claude
Code, Droid, or any other — **the `.axil` database is shared across every
agent you point at the project.**

## Install

Run bare for an interactive wizard that detects the agent tooling already in
the project and pre-selects it:

```bash
axil install
```

Or name agents explicitly (skips the wizard — the mode scripts and CI use):

```bash
axil install --codex --copilot          # two agents
axil install --all                       # every detected + supported agent
```

Each agent install writes its hook config, registers the MCP server, and —
for agents that read them — the cross-tool skills. The shared **AGENTS.md
contract** is written by default (opt out with `--no-agents-md`).

## What each agent gets

| Agent | Hooks | MCP | Skills / rules |
|-------|-------|-----|----------------|
| **codex** | `.codex/hooks.json` | project `.codex/config.toml` `[mcp_servers.axil]` | `.agents/skills/<name>/SKILL.md` |
| **copilot** | `.github/hooks/axil.json` | user `~/.copilot/mcp-config.json` | — |
| **droid** | `.factory/hooks.json` | project `.factory/mcp.json` | — |
| **antigravity** | `.agents/hooks.json` (key `axil-brain`) | project `.agents/mcp_config.json` | `.agents/rules/axil.md` + `.agents/skills/` |
| **qwen** | `.qwen/settings.json` `hooks` | `.qwen/settings.json` `mcpServers` | `context.fileName` gains `AGENTS.md` |
| **opencode** | `.opencode/plugins/axil.ts` (local plugin) | `opencode.json` `mcp.axil` | reads `AGENTS.md` / `CLAUDE.md` |

All writers **merge** — they touch only Axil's own entries and preserve
everything else in a shared config. Re-running is idempotent.

## Per-agent notes

**Codex** runs project hooks only after you *trust the project* **and** trust
the hook definitions — run `/hooks` inside Codex once after installing. Its
sandbox (`workspace-write`) blocks writes outside the repo, but the default DB
at `<repo>/.axil/memory.axil` is inside the workspace, so stores work out of
the box. For a global DB, add it to `sandbox_workspace_write.writable_roots`.

**Copilot CLI** hooks in `.github/hooks/axil.json` are **also loaded by the
Copilot cloud agent** from the cloned repo, so you get partial cloud coverage
for free. The MCP entry is per-user and pins no `--db` — it auto-detects
`.axil/memory.axil` from the launch directory, so one entry serves every
project.

**Antigravity CLI** rewrote the Gemini hook contract: it has no session-start
event, so boot rides the first `PreInvocation` (before the first model call)
and context queued by tool hooks is flushed there as an `injectSteps` message.
Its file-edit argument keys are inferred from docs — if edit capture looks
empty, see [Verifying dialects](#verifying-dialects).

**Qwen Code** ships its own LLM-driven auto-memory. To avoid double-capture,
set `memory.enableManagedAutoMemory = false` (and `enableManagedAutoDream`,
`enableAutoSkill`) in `~/.qwen/settings.json` so Axil is the single memory
layer.

**OpenCode** loads the plugin straight from `.opencode/plugins/axil.ts` — a
self-contained file embedded in the binary, **no npm package required**. It
forwards OpenCode's events to `axil hook run --dialect claude` and even injects
resume state into OpenCode's compaction summary, so memory survives context
compaction.

## MCP registration on its own

The hook loop is the primary integration; the MCP server is a structured-tool
fallback. To register it without the full install (or for an agent not listed
above):

```bash
axil mcp install <target>
# target: claude-code | cursor | windsurf | codex | copilot
#         | droid | qwen | antigravity | opencode
```

## Keeping it fresh & removing it

```bash
axil sync                 # refresh whatever is installed (auto-detect)
axil sync --codex --qwen  # refresh specific agents (works without a version bump)
axil install --uninstall  # remove ALL agent integrations — hooks, MCP entries,
                          # managed AGENTS.md/CONVENTIONS.md blocks, owned files
                          # (the database is always preserved)
```

## Verifying dialects

A few agents' hook payload shapes are inferred from docs rather than observed.
If an agent's edit/shell capture looks empty, run the built-in probe to record
exactly what that agent sends, then compare against the mapping:

```bash
# writes each raw hook payload to .axil/hook-capture.jsonl
axil hook capture --dialect <agent>
```

Wire it as the agent's hook command temporarily (or point one event at it),
drive a normal session, then inspect `.axil/hook-capture.jsonl`. The fields you
care about are the tool name and its arguments (shell command, edited file
path). File an issue if a mapping is off — the fix is a one-line field rename in
`hook_brain.rs`.
