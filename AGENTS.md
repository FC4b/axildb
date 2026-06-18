<!-- AXIL:BEGIN -->
# Axil Agent Memory

This repo uses Axil as persistent agent memory at `/Users/sek/Documents/GitHub/axildb/.axil/memory.axil`.

## Axil-First Rules

Use Axil before broad repo exploration. Aim for almost every project-context lookup to start with Axil, then open the cited files directly.

- Search/query gate: before any repo-discovery command (`rg`, `grep`, `git grep`, `find`, `fd`, `ls`, `tree`) or any "where/how/what changed" question, run the most relevant Axil command first.
- Start work with `axil recall "<user request>" --top-k 5`; use `axil boot` for status/history questions.
- For symbols, APIs, modules, or "where is X?", run `axil code-search "<query>" --top-k 5` before `rg` or `grep`.
- For exact text in docs/configs, run `axil fts "<term>" --limit 5` before a broad text scan.
- For recent work, decisions, or gotchas, run `axil since 7d --limit 20` or `axil recall "<topic>" --top-k 5`.
- After Axil points to files or symbols, read/edit files normally to verify current code.

Bypass Axil only for a user-named exact file/line, a command/test output you just produced, or a tiny local edit where no project context is needed.

## Write-Back Rules

- Store design choices immediately: `axil store decisions '{"summary":"<what>","reason":"<why>","files":["<path>"]}'`
- Store bugs/gotchas immediately: `axil store errors '{"error":"<what>","root_cause":"<why>","fix":"<how>"}'`
- Store architecture learned while reading: `axil store context '{"type":"architecture","summary":"<what you learned>","files":["<path>"]}'`
- Before a final response after substantive work, write a checkpoint: `axil checkpoint '{"state":"<where things stand>","next_steps":["<remaining work>"],"references":[{"kind":"file","ref":"<path>"}]}'`
<!-- AXIL:END -->


