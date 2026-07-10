# Hook dialect fixtures

Golden regression fixtures for the agent hook **dialects** in
[`hook_brain.rs`](../../../src/hook_brain.rs). Each `<dialect>.json` pairs
representative hook payloads with the canonical parse they must produce; the
test `hook_brain::tests::dialect_fixtures_parse_as_expected` feeds every
`payload` through the dialect parser and asserts the result equals `expect`.

They exist because several dialects' payload shapes were **inferred from
vendor docs rather than observed** (Antigravity's `toolCall.args` keys, Codex's
`tool_response`, Copilot's failure shape). These fixtures lock the mappings so
a tool's format change — or a regression in our parser — fails a fast, offline
unit test instead of silently degrading memory capture in the field.

## Format

```jsonc
{
  "dialect": "codex",
  "cases": [
    {
      "name": "human-readable case name",
      "event": null,           // --event override; null unless the dialect needs it (antigravity does)
      "payload": { /* the raw stdin the agent sends the hook */ },
      "expect": {
        "event": "PreTool",    // canonical EventKind (Debug form), or null if unparseable
        "tool":  { "kind": "shell", "command": "…", "exit_code": 0 }  // or null when the event has no tool
      }
    }
  ]
}
```

The `expect` object is **exactly** the `parsed` block that `axil hook capture`
writes — same `tool_summary` shape — so confirming a mapping against a real
session and locking it are the same step.

## Adding a real captured payload

1. Wire the probe as the agent's hook command temporarily:
   `axil hook capture --dialect <dialect>` (see the
   [Terminal Agents](../../../../../../docs/src/agents/terminal-agents.md) guide).
2. Drive a normal session, then open `.axil/hook-capture.jsonl`.
3. For any line where `parsed.tool` is `"other"` next to a `raw` payload that
   clearly names a file edit or command, the mapping is wrong — fix the field
   name in `hook_brain.rs`.
4. Once correct, copy that line's `raw` into a new fixture case's `payload` and
   its `parsed` into `expect`. The mapping is now regression-locked.

Keep at least ~20 cases across dialects; the test asserts a floor so a
truncated fixture set can't pass silently.
