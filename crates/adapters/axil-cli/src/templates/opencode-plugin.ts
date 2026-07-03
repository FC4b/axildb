// Axil brain — OpenCode plugin. Written by `axil install --opencode`;
// refreshed by `axil sync`. Do not edit in place: changes are overwritten.
//
// A pure event adapter: OpenCode events are mapped onto the Claude-style
// hook contract and piped to `axil hook run --dialect claude` — the same
// brain that serves Claude Code, Codex, Copilot, Droid, Antigravity, and
// Qwen. The brain's responses (context injections, stop-guard reasons)
// are applied through OpenCode's documented mutation points:
//
//   boot / recall      → queued, then pushed as a text part in-place at
//                        `chat.message` (and appended to tool output at
//                        `tool.execute.after`) — whichever fires first
//   session close      → `session.idle` forwards a Stop event; the brain
//                        runs the narrative guard + worker/heal/session
//                        record; a guard reason is queued as a nudge
//   compaction         → `experimental.session.compacting` injects the
//                        boot context so resume state survives compaction
//
// Self-contained: no npm dependencies (loaded straight from
// .opencode/plugins/ by OpenCode's Bun runtime; node:child_process is
// built in). Every path is best-effort — a memory plugin must never
// break the agent loop.

import { spawnSync } from "node:child_process"

// Resolved at call time so AXIL_BIN can be set by the host after load.
// Note the binary must be new enough to have `axil hook run` — run
// `axil sync` after upgrading a global install.
const axilBin = () => process.env.AXIL_BIN || "axil"

/** OpenCode tool ids → the Claude-contract tool names the brain expects. */
function mapTool(tool: string, args: any): { name: string; input: any } | null {
  switch (tool) {
    case "bash":
      return { name: "Bash", input: { command: args?.command ?? "" } }
    case "edit":
      return {
        name: "Edit",
        input: { file_path: args?.filePath ?? "", new_string: args?.newString ?? "" },
      }
    case "write":
      return {
        name: "Write",
        input: { file_path: args?.filePath ?? "", content: args?.content ?? "" },
      }
    case "read":
      return { name: "Read", input: { file_path: args?.filePath ?? "" } }
    case "todowrite":
      return { name: "TodoWrite", input: { todos: args?.todos ?? [] } }
    default:
      return null // unmapped tools are invisible to the brain
  }
}

export const AxilPlugin = async ({ directory }: any) => {
  const dir: string = directory || process.cwd()
  /** Context queued by the brain, per session, awaiting an injection point. */
  const pending = new Map<string, string[]>()
  const booted = new Set<string>()
  const lastIdle = new Map<string, number>()

  /** Pipe one Claude-shaped hook event to the brain; collect its response. */
  function brain(event: any): void {
    try {
      const res = spawnSync(axilBin(), ["hook", "run", "--dialect", "claude"], {
        cwd: dir,
        input: JSON.stringify(event),
        encoding: "utf8",
        timeout: 15_000,
        env: { ...process.env, CLAUDE_PROJECT_DIR: dir },
      })
      const stdout = (res.stdout || "").trim()
      if (!stdout) return
      const sid = event.session_id
      const queue = pending.get(sid) ?? []
      try {
        const parsed = JSON.parse(stdout)
        const ctx = parsed?.hookSpecificOutput?.additionalContext
        if (typeof ctx === "string" && ctx) queue.push(ctx)
        // The brain's stop guard: we cannot veto an OpenCode turn ending,
        // so the reason becomes a nudge on the next injection point.
        if (parsed?.decision === "block" && typeof parsed?.reason === "string") {
          queue.push(parsed.reason)
        }
      } catch {
        // UserPromptSubmit replies with raw context text, not JSON.
        queue.push(stdout)
      }
      if (queue.length) pending.set(sid, queue)
    } catch {
      // Never break the agent loop.
    }
  }

  function drain(sid: string): string {
    const queue = pending.get(sid)
    if (!queue || queue.length === 0) return ""
    pending.delete(sid)
    return queue.join("\n\n")
  }

  function base(sid: string, name: string): any {
    return { hook_event_name: name, session_id: sid, cwd: dir }
  }

  return {
    event: async ({ event }: any) => {
      // Session end-of-turn: forward Stop so the brain runs the narrative
      // guard and closes the session (record, worker, beliefs, heal).
      // `session.idle` is deprecated in favor of `session.status`; handle
      // both with a short dedupe window.
      let sid: string | undefined
      if (event?.type === "session.idle") sid = event?.properties?.sessionID
      if (event?.type === "session.status" && event?.properties?.status?.type === "idle") {
        sid = event?.properties?.sessionID
      }
      if (!sid) return
      const now = Date.now()
      if (now - (lastIdle.get(sid) ?? 0) < 5_000) return
      lastIdle.set(sid, now)
      brain({ ...base(sid, "Stop"), stop_hook_active: false })
    },

    "chat.message": async (input: any, output: any) => {
      const sid: string = input?.sessionID ?? ""
      if (!sid) return

      // First message of a session → boot. SessionStart makes the brain
      // emit the boot as additionalContext, which lands in the queue.
      if (!booted.has(sid)) {
        booted.add(sid)
        brain({ ...base(sid, "SessionStart"), source: "startup" })
      }

      // Prompt-time recall: forward the user's text.
      const prompt = (output?.parts ?? [])
        .filter((p: any) => p?.type === "text" && typeof p?.text === "string")
        .map((p: any) => p.text)
        .join("\n")
      if (prompt) brain({ ...base(sid, "UserPromptSubmit"), prompt })

      // Inject everything queued — mutate the parts array IN PLACE (the
      // caller keeps its own reference; reassigning does nothing).
      const ctx = drain(sid)
      if (ctx) output.parts.push({ type: "text", text: ctx })
    },

    "tool.execute.before": async (input: any, output: any) => {
      const mapped = mapTool(input?.tool ?? "", output?.args)
      if (!mapped) return
      brain({
        ...base(input.sessionID, "PreToolUse"),
        tool_name: mapped.name,
        tool_input: mapped.input,
      })
    },

    "tool.execute.after": async (input: any, output: any) => {
      // Args ride on `input` here (they were `output.args` in `before`).
      const mapped = mapTool(input?.tool ?? "", input?.args)
      if (mapped) {
        const exit = typeof output?.metadata?.exit === "number" ? output.metadata.exit : 0
        brain({
          ...base(input.sessionID, "PostToolUse"),
          tool_name: mapped.name,
          tool_input: mapped.input,
          tool_response: { exitCode: exit, output: String(output?.output ?? "").slice(0, 4000) },
        })
      }
      // Flush queued context by appending to the tool result the model
      // reads — the earliest injection point after a tool call.
      const ctx = drain(input?.sessionID ?? "")
      if (ctx) output.output = `${output.output ?? ""}\n\n${ctx}`
    },

    "experimental.session.compacting": async (input: any, output: any) => {
      // Resume state must survive compaction: inject the boot narrative
      // (recent decisions, checkpoint, errors) into the summary context.
      try {
        const res = spawnSync(
          axilBin(),
          ["--db", `${dir}/.axil/memory.axil`, "boot", "--boot-format", "narrative", "--budget", "800"],
          { cwd: dir, encoding: "utf8", timeout: 10_000 },
        )
        const boot = (res.stdout || "").trim()
        if (res.status === 0 && boot) output.context.push(boot)
      } catch {
        // best-effort
      }
    },
  }
}
