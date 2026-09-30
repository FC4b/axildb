//! Axil Brain — agent lifecycle hook runtime.
//!
//! Rust port of the former `.claude/hooks/axil-brain.sh` and
//! `store-on-task-complete.sh`. Living inside the binary removes the
//! bash + jq dependency, runs natively on Windows, and gives the hook
//! logic real unit tests. The CLI entry is `axil hook run --dialect <d>`;
//! the agent harness pipes the event JSON to stdin and reads the
//! dialect's response JSON (or injected context text) from stdout.
//!
//! A *dialect* is the JSON contract one agent family speaks. The brain
//! normalizes every dialect into canonical events (`EventKind`) and tool
//! actions (`ToolAction`), runs one shared cognitive loop over them, and
//! emits responses back in the dialect's own shape. `claude` covers
//! Claude Code; `codex`, `copilot`, and `droid` speak the same
//! shell-hook style with different field spellings; the Gemini-lineage
//! tools (Antigravity CLI, Qwen Code) arrive with their wave.
//!
//! The shared loop:
//!   user prompt      — inject a <context> block from recall
//!   session start    — boot push, and queue the close of this project's
//!                      sessions that went quiet without a session end
//!                      (installs without a session-start event emulate it
//!                      on the first pre-tool call via a sentinel)
//!   pre file-edit    — recall-for-file + store nudge
//!   pre shell        — axil-first search gate / paired search
//!   post file-edit   — manifest + snippet accumulation
//!   post shell       — heartbeat, commit capture, error capture
//!   post file-read   — fallback capture after empty recalls
//!   post todo-update — store reminder when a task completes
//!   stop             — narrative guard. Stop fires every turn, so it keeps
//!                      the session's state; dialects with no session-end
//!                      event also queue the session close here
//!   session end      — queue the session close, clean up
//!
//! Hooks never write to the database themselves. Each write becomes a job
//! file in `<db dir>/hook-queue/` (the intent log) and one detached
//! `axil hook drain` runs them, so a harness timeout can never kill a write
//! mid-commit. Lookups run with the slow-query log off for the same reason.
//!
//! Every path is best-effort: a memory hook must never break the agent
//! loop, so child failures are swallowed and the process always exits 0
//! once the input parses.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A session whose state files have been idle this long ended without a
/// session-end event (closed terminal, crash); the next session start in the
/// same project queues its close.
const STALE_SESSION_SECS: u64 = 6 * 3600;

/// Narrative tables that satisfy the Stop guard. If you add a new
/// narrative table, update both constants (list + human text).
const NARRATIVE_TABLES: &[&str] = &[
    "decisions",
    "errors",
    "context",
    "commits",
    "_checkpoint_records",
];
const NARRATIVE_TABLES_TEXT: &str = "decisions/errors/context/commits/checkpoint";

/// Cap the problems file so a runaway session (hundreds of empty recalls)
/// can't bloat the queue and the eventual session-heal load.
const PROBLEMS_MAX_BYTES: u64 = 262_144;

/// Cap the `axil hook capture` debug log — plenty for a probe session,
/// bounded if someone leaves it wired.
const CAPTURE_MAX_BYTES: u64 = 4_194_304;

// ── Dialect layer ────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dialect {
    Claude,
    Codex,
    Copilot,
    Droid,
    /// The legacy Gemini CLI contract (settings.json hooks, snake_case
    /// stdin, hookSpecificOutput.additionalContext) — spoken by Qwen Code,
    /// which forked Gemini CLI before Google's Antigravity rewrite.
    Gemini,
    /// Antigravity CLI (`agy`) rewrote the contract: 5 events, camelCase
    /// stdin (`toolCall.args`), context via PreInvocation `injectSteps`,
    /// no session-start event, Stop blocks with `decision: "continue"`.
    Antigravity,
}

impl Dialect {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "copilot" => Some(Self::Copilot),
            "droid" => Some(Self::Droid),
            "gemini" | "qwen" => Some(Self::Gemini),
            "antigravity" => Some(Self::Antigravity),
            _ => None,
        }
    }
}

/// Canonical lifecycle events the brain reasons about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EventKind {
    UserPrompt,
    SessionStart,
    PreTool,
    PostTool,
    Stop,
    SessionEnd,
    /// Fires before every model call (Antigravity's PreInvocation) — the
    /// only context-injection channel that dialect has. Carries the boot
    /// on first fire and flushes queued context after that.
    PreModel,
}

/// Canonical tool actions, extracted from each dialect's tool payloads.
#[derive(Debug, PartialEq)]
enum ToolAction {
    FileEdit {
        path: String,
        /// New content/patch snippet, when the dialect exposes it.
        snippet: Option<String>,
    },
    FileRead {
        path: String,
        offset: i64,
        limit: i64,
    },
    Shell {
        command: String,
        /// Post-tool only.
        exit_code: i64,
        stdout: String,
        stderr: String,
    },
    Todo {
        completed_count: i64,
    },
    /// One task marked completed (Claude Code's `TaskUpdate`), identified so
    /// the reminder fires once per task.
    TaskCompleted {
        task_id: String,
    },
    Other,
}

/// One parsed hook invocation, dialect-agnostic.
struct HookEvent {
    kind: EventKind,
    session_id: String,
    cwd: Option<String>,
    prompt: Option<String>,
    /// Harness already forced a continue for this stop — don't block again.
    stop_hook_active: bool,
    tool: Option<ToolAction>,
}

const SUPPORTED_DIALECTS: &str = "claude, codex, copilot, droid, antigravity, qwen";

/// Parse the dialect string or fail loudly — a bad value is a wiring
/// mistake in someone's hook config, not a runtime input to swallow.
fn parse_dialect(dialect: &str) -> Result<Dialect> {
    Dialect::parse(dialect).ok_or_else(|| {
        anyhow::anyhow!("unknown hook dialect '{dialect}' (supported: {SUPPORTED_DIALECTS})")
    })
}

/// Map one dialect's stdin JSON onto the canonical [`HookEvent`].
fn parse_event(dialect: Dialect, input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    match dialect {
        Dialect::Claude => parse_claude(input, event_override),
        Dialect::Codex => parse_codex(input, event_override),
        Dialect::Copilot => parse_copilot(input, event_override),
        Dialect::Droid => parse_droid(input, event_override),
        Dialect::Gemini => parse_gemini(input, event_override),
        Dialect::Antigravity => parse_antigravity(input, event_override),
    }
}

/// Read the hook payload from stdin, tolerant of agents that send the JSON
/// but DON'T close stdin (observed with Antigravity's `agy`: a plain
/// `read_to_string` there blocks forever waiting for an EOF that never
/// comes, hanging the hook and the whole agent turn). Strategy: read in a
/// detached thread, and return as soon as the buffer parses as complete
/// JSON — no EOF required. Falls back to whatever arrived if EOF or a hard
/// deadline comes first. Returns None on empty input.
fn read_hook_stdin() -> Option<String> {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    // Detached reader: pushes each chunk as it arrives. If stdin never
    // closes, this thread blocks in read() forever — harmless, the process
    // exits when main returns regardless of live threads.
    std::thread::spawn(move || {
        use std::io::Read as _;
        let mut stdin = std::io::stdin();
        let mut chunk = [0u8; 8192];
        loop {
            match stdin.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if tx.send(chunk[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut buf: Vec<u8> = Vec::new();
    loop {
        // As soon as what we have is valid JSON, we're done — this is the
        // fast path when the agent sent one complete object and stalled.
        if !buf.is_empty() {
            if let Ok(s) = std::str::from_utf8(&buf) {
                if serde_json::from_str::<Value>(s.trim()).is_ok() {
                    return Some(s.to_string());
                }
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining.min(Duration::from_millis(250))) {
            Ok(chunk) => buf.extend_from_slice(&chunk),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break, // EOF
        }
    }
    let s = String::from_utf8_lossy(&buf).into_owned();
    if s.trim().is_empty() {
        None
    } else {
        Some(s)
    }
}

pub(crate) fn run(dialect: &str, event_override: Option<&str>) -> Result<i32> {
    let dialect = parse_dialect(dialect)?;

    let Some(raw) = read_hook_stdin() else {
        return Ok(0);
    };
    let input: Value = match serde_json::from_str(raw.trim()) {
        Ok(v) => v,
        Err(_) => return Ok(0),
    };

    let Some(event) = parse_event(dialect, &input, event_override) else {
        return Ok(0);
    };
    let Some(ctx) = HookCtx::new(dialect, event) else {
        return Ok(0);
    };
    // Never propagate internal errors to the agent loop: report and exit 0.
    if let Err(e) = ctx.dispatch() {
        eprintln!("[axil hook] warn: {e}");
    }
    Ok(0)
}

/// A one-line summary of what the dialect parser extracted from a payload —
/// the debug view for the `capture` probe. A tool that comes out as `other`
/// next to a raw payload that clearly names a file edit or command is a
/// mapping miss to fix in this module.
fn tool_summary(tool: &ToolAction) -> Value {
    match tool {
        ToolAction::FileEdit { path, .. } => json!({"kind": "file_edit", "path": path}),
        ToolAction::FileRead { path, offset, limit } => {
            json!({"kind": "file_read", "path": path, "offset": offset, "limit": limit})
        }
        ToolAction::Shell {
            command,
            exit_code,
            ..
        } => json!({"kind": "shell", "command": command, "exit_code": exit_code}),
        ToolAction::Todo { completed_count } => {
            json!({"kind": "todo", "completed_count": completed_count})
        }
        ToolAction::TaskCompleted { task_id } => {
            json!({"kind": "task_completed", "task_id": task_id})
        }
        ToolAction::Other => json!({"kind": "other"}),
    }
}

/// `axil hook capture --dialect <d>` — a debugging probe. Records the raw
/// hook payload AND what the dialect parser understood from it to
/// `.axil/hook-capture.jsonl`, then runs the normal loop so the session
/// still functions while you record. Wire it as an agent's hook command
/// temporarily, drive a session, then inspect the file to confirm (or
/// correct) a dialect's field mappings against what the agent really sends.
pub(crate) fn capture(dialect: &str, event_override: Option<&str>) -> Result<i32> {
    let d = parse_dialect(dialect)?;

    let Some(raw) = read_hook_stdin() else {
        return Ok(0);
    };
    let trimmed = raw.trim();
    // Keep the parse result and the original text separately: a payload that
    // ISN'T valid JSON is the most debug-worthy case, so it must reach the
    // log verbatim rather than collapsing to null.
    let parsed_json: Option<Value> = serde_json::from_str(trimmed).ok();
    let input = parsed_json.clone().unwrap_or(Value::Null);
    let event = parse_event(d, &input, event_override);

    // Resolve the project dir the same way HookCtx does, so the capture log
    // lands in the same `.axil/` the loop uses.
    let cwd_hint = event.as_ref().and_then(|e| e.cwd.clone());
    let project_dir = project_dir_env_var(d)
        .and_then(std::env::var_os)
        .map(PathBuf::from)
        .or_else(|| cwd_hint.map(PathBuf::from))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let parsed = event.as_ref().map(|e| {
        json!({
            "event": format!("{:?}", e.kind),
            "session_id": e.session_id,
            "cwd": e.cwd,
            "tool": e.tool.as_ref().map(tool_summary),
        })
    });
    let record = json!({
        "at": now_iso(),
        "dialect": dialect,
        "event_override": event_override,
        "parsed": parsed,          // what the brain understood
        // What the agent actually sent — the parsed JSON, or the raw text
        // when it wasn't JSON (exactly the case worth capturing).
        "raw": parsed_json.unwrap_or_else(|| json!(trimmed)),
    });

    // Cap the log like the problems file — a probe left wired on Edit/Write
    // otherwise grows unbounded (it lives in .axil/, not the swept tmp dir).
    let axil_dir = project_dir.join(".axil");
    let _ = std::fs::create_dir_all(&axil_dir);
    let cap_path = axil_dir.join("hook-capture.jsonl");
    let over_cap = std::fs::metadata(&cap_path)
        .map(|m| m.len() >= CAPTURE_MAX_BYTES)
        .unwrap_or(false);
    if !over_cap {
        append_line(&cap_path, &record.to_string());
    }

    // Still run the real loop so wiring `capture` doesn't break the session.
    if let Some(event) = event {
        if let Some(ctx) = HookCtx::new(d, event) {
            let _ = ctx.dispatch();
        }
    }
    Ok(0)
}

// ── Dialect parsers ──────────────────────────────────────────────────
// Each maps one agent's stdin JSON onto the canonical HookEvent. Field
// spellings differ per tool; the loop itself never looks at raw JSON.

/// Claude Code: `hook_event_name` / `tool_name` / `tool_input` /
/// `tool_response`, snake_case fields, no session-start event.
fn parse_claude(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let raw_event = event_override
        .map(str::to_string)
        .or_else(|| str_field(input, "hook_event_name"))?;
    let kind = match raw_event.as_str() {
        "UserPromptSubmit" => EventKind::UserPrompt,
        "SessionStart" => EventKind::SessionStart,
        "PreToolUse" => EventKind::PreTool,
        "PostToolUse" | "PostToolUseFailure" => EventKind::PostTool,
        "Stop" => EventKind::Stop,
        "SessionEnd" => EventKind::SessionEnd,
        _ => return None,
    };
    let tool_name = str_field(input, "tool_name").unwrap_or_default();
    let tool = match kind {
        // A failed tool call arrives as its own event, with the failure text
        // in `tool_error` instead of a `tool_response`.
        EventKind::PostTool if raw_event == "PostToolUseFailure" => {
            Some(claude_failed_tool_action(input, &tool_name))
        }
        EventKind::PreTool | EventKind::PostTool => Some(claude_tool_action(input, &tool_name)),
        _ => None,
    };
    Some(HookEvent {
        kind,
        session_id: str_field(input, "session_id")?,
        cwd: str_field(input, "cwd"),
        prompt: str_field(input, "prompt"),
        stop_hook_active: input
            .get("stop_hook_active")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        tool,
    })
}

fn claude_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    match tool_name {
        "Edit" | "Write" => {
            let Some(path) = nested_str(input, &["tool_input", "file_path"]) else {
                return ToolAction::Other;
            };
            let snippet_field = if tool_name == "Edit" {
                "new_string"
            } else {
                "content"
            };
            ToolAction::FileEdit {
                path,
                snippet: nested_str(input, &["tool_input", snippet_field]),
            }
        }
        "Read" => {
            let Some(path) = nested_str(input, &["tool_input", "file_path"]) else {
                return ToolAction::Other;
            };
            ToolAction::FileRead {
                path,
                offset: nested_i64(input, &["tool_input", "offset"]).unwrap_or(1),
                limit: nested_i64(input, &["tool_input", "limit"]).unwrap_or(2000),
            }
        }
        "Bash" => ToolAction::Shell {
            command: nested_str(input, &["tool_input", "command"]).unwrap_or_default(),
            exit_code: response_exit_code(input),
            stdout: response_str(input, &["stdout", "output"]),
            stderr: response_str(input, &["stderr"]),
        },
        "TodoWrite" => ToolAction::Todo {
            completed_count: input
                .get("tool_input")
                .and_then(|t| t.get("todos"))
                .and_then(Value::as_array)
                .map(|todos| {
                    todos
                        .iter()
                        .filter(|t| t.get("status").and_then(Value::as_str) == Some("completed"))
                        .count() as i64
                })
                .unwrap_or(0),
        },
        // The task tools replaced TodoWrite; a completion is one TaskUpdate.
        "TaskUpdate" => {
            let input = input.get("tool_input");
            let completed =
                input.and_then(|t| t.get("status")).and_then(Value::as_str) == Some("completed");
            let task_id = ["taskId", "task_id", "id"]
                .iter()
                .find_map(|k| input.and_then(|t| t.get(*k)))
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
            match (completed, task_id) {
                (true, Some(task_id)) => ToolAction::TaskCompleted { task_id },
                _ => ToolAction::Other,
            }
        }
        _ => ToolAction::Other,
    }
}

/// A `PostToolUseFailure` payload. Only shell failures matter to the brain
/// (error capture); the text is `tool_error`, usually led by "Exit code N".
fn claude_failed_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    if tool_name != "Bash" {
        return ToolAction::Other;
    }
    let error = match input.get("tool_error") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    let exit_code = error
        .strip_prefix("Exit code ")
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse::<i64>().ok())
        .filter(|n| *n != 0)
        .unwrap_or(1);
    ToolAction::Shell {
        command: nested_str(input, &["tool_input", "command"]).unwrap_or_default(),
        exit_code,
        stdout: error,
        stderr: String::new(),
    }
}

/// OpenAI Codex CLI. Deliberately Claude Code wire-compatible: same event
/// spellings, snake_case stdin, camelCase stdout — per the generated
/// schemas in openai/codex (codex-rs/hooks/schema). Tool names differ:
/// the shell tool is literally `Bash`, file edits arrive as `apply_patch`.
fn parse_codex(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let mut ev = parse_claude(input, event_override)?;
    if matches!(ev.kind, EventKind::PreTool | EventKind::PostTool) {
        let tool_name = str_field(input, "tool_name").unwrap_or_default();
        ev.tool = Some(codex_tool_action(input, &tool_name));
    }
    Some(ev)
}

fn codex_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    match tool_name {
        // tool_input is schema-`any`; tolerate both a command string and
        // an argv array.
        "Bash" => ToolAction::Shell {
            command: shell_command_from(input),
            exit_code: response_exit_code(input),
            stdout: response_str(input, &["stdout", "output", "aggregated_output"]),
            stderr: response_str(input, &["stderr"]),
        },
        // Codex edits files through apply_patch; the patch body names the
        // touched file(s) — surface the first as the edited path. Observed
        // live: real Codex puts the patch body in `tool_input.command`
        // (the docs' `input`/`patch` never appear); keep those as fallbacks.
        "apply_patch" => {
            let patch = nested_str(input, &["tool_input", "command"])
                .or_else(|| nested_str(input, &["tool_input", "input"]))
                .or_else(|| nested_str(input, &["tool_input", "patch"]))
                .unwrap_or_default();
            match first_patch_path(&patch) {
                Some(path) => ToolAction::FileEdit {
                    path,
                    snippet: Some(patch),
                },
                None => ToolAction::Other,
            }
        }
        // Matcher aliases Edit/Write exist but hook input still reports
        // apply_patch; keep the Claude shapes as a fallback for future
        // tool surfacing.
        _ => claude_tool_action(input, tool_name),
    }
}

/// GitHub Copilot CLI. Two payload formats exist, selected by the event
/// name the hook was REGISTERED under: camelCase events → camelCase
/// fields and NO event name in the payload; PascalCase events ("VS Code
/// compatible") → snake_case fields + `hook_event_name`. Axil's config
/// writer registers PascalCase so the payload is self-describing; the
/// camelCase spellings are still accepted for hand-written configs
/// (which then need `--event`).
fn parse_copilot(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let raw_event = event_override
        .map(str::to_string)
        .or_else(|| str_field(input, "hook_event_name"))
        .or_else(|| str_field(input, "eventName"))?;
    let kind = match raw_event.as_str() {
        // PascalCase alias is UserPromptSubmit (not ...Submitted).
        "UserPromptSubmit" | "userPromptSubmitted" => EventKind::UserPrompt,
        "SessionStart" | "sessionStart" => EventKind::SessionStart,
        "PreToolUse" | "preToolUse" => EventKind::PreTool,
        "PostToolUse" | "postToolUse" | "PostToolUseFailure" | "postToolUseFailure" => {
            EventKind::PostTool
        }
        // agentStop's PascalCase alias is Stop.
        "Stop" | "agentStop" => EventKind::Stop,
        "SessionEnd" | "sessionEnd" => EventKind::SessionEnd,
        _ => return None,
    };
    let tool_name = str_field(input, "tool_name")
        .or_else(|| str_field(input, "toolName"))
        .unwrap_or_default();
    let tool = match kind {
        EventKind::PreTool | EventKind::PostTool => Some(copilot_tool_action(input, &tool_name)),
        _ => None,
    };
    Some(HookEvent {
        kind,
        session_id: str_field(input, "session_id").or_else(|| str_field(input, "sessionId"))?,
        cwd: str_field(input, "cwd"),
        prompt: str_field(input, "prompt"),
        stop_hook_active: false,
        tool,
    })
}

fn copilot_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    // Arguments live under tool_input (snake_case format) or toolArgs
    // (camelCase) — and toolArgs may arrive as a JSON-encoded STRING
    // (documented gotcha). Normalize to an object first.
    let args_obj: Option<Value> = ["tool_input", "toolArgs"].iter().find_map(|root| {
        let v = input.get(*root)?;
        if v.is_object() {
            Some(v.clone())
        } else if let Some(s) = v.as_str() {
            serde_json::from_str(s).ok()
        } else {
            None
        }
    });
    let arg = |keys: &[&str]| -> Option<String> {
        let obj = args_obj.as_ref()?;
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(Value::as_str))
            .map(str::to_string)
    };
    // postToolUseFailure carries no exit code — only a top-level `error`
    // string. Without this, response_exit_code returns 0 and a failed
    // command takes the success path (skipping error capture, and letting
    // a failed `git commit` capture the *previous* HEAD as a success).
    let failed = input.get("error").is_some();
    match tool_name {
        "bash" | "powershell" => ToolAction::Shell {
            command: arg(&["command", "cmd"]).unwrap_or_default(),
            exit_code: if failed { 1 } else { response_exit_code(input) },
            stdout: copilot_tool_result_text(input),
            stderr: String::new(),
        },
        "edit" | "create" | "str_replace_editor" | "apply_patch" => {
            match arg(&["path", "file_path", "filePath"]) {
                Some(path) => ToolAction::FileEdit {
                    path,
                    snippet: arg(&["new_str", "content", "new_string"]),
                },
                None => ToolAction::Other,
            }
        }
        "view" => match arg(&["path", "file_path", "filePath"]) {
            Some(path) => ToolAction::FileRead {
                path,
                offset: 1,
                limit: 2000,
            },
            None => ToolAction::Other,
        },
        "update_todo" => ToolAction::Todo {
            completed_count: args_obj
                .as_ref()
                .and_then(|o| o.get("todos"))
                .and_then(Value::as_array)
                .map(|todos| {
                    todos
                        .iter()
                        .filter(|t| t.get("status").and_then(Value::as_str) == Some("completed"))
                        .count() as i64
                })
                .unwrap_or(0),
        },
        _ => ToolAction::Other,
    }
}

/// Copilot's tool result: `tool_result.text_result_for_llm` (snake_case)
/// or `toolResult.textResultForLlm` (camelCase).
fn copilot_tool_result_text(input: &Value) -> String {
    nested_str(input, &["tool_result", "text_result_for_llm"])
        .or_else(|| nested_str(input, &["toolResult", "textResultForLlm"]))
        // postToolUseFailure carries a plain error string instead.
        .or_else(|| str_field(input, "error"))
        .unwrap_or_default()
}

/// Factory Droid: byte-for-byte the Claude Code hooks contract
/// (snake_case stdin, hookSpecificOutput stdout, exit-2 blocks) with
/// Droid's own tool names — per docs.factory.ai/reference/hooks-reference.
fn parse_droid(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let mut ev = parse_claude(input, event_override)?;
    if matches!(ev.kind, EventKind::PreTool | EventKind::PostTool) {
        let tool_name = str_field(input, "tool_name").unwrap_or_default();
        ev.tool = Some(droid_tool_action(input, &tool_name));
    }
    Some(ev)
}

fn droid_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    match tool_name {
        // Droid's shell tool is Execute (not Bash).
        "Execute" => ToolAction::Shell {
            command: nested_str(input, &["tool_input", "command"]).unwrap_or_default(),
            exit_code: response_exit_code(input),
            stdout: response_str(input, &["stdout", "output"]),
            stderr: response_str(input, &["stderr"]),
        },
        // File writes: Create (file_path + content), Edit, ApplyPatch.
        "Create" | "Edit" | "ApplyPatch" => {
            match nested_str(input, &["tool_input", "file_path"])
                .or_else(|| nested_str(input, &["tool_input", "path"]))
            {
                Some(path) => ToolAction::FileEdit {
                    path,
                    snippet: nested_str(input, &["tool_input", "new_string"])
                        .or_else(|| nested_str(input, &["tool_input", "content"])),
                },
                None => ToolAction::Other,
            }
        }
        // Read + TodoWrite share Claude's shapes.
        "Read" | "TodoWrite" => claude_tool_action(input, tool_name),
        _ => ToolAction::Other,
    }
}

/// Qwen Code (and the legacy Gemini CLI it forked): settings.json hooks,
/// snake_case stdin with `hook_event_name`, hookSpecificOutput responses,
/// exit 2 blocks. Qwen kept the Claude-style event spellings; the legacy
/// Gemini names (BeforeTool/AfterTool/BeforeAgent/AfterAgent) are accepted
/// as aliases. Tool names are Gemini-lineage snake_case canonicals.
fn parse_gemini(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let raw_event = event_override
        .map(str::to_string)
        .or_else(|| str_field(input, "hook_event_name"))?;
    let kind = match raw_event.as_str() {
        "SessionStart" => EventKind::SessionStart,
        "SessionEnd" => EventKind::SessionEnd,
        "UserPromptSubmit" | "BeforeAgent" => EventKind::UserPrompt,
        "PreToolUse" | "BeforeTool" => EventKind::PreTool,
        "PostToolUse" | "PostToolUseFailure" | "AfterTool" => EventKind::PostTool,
        "Stop" | "AfterAgent" => EventKind::Stop,
        _ => return None,
    };
    let tool_name = str_field(input, "tool_name").unwrap_or_default();
    let tool = match kind {
        EventKind::PreTool | EventKind::PostTool => Some(gemini_tool_action(input, &tool_name)),
        _ => None,
    };
    Some(HookEvent {
        kind,
        session_id: str_field(input, "session_id")?,
        cwd: str_field(input, "cwd"),
        prompt: str_field(input, "prompt"),
        stop_hook_active: input
            .get("stop_hook_active")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        tool,
    })
}

fn gemini_tool_action(input: &Value, tool_name: &str) -> ToolAction {
    match tool_name {
        "run_shell_command" => ToolAction::Shell {
            command: nested_str(input, &["tool_input", "command"]).unwrap_or_default(),
            exit_code: response_exit_code(input),
            stdout: response_str(input, &["stdout", "output", "result"]),
            stderr: response_str(input, &["stderr"]),
        },
        // `replace` is the legacy alias Qwen canonicalizes to `edit`.
        "write_file" | "edit" | "replace" => {
            match nested_str(input, &["tool_input", "file_path"])
                .or_else(|| nested_str(input, &["tool_input", "path"]))
            {
                Some(path) => ToolAction::FileEdit {
                    path,
                    snippet: nested_str(input, &["tool_input", "content"])
                        .or_else(|| nested_str(input, &["tool_input", "new_string"])),
                },
                None => ToolAction::Other,
            }
        }
        "read_file" => match nested_str(input, &["tool_input", "file_path"])
            .or_else(|| nested_str(input, &["tool_input", "path"]))
        {
            Some(path) => ToolAction::FileRead {
                path,
                offset: nested_i64(input, &["tool_input", "offset"]).unwrap_or(1),
                limit: nested_i64(input, &["tool_input", "limit"]).unwrap_or(2000),
            },
            None => ToolAction::Other,
        },
        // Qwen's todo tool (Gemini legacy: write_todos).
        "todo_write" | "write_todos" => ToolAction::Todo {
            completed_count: input
                .get("tool_input")
                .and_then(|t| t.get("todos"))
                .and_then(Value::as_array)
                .map(|todos| {
                    todos
                        .iter()
                        .filter(|t| t.get("status").and_then(Value::as_str) == Some("completed"))
                        .count() as i64
                })
                .unwrap_or(0),
        },
        _ => ToolAction::Other,
    }
}

/// Antigravity CLI (`agy`): the payload carries NO event name — the config
/// writer passes `--event <name>` per registration. camelCase fields;
/// session identity is `conversationId`, the workspace root is
/// `workspacePaths[0]`; tool args use PascalCase keys (Windsurf lineage).
fn parse_antigravity(input: &Value, event_override: Option<&str>) -> Option<HookEvent> {
    let kind = match event_override? {
        "PreInvocation" => EventKind::PreModel,
        "PreToolUse" => EventKind::PreTool,
        "PostToolUse" => EventKind::PostTool,
        "Stop" => EventKind::Stop,
        _ => return None,
    };
    let tool = match kind {
        EventKind::PreTool => Some(antigravity_tool_action(input)),
        // PostToolUse carries only stepIdx + an optional error string — no
        // toolCall or output. Surface a failure so error capture still runs.
        EventKind::PostTool => Some(match str_field(input, "error") {
            Some(err) => ToolAction::Shell {
                command: String::new(),
                exit_code: 1,
                stdout: err,
                stderr: String::new(),
            },
            None => ToolAction::Other,
        }),
        _ => None,
    };
    Some(HookEvent {
        kind,
        session_id: str_field(input, "conversationId")?,
        cwd: input
            .get("workspacePaths")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .map(str::to_string),
        prompt: None,
        stop_hook_active: false,
        tool,
    })
}

fn antigravity_tool_action(input: &Value) -> ToolAction {
    let name = nested_str(input, &["toolCall", "name"]).unwrap_or_default();
    let arg = |keys: &[&str]| -> Option<String> {
        let args = input.get("toolCall")?.get("args")?;
        keys.iter()
            .find_map(|k| args.get(*k).and_then(Value::as_str))
            .map(str::to_string)
    };
    match name.as_str() {
        "run_command" => ToolAction::Shell {
            command: arg(&["CommandLine", "Command"]).unwrap_or_default(),
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        },
        // File-write arg keys are not documented; try the plausible
        // PascalCase spellings and degrade to Other.
        "write_to_file" | "replace_file_content" | "multi_replace_file_content" => {
            match arg(&["TargetFile", "AbsolutePath", "FilePath", "Path", "File"]) {
                Some(path) => ToolAction::FileEdit {
                    path,
                    snippet: arg(&["CodeContent", "Content", "ReplacementContent", "TargetContent"]),
                },
                None => ToolAction::Other,
            }
        }
        "view_file" => match arg(&["TargetFile", "AbsolutePath", "FilePath", "Path", "File"]) {
            Some(path) => ToolAction::FileRead {
                path,
                offset: 1,
                limit: 2000,
            },
            None => ToolAction::Other,
        },
        _ => ToolAction::Other,
    }
}

/// Extract a flat command string from a shell tool's input, tolerating
/// both a plain string and an argv array (Codex uses `command: [..]`).
fn shell_command_from(input: &Value) -> String {
    let ti = input.get("tool_input");
    if let Some(cmd) = ti.and_then(|t| t.get("command")) {
        if let Some(s) = cmd.as_str() {
            return s.to_string();
        }
        if let Some(arr) = cmd.as_array() {
            return arr
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    String::new()
}

/// First file path named by an apply_patch body
/// (`*** Update File: src/x.rs` / `*** Add File: ...`).
fn first_patch_path(patch: &str) -> Option<String> {
    for line in patch.lines() {
        for marker in ["*** Update File: ", "*** Add File: ", "*** Delete File: "] {
            if let Some(rest) = line.strip_prefix(marker) {
                let p = rest.trim();
                if !p.is_empty() {
                    return Some(p.to_string());
                }
            }
        }
    }
    None
}

// ── The brain ────────────────────────────────────────────────────────

struct HookCtx {
    dialect: Dialect,
    event: HookEvent,
    sid: String,
    project_dir: PathBuf,
    exe: PathBuf,
    db: Option<PathBuf>,
    tmp: PathBuf,
    files: SessionFiles,
}

/// One session's state: `axil-session-<sid>.<suffix>` files in the temp
/// dir. They live for the whole session, not one turn, and are removed when
/// the session is closed.
struct SessionFiles {
    tmp: PathBuf,
    sid: String,
}

impl SessionFiles {
    fn path(&self, suffix: &str) -> PathBuf {
        self.tmp
            .join(format!("axil-session-{}.{}", self.sid, suffix))
    }

    fn prefix(&self) -> String {
        format!("axil-session-{}.", self.sid)
    }

    fn own(&self) -> Vec<PathBuf> {
        let prefix = self.prefix();
        std::fs::read_dir(&self.tmp)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(prefix.as_str()))
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Sweep every per-session temp file, including the one-per-file/query
    /// sentinel files (`.recalled-<hash>`, `.searched-<hash>`, …).
    fn cleanup(&self) {
        for path in self.own() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Seconds since any of this session's files last changed.
    fn idle_secs(&self) -> u64 {
        self.own()
            .iter()
            .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
            .filter_map(|t| t.elapsed().ok())
            .map(|d| d.as_secs())
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Lines of the edit manifest (one per edit, repeats included).
    fn manifest_lines(&self) -> Vec<String> {
        std::fs::read_to_string(self.path("manifest"))
            .map(|m| {
                m.lines()
                    .filter(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// A line-count mark into the manifest (`turn`, `flushed`).
    fn mark(&self, name: &str) -> usize {
        std::fs::read_to_string(self.path(name))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    fn set_mark(&self, name: &str, lines: usize) {
        let _ = std::fs::write(self.path(name), lines.to_string());
    }
}

impl HookCtx {
    fn new(dialect: Dialect, event: HookEvent) -> Option<Self> {
        let sid = sanitize_id(&event.session_id);
        if sid.is_empty() {
            return None;
        }
        let project_dir = project_dir_env_var(dialect)
            .and_then(std::env::var_os)
            .map(PathBuf::from)
            .or_else(|| event.cwd.clone().map(PathBuf::from))
            .or_else(|| std::env::current_dir().ok())?;
        let exe = std::env::current_exe().ok()?;
        let db = find_db(&project_dir);
        let tmp = std::env::temp_dir();
        let files = SessionFiles {
            tmp: tmp.clone(),
            sid: sid.clone(),
        };
        Some(Self {
            dialect,
            event,
            sid,
            project_dir,
            exe,
            db,
            tmp,
            files,
        })
    }

    fn dispatch(&self) -> Result<()> {
        match self.event.kind {
            EventKind::UserPrompt => self.on_user_prompt(),
            EventKind::SessionStart => {
                // A real session-start event (also after a compaction or
                // /clear, when the boot is worth re-injecting): mark booted
                // so the first-pre-tool emulation never double-boots.
                let _ = std::fs::write(self.sfile("booted"), "");
                self.start_session();
                self.boot_push();
                Ok(())
            }
            EventKind::PreModel => self.on_pre_model(),
            EventKind::PreTool => self.on_pre_tool(),
            EventKind::PostTool => self.on_post_tool(),
            EventKind::Stop => self.on_stop(),
            EventKind::SessionEnd => self.on_session_end(),
        }
    }

    // ── Dialect-shaped output ─────────────────────────────────────────

    /// Inject context the model will see.
    ///
    /// Claude/Codex/Droid/Qwen share `hookSpecificOutput.additionalContext`
    /// with `hookEventName` required to equal the event (Codex validates it
    /// against a const). Copilot takes a root-level `additionalContext` on
    /// a single line. Antigravity has NO context channel on tool events —
    /// text is queued and flushed as an `injectSteps` ephemeral message on
    /// the next PreInvocation.
    fn emit_context(&self, ctx: &str) {
        match self.dialect {
            Dialect::Copilot => {
                println!("{}", json!({ "additionalContext": ctx }));
            }
            Dialect::Antigravity => {
                append_line(&self.sfile("pending"), ctx);
                append_line(&self.sfile("pending"), "");
            }
            // Everything else (Claude/Codex/Droid/Qwen) shares the
            // hookSpecificOutput.additionalContext channel. Note: we do NOT
            // attach a `permissionDecision` on Qwen's PreToolUse — emitting
            // `allow` there would auto-approve the very tool the memory hook
            // is annotating, bypassing the user's confirmation gate.
            // Omitting it leaves the normal permission flow untouched.
            _ => {
                println!(
                    "{}",
                    json!({
                        "hookSpecificOutput": {
                            "hookEventName": claude_event_name(self.event.kind),
                            "additionalContext": ctx,
                        }
                    })
                );
            }
        }
    }

    /// Block a stop, demanding a narrative store first.
    /// `{"decision":"block","reason"}` is documented for Claude, Codex,
    /// Droid, Copilot, and Qwen; Antigravity spells it
    /// `{"decision":"continue"}` — "continue working", not "stop".
    fn emit_stop_block(&self, reason: &str) {
        let decision = if self.dialect == Dialect::Antigravity {
            "continue"
        } else {
            "block"
        };
        println!("{}", json!({"decision": decision, "reason": reason}));
    }

    // ── Session temp files ────────────────────────────────────────────

    fn sfile(&self, suffix: &str) -> PathBuf {
        self.files.path(suffix)
    }

    /// Claim a one-shot sentinel. `create_new` is atomic, so of two hooks
    /// racing for the same first call exactly one gets `true`.
    fn claim(&self, suffix: &str) -> bool {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.sfile(suffix))
            .is_ok()
    }

    /// Previous-session manifest, scoped per project so one repo's edited
    /// files never seed another repo's boot `--files`. Deliberately NOT
    /// prefixed `axil-session-<sid>.` so `cleanup_session_files` leaves it
    /// in place for the next session.
    fn prev_manifest(&self) -> PathBuf {
        let scope = fnv1a(&self.project_dir.to_string_lossy());
        self.tmp.join(format!("axil-prev-{scope}.manifest"))
    }

    fn log_problem(&self, line: &str) {
        let path = self.sfile("problems");
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() >= PROBLEMS_MAX_BYTES {
                return;
            }
        }
        append_line(&path, line);
    }

    // ── Heartbeat counters ────────────────────────────────────────────
    // One line per event in `<sid>.events` (`tools`, `recalls`, `stores`,
    // `errors`). A short append is a single write, so hooks running in
    // parallel never lose a count, as a read-modify-write file did.

    fn bump_count(&self, key: &str) {
        append_line(&self.sfile("events"), key);
    }

    fn read_count(&self, key: &str) -> i64 {
        std::fs::read_to_string(self.sfile("events"))
            .map(|s| s.lines().filter(|l| *l == key).count() as i64)
            .unwrap_or(0)
    }

    fn counts_compact(&self) -> String {
        format!(
            "{}s ∙ {}r ∙ {}t",
            self.read_count("stores"),
            self.read_count("recalls"),
            self.read_count("tools")
        )
    }

    // ── Child-process helpers (the brain shells out to its own binary) ─

    /// The brain's own binary for a lookup, with the slow-query log off:
    /// logging a slow read is a write, and a hook's reads must not write
    /// under the harness's kill timeout. The one exception is the graph's
    /// first read after an older binary wrote edges, which brings its
    /// adjacency tables up to date in commits a kill can't leave half done
    /// (see `GraphEngine::prepare` in axil-graph).
    fn axil_cmd(&self) -> Command {
        let mut cmd = Command::new(&self.exe);
        cmd.env("AXIL_SLOW_QUERY_LOG", "0");
        cmd
    }

    /// Run an axil lookup against the resolved DB; Some(stdout) on success.
    /// Child stderr is discarded — hook noise must not leak. Writes go
    /// through [`HookCtx::enqueue`] instead.
    fn axil_db_out(&self, args: &[&str]) -> Option<String> {
        let db = self.db.as_ref()?;
        run_capture(
            self.axil_cmd()
                .arg("--db")
                .arg(db)
                .args(args)
                .stdin(Stdio::null()),
        )
    }

    /// Spawn an axil subcommand with ALL stdio detached (null) and don't
    /// wait — for opportunistic background work whose output we don't need.
    /// Crucially, null stdout means the child never inherits the agent's
    /// hook pipe, so the agent's turn never blocks waiting on it.
    fn spawn_fire_and_forget(&self, args: &[&str]) {
        let Some(db) = self.db.as_ref() else { return };
        let mut cmd = Command::new(&self.exe);
        cmd.arg("--db")
            .arg(db)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        own_process_group(&mut cmd);
        let _ = cmd.spawn();
    }

    // ── Write queue ───────────────────────────────────────────────────

    /// Queue a database write and make sure a drainer is running. False
    /// when there is no database or the job file can't be written.
    fn enqueue(&self, job: &Job) -> bool {
        let Some(db) = self.db.as_ref() else {
            return false;
        };
        if !write_job(db, job) {
            return false;
        }
        spawn_drainer(&self.exe, db, &self.project_dir);
        true
    }

    // ── Session lifecycle ─────────────────────────────────────────────

    /// Record which project this session belongs to, and queue the close of
    /// this project's sessions that went quiet without a session-end event
    /// (a closed terminal, a crash) so their edits still reach `_sessions`.
    fn start_session(&self) {
        let project = self.project_dir.to_string_lossy().into_owned();
        let _ = std::fs::write(self.sfile("project"), &project);
        let Ok(entries) = std::fs::read_dir(&self.tmp) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(sid) = name
                .strip_prefix("axil-session-")
                .and_then(|rest| rest.strip_suffix(".project"))
            else {
                continue;
            };
            if sid == self.sid
                || std::fs::read_to_string(entry.path()).ok().as_deref() != Some(project.as_str())
            {
                continue;
            }
            let other = SessionFiles {
                tmp: self.tmp.clone(),
                sid: sid.to_string(),
            };
            if other.idle_secs() >= STALE_SESSION_SECS {
                self.flush_session(&other);
                other.cleanup();
            }
        }
    }

    /// Queue the close of everything a session recorded since its last
    /// flush: the `_sessions` row and its links, then worker, beliefs and
    /// session-heal, all run by the drainer. The snippet and problem logs
    /// are consumed; the manifest stays (the edit nudge counts it) with a
    /// mark past the flushed lines.
    fn flush_session(&self, files: &SessionFiles) {
        if self.db.is_none() {
            return;
        }
        let manifest = files.manifest_lines();
        let new_files: BTreeSet<String> = manifest
            .iter()
            .skip(files.mark("flushed"))
            .cloned()
            .collect();
        let problems = std::fs::read_to_string(files.path("problems"))
            .ok()
            .filter(|p| !p.trim().is_empty());
        if new_files.is_empty() && problems.is_none() {
            return;
        }
        let content = std::fs::read_to_string(files.path("content")).unwrap_or_default();
        let job = Job::CloseSession {
            session: files.sid.clone(),
            project_dir: self.project_dir.clone(),
            files: new_files.into_iter().collect(),
            content: truncate_utf8(&content, 4000).to_string(),
            problems,
            ended_at: now_iso(),
        };
        if !self.enqueue(&job) {
            return;
        }
        files.set_mark("flushed", manifest.len());
        let _ = std::fs::remove_file(files.path("content"));
        let _ = std::fs::remove_file(files.path("problems"));
        if !manifest.is_empty() {
            // The next session's boot seeds `--files` from this.
            let _ = std::fs::copy(files.path("manifest"), self.prev_manifest());
        }
    }

    /// Whether the harness will send a session-end event. Without one
    /// (Codex, Antigravity, Claude Code projects installed before Axil
    /// registered it), the session is flushed at every Stop instead.
    fn has_session_end_event(&self) -> bool {
        match self.dialect {
            Dialect::Copilot | Dialect::Droid | Dialect::Gemini => true,
            Dialect::Codex | Dialect::Antigravity => false,
            // Only Claude Code itself sends SessionEnd, and it sets
            // CLAUDE_PROJECT_DIR for its hooks; other harnesses speaking this
            // dialect (the OpenCode plugin) don't.
            Dialect::Claude => {
                std::env::var_os("CLAUDE_PROJECT_DIR").is_some()
                    && claude_registers_session_end(&self.project_dir)
            }
        }
    }

    /// Count narrative records stored in the last hour, or `None` when the
    /// lookup failed (a busy database is not "nothing stored").
    fn count_recent_narrative(&self) -> Option<i64> {
        let out = self.axil_db_out(&["since", "1h"])?;
        let rows = serde_json::from_str::<Value>(out.trim()).ok()?;
        let rows = rows.as_array()?;
        Some(
            rows.iter()
                .filter(|r| {
                    r.get("table")
                        .and_then(Value::as_str)
                        .map(|t| NARRATIVE_TABLES.contains(&t))
                        .unwrap_or(false)
                })
                .count() as i64,
        )
    }

    /// True when HEAD has a commit within the last hour. Lets the Stop
    /// guard accept a fresh commit even when the async PostToolUse
    /// `commits` row hasn't landed yet (async-hook vs sync-Stop race).
    fn has_recent_git_commit(&self) -> bool {
        let Some(out) = run_capture(
            Command::new("git")
                .arg("-C")
                .arg(&self.project_dir)
                .args(["log", "-1", "--pretty=%ct"])
                .stdin(Stdio::null()),
        ) else {
            return false;
        };
        let Ok(commit_ts) = out.trim().parse::<i64>() else {
            return false;
        };
        chrono::Utc::now().timestamp() - commit_ts < 3600
    }

    fn rel_path(&self, file_path: &str) -> String {
        let p = Path::new(file_path);
        let rel = p
            .strip_prefix(&self.project_dir)
            .map(|r| r.to_path_buf())
            .unwrap_or_else(|_| p.to_path_buf());
        // Store forward slashes so rel paths match the structural index on
        // every platform.
        rel.to_string_lossy().replace('\\', "/")
    }

    // ── User prompt: inject <context> block from recall ──────────────
    // Runs on every user prompt; must stay under ~2s wall-clock, so the
    // recall carries its own deadline.
    fn on_user_prompt(&self) -> Result<()> {
        // Copilot ignores userPromptSubmitted output entirely (documented:
        // "Output processed: No") — don't burn the recall latency there.
        if self.dialect == Dialect::Copilot {
            return Ok(());
        }
        let Some(prompt) = self.event.prompt.as_deref() else {
            return Ok(());
        };
        // Trivially short prompts (acks, confirmations) — no useful recall.
        if prompt.chars().count() < 8 {
            return Ok(());
        }
        if let Some(ctx) = self.axil_db_out(&[
            "recall",
            prompt,
            "--recall-format",
            "context-block",
            "--budget",
            "2000",
            "--timeout-ms",
            "1800",
            "--top-k",
            "5",
        ]) {
            if ctx.trim().is_empty() {
                return Ok(());
            }
            match self.dialect {
                // Claude and Droid document raw prompt-hook stdout being
                // added to context directly.
                Dialect::Claude | Dialect::Droid => print!("{ctx}"),
                // Codex documents only the JSON channel for this event.
                _ => self.emit_context(&ctx),
            }
        }
        Ok(())
    }

    // ── Pre-tool ──────────────────────────────────────────────────────

    fn on_pre_tool(&self) -> Result<()> {
        self.bump_count("tools");

        // Installs without a session-start event (older Claude Code
        // installs, some dialects) boot on the session's first tool call.
        // A real SessionStart already wrote the sentinel in dispatch().
        if self.claim("booted") {
            self.start_session();
            self.boot_push();
            // Fall through: if the session's first tool is a file edit the
            // file-recall context must still be injected below.
        }

        // Antigravity surfaces file edits ONLY at PreToolUse (its
        // PostToolUse payload carries no toolCall), so record the manifest
        // here — otherwise on_stop sees no manifest and skips the entire
        // session-close pipeline for this dialect. Other dialects log at
        // PostToolUse, once the edit has actually happened.
        if self.dialect == Dialect::Antigravity {
            if let Some(ToolAction::FileEdit { path, snippet }) = &self.event.tool {
                let _ = self.post_edit_log(path, snippet.as_deref());
            }
        }

        match &self.event.tool {
            Some(ToolAction::FileEdit { path, .. }) => self.pre_edit_context(path),
            Some(ToolAction::Shell { command, .. }) => self.pre_shell_search_gate(command),
            _ => Ok(()),
        }
    }

    /// Produce the boot context text (and fire the opportunistic background
    /// refreshes). The banner goes straight to stderr; how the boot text
    /// reaches the model is the caller's dialect-specific concern.
    fn boot_context_text(&self) -> Option<String> {
        self.db.as_ref()?;
        // Context-aware boot flags from the previous session's manifest.
        let mut args: Vec<String> = vec![
            "boot".into(),
            "--boot-format".into(),
            "narrative".into(),
            "--budget".into(),
            "800".into(),
        ];
        if let Ok(prev) = std::fs::read_to_string(self.prev_manifest()) {
            let files: Vec<&str> = prev
                .lines()
                .filter(|l| !l.is_empty())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .take(5)
                .collect();
            if !files.is_empty() {
                args.push("--files".into());
                args.push(files.join(","));
            }
        }

        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let boot = self.axil_db_out(&arg_refs);

        // Opportunistic background refreshes — fire-and-forget with stdio
        // fully detached. These spawn their own detached workers; if the
        // hook CAPTURED their stdout (via .output()), the agent's hook
        // pipe would stay open until those grandchildren exit, hanging the
        // whole turn (observed with Antigravity's `agy`). Spawning with
        // null stdio and not waiting keeps the hook instant.
        self.spawn_fire_and_forget(&["scip", "refresh", "--if-stale", "--in-background", "--quiet"]);
        self.spawn_fire_and_forget(&["maintain", "--if-stale", "--in-background", "--quiet"]);

        boot.filter(|b| !b.trim().is_empty())
    }

    fn boot_push(&self) {
        if let Some(boot) = self.boot_context_text() {
            // A real session-start event supports context injection — use
            // it so the model (not just the terminal) sees the boot.
            if self.event.kind == EventKind::SessionStart {
                self.emit_context(&boot);
            } else {
                eprintln!("{boot}");
            }
        }
    }

    /// Antigravity's PreInvocation: fires before every model call and is
    /// that dialect's only context-injection channel. First fire carries
    /// the boot; every fire flushes context queued by the tool handlers.
    fn on_pre_model(&self) -> Result<()> {
        let mut chunks: Vec<String> = Vec::new();

        let booted = self.sfile("booted");
        if !booted.exists() {
            let _ = std::fs::write(&booted, "");
            if let Some(boot) = self.boot_context_text() {
                chunks.push(boot);
            }
        }

        let pending = self.sfile("pending");
        if let Ok(queued) = std::fs::read_to_string(&pending) {
            if !queued.trim().is_empty() {
                chunks.push(queued.trim_end().to_string());
            }
            let _ = std::fs::remove_file(&pending);
        }

        if !chunks.is_empty() {
            println!(
                "{}",
                json!({ "injectSteps": [{ "ephemeralMessage": chunks.join("\n\n") }] })
            );
        }
        Ok(())
    }

    /// Surface past memories about a file BEFORE the agent edits it, plus
    /// the 5-edit "have you stored anything?" nudge — combined into one
    /// hookSpecificOutput so they don't fight for stdout.
    fn pre_edit_context(&self, file_path: &str) -> Result<()> {
        if is_skipped_path(file_path) || self.db.is_none() {
            return Ok(());
        }
        let rel = self.rel_path(file_path);

        // Per-file sentinel: recall-for-file once per file per session.
        // Multi-edit refactors of one file shouldn't pay 50-200ms per edit.
        let sentinel = self.sfile(&format!("recalled-{}", fnv1a(&rel)));
        let mut ctx = String::new();
        if !sentinel.exists() {
            if let Some(out) = self.axil_db_out(&["recall-for-file", &rel, "--top-k", "3"]) {
                // Only a lookup that ran marks the file done; a failed one
                // (say, a busy database) is retried on the next edit.
                let _ = std::fs::write(&sentinel, "");
                if let Ok(v) = serde_json::from_str::<Value>(out.trim()) {
                    let matches = v.get("matches").and_then(Value::as_i64).unwrap_or(0);
                    if matches > 0 {
                        let summaries: Vec<String> = v
                            .get("results")
                            .and_then(Value::as_array)
                            .map(|rows| {
                                rows.iter()
                                    .map(|r| {
                                        format!(
                                            "  • [{}] {}",
                                            r.get("table").and_then(Value::as_str).unwrap_or(""),
                                            r.get("summary").and_then(Value::as_str).unwrap_or("")
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !summaries.is_empty() {
                            ctx = format!(
                                "📎 AXIL — past memories about {rel} (read these before editing):\n{}",
                                summaries.join("\n")
                            );
                        }
                    }
                }
            }
        }

        // 5-edit nudge: fires at every 5th edit with no narrative stored.
        if let Ok(manifest) = std::fs::read_to_string(self.sfile("manifest")) {
            let edit_count = manifest.lines().count();
            if edit_count >= 5 && edit_count % 5 == 0 && self.count_recent_narrative() == Some(0) {
                let nudge = format!(
                    "⚠️ AXIL — {edit_count} files edited this session, no {NARRATIVE_TABLES_TEXT} stored. \
                     Store inline (axil store …) or commit; don't batch at the end."
                );
                if ctx.is_empty() {
                    ctx = nudge;
                } else {
                    ctx = format!("{ctx}\n\n{nudge}");
                }
            }
        }

        if !ctx.is_empty() {
            self.emit_context(&ctx);
        }
        Ok(())
    }

    /// Pair broad repo search with Axil's own index: run code-search/fts
    /// first and inject the compact result; when no query is extractable
    /// and the session hasn't recalled yet, inject the gate reminder.
    fn pre_shell_search_gate(&self, cmd: &str) -> Result<()> {
        let (is_repo_search, mut query) = detect_repo_search(cmd);
        if !is_repo_search {
            return Ok(());
        }
        if query.chars().count() < 3 {
            query = String::new();
        }

        if query.is_empty() {
            if self.read_count("recalls") == 0 {
                let ctx = "⚠️ AXIL search gate — this session has not used Axil recall yet. Before broad repo discovery, run one of:\n  axil recall \"<what you need>\" --top-k 5\n  axil code-search \"<symbol/module/API>\" --top-k 5\n  axil fts \"<exact term>\" --limit 5\n\nThen open the files Axil returns and verify current code.";
                self.emit_context(ctx);
            }
            return Ok(());
        }
        if self.db.is_none() {
            return Ok(());
        }

        let mode = if is_code_like_query(&query) {
            "code-search"
        } else {
            "fts"
        };
        // One paired search per (mode, query) per session.
        let sentinel = self.sfile(&format!("searched-{}", fnv1a(&format!("{mode}:{query}"))));
        if sentinel.exists() {
            return Ok(());
        }

        let hits = if mode == "code-search" {
            self.axil_db_out(&["code-search", &query, "--top-k", "3", "--format", "pretty"])
        } else {
            self.axil_db_out(&["fts", &query, "--limit", "3", "--format", "table"])
        };
        // A search that ran is done for the session, hits or not; a failed one
        // (say, a busy database) is retried the next time.
        let Some(hits) = hits else {
            return Ok(());
        };
        let _ = std::fs::write(&sentinel, "");
        if !is_empty_axil_output(&hits) {
            let ctx = format!(
                "📎 AXIL {mode}('{query}') — check this before spending tokens on repo-wide search:\n{hits}\n\nFor broad repo lookups, prefer 'axil {mode} <query>' first; use rg/grep after Axil points you at files or when verifying current text."
            );
            self.emit_context(&ctx);
        }
        Ok(())
    }

    // ── Post-tool ─────────────────────────────────────────────────────

    fn on_post_tool(&self) -> Result<()> {
        match &self.event.tool {
            Some(ToolAction::FileRead { path, offset, limit }) => {
                self.post_read_fallback_capture(path, *offset, *limit)
            }
            Some(ToolAction::FileEdit { path, snippet }) => {
                self.post_edit_log(path, snippet.as_deref())
            }
            Some(ToolAction::Shell {
                command,
                exit_code,
                stdout,
                stderr,
            }) => self.post_shell(command, *exit_code, stdout, stderr),
            Some(ToolAction::Todo { completed_count }) => {
                self.post_todo_store_reminder(*completed_count)
            }
            Some(ToolAction::TaskCompleted { task_id }) => {
                if self.claim(&format!("task-{}", fnv1a(task_id))) {
                    self.emit_context(STORE_REMINDER);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// After a Read that follows a recent empty recall/code-search/fts,
    /// attach a low-importance context row tying the missed query to the
    /// exact line range the agent opened — closing the miss→fallback loop.
    /// Rows carry `_origin: fallback_capture` and `_importance: 0.2` so they
    /// sit below the default recall floor.
    fn post_read_fallback_capture(&self, file_path: &str, offset: i64, limit: i64) -> Result<()> {
        let problems_path = self.sfile("problems");
        if !problems_path.exists() || is_skipped_path(file_path) {
            return Ok(());
        }

        // ISO 8601 strings sort lexically — string compare beats platform
        // date-parsing differences.
        let cutoff = (chrono::Utc::now() - chrono::Duration::minutes(5))
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();
        let problems = std::fs::read_to_string(&problems_path).unwrap_or_default();
        let recent_miss = problems
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| {
                v.get("kind").and_then(Value::as_str) == Some("empty_result")
                    && v.get("at")
                        .and_then(Value::as_str)
                        .map(|at| at >= cutoff.as_str())
                        .unwrap_or(false)
            })
            .last();
        let Some(miss) = recent_miss else {
            return Ok(());
        };
        let Some(missed_query) = miss
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.is_empty())
        else {
            return Ok(());
        };

        // A malformed limit (<= 0) would invert the range; clamp so
        // line_end is never below line_start.
        let line_start = offset;
        let line_end = offset.max(offset.saturating_add(limit).saturating_sub(1));
        let rel = self.rel_path(file_path);

        // Dedup: one capture per (query, path, range) per session.
        let key = fnv1a(&format!("{missed_query}:{rel}:{line_start}:{line_end}"));
        let sentinel = self.sfile(&format!("fallback-{key}"));
        if sentinel.exists() {
            return Ok(());
        }

        let payload = json!({
            "type": "fallback_capture",
            "summary": format!("Fallback capture for query: {missed_query}"),
            "query": missed_query,
            "code_refs": [{"path": rel, "line_start": line_start, "line_end": line_end}],
            "_origin": "fallback_capture",
            "_importance": 0.2,
        });
        if self.enqueue(&Job::axil(
            &["store", "context", &payload.to_string()],
            None,
        )) {
            let _ = std::fs::write(&sentinel, "");
            eprintln!(
                "🧠 Axil queued fallback capture: '{missed_query}' → {rel}:{line_start}-{line_end}"
            );
        }
        Ok(())
    }

    /// Track the edit manifest and accumulate content snippets for the
    /// end-of-session entity extraction.
    fn post_edit_log(&self, file_path: &str, snippet: Option<&str>) -> Result<()> {
        // An edited manifest or lockfile can change dependency versions:
        // re-ingest the docs of whatever changed (a no-op when nothing did).
        // Checked before the skip below, which drops lockfiles.
        if is_dependency_manifest(file_path) {
            self.enqueue(&Job::axil(
                &["deps", "refresh", "--if-stale", "--quiet"],
                None,
            ));
        }
        if is_skipped_path(file_path) {
            return Ok(());
        }
        append_line(&self.sfile("manifest"), &self.rel_path(file_path));
        if let Some(text) = snippet {
            if !text.is_empty() {
                append_line(&self.sfile("content"), truncate_utf8(text, 500));
            }
        }
        Ok(())
    }

    fn post_shell(&self, cmd: &str, exit_code: i64, stdout: &str, stderr: &str) -> Result<()> {
        let subcommands = axil_subcommands(cmd);
        let runs_any = |set: &[&str]| subcommands.iter().any(|s| set.contains(&s.as_str()));
        if exit_code == 0 && !cmd.is_empty() {
            // Heartbeat: the agent just interacted with its own brain.
            if runs_any(STORE_SUBCOMMANDS) {
                self.bump_count("stores");
                eprintln!("🧠 Axil stored (session: {})", self.counts_compact());
            }
            if runs_any(RECALL_SUBCOMMANDS) {
                self.bump_count("recalls");
            }

            // A commit message IS a decision/summary the agent already wrote —
            // capture it as narrative so the Stop guard doesn't demand a
            // re-statement of what's in the commit.
            if runs_git_commit(cmd) {
                self.capture_git_commit();
            }
        }

        if exit_code != 0 {
            self.bump_count("errors");
            if !stdout.is_empty() {
                // High confidence threshold to avoid noise.
                self.enqueue(&Job::axil(
                    &["auto-capture", "-", "--min-confidence", "0.8", "--source", "bash"],
                    Some(truncate_utf8(stdout, 2000)),
                ));
            }
            // Generic build/test failures already flow through auto-capture;
            // only axil-specific failures feed session-heal.
            if !subcommands.is_empty() {
                let event = json!({
                    "kind": "command_failure",
                    "subcommand": extract_axil_subcmd(cmd),
                    "query": cmd,
                    "exit_code": exit_code,
                    "stderr": truncate_utf8(stderr, 500),
                    "at": now_iso(),
                });
                self.log_problem(&event.to_string());
            }
        } else if runs_any(&[
            "recall",
            "code-search",
            "fts",
            "recall-for-file",
            "recall-for-entity",
        ]) {
            // axil read commands return 0 with empty output when nothing
            // matched; a session full of these tells session-heal the index
            // is stale or memory is sparse for the topics being asked.
            if is_empty_axil_output(stdout) {
                let event = json!({
                    "kind": "empty_result",
                    "subcommand": extract_axil_subcmd(cmd),
                    "query": first_quoted_arg(cmd).unwrap_or_default(),
                    "at": now_iso(),
                });
                self.log_problem(&event.to_string());
            }
        }
        Ok(())
    }

    fn capture_git_commit(&self) {
        if self.db.is_none() {
            return;
        }
        // %x1f (unit separator) splits headers in one git call; the body is
        // fetched separately because it can contain newlines.
        let Some(headers) = run_capture(
            Command::new("git")
                .arg("-C")
                .arg(&self.project_dir)
                .args(["log", "-1", "--pretty=%H%x1f%s%x1f%an%x1f%cI"])
                .stdin(Stdio::null()),
        ) else {
            return;
        };
        let parts: Vec<&str> = headers.trim_end().split('\u{1f}').collect();
        let (Some(sha), subject, author, committed_at) = (
            parts.first().filter(|s| !s.is_empty()),
            parts.get(1).copied().unwrap_or(""),
            parts.get(2).copied().unwrap_or(""),
            parts.get(3).copied().unwrap_or(""),
        ) else {
            return;
        };
        // Once per commit: a failed or no-op commit leaves HEAD where it
        // was, and storing it again only adds duplicate recall hits.
        if !self.claim(&format!("commit-{sha}")) {
            return;
        }
        let body = run_capture(
            Command::new("git")
                .arg("-C")
                .arg(&self.project_dir)
                .args(["log", "-1", "--pretty=%b"])
                .stdin(Stdio::null()),
        )
        .unwrap_or_default();
        let files: Vec<String> = run_capture(
            Command::new("git")
                .arg("-C")
                .arg(&self.project_dir)
                .args(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
                .stdin(Stdio::null()),
        )
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();

        // `summary` is what recall shows; `content` (subject and body) is
        // what gets embedded and searched, so the reasoning in the body is
        // findable instead of the sha and author.
        let body = body.trim_end();
        let content = if body.is_empty() {
            subject.to_string()
        } else {
            format!("{subject}\n\n{body}")
        };
        let payload = json!({
            "sha": sha,
            "summary": subject,
            "content": content,
            "subject": subject,
            "body": body,
            "author": author,
            "committed_at": committed_at,
            "files": files,
        });
        if self.enqueue(&Job::axil(
            &["store", "commits", &payload.to_string()],
            None,
        )) {
            self.bump_count("stores");
            let sha7: String = sha.chars().take(7).collect();
            eprintln!("🧠 Axil queued commit {sha7}: {subject}");
        }
    }

    /// When a todo flips to completed, inject the store reminder BEFORE the
    /// agent moves on. Dialects with a whole-list todo tool (`TodoWrite`
    /// and kin) report a completed count; Claude Code's task tools report
    /// one `TaskUpdate` per task instead (see `TaskCompleted`).
    fn post_todo_store_reminder(&self, completed: i64) -> Result<()> {
        let sentinel = self.sfile("todos");
        let last: i64 = std::fs::read_to_string(&sentinel)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let _ = std::fs::write(&sentinel, completed.to_string());

        if completed > last {
            self.emit_context(STORE_REMINDER);
        }
        Ok(())
    }

    // ── Stop and session end ──────────────────────────────────────────

    /// Stop fires at the end of every turn, so it only guards: when this
    /// turn edited several files and nothing narrative was stored, block
    /// the stop and ask for a store. Session state carries over to the next
    /// turn. Without a session-end event, the session is flushed here too.
    fn on_stop(&self) -> Result<()> {
        let manifest = self.files.manifest_lines();
        let turn: BTreeSet<&str> = manifest
            .iter()
            .skip(self.files.mark("turn"))
            .map(String::as_str)
            .collect();
        let file_count = turn.len();

        // The JSON {"decision":"block"} on stdout is the only channel the
        // harness re-injects into the model — stderr is invisible to it.
        if !self.event.stop_hook_active
            && file_count > 2
            && self.db.is_some()
            && self.count_recent_narrative() == Some(0)
            && !self.has_recent_git_commit()
        {
            let files_json = serde_json::to_string(&turn).unwrap_or_else(|_| "[]".into());
            let reason = format!(
                "Axil brain: {file_count} files were edited this turn but no {NARRATIVE_TABLES_TEXT} row was stored in the last hour (and no git commit). Before stopping, either: (a) commit the work — the commit message is captured as narrative — or (b) run: axil checkpoint '{{\"state\":\"<where things stand>\",\"next_steps\":[\"<remaining work>\"],\"references\":[{{\"kind\":\"file\",\"ref\":\"<path>\"}}]}}' (files touched this turn: {files_json}). After storing, you may stop."
            );
            self.emit_stop_block(&reason);
            // Keep the turn mark: the retried stop must see the same files.
            return Ok(());
        }
        self.files.set_mark("turn", manifest.len());

        if !self.has_session_end_event() {
            self.flush_session(&self.files);
        }
        Ok(())
    }

    /// The session is over: queue its close and remove its temp files. The
    /// harness gives this event very little time, so it only writes a job.
    fn on_session_end(&self) -> Result<()> {
        self.flush_session(&self.files);
        let (stores, recalls) = (self.read_count("stores"), self.read_count("recalls"));
        if stores != 0 || recalls != 0 {
            eprintln!(
                "🧠 Axil session: {stores} stored ∙ {recalls} recalled ∙ {} tools ∙ {} errors",
                self.read_count("tools"),
                self.read_count("errors")
            );
        }
        self.files.cleanup();
        Ok(())
    }
}

const STORE_REMINDER: &str = "You just marked a task completed. BEFORE doing anything else, run axil store with a summary of what you did and why. This is mandatory for every completed task.";

/// True when this project's Claude Code settings route SessionEnd to the
/// brain, which `axil install` does from this version on.
fn claude_registers_session_end(project_dir: &Path) -> bool {
    ["settings.json", "settings.local.json"].iter().any(|name| {
        std::fs::read_to_string(project_dir.join(".claude").join(name))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.pointer("/hooks/SessionEnd").map(Value::to_string))
            .is_some_and(|hooks| hooks.contains(" hook run"))
    })
}

// ── Write queue and drainer ──────────────────────────────────────────

/// How long the drainer waits, in total, for another process to release the
/// writer lock before giving a job up.
const BUSY_WAIT_MAX: Duration = Duration::from_secs(120);

/// A drain lock untouched this long belonged to a drainer that died.
const DRAIN_LOCK_STALE_SECS: u64 = 30 * 60;

/// Bound on `drain.log`; past it the log starts over.
const DRAIN_LOG_MAX_BYTES: u64 = 1_048_576;

/// A database write a hook asked for, run later by `axil hook drain`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Job {
    /// One axil command, such as `store commits '{…}'`.
    Axil {
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stdin: Option<String>,
    },
    /// A session's close: the `_sessions` row and its links, then worker,
    /// beliefs and session-heal.
    CloseSession {
        session: String,
        project_dir: PathBuf,
        files: Vec<String>,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        problems: Option<String>,
        ended_at: String,
    },
}

impl Job {
    fn axil(args: &[&str], stdin: Option<&str>) -> Self {
        Self::Axil {
            args: args.iter().map(|a| a.to_string()).collect(),
            stdin: stdin.map(str::to_string),
        }
    }
}

/// Hook jobs wait next to the database they change: `<db dir>/hook-queue/`.
fn queue_dir(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new("."))
        .join("hook-queue")
}

/// Write `job` to the queue atomically (temp file, then rename), so the
/// drainer never reads half a job. Names sort by creation time, which is the
/// order jobs run in.
fn write_job(db: &Path, job: &Job) -> bool {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let dir = queue_dir(db);
    let Ok(body) = serde_json::to_vec(job) else {
        return false;
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    let name = format!(
        "{}-{}-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%S%.9fZ"),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, body).is_ok()
        && std::fs::rename(&tmp, dir.join(format!("{name}.job"))).is_ok()
}

fn spawn_drainer(exe: &Path, db: &Path, cwd: &Path) {
    let args = [
        "--db".to_string(),
        db.to_string_lossy().into_owned(),
        "hook".into(),
        "drain".into(),
    ];
    spawn_detached(exe, &args, cwd, &queue_dir(db).join("drain.log"));
}

/// `axil hook drain`: run the queued hook writes in order, one drainer at a
/// time. Every enqueue starts one detached; a drainer that finds the lock
/// held exits at once, since the running one will reach the new job.
pub(crate) fn drain(db: &Path) -> Result<i32> {
    let exe = std::env::current_exe()?;
    let dir = queue_dir(db);
    loop {
        let Some(lock) = DrainLock::acquire(&dir) else {
            return Ok(0);
        };
        set_aside_interrupted(&dir);
        while let Some(job) = next_job(&dir) {
            lock.touch();
            run_job_file(&exe, db, &dir, &job);
        }
        drop(lock);
        // A job queued after the last scan but before the lock dropped saw a
        // live drainer and left it to us: look once more.
        if next_job(&dir).is_none() {
            return Ok(0);
        }
    }
}

/// `hook-queue/drain.lock`, created exclusively and refreshed per job.
struct DrainLock(PathBuf);

impl DrainLock {
    fn acquire(dir: &Path) -> Option<Self> {
        let path = dir.join("drain.lock");
        for _ in 0..2 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => {
                    let lock = Self(path);
                    lock.touch();
                    return Some(lock);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age.as_secs() > DRAIN_LOCK_STALE_SECS);
                    if !stale {
                        return None;
                    }
                    let _ = std::fs::remove_file(&path);
                }
                Err(_) => return None,
            }
        }
        None
    }

    fn touch(&self) {
        let _ = std::fs::write(&self.0, std::process::id().to_string());
    }
}

impl Drop for DrainLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn queue_files(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == ext))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

fn next_job(dir: &Path) -> Option<PathBuf> {
    queue_files(dir, "job").into_iter().next()
}

/// With the lock held, a `.running` job belonged to a drainer that died
/// mid-job, so some of its writes may have landed. It is never retried
/// (that could write twice): it moves to `interrupted/` for a person to
/// look at.
fn set_aside_interrupted(dir: &Path) {
    for path in queue_files(dir, "running") {
        let dest = dir.join("interrupted");
        let _ = std::fs::create_dir_all(&dest);
        if let Some(name) = path.file_name() {
            let _ = std::fs::rename(&path, dest.join(name));
            drain_log(
                dir,
                &format!(
                    "set aside interrupted job {}, not retried",
                    name.to_string_lossy()
                ),
            );
        }
    }
}

fn drain_log(dir: &Path, line: &str) {
    let path = dir.join("drain.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > DRAIN_LOG_MAX_BYTES) {
        let _ = std::fs::remove_file(&path);
    }
    append_line(&path, &format!("{} {line}", now_iso()));
}

/// Claim a job (rename to `.running`, the in-flight mark), run it, and
/// remove it. A failed job is logged and dropped, not retried.
fn run_job_file(exe: &Path, db: &Path, dir: &Path, job_path: &Path) {
    let running = job_path.with_extension("running");
    if std::fs::rename(job_path, &running).is_err() {
        return;
    }
    let name = running
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let result = std::fs::read(&running)
        .ok()
        .and_then(|body| serde_json::from_slice::<Job>(&body).ok())
        .ok_or_else(|| "unreadable job file".to_string())
        .and_then(|job| run_job(exe, db, &job));
    if let Err(e) = result {
        drain_log(dir, &format!("job {name} failed: {e}"));
    }
    let _ = std::fs::remove_file(&running);
}

fn run_job(exe: &Path, db: &Path, job: &Job) -> std::result::Result<(), String> {
    match job {
        Job::Axil { args, stdin } => run_write(exe, db, args, stdin.as_deref()).map(drop),
        Job::CloseSession {
            session,
            project_dir,
            files,
            content,
            problems,
            ended_at,
        } => {
            if !files.is_empty() {
                let entities: Value = Some(content.as_str())
                    .filter(|c| !c.is_empty())
                    .and_then(|c| {
                        run_capture_stdin(
                            Command::new(exe).args(["extract-entities", "-"]),
                            c.as_bytes(),
                        )
                    })
                    .and_then(|out| serde_json::from_str(out.trim()).ok())
                    .unwrap_or_else(|| json!([]));
                let entity_count = entities.as_array().map_or(0, Vec::len);
                let record = json!({
                    "session": session,
                    "files_changed": files,
                    "file_count": files.len(),
                    "entities": entities,
                    "entity_count": entity_count,
                    "ended_at": ended_at,
                });
                let stored = run_write(
                    exe,
                    db,
                    &["store".into(), "_sessions".into(), record.to_string()],
                    None,
                )?;
                if let Some(id) = serde_json::from_str::<Value>(stored.trim())
                    .ok()
                    .and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_string))
                {
                    let _ = run_write(exe, db, &["auto-link".into(), id], None);
                }
                // Consolidation, connections, inference, decay — then beliefs.
                let _ = run_write(exe, db, &["worker".into(), "run".into()], None);
                let _ = run_write(exe, db, &["beliefs".into(), "--generate".into()], None);
            }
            session_heal(exe, db, session, project_dir, problems.as_deref());
            Ok(())
        }
    }
}

/// Run one axil command against `db`, waiting while another process holds
/// the writer lock. Retrying a busy open is safe: it fails before anything
/// is written.
fn run_write(
    exe: &Path,
    db: &Path,
    args: &[String],
    stdin: Option<&str>,
) -> std::result::Result<String, String> {
    use std::io::Write as _;
    let mut delay = Duration::from_millis(250);
    let mut waited = Duration::ZERO;
    loop {
        let child = Command::new(exe)
            .arg("--db")
            .arg(db)
            .args(args)
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = child.map_err(|e| e.to_string())?;
        if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
            let _ = pipe.write_all(text.as_bytes());
        }
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if out.status.success() {
            return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
        }
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("database busy") && waited < BUSY_WAIT_MAX {
            std::thread::sleep(delay);
            waited += delay;
            delay = (delay * 2).min(Duration::from_secs(8));
            continue;
        }
        let command = args.first().map(String::as_str).unwrap_or("");
        let reason = err
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        return Err(format!("`axil {command}` failed: {reason}"));
    }
}

/// Run session-heal and act on its hints. The one auto-fix today:
/// `stale_structural_index` spawns a detached `axil index` so the next
/// session's queries hit a fresh index. The lock file throttles repeats
/// within the 5-minute stale window.
fn session_heal(exe: &Path, db: &Path, session: &str, project_dir: &Path, problems: Option<&str>) {
    let mut args: Vec<String> = vec!["session-heal".into(), "--session".into(), session.into()];
    let problems_file = problems.map(|text| {
        let path = queue_dir(db).join(format!("{}.problems", sanitize_id(session)));
        let _ = std::fs::write(&path, text);
        path
    });
    if let Some(path) = &problems_file {
        args.push("--problems-file".into());
        args.push(path.to_string_lossy().into_owned());
    }
    let out = run_write(exe, db, &args, None);
    if let Some(path) = &problems_file {
        let _ = std::fs::remove_file(path);
    }
    let Ok(out) = out else { return };
    let Ok(report) = serde_json::from_str::<Value>(out.trim()) else {
        return;
    };
    let stale = report
        .get("hints")
        .and_then(Value::as_array)
        .is_some_and(|hints| {
            hints
                .iter()
                .any(|h| h.get("kind").and_then(Value::as_str) == Some("stale_structural_index"))
        });
    if !stale {
        return;
    }

    let axil_dir = project_dir.join(".axil");
    let lock = axil_dir.join("index-refresh.lock");
    let log = axil_dir.join("index-refresh.log");
    let recent = std::fs::metadata(&lock)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|m| m.elapsed().ok())
        .is_some_and(|age| age.as_secs() < 300);
    if recent {
        return;
    }
    let _ = std::fs::create_dir_all(&axil_dir);
    let _ = std::fs::write(&lock, chrono::Utc::now().timestamp().to_string());
    let args = [
        "--db".to_string(),
        db.to_string_lossy().into_owned(),
        "index".into(),
        project_dir.to_string_lossy().into_owned(),
    ];
    if !spawn_detached(exe, &args, project_dir, &log) {
        let _ = std::fs::remove_file(&lock);
    }
}

// ── Pure helpers ─────────────────────────────────────────────────────

/// PascalCase event name shared by the Claude/Codex/Droid wire format —
/// required inside hookSpecificOutput (Codex validates it as a const).
fn claude_event_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::UserPrompt => "UserPromptSubmit",
        EventKind::SessionStart => "SessionStart",
        EventKind::PreTool => "PreToolUse",
        EventKind::PostTool => "PostToolUse",
        EventKind::Stop => "Stop",
        EventKind::SessionEnd => "SessionEnd",
        // Antigravity-only; never appears in a Claude-style response.
        EventKind::PreModel => "PreInvocation",
    }
}

/// The env var each harness sets to the project root.
fn project_dir_env_var(dialect: Dialect) -> Option<&'static str> {
    match dialect {
        Dialect::Claude => Some("CLAUDE_PROJECT_DIR"),
        // Codex/Copilot/Droid pass cwd in the payload; no dedicated env var
        // is documented for them.
        _ => None,
    }
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn nested_str(v: &Value, path: &[&str]) -> Option<String> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

fn nested_i64(v: &Value, path: &[&str]) -> Option<i64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_i64()
}

/// Exit code from `tool_response.exitCode` / `exit_code`, tolerating both
/// number and string encodings.
fn response_exit_code(input: &Value) -> i64 {
    for root in ["tool_response", "toolResult", "tool_output"] {
        let Some(resp) = input.get(root) else {
            continue;
        };
        for key in ["exitCode", "exit_code"] {
            if let Some(v) = resp.get(key) {
                if let Some(n) = v.as_i64() {
                    return n;
                }
                if let Some(s) = v.as_str() {
                    if let Ok(n) = s.trim().parse() {
                        return n;
                    }
                }
            }
        }
    }
    0
}

/// First non-empty string among the response object's `<keys>`.
fn response_str(input: &Value, keys: &[&str]) -> String {
    for root in ["tool_response", "toolResult", "tool_output"] {
        let Some(resp) = input.get(root) else {
            continue;
        };
        for key in keys {
            if let Some(s) = resp.get(*key).and_then(Value::as_str) {
                if !s.is_empty() {
                    return s.to_string();
                }
            }
        }
    }
    String::new()
}

/// Session IDs become temp-file names — keep them filesystem-safe.
fn sanitize_id(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Walk up from the project dir looking for `.axil/memory.axil`.
fn find_db(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join(".axil").join("memory.axil");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// Skip non-code paths: logs, lockfiles, build artifacts, the DB itself.
fn is_skipped_path(path: &str) -> bool {
    let p = path.replace('\\', "/");
    p.contains(".axil")
        || p.ends_with(".lock")
        || p.ends_with(".log")
        || p.contains("/node_modules/")
        || p.contains("/target/")
        || p.contains("/.git/")
}

/// Detect a broad repo search and extract what it is looking for.
/// Returns (is_repo_search, query).
///
/// Only a command in its own right counts: the first command of a pipeline,
/// or one joined by `&&`, `||`, `;` or `&`. A `grep` that filters another
/// command's output is not repo discovery, and heredoc bodies are ignored.
/// The query is the search tool's own pattern argument, never whatever
/// string happens to be quoted first in the command. `ls` and `tree` count
/// as discovery but carry no query.
fn detect_repo_search(cmd: &str) -> (bool, String) {
    let mut discovery = false;
    for words in primary_commands(&strip_heredoc_bodies(cmd)) {
        let words = skip_wrappers(&words);
        let Some(first) = words.first() else {
            continue;
        };
        let tool = first.rsplit('/').next().unwrap_or(first);
        let args = &words[1..];
        let query = match tool {
            "git" if args.first().map(String::as_str) == Some("grep") => {
                grep_pattern(&args[1..], GREP_VALUE_FLAGS)
            }
            "grep" | "egrep" | "fgrep" => grep_pattern(args, GREP_VALUE_FLAGS),
            "rg" => grep_pattern(args, RG_VALUE_FLAGS),
            "fd" | "fdfind" => args.iter().find(|a| !a.starts_with('-')).cloned(),
            "find" => find_name(args),
            "ls" | "tree" => {
                discovery = true;
                continue;
            }
            _ => continue,
        };
        let query = query.map(|p| pattern_to_query(&p)).unwrap_or_default();
        return (true, query);
    }
    (discovery, String::new())
}

/// Options that take a separate value argument, per search tool. A value
/// is never the pattern (`grep -A 3 foo` searches for `foo`).
const GREP_VALUE_FLAGS: &[&str] = &[
    "-e",
    "-f",
    "-m",
    "-A",
    "-B",
    "-C",
    "-d",
    "-D",
    "--regexp",
    "--file",
    "--max-count",
    "--context",
    "--after-context",
    "--before-context",
    "--include",
    "--exclude",
    "--exclude-dir",
    "--label",
];
const RG_VALUE_FLAGS: &[&str] = &[
    "-e",
    "-f",
    "-g",
    "-t",
    "-T",
    "-A",
    "-B",
    "-C",
    "-m",
    "-M",
    "-j",
    "-r",
    "-E",
    "-d",
    "--regexp",
    "--file",
    "--glob",
    "--iglob",
    "--type",
    "--type-not",
    "--type-add",
    "--max-count",
    "--max-columns",
    "--threads",
    "--replace",
    "--encoding",
    "--sort",
    "--sortr",
    "--colors",
    "--context",
    "--after-context",
    "--before-context",
    "--max-depth",
];

/// The pattern of a grep-like command: an explicit `-e`/`--regexp`, or else
/// the first argument that is neither an option nor an option's value.
fn grep_pattern(args: &[String], value_flags: &[&str]) -> Option<String> {
    let mut iter = args.iter();
    let mut first_positional = None;
    while let Some(arg) = iter.next() {
        if arg == "--" {
            return first_positional.or_else(|| iter.next().cloned());
        }
        if arg == "-e" || arg == "--regexp" {
            return iter.next().cloned();
        }
        if let Some(p) = arg.strip_prefix("--regexp=") {
            return Some(p.to_string());
        }
        if arg.starts_with('-') && arg.len() > 1 {
            if value_flags.contains(&arg.as_str()) {
                iter.next();
            }
            continue;
        }
        first_positional.get_or_insert_with(|| arg.clone());
    }
    first_positional
}

/// `find -name <glob>` (or `-iname`/`-path`), when the glob names something
/// more specific than a file extension.
fn find_name(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if matches!(arg.as_str(), "-name" | "-iname" | "-path" | "-ipath") {
            let name = iter.next()?.replace(['*', '?'], " ");
            let name = name.trim();
            return (!name.is_empty() && !name.starts_with('.')).then(|| name.to_string());
        }
    }
    None
}

/// Turn a search pattern into search words: alternatives (`a\|b`, `a|b`)
/// become separate words, regex syntax becomes spaces, and at most three
/// alternatives are kept.
fn pattern_to_query(pattern: &str) -> String {
    let alternatives = pattern.split("\\|").flat_map(|a| a.split('|'));
    let words: Vec<String> = alternatives
        .map(|alt| {
            let mut out = String::new();
            let mut chars = alt.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    // An escape: `\b`, `\w`, `\(` and friends are regex syntax,
                    // but an escaped `.`, `_`, `-`, `/` or `:` is part of a name.
                    '\\' => match chars.next() {
                        Some(e @ ('.' | '_' | '-' | '/' | ':')) => out.push(e),
                        Some(_) => out.push(' '),
                        None => {}
                    },
                    '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' => {
                        out.push(' ')
                    }
                    '.' if matches!(chars.peek(), Some('*' | '+' | '?')) => out.push(' '),
                    _ => out.push(c),
                }
            }
            out.split_whitespace().collect::<Vec<_>>().join(" ")
        })
        .filter(|w| !w.is_empty())
        .take(3)
        .collect();
    words.join(" ")
}

/// Cut heredoc bodies: everything after the line holding the `<<` belongs
/// to that command's stdin, not to the shell command line.
fn strip_heredoc_bodies(cmd: &str) -> String {
    match cmd.find("<<") {
        Some(at) => match cmd[at..].find('\n') {
            Some(nl) => cmd[..at + nl].to_string(),
            None => cmd.to_string(),
        },
        None => cmd.to_string(),
    }
}

/// Split a command line into its commands, as word lists, keeping only the
/// ones that are not the receiving end of a pipe. Quotes and backslash
/// escapes are honoured; `(`/`)` also separate commands (subshells).
fn primary_commands(cmd: &str) -> Vec<Vec<String>> {
    let mut commands = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut piped = false;
    let mut quote: Option<char> = None;
    let mut chars = cmd.chars().peekable();

    let mut end_command = |words: &mut Vec<String>, piped: bool| {
        if !piped && !words.is_empty() {
            commands.push(std::mem::take(words));
        }
        words.clear();
    };

    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            match c {
                c if c == q => quote = None,
                // In double quotes a backslash escapes only these; before
                // anything else it is kept, as the shell keeps it.
                '\\' if q == '"' => match chars.peek() {
                    Some('$' | '`' | '"' | '\\' | '\n') => word.push(chars.next().unwrap_or('\\')),
                    _ => word.push('\\'),
                },
                _ => word.push(c),
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                in_word = true;
            }
            '\\' => {
                if let Some(n) = chars.next() {
                    word.push(n);
                    in_word = true;
                }
            }
            c if c.is_whitespace() && c != '\n' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '|' | '&' | ';' | '\n' | '(' | ')' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
                // `||` and `&&` are one operator; a lone `|` pipes into the
                // next command, which then isn't primary.
                let doubled = matches!(c, '|' | '&') && chars.peek() == Some(&c);
                if doubled {
                    chars.next();
                }
                end_command(&mut words, piped);
                piped = c == '|' && !doubled;
            }
            _ => {
                word.push(c);
                in_word = true;
            }
        }
    }
    if in_word {
        words.push(word);
    }
    end_command(&mut words, piped);
    commands
}

/// Drop `VAR=value` prefixes and command wrappers (`rtk proxy`, `time`,
/// `sudo`, …) so the real command comes first.
fn skip_wrappers(words: &[String]) -> Vec<String> {
    let mut rest = words;
    loop {
        match rest.first().map(String::as_str) {
            Some(w) if w.contains('=') && !w.starts_with('-') && !w.starts_with('=') => {
                rest = &rest[1..];
            }
            Some("rtk") if rest.get(1).map(String::as_str) == Some("proxy") => rest = &rest[2..],
            Some("rtk" | "command" | "time" | "sudo" | "nice" | "nohup" | "env") => {
                rest = &rest[1..];
            }
            _ => return rest.to_vec(),
        }
    }
}

/// First double-quoted arg, then first single-quoted arg.
fn first_quoted_arg(cmd: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let mut parts = cmd.split(quote);
        parts.next()?; // text before the first quote
        if let Some(inner) = parts.next() {
            if !inner.is_empty() {
                return Some(inner.to_string());
            }
        }
    }
    None
}

/// Queries that look like code (identifiers, paths, Rust keywords) route to
/// the structural index; natural language goes to full-text search.
fn is_code_like_query(q: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "fn ", "impl ", "struct ", "trait ", "pub ", "async ", "mod ", "use ",
    ];
    if PREFIXES.iter().any(|p| q.starts_with(p)) {
        return true;
    }
    if q.contains('_') || q.contains("::") {
        return true;
    }
    // camelCase / mixedCase: a lowercase letter immediately followed by uppercase.
    q.as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase())
}

/// Empty-result sniffer for axil read commands: blank stdout, a JSON empty
/// array, or the textual sentinels axil prints.
fn is_empty_axil_output(out: &str) -> bool {
    let t = out.trim();
    t.is_empty()
        || t == "[]"
        || t == "(no results)"
        || t.starts_with("(no code proxies matched")
        || t.starts_with("(no matches)")
}

/// The axil subcommands a command line runs, in order: for each `axil`
/// word (or a path ending in it), the first word after it that is neither
/// an option nor a global option's value (`axil --db x store` is `store`).
fn axil_subcommands(cmd: &str) -> Vec<String> {
    const VALUE_OPTIONS: &[&str] = &["--db", "--format", "--agent"];
    let mut found = Vec::new();
    for words in primary_commands(&strip_heredoc_bodies(cmd)) {
        let words = skip_wrappers(&words);
        let mut iter = words.iter();
        while let Some(word) = iter.next() {
            let base = word.rsplit(['/', '\\']).next().unwrap_or(word);
            if base != "axil" && base != "axil.exe" {
                continue;
            }
            while let Some(next) = iter.next() {
                if VALUE_OPTIONS.contains(&next.as_str()) {
                    iter.next();
                } else if !next.starts_with('-') {
                    found.push(next.clone());
                    break;
                }
            }
        }
    }
    found
}

/// True when the command line itself runs `git commit` (as a command, not
/// text inside a heredoc, an echo or a commit message).
fn runs_git_commit(cmd: &str) -> bool {
    primary_commands(&strip_heredoc_bodies(cmd))
        .iter()
        .any(|words| {
            let words = skip_wrappers(words);
            let mut rest = words.iter().map(String::as_str);
            if rest.next() != Some("git") {
                return false;
            }
            // Skip git's own options: `git -C dir commit`, `git -c k=v commit`.
            while let Some(word) = rest.next() {
                match word {
                    "-C" | "-c" => {
                        rest.next();
                    }
                    w if w.starts_with('-') => {}
                    w => return w == "commit",
                }
            }
            false
        })
}

/// Manifests and lockfiles `axil deps` reads, across its five ecosystems.
fn is_dependency_manifest(path: &str) -> bool {
    const NAMES: &[&str] = &[
        "Cargo.toml",
        "Cargo.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pyproject.toml",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "go.mod",
        "pom.xml",
    ];
    let p = path.replace('\\', "/");
    let vendored = p.contains("/node_modules/") || p.contains("/target/") || p.contains("/.git/");
    let name = p.rsplit('/').next().unwrap_or(&p);
    !vendored && NAMES.contains(&name)
}

/// The first axil subcommand of a command line, or "".
fn extract_axil_subcmd(cmd: &str) -> String {
    axil_subcommands(cmd).into_iter().next().unwrap_or_default()
}

/// Subcommands that write the agent's knowledge (the store heartbeat).
const STORE_SUBCOMMANDS: &[&str] = &[
    "store",
    "observe",
    "believe",
    "checkpoint",
    "remember",
    "resolve",
    "know",
];
/// Subcommands that read memory; any of them satisfies the search gate.
const RECALL_SUBCOMMANDS: &[&str] = &[
    "recall",
    "boot",
    "recall-for-file",
    "recall-for-entity",
    "code-search",
    "code-context",
    "fts",
    "ask",
    "search",
    "know-about",
    "dep-docs",
];

/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// FNV-1a 64 — stable, dependency-free hash for sentinel filenames.
fn fnv1a(s: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:x}")
}

fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Append one line with a single write, so concurrent appenders (parallel
/// hooks bumping a counter) never interleave or lose each other's lines.
fn append_line(path: &Path, line: &str) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(format!("{line}\n").as_bytes());
    }
}

/// Put a background child in its own process group, so a harness that kills
/// the hook's whole group on timeout doesn't take the child with it.
fn own_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = cmd;
}

fn run_capture(cmd: &mut Command) -> Option<String> {
    let out = cmd.stderr(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn run_capture_stdin(cmd: &mut Command, stdin_bytes: &[u8]) -> Option<String> {
    use std::io::Write as _;
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_bytes);
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Spawn a child that survives this hook's exit, appending its output to
/// `log`. Unix goes through `nohup` (double-detach — direct spawn was
/// observed dying with the parent); Windows uses process-creation flags.
fn spawn_detached(exe: &Path, args: &[String], cwd: &Path, log: &Path) -> bool {
    #[cfg(unix)]
    {
        fn sh_quote(s: &str) -> String {
            format!("'{}'", s.replace('\'', "'\\''"))
        }
        let mut parts: Vec<String> = vec!["nohup".into(), sh_quote(&exe.to_string_lossy())];
        parts.extend(args.iter().map(|a| sh_quote(a)));
        parts.push(format!(">> {} 2>&1", sh_quote(&log.to_string_lossy())));
        parts.push("</dev/null &".into());
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(parts.join(" "))
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        own_process_group(&mut cmd);
        cmd.spawn().and_then(|mut c| c.wait()).is_ok()
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        const FLAGS: u32 = 0x0000_0008 | 0x0000_0200 | 0x0800_0000;
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .ok();
        let (out_io, err_io) = match log_file {
            Some(f) => match f.try_clone() {
                Ok(f2) => (Stdio::from(f), Stdio::from(f2)),
                Err(_) => (Stdio::null(), Stdio::null()),
            },
            None => (Stdio::null(), Stdio::null()),
        };
        Command::new(exe)
            .args(args)
            .current_dir(cwd)
            .creation_flags(FLAGS)
            .stdin(Stdio::null())
            .stdout(out_io)
            .stderr(err_io)
            .spawn()
            .is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (exe, args, cwd, log);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_paths_cover_artifacts_and_db() {
        for p in [
            "/repo/.axil/memory.axil",
            "/repo/db.axil.vec",
            "/repo/Cargo.lock",
            "/repo/build.log",
            "/repo/node_modules/x/index.js",
            "/repo/target/debug/axil",
            "/repo/.git/HEAD",
            r"C:\repo\target\debug\foo.rs",
        ] {
            assert!(is_skipped_path(p), "{p} should be skipped");
        }
        for p in ["/repo/src/main.rs", "/repo/docs/guide.md"] {
            assert!(!is_skipped_path(p), "{p} should not be skipped");
        }
    }

    #[test]
    fn repo_search_detection_and_query_extraction() {
        let search = |cmd: &str| detect_repo_search(cmd);
        assert_eq!(
            search(r#"rg "hnsw recall" src/"#),
            (true, "hnsw recall".into())
        );
        assert_eq!(
            search("grep -r 'adaptive_ef' crates/"),
            (true, "adaptive_ef".into())
        );
        assert_eq!(
            search("git grep install_agent_integrations"),
            (true, "install_agent_integrations".into())
        );
        assert_eq!(search("ls -la src/"), (true, String::new()));
        assert!(!search("cargo test --workspace").0);
        // Token-aware: substrings of other words must not trigger.
        assert!(!search("cargo run --example energy").0);
        assert!(!search("pgrep axil").0);
    }

    /// The query is the search tool's own pattern, never the first quoted
    /// string anywhere in the command.
    #[test]
    fn repo_search_query_is_the_tools_pattern() {
        let query = |cmd: &str| detect_repo_search(cmd).1;
        // A label echoed before the search.
        assert_eq!(
            query(r#"echo "== BGE prefix" && rg -n "embed_query" crates/"#),
            "embed_query"
        );
        // Option values are not the pattern.
        assert_eq!(query("grep -A 3 -m 5 fn_name src/lib.rs"), "fn_name");
        assert_eq!(query("rg -g '*.rs' -t rust HookCtx"), "HookCtx");
        assert_eq!(query("grep -e needle -r ."), "needle");
        // Wrappers and absolute paths.
        assert_eq!(
            query(r#"rtk proxy grep -n "record_slow_query" crates/"#),
            "record_slow_query"
        );
        assert_eq!(query("/usr/bin/grep -rn open_err crates/"), "open_err");
        // Alternations become words; regex syntax is dropped.
        assert_eq!(
            query(r#"grep -n "fn attach_detected_engines\|fn open_with_all_detected" main.rs"#),
            "fn attach_detected_engines fn open_with_all_detected"
        );
        assert_eq!(query(r#"rg "^pub fn \w+_probe\(" src"#), "pub fn _probe");
        // find: a specific name counts; an extension glob does not.
        assert_eq!(query("find . -name '*hook_brain*'"), "hook_brain");
        assert_eq!(
            detect_repo_search("find . -name '*.rs'"),
            (true, String::new())
        );
    }

    #[test]
    fn git_commit_is_detected_only_as_a_command() {
        assert!(runs_git_commit("git commit -m 'fix'"));
        assert!(runs_git_commit("git add a && git commit -q -F -"));
        assert!(runs_git_commit("git -C repo commit --amend --no-edit"));
        assert!(!runs_git_commit("git log --oneline -3"));
        assert!(!runs_git_commit(r#"echo "run git commit next""#));
        assert!(!runs_git_commit("cat <<'EOF'\ngit commit -m x\nEOF"));
        assert!(!runs_git_commit("grep -n 'git commit' notes.md"));
    }

    #[test]
    fn repo_search_ignores_filters_and_heredocs() {
        // grep filtering another command's output is not repo discovery.
        assert!(!detect_repo_search("cargo test 2>&1 | grep FAILED").0);
        assert!(!detect_repo_search("git log --oneline | head -5").0);
        // A heredoc body can hold anything; only the command line counts.
        let heredoc = "python3 - <<'EOF'\np = \"crates/axil-core/src/db.rs\"\nrg foo\nEOF";
        assert!(!detect_repo_search(heredoc).0);
        // A search after `&&` or `;` is still a command of its own.
        assert_eq!(
            detect_repo_search("cd crates && rg -l Busy"),
            (true, "Busy".into())
        );
        assert_eq!(
            detect_repo_search("cargo check; grep -rn DrainLock src"),
            (true, "DrainLock".into())
        );
    }

    #[test]
    fn code_like_query_routing() {
        for q in [
            "fn adaptive_ef",
            "install_agent_integrations",
            "axil_core::boot",
            "HookCtx",
            "camelCaseName",
        ] {
            assert!(is_code_like_query(q), "{q} should be code-like");
        }
        for q in ["hnsw recall quality", "release workflow macos"] {
            assert!(!is_code_like_query(q), "{q} should be prose");
        }
    }

    #[test]
    fn empty_axil_output_sniffer() {
        for out in ["", "  ", "[]", " [] ", "(no results)", "(no matches) for q"] {
            assert!(is_empty_axil_output(out), "{out:?} should read as empty");
        }
        assert!(is_empty_axil_output("(no code proxies matched 'q')"));
        assert!(!is_empty_axil_output("[{\"id\":\"x\"}]"));
        assert!(!is_empty_axil_output("hit: src/main.rs:42"));
    }

    #[test]
    fn axil_subcommand_extraction() {
        assert_eq!(extract_axil_subcmd("axil recall \"q\" --top-k 5"), "recall");
        assert_eq!(
            extract_axil_subcmd("./target/release/axil code-search q"),
            "code-search"
        );
        // Global options and their values are skipped.
        assert_eq!(
            extract_axil_subcmd("axil --db x.axil store errors '{}'"),
            "store"
        );
        assert_eq!(extract_axil_subcmd("cargo build"), "");
        assert_eq!(
            axil_subcommands("axil store decisions '{}' && axil recall \"x\""),
            vec!["store".to_string(), "recall".to_string()]
        );
        // A heredoc body is data, not commands.
        assert!(axil_subcommands("cat <<'EOF'\naxil store x\nEOF").is_empty());
    }

    #[test]
    fn exit_code_tolerates_string_number_and_roots() {
        assert_eq!(
            response_exit_code(&json!({"tool_response": {"exitCode": 1}})),
            1
        );
        assert_eq!(
            response_exit_code(&json!({"tool_response": {"exit_code": "2"}})),
            2
        );
        assert_eq!(
            response_exit_code(&json!({"toolResult": {"exitCode": 3}})),
            3
        );
        assert_eq!(response_exit_code(&json!({"tool_response": {}})), 0);
        assert_eq!(response_exit_code(&json!({})), 0);
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "aé漢字x";
        for max in 0..=s.len() {
            let t = truncate_utf8(s, max);
            assert!(t.len() <= max);
            assert!(s.starts_with(t));
        }
    }

    #[test]
    fn sanitized_session_ids_are_path_safe() {
        assert_eq!(sanitize_id("abc-123_D.4"), "abc-123_D.4");
        assert_eq!(sanitize_id("../../etc/passwd"), "..-..-etc-passwd");
        assert_eq!(sanitize_id("a b/c"), "a-b-c");
    }

    #[test]
    fn fnv_hash_is_stable() {
        assert_eq!(fnv1a("src/main.rs"), fnv1a("src/main.rs"));
        assert_ne!(fnv1a("a"), fnv1a("b"));
    }

    #[test]
    fn first_quoted_arg_prefers_double_quotes() {
        assert_eq!(
            first_quoted_arg(r#"rg "hello world" 'src'"#),
            Some("hello world".into())
        );
        assert_eq!(first_quoted_arg("rg 'single'"), Some("single".into()));
        assert_eq!(first_quoted_arg("rg plain"), None);
    }

    // ── Dialect parsing ───────────────────────────────────────────────

    #[test]
    fn claude_events_map_to_canonical_kinds() {
        let ev = parse_claude(
            &json!({
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "cwd": "/repo",
                "tool_name": "Edit",
                "tool_input": {"file_path": "/repo/src/a.rs", "new_string": "x"}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PreTool);
        assert_eq!(
            ev.tool,
            Some(ToolAction::FileEdit {
                path: "/repo/src/a.rs".into(),
                snippet: Some("x".into())
            })
        );

        let ev = parse_claude(
            &json!({"hook_event_name": "Stop", "session_id": "s1", "stop_hook_active": true}),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::Stop);
        assert!(ev.stop_hook_active);
    }

    #[test]
    fn claude_todowrite_counts_completed() {
        let ev = parse_claude(
            &json!({
                "hook_event_name": "PostToolUse",
                "session_id": "s1",
                "tool_name": "TodoWrite",
                "tool_input": {"todos": [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "pending"},
                    {"content": "c", "status": "completed"}
                ]}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.tool, Some(ToolAction::Todo { completed_count: 2 }));
    }

    #[test]
    fn codex_shell_accepts_argv_arrays_and_apply_patch() {
        // Codex's shell tool is literally "Bash"; tool_input is schema-any,
        // so both string and argv-array commands must parse.
        let ev = parse_codex(
            &json!({
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "tool_name": "Bash",
                "tool_input": {"command": ["rg", "adaptive_ef", "src/"]}
            }),
            None,
        )
        .unwrap();
        match ev.tool {
            Some(ToolAction::Shell { ref command, .. }) => {
                assert_eq!(command, "rg adaptive_ef src/")
            }
            other => panic!("expected Shell, got {other:?}"),
        }

        let patch = "*** Begin Patch\n*** Update File: src/lib.rs\n@@\n-a\n+b\n*** End Patch";
        let ev = parse_codex(
            &json!({
                "hook_event_name": "PostToolUse",
                "session_id": "s1",
                "tool_name": "apply_patch",
                "tool_input": {"input": patch}
            }),
            None,
        )
        .unwrap();
        match ev.tool {
            Some(ToolAction::FileEdit { ref path, .. }) => assert_eq!(path, "src/lib.rs"),
            other => panic!("expected FileEdit, got {other:?}"),
        }
    }

    #[test]
    fn copilot_pascalcase_snake_payloads_parse_like_claude() {
        // Axil registers PascalCase events → snake_case fields plus
        // hook_event_name, with Copilot's lowercase runtime tool names.
        let ev = parse_copilot(
            &json!({
                "hook_event_name": "PreToolUse",
                "session_id": "s1",
                "cwd": "/repo",
                "tool_name": "bash",
                "tool_input": {"command": "ls -la"}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PreTool);
        match ev.tool {
            Some(ToolAction::Shell { ref command, .. }) => assert_eq!(command, "ls -la"),
            other => panic!("expected Shell, got {other:?}"),
        }

        let ev = parse_copilot(
            &json!({"hook_event_name": "SessionStart", "session_id": "s1", "cwd": "/repo"}),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::SessionStart);

        // agentStop's PascalCase alias is Stop.
        let ev = parse_copilot(&json!({"hook_event_name": "Stop", "session_id": "s1"}), None)
            .unwrap();
        assert_eq!(ev.kind, EventKind::Stop);
    }

    #[test]
    fn copilot_camelcase_and_string_encoded_args_tolerated() {
        // Hand-written camelCase configs: no event name in the payload
        // (needs --event) and toolArgs may be a JSON-encoded STRING —
        // the documented gotcha from the official test payload.
        let ev = parse_copilot(
            &json!({
                "sessionId": "s1",
                "cwd": "/tmp",
                "toolName": "bash",
                "toolArgs": "{\"command\":\"ls\"}"
            }),
            Some("preToolUse"),
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PreTool);
        match ev.tool {
            Some(ToolAction::Shell { ref command, .. }) => assert_eq!(command, "ls"),
            other => panic!("expected Shell, got {other:?}"),
        }

        // edit tool with object args.
        let ev = parse_copilot(
            &json!({
                "eventName": "postToolUse",
                "sessionId": "s1",
                "toolName": "edit",
                "toolArgs": {"path": "src/a.rs", "new_str": "x"}
            }),
            None,
        )
        .unwrap();
        match ev.tool {
            Some(ToolAction::FileEdit { ref path, .. }) => assert_eq!(path, "src/a.rs"),
            other => panic!("expected FileEdit, got {other:?}"),
        }
    }

    #[test]
    fn droid_execute_maps_to_shell() {
        let ev = parse_droid(
            &json!({
                "hook_event_name": "PostToolUse",
                "session_id": "s1",
                "tool_name": "Execute",
                "tool_input": {"command": "cargo test"},
                "tool_response": {"exitCode": 1, "stdout": "boom"}
            }),
            None,
        )
        .unwrap();
        match ev.tool {
            Some(ToolAction::Shell {
                ref command,
                exit_code,
                ..
            }) => {
                assert_eq!(command, "cargo test");
                assert_eq!(exit_code, 1);
            }
            other => panic!("expected Shell, got {other:?}"),
        }
    }

    #[test]
    fn patch_path_extraction() {
        assert_eq!(
            first_patch_path("*** Begin Patch\n*** Add File: a/b.txt\n+hi"),
            Some("a/b.txt".into())
        );
        assert_eq!(first_patch_path("no markers here"), None);
    }

    #[test]
    fn qwen_snake_case_payloads_and_tools_parse() {
        // Qwen kept the Claude-style event spellings + snake_case fields.
        let ev = parse_gemini(
            &json!({
                "hook_event_name": "PreToolUse",
                "session_id": "q1",
                "cwd": "/repo",
                "tool_name": "run_shell_command",
                "tool_input": {"command": "rg adaptive_ef src/"}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PreTool);
        match ev.tool {
            Some(ToolAction::Shell { ref command, .. }) => {
                assert_eq!(command, "rg adaptive_ef src/")
            }
            other => panic!("expected Shell, got {other:?}"),
        }

        // Legacy Gemini CLI aliases still map.
        let ev = parse_gemini(
            &json!({
                "hook_event_name": "AfterTool",
                "session_id": "q1",
                "tool_name": "write_file",
                "tool_input": {"file_path": "src/a.rs", "content": "x"}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PostTool);
        match ev.tool {
            Some(ToolAction::FileEdit { ref path, .. }) => assert_eq!(path, "src/a.rs"),
            other => panic!("expected FileEdit, got {other:?}"),
        }

        // Qwen's todo tool.
        let ev = parse_gemini(
            &json!({
                "hook_event_name": "PostToolUse",
                "session_id": "q1",
                "tool_name": "todo_write",
                "tool_input": {"todos": [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "pending"}
                ]}
            }),
            None,
        )
        .unwrap();
        assert_eq!(ev.tool, Some(ToolAction::Todo { completed_count: 1 }));

        // Stop carries stop_hook_active.
        let ev = parse_gemini(
            &json!({"hook_event_name": "Stop", "session_id": "q1", "stop_hook_active": true}),
            None,
        )
        .unwrap();
        assert!(ev.stop_hook_active);
    }

    #[test]
    fn antigravity_events_need_override_and_use_conversation_id() {
        // No event name in the payload → --event is mandatory.
        let payload = json!({
            "toolCall": {"name": "run_command",
                "args": {"CommandLine": "npm test", "Cwd": "/workspace/p"}},
            "stepIdx": 19,
            "conversationId": "ec33ebf9",
            "workspacePaths": ["/workspace/p"]
        });
        assert!(parse_antigravity(&payload, None).is_none());

        let ev = parse_antigravity(&payload, Some("PreToolUse")).unwrap();
        assert_eq!(ev.kind, EventKind::PreTool);
        assert_eq!(ev.session_id, "ec33ebf9");
        assert_eq!(ev.cwd.as_deref(), Some("/workspace/p"));
        match ev.tool {
            Some(ToolAction::Shell { ref command, .. }) => assert_eq!(command, "npm test"),
            other => panic!("expected Shell, got {other:?}"),
        }

        // PreInvocation maps to the PreModel channel.
        let ev = parse_antigravity(
            &json!({"conversationId": "c1", "workspacePaths": ["/w"], "invocationNum": 1}),
            Some("PreInvocation"),
        )
        .unwrap();
        assert_eq!(ev.kind, EventKind::PreModel);

        // PostToolUse carries only an optional error — surfaced as a
        // failed shell action so error capture runs.
        let ev = parse_antigravity(
            &json!({"conversationId": "c1", "workspacePaths": ["/w"], "stepIdx": 3,
                    "error": "command exited 1"}),
            Some("PostToolUse"),
        )
        .unwrap();
        match ev.tool {
            Some(ToolAction::Shell { exit_code, ref stdout, .. }) => {
                assert_eq!(exit_code, 1);
                assert_eq!(stdout, "command exited 1");
            }
            other => panic!("expected Shell failure, got {other:?}"),
        }
    }

    #[test]
    fn unknown_events_are_ignored() {
        assert!(parse_claude(
            &json!({"hook_event_name": "PreCompact", "session_id": "s1"}),
            None
        )
        .is_none());
        assert!(parse_copilot(&json!({"eventName": "notification", "sessionId": "s1"}), None).is_none());
    }

    /// Golden-fixture regression gate for the dialect field mappings.
    ///
    /// Each `tests/fixtures/hooks/<dialect>.json` holds representative hook
    /// payloads and the canonical parse they must produce. The `expect`
    /// shape is exactly what `axil hook capture` writes in its `parsed`
    /// block (same [`tool_summary`]), so a real captured payload can be
    /// dropped straight into a fixture as `{payload, expect}` to lock it.
    /// A tool contract that drifts (or a mapping that regresses) fails here.
    #[test]
    fn dialect_fixtures_parse_as_expected() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("hooks");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read fixtures dir {}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no hook fixtures found in {}", dir.display());

        let mut cases_run = 0usize;
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            let doc: Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display()));
            let dialect_str = doc["dialect"].as_str().expect("fixture missing `dialect`");
            let dialect =
                Dialect::parse(dialect_str).unwrap_or_else(|| panic!("unknown dialect '{dialect_str}' in {}", path.display()));

            for case in doc["cases"].as_array().expect("fixture missing `cases`") {
                let name = case["name"].as_str().unwrap_or("<unnamed>");
                let event_override = case["event"].as_str();
                let payload = &case["payload"];

                let ev = parse_event(dialect, payload, event_override);
                let actual = json!({
                    "event": ev.as_ref().map(|e| format!("{:?}", e.kind)),
                    "tool": ev.as_ref().and_then(|e| e.tool.as_ref()).map(tool_summary),
                });
                assert_eq!(
                    actual, case["expect"],
                    "\n{} :: {name}\n  payload: {payload}\n  expected: {}\n  actual:   {actual}",
                    path.display(),
                    case["expect"],
                );
                cases_run += 1;
            }
        }
        // Guard against an empty/renamed fixture set silently passing.
        assert!(cases_run >= 20, "expected >=20 fixture cases, ran {cases_run}");
    }
}
