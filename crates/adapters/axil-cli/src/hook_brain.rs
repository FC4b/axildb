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
//!   session start    — boot push (Claude Code has no such event, so the
//!                      first pre-tool call emulates it via a sentinel)
//!   pre file-edit    — recall-for-file + store nudge
//!   pre shell        — axil-first search gate / paired search
//!   post file-edit   — manifest + snippet accumulation
//!   post shell       — heartbeat, commit capture, error capture
//!   post file-read   — fallback capture after empty recalls
//!   post todo-update — store reminder when a todo completes
//!   stop             — narrative guard, session close, worker
//!
//! Every path is best-effort: a memory hook must never break the agent
//! loop, so child failures are swallowed and the process always exits 0
//! once the input parses.

use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::Result;
use serde_json::{json, Value};

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

pub(crate) fn run(dialect: &str, event_override: Option<&str>) -> Result<i32> {
    let dialect = parse_dialect(dialect)?;

    let mut raw = String::new();
    if std::io::stdin().read_to_string(&mut raw).is_err() || raw.trim().is_empty() {
        return Ok(0);
    }
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

    let mut raw = String::new();
    if std::io::stdin().read_to_string(&mut raw).is_err() || raw.trim().is_empty() {
        return Ok(0);
    }
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
        "PostToolUse" => EventKind::PostTool,
        "Stop" => EventKind::Stop,
        "SessionEnd" => EventKind::SessionEnd,
        _ => return None,
    };
    let tool_name = str_field(input, "tool_name").unwrap_or_default();
    let tool = match kind {
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
        _ => ToolAction::Other,
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
        Some(Self {
            dialect,
            event,
            sid,
            project_dir,
            exe,
            db,
            tmp: std::env::temp_dir(),
        })
    }

    fn dispatch(&self) -> Result<()> {
        match self.event.kind {
            EventKind::UserPrompt => self.on_user_prompt(),
            EventKind::SessionStart => {
                // A real session-start event: mark booted so a Claude-style
                // first-pre-tool emulation never double-boots.
                let _ = std::fs::write(self.sfile("booted"), "");
                self.boot_push();
                Ok(())
            }
            EventKind::PreModel => self.on_pre_model(),
            EventKind::PreTool => self.on_pre_tool(),
            EventKind::PostTool => self.on_post_tool(),
            // SessionEnd can't block, so it skips the narrative guard but
            // still closes the session (summary, worker, heal, cleanup).
            EventKind::Stop => self.on_stop(true),
            EventKind::SessionEnd => self.on_stop(false),
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
        self.tmp.join(format!("axil-session-{}.{}", self.sid, suffix))
    }

    /// Previous-session manifest, scoped per project so one repo's edited
    /// files never seed another repo's boot `--files`. Deliberately NOT
    /// prefixed `axil-session-<sid>.` so `cleanup_session_files` leaves it
    /// in place for the next session.
    fn prev_manifest(&self) -> PathBuf {
        let scope = fnv1a(&self.project_dir.to_string_lossy());
        self.tmp.join(format!("axil-prev-{scope}.manifest"))
    }

    /// Sweep every per-session temp file, including the one-per-file/query
    /// sentinel files (`.recalled-<hash>`, `.searched-<hash>`, …).
    fn cleanup_session_files(&self) {
        let prefix = format!("axil-session-{}.", self.sid);
        if let Ok(entries) = std::fs::read_dir(&self.tmp) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(prefix.as_str())
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
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
    // One JSON file per session: { stores, recalls, tools, errors }.
    // Races on concurrent hook runs at worst drop a count; never corrupt.

    fn bump_count(&self, key: &str) {
        let path = self.sfile("counts");
        let mut counts: Value = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| json!({}));
        let next = counts.get(key).and_then(Value::as_i64).unwrap_or(0) + 1;
        counts[key] = json!(next);
        let _ = std::fs::write(&path, counts.to_string());
    }

    fn read_count(&self, key: &str) -> i64 {
        std::fs::read_to_string(self.sfile("counts"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.get(key).and_then(Value::as_i64))
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

    /// Run an axil subcommand against the resolved DB; Some(stdout) on
    /// success. Child stderr is discarded — hook noise must not leak.
    fn axil_db_out(&self, args: &[&str]) -> Option<String> {
        let db = self.db.as_ref()?;
        run_capture(
            Command::new(&self.exe)
                .arg("--db")
                .arg(db)
                .args(args)
                .stdin(Stdio::null()),
        )
    }

    /// Same but feeding bytes to the child's stdin (`auto-capture -` etc.).
    fn axil_db_out_stdin(&self, args: &[&str], stdin_bytes: &[u8]) -> Option<String> {
        let db = self.db.as_ref()?;
        run_capture_stdin(
            Command::new(&self.exe).arg("--db").arg(db).args(args),
            stdin_bytes,
        )
    }

    /// DB-less axil call (e.g. `extract-entities -`).
    fn axil_out_stdin(&self, args: &[&str], stdin_bytes: &[u8]) -> Option<String> {
        run_capture_stdin(Command::new(&self.exe).args(args), stdin_bytes)
    }

    /// Count narrative records stored in the last hour.
    fn count_recent_narrative(&self) -> i64 {
        let Some(out) = self.axil_db_out(&["since", "1h"]) else {
            return 0;
        };
        let Ok(rows) = serde_json::from_str::<Value>(out.trim()) else {
            return 0;
        };
        rows.as_array()
            .map(|arr| {
                arr.iter()
                    .filter(|r| {
                        r.get("table")
                            .and_then(Value::as_str)
                            .map(|t| NARRATIVE_TABLES.contains(&t))
                            .unwrap_or(false)
                    })
                    .count() as i64
            })
            .unwrap_or(0)
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

        // Claude Code has no session-start hook event: the first tool call
        // of the session carries the boot. Dialects with a real
        // SessionStart already wrote the sentinel in dispatch().
        let booted = self.sfile("booted");
        if !booted.exists() {
            let _ = std::fs::write(&booted, "");
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

        // Banner to stderr: hook stdout is parsed as JSON, and mixing prose
        // into it corrupts the parse (the old bash hook printed the banner
        // to stdout — a latent bug).
        if let Some(banner) = self.axil_db_out(&["brain-banner"]) {
            if !banner.trim().is_empty() {
                eprint!("{banner}");
            }
        }
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let boot = self.axil_db_out(&arg_refs);

        // Opportunistic background refreshes. Both subcommands self-detach
        // (`--in-background`) and gate on staleness, so these return fast;
        // errors are silent — the brain hook must never block the agent.
        let _ = self.axil_db_out(&["scip", "refresh", "--if-stale", "--in-background", "--quiet"]);
        let _ = self.axil_db_out(&["maintain", "--if-stale", "--in-background", "--quiet"]);

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
            let _ = std::fs::write(&sentinel, "");
        }

        // 5-edit nudge: fires at every 5th edit with no narrative stored.
        if let Ok(manifest) = std::fs::read_to_string(self.sfile("manifest")) {
            let edit_count = manifest.lines().count();
            if edit_count >= 5 && edit_count % 5 == 0 && self.count_recent_narrative() == 0 {
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
        }
        .unwrap_or_default();

        if !is_empty_axil_output(&hits) {
            let _ = std::fs::write(&sentinel, "");
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
        if self
            .axil_db_out(&["store", "context", &payload.to_string()])
            .is_some()
        {
            let _ = std::fs::write(&sentinel, "");
            eprintln!("🧠 Axil captured fallback: '{missed_query}' → {rel}:{line_start}-{line_end}");
        }
        Ok(())
    }

    /// Track the edit manifest and accumulate content snippets for the
    /// end-of-session entity extraction.
    fn post_edit_log(&self, file_path: &str, snippet: Option<&str>) -> Result<()> {
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
        if exit_code == 0 && !cmd.is_empty() {
            // Heartbeat: the agent just interacted with its own brain.
            if contains_any(cmd, &["axil store ", "axil observe ", "axil believe "]) {
                self.bump_count("stores");
                eprintln!("🧠 Axil stored (session: {})", self.counts_compact());
            } else if contains_any(cmd, &["axil recall", "axil boot", "axil recall-for-"]) {
                self.bump_count("recalls");
            }

            // A commit message IS a decision/summary the agent already wrote —
            // capture it as narrative so the Stop guard doesn't demand a
            // re-statement of what's in the commit.
            if cmd.contains("git commit") {
                self.capture_git_commit();
            }
        }

        if exit_code != 0 {
            self.bump_count("errors");
            if !stdout.is_empty() {
                // High confidence threshold to avoid noise.
                let _ = self.axil_db_out_stdin(
                    &["auto-capture", "-", "--min-confidence", "0.8", "--source", "bash"],
                    truncate_utf8(stdout, 2000).as_bytes(),
                );
            }
            // Generic build/test failures already flow through auto-capture;
            // only axil-specific failures feed session-heal.
            if cmd.contains("axil ") {
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
        } else if contains_any(
            cmd,
            &[
                "axil recall ",
                "axil code-search ",
                "axil fts ",
                "axil recall-for-file ",
                "axil recall-for-entity ",
            ],
        ) {
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

        let payload = json!({
            "sha": sha,
            "subject": subject,
            "body": body.trim_end(),
            "author": author,
            "committed_at": committed_at,
            "files": files,
        });
        if self
            .axil_db_out(&["store", "commits", &payload.to_string()])
            .is_some()
        {
            self.bump_count("stores");
            let sha7: String = sha.chars().take(7).collect();
            eprintln!("🧠 Axil captured commit {sha7}: {subject}");
        }
    }

    /// When a todo flips to completed, inject the store reminder BEFORE the
    /// agent moves on. (The old bash hook matched a `TaskUpdate` tool that
    /// stock Claude Code never emits — the real todo tool is `TodoWrite`
    /// with a `todos[]` payload, so it never fired.)
    fn post_todo_store_reminder(&self, completed: i64) -> Result<()> {
        let sentinel = self.sfile("todos");
        let last: i64 = std::fs::read_to_string(&sentinel)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let _ = std::fs::write(&sentinel, completed.to_string());

        if completed > last {
            self.emit_context(
                "You just marked a task completed. BEFORE doing anything else, run axil store with a summary of what you did and why. This is mandatory for every completed task.",
            );
        }
        Ok(())
    }

    // ── Stop: narrative guard, then session close ─────────────────────

    fn on_stop(&self, can_block: bool) -> Result<()> {
        let manifest_path = self.sfile("manifest");
        if !manifest_path.exists() {
            // Read-only session: still replay accumulated misses/failures so
            // recall-only sessions drive auto-fix and _heal_log entries.
            if self.sfile("problems").exists() && self.db.is_some() {
                self.run_session_heal_with_autofix();
            }
            self.cleanup_session_files();
            return Ok(());
        }
        if self.db.is_none() {
            self.cleanup_session_files();
            return Ok(());
        }

        let manifest = std::fs::read_to_string(&manifest_path).unwrap_or_default();
        let files: Vec<&str> = manifest
            .lines()
            .filter(|l| !l.is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let file_count = files.len();
        let files_json = serde_json::to_string(&files).unwrap_or_else(|_| "[]".into());

        // Guard: block the stop when substantive work (>2 distinct files)
        // happened with no narrative row and no fresh commit. The JSON
        // {"decision":"block"} on stdout is the only channel the harness
        // re-injects into the model — stderr is invisible to it.
        if can_block
            && !self.event.stop_hook_active
            && file_count > 2
            && self.count_recent_narrative() == 0
            && !self.has_recent_git_commit()
        {
            let reason = format!(
                "Axil brain: {file_count} files were edited this turn but no {NARRATIVE_TABLES_TEXT} row was stored in the last hour (and no git commit). Before stopping, either: (a) commit the work — the commit message is captured as narrative — or (b) run: axil checkpoint '{{\"state\":\"<where things stand>\",\"next_steps\":[\"<remaining work>\"],\"references\":[{{\"kind\":\"file\",\"ref\":\"<path>\"}}]}}' (files touched this turn: {files_json}). After storing, you may stop."
            );
            self.emit_stop_block(&reason);
            // Return WITHOUT closing or cleaning up: the session is still
            // live. Running the close here would write a premature _sessions
            // record + worker/beliefs, and deleting the temp files (booted,
            // counts, manifest) would make the next tool call spuriously
            // re-boot mid-session. When the agent stores and stops again,
            // that Stop passes the guard (narrative present or
            // stop_hook_active) and does the real close + cleanup. If a
            // stale async config ignores the block, the small per-session
            // temp-file set is orphaned until the OS clears the temp dir —
            // an acceptable trade for not corrupting a live session.
            return Ok(());
        }

        // Entity extraction from accumulated edit snippets.
        let entities: Value = std::fs::read_to_string(self.sfile("content"))
            .ok()
            .filter(|c| !c.is_empty())
            .and_then(|c| {
                self.axil_out_stdin(&["extract-entities", "-"], truncate_utf8(&c, 4000).as_bytes())
            })
            .and_then(|out| serde_json::from_str(out.trim()).ok())
            .unwrap_or_else(|| json!([]));
        let entity_count = entities.as_array().map(|a| a.len()).unwrap_or(0);

        let session_record = json!({
            "session": self.sid,
            "files_changed": files,
            "file_count": file_count,
            "entities": entities,
            "entity_count": entity_count,
            "ended_at": now_iso(),
        });
        let store_result = self
            .axil_db_out(&["store", "_sessions", &session_record.to_string()])
            .unwrap_or_default();
        if let Some(id) = serde_json::from_str::<Value>(store_result.trim())
            .ok()
            .and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_string))
        {
            let _ = self.axil_db_out(&["auto-link", &id]);
        }

        // Consolidation, connections, inference, decay — then beliefs.
        let _ = self.axil_db_out(&["worker", "run"]);
        let _ = self.axil_db_out(&["beliefs", "--generate"]);

        // session-heal always inspects detect_problems() even without an
        // explicit problems file, so every session gets a heal pass.
        self.run_session_heal_with_autofix();

        // Save the manifest for the next session's context-aware boot
        // (cleanup below would remove it).
        let _ = std::fs::copy(&manifest_path, self.prev_manifest());

        let (stores, recalls) = (self.read_count("stores"), self.read_count("recalls"));
        if stores != 0 || recalls != 0 {
            eprintln!(
                "🧠 Axil session: {stores} stored ∙ {recalls} recalled ∙ {} tools ∙ {} errors",
                self.read_count("tools"),
                self.read_count("errors")
            );
        }

        self.cleanup_session_files();
        Ok(())
    }

    /// Run session-heal and act on its hints. The one user-visible auto-fix
    /// today: `stale_structural_index` spawns a detached `axil index` so the
    /// next session's queries hit a fresh index. The lock file throttles
    /// back-to-back stops within the 5-minute stale window.
    fn run_session_heal_with_autofix(&self) {
        let problems = self.sfile("problems");
        let mut args: Vec<&str> = vec!["session-heal", "--session", &self.sid];
        let problems_str;
        if problems.exists() {
            problems_str = problems.to_string_lossy().into_owned();
            args.push("--problems-file");
            args.push(&problems_str);
        }
        let Some(out) = self.axil_db_out(&args) else {
            return;
        };
        let Ok(report) = serde_json::from_str::<Value>(out.trim()) else {
            return;
        };
        let stale = report
            .get("hints")
            .and_then(Value::as_array)
            .map(|hints| {
                hints.iter().any(|h| {
                    h.get("kind").and_then(Value::as_str) == Some("stale_structural_index")
                })
            })
            .unwrap_or(false);
        if !stale {
            return;
        }

        let axil_dir = self.project_dir.join(".axil");
        let lock = axil_dir.join("index-refresh.lock");
        let log = axil_dir.join("index-refresh.log");
        if let Ok(meta) = std::fs::metadata(&lock) {
            let age = meta
                .modified()
                .ok()
                .and_then(|m| m.elapsed().ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if age < 300 {
                return;
            }
        }
        let _ = std::fs::create_dir_all(&axil_dir);
        let _ = std::fs::write(&lock, chrono::Utc::now().timestamp().to_string());

        let Some(db) = self.db.as_ref() else { return };
        let args: Vec<String> = vec![
            "--db".into(),
            db.to_string_lossy().into_owned(),
            "index".into(),
            self.project_dir.to_string_lossy().into_owned(),
        ];
        if spawn_detached(&self.exe, &args, &self.project_dir, &log) {
            eprintln!(
                "🧠 Axil session-heal: stale structural index → spawned 'axil index' in background (log: .axil/index-refresh.log)"
            );
        } else {
            let _ = std::fs::remove_file(&lock);
        }
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

/// Detect broad repo-search commands and best-effort extract their query.
/// Returns (is_repo_search, query). `ls`/`tree` count as repo search but
/// never carry a query; for the others the query is the first quoted arg,
/// falling back to the first non-flag token after the search word.
fn detect_repo_search(cmd: &str) -> (bool, String) {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let mut search_tool = None;
    for (i, tok) in tokens.iter().enumerate() {
        match *tok {
            "git" if tokens.get(i + 1) == Some(&"grep") => {
                search_tool = Some((i + 1, "grep"));
                break;
            }
            "rg" | "grep" | "fd" | "find" => {
                search_tool = Some((i, *tok));
                break;
            }
            // Directory listings count as repo discovery but carry no query.
            "ls" | "tree" => return (true, String::new()),
            _ => {}
        }
    }
    let Some((tool_idx, _)) = search_tool else {
        return (false, String::new());
    };

    // Quoted args win — they're unambiguous.
    if let Some(q) = first_quoted_arg(cmd) {
        return (true, q);
    }
    // Otherwise: first non-flag token after the tool, minus shell operators.
    for tok in tokens.iter().skip(tool_idx + 1) {
        if tok.starts_with('-') || *tok == "." || *tok == "./" {
            continue;
        }
        let clean = tok
            .split(|c| matches!(c, ';' | '&' | '|'))
            .next()
            .unwrap_or("");
        return (true, clean.to_string());
    }
    (true, String::new())
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

/// Best-effort axil-subcommand extractor: the token after `axil` (or a
/// path ending in axil) that isn't a flag.
fn extract_axil_subcmd(cmd: &str) -> String {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    for (i, tok) in tokens.iter().enumerate() {
        let base = tok.rsplit(['/', '\\']).next().unwrap_or(tok);
        if base == "axil" || base == "axil.exe" {
            for next in tokens.iter().skip(i + 1) {
                if !next.starts_with('-') {
                    return next.to_string();
                }
            }
        }
    }
    String::new()
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

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

fn append_line(path: &Path, line: &str) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
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
        Command::new("sh")
            .arg("-c")
            .arg(parts.join(" "))
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .and_then(|mut c| c.wait())
            .is_ok()
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
        let (is, q) = detect_repo_search(r#"rg "hnsw recall" src/"#);
        assert!(is);
        assert_eq!(q, "hnsw recall");

        let (is, q) = detect_repo_search("grep -r 'adaptive_ef' crates/");
        assert!(is);
        assert_eq!(q, "adaptive_ef");

        let (is, q) = detect_repo_search("git grep install_agent_integrations");
        assert!(is);
        assert_eq!(q, "install_agent_integrations");

        let (is, q) = detect_repo_search("find . -name *.rs");
        assert!(is);
        assert_eq!(q, "*.rs");

        let (is, q) = detect_repo_search("ls -la src/");
        assert!(is);
        assert!(q.is_empty());

        let (is, _) = detect_repo_search("cargo test --workspace");
        assert!(!is);

        // Token-aware: substrings of other words must not trigger.
        let (is, _) = detect_repo_search("cargo run --example energy");
        assert!(!is);
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
        assert_eq!(extract_axil_subcmd("axil --db x.axil store errors '{}'"), "x.axil");
        assert_eq!(extract_axil_subcmd("cargo build"), "");
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
