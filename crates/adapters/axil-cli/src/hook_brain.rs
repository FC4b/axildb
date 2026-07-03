//! Axil Brain — agent lifecycle hook runtime.
//!
//! Rust port of the former `.claude/hooks/axil-brain.sh` and
//! `store-on-task-complete.sh`. Living inside the binary removes the
//! bash + jq dependency, runs natively on Windows, and gives the hook
//! logic real unit tests. The CLI entry is `axil hook run --dialect <d>`;
//! the agent harness pipes the event JSON to stdin and reads the
//! dialect's response JSON (or injected context text) from stdout.
//!
//! A *dialect* is the JSON contract one agent family speaks. `claude`
//! covers Claude Code today; Codex / Copilot CLI / Factory Droid use the
//! same shell-hook shape and land as thin field mappings, while the
//! Gemini-lineage tools (Antigravity CLI, Qwen Code) get their own
//! mapping. The cognitive logic below is shared by all of them.
//!
//! Events handled (claude dialect):
//!   UserPromptSubmit         — inject a <context> block from recall
//!   PreToolUse (first call)  — boot banner + context-aware boot push
//!   PreToolUse (Edit/Write)  — recall-for-file + store nudge
//!   PreToolUse (Bash)        — axil-first search gate / paired search
//!   PostToolUse (Edit/Write) — manifest + snippet accumulation
//!   PostToolUse (Bash)       — heartbeat, commit capture, error capture
//!   PostToolUse (Read)       — fallback capture after empty recalls
//!   PostToolUse (TodoWrite)  — store reminder when a todo completes
//!   Stop                     — narrative guard, session close, worker
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

pub(crate) fn run(dialect: &str, event_override: Option<&str>) -> Result<i32> {
    // Unknown dialect is a wiring mistake by whoever edited settings —
    // fail loudly so it's caught in development, not silently in sessions.
    if dialect != "claude" {
        anyhow::bail!(
            "unknown hook dialect '{dialect}' (supported: claude). \
             Codex/Copilot/Droid/Gemini dialects arrive with their integration waves."
        );
    }

    let mut raw = String::new();
    if std::io::stdin().read_to_string(&mut raw).is_err() || raw.trim().is_empty() {
        return Ok(0);
    }
    let input: Value = match serde_json::from_str(raw.trim()) {
        Ok(v) => v,
        Err(_) => return Ok(0),
    };

    let Some(ctx) = HookCtx::from_claude(&input, event_override) else {
        return Ok(0);
    };
    // Never propagate internal errors to the agent loop: report and exit 0.
    if let Err(e) = ctx.dispatch() {
        eprintln!("[axil hook] warn: {e}");
    }
    Ok(0)
}

struct HookCtx {
    input: Value,
    event: String,
    tool_name: String,
    sid: String,
    project_dir: PathBuf,
    exe: PathBuf,
    db: Option<PathBuf>,
    tmp: PathBuf,
}

impl HookCtx {
    fn from_claude(input: &Value, event_override: Option<&str>) -> Option<Self> {
        let event = event_override
            .map(str::to_string)
            .or_else(|| str_field(input, "hook_event_name"))?;
        let sid = sanitize_id(&str_field(input, "session_id")?);
        if sid.is_empty() {
            return None;
        }
        let project_dir = std::env::var_os("CLAUDE_PROJECT_DIR")
            .map(PathBuf::from)
            .or_else(|| str_field(input, "cwd").map(PathBuf::from))
            .or_else(|| std::env::current_dir().ok())?;
        let exe = std::env::current_exe().ok()?;
        let db = find_db(&project_dir);
        Some(Self {
            tool_name: str_field(input, "tool_name").unwrap_or_default(),
            input: input.clone(),
            event,
            sid,
            project_dir,
            exe,
            db,
            tmp: std::env::temp_dir(),
        })
    }

    fn dispatch(&self) -> Result<()> {
        match self.event.as_str() {
            "UserPromptSubmit" => self.on_user_prompt(),
            "PreToolUse" => self.on_pre_tool(),
            "PostToolUse" => self.on_post_tool(),
            "Stop" => self.on_stop(),
            _ => Ok(()),
        }
    }

    // ── Session temp files ────────────────────────────────────────────

    fn sfile(&self, suffix: &str) -> PathBuf {
        self.tmp.join(format!("axil-session-{}.{}", self.sid, suffix))
    }

    fn prev_manifest(&self) -> PathBuf {
        self.tmp.join("axil-session-prev.manifest")
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

    // ── UserPromptSubmit: inject <context> block from recall ─────────
    // Runs on every user prompt; must stay under ~2s wall-clock, so the
    // recall carries its own deadline.
    fn on_user_prompt(&self) -> Result<()> {
        let Some(prompt) = str_field(&self.input, "prompt") else {
            return Ok(());
        };
        // Trivially short prompts (acks, confirmations) — no useful recall.
        if prompt.chars().count() < 8 {
            return Ok(());
        }
        if let Some(ctx) = self.axil_db_out(&[
            "recall",
            &prompt,
            "--recall-format",
            "context-block",
            "--budget",
            "2000",
            "--timeout-ms",
            "1800",
            "--top-k",
            "5",
        ]) {
            // Claude Code injects UserPromptSubmit stdout back into the prompt.
            if !ctx.trim().is_empty() {
                print!("{ctx}");
            }
        }
        Ok(())
    }

    // ── PreToolUse ────────────────────────────────────────────────────

    fn on_pre_tool(&self) -> Result<()> {
        self.bump_count("tools");

        // First tool call of the session → context-aware boot.
        let booted = self.sfile("booted");
        if !booted.exists() {
            let _ = std::fs::write(&booted, "");
            self.boot_push();
            // Fall through: if the session's first tool is Edit/Write the
            // file-recall context must still be injected below.
        }

        match self.tool_name.as_str() {
            "Edit" | "Write" => self.pre_edit_context(),
            "Bash" => self.pre_bash_search_gate(),
            _ => Ok(()),
        }
    }

    fn boot_push(&self) {
        if self.db.is_none() {
            return;
        }
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

        // Banner + boot text both go to stderr: PreToolUse stdout is parsed
        // as hook JSON, and mixing prose into it corrupts the parse (the old
        // bash hook printed the banner to stdout — a latent bug).
        if let Some(banner) = self.axil_db_out(&["brain-banner"]) {
            if !banner.trim().is_empty() {
                eprint!("{banner}");
            }
        }
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        if let Some(boot) = self.axil_db_out(&arg_refs) {
            if !boot.trim().is_empty() {
                eprintln!("{boot}");
            }
        }

        // Opportunistic background refreshes. Both subcommands self-detach
        // (`--in-background`) and gate on staleness, so these return fast;
        // errors are silent — the brain hook must never block the agent.
        let _ = self.axil_db_out(&["scip", "refresh", "--if-stale", "--in-background", "--quiet"]);
        let _ = self.axil_db_out(&["maintain", "--if-stale", "--in-background", "--quiet"]);
    }

    /// Surface past memories about a file BEFORE the agent edits it, plus
    /// the 5-edit "have you stored anything?" nudge — combined into one
    /// hookSpecificOutput so they don't fight for stdout.
    fn pre_edit_context(&self) -> Result<()> {
        let Some(file_path) = nested_str(&self.input, &["tool_input", "file_path"]) else {
            return Ok(());
        };
        if is_skipped_path(&file_path) || self.db.is_none() {
            return Ok(());
        }
        let rel = self.rel_path(&file_path);

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
            emit_additional_context("PreToolUse", &ctx);
        }
        Ok(())
    }

    /// Pair broad repo search with Axil's own index: run code-search/fts
    /// first and inject the compact result; when no query is extractable
    /// and the session hasn't recalled yet, inject the gate reminder.
    fn pre_bash_search_gate(&self) -> Result<()> {
        let Some(cmd) = nested_str(&self.input, &["tool_input", "command"]) else {
            return Ok(());
        };
        let (is_repo_search, mut query) = detect_repo_search(&cmd);
        if !is_repo_search {
            return Ok(());
        }
        if query.chars().count() < 3 {
            query = String::new();
        }

        if query.is_empty() {
            if self.read_count("recalls") == 0 {
                let ctx = "⚠️ AXIL search gate — this session has not used Axil recall yet. Before broad repo discovery, run one of:\n  axil recall \"<what you need>\" --top-k 5\n  axil code-search \"<symbol/module/API>\" --top-k 5\n  axil fts \"<exact term>\" --limit 5\n\nThen open the files Axil returns and verify current code.";
                emit_additional_context("PreToolUse", ctx);
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
            emit_additional_context("PreToolUse", &ctx);
        }
        Ok(())
    }

    // ── PostToolUse ───────────────────────────────────────────────────

    fn on_post_tool(&self) -> Result<()> {
        match self.tool_name.as_str() {
            "Read" => self.post_read_fallback_capture(),
            "Edit" | "Write" => self.post_edit_log(),
            "Bash" => self.post_bash(),
            "TodoWrite" => self.post_todo_store_reminder(),
            _ => Ok(()),
        }
    }

    /// After a Read that follows a recent empty recall/code-search/fts,
    /// attach a low-importance context row tying the missed query to the
    /// exact line range the agent opened — closing the miss→fallback loop.
    /// Rows carry `_origin: fallback_capture` and `_importance: 0.2` so they
    /// sit below the default recall floor.
    fn post_read_fallback_capture(&self) -> Result<()> {
        let problems_path = self.sfile("problems");
        if !problems_path.exists() {
            return Ok(());
        }
        let Some(file_path) = nested_str(&self.input, &["tool_input", "file_path"]) else {
            return Ok(());
        };
        if is_skipped_path(&file_path) {
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

        let offset = nested_i64(&self.input, &["tool_input", "offset"]).unwrap_or(1);
        let limit = nested_i64(&self.input, &["tool_input", "limit"]).unwrap_or(2000);
        let (line_start, line_end) = (offset, offset + limit - 1);
        let rel = self.rel_path(&file_path);

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
    fn post_edit_log(&self) -> Result<()> {
        let Some(file_path) = nested_str(&self.input, &["tool_input", "file_path"]) else {
            return Ok(());
        };
        if is_skipped_path(&file_path) {
            return Ok(());
        }
        append_line(&self.sfile("manifest"), &self.rel_path(&file_path));

        let snippet_field = if self.tool_name == "Edit" {
            "new_string"
        } else {
            "content"
        };
        if let Some(text) = nested_str(&self.input, &["tool_input", snippet_field]) {
            if !text.is_empty() {
                append_line(&self.sfile("content"), truncate_utf8(&text, 500));
            }
        }
        Ok(())
    }

    fn post_bash(&self) -> Result<()> {
        let exit_code = response_exit_code(&self.input);
        let cmd = nested_str(&self.input, &["tool_input", "command"]).unwrap_or_default();

        if exit_code == 0 && !cmd.is_empty() {
            // Heartbeat: the agent just interacted with its own brain.
            if contains_any(&cmd, &["axil store ", "axil observe ", "axil believe "]) {
                self.bump_count("stores");
                eprintln!("🧠 Axil stored (session: {})", self.counts_compact());
            } else if contains_any(&cmd, &["axil recall", "axil boot", "axil recall-for-"]) {
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
            let output = response_str(&self.input, &["stdout", "output"]);
            if !output.is_empty() {
                // High confidence threshold to avoid noise.
                let _ = self.axil_db_out_stdin(
                    &["auto-capture", "-", "--min-confidence", "0.8", "--source", "bash"],
                    truncate_utf8(&output, 2000).as_bytes(),
                );
            }
            // Generic build/test failures already flow through auto-capture;
            // only axil-specific failures feed session-heal.
            if cmd.contains("axil ") {
                let stderr = response_str(&self.input, &["stderr"]);
                let event = json!({
                    "kind": "command_failure",
                    "subcommand": extract_axil_subcmd(&cmd),
                    "query": cmd,
                    "exit_code": exit_code,
                    "stderr": truncate_utf8(&stderr, 500),
                    "at": now_iso(),
                });
                self.log_problem(&event.to_string());
            }
        } else if contains_any(
            &cmd,
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
            let stdout = response_str(&self.input, &["stdout", "output"]);
            if is_empty_axil_output(&stdout) {
                let event = json!({
                    "kind": "empty_result",
                    "subcommand": extract_axil_subcmd(&cmd),
                    "query": first_quoted_arg(&cmd).unwrap_or_default(),
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
    fn post_todo_store_reminder(&self) -> Result<()> {
        let completed = self
            .input
            .get("tool_input")
            .and_then(|t| t.get("todos"))
            .and_then(Value::as_array)
            .map(|todos| {
                todos
                    .iter()
                    .filter(|t| t.get("status").and_then(Value::as_str) == Some("completed"))
                    .count() as i64
            })
            .unwrap_or(0);

        let sentinel = self.sfile("todos");
        let last: i64 = std::fs::read_to_string(&sentinel)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let _ = std::fs::write(&sentinel, completed.to_string());

        if completed > last {
            emit_additional_context(
                "PostToolUse",
                "You just marked a task completed. BEFORE doing anything else, run axil store with a summary of what you did and why. This is mandatory for every completed task.",
            );
        }
        Ok(())
    }

    // ── Stop: narrative guard, then session close ─────────────────────

    fn on_stop(&self) -> Result<()> {
        let stop_hook_active = self
            .input
            .get("stop_hook_active")
            .and_then(Value::as_bool)
            .unwrap_or(false);

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
        if !stop_hook_active
            && file_count > 2
            && self.count_recent_narrative() == 0
            && !self.has_recent_git_commit()
        {
            let reason = format!(
                "Axil brain: {file_count} files were edited this turn but no {NARRATIVE_TABLES_TEXT} row was stored in the last hour (and no git commit). Before stopping, either: (a) commit the work — the commit message is captured as narrative — or (b) run: axil checkpoint '{{\"state\":\"<where things stand>\",\"next_steps\":[\"<remaining work>\"],\"references\":[{{\"kind\":\"file\",\"ref\":\"<path>\"}}]}}' (files touched this turn: {files_json}). After storing, you may stop."
            );
            println!("{}", json!({"decision": "block", "reason": reason}));
            // Fall through to session close + cleanup: if the harness honors
            // the block a second Stop fires later and closes again; if an
            // old async config ignores it, falling through prevents leaks.
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
    let Some(resp) = input.get("tool_response") else {
        return 0;
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
    0
}

/// First non-empty string among `tool_response.<keys>`.
fn response_str(input: &Value, keys: &[&str]) -> String {
    let Some(resp) = input.get("tool_response") else {
        return String::new();
    };
    for key in keys {
        if let Some(s) = resp.get(*key).and_then(Value::as_str) {
            if !s.is_empty() {
                return s.to_string();
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

/// The additionalContext channel is the only PreToolUse/PostToolUse output
/// the model actually sees.
fn emit_additional_context(event_name: &str, ctx: &str) {
    println!(
        "{}",
        json!({
            "hookSpecificOutput": {
                "hookEventName": event_name,
                "additionalContext": ctx,
            }
        })
    );
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
        assert_eq!(q, "*.rs"); // -name skipped as flag? no: `.` skipped, -name skipped

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
    fn exit_code_tolerates_string_and_number() {
        assert_eq!(
            response_exit_code(&json!({"tool_response": {"exitCode": 1}})),
            1
        );
        assert_eq!(
            response_exit_code(&json!({"tool_response": {"exit_code": "2"}})),
            2
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
}
