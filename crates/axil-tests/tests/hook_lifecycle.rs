//! The Claude Code hook lifecycle, end to end through the `axil` binary:
//! Stop (every turn) only guards and never writes; the session close is a
//! queued job that a detached drainer runs; counters survive parallel hooks;
//! failed commands and completed tasks are seen.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use axil_core::Axil;
use serde_json::{json, Value};
use tempfile::TempDir;

fn axil_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("target/debug/axil");
    path.set_extension(std::env::consts::EXE_EXTENSION);
    assert!(
        path.exists(),
        "axil binary not found at {}. Run `cargo build -p axildb` first.",
        path.display()
    );
    path
}

/// A project with an Axil DB, and a private temp dir for the hook's
/// per-session state files.
struct Project {
    dir: TempDir,
    tmp: TempDir,
    db: PathBuf,
}

impl Project {
    /// `session_end`: whether `.claude/settings.json` routes SessionEnd to the
    /// brain, as `axil install` now does.
    fn new(session_end: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let db = dir.path().join(".axil").join("memory.axil");
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        Axil::open(&db).build().unwrap();
        if session_end {
            let settings = json!({"hooks": {"SessionEnd": [{"hooks": [
                {"type": "command", "command": "axil hook run --dialect claude"}
            ]}]}});
            std::fs::create_dir_all(dir.path().join(".claude")).unwrap();
            std::fs::write(
                dir.path().join(".claude/settings.json"),
                settings.to_string(),
            )
            .unwrap();
        }
        Self { dir, tmp, db }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Run one hook event; returns stdout.
    fn hook(&self, payload: Value) -> String {
        let mut child = Command::new(axil_bin())
            .args(["hook", "run", "--dialect", "claude"])
            .env("CLAUDE_PROJECT_DIR", self.path())
            .env("TMPDIR", self.tmp.path())
            .current_dir(self.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.to_string().as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn state(&self, sid: &str, suffix: &str) -> PathBuf {
        self.tmp.path().join(format!("axil-session-{sid}.{suffix}"))
    }

    /// Mark a session booted, so its first tool call skips the boot and the
    /// background refreshes the boot starts.
    fn booted(&self, sid: &str) {
        std::fs::write(self.state(sid, "booted"), "").unwrap();
    }

    fn edit(&self, sid: &str, file: &str) {
        self.hook(json!({
            "hook_event_name": "PostToolUse",
            "session_id": sid,
            "tool_name": "Edit",
            "tool_input": {
                "file_path": self.path().join(file).to_string_lossy(),
                "new_string": "fn handler() {}",
            },
            "tool_response": {},
        }));
    }

    fn stop(&self, sid: &str, stop_hook_active: bool) -> String {
        self.hook(json!({
            "hook_event_name": "Stop",
            "session_id": sid,
            "stop_hook_active": stop_hook_active,
        }))
    }

    fn queue(&self) -> PathBuf {
        self.db.parent().unwrap().join("hook-queue")
    }

    fn queued_jobs(&self) -> usize {
        std::fs::read_dir(self.queue())
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().extension().is_some_and(|x| x == "job"))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Wait until the drainer has emptied the queue and let go of its lock.
    fn wait_drained(&self) {
        wait_for("the hook queue to drain", || {
            let busy = std::fs::read_dir(self.queue()).is_ok_and(|d| {
                d.flatten().any(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    name.ends_with(".job") || name.ends_with(".running") || name == "drain.lock"
                })
            });
            !busy
        });
    }

    fn rows(&self, table: &str) -> Vec<Value> {
        Axil::open(&self.db)
            .build()
            .unwrap()
            .list(table)
            .unwrap()
            .into_iter()
            .map(|r| r.data)
            .collect()
    }

    fn count(&self, sid: &str, key: &str) -> usize {
        std::fs::read_to_string(self.state(sid, "events"))
            .unwrap_or_default()
            .lines()
            .filter(|l| *l == key)
            .count()
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(90);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn stop_only_guards_and_session_end_queues_the_close() {
    let p = Project::new(true);
    let sid = "s-guard";
    p.booted(sid);
    for file in ["src/a.rs", "src/b.rs", "src/c.rs"] {
        p.edit(sid, file);
    }

    // Three files edited and nothing stored: the stop is blocked.
    let out = p.stop(sid, false);
    assert!(out.contains(r#""decision":"block""#), "{out}");

    // The retried stop passes, and writes nothing: no job, session kept.
    assert!(p.stop(sid, true).trim().is_empty());
    assert_eq!(p.queued_jobs(), 0, "Stop must not queue a write");
    assert!(
        p.state(sid, "manifest").exists(),
        "state must outlive the turn"
    );

    // The next turn's stop only looks at that turn's edits.
    p.edit(sid, "src/d.rs");
    assert!(p.stop(sid, false).trim().is_empty());

    p.hook(json!({"hook_event_name": "SessionEnd", "session_id": sid, "reason": "other"}));
    assert!(!p.state(sid, "manifest").exists(), "session end cleans up");
    p.wait_drained();

    let sessions = p.rows("_sessions");
    assert_eq!(sessions.len(), 1, "{sessions:?}");
    assert_eq!(sessions[0]["session"], sid);
    assert_eq!(sessions[0]["file_count"], 4);
}

#[test]
fn without_a_session_end_event_stop_queues_the_close() {
    let p = Project::new(false);
    let sid = "s-legacy";
    p.booted(sid);
    p.edit(sid, "src/a.rs");

    assert!(p.stop(sid, false).trim().is_empty());
    p.wait_drained();
    assert_eq!(p.rows("_sessions").len(), 1);

    // Nothing new since that flush: the next stop records nothing more.
    assert!(p.stop(sid, false).trim().is_empty());
    p.wait_drained();
    assert_eq!(p.rows("_sessions").len(), 1);
    assert!(
        p.state(sid, "manifest").exists(),
        "state stays for the next turn"
    );
}

#[test]
fn parallel_hooks_count_every_tool_call() {
    let p = Project::new(true);
    let sid = "s-parallel";
    p.booted(sid);
    std::thread::scope(|scope| {
        for _ in 0..24 {
            scope.spawn(|| {
                p.hook(json!({
                    "hook_event_name": "PreToolUse",
                    "session_id": sid,
                    "tool_name": "Bash",
                    "tool_input": {"command": "true"},
                }));
            });
        }
    });
    assert_eq!(p.count(sid, "tools"), 24);
}

#[test]
fn a_completed_task_gets_one_store_reminder() {
    let p = Project::new(true);
    let sid = "s-task";
    let task = |id: &str, status: &str| {
        p.hook(json!({
            "hook_event_name": "PostToolUse",
            "session_id": sid,
            "tool_name": "TaskUpdate",
            "tool_input": {"taskId": id, "status": status},
            "tool_response": {},
        }))
    };
    // Nothing is stored in this project, so each newly completed task gets the
    // reminder once.
    assert!(task("1", "in_progress").trim().is_empty());
    assert!(task("1", "completed").contains("axil store"));
    assert!(task("1", "completed").trim().is_empty(), "once per task");
    assert!(task("2", "completed").contains("axil store"));
}

#[test]
fn a_failed_shell_command_is_seen() {
    let p = Project::new(true);
    let sid = "s-fail";
    p.hook(json!({
        "hook_event_name": "PostToolUseFailure",
        "session_id": sid,
        "tool_name": "Bash",
        "tool_input": {"command": "cargo test"},
        "tool_error": "Exit code 101\nerror: test failed, to rerun pass `-p axil-core --lib`",
    }));
    assert_eq!(p.count(sid, "errors"), 1);
    p.wait_drained();
}

#[test]
fn session_start_injects_boot_and_closes_stale_sessions() {
    let p = Project::new(true);

    // A session of this project that ended without a SessionEnd, 7h ago.
    let old = "s-stale";
    std::fs::write(
        p.state(old, "project"),
        p.path().to_string_lossy().as_bytes(),
    )
    .unwrap();
    std::fs::write(p.state(old, "manifest"), "src/old.rs\n").unwrap();
    let past = SystemTime::now() - Duration::from_secs(7 * 3600);
    for suffix in ["project", "manifest"] {
        std::fs::File::options()
            .write(true)
            .open(p.state(old, suffix))
            .unwrap()
            .set_modified(past)
            .unwrap();
    }

    let out = p.hook(json!({
        "hook_event_name": "SessionStart",
        "session_id": "s-new",
        "source": "startup",
    }));
    let boot: Value = serde_json::from_str(out.trim()).expect(&out);
    assert_eq!(boot["hookSpecificOutput"]["hookEventName"], "SessionStart");
    assert!(boot["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .is_some_and(|c| c.contains("Axil Boot Context")));

    assert!(!p.state(old, "manifest").exists());
    p.wait_drained();
    let sessions = p.rows("_sessions");
    assert!(
        sessions.iter().any(|s| s["session"] == old),
        "the stale session is closed: {sessions:?}"
    );
}

#[test]
fn drain_never_retries_an_interrupted_job() {
    let p = Project::new(true);
    let queue = p.queue();
    std::fs::create_dir_all(&queue).unwrap();
    let job = |summary: &str| {
        json!({"op": "axil", "args": ["store", "decisions", json!({"summary": summary}).to_string()]})
            .to_string()
    };
    std::fs::write(queue.join("20260101T000000Z-1-0.running"), job("cut off")).unwrap();
    std::fs::write(queue.join("20260101T000001Z-1-1.job"), job("queued")).unwrap();

    let status = Command::new(axil_bin())
        .arg("--db")
        .arg(&p.db)
        .args(["hook", "drain"])
        .status()
        .unwrap();
    assert!(status.success());

    let decisions = p.rows("decisions");
    assert_eq!(decisions.len(), 1, "{decisions:?}");
    assert_eq!(decisions[0]["summary"], "queued");
    assert!(queue
        .join("interrupted/20260101T000000Z-1-0.running")
        .exists());
    let log = std::fs::read_to_string(queue.join("drain.log")).unwrap();
    assert!(log.contains("not retried"), "{log}");
}
