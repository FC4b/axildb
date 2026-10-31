//! Converting a legacy table index, through the `axil` binary: a hook's
//! lookup never does it under the hook's time limit, doctor won't do it to a
//! database whose agent hooks run an axil too old to open it afterwards, and
//! a command that does it says so.
#![cfg(unix)]

use std::io::Write as _;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axil_core::{Storage, TableIndexState};
use serde_json::{json, Value};

/// `open`, again while it fails busy: another test's thread spawning a
/// child gives the child a copy of a database handle this thread just
/// closed, which holds the file's lock until the child execs.
fn retry_busy<T>(mut open: impl FnMut() -> axil_core::Result<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match open() {
            Err(e) if e.is_busy() && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            other => return other.unwrap(),
        }
    }
}

/// A project with a database, and a home of its own, so the user's own
/// agent configs stay out of it.
struct Project {
    dir: tempfile::TempDir,
    home: tempfile::TempDir,
    tmp: tempfile::TempDir,
}

impl Project {
    fn new() -> Self {
        let project = Self {
            dir: tempfile::tempdir().unwrap(),
            home: tempfile::tempdir().unwrap(),
            tmp: tempfile::tempdir().unwrap(),
        };
        std::fs::create_dir_all(project.db().parent().unwrap()).unwrap();
        project
    }

    fn db(&self) -> PathBuf {
        self.dir.path().join(".axil/memory.axil")
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_axil"));
        cmd.current_dir(self.dir.path())
            .env("HOME", self.home.path())
            .env("TMPDIR", self.tmp.path())
            .env("AXIL_DAEMON", "off")
            .env_remove("AXIL_DB")
            .env_remove("CLAUDE_PROJECT_DIR")
            .env_remove("AXIL_HOOK_DEADLINE_MS");
        cmd
    }

    fn axil(&self, args: &[&str]) -> Output {
        let db = self.db();
        self.command()
            .arg("--db")
            .arg(&db)
            .args(args)
            .output()
            .unwrap()
    }

    /// The database, one record in it, with its table index in the legacy
    /// layout a build from before the keyed index writes.
    fn legacy(&self) {
        let db = axil_core::Axil::open(self.db()).build().unwrap();
        db.insert(
            "notes",
            json!({ "summary": "auth timeouts on long uploads" }),
        )
        .unwrap();
        drop(db);
        retry_busy(|| Storage::open(self.db()))
            .downgrade_table_index()
            .unwrap();
        assert_eq!(self.state(), TableIndexState::Legacy);
    }

    /// The same, as an older build's writer leaves it when it is killed
    /// after a commit: not closed cleanly.
    fn unclean_legacy(&self) {
        let source = self.tmp.path().join("source.axil");
        let db = axil_core::Axil::open(&source).build().unwrap();
        db.insert(
            "notes",
            json!({ "summary": "auth timeouts on long uploads" }),
        )
        .unwrap();
        drop(db);
        retry_busy(|| Storage::open(&source))
            .downgrade_table_index()
            .unwrap();
        let held = retry_busy(|| Ok(redb::Database::create(&source)?));
        let txn = held.begin_write().unwrap();
        txn.open_table(redb::TableDefinition::<&str, &str>::new("older_writer"))
            .unwrap()
            .insert("wrote", "this")
            .unwrap();
        txn.commit().unwrap();
        std::fs::copy(&source, self.db()).unwrap();
        drop(held);
        assert_eq!(self.state(), TableIndexState::Unclean);
    }

    fn state(&self) -> TableIndexState {
        retry_busy(|| Storage::table_index_state(self.db()))
    }

    /// Claude hooks that run the axil at `exe`.
    fn hooks_run(&self, exe: &Path) {
        std::fs::create_dir_all(self.dir.path().join(".claude")).unwrap();
        let command = format!("{} hook run --dialect claude", exe.display());
        std::fs::write(
            self.dir.path().join(".claude/settings.json"),
            json!({ "hooks": { "Stop": [{ "hooks": [{ "type": "command", "command": command }] }] } })
                .to_string(),
        )
        .unwrap();
    }
}

/// An axil from before `axil hook formats`, outside the project, as an older
/// release on PATH is.
fn older_axil(dir: &Path) -> PathBuf {
    let path = dir.join("older").join("axil");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        "#!/bin/sh\necho \"error: unrecognized subcommand\" >&2\nexit 2\n",
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn doctor_checks(out: &Output) -> Vec<Value> {
    let report: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "doctor printed no report ({e}): {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    report["checks"].as_array().unwrap().clone()
}

/// A command run with a hook's deadline is refused a legacy table index and
/// writes nothing; the same command without one converts it.
#[test]
fn a_hook_lookup_never_converts_a_legacy_table_index() {
    let p = Project::new();
    p.legacy();
    let deadline = SystemTime::now().duration_since(UNIX_EPOCH).unwrap() + Duration::from_secs(60);
    let db = p.db();
    let refused = p
        .command()
        .env("AXIL_HOOK_DEADLINE_MS", deadline.as_millis().to_string())
        .arg("--db")
        .arg(&db)
        .args(["list", "notes"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert_eq!(p.state(), TableIndexState::Legacy);

    let listed = p.axil(&["list", "notes"]);
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    assert_eq!(p.state(), TableIndexState::Keyed);
}

/// Doctor does not convert a legacy table index that the agent hooks' axil
/// could not open afterwards: it reports why and runs no other check. Once
/// the hooks' axil can open it, doctor runs as usual.
#[test]
fn doctor_leaves_a_legacy_database_alone_when_the_hooks_axil_is_too_old() {
    let p = Project::new();
    p.legacy();
    let bin = tempfile::tempdir().unwrap();
    p.hooks_run(&older_axil(bin.path()));

    let out = p.axil(&["doctor"]);
    assert_eq!(out.status.code(), Some(1));
    let checks = doctor_checks(&out);
    let named = |name: &str| {
        checks
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("no {name} check: {checks:?}"))
            .clone()
    };
    assert_eq!(named("hook_binary")["status"], "warning");
    assert_eq!(named("table_index")["status"], "warning");
    assert_eq!(checks.len(), 2, "no other check: {checks:?}");
    assert_eq!(p.state(), TableIndexState::Legacy);

    p.hooks_run(Path::new(env!("CARGO_BIN_EXE_axil")));
    let out = p.axil(&["doctor"]);
    let checks = doctor_checks(&out);
    assert!(
        checks.iter().all(|c| c["name"] != "table_index"),
        "{checks:?}"
    );
    assert_eq!(p.state(), TableIndexState::Keyed);
}

/// Doctor does not convert a legacy table index when it can't tell which
/// axil the agent hooks run, such as one named `axil-old`: it doesn't run
/// it, so whether that axil can open the converted file is unknown.
#[test]
fn doctor_leaves_a_legacy_database_alone_when_it_cant_verify_the_hooks_axil() {
    let p = Project::new();
    p.legacy();
    let bin = tempfile::tempdir().unwrap();
    let ran = bin.path().join("ran");
    let renamed = bin.path().join("axil-old");
    std::fs::write(
        &renamed,
        format!(
            "#!/bin/sh\ntouch {}\necho table-index-keyed\n",
            ran.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&renamed, std::fs::Permissions::from_mode(0o755)).unwrap();
    p.hooks_run(&renamed);

    let out = p.axil(&["doctor"]);
    assert_eq!(out.status.code(), Some(1));
    let checks = doctor_checks(&out);
    let hook = checks.iter().find(|c| c["name"] == "hook_binary").unwrap();
    assert_eq!(hook["status"], "warning", "{hook}");
    assert!(
        hook["detail"]
            .as_str()
            .unwrap()
            .contains("can't verify which axil the hooks run"),
        "{hook}"
    );
    assert!(
        checks.iter().any(|c| c["name"] == "table_index"),
        "{checks:?}"
    );
    assert_eq!(p.state(), TableIndexState::Legacy);
    assert!(!ran.exists(), "doctor ran the hooks' axil");
}

/// A daemon whose open converts a legacy table index, while the agent hooks
/// run an older axil, writes the warning to its log and serves the
/// database: printed in its open, the warning would keep it from serving.
#[test]
fn a_daemon_that_converts_logs_the_warning_and_serves() {
    let p = Project::new();
    p.legacy();
    let bin = tempfile::tempdir().unwrap();
    let older = older_axil(bin.path());
    p.hooks_run(&older);

    let db = p.db();
    let daemon = |args: &[&str]| {
        p.command()
            .env_remove("AXIL_DAEMON")
            .arg("--db")
            .arg(&db)
            .arg("daemon")
            .args(args)
            .stdin(Stdio::null())
            .output()
            .unwrap()
    };
    let start = daemon(&["start", "--idle-timeout", "20s"]);
    let log_file = p.dir.path().join(".axil/run/memory.axil/daemon.log");
    let log = std::fs::read_to_string(&log_file).unwrap_or_default();
    daemon(&["stop"]);
    assert!(
        start.status.success(),
        "start: {}{}\nlog: {log}",
        String::from_utf8_lossy(&start.stdout),
        String::from_utf8_lossy(&start.stderr)
    );
    assert!(
        log.contains("converted the table index") && log.contains(&*older.to_string_lossy()),
        "{log}"
    );
    assert_eq!(p.state(), TableIndexState::Keyed);
}

/// A command that converts a legacy table index says so when the agent
/// hooks run an axil that can't open it any more.
#[test]
fn converting_warns_when_the_hooks_axil_is_too_old() {
    let p = Project::new();
    p.legacy();
    let bin = tempfile::tempdir().unwrap();
    let older = older_axil(bin.path());
    p.hooks_run(&older);

    let out = p.axil(&["list", "notes"]);
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("converted the table index") && stderr.contains(&*older.to_string_lossy()),
        "{stderr}"
    );
    assert_eq!(p.state(), TableIndexState::Keyed);

    // Converted once: the next command says nothing.
    let out = p.axil(&["list", "notes"]);
    assert!(!String::from_utf8_lossy(&out.stderr).contains("converted the table index"));
}

/// A prompt hook on a legacy file an older writer left unclean runs no
/// lookup, whose open would repair and convert it under the hook's time
/// limit: its drainer does both, and the session's problems say why the
/// prompt got no memory.
#[test]
fn a_prompt_hook_leaves_an_unclean_legacy_file_to_its_drainer() {
    let p = Project::new();
    p.unclean_legacy();
    let mut hook = p
        .command()
        .env("CLAUDE_PROJECT_DIR", p.dir.path())
        .args(["hook", "run", "--dialect", "claude"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let event = json!({
        "hook_event_name": "UserPromptSubmit",
        "session_id": "s1",
        "cwd": p.dir.path(),
        "prompt": "what did we note about auth timeouts?",
    });
    hook.stdin
        .take()
        .unwrap()
        .write_all(event.to_string().as_bytes())
        .unwrap();
    let out = hook.wait_with_output().unwrap();
    assert!(out.status.success());

    let log = p.dir.path().join(".axil/hook-queue/drain.log");
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if text.contains("repaired the database") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the drainer did not convert it: {text}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(p.state(), TableIndexState::Keyed);
    let problems = std::fs::read_to_string(p.tmp.path().join("axil-session-s1.problems")).unwrap();
    assert!(problems.contains("lookups_skipped"), "{problems}");
}

/// `axil bridge auto` only reads its members, so it opens them read-only: a
/// member whose table index is still in the legacy layout is read, and
/// stays legacy.
#[test]
fn the_bridge_scan_converts_no_member() {
    let p = Project::new();
    let member = |name: &str| p.dir.path().join(name).join(".axil/memory.axil");
    let symbol = "rust-analyzer cargo shared 0.1.0 lib/Shared#";
    for name in ["a", "b"] {
        let db = member(name);
        std::fs::create_dir_all(db.parent().unwrap()).unwrap();
        let handle = axil_core::Axil::open(&db).build().unwrap();
        handle
            .insert(
                "_entities",
                json!({ "name": "Shared", "canonical_id": symbol }),
            )
            .unwrap();
    }
    retry_busy(|| Storage::open(member("b")))
        .downgrade_table_index()
        .unwrap();
    let init = p.command().args(["workspace", "init"]).output().unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );

    let scan = p
        .command()
        .current_dir(p.dir.path().join("a"))
        .arg("--db")
        .arg(member("a"))
        .args(["bridge", "auto", "--dry-run"])
        .output()
        .unwrap();
    assert!(
        scan.status.success(),
        "{}",
        String::from_utf8_lossy(&scan.stderr)
    );
    let report: Value = serde_json::from_slice(&scan.stdout).unwrap();
    let plan = report["plan"].as_array().unwrap();
    assert!(
        plan.iter().any(|b| b["remote_canonical"] == symbol),
        "the legacy member was read: {report}"
    );
    assert_eq!(
        retry_busy(|| Storage::table_index_state(member("b"))),
        TableIndexState::Legacy
    );
}
