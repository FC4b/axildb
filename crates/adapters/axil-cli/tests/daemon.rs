//! The warm daemon as a user runs it: `axil daemon start`, `send`, `stop`
//! and `status`, against a temp database, with the built binary.
//!
//! Databases get a vector store only when the bge-small model is installed,
//! so no test downloads one; the vector checks are skipped loudly without
//! it. Every daemon is started with a short idle timeout and stopped when
//! its test ends, pass or fail.

#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_axil");

/// What the daemon reads from its environment, removed so each test sees
/// the defaults.
const ENV: &[&str] = &[
    "AXIL_DB",
    "AXIL_MCP_TOOLS",
    "AXIL_VERBOSE",
    "AXIL_EMBED_SESSIONS",
    "AXIL_SLOW_QUERY_LOG",
    "AXIL_DAEMON",
];

struct Fixture {
    _tmp: tempfile::TempDir,
    proj: PathBuf,
    db: PathBuf,
    vectors: bool,
}

fn model_available() -> bool {
    #[cfg(feature = "embed")]
    {
        use axil_vector::models::EmbeddingModel;
        if axil_vector::download::is_model_available(&EmbeddingModel::BgeSmall) {
            return true;
        }
    }
    eprintln!("no vector store: the bge-small model is not installed");
    false
}

/// A project with a database holding a few records.
fn fixture() -> Fixture {
    // Short enough for the socket to sit in the run directory.
    let tmp = tempfile::Builder::new().prefix("d").tempdir().unwrap();
    let proj = tmp.path().canonicalize().unwrap().join("p");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::create_dir_all(proj.join(".axil")).unwrap();
    std::fs::write(
        proj.join("src/auth.rs"),
        "/// Validate a login token.\npub fn validate_token(t: &str) -> bool { !t.is_empty() }\n",
    )
    .unwrap();
    std::fs::write(proj.join("Cargo.toml"), "[package]\nname = \"p\"\n").unwrap();
    let db = proj.join(".axil/memory.axil");
    let fx = |args: &[&str]| {
        let out = axil_in(BIN, &proj, args, None);
        assert!(out.status.success(), "{args:?}: {}", text(&out.stderr));
    };
    fx(&["init", db.to_str().unwrap()]);
    let vectors = model_available();
    if !vectors {
        let _ = std::fs::remove_file(proj.join(".axil/memory.axil.vec"));
    }
    for (table, data) in [
        (
            "decisions",
            r#"{"summary":"Cap login tokens at 512 bytes"}"#,
        ),
        (
            "errors",
            r#"{"error":"auth timeout during uploads","fix":"heartbeat"}"#,
        ),
        (
            "context",
            r#"{"type":"architecture","summary":"auth.rs validates tokens"}"#,
        ),
    ] {
        fx(&["store", table, data]);
    }
    Fixture {
        _tmp: tmp,
        proj,
        db,
        vectors,
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Run `exe` with `args` in `dir`, stdin from `stdin` or nothing.
fn axil_in(exe: &str, dir: &Path, args: &[&str], stdin: Option<&[u8]>) -> Output {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in ENV {
        cmd.env_remove(name);
    }
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = cmd.spawn().unwrap();
    if let Some(bytes) = stdin {
        child.stdin.take().unwrap().write_all(bytes).unwrap();
    }
    child.wait_with_output().unwrap()
}

impl Fixture {
    fn axil(&self, args: &[&str]) -> Output {
        axil_in(BIN, &self.proj, args, None)
    }

    fn start_with(&self, exe: &str, idle: &str) -> Value {
        let out = axil_in(
            exe,
            &self.proj,
            &["daemon", "start", "--idle-timeout", idle],
            None,
        );
        assert!(
            out.status.success(),
            "start: {}{}",
            text(&out.stdout),
            text(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }

    fn start(&self) -> Value {
        self.start_with(BIN, "60s")
    }

    fn status(&self) -> Value {
        let out = self.axil(&["daemon", "status", "--json"]);
        assert!(out.status.success(), "{}", text(&out.stderr));
        serde_json::from_slice(&out.stdout).unwrap()
    }

    fn send(&self, args: &[&str], stdin: Option<&[u8]>) -> Output {
        let mut argv = vec!["daemon", "send", "--"];
        argv.extend_from_slice(args);
        axil_in(BIN, &self.proj, &argv, stdin)
    }

    /// [`Fixture::send`], again while the daemon answers that it is
    /// reloading: a tick may have found the change first.
    fn send_settled(&self, args: &[&str]) -> Output {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let out = self.send(args, None);
            let reloading = text(&out.stderr).contains("\"reason\":\"unavailable\"");
            if !reloading || Instant::now() >= deadline {
                return out;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn run_dir(&self) -> PathBuf {
        self.proj.join(".axil/run/memory.axil")
    }

    /// Whether a daemon holds the lock.
    fn locked(&self) -> bool {
        let Ok(file) = std::fs::File::open(self.run_dir().join("daemon.lock")) else {
            return false;
        };
        matches!(file.try_lock(), Err(std::fs::TryLockError::WouldBlock))
    }

    fn info(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.run_dir().join("daemon.json")).unwrap()).unwrap()
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.run_dir().join("daemon.log")).unwrap_or_default()
    }

    /// The stores no handle has open any more: each opens read-only, which
    /// redb refuses for a file a writer left without closing.
    fn closed_cleanly(&self) -> Result<(), String> {
        for suffix in ["", ".vec", ".graph", ".ts"] {
            let path = PathBuf::from(format!("{}{suffix}", self.db.display()));
            if path.exists() {
                redb::ReadOnlyDatabase::open(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.locked() {
            let _ = self.axil(&["daemon", "stop", "--wait", "10"]);
        }
    }
}

fn wait_for(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    done()
}

/// The `daemon run` processes serving `db`.
fn daemons_for(db: &Path) -> Vec<String> {
    let out = Command::new("ps")
        .args(["-axo", "pid=,command="])
        .output()
        .unwrap();
    text(&out.stdout)
        .lines()
        .filter(|l| l.contains(&*db.to_string_lossy()) && l.contains("daemon run"))
        .map(str::to_string)
        .collect()
}

fn signal(pid: u64, sig: &str) {
    let status = Command::new("kill")
        .args([sig, &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

/// Send one raw request frame and read the answer frame.
fn raw_call(socket: &Path, request: &str) -> Value {
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream
        .write_all(&(request.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
    stream.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[test]
fn eight_starts_at_once_leave_one_daemon() {
    let fx = fixture();
    let starts: Vec<_> = (0..8)
        .map(|_| {
            let mut cmd = Command::new(BIN);
            cmd.args(["daemon", "start", "--idle-timeout", "60s"])
                .current_dir(&fx.proj)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for name in ENV {
                cmd.env_remove(name);
            }
            cmd.spawn().unwrap()
        })
        .collect();
    let reports: Vec<Value> = starts
        .into_iter()
        .map(|child| {
            let out = child.wait_with_output().unwrap();
            assert!(out.status.success(), "{}", text(&out.stderr));
            serde_json::from_slice(&out.stdout).unwrap()
        })
        .collect();
    let pid = &reports[0]["pid"];
    assert!(
        reports
            .iter()
            .all(|r| &r["pid"] == pid && r["state"] == "ready"),
        "{reports:?}"
    );
    let started = reports.iter().filter(|r| r["started"] == true).count();
    assert_eq!(started, 1, "{reports:?}");
    // The losers leave as soon as the winner answers.
    assert!(
        wait_for(Duration::from_secs(10), || daemons_for(&fx.db).len() == 1),
        "{:?}",
        daemons_for(&fx.db)
    );
    assert_eq!(fx.status()["live"]["pid"], *pid);
}

#[test]
fn a_killed_daemon_refuses_connections_and_the_next_start_takes_over() {
    let fx = fixture();
    let first = fx.start();
    let pid = first["pid"].as_u64().unwrap();
    let socket = PathBuf::from(first["socket"].as_str().unwrap());
    signal(pid, "-KILL");
    assert!(wait_for(Duration::from_secs(10), || !fx.locked()));

    // The socket file is still there, and nothing listens on it.
    let err = UnixStream::connect(&socket).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::ConnectionRefused);
    // Not run, as a command the daemon declines is not.
    let sent = fx.send(&["boot"], None);
    assert_eq!(sent.status.code(), Some(125), "{}", text(&sent.stderr));
    // The stores are left as a killed writer leaves them.
    assert!(fx.closed_cleanly().is_err());
    let status = fx.status();
    assert_eq!(status["running"], false);
    assert_eq!(status["last"]["state"], "ready");

    let second = fx.start();
    assert_ne!(second["pid"].as_u64().unwrap(), pid);
    let boot = fx.send(&["boot"], None);
    assert!(boot.status.success(), "{}", text(&boot.stderr));
    if fx.vectors {
        assert!(
            fx.log().contains("was not closed cleanly; repaired it"),
            "{}",
            fx.log()
        );
    }
}

#[test]
fn idle_sigterm_and_stop_each_close_every_store() {
    let fx = fixture();

    fx.start_with(BIN, "2s");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["exit_reason"], "idle");

    let pid = fx.start()["pid"].as_u64().unwrap();
    assert!(fx.send(&["boot"], None).status.success());
    signal(pid, "-TERM");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["exit_reason"], "signal 15");

    fx.start();
    assert!(fx.send(&["boot"], None).status.success());
    let out = fx.axil(&["daemon", "stop"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stopped["stopped"], true, "{stopped}");
    assert_eq!(stopped["clean"], true, "{stopped}");
    assert!(!fx.locked());
    fx.closed_cleanly().unwrap();
    assert_eq!(fx.info()["state"], "stopped");
    let status = fx.status();
    assert_eq!(status["running"], false);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(
        human.starts_with("daemon: not running; last one stopped at"),
        "{human}"
    );
}

#[test]
fn replacing_the_executable_retires_the_daemon() {
    let fx = fixture();
    let bin_dir = fx.proj.join("bin");
    std::fs::create_dir(&bin_dir).unwrap();
    let copy = bin_dir.join("axil");
    std::fs::copy(BIN, &copy).unwrap();
    let pid = fx.start_with(copy.to_str().unwrap(), "60s")["pid"]
        .as_u64()
        .unwrap();

    // A build installed over it, as `cargo install` does: a new file
    // renamed into place.
    let new = bin_dir.join("axil.new");
    std::fs::copy(BIN, &new).unwrap();
    std::fs::rename(&new, &copy).unwrap();
    assert!(
        wait_for(Duration::from_secs(10), || !fx.locked()),
        "pid {pid} still running"
    );
    assert_eq!(fx.info()["exit_reason"], "its executable changed");
    fx.closed_cleanly().unwrap();

    // A symlink moved to another build, as a package manager upgrades: the
    // binary the daemon runs is unchanged, the path it was started by now
    // leads to another.
    let (one, two) = (bin_dir.join("axil-one"), bin_dir.join("axil-two"));
    std::fs::copy(BIN, &one).unwrap();
    std::fs::copy(BIN, &two).unwrap();
    let link = bin_dir.join("axil-link");
    std::os::unix::fs::symlink(&one, &link).unwrap();
    let pid = fx.start_with(link.to_str().unwrap(), "60s")["pid"]
        .as_u64()
        .unwrap();
    let moved = bin_dir.join("axil-link.new");
    std::os::unix::fs::symlink(&two, &moved).unwrap();
    std::fs::rename(&moved, &link).unwrap();
    assert!(
        wait_for(Duration::from_secs(10), || !fx.locked()),
        "pid {pid} still running"
    );
    assert_eq!(fx.info()["exit_reason"], "its executable changed");
}

#[test]
fn a_dropped_engine_and_a_failed_graph_reload_the_daemon() {
    let fx = fixture();
    let socket = PathBuf::from(fx.start()["socket"].as_str().unwrap());
    let reloads = || fx.status()["live"]["reloads"].as_u64().unwrap();
    assert!(fx.send(&["since", "1h"], None).status.success());
    assert_eq!(reloads(), 0);

    // `compact --drop-engine` removes the store without opening the database.
    let dropped = fx.axil(&["compact", "--drop-engine", "ts"]);
    assert!(dropped.status.success(), "{}", text(&dropped.stderr));
    std::thread::sleep(Duration::from_millis(300));
    let since = fx.send_settled(&["since", "1h"]);
    assert_eq!(since.status.code(), Some(125), "{}", text(&since.stderr));
    assert!(text(&since.stderr).contains("\"reason\":\"lacks\""));
    assert_eq!(reloads(), 1);
    assert!(fx.status()["live"]["last_reload"]
        .as_str()
        .unwrap()
        .contains(".ts was removed"));

    let ack = raw_call(&socket, r#"{"op":"debug_fail_graph"}"#);
    assert_eq!(ack["ok"], true);
    assert!(fx.send_settled(&["boot"]).status.success());
    assert_eq!(reloads(), 2);
    assert_eq!(
        fx.status()["live"]["last_reload"],
        "the graph failed to load"
    );
    assert_eq!(fx.status()["live"]["state"], "ready");
}

/// What the hot commands print goes to the client, never into the
/// daemon's log: every line there is one of the daemon's own.
#[test]
fn the_hot_commands_print_nothing_into_the_log() {
    let fx = fixture();
    fx.start();
    let cp = r#"{"goal":"g","state":"s","next_steps":["n"]}"#;
    let cases: Vec<(Vec<&str>, Option<&[u8]>)> = vec![
        (vec!["boot"], None),
        (vec!["boot", "--boot-format", "narrative"], None),
        (vec!["boot", "--files", "src/auth.rs"], None),
        (vec!["recall-for-file", "src/auth.rs"], None),
        (vec!["recall", "auth timeout"], None),
        (
            vec!["recall", "auth timeout", "--recall-format", "context-block"],
            None,
        ),
        (vec!["--format", "table", "recall", "auth timeout"], None),
        (vec!["code-search", "validate token"], None),
        (vec!["fts", "tokens"], None),
        (vec!["fts", "tokens", "--jsonl"], None),
        (vec!["since", "1h"], None),
        (vec!["store", "decisions", r#"{"summary":"sent"}"#], None),
        (
            vec!["store", "context", "-"],
            Some(br#"{"summary":"piped"}"#),
        ),
        (vec!["store", "decisions", "{nope"], None),
        (vec!["checkpoint", cp], None),
        (vec!["checkpoint", "-"], Some(cp.as_bytes())),
        (vec!["--help"], None),
        (vec!["store"], None),
        (vec!["--version"], None),
    ];
    let mut served = 0;
    for (args, stdin) in &cases {
        let out = fx.send(args, *stdin);
        if out.status.code() != Some(125) {
            served += 1;
            assert!(
                !out.stdout.is_empty() || !out.stderr.is_empty(),
                "{args:?} printed nothing"
            );
        }
    }
    assert!(served >= cases.len() - 2, "only {served} served");
    let log = fx.log();
    let foreign: Vec<&str> = log
        .lines()
        .filter(|line| {
            let mut parts = line.splitn(3, ' ');
            let (stamp, who) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            !(stamp.len() == 20 && stamp.ends_with('Z') && who.starts_with("axil"))
        })
        .collect();
    assert!(
        foreign.is_empty(),
        "not the daemon's own lines:\n{}",
        foreign.join("\n")
    );
    assert!(log.contains("ready"), "{log}");

    // A checkpoint given its JSON does not read stdin, so an open pipe
    // nobody writes to does not hold it up.
    let mut cmd = Command::new(BIN);
    cmd.args(["daemon", "send", "--", "checkpoint", cp])
        .current_dir(&fx.proj)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in ENV {
        cmd.env_remove(name);
    }
    let mut child = cmd.spawn().unwrap();
    let open_stdin = child.stdin.take();
    let finished = wait_for(Duration::from_secs(60), || {
        child.try_wait().unwrap().is_some()
    });
    drop(open_stdin);
    let out = child.wait_with_output().unwrap();
    assert!(finished, "send waited on stdin");
    assert!(out.status.success(), "{}", text(&out.stderr));
}

/// `AXIL_DAEMON=off` keeps a daemon from starting.
#[test]
fn the_kill_switch_keeps_the_daemon_from_starting() {
    let fx = fixture();
    let out = Command::new(BIN)
        .args(["daemon", "start"])
        .current_dir(&fx.proj)
        .env_remove("AXIL_DB")
        .env("AXIL_DAEMON", "off")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("AXIL_DAEMON=off"),
        "{}",
        text(&out.stderr)
    );
    assert!(!fx.locked());
}

/// A database whose open prints a warning is not served: `start` says
/// why and fails, and so does `status`, and the next `start` tries again.
#[test]
fn a_database_whose_open_warns_is_not_served() {
    let fx = fixture();
    std::fs::write(fx.proj.join("axil.toml"), "[healing\nevent_log = true\n").unwrap();
    // Four at once: whichever daemon wins, every start reports its reason.
    let starts: Vec<_> = (0..4)
        .map(|_| {
            let mut cmd = Command::new(BIN);
            cmd.args(["daemon", "start", "--idle-timeout", "60s"])
                .current_dir(&fx.proj)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for name in ENV {
                cmd.env_remove(name);
            }
            cmd.spawn().unwrap()
        })
        .collect();
    for child in starts {
        let out = child.wait_with_output().unwrap();
        assert_eq!(out.status.code(), Some(1), "{}", text(&out.stderr));
        let report: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["started"], false);
        assert!(
            report["blocked"]
                .as_str()
                .unwrap()
                .contains("invalid config"),
            "{report}"
        );
    }
    assert!(!fx.locked());
    let status = fx.status();
    assert_eq!(status["running"], false);
    assert!(status["blocked"]["reason"]
        .as_str()
        .unwrap()
        .contains("invalid config"));
    assert!(text(&fx.axil(&["daemon", "status"]).stdout).contains("not served: "));

    std::fs::remove_file(fx.proj.join("axil.toml")).unwrap();
    assert_eq!(fx.start()["state"], "ready");
    assert_eq!(fx.status()["blocked"], Value::Null);
}

/// Two databases in one directory each have a daemon of their own:
/// starting, stopping and asking about one leaves the other be.
#[test]
fn two_databases_in_one_directory_have_a_daemon_each() {
    let fx = fixture();
    let other = fx.proj.join(".axil/other.axil");
    let other = other.to_str().unwrap();
    let init = fx.axil(&["init", other]);
    assert!(init.status.success(), "{}", text(&init.stderr));
    let on_other = |args: &[&str]| {
        let mut argv = vec!["--db", other];
        argv.extend_from_slice(args);
        fx.axil(&argv)
    };

    let first = fx.start();
    let status: Value =
        serde_json::from_slice(&on_other(&["daemon", "status", "--json"]).stdout).unwrap();
    assert_eq!(status["running"], false, "{status}");

    let out = on_other(&["daemon", "start", "--idle-timeout", "60s"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let second: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(second["started"], true, "{second}");
    assert_ne!(second["pid"], first["pid"]);
    assert_ne!(second["socket"], first["socket"]);

    let out = on_other(&["daemon", "stop"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stopped["stopped"], true, "{stopped}");
    assert_eq!(stopped["pid"], second["pid"]);
    assert_eq!(fx.status()["live"]["pid"], first["pid"]);
}

/// A daemon slow to shut down (here holding the database 7 s, as a long
/// warm-up would): `status` says it is stopping, `stop` waits for it and
/// reports it closed the database, and a `start` meanwhile waits for it to
/// go and takes over.
#[test]
fn a_slow_shutdown_is_waited_for() {
    let fx = fixture();
    let slow_shutdown = || {
        let first = fx.start();
        let socket = PathBuf::from(first["socket"].as_str().unwrap());
        let hold = raw_call(&socket, r#"{"op":"debug_hold","ms":7000}"#);
        assert_eq!(hold["ok"], true);
        let shutdown = raw_call(&socket, r#"{"op":"shutdown","reason":"test"}"#);
        assert_eq!(shutdown["ok"], true);
        assert!(wait_for(Duration::from_secs(10), || fx.info()["state"] == "stopping"));
        first["pid"].as_u64().unwrap()
    };

    let pid = slow_shutdown();
    assert_eq!(fx.status()["stopping"], true);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(human.starts_with("daemon: stopping"), "{human}");
    let out = fx.axil(&["daemon", "stop", "--wait", "20"]);
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        (
            &stopped["stopped"],
            &stopped["clean"],
            stopped["pid"].as_u64()
        ),
        (&Value::Bool(true), &Value::Bool(true), Some(pid)),
        "{stopped}"
    );
    fx.closed_cleanly().unwrap();

    let pid = slow_shutdown();
    let next = fx.start();
    assert_ne!(next["pid"].as_u64().unwrap(), pid);
    assert_eq!(next["started"], true, "{next}");
}

/// A backup copied over the database in place, as `cp` does, keeping its
/// inode: the daemon stops without closing the database, which would write
/// its stale view over the copy, so the copy is what the next command
/// reads.
#[test]
fn a_backup_copied_over_the_database_is_kept() {
    let fx = fixture();
    let backup = std::fs::read(&fx.db).unwrap();
    fx.start();
    let stored = fx.send(
        &["store", "decisions", r#"{"summary":"after the backup"}"#],
        None,
    );
    assert!(stored.status.success(), "{}", text(&stored.stderr));
    let id = serde_json::from_slice::<Value>(&stored.stdout).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&fx.db)
            .unwrap();
        file.write_all(&backup).unwrap();
    }
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    let info = fx.info();
    assert!(
        info["exit_reason"]
            .as_str()
            .unwrap()
            .starts_with("its files were overwritten"),
        "{info}"
    );
    assert_eq!(info["closed_cleanly"], false);
    let human = text(&fx.axil(&["daemon", "status"]).stdout);
    assert!(human.contains("for the next open to repair"), "{human}");
    assert!(
        std::fs::read(&fx.db).unwrap() == backup,
        "the copy was written over"
    );
    let get = fx.axil(&["get", &id]);
    assert!(
        text(&get.stderr).contains("not found"),
        "{}",
        text(&get.stderr)
    );
}

/// A daemon that holds the lock but answers nothing (stopped with SIGSTOP
/// here) is sent SIGTERM by `stop`, which first makes sure the process is
/// this database's daemon.
#[test]
fn stop_signals_a_daemon_that_does_not_answer() {
    let fx = fixture();
    let pid = fx.start()["pid"].as_u64().unwrap();
    signal(pid, "-STOP");
    let out = fx.axil(&["daemon", "stop", "--wait", "2"]);
    signal(pid, "-CONT");
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    // Still stopped when the wait ran out; the signal was waiting for it.
    assert_eq!(report["running"], true, "{report}");
    assert!(wait_for(Duration::from_secs(15), || !fx.locked()));
    assert_eq!(fx.info()["exit_reason"], "signal 15");
    fx.closed_cleanly().unwrap();
}
