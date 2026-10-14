//! `axil daemon {start, stop, status, logs}`: running a database's daemon by
//! hand. Nothing forwards commands to it on its own yet; `daemon send` sends
//! one command line, to try it. Every other command leases the database from
//! it (see [`super::lease`]).
//!
//! `AXIL_DAEMON` is the kill switch: `off` keeps `daemon start` from
//! starting one. Unset, it is `on`: a daemon runs only when started by hand.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::paths::{write_atomic, RunDir};
use super::proto::{self, Ack, Hello, Identity, Op, RunReply};
use super::server;
use super::state::{graph_name, DaemonInfo};
use crate::cli_io;
use crate::{Output, EXIT_ERROR, EXIT_OK};

/// The exit code of a `daemon send` the daemon did not run.
const NOT_SERVED_EXIT: i32 = 125;

/// The daemon commands.
#[derive(clap::Subcommand)]
pub enum DaemonCommand {
    /// Start the daemon for this database in the background, and wait
    /// until it serves. Does nothing when one is running already.
    Start {
        /// Stop after this long without a command to run (90s, 30m, 2h).
        #[arg(long, default_value = "30m", value_parser = parse_duration)]
        idle_timeout: Duration,
        /// How long to wait for it to be ready, in seconds.
        #[arg(long, default_value = "30")]
        wait: u64,
    },
    /// Be the daemon, in this process. What `start` runs in the background.
    #[command(hide = true)]
    Run {
        /// Stop after this long without a command to run.
        #[arg(long, default_value = "30m", value_parser = parse_duration)]
        idle_timeout: Duration,
        /// The path `start` was run by, symlinks unresolved: the daemon
        /// retires once it leads to another binary.
        #[arg(long)]
        launch_path: Option<PathBuf>,
    },
    /// Ask the daemon to stop, and wait until it has exited. `clean`
    /// says whether it closed the database; when it could not, the next
    /// open repairs it.
    Stop {
        /// How long to wait, in seconds.
        #[arg(long, default_value = "15")]
        wait: u64,
    },
    /// Report on the daemon: what it is doing while it runs, and how it
    /// ended when it doesn't.
    Status {
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Print the end of the daemon's log.
    Logs {
        /// How many lines.
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,
    },
    /// Send one command line to the daemon and print what it printed,
    /// exiting with its exit code; 125 when it was not run, by the daemon's
    /// choice or because none answers. Give the command line after `--`.
    #[command(hide = true)]
    Send {
        /// The command line, without the program name.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        args: Vec<String>,
    },
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    axil_core::config::parse_duration_secs(s)
        .map(Duration::from_secs)
        .ok_or_else(|| format!("'{s}' is not a duration: use a number of seconds, or 90s, 30m, 2h"))
}

/// What `AXIL_DAEMON` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// No daemon runs.
    Off,
    /// A daemon runs when started by hand.
    On,
    /// A daemon is also started on demand. Treated as `on` for now.
    Auto,
}

/// `AXIL_DAEMON`: `off`, `on` or `auto`; `on` when unset or anything else.
pub fn mode() -> Mode {
    match std::env::var("AXIL_DAEMON").as_deref() {
        Ok("off") => Mode::Off,
        Ok("auto") => Mode::Auto,
        _ => Mode::On,
    }
}

/// Run a daemon command for the database `db_opt` names (or the one found
/// from the working directory).
pub fn dispatch(db_opt: &Option<PathBuf>, command: DaemonCommand, out: &Output) -> Result<i32> {
    let db = crate::require_db(db_opt)?;
    match command {
        DaemonCommand::Start { idle_timeout, wait } => {
            start(&db, idle_timeout, Duration::from_secs(wait), out)
        }
        DaemonCommand::Run {
            idle_timeout,
            launch_path,
        } => server::run_daemon(&db, idle_timeout, launch_path),
        DaemonCommand::Stop { wait } => stop(&db, Duration::from_secs(wait), out),
        DaemonCommand::Status { json } => status(&db, json, out),
        DaemonCommand::Logs { lines } => logs(&db, lines),
        DaemonCommand::Send { args } => send(&db, args),
    }
}

/// The run directory of the database at `db`, which must exist.
fn run_dir(db: &Path) -> Result<(PathBuf, RunDir)> {
    let key = cli_io::db_key(db).with_context(|| format!("no database at {}", db.display()))?;
    anyhow::ensure!(key.is_file(), "no database at {}", key.display());
    let run = RunDir::of(&key);
    Ok((key, run))
}

fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The daemon answering on `run`'s socket, if it is the daemon of `key`.
/// One that serves another database (two whose socket paths were too long
/// share the fallback directory, and could share a name) is an error: it
/// is not this database's to start, stop or report on.
fn hello_for(key: &Path, run: &RunDir, timeout: Duration) -> Result<Option<Hello>> {
    let Ok(hello) = proto::hello(run.socket(), timeout) else {
        return Ok(None);
    };
    // One that does not say is taken to be this database's: the socket is
    // in its run directory.
    anyhow::ensure!(
        hello.db.is_empty() || Path::new(&hello.db) == key,
        "{} is answered by the daemon of another database, {}",
        run.socket().display(),
        hello.db
    );
    Ok(Some(hello))
}

fn start(db: &Path, idle_timeout: Duration, wait: Duration, out: &Output) -> Result<i32> {
    anyhow::ensure!(
        mode() != Mode::Off,
        "the daemon is turned off (AXIL_DAEMON=off)"
    );
    let started_ms = epoch_ms();
    let (key, run) = run_dir(db)?;
    run.create()
        .with_context(|| format!("can't use {}", run.dir().display()))?;
    let own = proto::ExeId::current()?.build_id();
    // One that is starting or reloading is waited for, not replaced.
    let running = hello_for(&key, &run, Duration::from_secs(1))?
        .filter(|hello| hello.state != "stopping")
        .map(|hello| hello.pid);
    let mut child = match running {
        Some(_) => None,
        None => Some(spawn(&key, &run, idle_timeout)?),
    };

    let deadline = Instant::now() + wait;
    let mut blocked: Option<String> = None;
    loop {
        if let Some(hello) = hello_for(&key, &run, Duration::from_secs(1))? {
            // One that let go of the database for a lease serves again
            // once the lease ends.
            if hello.state == "ready" || hello.state == "leased" {
                out.print(&json!({
                    "started": child.as_ref().is_some_and(|c| c.id() == hello.pid),
                    "running": true,
                    "pid": hello.pid,
                    "state": hello.state,
                    "build_id": hello.build_id,
                    "same_build": hello.build_id == own,
                    "socket": run.socket(),
                }));
                return Ok(EXIT_OK);
            }
        }
        if let Some(c) = child.as_mut() {
            // Reaped, so it is not left a zombie for as long as this waits.
            if c.try_wait()?.is_some() {
                child = None;
            }
        }
        // A daemon that would not serve says why, and stops: the one this
        // spawned, or the one it lost the election to. A reason recorded
        // before this started is an earlier daemon's.
        if blocked.is_none() {
            blocked = read_json_file::<Value>(&run.blocked())
                .filter(|b| b["at_ms"].as_u64().is_some_and(|at| at >= started_ms))
                .map(|b| b["reason"].as_str().unwrap_or("unknown").to_string());
        }
        // Reported once that daemon has let the lock go and the one this
        // spawned has exited, so none is left behind.
        if let Some(reason) = &blocked {
            if (child.is_none() && !server::lock_held(&run)) || Instant::now() >= deadline {
                out.print(&json!({"started": false, "running": false, "blocked": reason}));
                return Ok(EXIT_ERROR);
            }
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "the daemon was not ready after {}s; see `axil daemon logs`",
                wait.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Spawn `axil daemon run` for the database `key`, in a session of its
/// own, its output appended to the log. Whatever kept the last one from
/// serving is tried again; its record stays until a daemon serves.
fn spawn(key: &Path, run: &RunDir, idle_timeout: Duration) -> Result<std::process::Child> {
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.log())
        .with_context(|| format!("can't open {}", run.log().display()))?;
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--db")
        .arg(key)
        .args(["daemon", "run", "--idle-timeout"])
        .arg(format!("{}s", idle_timeout.as_secs()));
    if let Some(launch) = launch_path() {
        cmd.arg("--launch-path").arg(launch);
    }
    cmd.current_dir("/")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe, as code between fork and
        // exec must be. A session of its own keeps the daemon from the
        // terminal's hangup.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let child = cmd.spawn().context("can't start the daemon")?;
    let stamp = json!({"pid": child.id(), "at": chrono::Utc::now().to_rfc3339()});
    let _ = write_atomic(&run.spawn_stamp(), stamp.to_string().as_bytes());
    Ok(child)
}

/// The path this binary was started by, before symlinks are resolved:
/// `argv[0]` when it names a file, made absolute, else its first match on
/// `PATH`. Only a path that leads to this very binary is given; the daemon
/// started with it retires once it leads elsewhere, as when a package
/// manager moves a symlink to a new build.
fn launch_path() -> Option<PathBuf> {
    let arg0 = PathBuf::from(std::env::args_os().next()?);
    let path = if arg0.components().count() > 1 {
        std::env::current_dir().ok()?.join(arg0)
    } else {
        let dirs = std::env::var_os("PATH")?;
        std::env::split_paths(&dirs)
            .map(|dir| dir.join(&arg0))
            .find(|candidate| candidate.is_file())?
    };
    let this = proto::ExeId::current().ok()?;
    (proto::ExeId::of(&path).ok()? == this).then_some(path)
}

fn stop(db: &Path, wait: Duration, out: &Output) -> Result<i32> {
    let (key, run) = run_dir(db)?;
    let info = read_json_file::<DaemonInfo>(&run.info());
    let pid = info.as_ref().map(|info| info.pid);
    let op = Op::Shutdown {
        reason: "axil daemon stop".into(),
    };
    let asked = match hello_for(&key, &run, Duration::from_secs(5))? {
        Some(_) => proto::call::<Ack>(run.socket(), &op, Duration::from_secs(5)).is_ok(),
        None => false,
    };
    let mut signalled = false;
    if !asked {
        if !server::lock_held(&run) {
            out.print(&json!({"stopped": false, "running": false}));
            return Ok(EXIT_OK);
        }
        // One shutting down already has unlinked its socket: wait for it.
        // One that holds the lock but does not answer (every connection
        // taken, say) is sent SIGTERM, once it is sure to be this
        // database's daemon.
        if !server::said_stopping(&run) {
            if let Some(pid) = pid.filter(|pid| is_daemon_of(*pid, &key)) {
                // SAFETY: kill only sends a signal.
                signalled = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } == 0;
            }
            if !signalled {
                out.print(&json!({
                    "stopped": false,
                    "running": true,
                    "pid": pid,
                    "error": "a daemon holds the lock but does not answer",
                }));
                return Ok(EXIT_ERROR);
            }
        }
    }
    let deadline = Instant::now() + wait;
    while server::lock_held(&run) {
        if Instant::now() >= deadline {
            out.print(&json!({
                "stopped": false,
                "running": true,
                "pid": pid,
                "error": format!("still running after {}s", wait.as_secs()),
            }));
            return Ok(EXIT_ERROR);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // The lock goes with the process when the daemon could not close the
    // database; its record says which.
    let last = read_json_file::<DaemonInfo>(&run.info());
    let clean = last
        .as_ref()
        .is_some_and(|info| info.state == "stopped" && info.closed_cleanly == Some(true));
    let mut report = json!({"stopped": true, "pid": pid, "clean": clean});
    if signalled {
        report["signalled"] = json!(true);
    }
    out.print(&report);
    Ok(EXIT_OK)
}

/// Whether process `pid` is `axil daemon run` for the database `key`.
fn is_daemon_of(pid: u32, key: &Path) -> bool {
    let Ok(out) = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .stderr(Stdio::null())
        .output()
    else {
        return false;
    };
    let command = String::from_utf8_lossy(&out.stdout);
    command.contains(&format!("--db {} daemon run", key.display()))
}

/// The status report: the record the daemon left, the lock and socket
/// probed now, and the live figures when it answers.
fn report(key: &Path, run: &RunDir) -> Value {
    let last = read_json_file::<DaemonInfo>(&run.info());
    let answer = proto::call::<Value>(run.socket(), &Op::Status, Duration::from_secs(2)).ok();
    // Another database's daemon on this socket is not this one's.
    let other_db = answer
        .as_ref()
        .and_then(|a| a["db"].as_str())
        .filter(|db| Path::new(db) != key)
        .map(str::to_string);
    let live = answer.filter(|_| other_db.is_none());
    let lock_held = server::lock_held(run);
    let own = proto::ExeId::current().map(|e| e.build_id()).ok();
    let same_build = live
        .as_ref()
        .and_then(|l| l["build_id"].as_str())
        .zip(own.as_deref())
        .map(|(theirs, ours)| theirs == ours);
    json!({
        "db": key,
        "running": live.is_some(),
        "lock_held": lock_held,
        "stopping": live.is_none() && lock_held && server::said_stopping(run),
        "socket": {
            "path": run.socket(),
            "exists": run.socket().exists(),
            "answers": live.is_some(),
            "other_db": other_db,
        },
        "this_build": own,
        "same_build": same_build,
        "blocked": read_json_file::<Value>(&run.blocked()),
        "last": last,
        "live": live,
    })
}

fn status(db: &Path, as_json: bool, out: &Output) -> Result<i32> {
    let (key, run) = run_dir(db)?;
    let report = report(&key, &run);
    if as_json {
        out.print(&report);
    } else {
        for line in describe(&report) {
            outln!("{line}");
        }
    }
    Ok(EXIT_OK)
}

/// `report` in lines of text.
fn describe(report: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let live = &report["live"];
    let last = &report["last"];
    if live.is_object() {
        lines.push(format!(
            "daemon: running, {} (pid {})",
            live["state"].as_str().unwrap_or("?"),
            live["pid"]
        ));
    } else if report["stopping"] == true {
        lines.push(format!("daemon: stopping (pid {})", last["pid"]));
    } else if report["lock_held"] == true {
        lines.push("daemon: holds the lock but does not answer".to_string());
    } else if last.is_object() {
        let how = match (last["state"].as_str(), last["exit_reason"].as_str()) {
            (Some("stopped"), Some(reason)) => format!(
                "stopped at {} ({reason}){}",
                last["stopped_at"].as_str().unwrap_or("?"),
                if last["closed_cleanly"] == false {
                    ", leaving the database for the next open to repair"
                } else {
                    ""
                }
            ),
            (Some(state), _) => format!(
                "gone without stopping: it was {state} (pid {})",
                last["pid"]
            ),
            _ => "gone".to_string(),
        };
        lines.push(format!("daemon: not running; last one {how}"));
    } else {
        lines.push("daemon: not running; none has run for this database".to_string());
    }
    lines.push(format!(
        "database: {}",
        report["db"].as_str().unwrap_or("?")
    ));
    lines.push(format!(
        "socket: {}",
        report["socket"]["path"].as_str().unwrap_or("?")
    ));
    if let Some(reason) = report["blocked"]["reason"].as_str() {
        lines.push(format!("not served: {reason}"));
    }
    if !live.is_object() {
        return lines;
    }
    let build = match report["same_build"].as_bool() {
        Some(true) => "the same as this binary's",
        Some(false) => "not this binary's: commands from this binary are not served",
        None => "unknown",
    };
    lines.push(format!(
        "build: {} ({}), {build}",
        live["build_id"].as_str().unwrap_or("?"),
        live["version"].as_str().unwrap_or("?")
    ));
    lines.push(format!(
        "idle: {}s, stops after {}s more",
        live["idle_secs"], live["idle_left_secs"]
    ));
    if let Some(holders) = live["lease_holders"].as_array().filter(|h| !h.is_empty()) {
        let pids: Vec<String> = holders
            .iter()
            .map(|h| {
                let waiting = if h["granted"] == true {
                    ""
                } else {
                    ", waiting"
                };
                format!("pid {} for {}s{waiting}", h["pid"], h["secs"])
            })
            .collect();
        lines.push(format!("leased to: {}", pids.join("; ")));
    }
    lines.push(format!(
        "in flight: {}, abandoned recalls: {}, panics: {}, reloads: {}{}",
        live["in_flight"],
        live["abandoned_recalls"],
        live["panics"],
        live["reloads"],
        live["last_reload"]
            .as_str()
            .map(|why| format!(" (last: {why})"))
            .unwrap_or_default()
    ));
    let graph = serde_json::from_value(live["graph"].clone()).ok();
    let engines: Vec<&str> = ["vector", "embedder", "graph", "fts", "timeseries"]
        .into_iter()
        .filter(|e| live["engines"][e] == true)
        .collect();
    lines.push(format!(
        "engines: {}; graph backend: {}",
        if engines.is_empty() {
            "none".to_string()
        } else {
            engines.join(", ")
        },
        graph_name(&graph)
    ));
    if let Some(ops) = live["ops"].as_object() {
        for (op, figures) in ops {
            let ms = |v: &Value| {
                v.as_f64()
                    .map_or("-".to_string(), |ms| format!("{ms:.1} ms"))
            };
            lines.push(format!(
                "  {op:<16} served {:<6} not served {:<4} p50 {:<10} p95 {}",
                figures["served"],
                figures["not_served"],
                ms(&figures["p50_ms"]),
                ms(&figures["p95_ms"])
            ));
        }
    }
    lines
}

fn logs(db: &Path, lines: usize) -> Result<i32> {
    let (_, run) = run_dir(db)?;
    let mut text = String::new();
    for path in [run.old_log(), run.log()] {
        if let Ok(bytes) = fs::read(&path) {
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
    }
    let all: Vec<&str> = text.lines().collect();
    for line in &all[all.len().saturating_sub(lines)..] {
        outln!("{line}");
    }
    Ok(EXIT_OK)
}

/// Whether the command line `argv` reads stdin, as the command would:
/// one with a `-` argument (`store <table> -`, `checkpoint -`), or a
/// `checkpoint` given no JSON. Another leaves stdin be.
fn reads_stdin(argv: &[String]) -> bool {
    if argv.iter().any(|a| a == "-") {
        return true;
    }
    #[cfg(feature = "checkpoint")]
    {
        use clap::Parser;
        if let Ok(cli) = crate::Cli::try_parse_from(argv) {
            return matches!(cli.command, crate::Command::Checkpoint { arg: None, .. });
        }
    }
    false
}

fn send(db: &Path, args: Vec<String>) -> Result<i32> {
    let (_, run) = run_dir(db)?;
    let identity = Identity::current()?;
    let argv: Vec<String> = std::iter::once("axil".to_string()).chain(args).collect();
    let stdin = if reads_stdin(&argv) && !cli_io::stdin_is_terminal() {
        Some(cli_io::read_stdin(u64::MAX)?.into_bytes())
    } else {
        None
    };
    let cwd = std::env::current_dir()?;
    let header = identity.header(argv, &cwd, stdin.is_some());
    let outcome = match proto::run(
        run.socket(),
        &header,
        stdin.as_deref(),
        Duration::from_secs(300),
    ) {
        Ok(outcome) => outcome,
        // Not run, as when the daemon declines it.
        Err(e) => {
            errln!("axil: no daemon answers on {}: {e}", run.socket().display());
            return Ok(NOT_SERVED_EXIT);
        }
    };
    match outcome.reply {
        RunReply::Served { exit } => {
            let mut stdout = cli_io::stdout();
            stdout.write_all(&outcome.stdout)?;
            stdout.flush()?;
            errp!("{}", String::from_utf8_lossy(&outcome.stderr));
            Ok(exit)
        }
        other => {
            errln!("{}", serde_json::to_string(&other)?);
            Ok(NOT_SERVED_EXIT)
        }
    }
}
