//! `axil daemon {start, stop, status, logs}`: running a database's daemon by
//! hand. Nothing forwards commands to it on its own yet; `daemon send` sends
//! one command line, to try it.
//!
//! `AXIL_DAEMON` is the kill switch: `off` keeps `daemon start` from
//! starting one. Unset, it is `on`: a daemon runs only when started by hand.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::paths::{write_atomic, RunDir};
use super::proto::{self, Ack, Identity, Op, RunReply};
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
    },
    /// Ask the daemon to stop, and wait until it has closed the database.
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
    /// exiting with its exit code; 125 when the daemon did not run it.
    /// Give the command line after `--`.
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
        DaemonCommand::Run { idle_timeout } => server::run_daemon(&db, idle_timeout),
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

fn start(db: &Path, idle_timeout: Duration, wait: Duration, out: &Output) -> Result<i32> {
    anyhow::ensure!(
        mode() != Mode::Off,
        "the daemon is turned off (AXIL_DAEMON=off)"
    );
    let (key, run) = run_dir(db)?;
    run.create()
        .with_context(|| format!("can't use {}", run.dir().display()))?;
    let own = proto::ExeId::current()?.build_id();
    if let Ok(hello) = proto::hello(run.socket(), Duration::from_secs(1)) {
        if hello.state != "stopping" {
            out.print(&json!({
                "started": false,
                "running": true,
                "pid": hello.pid,
                "state": hello.state,
                "build_id": hello.build_id,
                "same_build": hello.build_id == own,
            }));
            return Ok(EXIT_OK);
        }
    }

    // Started by hand: whatever kept the last one from serving is tried
    // again.
    let _ = fs::remove_file(run.blocked());
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(run.log())
        .with_context(|| format!("can't open {}", run.log().display()))?;
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--db")
        .arg(&key)
        .args(["daemon", "run", "--idle-timeout"])
        .arg(format!("{}s", idle_timeout.as_secs()))
        .current_dir("/")
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
    let mut child = cmd.spawn().context("can't start the daemon")?;
    let stamp = json!({"pid": child.id(), "at": chrono::Utc::now().to_rfc3339()});
    let _ = write_atomic(&run.spawn_stamp(), stamp.to_string().as_bytes());

    let deadline = Instant::now() + wait;
    let mut exited = None;
    loop {
        if let Ok(hello) = proto::hello(run.socket(), Duration::from_secs(1)) {
            if hello.state == "ready" {
                out.print(&json!({
                    "started": hello.pid == child.id(),
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
        if exited.is_none() {
            exited = child.try_wait()?;
        }
        // Exited without a daemon to wait for: it would not serve, or lost
        // to one that is gone again.
        if exited.is_some() {
            if let Some(blocked) = read_json_file::<Value>(&run.blocked()) {
                let reason = blocked["reason"].as_str().unwrap_or("unknown").to_string();
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

fn stop(db: &Path, wait: Duration, out: &Output) -> Result<i32> {
    let (_, run) = run_dir(db)?;
    let pid = read_json_file::<DaemonInfo>(&run.info()).map(|info| info.pid);
    let op = Op::Shutdown {
        reason: "axil daemon stop".into(),
    };
    if proto::call::<Ack>(run.socket(), &op, Duration::from_secs(5)).is_err() {
        if server::lock_held(&run) {
            out.print(&json!({
                "stopped": false,
                "running": true,
                "pid": pid,
                "error": "a daemon holds the lock but does not answer",
            }));
            return Ok(EXIT_ERROR);
        }
        out.print(&json!({"stopped": false, "running": false}));
        return Ok(EXIT_OK);
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
    out.print(&json!({"stopped": true, "pid": pid}));
    Ok(EXIT_OK)
}

/// The status report: the record the daemon left, the lock and socket
/// probed now, and the live figures when it answers.
fn report(key: &Path, run: &RunDir) -> Value {
    let last = read_json_file::<DaemonInfo>(&run.info());
    let live = proto::call::<Value>(run.socket(), &Op::Status, Duration::from_secs(2)).ok();
    let own = proto::ExeId::current().map(|e| e.build_id()).ok();
    let same_build = live
        .as_ref()
        .and_then(|l| l["build_id"].as_str())
        .zip(own.as_deref())
        .map(|(theirs, ours)| theirs == ours);
    json!({
        "db": key,
        "running": live.is_some(),
        "lock_held": server::lock_held(run),
        "socket": {
            "path": run.socket(),
            "exists": run.socket().exists(),
            "answers": live.is_some(),
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
    } else if report["lock_held"] == true {
        lines.push("daemon: holds the lock but does not answer".to_string());
    } else if last.is_object() {
        let how = match (last["state"].as_str(), last["exit_reason"].as_str()) {
            (Some("stopped"), Some(reason)) => format!(
                "stopped at {} ({reason})",
                last["stopped_at"].as_str().unwrap_or("?")
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

fn send(db: &Path, args: Vec<String>) -> Result<i32> {
    let (_, run) = run_dir(db)?;
    let identity = Identity::current()?;
    let argv: Vec<String> = std::iter::once("axil".to_string()).chain(args).collect();
    // Stdin is read only for a command that reads it, `-` or a checkpoint
    // without its JSON, as the command would: another would leave it be.
    let reads_stdin =
        argv.iter().any(|a| a == "-") || super::state::op_label(&argv) == "checkpoint";
    let stdin = if reads_stdin && !cli_io::stdin_is_terminal() {
        Some(cli_io::read_stdin(u64::MAX)?.into_bytes())
    } else {
        None
    };
    let cwd = std::env::current_dir()?;
    let header = identity.header(argv, &cwd, stdin.is_some());
    let outcome = proto::run(
        run.socket(),
        &header,
        stdin.as_deref(),
        Duration::from_secs(300),
    )
    .with_context(|| format!("no daemon answers on {}", run.socket().display()))?;
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
