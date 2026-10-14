//! The daemon process: it keeps one database open and runs the hot commands
//! clients send over its socket, until it is idle, told to stop, signalled,
//! or its own binary changes.
//!
//! [`run_daemon`] is `axil daemon run`. In order:
//!
//! 1. Take `run/daemon.lock`, retrying for 5 s to outlast a daemon that is
//!    retiring. When a daemon already answers, leave it to serve and exit 0.
//! 2. Point stdout and stderr at `run/daemon.log`, remove from the
//!    environment what each request carries itself ([`STRIPPED_ENV`]), and
//!    handle SIGTERM, SIGINT and SIGHUP by shutting down.
//! 3. Replace any stale socket and bind it 0600. Clients connecting from
//!    here on are told the daemon is starting.
//! 4. Open the database ([`WarmState::open`]), retrying for 10 s while
//!    another process holds its lock. A database that can't be served (its
//!    open printed, an engine is missing, or it is read-only) is written to
//!    `run/daemon.blocked`, and the daemon exits 1.
//! 5. Serve, and warm the embedder and the graph up in the background.
//!
//! The process-wide setup a command runs under is the one-shot binary's:
//! `daemon run` goes through the same `process_setup`, so the vector index
//! answers with the same exact scan up to the same size, and a forwarded
//! recall ranks as the one-shot recall it stands in for. The approximate
//! graph a long-lived process could amortize is left unused.
//!
//! While serving, a request runs on its own thread, at most
//! [`MAX_IN_FLIGHT`] at a time (more are answered `overloaded`). Writes run
//! one at a time under the warm state's writer lock. Before a request runs,
//! the daemon checks (at most every 250 ms) whether the files behind the
//! open handle changed without it ([`FileFp`]) or the graph failed to load;
//! either reopens the database once the requests on the old handle finish.
//! A tick each second does the same check, and stops the daemon when it has
//! been idle for its idle timeout, when it was signalled, or when its own
//! executable changed on disk.
//!
//! Shutting down unlinks the socket first, so new clients fall back at
//! once, answers queued connections `stopping`, lets running requests
//! finish, and closes the database cleanly before the lock is let go. After
//! a request panicked the database is never closed (see
//! [`WarmState::poisoned`]): the daemon exits 1 without dropping it.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axil_core::GraphState;
use serde_json::{json, Value};

use super::exec::{self, ReopenError, WarmState};
use super::paths::{write_atomic, RunDir};
use super::proto::{
    self, read_frame, read_json, write_frame, write_json, Ack, ExeId, Hello, Identity, Op,
    RunHeader, RunReply, DAEMON_PROTO, MAX_FRAME, STRIPPED_ENV, VERSION,
};
use super::state::{op_label, DaemonInfo, FileFp, Phase, PhaseCell, Stats};
use crate::cli_io::{self, ReqCtx};

/// Run requests running at once; more are answered `overloaded`.
pub const MAX_IN_FLIGHT: usize = 16;

/// Connections handled at once, whatever they ask.
const MAX_CONNECTIONS: usize = 64;

/// How long a connection may take to send its request.
const READ_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a reply may take to reach a client that stopped reading.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a freshness check stands.
const FRESH_FOR: Duration = Duration::from_millis(250);

/// How long a reopen waits for the requests on the old handle.
const REOPEN_WAIT: Duration = Duration::from_secs(10);

/// How long shutting down waits for running requests.
const DRAIN_WAIT: Duration = Duration::from_secs(10);

/// How long shutting down waits for the last references to the handle,
/// after the requests are done: recall workers a deadline left running.
const RELEASE_WAIT: Duration = Duration::from_secs(3);

/// How often the tick runs.
const TICK: Duration = Duration::from_secs(1);

/// The size at which the log moves to `daemon.log.1`.
const LOG_ROTATE_BYTES: u64 = 1024 * 1024;

/// How many times in a row a failed graph is reloaded before the daemon
/// gives up on the database.
const GRAPH_RELOADS: u32 = 1;

/// How a daemon runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Stop after this long without a run request.
    pub idle_timeout: Duration,
    /// How long to retry an open that another process's lock blocks.
    pub open_wait: Duration,
    /// This is the daemon process: bind under umask 077 and rotate the log
    /// at fd 1 and 2. Both are process-wide.
    pub process: bool,
}

/// Why the daemon stops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Idle for its idle timeout.
    Idle,
    /// A client sent `shutdown`.
    Requested(String),
    /// SIGTERM, SIGINT or SIGHUP.
    Signal(i32),
    /// Its executable changed on disk.
    ExeChanged,
    /// The database can't be served.
    Blocked(String),
    /// A request panicked.
    Panicked,
}

impl StopReason {
    fn describe(&self) -> String {
        match self {
            StopReason::Idle => "idle".to_string(),
            StopReason::Requested(why) if why.is_empty() => "asked to stop".to_string(),
            StopReason::Requested(why) => format!("asked to stop: {why}"),
            StopReason::Signal(sig) => format!("signal {sig}"),
            StopReason::ExeChanged => "its executable changed".to_string(),
            StopReason::Blocked(why) => format!("blocked: {why}"),
            StopReason::Panicked => "a request panicked".to_string(),
        }
    }

    fn exit_code(&self) -> i32 {
        match self {
            StopReason::Blocked(_) | StopReason::Panicked => crate::EXIT_ERROR,
            _ => crate::EXIT_OK,
        }
    }
}

/// How a shutdown ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Finish {
    /// The database is closed; exit with this code.
    Exit(i32),
    /// A request panicked: exit 1 without dropping anything (see
    /// [`WarmState::poisoned`]).
    Poisoned,
}

/// One line in the daemon's log.
fn log(message: impl std::fmt::Display) {
    errln!(
        "{} axil daemon[{}]: {message}",
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
        std::process::id()
    );
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The open handle's view of when its files last matched it.
struct Freshness {
    baseline: FileFp,
    checked: Instant,
}

/// A running daemon.
pub struct Server {
    key: PathBuf,
    run: RunDir,
    opts: Options,
    identity: Identity,
    exe: Option<ExeId>,
    phase: PhaseCell,
    state: OnceLock<WarmState>,
    info: Mutex<DaemonInfo>,
    stats: Stats,
    stop: Mutex<Option<StopReason>>,
    stopped: Condvar,
    in_flight: AtomicUsize,
    connections: AtomicUsize,
    last_active: Mutex<Instant>,
    fresh: Mutex<Option<Freshness>>,
    graph_reloads: AtomicU32,
    fail_graph: AtomicBool,
    accept: Mutex<Option<JoinHandle<()>>>,
}

/// A run request counted against [`MAX_IN_FLIGHT`] until dropped.
struct Admitted<'a>(&'a AtomicUsize);

impl<'a> Admitted<'a> {
    fn new(count: &'a AtomicUsize, cap: usize) -> Option<Self> {
        if count.fetch_add(1, Ordering::AcqRel) >= cap {
            count.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Self(count))
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Server {
    /// Bind the socket, open the database at `key` and start serving it.
    /// Fails, with the socket gone and the reason written to
    /// `daemon.blocked`, when the database can't be served.
    pub fn start(key: PathBuf, run: RunDir, opts: Options) -> Result<Arc<Server>, String> {
        let identity = Identity::current().map_err(|e| format!("can't identify this binary: {e}"))?;
        let exe = ExeId::current().ok();
        let info = DaemonInfo {
            proto: DAEMON_PROTO,
            pid: std::process::id(),
            state: Phase::Starting.name().to_string(),
            version: VERSION.to_string(),
            build_id: identity.build_id.clone(),
            exe: exe
                .as_ref()
                .map(|e| e.path.display().to_string())
                .unwrap_or_default(),
            features: identity.features.clone(),
            db: key.display().to_string(),
            socket: run.socket().display().to_string(),
            started_at: now_rfc3339(),
            idle_timeout_secs: opts.idle_timeout.as_secs(),
            ..DaemonInfo::default()
        };
        let server = Arc::new(Server {
            key,
            run,
            opts,
            identity,
            exe,
            phase: PhaseCell::new(Phase::Starting),
            state: OnceLock::new(),
            info: Mutex::new(info),
            stats: Stats::default(),
            stop: Mutex::new(None),
            stopped: Condvar::new(),
            in_flight: AtomicUsize::new(0),
            connections: AtomicUsize::new(0),
            last_active: Mutex::new(Instant::now()),
            fresh: Mutex::new(None),
            graph_reloads: AtomicU32::new(0),
            fail_graph: AtomicBool::new(false),
            accept: Mutex::new(None),
        });
        server.write_info();

        let listener = server
            .bind()
            .map_err(|e| format!("can't listen on {}: {e}", server.run.socket().display()))?;
        let accepting = server.clone();
        let accept = thread::Builder::new()
            .name("axil-accept".into())
            .spawn(move || accepting.accept_loop(listener))
            .map_err(|e| format!("can't start the accept thread: {e}"))?;
        *server.accept.lock().unwrap_or_else(PoisonError::into_inner) = Some(accept);
        log(format!(
            "listening on {} for {}",
            server.run.socket().display(),
            server.key.display()
        ));

        if let Err(why) = server.open() {
            server.block(&why);
            server.shutdown(&StopReason::Blocked(why.clone()));
            return Err(why);
        }
        Ok(server)
    }

    /// Open the database, retrying while another process holds its lock.
    fn open(self: &Arc<Self>) -> Result<(), String> {
        let deadline = Instant::now() + self.opts.open_wait;
        let state = loop {
            if let Some(reason) = self.stop_reason() {
                return Err(format!("stopped while opening the database: {}", reason.describe()));
            }
            let fp = FileFp::read(&self.key);
            match WarmState::open(&self.key) {
                Ok(state) => {
                    self.set_fresh(fp);
                    break state;
                }
                Err(e) if crate::is_lock_contention(&e) && Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(format!("{e:#}")),
            }
        };
        if let Some(why) = unservable(&state) {
            return Err(why);
        }
        let _ = self.state.set(state);
        self.phase.advance(Phase::Starting, Phase::Ready);
        {
            let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
            info.ready_at = Some(now_rfc3339());
        }
        self.write_info();
        log("ready");
        self.prewarm();
        Ok(())
    }

    /// Serve until something stops the daemon, running the tick each
    /// second, and return why it stops. The caller then shuts it down.
    pub fn serve(self: &Arc<Self>) -> StopReason {
        loop {
            {
                let stop = self.stop.lock().unwrap_or_else(PoisonError::into_inner);
                let (mut stop, _) = self
                    .stopped
                    .wait_timeout_while(stop, TICK, |s| s.is_none())
                    .unwrap_or_else(PoisonError::into_inner);
                if let Some(reason) = stop.take() {
                    return reason;
                }
            }
            if let Some(reason) = self.tick() {
                return reason;
            }
        }
    }

    /// The once-a-second checks.
    fn tick(self: &Arc<Self>) -> Option<StopReason> {
        if let Some(sig) = take_signal() {
            return Some(StopReason::Signal(sig));
        }
        if self.in_flight.load(Ordering::Acquire) == 0
            && self.idle_for() >= self.opts.idle_timeout
        {
            return Some(StopReason::Idle);
        }
        if let Some(exe) = &self.exe {
            if ExeId::of(&exe.path).ok().as_ref() != Some(exe) {
                return Some(StopReason::ExeChanged);
            }
        }
        if self.opts.process {
            rotate_log(&self.run);
        }
        if self.phase.get() == Phase::Ready {
            if let Some(why) = self.stale() {
                let server = self.clone();
                let _ = thread::Builder::new()
                    .name("axil-reload".into())
                    .spawn(move || server.reload(&why));
            }
        }
        None
    }

    /// Stop serving and close the database, as [`Finish`] says.
    pub fn shutdown(self: &Arc<Self>, reason: &StopReason) -> Finish {
        self.phase.set(Phase::Stopping);
        log(format!("stopping: {}", reason.describe()));
        // New clients find no socket and fall back at once.
        let _ = fs::remove_file(self.run.socket());
        if let Some(accept) = self
            .accept
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            let _ = accept.join();
        }
        let deadline = Instant::now() + DRAIN_WAIT;
        while self.in_flight.load(Ordering::Acquire) > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let left = self.in_flight.load(Ordering::Acquire);
        if left > 0 {
            log(format!("{left} requests still running after {DRAIN_WAIT:?}"));
        }

        let poisoned = self.state.get().is_some_and(WarmState::poisoned);
        if !poisoned {
            if let Some(state) = self.state.get() {
                match state.release(RELEASE_WAIT) {
                    Ok(()) => log("closed the database"),
                    Err(e) => log(format!("did not close the database: {e}")),
                }
            }
        }
        {
            let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
            info.stopped_at = Some(now_rfc3339());
            info.exit_reason = Some(reason.describe());
        }
        self.write_info_as("stopped");
        if poisoned {
            Finish::Poisoned
        } else {
            Finish::Exit(reason.exit_code())
        }
    }

    /// Ask the daemon to stop; the first reason given stands.
    pub fn request_stop(&self, reason: StopReason) {
        let mut stop = self.stop.lock().unwrap_or_else(PoisonError::into_inner);
        if stop.is_none() && self.phase.get() != Phase::Stopping {
            *stop = Some(reason);
        }
        self.stopped.notify_all();
    }

    fn stop_reason(&self) -> Option<StopReason> {
        self.stop
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record why the database can't be served, for `daemon status` and
    /// `daemon start`.
    fn block(&self, why: &str) {
        log(format!("not serving {}: {why}", self.key.display()));
        let record = json!({
            "reason": why,
            "pid": std::process::id(),
            "at": now_rfc3339(),
        });
        if let Err(e) = write_atomic(&self.run.blocked(), record.to_string().as_bytes()) {
            log(format!("can't write {}: {e}", self.run.blocked().display()));
        }
    }

    fn stop_blocked(&self, why: String) {
        self.block(&why);
        self.request_stop(StopReason::Blocked(why));
    }

    fn write_info(&self) {
        self.write_info_as(self.phase.get().name());
    }

    fn write_info_as(&self, state: &str) {
        let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
        info.state = state.to_string();
        let bytes = serde_json::to_vec_pretty(&*info).unwrap_or_default();
        if let Err(e) = write_atomic(&self.run.info(), &bytes) {
            log(format!("can't write {}: {e}", self.run.info().display()));
        }
    }

    fn idle_for(&self) -> Duration {
        self.last_active
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .elapsed()
    }

    fn touch(&self) {
        *self.last_active.lock().unwrap_or_else(PoisonError::into_inner) = Instant::now();
    }

    fn set_fresh(&self, baseline: FileFp) {
        *self.fresh.lock().unwrap_or_else(PoisonError::into_inner) = Some(Freshness {
            baseline,
            checked: Instant::now(),
        });
    }

    /// Bind the socket, replacing a stale one: the lock says no daemon is
    /// using it.
    fn bind(&self) -> io::Result<UnixListener> {
        let path = self.run.socket();
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        // SAFETY: umask only swaps the process's file mode mask.
        let old = self.opts.process.then(|| unsafe { libc::umask(0o077) });
        let bound = UnixListener::bind(path);
        if let Some(old) = old {
            // SAFETY: as above.
            unsafe { libc::umask(old) };
        }
        let listener = bound?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(listener)
    }

    fn accept_loop(self: Arc<Self>, listener: UnixListener) {
        let fd = listener.as_raw_fd();
        while self.phase.get() != Phase::Stopping {
            if !readable(fd, 200) {
                continue;
            }
            match listener.accept() {
                Ok((stream, _)) => self.spawn_connection(stream),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    log(format!("accept failed: {e}"));
                    thread::sleep(Duration::from_millis(50));
                }
            }
        }
        // Whoever connected before the socket went is answered, not cut off.
        if listener.set_nonblocking(true).is_ok() {
            while let Ok((stream, _)) = listener.accept() {
                if stream.set_nonblocking(false).is_ok() {
                    self.spawn_connection(stream);
                }
            }
        }
    }

    fn spawn_connection(self: &Arc<Self>, mut stream: UnixStream) {
        if self.connections.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
            self.connections.fetch_sub(1, Ordering::AcqRel);
            let _ = stream.set_write_timeout(Some(Duration::from_millis(100)));
            let _ = write_json(&mut stream, &RunReply::Overloaded);
            return;
        }
        let server = self.clone();
        let spawned = thread::Builder::new()
            .name("axil-conn".into())
            .spawn(move || {
                struct Done<'a>(&'a AtomicUsize);
                impl Drop for Done<'_> {
                    fn drop(&mut self) {
                        self.0.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                let _done = Done(&server.connections);
                server.handle(stream);
            });
        if spawned.is_err() {
            self.connections.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// One connection: one request and its answer.
    fn handle(self: &Arc<Self>, mut stream: UnixStream) {
        let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
        let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
        let op: Op = match read_json(&mut stream) {
            Ok(op) => op,
            Err(e) => {
                if e.kind() == io::ErrorKind::InvalidData {
                    let message = format!("bad request: {e}");
                    let _ = write_json(&mut stream, &RunReply::Error { message });
                }
                return;
            }
        };
        let _ = match op {
            Op::Hello => write_json(&mut stream, &self.hello()),
            Op::Shutdown { reason } => {
                let ack = Ack {
                    op: "shutdown".into(),
                    ok: true,
                };
                let sent = write_json(&mut stream, &ack);
                self.request_stop(StopReason::Requested(reason));
                sent
            }
            Op::Status => write_json(&mut stream, &self.status()),
            Op::Run(header) => {
                self.handle_run(&mut stream, header);
                Ok(())
            }
            #[cfg(debug_assertions)]
            Op::DebugFailGraph => {
                self.fail_graph.store(true, Ordering::Release);
                let ack = Ack {
                    op: "debug_fail_graph".into(),
                    ok: true,
                };
                write_json(&mut stream, &ack)
            }
        };
    }

    fn handle_run(self: &Arc<Self>, stream: &mut UnixStream, header: RunHeader) {
        let stdin = if header.has_stdin {
            match read_frame(stream) {
                Ok(bytes) => Some(bytes),
                Err(e) => {
                    let message = format!("could not read stdin: {e}");
                    let _ = write_json(stream, &RunReply::Error { message });
                    return;
                }
            }
        } else {
            None
        };
        let (reply, output) = self.run(header, stdin);
        if write_json(stream, &reply).is_err() {
            return;
        }
        if let Some(response) = output {
            let _ = write_frame(stream, &response.stdout)
                .and_then(|()| write_frame(stream, &response.stderr));
        }
    }

    /// Run one request, or say why not.
    fn run(
        self: &Arc<Self>,
        header: RunHeader,
        stdin: Option<Vec<u8>>,
    ) -> (RunReply, Option<exec::Response>) {
        let refused = |reply: RunReply| {
            self.stats.refused.fetch_add(1, Ordering::AcqRel);
            (reply, None)
        };
        match self.phase.get() {
            Phase::Starting => return refused(RunReply::Starting),
            Phase::Stopping => return refused(RunReply::Stopping),
            Phase::Ready | Phase::Reloading => {}
        }
        if let Some(reason) = self.identity.mismatch(&header) {
            return refused(RunReply::Incompatible {
                reason: reason.to_string(),
            });
        }
        let Some(_admitted) = Admitted::new(&self.in_flight, MAX_IN_FLIGHT) else {
            return refused(RunReply::Overloaded);
        };
        self.touch();
        if header.deadline_ms.is_some_and(|deadline| epoch_ms() >= deadline) {
            return refused(RunReply::Expired);
        }
        let Some(state) = self.state.get() else {
            return refused(RunReply::Starting);
        };
        if let Some(why) = self.stale() {
            self.reload(&why);
        }

        let op = op_label(&header.argv);
        let started = Instant::now();
        let result = exec::execute_captured(header.to_request(stdin), state);
        let took = started.elapsed();
        self.touch();
        if state.poisoned() {
            self.request_stop(StopReason::Panicked);
        }
        match result {
            Ok(response) if response.stdout.len() > MAX_FRAME || response.stderr.len() > MAX_FRAME => {
                self.stats.not_served(op);
                let reply = RunReply::NotServed {
                    reason: "too_large".into(),
                    detail: "the output is over the 32 MiB frame limit".into(),
                };
                (reply, None)
            }
            Ok(response) => {
                self.stats.served(op, took);
                (RunReply::Served { exit: response.exit }, Some(response))
            }
            Err(not_served) => {
                self.stats.not_served(op);
                (RunReply::not_served(not_served), None)
            }
        }
    }

    /// Why the open handle is stale, if it is: its files changed under it,
    /// or the graph failed to load. The file check stands for 250 ms.
    fn stale(&self) -> Option<String> {
        if self.graph_failed() {
            return Some("the graph failed to load".to_string());
        }
        let mut fresh = self.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        let fresh = fresh.as_mut()?;
        if fresh.checked.elapsed() < FRESH_FOR {
            return None;
        }
        fresh.checked = Instant::now();
        fresh.baseline.changed(&FileFp::read(&self.key))
    }

    fn graph_failed(&self) -> bool {
        if self.fail_graph.load(Ordering::Acquire) {
            return true;
        }
        let Some(warm) = self.state.get().and_then(WarmState::handle) else {
            return false;
        };
        matches!(warm.db().graph_load_state(), Some(GraphState::Failed(_)))
    }

    /// Reopen the database because `why`. One reload runs at a time: a
    /// caller that finds one running, or the daemon stopping, returns at
    /// once. Requests are not served while it runs. A database that doesn't
    /// open again stops the daemon.
    fn reload(self: &Arc<Self>, why: &str) {
        let Some(state) = self.state.get() else {
            return;
        };
        if why.starts_with("the graph") {
            let tries = self.graph_reloads.fetch_add(1, Ordering::AcqRel);
            if tries >= GRAPH_RELOADS {
                self.stop_blocked(format!("{why} again after a reload"));
                return;
            }
        }
        if !self.phase.advance(Phase::Ready, Phase::Reloading) {
            return;
        }
        log(format!("reloading: {why}"));
        let deadline = Instant::now() + self.opts.open_wait;
        let outcome = loop {
            let fp = FileFp::read(&self.key);
            match state.reopen(REOPEN_WAIT) {
                Ok(()) => break Ok(fp),
                Err(ReopenError::Running) | Err(ReopenError::InUse) => break Err(None),
                Err(ReopenError::Poisoned) => break Err(Some("a request panicked".to_string())),
                Err(ReopenError::Open(e))
                    if crate::is_lock_contention(&e) && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(ReopenError::Open(e)) => break Err(Some(format!("{e:#}"))),
            }
        };
        match outcome {
            Ok(fp) => {
                if let Some(why) = unservable(state) {
                    self.stop_blocked(why);
                    return;
                }
                self.set_fresh(fp);
                self.fail_graph.store(false, Ordering::Release);
                self.stats.reloaded(why);
                self.phase.advance(Phase::Reloading, Phase::Ready);
                log("reloaded");
                self.prewarm_on(state);
            }
            Err(None) => {
                log("reload postponed: requests still hold the database");
                self.phase.advance(Phase::Reloading, Phase::Ready);
            }
            Err(Some(failed)) => self.stop_blocked(format!("reopening after {why}: {failed}")),
        }
    }

    fn prewarm(self: &Arc<Self>) {
        if let Some(state) = self.state.get() {
            self.prewarm_on(state);
        }
    }

    /// Load the embedding model and the graph's backend in the background.
    /// What they print is logged, and the policy for prints at open holds:
    /// a line a command would print, once per process, is a database that
    /// is not served. A graph left in memory prints that warning in each
    /// one-shot command; a failed graph is reloaded by the next check.
    fn prewarm_on(self: &Arc<Self>, state: &WarmState) {
        let Some(warm) = state.handle() else {
            return;
        };
        let server = Arc::clone(self);
        let _ = thread::Builder::new()
            .name("axil-prewarm".into())
            .spawn(move || {
                let started = Instant::now();
                let (out, err, report) =
                    cli_io::capture(ReqCtx::default(), || warm.db().prewarm());
                drop(warm);
                let printed = String::from_utf8_lossy(&[out, err].concat()).into_owned();
                log(format!(
                    "warmed up in {} ms: embedder {}, graph {}",
                    started.elapsed().as_millis(),
                    match &report.embedder {
                        None => "none".to_string(),
                        Some(Ok(())) => "loaded".to_string(),
                        Some(Err(e)) => format!("failed ({e})"),
                    },
                    super::state::graph_name(&report.graph),
                ));
                for line in printed.lines() {
                    log(format!("warm-up printed: {line}"));
                }
                let graph = report.graph.clone();
                server.stats.set_prewarm(report);
                match graph {
                    Some(GraphState::Failed(_)) => {}
                    Some(GraphState::Memory) => server.stop_blocked(
                        "the graph's adjacency tables can't be used, so every command would \
                         print that it loads every edge into memory"
                            .to_string(),
                    ),
                    _ if !printed.trim().is_empty() => {
                        server.stop_blocked(format!("warming up printed:\n{}", printed.trim_end()))
                    }
                    _ => server.graph_reloads.store(0, Ordering::Release),
                }
            });
    }

    fn hello(&self) -> Hello {
        Hello {
            op: "hello".into(),
            proto: DAEMON_PROTO,
            build_id: self.identity.build_id.clone(),
            version: VERSION.into(),
            pid: std::process::id(),
            state: self.phase.get().name().into(),
        }
    }

    /// What `daemon status` shows while the daemon runs.
    fn status(&self) -> Value {
        let info = self
            .info
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let warm = self.state.get().and_then(WarmState::handle);
        let engines = warm.as_ref().map(|warm| {
            let db = warm.db();
            json!({
                "vector": db.has_vector_index(),
                "embedder": db.has_embedder(),
                "graph": db.has_graph_index(),
                "fts": db.has_fts_index(),
                "timeseries": db.has_timeseries_index(),
                "read_only": db.storage().is_read_only(),
                "degraded": db.degraded_engines().iter().map(|d| d.summary()).collect::<Vec<_>>(),
            })
        });
        let graph = warm.as_ref().and_then(|warm| warm.db().graph_load_state());
        drop(warm);
        let idle = self.idle_for();
        let busy = self.in_flight.load(Ordering::Acquire) > 0;
        json!({
            "op": "status",
            "state": self.phase.get().name(),
            "pid": std::process::id(),
            "proto": DAEMON_PROTO,
            "version": VERSION,
            "build_id": self.identity.build_id,
            "exe": info.exe,
            "features": self.identity.features,
            "db": info.db,
            "socket": info.socket,
            "started_at": info.started_at,
            "ready_at": info.ready_at,
            "idle_timeout_secs": self.opts.idle_timeout.as_secs(),
            "idle_secs": if busy { 0 } else { idle.as_secs() },
            "idle_left_secs": if busy {
                self.opts.idle_timeout.as_secs()
            } else {
                self.opts.idle_timeout.saturating_sub(idle).as_secs()
            },
            "in_flight": self.in_flight.load(Ordering::Acquire),
            "lease_holder": Value::Null,
            "abandoned_recalls": cli_io::RECALL_SLOTS.live(),
            "panics": u64::from(self.state.get().is_some_and(WarmState::poisoned)),
            "reloads": self.stats.reloads.load(Ordering::Acquire),
            "last_reload": self.stats.last_reload(),
            "refused": self.stats.refused.load(Ordering::Acquire),
            "ops": self.stats.ops(),
            "engines": engines,
            "graph": graph,
            "prewarm": self.stats.prewarm(),
        })
    }
}

/// Why `state`'s database must not be served: an engine whose store exists
/// did not attach, or the handle is read-only.
fn unservable(state: &WarmState) -> Option<String> {
    let warm = state.handle()?;
    let db = warm.db();
    if let Some(degraded) = db.degraded_engines().first() {
        return Some(degraded.summary());
    }
    if db.storage().is_read_only() {
        return Some("the database opened read-only".to_string());
    }
    None
}

/// Whether `fd` has a connection waiting, within `timeout_ms`.
fn readable(fd: i32, timeout_ms: i32) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd, for the duration of the call.
    let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
    n > 0 && pfd.revents & libc::POLLIN != 0
}

/// The last of SIGTERM, SIGINT and SIGHUP received, until taken.
static SIGNAL: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(sig: libc::c_int) {
    SIGNAL.store(sig, Ordering::Relaxed);
}

fn take_signal() -> Option<i32> {
    match SIGNAL.swap(0, Ordering::Relaxed) {
        0 => None,
        sig => Some(sig),
    }
}

/// Shut down on SIGTERM, SIGINT and SIGHUP: the handler only records the
/// signal, and the tick acts on it.
fn handle_signals() {
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: the handler only stores to an atomic, which is
        // async-signal-safe, and the sigaction struct is fully initialized.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as libc::sighandler_t;
            libc::sigemptyset(&mut action.sa_mask);
            action.sa_flags = libc::SA_RESTART;
            libc::sigaction(sig, &action, std::ptr::null_mut());
        }
    }
}

/// Point stdout and stderr at `log`, appending.
fn redirect_output(log: &Path) -> io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(log)?;
    let null = File::open("/dev/null")?;
    // SAFETY: dup2 onto the standard descriptors, from files open here.
    unsafe {
        if libc::dup2(file.as_raw_fd(), 1) < 0
            || libc::dup2(file.as_raw_fd(), 2) < 0
            || libc::dup2(null.as_raw_fd(), 0) < 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Move a log over [`LOG_ROTATE_BYTES`] to `daemon.log.1` and start a new
/// one.
fn rotate_log(run: &RunDir) {
    let big = fs::metadata(run.log()).is_ok_and(|m| m.len() > LOG_ROTATE_BYTES);
    if big && fs::rename(run.log(), run.old_log()).is_ok() {
        let _ = redirect_output(&run.log());
        log("rotated the log");
    }
}

/// Take the daemon lock, or find that another daemon has it. `None` means
/// another daemon answers (or still held the lock after 5 s): this one
/// exits.
fn elect(run: &RunDir) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(run.lock())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(Some(file)),
            Err(fs::TryLockError::WouldBlock) => {
                // One that answers is serving; one that doesn't may be
                // retiring, so wait for it to let go.
                let answers = proto::hello(run.socket(), Duration::from_millis(500))
                    .is_ok_and(|hello| hello.state != Phase::Stopping.name());
                if answers || Instant::now() >= deadline {
                    return Ok(None);
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(fs::TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// Whether a daemon holds `run`'s lock, probed without waiting.
pub fn lock_held(run: &RunDir) -> bool {
    let Ok(file) = OpenOptions::new().read(true).write(true).open(run.lock()) else {
        return false;
    };
    matches!(file.try_lock(), Err(fs::TryLockError::WouldBlock))
}

/// `axil daemon run`: be the daemon for the database at `db` until it
/// stops; see the module docs. Returns the exit code.
pub fn run_daemon(db: &Path, idle_timeout: Duration) -> anyhow::Result<i32> {
    let key = cli_io::db_key(db)?;
    anyhow::ensure!(key.is_file(), "no database at {}", key.display());
    let run = RunDir::of(&key);
    run.create()?;
    let Some(lock) = elect(&run)? else {
        return Ok(crate::EXIT_OK);
    };
    redirect_output(&run.log())?;
    for name in STRIPPED_ENV {
        std::env::remove_var(name);
    }
    handle_signals();

    let opts = Options {
        idle_timeout,
        open_wait: Duration::from_secs(10),
        process: true,
    };
    let server = match Server::start(key, run, opts) {
        Ok(server) => server,
        Err(_) => return Ok(crate::EXIT_ERROR),
    };
    let reason = server.serve();
    match server.shutdown(&reason) {
        Finish::Exit(code) => {
            drop(server);
            drop(lock);
            Ok(code)
        }
        Finish::Poisoned => {
            std::mem::forget(server);
            std::process::exit(crate::EXIT_ERROR)
        }
    }
}
