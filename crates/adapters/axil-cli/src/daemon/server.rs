//! The daemon process: it keeps one database open and runs the hot commands
//! clients send over its socket, until it is idle, told to stop, signalled,
//! or its own binary changes.
//!
//! [`run_daemon`] is `axil daemon run`. In order:
//!
//! 1. Take the run directory's `daemon.lock` (see [`RunDir`]), retrying for
//!    5 s to outlast a daemon that is retiring, and for 30 s while one says
//!    it is stopping. When a daemon already answers, leave it to serve and
//!    exit 0.
//! 2. Point stdout and stderr at `daemon.log`, remove from the environment
//!    what each request carries itself ([`STRIPPED_ENV`]), and handle
//!    SIGTERM, SIGINT and SIGHUP by shutting down.
//! 3. Replace any stale socket and bind it 0600. Clients connecting from
//!    here on are told the daemon is starting. A connection from another
//!    user is dropped.
//! 4. Open the database ([`WarmState::open`]), retrying for 10 s while
//!    another process holds its lock. A database that can't be served (its
//!    open printed, an engine is missing, or it is read-only) is written to
//!    `daemon.blocked`, and the daemon exits 1. Once it serves, a
//!    `daemon.blocked` left from before is removed.
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
//! the daemon checks (at most every 250 ms, and again for a write once it
//! holds the writer lock) whether the files behind the open handle changed
//! without it ([`FileFp`]) or the graph failed to load; either reopens the
//! database, on a thread of its own, once the requests on the old handle
//! finish. Until a reopen succeeds, no request runs on the old handle: each
//! is answered `unavailable`, and the client asks again, so a write is never
//! made to a file that is no longer the database. A store written in place by another process (a backup copied
//! over it) can't be reopened: the daemon stops without closing it, since
//! closing would write the handle's stale view over the copy. A tick each
//! second does the same checks, and stops the daemon when it has been idle
//! for its idle timeout (by the monotonic clock or the wall clock, which
//! counts a sleep), when it was signalled, or when its executable changed
//! on disk, or the path it was started by now leads to another.
//!
//! A process about to open the database leases it first (see
//! [`super::lease`]). The daemon answers new requests `leased`, waits for
//! the ones running and for the warm-up, closes the database and grants the
//! lease; one it can't close in the time the process waits is refused
//! (`lease_timeout`), and the daemon serves on. While any process holds a
//! lease the database stays closed. Once the last one ends, by closing its
//! connection or by exiting, however it exits, the daemon waits 250 ms,
//! opens the database again and warms it up, unless another lease was asked
//! for meanwhile.
//!
//! Shutting down records `stopping`, unlinks the socket, so new clients fall
//! back at once, answers queued connections `stopping`, waits for running
//! requests and for the warm-up or a reload still running, and closes the
//! database before the lock is let go. When the database can't be closed
//! (a request panicked, see [`WarmState::poisoned`]; its files were
//! overwritten; or something still held it when the wait ran out), the
//! daemon records that it did not close it and exits 1 without dropping
//! anything, holding the lock until the process is gone: the next open
//! repairs the files, as after a one-shot command that was killed.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axil_core::GraphState;
use serde_json::{json, Value};

use super::drain::QueueStats;
use super::exec::{self, NotServed, ReopenError, WarmState};
use super::paths::{self, write_atomic, RunDir};
use super::proto::{
    self, read_frame, read_json, write_frame, write_json, Ack, ExeId, Hello, Identity, LeaseReply,
    Op, RunHeader, RunReply, DAEMON_PROTO, MAX_FRAME, STRIPPED_ENV, VERSION,
};
use super::state::{op_label, Change, DaemonInfo, FileFp, Phase, PhaseCell, Stats};
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

/// How long shutting down waits for running requests and background work.
const DRAIN_WAIT: Duration = Duration::from_secs(10);

/// How long a connection over [`MAX_CONNECTIONS`] may take to send its
/// request, which is answered on the accepting thread.
const OVER_CAP_READ: Duration = Duration::from_millis(200);

/// How long shutting down waits for the last references to the handle,
/// after the requests and the background work are done: recall workers a
/// deadline left running.
const RELEASE_WAIT: Duration = Duration::from_secs(3);

/// How long a daemon starting waits for the lock while the daemon holding
/// it says it is stopping: that one waits up to 10 s for its requests and
/// 3 s for the last references to the handle.
const STOPPING_WAIT: Duration = Duration::from_secs(30);

/// How long the daemon waits after the last lease ends before it opens the
/// database again: a process that exited may hold its locks a moment longer,
/// and another often follows it.
const UNLEASE_DELAY: Duration = Duration::from_millis(250);

/// The longest a lease request may have the daemon wait for its requests.
const MAX_LEASE_WAIT: Duration = Duration::from_secs(60);

/// How often the tick runs.
const TICK: Duration = Duration::from_secs(1);

/// The size at which the log moves to `daemon.log.1`.
const LOG_ROTATE_BYTES: u64 = 1024 * 1024;

/// Between two tries to open the database while another process holds it
/// without a lease.
const HELD_ELSEWHERE_RETRY: Duration = Duration::from_millis(100);

/// How many times in a row a failed graph is reloaded before the daemon
/// gives up on the database.
const GRAPH_RELOADS: u32 = 1;

/// The reload reason for a graph that failed to load.
const GRAPH_FAILED: &str = "the graph failed to load";

/// Why the handle is stale from a lease until the database is opened again.
const LEASED: &str = "the database was leased";

/// How a daemon runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Stop after this long without a run request.
    pub idle_timeout: Duration,
    /// How long to retry the first open while another process's lock
    /// blocks it. Opening again after a lease or a change waits for such a
    /// process for as long as it holds the database.
    pub open_wait: Duration,
    /// How long a reopen waits for the requests on the old handle.
    pub reopen_wait: Duration,
    /// How long shutting down waits for running requests and background
    /// work.
    pub drain_wait: Duration,
    /// The path the daemon's binary was started by, before symlinks are
    /// resolved: the daemon retires when it leads to another binary.
    pub launch_path: Option<PathBuf>,
    /// This is the daemon process: bind under umask 077 and rotate the log
    /// at fd 1 and 2. Both are process-wide.
    pub process: bool,
    /// Drain the hook queue (see [`super::drain`]).
    pub drain_queue: bool,
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
    /// A request, or the warm-up, panicked.
    Panicked,
    /// A store was written in place under the open handle.
    Overwritten(String),
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
            StopReason::Overwritten(why) => format!("its files were overwritten: {why}"),
        }
    }

    fn exit_code(&self) -> i32 {
        match self {
            StopReason::Blocked(_) | StopReason::Panicked | StopReason::Overwritten(_) => {
                crate::EXIT_ERROR
            }
            _ => crate::EXIT_OK,
        }
    }
}

/// Why the daemon did not get to serve.
enum Unopened {
    /// It was told to stop first.
    Stopped(StopReason),
    /// The database can't be served, for this reason.
    Unservable(String),
}

/// How a shutdown ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Finish {
    /// The database is closed; exit with this code.
    Exit(i32),
    /// The database was left open: a request panicked (see
    /// [`WarmState::poisoned`]), its files were overwritten, or something
    /// still held it. Exit 1 without dropping anything, and without letting
    /// the lock go before the process is gone.
    Abandoned,
}

/// One entry in the daemon's log, each of its lines stamped. Unit tests
/// run daemons in the test process, whose stderr is not a log.
pub(super) fn log(message: impl std::fmt::Display) {
    if cfg!(test) {
        return;
    }
    let stamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let pid = std::process::id();
    for line in message.to_string().lines() {
        errln!("{stamp} axil daemon[{pid}]: {line}");
    }
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Whether a request's deadline has passed.
fn expired(deadline_ms: Option<u64>) -> bool {
    deadline_ms.is_some_and(|deadline| epoch_ms() >= deadline)
}

/// The processes that asked for a lease and have not let it go.
#[derive(Default)]
struct Leases {
    holders: Vec<Holder>,
    next_id: u64,
    /// Moves on at each request, so a reopen meant for after the last lease
    /// ended is called off when another is asked for meanwhile.
    generation: u64,
    /// Leases granted since the daemon started.
    granted: u64,
}

/// A process that asked for a lease.
struct Holder {
    id: u64,
    pid: u32,
    since: Instant,
    granted: bool,
    /// The daemon's own, for the queue's steps that run in their own
    /// process (see [`Server::yield_db`]).
    queue: bool,
}

/// The open handle's view of when its files last matched it.
struct Freshness {
    /// The files as the handle left them.
    baseline: FileFp,
    /// When they were last compared.
    checked: Instant,
    /// A change found and not reloaded yet. Until a reload succeeds, no
    /// request runs on the handle.
    stale: Option<String>,
    /// Work running on the handle ([`Busy`]).
    busy: usize,
    /// The baseline's store contents are what the handle's own work last
    /// left them as, so they can be compared; see [`FileFp`].
    settled: bool,
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
    /// When the last run request came or finished, on both clocks: an
    /// [`Instant`] does not advance while the machine sleeps.
    last_active: Mutex<(Instant, SystemTime)>,
    fresh: Mutex<Freshness>,
    leases: Mutex<Leases>,
    graph_reloads: AtomicU32,
    fail_graph: AtomicBool,
    /// The warm-up of the open handle was put off for a lease asked for
    /// meanwhile; see [`Server::prewarm_on`].
    unwarmed: AtomicBool,
    /// Opening the database again found another process holding it without
    /// a lease; see [`Server::held_elsewhere`].
    held_elsewhere: AtomicBool,
    accept: Mutex<Option<JoinHandle<()>>>,
    /// The warm-ups and reloads started, which shutting down waits for;
    /// `None` once it has begun, when nothing more starts.
    background: Mutex<Option<Vec<JoinHandle<()>>>>,
    /// [`Options::launch_path`], when it leads to this binary.
    launch: Option<PathBuf>,
    /// Set to wake the queue's drain thread: a job was queued, or the
    /// database opened again.
    woken: Mutex<bool>,
    wake: Condvar,
    /// The queue's drain thread, once the database first opened.
    drain: Mutex<Option<JoinHandle<()>>>,
    queue: QueueStats,
    #[cfg(test)]
    panic_prewarm: AtomicBool,
}

/// What a step of a queued job came to; see [`Server::run_queued`].
pub(super) enum Queued {
    /// It ran on the open handle.
    Ran(exec::Response),
    /// The daemon does not run it on its handle; it runs in a process of
    /// its own.
    NotServed(NotServed),
    /// Not now: the database is opening, opening again or leased.
    Wait,
    /// Not at all: the daemon is stopping with the database closed, or a
    /// request panicked.
    Stopped,
}

/// A run request counted against [`MAX_IN_FLIGHT`] until dropped.
struct Admitted<'a>(&'a AtomicUsize);

impl<'a> Admitted<'a> {
    /// Sequentially consistent with the phase: see [`PhaseCell`].
    fn new(count: &'a AtomicUsize, cap: usize) -> Option<Self> {
        if count.fetch_add(1, Ordering::SeqCst) >= cap {
            count.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Self(count))
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Work on the open handle, counted until dropped: a request, or the
/// warm-up. When the last of it ends, the stores' contents as it left them
/// become the baseline ([`FileFp`]).
struct Busy<'a>(&'a Server);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        let mut fresh = self.0.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        fresh.busy -= 1;
        if fresh.busy == 0 {
            let now = FileFp::read(&self.0.key);
            self.0.settle(&mut fresh, &now);
        }
    }
}

impl Server {
    /// Bind the socket, open the database at `key` and start serving it.
    /// Fails, with the socket gone and the reason written to
    /// `daemon.blocked`, when the database can't be served.
    pub fn start(key: PathBuf, run: RunDir, opts: Options) -> Result<Arc<Server>, String> {
        let identity =
            Identity::current().map_err(|e| format!("can't identify this binary: {e}"))?;
        let exe = ExeId::current().ok();
        let launch = opts.launch_path.clone().filter(|path| {
            let same = exe.is_some() && ExeId::of(path).ok() == exe;
            if !same {
                log(format!(
                    "not watching {}: it does not lead to this binary",
                    path.display()
                ));
            }
            same
        });
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
        let baseline = FileFp::read(&key);
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
            last_active: Mutex::new((Instant::now(), SystemTime::now())),
            fresh: Mutex::new(Freshness {
                baseline,
                checked: Instant::now(),
                stale: None,
                busy: 0,
                settled: false,
            }),
            leases: Mutex::new(Leases::default()),
            graph_reloads: AtomicU32::new(0),
            fail_graph: AtomicBool::new(false),
            unwarmed: AtomicBool::new(false),
            held_elsewhere: AtomicBool::new(false),
            accept: Mutex::new(None),
            background: Mutex::new(Some(Vec::new())),
            launch,
            woken: Mutex::new(false),
            wake: Condvar::new(),
            drain: Mutex::new(None),
            queue: QueueStats::default(),
            #[cfg(test)]
            panic_prewarm: AtomicBool::new(false),
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

        match server.open() {
            Ok(()) => Ok(server),
            Err(Unopened::Stopped(reason)) => {
                server.shutdown(&reason);
                Err(format!("stopped while opening: {}", reason.describe()))
            }
            Err(Unopened::Unservable(why)) => {
                server.block(&why);
                server.shutdown(&StopReason::Blocked(why.clone()));
                Err(why)
            }
        }
    }

    /// Open the database, retrying while another process holds its lock.
    fn open(self: &Arc<Self>) -> Result<(), Unopened> {
        let deadline = Instant::now() + self.opts.open_wait;
        let state = loop {
            if let Some(sig) = self.signalled() {
                return Err(Unopened::Stopped(StopReason::Signal(sig)));
            }
            if let Some(reason) = self.stop_reason() {
                return Err(Unopened::Stopped(reason));
            }
            let before = FileFp::read(&self.key);
            match WarmState::open(&self.key) {
                Ok(state) => {
                    self.set_fresh(before);
                    break state;
                }
                Err(e) if crate::is_lock_contention(&e) && Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(Unopened::Unservable(format!("{e:#}"))),
            }
        };
        if let Some(why) = unservable(&state) {
            return Err(Unopened::Unservable(why));
        }
        let _ = self.state.set(state);
        // Whatever kept an earlier daemon from serving is gone, which a
        // client that finds this one ready must not be told otherwise.
        let _ = fs::remove_file(self.run.blocked());
        // On disk before any client can see it: `daemon status` must not
        // report a daemon that served as one still starting.
        {
            let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
            info.ready_at = Some(now_rfc3339());
        }
        self.write_info_as(Phase::Ready.name());
        self.phase.advance(Phase::Starting, Phase::Ready);
        log("ready");
        self.prewarm();
        if self.opts.drain_queue {
            match super::drain::spawn(self.clone()) {
                Ok(handle) => {
                    *self.drain.lock().unwrap_or_else(PoisonError::into_inner) = Some(handle)
                }
                Err(e) => log(format!("can't start draining the hook queue: {e}")),
            }
        }
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
        if let Some(sig) = self.signalled() {
            return Some(StopReason::Signal(sig));
        }
        if self.in_flight.load(Ordering::SeqCst) == 0 && self.idle_for() >= self.opts.idle_timeout {
            return Some(StopReason::Idle);
        }
        if let Some(exe) = &self.exe {
            // Through the path it was started by: a symlink moved to a new
            // build leads elsewhere, though this binary is unchanged.
            let watched = self.launch.as_deref().unwrap_or(&exe.path);
            if ExeId::of(watched).ok().as_ref() != Some(exe) {
                return Some(StopReason::ExeChanged);
            }
        }
        if self.opts.process {
            rotate_log(&self.run);
        }
        if self.phase.get() == Phase::Ready {
            match self.check(Duration::ZERO, 0) {
                Some(Change::Reload(why)) => {
                    self.spawn_background("axil-reload", move |server| server.reload(&why))
                }
                Some(Change::Overwritten(why)) => {
                    self.abandon(&why);
                    return Some(StopReason::Overwritten(why));
                }
                None => {}
            }
        }
        None
    }

    /// Run `work` on a thread of its own, which shutting down waits for.
    /// Once shutting down has begun, nothing starts.
    fn spawn_background(
        self: &Arc<Self>,
        name: &str,
        work: impl FnOnce(Arc<Server>) + Send + 'static,
    ) {
        let mut background = self
            .background
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(handles) = background.as_mut() else {
            return;
        };
        handles.retain(|handle| !handle.is_finished());
        let server = self.clone();
        match thread::Builder::new()
            .name(name.into())
            .spawn(move || work(server))
        {
            Ok(handle) => handles.push(handle),
            Err(e) => log(format!("can't start {name}: {e}")),
        }
    }

    /// The signal the daemon process received, if any. Signals are the
    /// process's, so only the daemon process takes them.
    fn signalled(&self) -> Option<i32> {
        if self.opts.process {
            take_signal()
        } else {
            None
        }
    }

    /// Stop serving and close the database, as [`Finish`] says.
    pub fn shutdown(self: &Arc<Self>, reason: &StopReason) -> Finish {
        self.phase.set(Phase::Stopping);
        // `daemon stop` and a daemon starting wait for one that says so.
        self.write_info_as(Phase::Stopping.name());
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
        let mut background = self
            .background
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_default();
        // The queue's drain thread finishes the job it is on, and stops.
        self.wake_drain();
        if let Some(drain) = self
            .drain
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            background.push(drain);
        }
        let working = || background.iter().filter(|h| !h.is_finished()).count();
        let deadline = Instant::now() + self.opts.drain_wait;
        while (self.in_flight.load(Ordering::SeqCst) > 0 || working() > 0)
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(10));
        }
        let (left, busy) = (self.in_flight.load(Ordering::SeqCst), working());
        if left > 0 || busy > 0 {
            log(format!(
                "{left} requests and {busy} warm-ups, reloads or queued jobs still running after {:?}",
                self.opts.drain_wait
            ));
        }
        for handle in background {
            if handle.is_finished() {
                let _ = handle.join();
            }
        }

        let closed = match self.state.get() {
            None => true,
            Some(state) if state.poisoned() => false,
            Some(state) => match state.release(RELEASE_WAIT) {
                Ok(()) => {
                    log("closed the database");
                    true
                }
                Err(e) => {
                    log(format!("did not close the database: {e}"));
                    false
                }
            },
        };
        {
            let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
            info.stopped_at = Some(now_rfc3339());
            info.exit_reason = Some(reason.describe());
            info.closed_cleanly = Some(closed);
        }
        self.write_info_as("stopped");
        if closed {
            Finish::Exit(reason.exit_code())
        } else {
            Finish::Abandoned
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
            "at_ms": epoch_ms(),
        });
        if let Err(e) = write_atomic(&self.run.blocked(), record.to_string().as_bytes()) {
            log(format!("can't write {}: {e}", self.run.blocked().display()));
        }
    }

    fn stop_blocked(&self, why: String) {
        self.block(&why);
        self.request_stop(StopReason::Blocked(why));
    }

    /// Opening the database again found another process holding it without
    /// a lease: a build from before leases, or one run with
    /// `AXIL_DAEMON=off`. The daemon waits for it for as long as it holds
    /// the database, and a command sent meanwhile is answered `leased`, so
    /// it runs in its own process at once, as beside a lease holder.
    fn held_elsewhere(&self) {
        if !self.held_elsewhere.swap(true, Ordering::AcqRel) {
            log("another process has the database open without a lease: waiting for it");
        }
    }

    /// The database opened again: no other process holds it.
    fn let_go_elsewhere(&self) {
        if self.held_elsewhere.swap(false, Ordering::AcqRel) {
            log("the other process let go of the database");
        }
    }

    /// Record the phase. Read with the record locked: a shutdown sets its
    /// phase before it records it, so whichever records last records the
    /// phase that stands.
    fn write_info(&self) {
        let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
        let state = self.phase.get().name();
        self.write_locked(&mut info, state);
    }

    fn write_info_as(&self, state: &str) {
        let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
        self.write_locked(&mut info, state);
    }

    /// Record `state` ahead of the phase it stands for, as long as the phase
    /// is still `now`: never over a shutdown that began meanwhile.
    fn write_info_while(&self, now: Phase, state: &str) -> bool {
        let mut info = self.info.lock().unwrap_or_else(PoisonError::into_inner);
        if self.phase.get() != now {
            return false;
        }
        self.write_locked(&mut info, state);
        true
    }

    fn write_locked(&self, info: &mut DaemonInfo, state: &str) {
        info.state = state.to_string();
        let bytes = serde_json::to_vec_pretty(&*info).unwrap_or_default();
        if let Err(e) = write_atomic(&self.run.info(), &bytes) {
            log(format!("can't write {}: {e}", self.run.info().display()));
        }
    }

    /// How long since the last run request, by the longer of the two
    /// clocks: the monotonic one misses a sleep, the wall clock can be set
    /// back.
    fn idle_for(&self) -> Duration {
        let (instant, wall) = *self
            .last_active
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let slept = SystemTime::now().duration_since(wall).unwrap_or_default();
        instant.elapsed().max(slept)
    }

    fn touch(&self) {
        *self
            .last_active
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = (Instant::now(), SystemTime::now());
    }

    /// A handle opened on the files `before` describes: the baseline from
    /// now on. Their identities are taken from before the open, so a file
    /// replaced while it opened is found; the stores' contents from after
    /// it, since the open may write.
    fn set_fresh(&self, before: FileFp) {
        let mut baseline = before;
        baseline.absorb(&FileFp::read(&self.key));
        let mut fresh = self.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        fresh.baseline = baseline;
        fresh.checked = Instant::now();
        fresh.stale = None;
        fresh.settled = true;
    }

    /// Count work on the handle until the guard is dropped.
    fn busy(&self) -> Busy<'_> {
        self.fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .busy += 1;
        Busy(self)
    }

    /// With no work on the handle, take the stores' contents in `now` as
    /// what the handle's own work left them as. A recall worker a deadline
    /// left running may still write (a recall records which memories it
    /// returned), so while one runs nothing is taken, and the contents are
    /// not compared until a check finds it gone.
    fn settle(&self, fresh: &mut Freshness, now: &FileFp) {
        if cli_io::RECALL_SLOTS.live() > 0 {
            fresh.settled = false;
            return;
        }
        // Files replaced since are left for the reload to take in; should
        // the change be undone instead, the contents are taken by the next
        // check that finds nothing running, not compared.
        if fresh.baseline.changed(now, false).is_none() {
            fresh.baseline.absorb(now);
            fresh.settled = true;
        } else {
            fresh.settled = false;
        }
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
                Ok((stream, _)) => self.accepted(stream),
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
                    self.accepted(stream);
                }
            }
        }
    }

    /// A connection, served when it comes from this user. The socket is
    /// 0600 in a directory only this user can enter; this holds even when
    /// someone changed that.
    fn accepted(self: &Arc<Self>, stream: UnixStream) {
        match proto::peer_uid(&stream) {
            Ok(peer) if peer == paths::uid() => self.spawn_connection(stream),
            Ok(peer) => log(format!("refused a connection from uid {peer}")),
            Err(e) => log(format!(
                "refused a connection that can't be told apart: {e}"
            )),
        }
    }

    fn spawn_connection(self: &Arc<Self>, mut stream: UnixStream) {
        if self.connections.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
            self.connections.fetch_sub(1, Ordering::AcqRel);
            // Answered here, briefly: `hello` and `shutdown` work whatever
            // the load, so `daemon stop` does; anything else is told to
            // come back.
            let _ = stream.set_read_timeout(Some(OVER_CAP_READ));
            let _ = stream.set_write_timeout(Some(OVER_CAP_READ));
            if let Ok(op) = read_json::<Op>(&mut stream) {
                let _ = match op {
                    Op::Hello => write_json(&mut stream, &self.hello()),
                    Op::Shutdown { reason } => self.acknowledge_shutdown(&mut stream, reason),
                    Op::Lease { .. } => write_json(&mut stream, &LeaseReply::refused("overloaded")),
                    _ => write_json(&mut stream, &RunReply::Overloaded),
                };
            }
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
            Op::Shutdown { reason } => self.acknowledge_shutdown(&mut stream, reason),
            Op::Status => write_json(&mut stream, &self.status()),
            Op::Check(header) => {
                let mismatch = self.identity.mismatch(&header);
                write_json(&mut stream, &json!({"op": "check", "mismatch": mismatch}))
            }
            Op::Lease { db, wait_ms, pid } => self.handle_lease(&mut stream, &db, wait_ms, pid),
            Op::Poke => {
                self.wake_drain();
                let ack = Ack {
                    op: "poke".into(),
                    ok: self.drains(),
                };
                write_json(&mut stream, &ack)
            }
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
            #[cfg(debug_assertions)]
            Op::DebugHold { ms, writer } => {
                self.spawn_background("axil-debug-hold", move |server| {
                    let _busy = server.busy();
                    let state = server.state.get();
                    let held = state.and_then(WarmState::handle);
                    let writing = state.filter(|_| writer).map(WarmState::debug_writer);
                    thread::sleep(Duration::from_millis(ms));
                    drop(writing);
                    drop(held);
                });
                let ack = Ack {
                    op: "debug_hold".into(),
                    ok: true,
                };
                write_json(&mut stream, &ack)
            }
            #[cfg(debug_assertions)]
            Op::DebugOpenGuard => {
                let key = self.key.clone();
                let tripped = panic::catch_unwind(move || axil_core::open_guard(&key)).is_err();
                write_json(
                    &mut stream,
                    &json!({"op": "debug_open_guard", "tripped": tripped}),
                )
            }
        };
    }

    /// Answer `lease` for process `pid`, which is about to open the database
    /// `db`, and hold the lease until the connection closes.
    fn handle_lease(
        self: &Arc<Self>,
        stream: &mut UnixStream,
        db: &str,
        wait_ms: u64,
        pid: u32,
    ) -> io::Result<()> {
        if !db.is_empty() && !proto::names_db(db, &self.key) {
            return write_json(stream, &LeaseReply::refused("wrong_db"));
        }
        let wait = Duration::from_millis(wait_ms).min(MAX_LEASE_WAIT);
        let id = self.ask_lease(pid, false);
        let result = self.let_go(wait);
        let sent = match result {
            Ok(()) => {
                self.grant_lease(id);
                log(format!("leased the database to pid {pid}"));
                write_json(stream, &LeaseReply::granted())
            }
            Err(reason) => {
                log(format!("did not lease the database to pid {pid}: {reason}"));
                write_json(stream, &LeaseReply::refused(reason))
            }
        };
        if result.is_ok() && sent.is_ok() {
            // The lease lasts until the holder's end closes, as it does
            // when the holder exits, however it exits.
            let _ = stream.set_read_timeout(None);
            let mut buf = [0u8; 64];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            log(format!("pid {pid} let go of its lease"));
        }
        self.end_lease(id);
        sent
    }

    /// Count process `pid` in as asking for a lease, the queue's own when
    /// `queue`; see [`Server::end_lease`].
    fn ask_lease(&self, pid: u32, queue: bool) -> u64 {
        let mut leases = self.leases.lock().unwrap_or_else(PoisonError::into_inner);
        leases.next_id += 1;
        leases.generation += 1;
        let id = leases.next_id;
        leases.holders.push(Holder {
            id,
            pid,
            since: Instant::now(),
            granted: false,
            queue,
        });
        id
    }

    fn grant_lease(&self, id: u64) {
        let mut leases = self.leases.lock().unwrap_or_else(PoisonError::into_inner);
        leases.granted += 1;
        if let Some(holder) = leases.holders.iter_mut().find(|h| h.id == id) {
            holder.granted = true;
        }
    }

    /// Whether any process holds a lease or waits for one.
    fn lease_asked(&self) -> bool {
        !self
            .leases
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .holders
            .is_empty()
    }

    /// Count the lease `id` out. When it was the last, the database is
    /// opened again, after [`UNLEASE_DELAY`], unless another lease is asked
    /// for by then.
    fn end_lease(self: &Arc<Self>, id: u64) {
        self.leases
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .holders
            .retain(|h| h.id != id);
        self.unlease_if_unheld();
    }

    /// With the database closed for leases none holds any more, open it
    /// again (see [`Server::unlease`]).
    fn unlease_if_unheld(self: &Arc<Self>) {
        let generation = {
            let leases = self.leases.lock().unwrap_or_else(PoisonError::into_inner);
            if !leases.holders.is_empty() {
                return;
            }
            leases.generation
        };
        match self.phase.get() {
            Phase::Leased => {
                self.spawn_background("axil-unlease", move |server| server.unlease(generation))
            }
            // The warm-up was put off for a lease that was then refused.
            Phase::Ready if self.unwarmed.load(Ordering::Acquire) => self.prewarm(),
            _ => {}
        }
    }

    /// Close the database for a lease, within `wait`: stop serving it, and
    /// wait for the requests and the warm-up running on it. When the time
    /// runs out first, serve on and return `lease_timeout`. With the
    /// database closed for another lease already, return at once.
    fn let_go(&self, wait: Duration) -> Result<(), &'static str> {
        let deadline = Instant::now() + wait;
        // One caller at a time moves the daemon out of ready; another
        // waits for what that one gets to.
        loop {
            match self.phase.get() {
                Phase::Leased => return Ok(()),
                Phase::Stopping => return Err("stopping"),
                Phase::Ready => {
                    if self.phase.advance(Phase::Ready, Phase::Leasing) {
                        break;
                    }
                    continue;
                }
                // Opening, reopening, or letting go for another lease.
                Phase::Starting | Phase::Reloading | Phase::Leasing => {}
            }
            if Instant::now() >= deadline {
                return Err("lease_timeout");
            }
            thread::sleep(Duration::from_millis(10));
        }
        let Some(state) = self.state.get() else {
            self.phase.advance(Phase::Leasing, Phase::Ready);
            return Err("lease_timeout");
        };
        log("letting go of the database for a lease");
        let left = deadline.saturating_duration_since(Instant::now());
        match state.release(left) {
            Ok(()) => {
                // What the holder does to the files is its own business: until
                // the database is opened again, a request finds the handle
                // stale and is not served, rather than take the holder's
                // writes for someone else's overwrite.
                self.fresh
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .stale = Some(LEASED.to_string());
                // A shutdown that began meanwhile finds the database closed.
                if self.phase.advance(Phase::Leasing, Phase::Leased) {
                    self.write_info();
                }
                Ok(())
            }
            // A request panicked on the handle, which is never closed: the
            // daemon is stopping, and its exit lets the database go.
            Err(ReopenError::Poisoned) => Err("stopping"),
            Err(_) => {
                self.phase.advance(Phase::Leasing, Phase::Ready);
                Err("lease_timeout")
            }
        }
    }

    /// Open the database again once the last lease has ended, unless one
    /// is asked for again by the time it opens: the leases of `generation`
    /// were the last.
    fn unlease(self: &Arc<Self>, generation: u64) {
        thread::sleep(UNLEASE_DELAY);
        // A lease asked for during the wait was granted at once, the
        // database being closed, and its end opens it again.
        let last = {
            let leases = self.leases.lock().unwrap_or_else(PoisonError::into_inner);
            leases.generation == generation && leases.holders.is_empty()
        };
        if !last || !self.phase.advance(Phase::Leased, Phase::Reloading) {
            return;
        }
        let Some(state) = self.state.get() else {
            return;
        };
        log("the last lease ended: opening the database again");
        let opened = loop {
            if self.phase.get() == Phase::Stopping {
                return;
            }
            if self.lease_asked() {
                // Closed it stays, for the lease just asked for. One whose
                // wait ran out meanwhile was refused, and its end found the
                // database still opening: open it again for that one.
                self.phase.advance(Phase::Reloading, Phase::Leased);
                self.unlease_if_unheld();
                return;
            }
            let before = FileFp::read(&self.key);
            match state.reopen(self.opts.reopen_wait) {
                Ok(()) => {
                    self.let_go_elsewhere();
                    break Ok(before);
                }
                // A request panicked; the daemon is stopping.
                Err(ReopenError::Poisoned) => return,
                Err(ReopenError::Open(e)) if crate::is_lock_contention(&e) => {
                    self.held_elsewhere();
                    thread::sleep(HELD_ELSEWHERE_RETRY);
                }
                Err(ReopenError::Open(e)) => break Err(format!("{e:#}")),
                Err(e) => break Err(e.to_string()),
            }
        };
        match opened {
            Ok(_) if self.phase.get() == Phase::Stopping => {}
            Ok(before) => {
                if let Some(why) = unservable(state) {
                    self.stop_blocked(why);
                    return;
                }
                self.set_fresh(before);
                self.fail_graph.store(false, Ordering::Release);
                log("opened the database again");
                // On disk before a client can see it, as at the first open,
                // unless a shutdown began.
                if self.write_info_while(Phase::Reloading, Phase::Ready.name()) {
                    self.phase.advance(Phase::Reloading, Phase::Ready);
                }
                self.prewarm_on(state);
                self.wake_drain();
            }
            Err(failed) => self.stop_blocked(format!("reopening after a lease: {failed}")),
        }
    }

    /// Answer `shutdown`, then begin it.
    fn acknowledge_shutdown(&self, stream: &mut UnixStream, reason: String) -> io::Result<()> {
        let ack = Ack {
            op: "shutdown".into(),
            ok: true,
        };
        let sent = write_json(stream, &ack);
        self.request_stop(StopReason::Requested(reason));
        sent
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
            let reason = match &reply {
                RunReply::Incompatible { reason } => format!("incompatible {reason}"),
                other => serde_json::to_value(other)
                    .ok()
                    .and_then(|v| v["reply"].as_str().map(str::to_string))
                    .unwrap_or_default(),
            };
            self.stats.refused(reason);
            (reply, None)
        };
        // Counted in before the phase is read: a shutdown that set the
        // phase and then found no request running can't have this one
        // start after it (see `PhaseCell`).
        let Some(_admitted) = Admitted::new(&self.in_flight, MAX_IN_FLIGHT) else {
            return refused(RunReply::Overloaded);
        };
        // Before any reply a client of another protocol version may not
        // know: it is told only that it does not match.
        if let Some(reason) = self.identity.mismatch(&header) {
            return refused(RunReply::Incompatible {
                reason: reason.to_string(),
            });
        }
        match self.phase.get() {
            Phase::Starting => return refused(RunReply::Starting),
            Phase::Stopping => return refused(RunReply::Stopping),
            // Another process holds the database without a lease: the
            // command runs in its own process, as beside a lease holder.
            Phase::Leased | Phase::Reloading if self.held_elsewhere.load(Ordering::Acquire) => {
                return refused(RunReply::Leased)
            }
            // With no holder left, the database is about to be opened
            // again: the client may wait for it as for a daemon starting.
            Phase::Leasing | Phase::Leased if !self.lease_asked() => {
                return refused(RunReply::Starting)
            }
            Phase::Leasing | Phase::Leased => return refused(RunReply::Leased),
            Phase::Ready | Phase::Reloading => {}
        }
        if expired(header.deadline_ms) {
            return refused(RunReply::Expired);
        }
        let Some(state) = self.state.get() else {
            return refused(RunReply::Starting);
        };
        let op = op_label(&header.argv);
        match self.check(FRESH_FOR, 0) {
            // Reopened on a thread of its own, which can take a while: the
            // client asks again, as while the daemon opens the database at
            // its start.
            Some(Change::Reload(why)) => {
                if self.phase.get() == Phase::Ready {
                    self.spawn_background("axil-reload", move |server| server.reload(&why));
                }
                self.stats.not_served(op);
                return (RunReply::not_served(NotServed::Unavailable), None);
            }
            Some(Change::Overwritten(why)) => {
                self.abandon(&why);
                self.request_stop(StopReason::Overwritten(why));
                self.stats.not_served(op);
                return (RunReply::not_served(NotServed::Unavailable), None);
            }
            None => {}
        }

        let started = Instant::now();
        let deadline = header.deadline_ms;
        let gate = |writes: bool| self.gate(deadline, writes);
        let result = {
            let _busy = self.busy();
            exec::execute_gated(header.to_request(stdin), state, &gate)
        };
        let took = started.elapsed();
        if state.poisoned() {
            self.request_stop(StopReason::Panicked);
        }
        match result {
            Ok(response)
                if response.stdout.len() > MAX_FRAME || response.stderr.len() > MAX_FRAME =>
            {
                self.stats.not_served(op);
                let reply = RunReply::NotServed {
                    reason: "too_large".into(),
                    detail: "the output is over the 32 MiB frame limit".into(),
                };
                (reply, None)
            }
            Ok(response) => {
                // Only a command run is use: a client that is refused, or
                // whose own command leases the database, keeps no daemon up.
                self.touch();
                self.stats.served(op, took);
                (RunReply::served(&response), Some(response))
            }
            Err(not_served) => {
                self.stats.not_served(op);
                (RunReply::not_served(not_served), None)
            }
        }
    }

    /// Asked last before a request runs, once a write holds the writer
    /// lock: whether its deadline passed while it waited, and whether the
    /// handle still is the database's. A write is checked against the
    /// files as they are now, not as they were up to 250 ms ago: a write
    /// to a handle whose file was replaced is lost with the handle.
    fn gate(&self, deadline_ms: Option<u64>, writes: bool) -> Result<(), NotServed> {
        if expired(deadline_ms) {
            return Err(NotServed::Expired);
        }
        // A write that waited for the writer lock while the database was
        // let go of.
        if matches!(self.phase.get(), Phase::Leasing | Phase::Leased) {
            return Err(NotServed::Leased);
        }
        let fresh_for = if writes { Duration::ZERO } else { FRESH_FOR };
        match self.check(fresh_for, 1) {
            None => Ok(()),
            Some(Change::Reload(_)) => Err(NotServed::Unavailable),
            Some(Change::Overwritten(why)) => {
                self.abandon(&why);
                self.request_stop(StopReason::Overwritten(why));
                Err(NotServed::Unavailable)
            }
        }
    }

    /// What changed under the open handle, if anything: its files changed
    /// without it, or the graph failed to load. A change found stands until
    /// a reload takes it in. Otherwise the file check stands for
    /// `fresh_for`. The stores' contents are compared only when no work
    /// but the caller's own (`own`) runs on the handle.
    fn check(&self, fresh_for: Duration, own: usize) -> Option<Change> {
        if self.graph_failed() {
            return Some(Change::Reload(GRAPH_FAILED.to_string()));
        }
        let mut fresh = self.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(why) = &fresh.stale {
            return Some(Change::Reload(why.clone()));
        }
        if fresh.checked.elapsed() < fresh_for {
            return None;
        }
        fresh.checked = Instant::now();
        let now = FileFp::read(&self.key);
        let quiet = fresh.busy <= own && cli_io::RECALL_SLOTS.live() == 0;
        if quiet && !fresh.settled {
            self.settle(&mut fresh, &now);
        }
        let change = fresh.baseline.changed(&now, quiet && fresh.settled);
        if let Some(Change::Reload(why)) = &change {
            fresh.stale = Some(why.clone());
        }
        change
    }

    /// Whether the files behind the handle were replaced, added, removed or
    /// edited since it opened, read now.
    fn files_changed(&self) -> bool {
        let fresh = self.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        fresh
            .baseline
            .changed(&FileFp::read(&self.key), false)
            .is_some()
    }

    /// Stop serving the handle for good and never close it: a store was
    /// written in place under it (see [`WarmState::abandon`]).
    fn abandon(&self, why: &str) {
        let Some(state) = self.state.get().filter(|state| !state.poisoned()) else {
            return;
        };
        log(format!(
            "not serving {} any more, and leaving it unclosed: {why}",
            self.key.display()
        ));
        state.abandon();
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
    /// once. Requests are not served while it runs, nor after it until one
    /// succeeds (the change found stands; see [`Server::check`]): a reload
    /// put off because requests still hold the old handle is tried again
    /// by the next tick. A database that doesn't open again stops the
    /// daemon.
    fn reload(self: &Arc<Self>, why: &str) {
        let Some(state) = self.state.get() else {
            return;
        };
        if !self.phase.advance(Phase::Ready, Phase::Reloading) {
            return;
        }
        // Another caller may have reloaded since `why` was found, or the
        // change been undone.
        let graph = why == GRAPH_FAILED;
        let still = if graph {
            self.graph_failed()
        } else {
            self.files_changed()
        };
        if !still {
            self.fresh
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .stale = None;
            self.phase.advance(Phase::Reloading, Phase::Ready);
            return;
        }
        if graph && self.graph_reloads.fetch_add(1, Ordering::AcqRel) >= GRAPH_RELOADS {
            self.stop_blocked(format!("{why} again after a reload"));
            return;
        }
        log(format!("reloading: {why}"));
        enum Outcome {
            Reopened(FileFp),
            Postponed,
            Leased,
            Stopping,
            Failed(String),
        }
        let outcome = loop {
            if self.phase.get() == Phase::Stopping {
                break Outcome::Stopping;
            }
            let before = FileFp::read(&self.key);
            match state.reopen(self.opts.reopen_wait) {
                Ok(()) => {
                    self.let_go_elsewhere();
                    break Outcome::Reopened(before);
                }
                Err(ReopenError::Running) | Err(ReopenError::InUse) => break Outcome::Postponed,
                Err(ReopenError::Poisoned) => {
                    break Outcome::Failed("a request panicked".to_string())
                }
                Err(ReopenError::Open(e)) if crate::is_lock_contention(&e) => {
                    self.held_elsewhere();
                    // The old handle is closed: a lease asked for is
                    // granted, and its end opens the database again.
                    if self.lease_asked() {
                        break Outcome::Leased;
                    }
                    thread::sleep(HELD_ELSEWHERE_RETRY);
                }
                Err(ReopenError::Open(e)) => break Outcome::Failed(format!("{e:#}")),
            }
        };
        match outcome {
            // Shutting down closes whatever handle there is.
            Outcome::Stopping => {}
            Outcome::Leased => {
                if self.phase.advance(Phase::Reloading, Phase::Leased) {
                    self.write_info();
                }
                self.unlease_if_unheld();
            }
            Outcome::Reopened(_) if self.phase.get() == Phase::Stopping => {}
            Outcome::Reopened(before) => {
                if let Some(why) = unservable(state) {
                    self.stop_blocked(why);
                    return;
                }
                self.set_fresh(before);
                self.fail_graph.store(false, Ordering::Release);
                self.stats.reloaded(why);
                self.phase.advance(Phase::Reloading, Phase::Ready);
                log("reloaded");
                self.prewarm_on(state);
                self.wake_drain();
            }
            Outcome::Postponed => {
                log("reload postponed: requests still hold the database");
                self.phase.advance(Phase::Reloading, Phase::Ready);
                self.wake_drain();
            }
            Outcome::Failed(failed) => {
                self.stop_blocked(format!("reopening after {why}: {failed}"))
            }
        }
    }

    fn prewarm(self: &Arc<Self>) {
        if let Some(state) = self.state.get() {
            self.prewarm_on(state);
        }
    }

    /// Load the embedding model and the graph's backend in the background.
    /// What they print is logged. Beyond the embedder's banners, a line
    /// printed here is one every one-shot command would print and no served
    /// command would, so the database is not served: a graph left in memory
    /// prints that warning in each one-shot command. A failed graph is
    /// reloaded by the next check instead.
    ///
    /// The graph's backend may write its adjacency tables. A panic there
    /// can unwind through a write transaction, so it leaves the handle as a
    /// request that panicked does ([`WarmState::poisoned`]).
    ///
    /// A process waiting for a lease would only wait longer for it, so then
    /// nothing is warmed up: the reopen after its lease does it.
    fn prewarm_on(self: &Arc<Self>, state: &WarmState) {
        if self.lease_asked() {
            self.unwarmed.store(true, Ordering::Release);
            return;
        }
        let Some(warm) = state.handle() else {
            return;
        };
        self.unwarmed.store(false, Ordering::Release);
        self.spawn_background("axil-prewarm", move |server| {
            let _busy = server.busy();
            let started = Instant::now();
            let (out, err, report) = cli_io::capture(ReqCtx::default(), || {
                panic::catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    if server.panic_prewarm.swap(false, Ordering::AcqRel) {
                        panic!("the warm-up panicked on purpose");
                    }
                    warm.db().prewarm()
                }))
            });
            let report = match report {
                Ok(report) => {
                    drop(warm);
                    report
                }
                Err(_) => {
                    log("the warm-up panicked; the database is left unclosed");
                    if let Some(state) = server.state.get() {
                        state.poison(warm);
                    }
                    server.request_stop(StopReason::Panicked);
                    return;
                }
            };
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
            // The embedder's banners (its execution provider, its
            // session pool) print once per process in the daemon, in
            // whichever command loads it first in a one-shot run; any
            // other line reports something wrong with the database.
            let problems: Vec<&str> = printed
                .lines()
                .filter(|line| !line.trim().is_empty() && !line.starts_with("[axil-vector]"))
                .collect();
            match graph {
                Some(GraphState::Failed(_)) => {}
                Some(GraphState::Memory) => server.stop_blocked(
                    "the graph's adjacency tables can't be used, so every command would \
                         print that it loads every edge into memory"
                        .to_string(),
                ),
                _ if !problems.is_empty() => {
                    server.stop_blocked(format!("warming up printed:\n{}", problems.join("\n")))
                }
                _ => server.graph_reloads.store(0, Ordering::Release),
            }
        });
    }

    /// The database the daemon serves.
    pub(super) fn key(&self) -> &Path {
        &self.key
    }

    /// The queue's figures.
    pub(super) fn queue(&self) -> &QueueStats {
        &self.queue
    }

    /// Whether the daemon is shutting down.
    pub(super) fn stopping(&self) -> bool {
        self.phase.get() == Phase::Stopping
    }

    /// Whether the daemon serves: the database open and ready.
    pub(super) fn serving(&self) -> bool {
        self.phase.get() == Phase::Ready
            && self
                .state
                .get()
                .is_some_and(|state| !state.poisoned() && state.handle().is_some())
    }

    /// Whether the daemon drains the hook queue, and goes on to.
    fn drains(&self) -> bool {
        !self.stopping()
            && self
                .drain
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
                .is_some_and(|drain| !drain.is_finished())
    }

    /// The writes clients sent waiting for the one running.
    pub(super) fn writers_waiting(&self) -> usize {
        self.state.get().map_or(0, WarmState::writers_waiting)
    }

    /// Wake the queue's drain thread.
    pub(super) fn wake_drain(&self) {
        *self.woken.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.wake.notify_all();
    }

    /// For the queue's drain thread: wait until woken, or `timeout`.
    pub(super) fn wait_for_wake(&self, timeout: Duration) {
        let woken = self.woken.lock().unwrap_or_else(PoisonError::into_inner);
        let (mut woken, _) = self
            .wake
            .wait_timeout_while(woken, timeout, |woken| !*woken)
            .unwrap_or_else(PoisonError::into_inner);
        *woken = false;
    }

    /// Run one step of a queued job, `argv` run in `cwd` with `stdin`, on
    /// the open handle, as a client's request runs (see [`Server::run`]),
    /// for the queue's drain thread. It is not refused for the cap on
    /// requests, nor once shutting down has begun while the database is
    /// still open: the job it belongs to finishes first. It counts as use,
    /// as a served request does.
    pub(super) fn run_queued(
        self: &Arc<Self>,
        argv: &[String],
        cwd: &Path,
        stdin: &[u8],
    ) -> Queued {
        let Some(state) = self.state.get() else {
            return Queued::Wait;
        };
        if state.poisoned() {
            return Queued::Stopped;
        }
        match self.phase.get() {
            Phase::Ready => {}
            Phase::Stopping if state.handle().is_some() => {}
            Phase::Stopping => return Queued::Stopped,
            Phase::Starting | Phase::Reloading | Phase::Leasing | Phase::Leased => {
                return Queued::Wait
            }
        }
        match self.check(FRESH_FOR, 0) {
            Some(Change::Reload(why)) => {
                if self.phase.get() == Phase::Ready {
                    self.spawn_background("axil-reload", move |server| server.reload(&why));
                }
                return Queued::Wait;
            }
            Some(Change::Overwritten(why)) => {
                self.abandon(&why);
                self.request_stop(StopReason::Overwritten(why));
                return Queued::Stopped;
            }
            None => {}
        }
        // Counted in, so shutting down waits for it.
        let Some(_admitted) = Admitted::new(&self.in_flight, usize::MAX) else {
            return Queued::Wait;
        };
        let op = op_label(argv);
        let request = exec::Request {
            argv: argv.iter().map(std::ffi::OsString::from).collect(),
            cwd: cwd.to_path_buf(),
            env: Vec::new(),
            stdin: Some(stdin.to_vec()),
            slow_query_log: true,
            queued: true,
        };
        let started = Instant::now();
        let gate = |writes: bool| self.gate(None, writes);
        let result = {
            let _busy = self.busy();
            exec::execute_gated(request, state, &gate)
        };
        if state.poisoned() {
            self.request_stop(StopReason::Panicked);
            return Queued::Stopped;
        }
        match result {
            Ok(response) => {
                self.touch();
                self.stats.served(op, started.elapsed());
                Queued::Ran(response)
            }
            Err(NotServed::Unavailable | NotServed::Expired | NotServed::Leased) => Queued::Wait,
            Err(why) => {
                self.stats.not_served(op);
                Queued::NotServed(why)
            }
        }
    }

    /// Let go of the database for the queue's drain thread, to run steps
    /// the daemon does not run in a process of their own: a lease the
    /// daemon holds itself, granted as another process's is (see
    /// [`Server::let_go`]). Returns the lease's id, for [`Server::unyield`].
    pub(super) fn yield_db(self: &Arc<Self>) -> Result<u64, &'static str> {
        let id = self.ask_lease(std::process::id(), true);
        match self.let_go(super::lease::LEASE_WAIT) {
            Ok(()) => {
                self.grant_lease(id);
                log("let go of the database to run queued steps in their own process");
                Ok(id)
            }
            Err(why) => {
                self.end_lease(id);
                Err(why)
            }
        }
    }

    /// End the lease [`Server::yield_db`] took: the database opens again
    /// once no other process holds one.
    pub(super) fn unyield(self: &Arc<Self>, id: u64) {
        log("the queued steps run in their own process are done");
        self.end_lease(id);
    }

    fn hello(&self) -> Hello {
        Hello {
            op: "hello".into(),
            proto: DAEMON_PROTO,
            build_id: self.identity.build_id.clone(),
            version: VERSION.into(),
            pid: std::process::id(),
            state: self.phase.get().name().into(),
            db: self.key.display().to_string(),
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
        let busy = self.in_flight.load(Ordering::SeqCst) > 0;
        let stale = self
            .fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stale
            .clone();
        let (leases, leases_granted) = {
            let leases = self.leases.lock().unwrap_or_else(PoisonError::into_inner);
            let holders: Vec<Value> = leases
                .holders
                .iter()
                .map(|h| {
                    json!({
                        "pid": h.pid,
                        "granted": h.granted,
                        "secs": h.since.elapsed().as_secs(),
                        "queue": h.queue,
                    })
                })
                .collect();
            (holders, leases.granted)
        };
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
            "in_flight": self.in_flight.load(Ordering::SeqCst),
            "stale": stale,
            "held_elsewhere": self.held_elsewhere.load(Ordering::Acquire),
            "lease_holders": leases,
            "leases_granted": leases_granted,
            "abandoned_recalls": cli_io::RECALL_SLOTS.live(),
            "panics": u64::from(self.state.get().is_some_and(WarmState::poisoned)),
            "reloads": self.stats.reloads.load(Ordering::Acquire),
            "last_reload": self.stats.last_reload(),
            "refused": self.stats.refused.load(Ordering::Acquire),
            "refused_by": self.stats.refused_by(),
            "ops": self.stats.ops(),
            "engines": engines,
            "graph": graph,
            "prewarm": self.stats.prewarm(),
            "queue": self.queue.report(),
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

/// How long a daemon starting waits for the lock while the daemon holding
/// it says nothing: it may be retiring, or done shutting down and about to
/// let go.
const QUIET_WAIT: Duration = Duration::from_secs(5);

/// Take the daemon lock, or find that another daemon has it. `None` means
/// another daemon answers, or still held the lock [`QUIET_WAIT`] after it
/// last said anything, or [`STOPPING_WAIT`] after this began while it said
/// it was stopping: this one exits.
fn elect(run: &RunDir) -> io::Result<Option<File>> {
    elect_within(run, QUIET_WAIT, STOPPING_WAIT)
}

/// [`elect`], with its two waits.
fn elect_within(run: &RunDir, quiet: Duration, stopping: Duration) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(run.lock())?;
    let started = Instant::now();
    // When the holder last answered or said it was shutting down. One that
    // is done records it stopped before it lets the lock go: the wait for
    // it then counts from there, not from when this began to wait.
    let mut heard = started;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(Some(file)),
            Err(fs::TryLockError::WouldBlock) => {
                // One that answers is serving; one that doesn't may be
                // retiring, so wait for it to let go, longer when it says
                // it is shutting down.
                let hello = proto::hello(run.socket(), Duration::from_millis(500)).ok();
                let stopping_name = Phase::Stopping.name();
                if hello
                    .as_ref()
                    .is_some_and(|hello| hello.state != stopping_name)
                {
                    return Ok(None);
                }
                if hello.is_some() || said_stopping(run) {
                    heard = Instant::now();
                    if started.elapsed() >= stopping {
                        return Ok(None);
                    }
                } else if heard.elapsed() >= quiet {
                    return Ok(None);
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(fs::TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// Whether the daemon's record says it is shutting down.
pub fn said_stopping(run: &RunDir) -> bool {
    fs::read(run.info())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<DaemonInfo>(&bytes).ok())
        .is_some_and(|info| info.state == Phase::Stopping.name())
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
pub fn run_daemon(
    db: &Path,
    idle_timeout: Duration,
    launch_path: Option<PathBuf>,
) -> anyhow::Result<i32> {
    let key = cli_io::db_key(db)?;
    anyhow::ensure!(key.is_file(), "no database at {}", key.display());
    // From here on, the database is this process's to open, not to lease.
    super::lease::serve_as_daemon(&key);
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
        reopen_wait: REOPEN_WAIT,
        drain_wait: DRAIN_WAIT,
        launch_path,
        process: true,
        drain_queue: true,
    };
    let server = match Server::start(key.clone(), run, opts) {
        Ok(server) => server,
        Err(_) => return Ok(crate::EXIT_ERROR),
    };
    let reason = server.serve();
    let finish = server.shutdown(&reason);
    // Jobs queued since the drain thread stopped go to a drainer of their
    // own, once the queue's lock is free.
    if !server.queue.holding() {
        super::drain::hand_back(&key);
    }
    match finish {
        Finish::Exit(code) => {
            drop(server);
            drop(lock);
            Ok(code)
        }
        Finish::Abandoned => {
            // Nothing is dropped: the handle stays open, and the lock held,
            // until the process is gone, so no daemon opens the database
            // before then and `daemon stop` waits for the exit.
            std::mem::forget(server);
            std::mem::forget(lock);
            std::process::exit(crate::EXIT_ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_io::tests::parity::{
        self, build_fixture, check_parity_with, matrix, one_shot_at, Case, Run,
    };
    use crate::daemon::client;
    use crate::daemon::lease::{self, Lease};
    use crate::daemon::proto::RunOutcome;

    fn options(idle_timeout: Duration) -> Options {
        Options {
            idle_timeout,
            open_wait: Duration::from_secs(5),
            reopen_wait: REOPEN_WAIT,
            drain_wait: DRAIN_WAIT,
            launch_path: None,
            process: false,
            drain_queue: false,
        }
    }

    fn start_with(db: &Path, opts: Options) -> Arc<Server> {
        let key = cli_io::db_key(db).unwrap();
        let run = RunDir::of(&key);
        run.create().unwrap();
        Server::start(key, run, opts).unwrap_or_else(|why| panic!("not started: {why}"))
    }

    fn start(db: &Path) -> Arc<Server> {
        start_with(db, options(Duration::from_secs(3600)))
    }

    fn stop(server: Arc<Server>) {
        let finish = server.shutdown(&StopReason::Requested("test".into()));
        assert_eq!(finish, Finish::Exit(0));
    }

    /// A client in `cwd` with no `AXIL_DB`, whose stdin is piped and holds
    /// `stdin`, empty by default, like the one-shot children's.
    fn send(server: &Server, args: &[&str], cwd: &Path, stdin: Option<&[u8]>) -> RunOutcome {
        let mut header = Identity::current()
            .unwrap()
            .header(parity::argv(args), cwd, true);
        header.env.clear();
        let stdin = stdin.unwrap_or_default();
        proto::run(
            server.run.socket(),
            &header,
            Some(stdin),
            Duration::from_secs(120),
        )
        .unwrap()
    }

    fn served(outcome: RunOutcome) -> Run {
        match outcome.reply {
            RunReply::Served { exit, .. } => (outcome.stdout, outcome.stderr, exit),
            other => panic!("not served: {other:?}"),
        }
    }

    fn unavailable(outcome: &RunOutcome) -> bool {
        matches!(&outcome.reply, RunReply::NotServed { reason, .. } if reason == "unavailable")
    }

    /// [`send`], again while the daemon reopens the database, as a client
    /// does.
    fn send_settled(server: &Server, args: &[&str], cwd: &Path) -> RunOutcome {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let outcome = send(server, args, cwd, None);
            if !unavailable(&outcome) || Instant::now() >= deadline {
                return outcome;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    /// Until the warm-up and any reload are done, so they hold nothing.
    fn quiet(server: &Server) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while server
            .background
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|handles| handles.iter().any(|h| !h.is_finished()))
        {
            assert!(Instant::now() < deadline, "the warm-up did not finish");
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn info_of(server: &Server) -> DaemonInfo {
        serde_json::from_slice(&fs::read(server.run.info()).unwrap()).unwrap()
    }

    /// The cli_io parity matrix sent through the socket of a daemon on the
    /// database, against the one-shot binary, both run in the project's
    /// directory. The daemon's run directory sits beside the database
    /// while it serves.
    #[test]
    fn served_output_matches_one_shot_through_the_socket() {
        let tmp = tempfile::tempdir().unwrap();
        let fixture = build_fixture(&tmp.path().join("fixture"));
        let proj = tmp.path().join("work/proj");
        let db_path = proj.join(".axil/memory.axil");
        let db = db_path.to_str().unwrap();
        let auth_rs = proj.join("src/auth.rs");
        let auth_rs = auth_rs.to_str().unwrap();
        // Refused before they run; exec's tests cover them.
        let refused = |c: &Case| c.cold_only || c.name == "clap unknown flag";
        let cases: Vec<Case> = matrix(db, auth_rs)
            .into_iter()
            .filter(|c| !refused(c))
            .collect();
        let cwd = || proj.canonicalize().unwrap();
        let results = check_parity_with(
            &fixture,
            &proj,
            &cases,
            |c, args| {
                let server = start(&db_path);
                let run = served(send(&server, args, &cwd(), c.stdin));
                stop(server);
                run
            },
            |c, args| one_shot_at(args, c.stdin, &cwd(), None),
        );
        let (_, boot) = results.iter().find(|(n, _)| *n == "boot json").unwrap();
        assert!(boot.2 == 0 && boot.0.starts_with(b"{"));
    }

    /// The same matrix sent as the CLI sends it ([`client::forward`]): the
    /// client's own command line, environment and stdin, and no deadline.
    #[test]
    fn forwarded_output_matches_one_shot() {
        let tmp = tempfile::tempdir().unwrap();
        let fixture = build_fixture(&tmp.path().join("fixture"));
        let proj = tmp.path().join("work/proj");
        let db_path = proj.join(".axil/memory.axil");
        let db = db_path.to_str().unwrap();
        let auth_rs = proj.join("src/auth.rs");
        let auth_rs = auth_rs.to_str().unwrap();
        let refused = |c: &Case| c.cold_only || c.name == "clap unknown flag";
        let cases: Vec<Case> = matrix(db, auth_rs)
            .into_iter()
            .filter(|c| !refused(c))
            .collect();
        let cwd = || proj.canonicalize().unwrap();
        let opts = client::Opts {
            slow_query_log: true,
            deadline: None,
            starting_wait: Duration::from_secs(10),
            until: None,
        };
        check_parity_with(
            &fixture,
            &proj,
            &cases,
            |c, args| {
                let server = start(&db_path);
                let stdin = Some(c.stdin.unwrap_or_default().to_vec());
                let forwarded = client::forward(&db_path, parity::argv(args), &cwd(), stdin, &opts);
                stop(server);
                match forwarded {
                    client::Forwarded::Served {
                        exit,
                        stdout,
                        stderr,
                        ..
                    } => (stdout, stderr, exit),
                    other => panic!("{}: not served: {other:?}", c.name),
                }
            },
            |c, args| one_shot_at(args, c.stdin, &cwd(), None),
        );
    }

    /// Sixteen recalls at once, with stores running beside them: every one
    /// is served, and every store is in the database afterwards. A request
    /// over the in-flight cap is answered `overloaded` and sent again.
    #[test]
    fn sixteen_recalls_beside_stores_are_all_served() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        let cwd = proj.canonicalize().unwrap();
        let overloaded = AtomicUsize::new(0);
        let retrying = |args: &[&str], stdin: Option<&[u8]>| loop {
            let outcome = send(&server, args, &cwd, stdin);
            if outcome.reply == RunReply::Overloaded {
                overloaded.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            // Nothing here reloads the handle; say what did, if anything.
            assert!(!unavailable(&outcome), "{:?}: {}", outcome.reply, {
                let status = server.status();
                json!({"stale": status["stale"], "last_reload": status["last_reload"],
                    "reloads": status["reloads"], "graph": status["graph"]})
            });
            return served(outcome);
        };
        let read: &[&str] = if cfg!(feature = "embed") {
            &["recall", "auth timeout"]
        } else {
            &["boot"]
        };
        let stored: Vec<(String, String)> = thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| {
                    for _ in 0..2 {
                        let (out, err, exit) = retrying(read, None);
                        assert!(
                            exit == 0 && !out.is_empty(),
                            "{}",
                            String::from_utf8_lossy(&err)
                        );
                    }
                });
            }
            let writers: Vec<_> = (0..4)
                .map(|w| {
                    let retrying = &retrying;
                    s.spawn(move || {
                        (0..4)
                            .map(|n| {
                                let summary = format!("socket store {w}-{n}");
                                let data = json!({ "summary": summary }).to_string();
                                let (out, err, exit) = if n % 2 == 0 {
                                    retrying(&["store", "decisions", &data], None)
                                } else {
                                    retrying(&["store", "decisions", "-"], Some(data.as_bytes()))
                                };
                                assert_eq!(exit, 0, "{}", String::from_utf8_lossy(&err));
                                let record: Value = serde_json::from_slice(&out).unwrap();
                                (record["id"].as_str().unwrap().to_string(), summary)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            writers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        let warm = server.state.get().unwrap().handle().unwrap();
        for (id, summary) in &stored {
            let id = axil_core::RecordId::from_string(id).unwrap();
            let record = warm.db().get(&id).unwrap().expect("stored");
            assert_eq!(record.data["summary"], summary.as_str());
        }
        drop(warm);
        assert_eq!(stored.len(), 16);
        let ops = server.stats.ops();
        assert_eq!(ops["store"].served, 16);
        eprintln!(
            "{} requests were over the cap and sent again",
            overloaded.load(Ordering::Relaxed)
        );
        stop(server);
    }

    /// A change the handle didn't make reloads it, and the request that
    /// found it is asked again: an engines edit in `axil.toml`, a backup
    /// renamed over the database, a dropped store, and a graph that failed
    /// to load.
    #[test]
    fn a_change_under_the_open_handle_reloads_it() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let backup = proj.join(".axil/backup.axil");
        fs::copy(&db, &backup).unwrap();
        let server = start(&db);
        let cwd = proj.canonicalize().unwrap();
        let reloads = || server.stats.reloads.load(Ordering::Acquire);
        let after_change = || thread::sleep(FRESH_FOR + Duration::from_millis(50));

        let (_, _, exit) = served(send(&server, &["boot"], &cwd, None));
        assert_eq!((exit, reloads()), (0, 0));

        fs::write(proj.join("axil.toml"), "[engines]\ndisabled = [\"ts\"]\n").unwrap();
        after_change();
        let since = send(&server, &["since", "1h"], &cwd, None);
        assert!(unavailable(&since), "{since:?}");
        let since = send_settled(&server, &["since", "1h"], &cwd);
        assert_eq!(reloads(), 1);
        assert!(server
            .stats
            .last_reload()
            .unwrap()
            .ends_with("axil.toml changed"));
        if cfg!(feature = "timeseries") {
            assert!(
                matches!(&since.reply, RunReply::NotServed { reason, .. } if reason == "lacks"),
                "{since:?}"
            );
        }

        fs::rename(&backup, &db).unwrap();
        after_change();
        served(send_settled(&server, &["boot"], &cwd));
        assert_eq!(reloads(), 2);
        assert_eq!(
            server.stats.last_reload().as_deref(),
            Some("the database was replaced")
        );

        // A build without the store or the graph has neither to lose.
        let mut expected = 2;
        if cfg!(feature = "fts") {
            axil_core::drop_engine_companion(&db, "fts").unwrap();
            after_change();
            let fts = send_settled(&server, &["fts", "cache"], &cwd);
            expected += 1;
            assert_eq!(reloads(), expected);
            assert!(
                matches!(&fts.reply, RunReply::NotServed { reason, .. } if reason == "lacks"),
                "{fts:?}"
            );
        }
        if cfg!(feature = "graph") {
            server.fail_graph.store(true, Ordering::Release);
            served(send_settled(&server, &["boot"], &cwd));
            expected += 1;
            assert_eq!(reloads(), expected);
            assert_eq!(
                server.stats.last_reload().as_deref(),
                Some("the graph failed to load")
            );
        }
        assert_eq!(server.phase.get(), Phase::Ready);
        stop(server);
    }

    /// A database replaced under a handle that requests still hold: the
    /// reload is put off, and until one succeeds nothing runs on the old
    /// handle, a write least of all, not even in the 250 ms a clean check
    /// stands for. A request is not held up while the reload waits for the
    /// old handle. Once it is let go, the write lands in the database on
    /// disk.
    #[test]
    fn nothing_runs_on_a_handle_known_to_be_stale() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let backup = proj.join(".axil/backup.axil");
        fs::copy(&db, &backup).unwrap();
        let opts = Options {
            reopen_wait: Duration::from_millis(200),
            ..options(Duration::from_secs(3600))
        };
        let server = start_with(&db, opts);
        quiet(&server);
        let cwd = proj.canonicalize().unwrap();
        let reloads = || server.stats.reloads.load(Ordering::Acquire);

        // A request still running on the old handle.
        let held = server.state.get().unwrap().handle().unwrap();
        fs::rename(&backup, &db).unwrap();
        thread::sleep(FRESH_FOR + Duration::from_millis(50));
        let data = json!({ "summary": "written while stale" }).to_string();
        let store = ["store", "decisions", data.as_str()];
        let first = send(&server, &store, &cwd, None);
        assert!(unavailable(&first), "{first:?}");
        let again = send(&server, &store, &cwd, None);
        assert!(unavailable(&again), "{again:?}");
        let read = send(&server, &["boot"], &cwd, None);
        assert!(unavailable(&read), "{read:?}");
        assert_eq!(reloads(), 0);

        let mut header = Identity::current()
            .unwrap()
            .header(parity::argv(&store), &cwd, false);
        header.env.clear();
        header.deadline_ms = Some(epoch_ms() + 10_000);
        let late = proto::run(server.run.socket(), &header, None, Duration::from_secs(30)).unwrap();
        assert!(unavailable(&late), "{late:?}");

        drop(held);
        let (out, err, exit) = served(send_settled(&server, &store, &cwd));
        assert_eq!(exit, 0, "{}", String::from_utf8_lossy(&err));
        assert_eq!(reloads(), 1);
        let record: Value = serde_json::from_slice(&out).unwrap();
        let id = axil_core::RecordId::from_string(record["id"].as_str().unwrap()).unwrap();
        let warm = server.state.get().unwrap().handle().unwrap();
        assert!(warm.db().get(&id).unwrap().is_some());
        drop(warm);
        stop(server);
    }

    /// A backup copied over the database in place, its inode kept: the
    /// daemon serves nothing more and stops without closing the handle,
    /// which would write its view of the file over the copy.
    #[test]
    fn a_store_written_in_place_is_left_unclosed() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let backup = fs::read(&db).unwrap();
        let server = start(&db);
        quiet(&server);
        let cwd = proj.canonicalize().unwrap();
        let data = json!({ "summary": "after the backup" }).to_string();
        let (_, err, exit) = served(send(&server, &["store", "decisions", &data], &cwd, None));
        assert_eq!(exit, 0, "{}", String::from_utf8_lossy(&err));
        served(send(&server, &["boot"], &cwd, None));

        {
            use std::io::Write;
            let mut file = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&db)
                .unwrap();
            file.write_all(&backup).unwrap();
        }
        thread::sleep(FRESH_FOR + Duration::from_millis(50));
        let after = send(&server, &["boot"], &cwd, None);
        assert!(unavailable(&after), "{after:?}");
        let reason = server.serve();
        assert!(
            matches!(&reason, StopReason::Overwritten(why) if why.starts_with("the database")),
            "{reason:?}"
        );
        assert_eq!(server.shutdown(&reason), Finish::Abandoned);
        assert_eq!(
            fs::read(&db).unwrap(),
            backup,
            "the copy is left as written"
        );
        let info = info_of(&server);
        assert_eq!(
            (info.state.as_str(), info.closed_cleanly),
            ("stopped", Some(false))
        );
        // As the daemon process does: exit without dropping it.
        std::mem::forget(server);
    }

    /// Shutting down waits for the background work on the handle, then
    /// closes it. A handle something still holds when the waits run out is
    /// left open, and the record says so.
    #[test]
    fn shutting_down_waits_for_the_background_work() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");

        let server = start(&db);
        quiet(&server);
        server.spawn_background("test-hold", |server| {
            let _busy = server.busy();
            let held = server.state.get().and_then(WarmState::handle);
            thread::sleep(Duration::from_millis(500));
            drop(held);
        });
        let started = Instant::now();
        stop(server.clone());
        assert!(started.elapsed() >= Duration::from_millis(400));
        assert_eq!(info_of(&server).closed_cleanly, Some(true));
        drop(server);

        let opts = Options {
            drain_wait: Duration::from_millis(100),
            ..options(Duration::from_secs(3600))
        };
        let server = start_with(&db, opts);
        quiet(&server);
        let held = server.state.get().unwrap().handle().unwrap();
        let reason = StopReason::Requested("test".into());
        assert_eq!(server.shutdown(&reason), Finish::Abandoned);
        let info = info_of(&server);
        assert_eq!(
            (info.state.as_str(), info.closed_cleanly),
            ("stopped", Some(false))
        );
        drop(held);
        std::mem::forget(server);
    }

    /// The warm-up panicking leaves the handle as a panicked request does:
    /// never used or closed again.
    #[test]
    fn a_warm_up_that_panics_poisons_the_handle() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        quiet(&server);
        let cwd = proj.canonicalize().unwrap();
        // A reload warms up again.
        server.panic_prewarm.store(true, Ordering::Release);
        server.fail_graph.store(true, Ordering::Release);
        let _ = send(&server, &["boot"], &cwd, None);
        assert_eq!(server.serve(), StopReason::Panicked);
        assert!(server.state.get().unwrap().poisoned());
        assert_eq!(server.shutdown(&StopReason::Panicked), Finish::Abandoned);
        std::mem::forget(server);
    }

    /// With every connection taken by clients that send nothing, `hello`
    /// and `shutdown` are still answered, so `daemon stop` works; a run is
    /// told to come back.
    #[test]
    fn the_frozen_ops_are_answered_with_every_connection_taken() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        let socket = server.run.socket().to_path_buf();
        let idle: Vec<UnixStream> = (0..MAX_CONNECTIONS)
            .map(|_| UnixStream::connect(&socket).unwrap())
            .collect();
        let deadline = Instant::now() + Duration::from_secs(10);
        while server.connections.load(Ordering::Acquire) < MAX_CONNECTIONS {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(10));
        }
        let hello = proto::hello(&socket, Duration::from_secs(5)).unwrap();
        assert_eq!(hello.state, "ready");
        let cwd = proj.canonicalize().unwrap();
        let header = Identity::current()
            .unwrap()
            .header(parity::argv(&["boot"]), &cwd, false);
        let run = proto::run(&socket, &header, None, Duration::from_secs(5)).unwrap();
        assert_eq!(run.reply, RunReply::Overloaded);
        let ack: Ack = proto::call(
            &socket,
            &Op::Shutdown {
                reason: "full".into(),
            },
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(ack.ok);
        assert_eq!(server.serve(), StopReason::Requested("full".into()));
        drop(idle);
        stop(server);
    }

    /// What a request is refused for, and the ops beside `run`.
    #[test]
    fn requests_that_must_not_run_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        let cwd = proj.canonicalize().unwrap();
        let id = Identity::current().unwrap();
        let header = id.header(parity::argv(&["boot"]), &cwd, false);
        let socket = server.run.socket().to_path_buf();
        let reply = |header: RunHeader| {
            proto::run(&socket, &header, None, Duration::from_secs(30))
                .unwrap()
                .reply
        };
        let incompatible = |reason: &str| RunReply::Incompatible {
            reason: reason.into(),
        };

        assert!(matches!(
            reply(header.clone()),
            RunReply::Served { exit: 0, .. }
        ));
        let mut other = header.clone();
        other.build_id = "0000000000000000".into();
        assert_eq!(reply(other), incompatible("build"));
        let mut other = header.clone();
        other.env_fp = "0000000000000000".into();
        assert_eq!(reply(other), incompatible("env"));
        let mut other = header.clone();
        other.features.push("not-a-feature".into());
        assert_eq!(reply(other), incompatible("features"));
        let mut other = header.clone();
        other.proto = DAEMON_PROTO + 1;
        assert_eq!(reply(other), incompatible("proto"));
        let mut other = header.clone();
        other.deadline_ms = Some(1);
        assert_eq!(reply(other), RunReply::Expired);
        let mut other = header.clone();
        other.argv = parity::argv(&["--db", "/elsewhere/.axil/memory.axil", "boot"]);
        assert!(matches!(
            reply(other),
            RunReply::NotServed { reason, .. } if reason == "wrong_db"
        ));

        let hello = proto::hello(&socket, Duration::from_secs(5)).unwrap();
        assert_eq!(
            (hello.op.as_str(), hello.proto, hello.state.as_str()),
            ("hello", DAEMON_PROTO, "ready")
        );
        assert_eq!(Path::new(&hello.db), server.key);
        assert_eq!(hello.build_id, id.build_id);
        assert_eq!(hello.pid, std::process::id());
        let status: Value = proto::call(&socket, &Op::Status, Duration::from_secs(5)).unwrap();
        assert_eq!(status["state"], "ready");
        assert_eq!(status["ops"]["boot"]["served"], 1);
        assert_eq!(status["refused"], 5);
        assert_eq!(status["lease_holders"], json!([]));
        assert_eq!(status["leases_granted"], 0);

        let mut garbage = UnixStream::connect(&socket).unwrap();
        write_frame(&mut garbage, b"{\"op\":\"nonsense\"}").unwrap();
        let answer: RunReply = read_json(&mut garbage).unwrap();
        assert!(matches!(answer, RunReply::Error { .. }), "{answer:?}");

        let ack: Ack = proto::call(
            &socket,
            &Op::Shutdown {
                reason: "done".into(),
            },
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(ack.ok);
        assert_eq!(server.serve(), StopReason::Requested("done".into()));
        stop(server);
        assert!(!socket.exists(), "the socket goes with the daemon");
    }

    #[test]
    fn requests_over_the_cap_are_not_admitted() {
        let count = AtomicUsize::new(0);
        let first = Admitted::new(&count, 2).unwrap();
        let second = Admitted::new(&count, 2).unwrap();
        assert!(Admitted::new(&count, 2).is_none());
        drop(first);
        let third = Admitted::new(&count, 2);
        assert!(third.is_some());
        drop((second, third));
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    /// An idle daemon stops by itself, leaves every store closed cleanly
    /// and says how it ended. (Signals are the process's: tests/daemon.rs
    /// signals a daemon process.)
    #[test]
    fn an_idle_daemon_stops_and_closes_its_stores() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let closed_cleanly = || {
            for suffix in ["", ".graph", ".ts", ".vec"] {
                let path = axil_core::companion_path(&db, suffix);
                if path.exists() {
                    redb::ReadOnlyDatabase::open(&path)
                        .unwrap_or_else(|e| panic!("{} was left open: {e}", path.display()));
                }
            }
        };

        let server = start_with(&db, options(Duration::from_secs(1)));
        let started = Instant::now();
        assert_eq!(server.serve(), StopReason::Idle);
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(server.shutdown(&StopReason::Idle), Finish::Exit(0));
        closed_cleanly();
        let info = info_of(&server);
        assert_eq!(info.state, "stopped");
        assert_eq!(info.exit_reason.as_deref(), Some("idle"));
        assert_eq!(info.closed_cleanly, Some(true));
    }

    /// Only a command it runs keeps a daemon from idling out: neither the
    /// requests it refuses nor the leases of the clients it refuses, and
    /// the reopen after each.
    #[test]
    fn a_daemon_no_client_can_use_still_idles_out() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let server = start_with(&db, options(Duration::from_secs(2)));
        let cwd = proj.canonicalize().unwrap();
        let done = AtomicBool::new(false);
        let started = Instant::now();
        thread::scope(|s| {
            s.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    let mut header =
                        Identity::current()
                            .unwrap()
                            .header(parity::argv(&["boot"]), &cwd, false);
                    header.build_id = "another build".into();
                    let _ = proto::run(server.run.socket(), &header, None, Duration::from_secs(5));
                    if let Ok(mut stream) =
                        proto::connect(server.run.socket(), Duration::from_secs(5))
                    {
                        let op = Op::Lease {
                            db: db.display().to_string(),
                            wait_ms: 5_000,
                            pid: std::process::id(),
                        };
                        let _ = write_json(&mut stream, &op)
                            .and_then(|()| read_json::<LeaseReply>(&mut stream));
                    }
                    thread::sleep(Duration::from_millis(300));
                }
            });
            assert_eq!(server.serve(), StopReason::Idle);
            done.store(true, Ordering::Release);
        });
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(server.shutdown(&StopReason::Idle), Finish::Exit(0));
    }

    /// A machine that slept through the idle timeout: the monotonic clock
    /// says a moment passed, the wall clock says hours did.
    #[test]
    fn idle_time_counts_the_time_the_machine_slept() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start_with(
            &proj.join(".axil/memory.axil"),
            options(Duration::from_secs(60)),
        );
        assert_eq!(server.tick(), None);
        let hours_ago = SystemTime::now() - Duration::from_secs(7200);
        *server.last_active.lock().unwrap() = (Instant::now(), hours_ago);
        assert_eq!(server.tick(), Some(StopReason::Idle));
        stop(server);
    }

    /// Ask `server` for a lease with `db` and `wait_ms`, and return its
    /// answer with the connection that holds the lease.
    fn ask(server: &Server, db: &Path, wait_ms: u64) -> (LeaseReply, UnixStream) {
        let mut stream = proto::connect(server.run.socket(), Duration::from_secs(30)).unwrap();
        let op = Op::Lease {
            db: db.display().to_string(),
            wait_ms,
            pid: std::process::id(),
        };
        write_json(&mut stream, &op).unwrap();
        (read_json(&mut stream).unwrap(), stream)
    }

    fn wait_for_phase(server: &Server, phase: Phase) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while server.phase.get() != phase {
            assert!(
                Instant::now() < deadline,
                "still {:?}, not {phase:?}",
                server.phase.get()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn holders(server: &Server) -> usize {
        server.status()["lease_holders"].as_array().unwrap().len()
    }

    /// A lease closes the database, so another process can open it, and
    /// keeps it closed until the last holder lets go; then the daemon opens
    /// it again and serves.
    #[test]
    fn a_lease_closes_the_database_until_the_last_holder_lets_go() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let server = start(&db);
        quiet(&server);

        let Lease::Held(first) = lease::lease(&server.key, Duration::from_secs(10)) else {
            panic!("not granted");
        };
        assert_eq!(server.phase.get(), Phase::Leased);
        assert!(server.state.get().unwrap().handle().is_none());
        assert_eq!(info_of(&server).state, "leased");
        assert_eq!(server.hello().state, "leased");
        // The holder can open it, as no process could while the daemon
        // had it open.
        drop(axil_core::Axil::open(&db).build().unwrap());
        let outcome = send(&server, &["boot"], &proj, None);
        assert_eq!(outcome.reply, RunReply::Leased);
        // A client of the protocol from before `leased` is told only that
        // it does not match.
        let mut header = Identity::current()
            .unwrap()
            .header(parity::argv(&["boot"]), &proj, true);
        header.proto = DAEMON_PROTO - 1;
        let old = proto::run(
            server.run.socket(),
            &header,
            Some(b""),
            Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(
            old.reply,
            RunReply::Incompatible {
                reason: "proto".into()
            }
        );

        // A second holder is granted at once.
        let (reply, second) = ask(&server, &server.key, 1);
        assert_eq!(reply, LeaseReply::granted());
        assert_eq!(holders(&server), 2);
        drop(second);
        let deadline = Instant::now() + Duration::from_secs(10);
        while holders(&server) > 1 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(10));
        }
        thread::sleep(UNLEASE_DELAY * 3);
        assert_eq!(server.phase.get(), Phase::Leased, "reopened under a lease");

        // A lease asked for while the reopen waits calls it off.
        drop(first);
        let (reply, third) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        thread::sleep(UNLEASE_DELAY * 3);
        assert_eq!(server.phase.get(), Phase::Leased, "reopened under a lease");

        drop(third);
        wait_for_phase(&server, Phase::Ready);
        assert_eq!(info_of(&server).state, "ready");
        served(send(&server, &["boot"], &proj, None));
        let status = server.status();
        assert_eq!(status["leases_granted"], 3);
        assert_eq!(status["reloads"], 0);
        stop(server);
    }

    /// A lease the requests running on the database outlast is refused,
    /// and the daemon serves on; one that waits long enough gets it.
    #[test]
    fn a_lease_waits_for_the_running_requests_or_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let server = start(&db);
        quiet(&server);
        // Taken here, so it is held before the lease is asked for.
        let hold = |server: &Arc<Server>, ms: u64| {
            let held = server.state.get().and_then(WarmState::handle).unwrap();
            server.spawn_background("test-hold", move |server| {
                let _busy = server.busy();
                thread::sleep(Duration::from_millis(ms));
                drop(held);
            });
        };

        hold(&server, 1500);
        let (reply, refused) = ask(&server, &server.key, 200);
        assert_eq!(reply, LeaseReply::refused("lease_timeout"));
        drop(refused);
        assert_eq!(server.phase.get(), Phase::Ready);
        quiet(&server);
        served(send(&server, &["boot"], &proj, None));

        hold(&server, 500);
        let started = Instant::now();
        let (reply, held) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        assert!(started.elapsed() >= Duration::from_millis(300));
        drop(held);
        wait_for_phase(&server, Phase::Ready);

        // Another database's lease is not this daemon's to grant.
        let (reply, _) = ask(&server, &proj.join("other.axil"), 10_000);
        assert_eq!(reply, LeaseReply::refused("wrong_db"));
        assert_eq!(server.phase.get(), Phase::Ready);
        stop(server);
    }

    /// A process with the database open but no lease (a build from before
    /// leases, say) when the last lease ends, or when a change has the
    /// daemon reopen it: the daemon waits for it for as long as it holds the
    /// database, past the open wait, and survives. Meanwhile a command is
    /// answered `leased`, so it runs in its own process at once, and a lease
    /// is granted at once. The holder's writes are not taken for an
    /// overwrite.
    #[test]
    fn a_holder_without_a_lease_is_waited_for_and_commands_run_elsewhere() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let backup = proj.join(".axil/backup.axil");
        fs::copy(&db, &backup).unwrap();
        let opts = Options {
            open_wait: Duration::from_millis(300),
            ..options(Duration::from_secs(3600))
        };
        let open_wait = opts.open_wait;
        let server = start_with(&db, opts);
        quiet(&server);
        let held_elsewhere = || {
            let deadline = Instant::now() + Duration::from_secs(30);
            while !server.held_elsewhere.load(Ordering::Acquire) {
                assert!(Instant::now() < deadline, "{}", server.status());
                thread::sleep(Duration::from_millis(10));
            }
        };
        let leased = |args: &[&str]| send(&server, args, &proj, None).reply == RunReply::Leased;

        let (reply, held) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        let holder = axil_core::Axil::open(&db).build().unwrap();
        holder
            .insert("decisions", json!({"summary": "written under the lease"}))
            .unwrap();
        drop(held);
        held_elsewhere();
        assert!(leased(&["boot"]));
        let started = Instant::now();
        let (reply, other) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(other);
        thread::sleep(UNLEASE_DELAY + open_wait * 3);
        assert_eq!(server.stop_reason(), None);
        assert_eq!(server.phase.get(), Phase::Reloading);
        assert_eq!(server.status()["held_elsewhere"], true);
        assert!(leased(&["boot"]));
        assert!(!server.state.get().unwrap().poisoned());
        drop(holder);
        wait_for_phase(&server, Phase::Ready);
        assert_eq!(server.status()["held_elsewhere"], false);
        let (out, _, exit) = served(send(&server, &["boot"], &proj, None));
        assert_eq!(exit, 0);
        assert!(String::from_utf8_lossy(&out).contains("written under the lease"));

        // A backup renamed over the database, which another process opens
        // before the daemon reopens it.
        fs::rename(&backup, &db).unwrap();
        let holder = axil_core::Axil::open(&db).build().unwrap();
        thread::sleep(FRESH_FOR + Duration::from_millis(50));
        let first = send(&server, &["boot"], &proj, None);
        assert!(unavailable(&first), "{first:?}");
        held_elsewhere();
        assert!(leased(&["boot"]));
        let (reply, other) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        drop(other);
        thread::sleep(UNLEASE_DELAY + open_wait * 3);
        assert_eq!(server.stop_reason(), None);
        drop(holder);
        wait_for_phase(&server, Phase::Ready);
        served(send(&server, &["boot"], &proj, None));
        stop(server);
    }

    /// A daemon starting while another shuts down takes the lock once the
    /// other lets go, also when the other records that it stopped a while
    /// before it does, and shut down for longer than a quiet holder is
    /// waited for: the wait for it counts from when it last said anything.
    #[test]
    fn a_daemon_starting_waits_for_one_that_just_stopped() {
        let tmp = tempfile::Builder::new().prefix("e").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        fs::write(&key, b"").unwrap();
        let run = RunDir::of(&key);
        run.create().unwrap();
        let held = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(run.lock())
            .unwrap();
        held.try_lock().unwrap();
        let record = |state: &str| {
            let info = DaemonInfo {
                state: state.into(),
                ..DaemonInfo::default()
            };
            write_atomic(&run.info(), &serde_json::to_vec(&info).unwrap()).unwrap();
        };
        record(Phase::Stopping.name());
        let quiet = Duration::from_millis(300);
        thread::scope(|s| {
            let electing = s.spawn(|| elect_within(&run, quiet, Duration::from_secs(30)));
            thread::sleep(quiet * 3);
            record("stopped");
            thread::sleep(quiet / 2);
            drop(held);
            assert!(electing.join().unwrap().unwrap().is_some());
        });

        // One that never says anything is waited for no longer than that.
        let held = OpenOptions::new()
            .read(true)
            .write(true)
            .open(run.lock())
            .unwrap();
        // A process another test starts keeps a copy of every handle open
        // here, the lock just let go of included, until it execs.
        let deadline = Instant::now() + Duration::from_secs(5);
        while let Err(e) = held.try_lock() {
            assert!(Instant::now() < deadline, "{e}");
            thread::sleep(Duration::from_millis(10));
        }
        let started = Instant::now();
        assert!(elect_within(&run, quiet, Duration::from_secs(30))
            .unwrap()
            .is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// A client that goes away before its answer, with the command sent or
    /// halfway through sending it, leaves the daemon serving; a write it
    /// sent whole runs once.
    #[test]
    fn a_client_that_leaves_early_leaves_the_daemon_serving() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        let cwd = proj.canonicalize().unwrap();
        let identity = Identity::current().unwrap();
        let store = parity::argv(&["store", "decisions", r#"{"summary":"left early"}"#]);
        let header = identity.header(store, &cwd, false);
        let mut stream = proto::connect(server.run.socket(), Duration::from_secs(10)).unwrap();
        write_json(&mut stream, &Op::Run(header)).unwrap();
        drop(stream);
        let piped = parity::argv(&["store", "context", "-"]);
        let header = identity.header(piped, &cwd, true);
        let mut stream = proto::connect(server.run.socket(), Duration::from_secs(10)).unwrap();
        write_json(&mut stream, &Op::Run(header)).unwrap();
        std::io::Write::write_all(&mut stream, &64u32.to_be_bytes()).unwrap();
        drop(stream);

        let deadline = Instant::now() + Duration::from_secs(30);
        while server.stats.ops().get("store").map_or(0, |s| s.served) < 1 {
            assert!(Instant::now() < deadline, "the store was not run");
            thread::sleep(Duration::from_millis(10));
        }
        let (out, _, exit) = served(send(&server, &["boot"], &cwd, None));
        assert_eq!(exit, 0);
        assert_eq!(
            String::from_utf8_lossy(&out).matches("left early").count(),
            1
        );
        assert_eq!(server.stats.ops()["store"].served, 1);
        stop(server);
    }

    /// While a lease is held a request is told so; once the last ended, it
    /// is told the database is opening, as at the start (`unavailable`
    /// once the open has begun), so a client waits for the daemon rather
    /// than lease the database again.
    #[test]
    fn after_the_last_lease_a_request_is_told_the_database_is_opening() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        quiet(&server);
        let (reply, held) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        assert_eq!(
            send(&server, &["boot"], &proj, None).reply,
            RunReply::Leased
        );
        drop(held);
        let deadline = Instant::now() + Duration::from_secs(10);
        while holders(&server) > 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        let outcome = send(&server, &["boot"], &proj, None);
        assert!(
            outcome.reply == RunReply::Starting || unavailable(&outcome),
            "{:?}",
            outcome.reply
        );
        wait_for_phase(&server, Phase::Ready);
        let (_, _, exit) = served(send(&server, &["boot"], &proj, None));
        assert_eq!(exit, 0);
        stop(server);
    }

    /// A lease refused while the database opens again after another (its
    /// wait ran out) does not leave the database closed for nobody.
    #[test]
    fn a_lease_refused_during_the_reopen_leaves_it_to_finish() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let server = start(&db);
        quiet(&server);
        let (reply, held) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        // The reopen waits for the holder, which keeps the database open.
        let holder = axil_core::Axil::open(&db).build().unwrap();
        drop(held);
        wait_for_phase(&server, Phase::Reloading);
        let (reply, other) = ask(&server, &server.key, 0);
        assert!(
            reply == LeaseReply::refused("lease_timeout") || reply == LeaseReply::granted(),
            "{reply:?}"
        );
        drop(other);
        thread::sleep(UNLEASE_DELAY * 2);
        drop(holder);
        wait_for_phase(&server, Phase::Ready);
        assert_eq!(holders(&server), 0);
        served(send(&server, &["boot"], &proj, None));
        stop(server);
    }

    /// The record of a phase written ahead of it never lands over a
    /// shutdown that began meanwhile.
    #[test]
    fn a_phase_recorded_ahead_never_overwrites_stopping() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        quiet(&server);
        assert!(server.phase.advance(Phase::Ready, Phase::Reloading));
        assert!(server.write_info_while(Phase::Reloading, "ready"));
        assert_eq!(info_of(&server).state, "ready");
        server.phase.set(Phase::Stopping);
        server.write_info_as("stopping");
        assert!(!server.write_info_while(Phase::Reloading, "ready"));
        assert_eq!(info_of(&server).state, "stopping");
        server.write_info();
        assert_eq!(info_of(&server).state, "stopping");
        stop(server);
    }

    /// A warm-up put off for a lease runs once that lease is refused: the
    /// daemon serves on, and should not pay the embedder's load on its next
    /// recall.
    #[test]
    fn a_warm_up_put_off_for_a_refused_lease_runs_after_it() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        quiet(&server);
        let id = server.ask_lease(std::process::id(), false);
        server.prewarm();
        assert!(server.unwarmed.load(Ordering::Acquire));
        assert!(server
            .background
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .iter()
            .all(|h| h.is_finished()));
        server.end_lease(id);
        assert!(!server.unwarmed.load(Ordering::Acquire));
        quiet(&server);
        stop(server);
    }

    /// A daemon shut down while leased has nothing left to close.
    #[test]
    fn a_daemon_stopped_while_leased_exits_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let server = start(&proj.join(".axil/memory.axil"));
        quiet(&server);
        let (reply, held) = ask(&server, &server.key, 10_000);
        assert_eq!(reply, LeaseReply::granted());
        stop(server.clone());
        assert_eq!(info_of(&server).closed_cleanly, Some(true));
        drop(held);
    }
}
