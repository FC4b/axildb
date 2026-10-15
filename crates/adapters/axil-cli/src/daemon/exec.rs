//! Run one client's command line in-process, against the database the host
//! keeps open, and return what the one-shot `axil` would have printed and
//! exited with in the client's own process.
//!
//! [`execute_captured`] parses the command line as `axil` does, finds the
//! database the client's process would have found, and runs the command
//! through [`dispatch_command`](crate::dispatch_command) with its stdio
//! captured ([`cli_io::capture`]) and the open handle installed
//! ([`cli_io::hot_db`]). It serves only the commands [`warm_route`] accepts,
//! on the database it holds. Anything else comes back as [`NotServed`]
//! before anything is printed or written, and the client runs it itself.
//!
//! What the client would see differently from a one-shot run:
//! - A diagnostic an engine prints once per process (a config warning, a
//!   verbose-mode banner, the embedder's execution-provider line) prints once
//!   in the host's life: in the first request that triggers it, or on the
//!   host's stderr when the host's own open triggers it.
//! - A database whose open prints anything (an unloadable or corrupt row, a
//!   config warning, a migration's note) is not served at all:
//!   [`WarmState::open`] fails with [`OpenPrinted`], and a reopen that
//!   prints leaves no handle. Each command then runs in its own process and
//!   prints the line, with its hint, every time, until the cause is fixed.
//!   Replaying the lines into each response instead would not match: each
//!   command's one-shot open attaches only the engines it needs, and prints
//!   only theirs.
//! - A vector store a killed writer left needing repair is repaired by the
//!   host's open, which says so on the host's stderr; the one-shot binary
//!   says so in the first command after the crash.
//! - A line an engine prints the first time a handle uses something (the
//!   graph store when its adjacency tables can't be used, say) prints in
//!   the first request that does, not in every one.
//! - Lines printed inside recall's ranking (`query.rs` and `Axil::recall`)
//!   and by the recall workers a `--timeout-ms` deadline leaves running go
//!   to the host's stderr: those threads carry no request.
//! - A command that panics exits 1 with one `axil: the command panicked`
//!   line on stderr, not the panic hook's message. The host then serves
//!   nothing more and must exit; see [`WarmState::poisoned`].
//! - Help and usage errors render as for a pipe, without color. They are
//!   served only to a client whose [`CLI_ENV`] variables match the host's
//!   (unset, once the host has removed its own), since clap prints them
//!   differently with a variable set.
//!
//! The process-wide setup a one-shot run does first (the vector exact-scan
//! ceiling, the encryption cipher) is the host's to do once, and so is
//! removing its own [`CLI_ENV`] variables, before the first request: the
//! request carries the client's.

use std::any::Any;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError, RwLock};
use std::time::Duration;

use axil_core::Axil;
use clap::error::ErrorKind;
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches};

use crate::cli_io::{self, Need, ReqCtx, WarmDb, WarmLacks};
use crate::{Cli, Command, Output};

/// The stack a request runs on: `main`'s. In debug builds clap's parser for
/// the whole command set overflows a default thread stack.
const REQUEST_STACK: usize = 32 * 1024 * 1024;

/// Every environment variable the command line reads through clap (an
/// argument's `env =`). clap reads each one as it makes the argument, so
/// the host's parser keeps the values the host had then, and a request
/// carries the client's.
pub const CLI_ENV: &[&str] = &["AXIL_DB", "AXIL_MCP_TOOLS"];

/// One command line, as the client's process was asked to run it.
pub struct Request {
    /// The command line, program name first.
    pub argv: Vec<OsString>,
    /// The client's working directory, absolute. A relative `--db` or
    /// `AXIL_DB` resolves against it, and with neither the database is the
    /// `.axil/memory.axil` found walking up from it.
    pub cwd: PathBuf,
    /// The client's values of the [`CLI_ENV`] variables it has set; one
    /// missing here is unset. The host's own are never consulted.
    pub env: Vec<(OsString, OsString)>,
    /// Piped stdin. `None` stands for a terminal: nothing was piped.
    pub stdin: Option<Vec<u8>>,
    /// Whether the command may record a slow query in the database's log.
    pub slow_query_log: bool,
    /// A step of a job the hooks queued, run by the host's drain thread. A
    /// write a client sent goes ahead of the queue's between its steps
    /// (see [`WarmState::writers_waiting`]).
    pub queued: bool,
}

impl Request {
    /// The client's value of the environment variable `name`.
    fn env(&self, name: &str) -> Option<&OsStr> {
        self.env
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value.as_os_str())
    }
}

/// What a served command printed, and the code it exited with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// The process exit code.
    pub exit: i32,
    /// Bytes for stdout.
    pub stdout: Vec<u8>,
    /// Bytes for stderr.
    pub stderr: Vec<u8>,
    /// The order they were printed in ([`cli_io::Captured::order`]).
    pub order: Vec<(bool, usize)>,
}

impl Response {
    fn of(exit: i32, printed: cli_io::Captured) -> Response {
        Response {
            exit,
            stdout: printed.out,
            stderr: printed.err,
            order: printed.order,
        }
    }
}

/// Why the host did not run a request. It printed and wrote nothing, so the
/// client can run the command in its own process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotServed {
    /// The command line holds an argument or subcommand this build doesn't
    /// know, or it would parse differently in the client's process: it
    /// doesn't parse, or the command reads a [`CLI_ENV`] variable other than
    /// `AXIL_DB`, and the client's [`CLI_ENV`] values differ from the ones
    /// the host's parser was made with (an empty `AXIL_DB` is itself a usage
    /// error).
    Parse,
    /// The command is for another database than the one held, or for none.
    WrongDb,
    /// The command is not one the host runs in-process.
    Command,
    /// The open handle lacks what the command needs: the command's own open
    /// would attach or create it.
    Lacks(Need),
    /// The command reads stdin and the client's is a terminal: the one-shot
    /// binary would wait there for the user to type the input.
    Terminal,
    /// The host has no handle to run it on: a reopen is running or failed,
    /// a request panicked ([`WarmState::poisoned`]), the host's gate found
    /// the handle stale, or the request's thread couldn't start.
    Unavailable,
    /// The request's deadline passed before the command could start.
    Expired,
    /// The host let go of the database for a process that leased it.
    Leased,
}

impl fmt::Display for NotServed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotServed::Parse => f.write_str("the command line does not parse here"),
            NotServed::WrongDb => f.write_str("the command is for another database"),
            NotServed::Command => f.write_str("the command is not run in-process"),
            NotServed::Lacks(need) => WarmLacks(*need).fmt(f),
            NotServed::Terminal => f.write_str("the command reads stdin from a terminal"),
            NotServed::Unavailable => f.write_str("the database is not open"),
            NotServed::Expired => f.write_str("the request's deadline passed before it ran"),
            NotServed::Leased => f.write_str("the database is leased to another process"),
        }
    }
}

impl std::error::Error for NotServed {}

/// The database a host keeps open, shared by every request it runs.
pub struct WarmState {
    /// The open handle. `None` while a reopen waits for the old one to be
    /// released, and after a reopen that failed to open again.
    warm: RwLock<Option<WarmDb>>,
    /// Where the handle reopens from.
    path: PathBuf,
    /// Held by a request that writes, so writes run one at a time.
    writer: Mutex<()>,
    /// The writes clients sent that wait for [`WarmState::writer`].
    writers_waiting: AtomicUsize,
    /// Set for good by a request that panicked; see [`WarmState::poisoned`].
    poisoned: AtomicBool,
    /// Held by the one [`WarmState::reopen`] running.
    reopening: AtomicBool,
    #[cfg(test)]
    panic_next: AtomicBool,
}

/// Opening the database printed: it is not served in-process. Each line,
/// such as an unloadable row and the `axil heal` that clears it, would
/// otherwise show once on the host's stderr and never in the commands the
/// user runs. Carries what the open printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenPrinted(pub String);

impl fmt::Display for OpenPrinted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the database is not served in-process: opening it printed\n{}",
            self.0.trim_end()
        )
    }
}

impl std::error::Error for OpenPrinted {}

/// [`WarmDb::open`], as long as the open prints nothing; [`OpenPrinted`]
/// otherwise, with the handle closed again.
///
/// A vector store a killed writer left needing repair is repaired first,
/// and said so on the host's own stderr. The open would repair it too and
/// print that it did, which would keep the database from being served for
/// a line a one-shot run prints once, in the first command after the crash.
///
/// The host's own open skips the open guard ([`axil_core::without_open_guard`]):
/// the guard asks the host to let go, or, in the daemon, checks that nothing
/// but this opens the database.
fn open_warm(path: &Path) -> anyhow::Result<WarmDb> {
    let (stdout, stderr, warm) = axil_core::without_open_guard(|| {
        repair_vector_store(path);
        cli_io::capture(ReqCtx::default(), || WarmDb::open(path))
    });
    let warm = warm?;
    if stdout.is_empty() && stderr.is_empty() {
        return Ok(warm);
    }
    drop(warm);
    let printed = [stdout, stderr].concat();
    Err(OpenPrinted(String::from_utf8_lossy(&printed).into_owned()).into())
}

/// Repair the default vector store at `path` if a killed writer left it
/// needing it, as the open's probe would; see [`open_warm`]. A store that
/// can't be probed is left for the open to report.
fn repair_vector_store(path: &Path) {
    #[cfg(feature = "vector")]
    {
        let disabled = path
            .parent()
            .and_then(|dir| axil_core::load_config_from(dir).ok())
            .is_some_and(|config| config.is_engine_disabled("vec"));
        if disabled {
            return;
        }
        if let Ok(axil_vector::VectorStoreProbe::Ready { repaired: true, .. }) =
            axil_vector::probe_vector_store(path)
        {
            errln!(
                "axil: vector store {} was not closed cleanly; repaired it",
                axil_vector::vector_db_path(path).display()
            );
        }
    }
    #[cfg(not(feature = "vector"))]
    let _ = path;
}

/// Why [`WarmState::reopen`] did not reopen the database.
#[derive(Debug)]
pub enum ReopenError {
    /// A request panicked on the handle: it can't be reopened in this
    /// process (see [`WarmState::poisoned`]).
    Poisoned,
    /// Another reopen is running. Requests get [`NotServed::Unavailable`]
    /// until it is done.
    Running,
    /// Requests still held the old handle when the timeout ran out. The
    /// old handle is kept and serves on.
    InUse,
    /// The database did not open again, or the open printed
    /// ([`OpenPrinted`]). No handle is left, and requests get
    /// [`NotServed::Unavailable`] until a later reopen succeeds.
    Open(anyhow::Error),
}

impl fmt::Display for ReopenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReopenError::Poisoned => f.write_str("a request panicked on the open database"),
            ReopenError::Running => f.write_str("another reopen is running"),
            ReopenError::InUse => f.write_str("the database was still in use"),
            ReopenError::Open(e) => write!(f, "the database did not open again: {e:#}"),
        }
    }
}

impl std::error::Error for ReopenError {}

impl WarmState {
    /// Open the database at `path` (see [`WarmDb::open`]) to serve requests.
    /// An open that prints anything fails with [`OpenPrinted`], and leaves
    /// the database closed.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        cli_io::install_diagnostic_sink();
        let warm = open_warm(path)?;
        Ok(Self {
            path: warm.key().to_path_buf(),
            warm: RwLock::new(Some(warm)),
            writer: Mutex::new(()),
            writers_waiting: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            reopening: AtomicBool::new(false),
            #[cfg(test)]
            panic_next: AtomicBool::new(false),
        })
    }

    /// The open handle, if there is one.
    pub fn handle(&self) -> Option<WarmDb> {
        self.slot_read().clone()
    }

    /// Debug builds only: the writer lock, held as a write holds it.
    #[cfg(debug_assertions)]
    pub fn debug_writer(&self) -> std::sync::MutexGuard<'_, ()> {
        self.writer.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How many writes clients sent are waiting for the one running. The
    /// queue's drain thread lets them go first: it starts its next step
    /// only once none waits.
    pub fn writers_waiting(&self) -> usize {
        self.writers_waiting.load(Ordering::Acquire)
    }

    /// Whether a request panicked on the handle. Nothing is served after
    /// that, and the host must exit the process with
    /// [`std::process::exit`].
    ///
    /// A panic can unwind through an open redb write transaction, and redb
    /// skips rolling one back on a panicking thread. The handle then counts
    /// pages as allocated that no table references, and the next commit on
    /// it, or the clean shutdown that closing it runs, would record them so
    /// for good. So the handle is never used or closed again: it is leaked,
    /// and its files stay open and locked, which also means this process
    /// can't open the database again. When the process exits without
    /// closing it, the next open finds the database not closed cleanly and
    /// repairs it, as after a one-shot run that panicked.
    ///
    /// A request already running on another thread when the panic came is
    /// not stopped, and may still commit.
    pub fn poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Close the handle and open the database again, for a host that found
    /// the open handle stale.
    ///
    /// Only one reopen runs at a time; another caller gets
    /// [`ReopenError::Running`] at once. While it runs, requests are
    /// [`NotServed::Unavailable`] rather than queued behind it. It waits up
    /// to `timeout` for the requests still using the old handle (see
    /// [`WarmDb::release`]) and keeps that handle if they don't finish.
    pub fn reopen(&self, timeout: Duration) -> Result<(), ReopenError> {
        if self.poisoned() {
            return Err(ReopenError::Poisoned);
        }
        if self
            .reopening
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(ReopenError::Running);
        }
        struct Claim<'a>(&'a AtomicBool);
        impl Drop for Claim<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _claim = Claim(&self.reopening);

        // The lock is held only to take the handle out and to put one back,
        // never across the wait or the open.
        let old = self.slot_write().take();
        if let Some(old) = old {
            if let Err(old) = old.release(timeout) {
                *self.slot_write() = Some(old);
                return Err(ReopenError::InUse);
            }
        }
        // Opening creates a database that is not there: one removed from
        // under the handle is not brought back empty.
        if !self.path.is_file() {
            return Err(ReopenError::Open(anyhow::anyhow!(
                "{} no longer exists",
                self.path.display()
            )));
        }
        let warm = open_warm(&self.path).map_err(ReopenError::Open)?;
        *self.slot_write() = Some(warm);
        Ok(())
    }

    /// Close the database, for a host that is shutting down: wait up to
    /// `timeout` for a reopen that is running and for the requests still
    /// using the handle (see [`WarmDb::release`]), then close it. Requests
    /// are [`NotServed::Unavailable`] from then on.
    ///
    /// A poisoned handle is never closed ([`ReopenError::Poisoned`]; see
    /// [`WarmState::poisoned`]). One still in use when the time runs out is
    /// kept ([`ReopenError::InUse`]), and so is everything while a reopen
    /// still runs ([`ReopenError::Running`]).
    pub fn release(&self, timeout: Duration) -> Result<(), ReopenError> {
        let deadline = std::time::Instant::now() + timeout;
        // A reopen that is running would put a handle back.
        while self
            .reopening
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            if std::time::Instant::now() >= deadline {
                return Err(ReopenError::Running);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let result = self.release_claimed(deadline);
        self.reopening.store(false, Ordering::Release);
        result
    }

    fn release_claimed(&self, deadline: std::time::Instant) -> Result<(), ReopenError> {
        if self.poisoned() {
            return Err(ReopenError::Poisoned);
        }
        let Some(old) = self.slot_write().take() else {
            return Ok(());
        };
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match old.release(left) {
            Ok(_) => Ok(()),
            Err(old) => {
                *self.slot_write() = Some(old);
                Err(ReopenError::InUse)
            }
        }
    }

    /// The handle a request runs on.
    fn current(&self) -> Result<WarmDb, NotServed> {
        if self.poisoned() {
            return Err(NotServed::Unavailable);
        }
        self.handle().ok_or(NotServed::Unavailable)
    }

    /// Record that a request panicked on `warm`, and leak it so the
    /// database is never closed cleanly (see [`WarmState::poisoned`]).
    pub fn poison(&self, warm: WarmDb) {
        self.poisoned.store(true, Ordering::Release);
        std::mem::forget(warm);
    }

    /// Never serve or close the handle again, as after a panic (see
    /// [`WarmState::poisoned`]): for a host that found the database's files
    /// written by someone else under the open handle. Closing it would write
    /// the handle's view of the file over theirs.
    pub fn abandon(&self) {
        match self.handle() {
            Some(warm) => self.poison(warm),
            None => self.poisoned.store(true, Ordering::Release),
        }
    }

    fn slot_read(&self) -> std::sync::RwLockReadGuard<'_, Option<WarmDb>> {
        self.warm.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn slot_write(&self) -> std::sync::RwLockWriteGuard<'_, Option<WarmDb>> {
        self.warm.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Run `req` against `state`'s database, as `axil` would have run it in the
/// client's process, and return what it printed and its exit code: the
/// one-shot binary's bytes. A command [`warm_route`] refuses, a command line
/// that names an unknown argument, and a database other than the one held
/// come back as [`NotServed`] with nothing printed or written.
///
/// The command runs on a thread of its own with `main`'s stack. An error
/// prints as the one-shot binary prints it and exits 1. A panic exits 1 too,
/// and poisons `state` ([`WarmState::poisoned`]).
#[cfg_attr(not(test), allow(dead_code))] // the daemon runs requests through its gate
pub fn execute_captured(req: Request, state: &WarmState) -> Result<Response, NotServed> {
    execute_gated(req, state, &|_| Ok(()))
}

/// [`execute_captured`], with `gate` asked last before the command runs:
/// after a write has taken the writer lock and before the request takes the
/// handle, with whether the command writes. A [`NotServed`] it returns is
/// the request's answer, with nothing run. A host checks there what may
/// have changed while the request waited: its deadline, and whether the
/// handle is still the database's.
pub fn execute_gated(
    req: Request,
    state: &WarmState,
    gate: &(dyn Fn(bool) -> Result<(), NotServed> + Sync),
) -> Result<Response, NotServed> {
    std::thread::scope(|scope| {
        let request = std::thread::Builder::new()
            .name("axil-request".into())
            .stack_size(REQUEST_STACK)
            .spawn_scoped(scope, move || execute(req, state, gate));
        match request {
            // A panic outside the command itself (parsing, say) leaves the
            // handle as it was.
            Ok(request) => request
                .join()
                .unwrap_or_else(|payload| Ok(panicked(cli_io::Captured::default(), &*payload))),
            Err(_) => Err(NotServed::Unavailable),
        }
    })
}

fn execute(
    req: Request,
    state: &WarmState,
    gate: &(dyn Fn(bool) -> Result<(), NotServed> + Sync),
) -> Result<Response, NotServed> {
    let mut cli = match parse(&req)? {
        Parsed::Run(cli) => cli,
        Parsed::Printed(response) => return Ok(response),
    };
    if req.stdin.is_none() && reads_stdin(&cli.command) {
        return Err(NotServed::Terminal);
    }
    let db_path = request_db(cli.db.as_deref(), &req.cwd).ok_or(NotServed::WrongDb)?;
    let writes = writes(&cli.command);
    // A write waits for the writes before it without holding the handle,
    // so a queue of writes never keeps a reopen waiting on it.
    let _writing = writes.then(|| {
        let waiting = (!req.queued).then(|| Waiting::new(&state.writers_waiting));
        let lock = state.writer.lock().unwrap_or_else(PoisonError::into_inner);
        drop(waiting);
        lock
    });
    // A write that waited on the lock behind one that panicked.
    if state.poisoned() {
        return Err(NotServed::Unavailable);
    }
    gate(writes)?;
    let warm = state.current()?;
    if cli_io::db_key(&db_path).ok().as_deref() != Some(warm.key()) {
        return Err(NotServed::WrongDb);
    }
    warm_route(&cli.command, warm.db())?;
    cli.db = Some(db_path);
    let held = warm.clone();
    let ctx = ReqCtx {
        stdin: req.stdin,
        slow_query_log: req.slow_query_log,
        warm: Some(warm),
        ..ReqCtx::default()
    };
    let (printed, result) = cli_io::capture_in_order(ctx, || {
        panic::catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            if state.panic_next.swap(false, Ordering::AcqRel) {
                panic!("a request panicked on purpose");
            }
            let out = Output::for_cli(&cli);
            crate::dispatch_command(cli, &out)
        }))
    });
    match result {
        Ok(result) => finish(result, printed),
        Err(payload) => {
            state.poison(held);
            Ok(panicked(printed, &*payload))
        }
    }
}

/// A write a client sent, counted in [`WarmState::writers_waiting`] until
/// dropped.
struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn new(count: &'a AtomicUsize) -> Self {
        count.fetch_add(1, Ordering::AcqRel);
        Self(count)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The response for what [`dispatch_command`](crate::dispatch_command)
/// returned: an error printed as `run_cli` prints it, or [`NotServed`] when
/// the handle lacked what the command needs (the command wrote nothing).
fn finish(
    result: anyhow::Result<i32>,
    mut printed: cli_io::Captured,
) -> Result<Response, NotServed> {
    match result {
        Ok(exit) => Ok(Response::of(exit, printed)),
        Err(e) => {
            if let Some(WarmLacks(need)) = e.downcast_ref::<WarmLacks>() {
                return Err(NotServed::Lacks(*need));
            }
            let line = format!("{}\n", crate::format_cli_error(&e));
            printed.push(true, line.as_bytes());
            Ok(Response::of(crate::EXIT_ERROR, printed))
        }
    }
}

fn panicked(mut printed: cli_io::Captured, payload: &(dyn Any + Send)) -> Response {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a non-string payload".to_string());
    let line = format!("axil: the command panicked: {message}\n");
    printed.push(true, line.as_bytes());
    Response::of(crate::EXIT_ERROR, printed)
}

/// Whether a host runs `command` in-process at all, and what its open
/// handle must have to run it. A client can ask this before it sends a
/// request; [`warm_route`] adds the handle's side.
pub fn route(command: &Command) -> Result<Vec<Need>, NotServed> {
    // A raw vector for the default space has its dimensions checked against
    // the vector store on disk before the command opens the database.
    #[cfg(feature = "embed")]
    if let Command::Store {
        vector: Some(_),
        space: None,
        ..
    } = command
    {
        return Err(NotServed::Command);
    }
    crate::hot_needs(command).ok_or(NotServed::Command)
}

/// Whether a host runs the command line `argv` (program name first)
/// in-process at all, as far as [`route`] tells without a handle; one that
/// does not parse it doesn't. Parsing needs `main`'s stack (see
/// [`REQUEST_STACK`]).
pub fn routes(argv: &[String]) -> bool {
    let parser = parser();
    let Ok(matches) = parser.command.clone().try_get_matches_from(argv) else {
        return false;
    };
    Cli::from_arg_matches(&matches).is_ok_and(|cli| route(&cli.command).is_ok())
}

/// Whether `db` can serve `command` in-process: [`route`] accepts it and
/// `db` has everything it needs, the same needs the command checks through
/// [`cli_io::hot_db`]. A host can refuse a request on this alone, without
/// running anything.
pub fn warm_route(command: &Command, db: &Axil) -> Result<(), NotServed> {
    match route(command)?.into_iter().find(|need| !need.met_by(db)) {
        Some(need) => Err(NotServed::Lacks(need)),
        None => Ok(()),
    }
}

/// Whether `command` writes, and so runs under [`WarmState`]'s writer lock:
/// `store`, a `checkpoint` other than `show`, and a `recall --fresh-only`,
/// which may re-index the files it found stale.
pub fn writes(command: &Command) -> bool {
    match command {
        Command::Store { .. } => true,
        Command::AutoCapture { dry_run, .. } => !dry_run,
        #[cfg(feature = "checkpoint")]
        Command::Checkpoint { arg, .. } => arg.as_deref() != Some("show"),
        #[cfg(feature = "embed")]
        Command::Recall { fresh_only, .. } => *fresh_only,
        _ => false,
    }
}

/// Whether `command` reads stdin whatever stdin is: `store <table> -`,
/// `auto-capture -` and `extract-entities -`. A `checkpoint -` reads it
/// only when it is not a terminal.
fn reads_stdin(command: &Command) -> bool {
    match command {
        Command::Store { json_data, .. } => json_data == "-",
        Command::AutoCapture { text, .. } | Command::ExtractEntities { text } => text == "-",
        _ => false,
    }
}

/// The database a command line runs on in the client's process: its `--db`
/// or `AXIL_DB` resolved against `cwd`, else the `.axil/memory.axil` found
/// walking up from `cwd`. Never the host's own working directory.
fn request_db(db: Option<&Path>, cwd: &Path) -> Option<PathBuf> {
    if !cwd.is_absolute() {
        return None;
    }
    match db {
        Some(path) => Some(cwd.join(path)),
        None => crate::find_axil_dir_from(cwd.to_path_buf()),
    }
}

enum Parsed {
    /// A command to run.
    Run(Cli),
    /// Help, the version, or a usage error, already rendered.
    Printed(Response),
}

/// The CLI's clap command, made once. Each parse builds what it reaches, as
/// a one-shot parse does: building the whole tree would also run clap's
/// debug-build checks on subcommands the command line never names.
struct CliParser {
    command: clap::Command,
    /// The [`CLI_ENV`] values the command was made with, in that order:
    /// clap reads each variable as it makes the argument, and both what a
    /// command line parses to and what help and usage errors print depend
    /// on them.
    env: Vec<Option<OsString>>,
}

impl CliParser {
    /// Whether `req`'s client has any [`CLI_ENV`] variable other than the
    /// parser was made with, among those `wanted` picks.
    fn env_differs(&self, req: &Request, wanted: impl Fn(&str) -> bool) -> bool {
        CLI_ENV
            .iter()
            .zip(&self.env)
            .any(|(name, host)| wanted(name) && req.env(name) != host.as_deref())
    }
}

fn parser() -> &'static CliParser {
    static PARSER: OnceLock<CliParser> = OnceLock::new();
    PARSER.get_or_init(|| {
        let env = CLI_ENV.iter().map(std::env::var_os).collect();
        let command = Cli::command();
        CliParser { command, env }
    })
}

/// Parse `req`'s command line as `Cli::parse` would in the client's
/// process, with the client's [`CLI_ENV`] values.
fn parse(req: &Request) -> Result<Parsed, NotServed> {
    // An empty `AXIL_DB` makes every command line a usage error.
    if req.env("AXIL_DB").is_some_and(OsStr::is_empty) {
        return Err(NotServed::Parse);
    }
    let parser = parser();
    let matches = match parser.command.clone().try_get_matches_from(&req.argv) {
        Ok(matches) => matches,
        Err(e) => return printed(e, req, parser),
    };
    let mut cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => return printed(e.format(&mut parser.command.clone()), req, parser),
    };
    // Without `--db` on the command line, the client's `AXIL_DB` applies,
    // not one the host's environment supplied. Any other variable the
    // command reads would have to be applied the same way, field by field;
    // no command served in-process reads one, so the client runs it itself.
    if matches.value_source("db") != Some(ValueSource::CommandLine) {
        cli.db = req.env("AXIL_DB").map(PathBuf::from);
    }
    let mut read = Vec::new();
    env_read(&parser.command, &matches, &mut read);
    if parser.env_differs(req, |name| name != "AXIL_DB" && read.contains(&name)) {
        return Err(NotServed::Parse);
    }
    Ok(Parsed::Run(cli))
}

/// The environment variables the arguments of `command`, and of each
/// subcommand `matches` goes on to name, read.
fn env_read<'a>(command: &'a clap::Command, matches: &clap::ArgMatches, read: &mut Vec<&'a str>) {
    read.extend(
        command
            .get_arguments()
            .filter_map(|arg| arg.get_env()?.to_str()),
    );
    if let Some((name, sub)) = matches.subcommand() {
        if let Some(command) = command.find_subcommand(name) {
            env_read(command, sub, read);
        }
    }
}

/// What `clap::Error::exit` would print for `e`, as a response.
///
/// An unknown argument or subcommand is [`NotServed::Parse`]: a client of a
/// newer build runs it in its own process. So is any of this when the
/// client's [`CLI_ENV`] values are not the ones the parser was made with.
/// clap reads them as it builds the command: help shows an argument's value,
/// a usage line lists `--db` when `AXIL_DB` is set, a command line without a
/// subcommand prints help without it and an error with it, and an invalid
/// value is an error only where it is set.
fn printed(e: clap::Error, req: &Request, parser: &CliParser) -> Result<Parsed, NotServed> {
    let unknown = matches!(
        e.kind(),
        ErrorKind::UnknownArgument | ErrorKind::InvalidSubcommand
    );
    if unknown || parser.env_differs(req, |_| true) {
        return Err(NotServed::Parse);
    }
    let mut printed = cli_io::Captured::default();
    printed.push(e.use_stderr(), e.render().to_string().as_bytes());
    Ok(Parsed::Printed(Response::of(e.exit_code(), printed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_io::tests::parity::{
        self, build_fixture, case, check_parity_with, copy_dir, matrix, one_shot_at, Case, Run,
    };
    use std::sync::Arc;
    use std::time::Instant;

    /// A request from a client working in `cwd`. Its stdin is piped and
    /// holds `stdin`, empty by default, like the one-shot children's.
    fn request(args: &[&str], cwd: &Path, axil_db: Option<&str>, stdin: Option<&[u8]>) -> Request {
        Request {
            argv: parity::argv(args).into_iter().map(OsString::from).collect(),
            cwd: cwd.to_path_buf(),
            env: axil_db
                .map(|db| vec![("AXIL_DB".into(), db.into())])
                .unwrap_or_default(),
            stdin: Some(stdin.unwrap_or_default().to_vec()),
            slow_query_log: true,
            queued: false,
        }
    }

    fn served(req: Request, state: &WarmState) -> Run {
        let r = execute_captured(req, state).unwrap_or_else(|e| panic!("not served: {e}"));
        (r.stdout, r.stderr, r.exit)
    }

    fn show(run: &Run) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            run.2,
            String::from_utf8_lossy(&run.0),
            String::from_utf8_lossy(&run.1)
        )
    }

    /// How a case's command line names its database.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum DbFrom {
        /// No `--db`: the `.axil/memory.axil` above the client's directory.
        Found,
        /// `--db` with the absolute path, as the cli_io matrix has it.
        AbsoluteFlag,
        /// `--db .axil/memory.axil`, relative to the client's directory.
        RelativeFlag,
        /// `AXIL_DB=.axil/memory.axil`, relative to the client's directory.
        Env,
    }

    const RELATIVE_DB: &str = ".axil/memory.axil";

    /// `c` with its database named the `from` way, and with the project file
    /// it names given relative to the project unless the database path is
    /// absolute.
    fn rewrite(c: &Case, db: &str, auth_rs: &str, from: DbFrom) -> Case {
        let mut args = Vec::new();
        let mut given = c.args.iter();
        while let Some(arg) = given.next() {
            if arg == "--db" {
                given.next();
                match from {
                    DbFrom::AbsoluteFlag => args.extend(["--db", db]),
                    DbFrom::RelativeFlag => args.extend(["--db", RELATIVE_DB]),
                    DbFrom::Found | DbFrom::Env => {}
                }
            } else if arg == auth_rs && from != DbFrom::AbsoluteFlag {
                args.push("src/auth.rs");
            } else {
                args.push(arg);
            }
        }
        let mut out = case(c.name, &args);
        out.stdin = c.stdin;
        out.volatile = c.volatile;
        out.cold_only = c.cold_only;
        out
    }

    /// The cli_io parity matrix, run by [`execute_captured`] on an open
    /// handle for a client in the project's directory, against the one-shot
    /// binary run in that directory. The host's own working directory is
    /// elsewhere, so a command that resolved anything against it would
    /// differ. Every way of naming the database is covered: found from the
    /// directory (the whole matrix), and a few cases each through an
    /// absolute `--db`, a relative `--db` and a relative `AXIL_DB`.
    #[test]
    fn served_output_matches_one_shot_in_the_clients_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let fixture = build_fixture(&tmp.path().join("fixture"));
        let proj = tmp.path().join("work/proj");
        let db_path = proj.join(".axil/memory.axil");
        let db = db_path.to_str().unwrap();
        let auth_rs = proj.join("src/auth.rs");
        let auth_rs = auth_rs.to_str().unwrap();
        let host_cwd = std::env::current_dir().unwrap();
        assert!(!host_cwd.starts_with(tmp.path()));

        // Refused before they run; checked below.
        let refused = |c: &Case| c.cold_only || c.name == "clap unknown flag";
        let few = [
            "boot json",
            "boot --files",
            "recall-for-file",
            "store inline",
            "store stdin",
            "store bad JSON (Err)",
            "--help",
            "fts json",
            "checkpoint -",
            "context-block",
        ];
        for from in [
            DbFrom::Found,
            DbFrom::AbsoluteFlag,
            DbFrom::RelativeFlag,
            DbFrom::Env,
        ] {
            // With `AXIL_DB` set, help and usage errors are the client's to
            // print (see `help_version_and_usage_errors_match_one_shot`).
            let printed_by_clap = |c: &Case| !c.args.iter().any(|a| a == "--db");
            let cases: Vec<Case> = matrix(db, auth_rs)
                .iter()
                .filter(|c| !refused(c) && (from == DbFrom::Found || few.contains(&c.name)))
                .filter(|c| from != DbFrom::Env || !printed_by_clap(c))
                .map(|c| rewrite(c, db, auth_rs, from))
                .collect();
            let env = (from == DbFrom::Env).then_some(RELATIVE_DB);
            let cwd = || proj.canonicalize().unwrap();
            let results = check_parity_with(
                &fixture,
                &proj,
                &cases,
                |c, args| {
                    let state = WarmState::open(&db_path).unwrap();
                    served(request(args, &cwd(), env, c.stdin), &state)
                },
                |c, args| one_shot_at(args, c.stdin, &cwd(), env),
            );
            // Equal could still mean equally failed to find the database.
            let (name, run) = results.iter().find(|(n, _)| *n == "boot json").unwrap();
            assert!(
                run.2 == 0 && run.0.starts_with(b"{"),
                "{from:?} {name}: {}",
                show(run)
            );
        }

        // The matrix's cases the host leaves to the client.
        copy_dir(&fixture, &proj);
        let state = WarmState::open(&db_path).unwrap();
        for c in matrix(db, auth_rs).iter().filter(|c| refused(c)) {
            let args: Vec<&str> = c.args.iter().map(String::as_str).collect();
            let expected = if c.cold_only {
                NotServed::Command
            } else {
                NotServed::Parse
            };
            let got = execute_captured(request(&args, &proj, None, None), &state);
            assert_eq!(got, Err(expected), "{}", c.name);
        }
    }

    /// Help, the version and usage errors print what the one-shot binary
    /// prints. With the client's `AXIL_DB` set, clap prints them differently
    /// (help shows the value, usage lines list `--db`), so the host, made
    /// without it, leaves them to the client.
    #[test]
    fn help_version_and_usage_errors_match_one_shot() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let cwd = proj.canonicalize().unwrap();
        // Making the parser needs a request thread's stack.
        let host_env = std::thread::Builder::new()
            .stack_size(REQUEST_STACK)
            .spawn(|| parser().env.clone())
            .unwrap()
            .join()
            .unwrap();
        let cases: Vec<(Vec<&str>, Option<&str>)> = vec![
            (vec!["--help"], None),
            (vec!["-h"], None),
            (vec!["store", "--help"], None),
            (vec!["help", "recall-for-file"], None),
            (vec!["--version"], None),
            (vec!["store"], None),
            (vec![], None),
            (vec!["boot", "--boot-format", "nope"], None),
            (vec!["--help"], Some("/elsewhere/memory.axil")),
            (vec!["store"], Some(RELATIVE_DB)),
            (vec![], Some(RELATIVE_DB)),
        ];
        for (args, env) in cases {
            let req = request(&args, &cwd, env, None);
            let client_env: Vec<Option<OsString>> = CLI_ENV
                .iter()
                .map(|name| match *name {
                    "AXIL_DB" => env.map(OsString::from),
                    other => std::env::var_os(other),
                })
                .collect();
            if host_env != client_env {
                let got = execute_captured(req, &state);
                assert_eq!(got, Err(NotServed::Parse), "{args:?} AXIL_DB={env:?}");
                continue;
            }
            let got = served(req, &state);
            let want = one_shot_at(&args, None, &cwd, env);
            assert!(
                got == want,
                "{args:?} AXIL_DB={env:?}\n=== served\n{}\n=== one-shot\n{}",
                show(&got),
                show(&want)
            );
            if args == ["--help"] {
                let out = String::from_utf8_lossy(&got.0);
                assert!(got.2 == 0 && out.starts_with("One file."), "{out}");
            }
        }
    }

    /// A command that reads a variable other than `AXIL_DB` from the
    /// environment is the client's to run when the client's value is not
    /// the host's, and so is its help; a command that doesn't read it is
    /// served whatever the client has.
    #[test]
    fn a_variable_the_command_reads_is_the_clients() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let host_tools = std::env::var_os("AXIL_MCP_TOOLS");
        let with_tools = |args: &[&str], tools: Option<&str>| Request {
            env: tools
                .map(|v| vec![("AXIL_MCP_TOOLS".into(), v.into())])
                .unwrap_or_default(),
            ..request(args, &proj, None, None)
        };
        let differs = Some("bogus");
        assert_ne!(host_tools.as_deref(), differs.map(OsStr::new));
        // Only `mcp` reads one, and a build without it has no such command.
        let mcp = cfg!(feature = "mcp");
        for args in [&["mcp"][..], &["mcp", "--help"]] {
            let got = execute_captured(with_tools(args, differs), &state);
            if mcp {
                assert_eq!(got, Err(NotServed::Parse), "{args:?}");
            }
        }
        let r = execute_captured(with_tools(&["boot"], differs), &state).unwrap();
        assert_eq!(r.exit, 0, "{}", String::from_utf8_lossy(&r.stderr));
        if mcp && host_tools.is_none() {
            let got = execute_captured(with_tools(&["mcp"], None), &state);
            assert_eq!(got, Err(NotServed::Command));
        }
    }

    /// Every `env =` on an argument is a variable the request carries. A new
    /// one the parser reads but a request lacks would quietly take the
    /// host's value.
    #[test]
    fn every_variable_the_parser_reads_is_carried() {
        let source = include_str!("../main.rs");
        let mut literal: Vec<&str> = source
            .split("env = \"")
            .skip(1)
            .map(|rest| &rest[..rest.find('"').unwrap()])
            .collect();
        literal.sort_unstable();
        literal.dedup();
        let mut carried = CLI_ENV.to_vec();
        carried.sort_unstable();
        assert_eq!(literal, carried, "the `env = \"…\"` attributes in main.rs");

        fn walk(command: &clap::Command, found: &mut Vec<String>) {
            for arg in command.get_arguments() {
                if let Some(name) = arg.get_env() {
                    found.push(name.to_string_lossy().into_owned());
                }
            }
            for sub in command.get_subcommands() {
                walk(sub, found);
            }
        }
        let mut parsed = std::thread::Builder::new()
            .stack_size(REQUEST_STACK)
            .spawn(|| {
                let mut found = Vec::new();
                walk(&Cli::command(), &mut found);
                found
            })
            .unwrap()
            .join()
            .unwrap();
        parsed.sort_unstable();
        parsed.dedup();
        // A build without `mcp` lacks the one argument that reads this.
        if !cfg!(feature = "mcp") {
            carried.retain(|name| *name != "AXIL_MCP_TOOLS");
        }
        assert_eq!(parsed, carried, "the variables the parser reads");
    }

    /// A request runs on the database its client's process would have found,
    /// from the client's directory, and only when that is the one held.
    #[test]
    fn the_database_is_the_one_the_client_would_find() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let src = proj.join("src").canonicalize().unwrap();
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        let elsewhere = elsewhere.canonicalize().unwrap();

        // Its absolute path reaches it from anywhere; from a subdirectory of
        // the project it is found walking up, or named relative to it.
        let abs = proj.join(".axil/memory.axil");
        let abs = abs.to_str().unwrap();
        let want = served(request(&["boot"], &elsewhere, Some(abs), None), &state);
        assert!(want.2 == 0 && want.0.starts_with(b"{"), "{}", show(&want));
        let up = "../.axil/memory.axil";
        for (args, env) in [
            (vec!["boot"], None),
            (vec!["--db", up, "boot"], None),
            (vec!["boot"], Some(up)),
        ] {
            let got = served(request(&args, &src, env, None), &state);
            assert!(got == want, "{args:?} AXIL_DB={env:?}: {}", show(&got));
        }

        // No database above the client's directory, another database, or a
        // directory that is no directory: the client's to run.
        let other = elsewhere.join("other.axil");
        let other = other.to_str().unwrap();
        for (args, cwd, env) in [
            (vec!["boot"], elsewhere.as_path(), None),
            (vec!["--db", other, "boot"], src.as_path(), None),
            (vec!["boot"], src.as_path(), Some(other)),
            (vec!["--db", "memory.axil", "boot"], src.as_path(), None),
            (vec!["boot"], Path::new("proj"), None),
        ] {
            let got = execute_captured(request(&args, cwd, env, None), &state);
            assert_eq!(got, Err(NotServed::WrongDb), "{args:?} in {cwd:?}");
        }
        assert_eq!(
            execute_captured(request(&["boot"], &src, Some(""), None), &state),
            Err(NotServed::Parse)
        );
    }

    /// What the handle can't serve comes back as a typed refusal, with
    /// nothing written and no store created: commands the host never runs,
    /// and commands whose own open would attach or create what the handle
    /// lacks.
    #[test]
    fn what_the_handle_lacks_is_not_served() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db_path = proj.join(".axil/memory.axil");
        let companion = |suffix: &str| axil_core::companion_path(&db_path, suffix);
        for suffix in [".fts", ".ts"] {
            let _ = std::fs::remove_dir_all(companion(suffix));
            let _ = std::fs::remove_file(companion(suffix));
        }
        let state = WarmState::open(&db_path).unwrap();
        let stored = r#"{"summary":"stored through a handle that lacks a store"}"#;
        let mut cases: Vec<(Vec<String>, NotServed)> = vec![
            (
                parity::argv(&["store", "decisions", stored, "--llm"]),
                NotServed::Lacks(Need::Llm),
            ),
            (parity::argv(&["get", "x"]), NotServed::Command),
            (parity::argv(&["index", "."]), NotServed::Command),
            (parity::argv(&["no-such-command"]), NotServed::Command),
        ];
        if cfg!(feature = "fts") {
            let fts = parity::argv(&["fts", "cache"]);
            cases.push((fts, NotServed::Lacks(Need::FullText)));
        }
        if cfg!(feature = "timeseries") {
            let since = parity::argv(&["since", "1h"]);
            cases.push((since, NotServed::Lacks(Need::TimeSeries)));
        }
        #[cfg(feature = "embed")]
        {
            let embed = parity::argv(&["store", "decisions", stored, "--embed", "summary"]);
            cases.push((embed, NotServed::Lacks(Need::Vectors)));
            let vector = serde_json::to_string(&vec![0.5f32; 384]).unwrap();
            let raw = parity::argv(&["store", "decisions", stored, "--vector", &vector]);
            cases.push((raw, NotServed::Command));
        }

        let decisions = || {
            state
                .handle()
                .unwrap()
                .db()
                .list("decisions")
                .unwrap()
                .len()
        };
        let before = decisions();
        for (argv, expected) in &cases {
            let args: Vec<&str> = argv[1..].iter().map(String::as_str).collect();
            let got = execute_captured(request(&args, &proj, None, None), &state);
            assert_eq!(got, Err(*expected), "{args:?}");
        }
        assert_eq!(decisions(), before, "a refused command wrote a record");
        for suffix in [".fts", ".ts", ".vec"] {
            assert!(
                !companion(suffix).exists(),
                "a refused command made {suffix}"
            );
        }
    }

    /// A record to read from a terminal is not served, before anything is
    /// written; piped, it is. A checkpoint at a terminal reads nothing, as
    /// in the one-shot binary, and is served.
    #[test]
    fn a_store_from_a_terminal_is_not_served() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let at_terminal = |args: &[&str]| Request {
            stdin: None,
            ..request(args, &proj, None, None)
        };
        let decisions = || {
            state
                .handle()
                .unwrap()
                .db()
                .list("decisions")
                .unwrap()
                .len()
        };
        let before = decisions();
        let store = ["store", "decisions", "-"];
        assert_eq!(
            execute_captured(at_terminal(&store), &state),
            Err(NotServed::Terminal)
        );
        assert_eq!(decisions(), before);
        let piped = request(&store, &proj, None, Some(br#"{"summary":"piped"}"#));
        assert_eq!(execute_captured(piped, &state).unwrap().exit, 0);
        assert_eq!(decisions(), before + 1);
        if cfg!(feature = "checkpoint") {
            assert!(execute_captured(at_terminal(&["checkpoint", "-"]), &state).is_ok());
        }
    }

    /// The handle refusing from inside the command, as [`cli_io::hot_db`]
    /// does, is the same typed refusal; any other error prints as the
    /// one-shot binary prints it.
    #[test]
    fn a_command_error_prints_and_a_refusal_does_not() {
        let printed = |to_err: bool, bytes: &[u8]| {
            let mut printed = cli_io::Captured::default();
            printed.push(to_err, bytes);
            printed
        };
        let refused = finish(
            Err(WarmLacks(Need::FullText).into()),
            printed(false, b"half"),
        );
        assert_eq!(refused, Err(NotServed::Lacks(Need::FullText)));
        let mut before = printed(true, b"warning\n");
        before.push(false, b"out\n");
        let failed = finish(Err(anyhow::anyhow!("boom").context("ctx")), before);
        let error = b"{\"error\":\"ctx: boom\"}\n";
        assert_eq!(
            failed,
            Ok(Response {
                exit: 1,
                stdout: b"out\n".to_vec(),
                stderr: [b"warning\n".as_slice(), error].concat(),
                order: vec![(true, 8), (false, 4), (true, error.len())],
            })
        );
    }

    /// A request that panics is answered, and poisons the state: the
    /// requests after it, however many arrive at once, are refused at once,
    /// nothing reopens, and the handle is never closed, so its files stay
    /// locked until the process exits.
    #[test]
    fn a_panicking_request_is_answered_and_poisons_the_state() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let state = WarmState::open(&db).unwrap();
        let boot = || request(&["boot"], &proj, None, None);

        // A reference held elsewhere, as by a recall worker left running
        // past its deadline, must not make the refusals wait for it.
        let held = state.handle().unwrap();
        state.panic_next.store(true, Ordering::Release);
        let data = r#"{"summary":"a store that panics"}"#;
        let store = request(&["store", "decisions", data], &proj, None, None);
        let r = execute_captured(store, &state).unwrap();
        assert_eq!(r.exit, 1);
        let err = String::from_utf8_lossy(&r.stderr);
        assert!(
            err.ends_with("axil: the command panicked: a request panicked on purpose\n"),
            "{err}"
        );
        assert!(state.poisoned());

        let started = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    assert_eq!(
                        execute_captured(boot(), &state),
                        Err(NotServed::Unavailable)
                    )
                });
            }
        });
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "{:?}",
            started.elapsed()
        );
        assert!(matches!(
            state.reopen(Duration::from_secs(1)),
            Err(ReopenError::Poisoned)
        ));

        // Every other reference gone, the database is still open and locked.
        let weak = Arc::downgrade(held.db());
        drop(held);
        drop(state);
        assert!(weak.upgrade().is_some());
        assert!(WarmDb::open(&db).is_err());
    }

    /// One reopen runs at a time, and while it waits for the old handle the
    /// requests are refused rather than queued behind it. A reopen that
    /// times out keeps the old handle serving.
    #[test]
    fn a_reopen_runs_once_and_never_holds_up_requests() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let boot = || request(&["boot"], &proj, None, None);
        assert_eq!(execute_captured(boot(), &state).unwrap().exit, 0);

        let held = state.handle().unwrap();
        let first = Arc::downgrade(held.db());
        assert!(matches!(
            state.reopen(Duration::from_millis(50)),
            Err(ReopenError::InUse)
        ));
        assert_eq!(execute_captured(boot(), &state).unwrap().exit, 0);

        std::thread::scope(|s| {
            let reopen = s.spawn(|| state.reopen(Duration::from_secs(30)));
            // The reopen has taken the handle out and waits on `held`.
            while state.handle().is_some() {
                std::thread::sleep(Duration::from_millis(1));
            }
            let started = Instant::now();
            for _ in 0..4 {
                assert!(matches!(
                    state.reopen(Duration::from_secs(30)),
                    Err(ReopenError::Running)
                ));
                assert_eq!(
                    execute_captured(boot(), &state),
                    Err(NotServed::Unavailable)
                );
            }
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "{:?}",
                started.elapsed()
            );
            drop(held);
            reopen.join().unwrap().unwrap();
        });
        assert!(first.upgrade().is_none(), "the old handle is still open");
        assert_eq!(execute_captured(boot(), &state).unwrap().exit, 0);
    }

    /// A reopen of a database removed from under the open handle fails
    /// rather than create an empty one in its place.
    #[test]
    fn a_reopen_never_creates_a_removed_database() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        let state = WarmState::open(&db).unwrap();
        std::fs::remove_file(&db).unwrap();
        assert!(matches!(
            state.reopen(Duration::from_secs(5)),
            Err(ReopenError::Open(_))
        ));
        assert!(!db.exists());
    }

    /// The host relies on catching a request's panic, so no build may abort
    /// on one (the daemon module also refuses to compile that way).
    #[test]
    fn panics_unwind() {
        assert!(cfg!(panic = "unwind"));
        let manifest: toml::Value =
            toml::from_str(include_str!("../../../../../Cargo.toml")).unwrap();
        let profiles = manifest.get("profile").and_then(toml::Value::as_table);
        for (name, profile) in profiles.into_iter().flatten() {
            let strategy = profile.get("panic").and_then(toml::Value::as_str);
            assert_ne!(strategy, Some("abort"), "[profile.{name}] aborts on panic");
        }
    }

    /// Requests from many threads at once on one handle: each gets its own
    /// command's output, whole and unmixed. Reads print what they print
    /// alone; each write answers with its own record, and every record lands.
    #[test]
    fn parallel_requests_get_their_own_output() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let run = |args: &[&str], stdin: Option<&[u8]>| {
            let mut req = request(args, &proj, None, stdin);
            req.slow_query_log = false;
            execute_captured(req, &state).unwrap()
        };

        let mut reads: Vec<Vec<&str>> = vec![
            vec!["boot"],
            vec!["boot", "--boot-format", "narrative"],
            vec!["recall-for-file", "src/auth.rs"],
        ];
        if cfg!(feature = "checkpoint") {
            reads.push(vec!["checkpoint", "show"]);
        }
        if cfg!(feature = "fts") {
            reads.push(vec!["fts", "cache"]);
            reads.push(vec!["--format", "table", "fts", "token"]);
        }
        if cfg!(feature = "indexer") {
            reads.push(vec!["code-search", "validate token", "--json"]);
        }
        if cfg!(feature = "timeseries") {
            reads.push(vec!["since", "1h"]);
        }
        let alone: Vec<Response> = reads.iter().map(|args| run(args, None)).collect();
        for (args, r) in reads.iter().zip(&alone) {
            assert!(r.exit == 0 && !r.stdout.is_empty(), "{args:?}: {r:?}");
        }
        std::thread::scope(|s| {
            for client in 0..8 {
                let (reads, alone, run) = (&reads, &alone, &run);
                s.spawn(move || {
                    for round in 0..4 {
                        for i in 0..reads.len() {
                            let k = (i + client + round) % reads.len();
                            assert_eq!(run(&reads[k], None), alone[k], "{:?}", reads[k]);
                        }
                    }
                });
            }
        });

        // Writes, with recalls running between them.
        let ids: Vec<String> = std::thread::scope(|s| {
            let clients: Vec<_> = (0..8)
                .map(|client| {
                    let run = &run;
                    s.spawn(move || {
                        let mut ids = Vec::new();
                        for n in 0..4 {
                            let summary = format!("parallel store {client}-{n}");
                            let data = serde_json::json!({ "summary": summary }).to_string();
                            let r = if n % 2 == 0 {
                                run(&["store", "decisions", &data], None)
                            } else {
                                run(&["store", "decisions", "-"], Some(data.as_bytes()))
                            };
                            assert!(r.exit == 0 && r.stderr.is_empty(), "{r:?}");
                            let text = String::from_utf8(r.stdout).unwrap();
                            assert_eq!(text.matches('\n').count(), 1, "{text}");
                            let stored: serde_json::Value = serde_json::from_str(&text).unwrap();
                            assert_eq!(stored["table"], "decisions");
                            ids.push((stored["id"].as_str().unwrap().to_string(), summary));
                            if cfg!(feature = "embed") {
                                let r = run(&["recall", "auth timeout"], None);
                                assert_eq!(r.exit, 0, "{r:?}");
                                let hits: Vec<serde_json::Value> =
                                    serde_json::from_slice(&r.stdout).unwrap();
                                assert!(!hits.is_empty());
                                assert_eq!(
                                    r.stderr,
                                    b"[axil] compact view \xe2\x80\x94 expand any hit with: axil get <id>\n"
                                );
                            }
                        }
                        ids
                    })
                })
                .collect();
            let warm = state.handle().unwrap();
            clients
                .into_iter()
                .flat_map(|c| c.join().unwrap())
                .map(|(id, summary)| {
                    let record_id = axil_core::RecordId::from_string(&id).unwrap();
                    let record = warm.db().get(&record_id).unwrap().unwrap();
                    assert_eq!(record.data["summary"], summary.as_str());
                    id
                })
                .collect()
        });
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!((ids.len(), unique.len()), (32, 32));
    }

    /// A database whose open prints, here a vector row the load skips with
    /// a hint to heal it, is not served, and is left closed for the
    /// commands the client runs itself, which print the line every time.
    #[cfg(feature = "vector")]
    #[test]
    fn a_database_whose_open_prints_is_not_served() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let db = proj.join(".axil/memory.axil");
        // The store a detected open attaches: the default model's, with its
        // embedder when the build has one.
        #[cfg(feature = "embed")]
        let dims = {
            let model = crate::resolve_embedding_model(&db);
            if !axil_vector::download::is_model_available(&model) {
                eprintln!("skipped: {} is not installed", model.name());
                return;
            }
            model.dimensions()
        };
        #[cfg(not(feature = "embed"))]
        let dims = 4;
        drop(axil_vector::VectorEngine::open(&db, dims).unwrap());
        {
            let vec_file = axil_core::companion_path(&db, ".vec");
            let vectors = redb::Database::open(vec_file).unwrap();
            let txn = vectors.begin_write().unwrap();
            {
                let table: redb::TableDefinition<&str, &[u8]> =
                    redb::TableDefinition::new("vectors");
                let mut table = txn.open_table(table).unwrap();
                table.insert("zzzz", b"bad".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        for _ in 0..2 {
            let err = WarmState::open(&db).err().expect("served");
            let printed = &err
                .downcast_ref::<OpenPrinted>()
                .unwrap_or_else(|| panic!("{err:#}"))
                .0;
            assert!(
                printed.contains("unloadable") && printed.contains("axil heal --reindex"),
                "{printed}"
            );
        }
        let (_, err, code) = one_shot_at(&["boot"], None, &proj, None);
        let err = String::from_utf8_lossy(&err);
        assert!(code == 0 && err.contains("axil heal --reindex"), "{err}");
    }

    /// A warning an engine prints during the command reaches the request's
    /// stderr, where the one-shot binary prints it, not the host's.
    #[cfg(feature = "graph")]
    #[test]
    fn an_engine_warning_reaches_the_requests_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let fixture = build_fixture(&tmp.path().join("fixture"));
        // An undecodable edge whose id sorts after every real one, as if an
        // older binary had added it. The graph engine meets it on its first
        // operation, inside recall, and says it is removing it.
        {
            let graph = redb::Database::open(fixture.join(".axil/memory.axil.graph")).unwrap();
            let txn = graph.begin_write().unwrap();
            {
                let edges: redb::TableDefinition<&str, &[u8]> = redb::TableDefinition::new("edges");
                let mut edges = txn.open_table(edges).unwrap();
                edges.insert("zzzz", b"garbage".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }
        let proj = tmp.path().join("work/proj");
        let db_path = proj.join(".axil/memory.axil");
        let cwd = || proj.canonicalize().unwrap();
        let results = check_parity_with(
            &fixture,
            &proj,
            &[case("recall", &["recall", "auth timeout"])],
            |c, args| {
                let state = WarmState::open(&db_path).unwrap();
                served(request(args, &cwd(), None, c.stdin), &state)
            },
            |c, args| one_shot_at(args, c.stdin, &cwd(), None),
        );
        let err = String::from_utf8_lossy(&results[0].1 .1);
        assert!(err.contains("warning: removing corrupt edge zzzz"), "{err}");
    }
}
