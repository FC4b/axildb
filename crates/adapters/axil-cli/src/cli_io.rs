//! Capturable stdio for CLI commands.
//!
//! Every byte a command prints and every byte it reads from stdin goes
//! through this module, so a command can run in-process with its output
//! collected into buffers instead of landing on the process's fds.
//!
//! A one-shot `axil` run installs no request context. The print macros then
//! write their arguments to the locked process stdout / stderr, so the bytes
//! and the line buffering are exactly a bare `println!`'s.
//!
//! A reader that goes away (`axil recall … | head`) is the one difference.
//! Rust ignores SIGPIPE, so the write fails with `BrokenPipe`, where
//! `println!` would panic. A panic is the wrong way out: unwinding still drops
//! the database, but redb skips its clean shutdown on a panicking thread, so
//! the next open finds the stores "not closed cleanly" and repairs them.
//! Restoring SIGPIPE's default action would be worse: the signal kills the
//! process before anything closes. So the stream is marked closed instead
//! ([`stdout_closed`]), and that write and every later one to it is
//! discarded. The command runs to its end, the database closes normally, and
//! the process exits with the command's own code: 0 for a command that
//! succeeded. That is the convention ripgrep and fd follow, and it fits a
//! reader that stopped because it had what it wanted. 141 (what a process
//! killed by SIGPIPE reports) would fail every `set -o pipefail` script that
//! pipes `axil` into `head`. Running on also means a closed pipe never cuts a
//! write command off halfway. The loops that print to stdout without end
//! (`stats --watch`, `ingest-scip --watch`) check [`stdout_closed`] and stop.
//! Any other write error panics with `println!`'s message.
//!
//! [`capture`] installs a [`ReqCtx`] on the current thread for the duration
//! of a closure. Output then appends to the context's buffers and stdin is
//! served from its bytes.
//!
//! The context can also carry a database the host already has open, a
//! [`WarmDb`]. The hot commands get their handle through [`hot_db`], which
//! returns that handle instead of opening the file again, or refuses with
//! [`WarmLacks`] when the handle is missing something the command's own open
//! would have provided. [`RECALL_SLOTS`] bounds the recall workers a deadline
//! leaves running, which a one-shot run never outlives but a long-lived host
//! would accumulate.

use std::cell::RefCell;
use std::fmt;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axil_core::Axil;

thread_local! {
    static REQ: RefCell<Option<ReqCtx>> = const { RefCell::new(None) };
}

/// Set once a write finds the process's stdout closed by its reader.
static STDOUT_CLOSED: AtomicBool = AtomicBool::new(false);
/// Stdin the process read ahead to send to the database's daemon, which then
/// did not run the command: the command reads these bytes instead.
static READ_AHEAD: Mutex<Option<Vec<u8>>> = Mutex::new(None);
/// Set once a write finds the process's stderr closed by its reader.
static STDERR_CLOSED: AtomicBool = AtomicBool::new(false);

/// I/O state for one command run in-process.
pub struct ReqCtx {
    /// Bytes the command wrote to stdout.
    pub out: Vec<u8>,
    /// Bytes the command wrote to stderr.
    pub err: Vec<u8>,
    /// The order the command wrote in (see [`Captured::order`]).
    pub order: Vec<(bool, usize)>,
    /// Piped stdin. `None` stands for a terminal: nothing was piped.
    pub stdin: Option<Vec<u8>>,
    /// Whether the command records slow queries in the database's log.
    pub slow_query_log: bool,
    /// An open database the command uses instead of opening its own, when
    /// it is the command's database. See [`hot_db`].
    pub warm: Option<WarmDb>,
}

impl Default for ReqCtx {
    fn default() -> Self {
        Self {
            out: Vec::new(),
            err: Vec::new(),
            order: Vec::new(),
            stdin: None,
            slow_query_log: true,
            warm: None,
        }
    }
}

/// Run `f` with `ctx` installed as this thread's request context, and return
/// what it wrote to stdout and stderr along with its result.
///
/// The previously installed context, if any, is restored afterwards, also
/// when `f` panics (the panic carries on, and what `f` wrote is dropped).
#[cfg_attr(not(test), allow(dead_code))] // the in-process host is its first non-test caller
pub fn capture<R>(ctx: ReqCtx, f: impl FnOnce() -> R) -> (Vec<u8>, Vec<u8>, R) {
    let (printed, result) = capture_in_order(ctx, f);
    (printed.out, printed.err, result)
}

/// What a captured command wrote.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Captured {
    /// Its stdout.
    pub out: Vec<u8>,
    /// Its stderr.
    pub err: Vec<u8>,
    /// The order it wrote them in: runs of so many bytes of its stderr
    /// (`true`) or its stdout, which cover both, two runs in a row never to
    /// the same stream. A terminal showing both streams shows them so.
    pub order: Vec<(bool, usize)>,
}

impl Captured {
    /// Append `bytes` to stderr (`to_err`) or stdout, after what is there.
    #[cfg_attr(not(unix), allow(dead_code))] // the daemon's
    pub fn push(&mut self, to_err: bool, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let buf = if to_err { &mut self.err } else { &mut self.out };
        buf.extend_from_slice(bytes);
        push_run(&mut self.order, to_err, bytes.len());
    }
}

/// Count `len` more bytes to stderr (`to_err`) or stdout into `order`.
fn push_run(order: &mut Vec<(bool, usize)>, to_err: bool, len: usize) {
    match order.last_mut() {
        Some((err, n)) if *err == to_err => *n += len,
        _ => order.push((to_err, len)),
    }
}

/// [`capture`], with the order `f` wrote its stdout and stderr in.
pub fn capture_in_order<R>(ctx: ReqCtx, f: impl FnOnce() -> R) -> (Captured, R) {
    struct Restore(Option<ReqCtx>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            REQ.with(|r| *r.borrow_mut() = prev);
        }
    }

    let _restore = Restore(REQ.with(|r| r.borrow_mut().replace(ctx)));
    let result = f();
    let ctx = REQ
        .with(|r| r.borrow_mut().take())
        .expect("request context removed while its capture was running");
    let printed = Captured {
        out: ctx.out,
        err: ctx.err,
        order: ctx.order,
    };
    (printed, result)
}

fn capturing() -> bool {
    REQ.with(|r| r.borrow().is_some())
}

/// Append to the installed request's stdout or stderr buffer. Returns
/// `false`, writing nothing, when no request is installed.
fn append(to_err: bool, bytes: &[u8]) -> bool {
    REQ.with(|r| match r.borrow_mut().as_mut() {
        Some(ctx) => {
            let buf = if to_err { &mut ctx.err } else { &mut ctx.out };
            buf.extend_from_slice(bytes);
            if !bytes.is_empty() {
                push_run(&mut ctx.order, to_err, bytes.len());
            }
            true
        }
        None => false,
    })
}

/// Format into the request's buffer. The text is formatted before the
/// context is borrowed, so a `Display` impl that prints cannot re-enter it.
fn append_fmt(to_err: bool, args: fmt::Arguments<'_>, newline: bool) -> bool {
    if !capturing() {
        return false;
    }
    let mut text = fmt::format(args);
    if newline {
        text.push('\n');
    }
    append(to_err, text.as_bytes())
}

/// Whether a write found the process's stdout closed by its reader. Output
/// printed since then was discarded. Never set by a captured request.
pub fn stdout_closed() -> bool {
    STDOUT_CLOSED.load(Ordering::Relaxed)
}

/// Write to the process's stdout or stderr the way `print!` / `eprint!` do,
/// except for a reader that has gone away: the stream is marked closed and
/// this write and every later one to it is dropped. See the module docs.
fn write_process(to_err: bool, args: fmt::Arguments<'_>, newline: bool) {
    write_process_with(to_err, |w| {
        if newline {
            writeln!(w, "{args}")
        } else {
            write!(w, "{args}")
        }
    });
}

/// Write `bytes` to the process's stdout or stderr as [`write_process`]
/// writes: what a command the database's daemon ran printed, byte for byte.
#[cfg(unix)]
pub fn write_process_bytes(to_err: bool, bytes: &[u8]) {
    write_process_with(to_err, |w| w.write_all(bytes).and_then(|()| w.flush()));
}

/// Write what a command the database's daemon ran printed to the process's
/// stdout and stderr, in the order it printed it (see [`Captured::order`]).
/// What `order` leaves out, which it should not, follows: stdout, then
/// stderr.
#[cfg(unix)]
pub fn write_process_in_order(out: &[u8], err: &[u8], order: &[(bool, usize)]) {
    let (mut out, mut err) = (out, err);
    for &(to_err, len) in order {
        let from = if to_err { &mut err } else { &mut out };
        let bytes: &[u8] = from;
        let (run, rest) = bytes.split_at(len.min(bytes.len()));
        *from = rest;
        if !run.is_empty() {
            write_process_bytes(to_err, run);
        }
    }
    for (to_err, rest) in [(false, out), (true, err)] {
        if !rest.is_empty() {
            write_process_bytes(to_err, rest);
        }
    }
}

fn write_process_with(to_err: bool, write: impl FnOnce(&mut dyn Write) -> io::Result<()>) {
    let (closed, label) = if to_err {
        (&STDERR_CLOSED, "stderr")
    } else {
        (&STDOUT_CLOSED, "stdout")
    };
    if closed.load(Ordering::Relaxed) {
        return;
    }
    let written = if to_err {
        write(&mut io::stderr().lock())
    } else {
        write(&mut io::stdout().lock())
    };
    match written {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => closed.store(true, Ordering::Relaxed),
        Err(e) => panic!("failed printing to {label}: {e}"),
    }
}

/// Backs [`outln!`] and [`outp!`].
#[doc(hidden)]
pub fn out_fmt(args: fmt::Arguments<'_>, newline: bool) {
    if !append_fmt(false, args, newline) {
        write_process(false, args, newline);
    }
}

/// Backs [`errln!`] and [`errp!`].
#[doc(hidden)]
pub fn err_fmt(args: fmt::Arguments<'_>, newline: bool) {
    if !append_fmt(true, args, newline) {
        write_process(true, args, newline);
    }
}

/// Send the engines' diagnostic lines ([`axil_core::diagln!`]) through
/// [`err_fmt`], so a command run in-process gets the warnings its engines
/// print on its own stderr. Without a request they go to the process's
/// stderr as before. Setting it again does nothing.
pub fn install_diagnostic_sink() {
    axil_core::diag::set_diagnostic_sink(|args| err_fmt(args, true));
}

/// `println!` through the capture layer.
macro_rules! outln {
    () => {
        $crate::cli_io::out_fmt(format_args!(""), true)
    };
    ($($arg:tt)*) => {
        $crate::cli_io::out_fmt(format_args!($($arg)*), true)
    };
}

/// `print!` through the capture layer.
macro_rules! outp {
    ($($arg:tt)*) => {
        $crate::cli_io::out_fmt(format_args!($($arg)*), false)
    };
}

/// `eprintln!` through the capture layer.
macro_rules! errln {
    () => {
        $crate::cli_io::err_fmt(format_args!(""), true)
    };
    ($($arg:tt)*) => {
        $crate::cli_io::err_fmt(format_args!($($arg)*), true)
    };
}

/// `eprint!` through the capture layer.
macro_rules! errp {
    ($($arg:tt)*) => {
        $crate::cli_io::err_fmt(format_args!($($arg)*), false)
    };
}

/// A [`Write`] handle on the command's stdout, for output streamed through a
/// writer rather than printed: the request's buffer when one is installed,
/// the locked process stdout otherwise. A closed stdout is handled as the
/// print macros handle it: the bytes are dropped and the write reports
/// success, so the command runs to its end.
pub struct Stdout(Option<io::StdoutLock<'static>>);

/// The command's stdout as a [`Write`] handle; see [`Stdout`].
pub fn stdout() -> Stdout {
    if capturing() {
        Stdout(None)
    } else {
        Stdout(Some(io::stdout().lock()))
    }
}

/// Turn a write to the process stdout that found its reader gone into a
/// success worth `dropped`, marking stdout closed.
fn unless_closed<T>(result: io::Result<T>, dropped: T) -> io::Result<T> {
    match result {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
            STDOUT_CLOSED.store(true, Ordering::Relaxed);
            Ok(dropped)
        }
        other => other,
    }
}

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.0 {
            Some(_) if stdout_closed() => Ok(buf.len()),
            Some(lock) => unless_closed(lock.write(buf), buf.len()),
            None if append(false, buf) => Ok(buf.len()),
            None if stdout_closed() => Ok(buf.len()),
            None => unless_closed(io::stdout().write(buf), buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.0 {
            Some(_) if stdout_closed() => Ok(()),
            Some(lock) => unless_closed(lock.flush(), ()),
            None => Ok(()),
        }
    }
}

/// Take up to `limit` bytes of the request's stdin, or `None` when no
/// request is installed. A request without stdin reads as empty.
fn take_request_stdin(limit: u64) -> Option<Vec<u8>> {
    REQ.with(|r| {
        r.borrow_mut().as_mut().map(|ctx| match ctx.stdin.as_mut() {
            Some(data) => {
                let n = usize::try_from(limit).unwrap_or(usize::MAX).min(data.len());
                data.drain(..n).collect()
            }
            None => Vec::new(),
        })
    })
}

/// Read stdin to a string, stopping after `limit` bytes:
/// `io::stdin().take(limit).read_to_string(..)`, with the same errors
/// (invalid UTF-8 is `InvalidData`) when the bytes come from a request.
pub fn read_stdin(limit: u64) -> io::Result<String> {
    read_stdin_or(limit, io::stdin)
}

/// [`read_stdin`], with `process` giving the process's stdin. Stdin read
/// ahead is read first, then the rest of the process's.
fn read_stdin_or<R: Read>(limit: u64, process: impl FnOnce() -> R) -> io::Result<String> {
    let mut buf = String::new();
    match take_request_stdin(limit) {
        Some(bytes) => bytes.as_slice().read_to_string(&mut buf)?,
        None => match take_read_ahead(limit) {
            Some(bytes) => {
                let left = limit.saturating_sub(bytes.len() as u64);
                bytes
                    .as_slice()
                    .chain(process().take(left))
                    .read_to_string(&mut buf)?
            }
            None => process().take(limit).read_to_string(&mut buf)?,
        },
    };
    Ok(buf)
}

/// Stdin as a buffered reader, for input consumed as a stream.
pub fn stdin_reader() -> Box<dyn io::BufRead> {
    stdin_reader_or(|| io::stdin().lock())
}

/// [`stdin_reader`], with `process` giving the process's stdin.
fn stdin_reader_or<R: io::BufRead + 'static>(process: impl FnOnce() -> R) -> Box<dyn io::BufRead> {
    if let Some(bytes) = take_request_stdin(u64::MAX) {
        return Box::new(io::Cursor::new(bytes));
    }
    match take_read_ahead(u64::MAX) {
        Some(bytes) => Box::new(io::Cursor::new(bytes).chain(process())),
        None => Box::new(process()),
    }
}

/// Read the process's stdin to its end, to send to the database's daemon,
/// but no more than `cap` bytes and one: more than `cap` shows as more.
/// Should the daemon not run the command, [`put_back_stdin`] hands the bytes
/// to the command.
#[cfg(unix)]
pub fn read_stdin_ahead(cap: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    io::stdin()
        .take(u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1))
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Give back stdin [`read_stdin_ahead`] read: the command reads these bytes,
/// and then the rest of stdin, as if nothing had been read ahead.
#[cfg(unix)]
pub fn put_back_stdin(bytes: Vec<u8>) {
    *READ_AHEAD.lock().unwrap_or_else(PoisonError::into_inner) = Some(bytes);
}

/// Take up to `limit` bytes of the stdin read ahead, or `None` when none was.
fn take_read_ahead(limit: u64) -> Option<Vec<u8>> {
    let mut ahead = READ_AHEAD.lock().unwrap_or_else(PoisonError::into_inner);
    let data = ahead.as_mut()?;
    let n = usize::try_from(limit).unwrap_or(usize::MAX).min(data.len());
    Some(data.drain(..n).collect())
}

/// Whether stdin is a terminal, i.e. nothing was piped in. A request
/// counts as a terminal when it carries no stdin.
pub fn stdin_is_terminal() -> bool {
    REQ.with(|r| r.borrow().as_ref().map(|ctx| ctx.stdin.is_none()))
        .unwrap_or_else(|| io::stdin().is_terminal())
}

/// Whether stderr is a terminal. Never for a request: its stderr is a
/// buffer, so progress bars drawn on the process's stderr must stay off.
#[cfg_attr(not(feature = "indexer"), allow(dead_code))] // only indexer commands draw progress
pub fn stderr_is_terminal() -> bool {
    !capturing() && io::stderr().is_terminal()
}

/// Whether the command should record slow queries in the database's log.
/// True unless the installed request turned it off.
pub fn slow_query_log_enabled() -> bool {
    REQ.with(|r| r.borrow().as_ref().is_none_or(|ctx| ctx.slow_query_log))
}

/// The identity of the database at `path`: its directory, canonicalized,
/// joined with its file name as given.
///
/// The companion files (`.vec`, `.fts`, `.ts`, ...) are named after the path as
/// given, so two paths are one database only when they agree on this, not
/// merely when they resolve to the same core file: a symlink to the core file
/// under another name has companions of its own.
pub fn db_key(path: &Path) -> io::Result<PathBuf> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} names no database file", path.display()),
        )
    })?;
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    Ok(std::fs::canonicalize(dir)?.join(name))
}

/// A database a long-lived host keeps open for the commands it runs
/// in-process, installed through [`ReqCtx::warm`].
///
/// The only way to make one is [`WarmDb::open`], which opens it exactly as a
/// hot command's own open finds the database: every engine whose store
/// exists, no LLM provider, and the installed plugins registered as a
/// one-shot `boot` registers them. A store that appears after the open is
/// not attached: commands that need it get [`WarmLacks`].
#[derive(Clone)]
pub struct WarmDb {
    db: Arc<Axil>,
    key: PathBuf,
}

impl WarmDb {
    /// Open the database at `path` for in-process commands.
    #[cfg_attr(not(test), allow(dead_code))] // the in-process host is its first non-test caller
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let key = db_key(path)?;
        let db = Arc::new(crate::open_with_all_detected(path)?);
        crate::register_installed_plugins(&db, path);
        Ok(Self { db, key })
    }

    /// The open handle.
    #[cfg_attr(not(test), allow(dead_code))] // read by tests until a host uses it
    pub fn db(&self) -> &Arc<Axil> {
        &self.db
    }

    /// The database's identity (see [`db_key`]): a command is for this
    /// handle when its database path has this key.
    #[cfg_attr(not(test), allow(dead_code))] // the in-process host is its first non-test caller
    pub fn key(&self) -> &Path {
        &self.key
    }

    /// Close the database, so another process can open it, and return the
    /// path to reopen it from with [`WarmDb::open`].
    ///
    /// The database closes when the last reference to the handle goes, so
    /// this waits, up to `timeout`, for every other one: clones of this
    /// `WarmDb` in requests still running, the handles their commands took
    /// from [`hot_db`], and recall workers still searching past their
    /// deadline. The host must have stopped installing it in new requests.
    /// After `timeout` the handle comes back unchanged.
    #[cfg_attr(not(test), allow(dead_code))] // the in-process host is its first non-test caller
    pub fn release(self, timeout: Duration) -> Result<PathBuf, WarmDb> {
        let deadline = Instant::now() + timeout;
        let mut db = self.db;
        loop {
            match Arc::try_unwrap(db) {
                Ok(db) => {
                    drop(db);
                    return Ok(self.key);
                }
                Err(shared) if Instant::now() >= deadline => {
                    return Err(WarmDb {
                        db: shared,
                        key: self.key,
                    })
                }
                Err(shared) => {
                    db = shared;
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }
}

/// The installed request's open database, when `path` names that database
/// (see [`db_key`]). `None` when no request is installed, it carries no
/// handle, or `path` is another database (or can't be resolved).
pub fn warm_db(path: &Path) -> Option<Arc<Axil>> {
    let warm = REQ.with(|r| r.borrow().as_ref()?.warm.clone())?;
    (db_key(path).ok()? == warm.key).then_some(warm.db)
}

/// Something a command needs from its database handle beyond the core store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // which variants a build constructs depends on its features
pub enum Need {
    /// The default vector store.
    Vectors,
    /// A text embedder.
    Embedder,
    /// The full-text index.
    FullText,
    /// The time-series index.
    TimeSeries,
    /// An LLM provider.
    Llm,
}

impl Need {
    /// Whether `db` has it.
    pub fn met_by(self, db: &Axil) -> bool {
        match self {
            Need::Vectors => db.has_vector_index(),
            Need::Embedder => db.has_embedder(),
            Need::FullText => db.has_fts_index(),
            Need::TimeSeries => db.has_timeseries_index(),
            Need::Llm => db.has_llm(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Need::Vectors => "vector store",
            Need::Embedder => "embedder",
            Need::FullText => "full-text index",
            Need::TimeSeries => "time-series index",
            Need::Llm => "LLM provider",
        }
    }
}

/// The request's open database can't serve a command: it lacks something
/// the command's own open would have attached or created. Nothing has been
/// written when a command returns this, so a host can run the command again
/// in a process of its own, which opens the database the one-shot way. That
/// process can't open the database while the host holds it: the host first
/// calls [`WarmDb::release`], then runs the command, then reopens with
/// [`WarmDb::open`], which attaches what the command created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmLacks(pub Need);

impl fmt::Display for WarmLacks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the open database handle has no {}; run the command in its own process",
            self.0.name()
        )
    }
}

impl std::error::Error for WarmLacks {}

/// The handle a command runs against: the request's open database when it is
/// the one at `path` (see [`warm_db`]), otherwise a new handle from `open`.
///
/// A warm handle must meet every one of `needs`, or this returns
/// [`WarmLacks`] before the command has done anything. `open` is the
/// command's one-shot open, which attaches or creates what it needs itself.
///
/// Without such a request this is exactly `open`, so a one-shot run opens and
/// closes the database as it always has. A warm handle is shared: the command
/// neither opens nor closes it, and only drops its own reference.
pub fn hot_db(
    path: &Path,
    needs: &[Need],
    open: impl FnOnce() -> anyhow::Result<Axil>,
) -> anyhow::Result<Arc<Axil>> {
    let Some(db) = warm_db(path) else {
        return open().map(Arc::new);
    };
    match needs.iter().find(|need| !need.met_by(&db)) {
        Some(&need) => Err(WarmLacks(need).into()),
        None => Ok(db),
    }
}

/// A process-wide cap on threads that run a recall past its caller's
/// deadline.
///
/// `recall --timeout-ms` runs the search on a worker and stops waiting at the
/// deadline; the worker carries on until the search finishes. A one-shot
/// process exits right after and takes the worker with it. A long-lived
/// process keeps it, so without a cap each timed-out request would leave
/// another thread behind.
#[cfg_attr(not(feature = "embed"), allow(dead_code))] // recall, the one taker, needs `embed`
pub struct RecallSlots {
    live: Mutex<usize>,
    freed: Condvar,
    cap: usize,
}

/// The recall worker cap shared by every command in this process.
#[cfg_attr(not(feature = "embed"), allow(dead_code))]
pub static RECALL_SLOTS: RecallSlots = RecallSlots::new(4);

/// A taken [`RecallSlots`] slot. The worker holds it while it runs, and
/// dropping it frees the slot.
#[cfg_attr(not(feature = "embed"), allow(dead_code))]
pub struct RecallSlot(&'static RecallSlots);

#[cfg_attr(not(feature = "embed"), allow(dead_code))]
impl RecallSlots {
    /// A cap of `cap` concurrent workers.
    pub const fn new(cap: usize) -> Self {
        Self {
            live: Mutex::new(0),
            freed: Condvar::new(),
            cap,
        }
    }

    /// Take a slot, waiting for one to be freed until `deadline`. `None` once
    /// `deadline` has passed, even with a slot free by then: the caller has
    /// stopped waiting for the search the slot would run.
    pub fn take_by(&'static self, deadline: Instant) -> Option<RecallSlot> {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let now = Instant::now();
            if now > deadline {
                if *live < self.cap {
                    // The wakeup may have been meant for this waiter: pass it on.
                    self.freed.notify_one();
                }
                return None;
            }
            if *live < self.cap {
                *live += 1;
                return Some(RecallSlot(self));
            }
            live = self
                .freed
                .wait_timeout(live, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// How many slots are taken right now.
    #[cfg_attr(not(test), allow(dead_code))] // read by tests until a host reports it
    pub fn live(&self) -> usize {
        *self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for RecallSlot {
    fn drop(&mut self) {
        *self.0.live.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        self.0.freed.notify_one();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn piped(bytes: &[u8]) -> ReqCtx {
        ReqCtx {
            stdin: Some(bytes.to_vec()),
            ..ReqCtx::default()
        }
    }

    #[test]
    fn capture_collects_each_stream_byte_for_byte() {
        let (out, err, n) = capture(ReqCtx::default(), || {
            outln!("a {}", 1);
            outp!("b");
            outln!();
            errp!("c{:>3}", "x");
            errln!("{:#}", anyhow::anyhow!("outer").context("ctx"));
            7
        });
        assert_eq!(out, b"a 1\nb\n");
        assert_eq!(err, b"c  xctx: outer\n");
        assert_eq!(n, 7);
    }

    /// The order the streams were written in is kept, a run per stream
    /// switch, and covers every byte.
    #[test]
    fn capture_keeps_the_order_the_streams_were_written_in() {
        let (printed, ()) = capture_in_order(ReqCtx::default(), || {
            outln!("one");
            outp!("two ");
            errln!("warning");
            errp!("");
            outln!("three");
            errp!("x");
        });
        assert_eq!(printed.out, b"one\ntwo three\n");
        assert_eq!(printed.err, b"warning\nx");
        assert_eq!(
            printed.order,
            [(false, 8), (true, 8), (false, 6), (true, 1)]
        );
        let mut more = printed;
        more.push(true, b"y\n");
        more.push(false, b"");
        assert_eq!(more.order.last(), Some(&(true, 3)));
    }

    #[test]
    fn capture_restores_the_outer_request() {
        let (out, _, inner) = capture(ReqCtx::default(), || {
            outp!("outer ");
            let inner = capture(ReqCtx::default(), || outp!("inner"));
            outp!("again");
            inner
        });
        assert_eq!(out, b"outer again");
        assert_eq!(inner.0, b"inner");
        assert!(!capturing());
    }

    #[test]
    fn capture_restores_after_a_panic() {
        let caught = std::panic::catch_unwind(|| {
            capture(ReqCtx::default(), || {
                panic!("boom");
            })
        });
        assert!(caught.is_err());
        assert!(!capturing());
    }

    #[test]
    fn stdout_writer_appends_to_the_request() {
        let (out, _, ()) = capture(ReqCtx::default(), || {
            let mut w = io::BufWriter::new(stdout());
            writeln!(w, "{{\"a\":1}}").unwrap();
            w.flush().unwrap();
        });
        assert_eq!(out, b"{\"a\":1}\n");
    }

    #[test]
    fn read_stdin_honors_the_limit_and_consumes_what_it_reads() {
        let (_, _, (first, rest, after)) = capture(piped(b"hello world"), || {
            let first = read_stdin(5).unwrap();
            let rest = read_stdin(u64::MAX).unwrap();
            (first, rest, read_stdin(u64::MAX).unwrap())
        });
        assert_eq!(first, "hello");
        assert_eq!(rest, " world");
        assert_eq!(after, "");
    }

    #[test]
    fn read_stdin_rejects_invalid_utf8_like_read_to_string() {
        let (_, _, res) = capture(piped(b"ok\xff"), || read_stdin(u64::MAX));
        let err = res.unwrap_err();
        let mut s = String::new();
        let std_err = (&b"ok\xff"[..]).read_to_string(&mut s).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(err.to_string(), std_err.to_string());
    }

    #[test]
    fn a_request_without_stdin_is_a_terminal_that_reads_empty() {
        let (_, _, (tty, text)) = capture(ReqCtx::default(), || {
            (stdin_is_terminal(), read_stdin(u64::MAX).unwrap())
        });
        assert!(tty);
        assert_eq!(text, "");
        let (_, _, tty) = capture(piped(b""), stdin_is_terminal);
        assert!(!tty, "an empty pipe is still a pipe");
        let (_, _, tty) = capture(ReqCtx::default(), stderr_is_terminal);
        assert!(!tty);
    }

    #[test]
    fn slow_query_log_is_on_unless_the_request_turns_it_off() {
        assert!(slow_query_log_enabled());
        let (_, _, on) = capture(ReqCtx::default(), slow_query_log_enabled);
        assert!(on);
        let off = ReqCtx {
            slow_query_log: false,
            ..ReqCtx::default()
        };
        let (_, _, on) = capture(off, slow_query_log_enabled);
        assert!(!on);
    }

    #[test]
    fn stdin_reader_streams_the_request_bytes() {
        let (_, _, lines) = capture(piped(b"a\nb\n"), || {
            use std::io::BufRead;
            stdin_reader()
                .lines()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
        });
        assert_eq!(lines, ["a", "b"]);
    }

    /// Stdin read ahead for the database's daemon and given back is what
    /// the command reads, by either reader, and then the rest of the
    /// process's stdin, which was not read ahead.
    #[cfg(unix)]
    #[test]
    fn stdin_given_back_is_read_before_the_process_stdin() {
        let process = |rest: &'static str| move || io::Cursor::new(rest.as_bytes());
        put_back_stdin(b"abc".to_vec());
        assert_eq!(read_stdin_or(2, process("")).unwrap(), "ab");
        let mut rest = String::new();
        stdin_reader_or(process(""))
            .read_to_string(&mut rest)
            .unwrap();
        assert_eq!(rest, "c");
        assert_eq!(read_stdin_or(u64::MAX, process("")).unwrap(), "");

        put_back_stdin(b"abc".to_vec());
        assert_eq!(read_stdin_or(5, process("def")).unwrap(), "abcde");
        put_back_stdin(b"abc".to_vec());
        let mut all = String::new();
        stdin_reader_or(process("def"))
            .read_to_string(&mut all)
            .unwrap();
        assert_eq!(all, "abcdef");
        *READ_AHEAD.lock().unwrap() = None;
    }

    #[test]
    fn hot_db_serves_the_warm_handle_only_for_its_own_database() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("db");
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("a.axil");
        let warm = WarmDb::open(&path).unwrap();
        let ctx = || ReqCtx {
            warm: Some(warm.clone()),
            ..ReqCtx::default()
        };
        let warm_hit = |p: &Path, needs: &[Need]| {
            capture(ctx(), || {
                hot_db(p, needs, || panic!("opened a database the request holds"))
            })
            .2
        };
        let opens = |p: &Path| {
            let (_, _, got) = capture(ctx(), || hot_db(p, &[], || anyhow::bail!("cold open")));
            got.is_err_and(|e| e.to_string() == "cold open")
        };

        // The same database under another spelling: no open.
        let got = warm_hit(&dir.join(".").join("a.axil"), &[]).unwrap();
        assert!(Arc::ptr_eq(&got, warm.db()));
        drop(got);

        // Another database: the command opens its own.
        assert!(opens(&dir.join("b.axil")));

        // A symlinked directory holds the same companion files, so it is the
        // same database. A symlink to the core file under another name has
        // companions of its own, so it is another database.
        #[cfg(unix)]
        {
            let dir_link = tmp.path().join("db-link");
            std::os::unix::fs::symlink(&dir, &dir_link).unwrap();
            let got = warm_hit(&dir_link.join("a.axil"), &[]).unwrap();
            assert!(Arc::ptr_eq(&got, warm.db()));
            drop(got);
            let file_link = dir.join("link.axil");
            std::os::unix::fs::symlink(&path, &file_link).unwrap();
            assert!(opens(&file_link));
        }

        // The handle lacks what the command needs: refused, unopened.
        let Err(err) = warm_hit(&path, &[Need::FullText]) else {
            panic!("served a command the handle lacks a store for");
        };
        assert_eq!(
            err.downcast_ref::<WarmLacks>(),
            Some(&WarmLacks(Need::FullText))
        );

        // No request: always the command's own open.
        let mut opened = false;
        let got = hot_db(&path, &[Need::FullText], || {
            opened = true;
            Err(anyhow::anyhow!("cold open"))
        });
        assert!(opened && got.is_err());
        assert!(warm_db(&path).is_none());
        assert_eq!(
            Arc::strong_count(warm.db()),
            1,
            "a request's clone outlived it"
        );
    }

    #[test]
    fn recall_slots_cap_holders_and_hand_a_freed_slot_to_a_waiter() {
        static SLOTS: RecallSlots = RecallSlots::new(2);
        let soon = || Instant::now() + Duration::from_secs(1);
        let a = SLOTS.take_by(soon()).unwrap();
        // Slots are per process, not per thread.
        let b = std::thread::spawn(move || SLOTS.take_by(soon()));
        let b = b.join().unwrap().unwrap();
        assert_eq!(SLOTS.live(), 2);

        // All taken: wait for the deadline, then give up.
        let start = Instant::now();
        assert!(SLOTS.take_by(start + Duration::from_millis(50)).is_none());
        assert!(start.elapsed() >= Duration::from_millis(50));

        // A slot freed during the wait goes to the waiter, long before its
        // deadline.
        let start = Instant::now();
        let freer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            drop(a);
        });
        let c = SLOTS.take_by(start + Duration::from_secs(60));
        assert!(c.is_some() && start.elapsed() < Duration::from_secs(30));
        freer.join().unwrap();
        drop((b, c));
        assert_eq!(SLOTS.live(), 0);
    }

    #[test]
    fn a_slot_freed_after_the_deadline_is_not_taken() {
        static SLOTS: RecallSlots = RecallSlots::new(1);
        let soon = Instant::now() + Duration::from_secs(1);
        let held = SLOTS.take_by(soon).unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(50);
        let waiter = std::thread::spawn(move || SLOTS.take_by(deadline).is_some());

        // Hold the lock across the waiter's deadline and free the slot under
        // it: the waiter's timed-out wait can only return once the slot is
        // free, and must still give up.
        std::thread::sleep(Duration::from_millis(20));
        let mut live = SLOTS.live.lock().unwrap();
        while Instant::now() <= deadline + Duration::from_millis(30) {
            std::thread::sleep(Duration::from_millis(5));
        }
        *live -= 1;
        std::mem::forget(held);
        SLOTS.freed.notify_one();
        drop(live);
        assert!(!waiter.join().unwrap(), "took a slot after its deadline");
        assert_eq!(SLOTS.live(), 0);
    }

    /// `src` with comments and the contents of string and char literals
    /// replaced by spaces. Newlines are kept, so byte offsets still map to
    /// the original lines.
    pub(crate) fn code_only(src: &str) -> Vec<u8> {
        let b = src.as_bytes();
        let mut out = b.to_vec();
        let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
            for c in &mut out[from..to] {
                if *c != b'\n' {
                    *c = b' ';
                }
            }
        };
        let mut i = 0;
        while i < b.len() {
            let start = i;
            let prev_ident = i > 0 && is_ident(b[i - 1]);
            if b[i..].starts_with(b"//") {
                i = b[i..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map_or(b.len(), |p| i + p);
            } else if b[i..].starts_with(b"/*") {
                let mut depth = 0;
                while i < b.len() {
                    if b[i..].starts_with(b"/*") {
                        depth += 1;
                        i += 2;
                    } else if b[i..].starts_with(b"*/") {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
            } else if b[i] == b'r'
                && (!prev_ident || (b[i - 1] == b'b' && (i < 2 || !is_ident(b[i - 2]))))
                && b[i + 1..]
                    .iter()
                    .position(|&c| c != b'#')
                    .map(|p| b[i + 1 + p])
                    == Some(b'"')
            {
                let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
                let mut close = vec![b'"'];
                close.extend(std::iter::repeat_n(b'#', hashes));
                let body = i + 2 + hashes;
                let end = b[body..]
                    .windows(close.len())
                    .position(|w| w == close.as_slice())
                    .expect("unterminated raw string");
                i = body + end + close.len();
            } else if b[i] == b'"' {
                i += 1;
                while b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            } else if b[i] == b'\'' {
                // A char literal ('x', '\n', '…'); otherwise a lifetime.
                let len = if b.get(i + 1) == Some(&b'\\') {
                    b[i + 2..].iter().position(|&c| c == b'\'').map(|p| p + 3)
                } else {
                    let ch = src[i + 1..].chars().next().map_or(1, char::len_utf8);
                    (b.get(i + 1 + ch) == Some(&b'\'')).then_some(ch + 2)
                };
                i += len.unwrap_or(1);
                if len.is_none() {
                    continue;
                }
            } else {
                i += 1;
                continue;
            }
            blank(&mut out, start, i);
        }
        out
    }

    /// Blank every `#[cfg(test)]` / `#[cfg(all(test, ...))]` module: from its
    /// attribute to the first unindented `}` after it.
    pub(crate) fn without_test_modules(code: &mut [u8]) {
        let text = String::from_utf8_lossy(code).into_owned();
        let mut offset = 0;
        let mut test_mod_start = None;
        let mut pending_attr = None;
        for line in text.split_inclusive('\n') {
            let end = offset + line.len();
            if let Some(start) = test_mod_start {
                if line.trim_end() == "}" {
                    for c in &mut code[start..end] {
                        if *c != b'\n' {
                            *c = b' ';
                        }
                    }
                    test_mod_start = None;
                }
            } else if line.starts_with("#[cfg(test)]") || line.starts_with("#[cfg(all(test") {
                pending_attr = Some(offset);
            } else if let Some(attr) = pending_attr.take() {
                if ["mod ", "pub mod ", "pub(crate) mod "]
                    .iter()
                    .any(|m| line.starts_with(m))
                {
                    test_mod_start = Some(attr);
                }
            }
            offset = end;
        }
    }

    /// The lines of `src` that use one of `banned` outside comments, string
    /// literals and test modules.
    fn uses_of(src: &str, banned: &[&str]) -> Vec<usize> {
        let mut code = code_only(src);
        without_test_modules(&mut code);
        let mut lines = Vec::new();
        for needle in banned {
            let n = needle.as_bytes();
            for (pos, w) in code.windows(n.len()).enumerate() {
                // `eprintln!` holds `println!`, and `cli_io::stdout()` holds
                // `io::stdout()`: only a match that starts a token counts.
                let starts_token =
                    pos == 0 || !(code[pos - 1].is_ascii_alphanumeric() || code[pos - 1] == b'_');
                if w == n && starts_token {
                    lines.push(src[..pos].matches('\n').count() + 1);
                }
            }
        }
        lines.sort_unstable();
        lines
    }

    const PRINTS: [&str; 4] = ["println!", "print!", "eprintln!", "eprint!"];

    /// The commands, and the daemon that runs them in-process, print and
    /// read stdin only through this module. `hook_brain.rs` is exempt: the
    /// hook commands never run in-process.
    #[test]
    fn main_rs_and_the_daemon_reach_stdio_only_through_cli_io() {
        let sources = [
            ("main.rs", include_str!("main.rs")),
            ("daemon/mod.rs", include_str!("daemon/mod.rs")),
            ("daemon/exec.rs", include_str!("daemon/exec.rs")),
            ("daemon/control.rs", include_str!("daemon/control.rs")),
            ("daemon/paths.rs", include_str!("daemon/paths.rs")),
            ("daemon/proto.rs", include_str!("daemon/proto.rs")),
            ("daemon/server.rs", include_str!("daemon/server.rs")),
            ("daemon/state.rs", include_str!("daemon/state.rs")),
        ];
        let banned = [
            &PRINTS[..],
            &["io::stdout()", "io::stderr()", "io::stdin()"],
        ]
        .concat();
        let found: Vec<String> = sources
            .iter()
            .flat_map(|(file, src)| {
                uses_of(src, &banned)
                    .into_iter()
                    .map(move |l| format!("{file}:{l}"))
            })
            .collect();
        assert!(
            found.is_empty(),
            "use the cli_io macros and helpers instead:\n{}",
            found.join("\n")
        );
    }

    /// The library crates print only through `axil_core::diagln!`, which a
    /// host running commands in-process points at the request's stderr. The
    /// exceptions print where no request is: recall's ranking (`query.rs`,
    /// and `Axil::recall` in `db.rs`, which runs on recall workers too) and
    /// the sink's own fallback. Adapters are left out: the CLI has the lint
    /// above, and the MCP server and AxilQL print their protocol's output.
    #[test]
    fn the_libraries_print_only_through_the_diagnostic_sink() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let allowed = [
            ("crates/axil-core/src/query.rs", 2),
            ("crates/axil-core/src/db.rs", 1),
            ("crates/axil-core/src/diag.rs", 1),
        ];
        let mut dirs = vec![
            root.join("crates/axil-core/src"),
            root.join("crates/axil-bundle/src"),
            root.join("crates/adapters/axil-runtime/src"),
        ];
        for group in ["crates/engines", "crates/extensions"] {
            for entry in std::fs::read_dir(root.join(group)).unwrap() {
                dirs.push(entry.unwrap().path().join("src"));
            }
        }
        let mut files = 0;
        let mut found = Vec::new();
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                files += 1;
                let src = std::fs::read_to_string(&path).unwrap();
                let file = path
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let lines = uses_of(&src, &PRINTS);
                let allow = allowed
                    .iter()
                    .find(|(f, _)| *f == file)
                    .map_or(0, |(_, n)| *n);
                if lines.len() != allow {
                    found.push(format!("{file}: lines {lines:?}, {allow} allowed"));
                }
            }
        }
        assert!(
            files > 100,
            "read only {files} files under {}",
            root.display()
        );
        assert!(
            found.is_empty(),
            "print through axil_core::diagln! instead:\n{}",
            found.join("\n")
        );
    }

    #[test]
    fn the_lint_sees_through_literals_comments_and_test_modules() {
        let src = "fn a() { outln!(\"println!(x)\"); } // println!\n\
                   /* eprint! /* nested */ */ let r = r#\"print!\"#; let c = '\"';\n\
                   #[cfg(test)]\nmod t {\n    fn b() { println!(); }\n}\n\
                   fn c<'a>() { eprintln!(); }\n";
        let mut code = code_only(src);
        without_test_modules(&mut code);
        let code = String::from_utf8(code).unwrap();
        assert_eq!(code.matches("println!").count(), 1, "{code}");
        assert!(code.contains("fn c<'a>() { eprintln!(); }"), "{code}");
        assert!(!code.contains("print!\""), "{code}");
    }

    /// One-shot vs in-process parity. Each command runs twice against
    /// identical copies of one fixture database: once in a child process on
    /// the real fds with no request installed, the way `axil` runs, and once
    /// in-process under [`capture`]. Both must print the same bytes and exit
    /// with the same code.
    ///
    /// The child is this test binary re-run with only [`one_shot_child`]
    /// selected, so it carries exactly the code under test. The fixture has
    /// no vector store, so no command loads an embedding model.
    pub(crate) mod parity {
        use super::*;
        use crate::{run_cli, Cli};
        use clap::Parser;
        use std::path::Path;
        use std::process::{Command, Stdio};

        const CHILD_ARGS_ENV: &str = "AXIL_CLI_IO_PARITY_ARGS";
        /// Set when the child should wait, after writing [`MARK`], until its
        /// stdin closes before it runs the command.
        const CHILD_WAIT_ENV: &str = "AXIL_CLI_IO_PARITY_WAIT";
        /// Written on both streams before the command runs, so the child's
        /// output can be told apart from the test harness's.
        const MARK: &[u8] = b"\x01axil-cli-io-parity\x01\n";

        /// Run `f` on a thread with `main`'s stack size: in debug builds,
        /// clap's parser for the whole `Command` enum overflows the default
        /// test-thread stack.
        fn on_cli_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
            std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(f)
                .unwrap()
                .join()
                .unwrap()
        }

        pub(crate) fn argv(args: &[&str]) -> Vec<String> {
            std::iter::once("axil")
                .chain(args.iter().copied())
                .map(str::to_string)
                .collect()
        }

        /// Not a test on its own: the one-shot side of the parity test. A
        /// usage error makes clap print and exit inside `parse_from`, exactly
        /// as it does in `axil`.
        #[test]
        fn one_shot_child() {
            let Some(args) = std::env::var_os(CHILD_ARGS_ENV) else {
                return;
            };
            let args: Vec<String> = serde_json::from_str(args.to_str().unwrap()).unwrap();
            for mut stream in [
                Box::new(io::stdout()) as Box<dyn Write>,
                Box::new(io::stderr()),
            ] {
                stream.write_all(MARK).unwrap();
                stream.flush().unwrap();
            }
            if std::env::var_os(CHILD_WAIT_ENV).is_some() {
                io::stdin().read_to_end(&mut Vec::new()).unwrap();
            }
            let code = on_cli_stack(move || run_cli(Cli::parse_from(args)));
            std::process::exit(code);
        }

        /// This test binary, set to run [`one_shot_child`] on `args`.
        fn one_shot_command(args: &[&str]) -> Command {
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.args([
                "cli_io::tests::parity::one_shot_child",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ARGS_ENV, serde_json::to_string(&argv(args)).unwrap());
            cmd
        }

        /// The bytes of `stream` after [`MARK`].
        fn after_mark(stream: &[u8]) -> Option<Vec<u8>> {
            let at = stream.windows(MARK.len()).position(|w| w == MARK)?;
            Some(stream[at + MARK.len()..].to_vec())
        }

        fn one_shot(args: &[&str], stdin: Option<&[u8]>) -> Run {
            one_shot_with(args, stdin, |_| {})
        }

        /// [`one_shot`] in the working directory `cwd`, with `AXIL_DB` set to
        /// `axil_db` or unset.
        pub(crate) fn one_shot_at(
            args: &[&str],
            stdin: Option<&[u8]>,
            cwd: &Path,
            axil_db: Option<&str>,
        ) -> Run {
            one_shot_with(args, stdin, |child| {
                child.current_dir(cwd);
                match axil_db {
                    Some(value) => child.env("AXIL_DB", value),
                    None => child.env_remove("AXIL_DB"),
                };
            })
        }

        fn one_shot_with(
            args: &[&str],
            stdin: Option<&[u8]>,
            setup: impl FnOnce(&mut Command),
        ) -> Run {
            let mut child = one_shot_command(args);
            setup(&mut child);
            let mut child = child
                .stdin(if stdin.is_some() {
                    Stdio::piped()
                } else {
                    Stdio::null()
                })
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            if let Some(bytes) = stdin {
                let mut pipe = child.stdin.take().unwrap();
                let bytes = bytes.to_vec();
                std::thread::spawn(move || pipe.write_all(&bytes).unwrap());
            }
            let output = child.wait_with_output().unwrap();
            let after_mark = |stream: &[u8]| {
                after_mark(stream).unwrap_or_else(|| {
                    panic!(
                        "child never ran: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                })
            };
            (
                after_mark(&output.stdout),
                after_mark(&output.stderr),
                output.status.code().unwrap(),
            )
        }

        /// The in-process side. `stdin: None` matches the child's null stdin:
        /// piped, and empty.
        fn in_process(args: &[&str], stdin: Option<&[u8]>) -> (Vec<u8>, Vec<u8>, i32) {
            let ctx = ReqCtx {
                stdin: Some(stdin.unwrap_or_default().to_vec()),
                ..ReqCtx::default()
            };
            in_process_with(ctx, args)
        }

        fn in_process_with(ctx: ReqCtx, args: &[&str]) -> (Vec<u8>, Vec<u8>, i32) {
            let args = argv(args);
            on_cli_stack(move || {
                capture(ctx, || match Cli::try_parse_from(&args) {
                    Ok(cli) => run_cli(cli),
                    // What `clap::Error::exit` prints, minus the exit.
                    Err(e) => {
                        let text = e.render().to_string();
                        if e.use_stderr() {
                            errp!("{text}");
                        } else {
                            outp!("{text}");
                        }
                        e.exit_code()
                    }
                })
            })
        }

        pub(crate) fn copy_dir(from: &Path, to: &Path) {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                let dest = to.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_dir(&entry.path(), &dest);
                } else {
                    std::fs::copy(entry.path(), dest).unwrap();
                }
            }
        }

        /// Replace what differs between two runs of a writing command: ULIDs
        /// (renumbered in order of appearance, so a repeated id still
        /// matches) and ISO-8601 timestamps.
        fn normalize(bytes: &[u8]) -> String {
            const CROCKFORD: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
            let text = String::from_utf8_lossy(bytes);
            let b = text.as_bytes();
            let mut ids: Vec<&[u8]> = Vec::new();
            let mut out = String::new();
            let mut i = 0;
            while i < b.len() {
                let boundary = i == 0 || !b[i - 1].is_ascii_alphanumeric();
                let ulid = b.get(i..i + 26).filter(|w| {
                    w[0] == b'0'
                        && w.iter().all(|c| CROCKFORD.contains(c))
                        && b.get(i + 26).is_none_or(|c| !c.is_ascii_alphanumeric())
                });
                if let (true, Some(id)) = (boundary, ulid) {
                    let n = ids.iter().position(|seen| *seen == id).unwrap_or_else(|| {
                        ids.push(id);
                        ids.len() - 1
                    });
                    out.push_str(&format!("<ID{n}>"));
                    i += 26;
                    continue;
                }
                let digits = |at: usize, n: usize| {
                    b.get(at..at + n)
                        .is_some_and(|w| w.iter().all(u8::is_ascii_digit))
                };
                let timestamp = digits(i, 4)
                    && b.get(i + 4) == Some(&b'-')
                    && digits(i + 5, 2)
                    && b.get(i + 10) == Some(&b'T')
                    && digits(i + 11, 2)
                    && b.get(i + 13) == Some(&b':');
                if timestamp {
                    let mut j = i + 19;
                    while j < b.len() && (b[j].is_ascii_digit() || b"+-:.Z".contains(&b[j])) {
                        j += 1;
                    }
                    out.push_str("<TS>");
                    i = j;
                    continue;
                }
                // How long the command took.
                const DURATION: &[u8] = b"\"duration_ms\":";
                if b[i..].starts_with(DURATION) {
                    let mut j = i + DURATION.len();
                    while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
                        j += 1;
                    }
                    out.push_str("\"duration_ms\":<MS>");
                    i = j;
                    continue;
                }
                let ch = text[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
            out
        }

        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) args: Vec<String>,
            pub(crate) stdin: Option<&'static [u8]>,
            /// Writes a record or stamps the time, so ids and timestamps
            /// are compared normalized.
            pub(crate) volatile: bool,
            /// Opens the database itself rather than through [`hot_db`], so
            /// it can't run while a warm handle holds the file.
            pub(crate) cold_only: bool,
        }

        pub(crate) fn case(name: &'static str, args: &[&str]) -> Case {
            Case {
                name,
                args: args.iter().map(|a| a.to_string()).collect(),
                stdin: None,
                volatile: false,
                cold_only: false,
            }
        }

        impl Case {
            fn stdin(mut self, bytes: &'static [u8]) -> Self {
                self.stdin = Some(bytes);
                self
            }
            fn volatile(mut self) -> Self {
                self.volatile = true;
                self
            }
            fn cold_only(mut self) -> Self {
                self.cold_only = true;
                self
            }
        }

        /// The fixture project and its database, built through the CLI
        /// itself. Returns the project directory.
        pub(crate) fn build_fixture(root: &Path) -> std::path::PathBuf {
            let proj = root.join("proj");
            std::fs::create_dir_all(proj.join("src")).unwrap();
            std::fs::create_dir_all(proj.join(".axil")).unwrap();
            std::fs::write(
                proj.join("src/auth.rs"),
                "/// Validate a login token and refresh the session timeout.\n\
                 pub fn validate_token(token: &str) -> bool {\n    !token.is_empty()\n}\n",
            )
            .unwrap();
            std::fs::write(
                proj.join("src/cache.rs"),
                "/// LRU cache eviction for the query planner.\n\
                 pub struct PlanCache {\n    cap: usize,\n}\n",
            )
            .unwrap();
            std::fs::write(
                proj.join("README.md"),
                "# Fixture\n\n## Cache\nThe plan cache evicts half its entries.\n",
            )
            .unwrap();
            // A project marker, so commands that look for the project root
            // stop here instead of scanning the database's own directory.
            std::fs::write(
                proj.join("Cargo.toml"),
                "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
            )
            .unwrap();

            let db = proj.join(".axil/memory.axil");
            let db_s = db.to_str().unwrap();
            let run = |args: &[&str]| {
                let (out, err, code) = in_process(args, None);
                assert_eq!(
                    code,
                    0,
                    "fixture step {args:?} failed: {}{}",
                    String::from_utf8_lossy(&out),
                    String::from_utf8_lossy(&err)
                );
            };
            run(&["init", db_s]);
            // No vector store: no command may need the embedding model.
            let _ = std::fs::remove_file(proj.join(".axil/memory.axil.vec"));
            if cfg!(feature = "indexer") {
                run(&["--db", db_s, "index", proj.to_str().unwrap(), "--quiet"]);
            }
            for (table, data) in [
                (
                    "decisions",
                    r#"{"summary":"Cap login tokens at 512 bytes in validate_token","reason":"oversized tokens were a DoS vector"}"#,
                ),
                (
                    "errors",
                    r#"{"error":"auth timeout fired during long uploads","root_cause":"no heartbeat refresh","fix":"refresh the session on heartbeat"}"#,
                ),
                (
                    "context",
                    r#"{"type":"architecture","summary":"auth.rs validates tokens; cache.rs owns the plan cache"}"#,
                ),
                (
                    "rules",
                    r#"{"rule":"Never log raw auth tokens","summary":"Never log raw auth tokens"}"#,
                ),
            ] {
                run(&["--db", db_s, "store", table, data]);
            }
            proj
        }

        /// The hot commands in each output format, with the errors, usage
        /// errors and help text around them. `db` is the work copy's
        /// database and `auth_rs` a file in its project.
        pub(crate) fn matrix(db: &str, auth_rs: &str) -> Vec<Case> {
            let cp = r#"{"goal":"parity","state":"fixture built","next_steps":["diff"],"references":[{"kind":"file","ref":"src/auth.rs"}]}"#;

            let mut cases = vec![
                case("boot json", &["--db", db, "boot"]),
                case(
                    "boot narrative",
                    &["--db", db, "boot", "--boot-format", "narrative"],
                ),
                case("boot --files", &["--db", db, "boot", "--files", auth_rs]),
                case(
                    "boot --schema v1 (Err)",
                    &["--db", db, "boot", "--schema", "v1"],
                ),
                case("boot --schema v2", &["--db", db, "boot", "--schema", "v2"]).volatile(),
                case("recall-for-file", &["--db", db, "recall-for-file", auth_rs]),
                case(
                    "get missing (exit 2)",
                    &["--db", db, "get", "01ZZZZZZZZZZZZZZZZZZZZZZZZ"],
                )
                .cold_only(),
                case(
                    "store bad JSON (Err)",
                    &["--db", db, "store", "decisions", "{nope"],
                ),
                case(
                    "store inline",
                    &[
                        "--db",
                        db,
                        "store",
                        "decisions",
                        r#"{"summary":"inline parity store"}"#,
                    ],
                )
                .volatile(),
                case("store stdin", &["--db", db, "store", "context", "-"])
                    .stdin(br#"{"type":"howto","summary":"stdin parity store"}"#)
                    .volatile(),
                case(
                    "store stdin empty (Err)",
                    &["--db", db, "store", "context", "-"],
                ),
                case(
                    "auto-capture dry run",
                    &[
                        "--db",
                        db,
                        "auto-capture",
                        "Decided to cap login tokens at 512 bytes because of the DoS",
                        "--dry-run",
                    ],
                ),
                case(
                    "auto-capture stdin",
                    &[
                        "--db",
                        db,
                        "auto-capture",
                        "-",
                        "--min-confidence",
                        "0.0",
                        "--source",
                        "bash",
                    ],
                )
                .stdin(
                    b"error[E0425]: cannot find value `timeout_ms` in this scope\n\
                      Decided to pass the timeout through the session config.",
                )
                .volatile(),
                case(
                    "extract-entities",
                    &[
                        "--db",
                        db,
                        "extract-entities",
                        "validate_token in src/auth.rs uses PlanCache",
                    ],
                ),
                case(
                    "extract-entities stdin",
                    &["--db", db, "extract-entities", "-"],
                )
                .stdin(b"Fixed the auth timeout in validate_token (src/auth.rs)"),
                case("worker run", &["--db", db, "worker", "run"]).volatile(),
                case("beliefs --generate", &["--db", db, "beliefs", "--generate"]).volatile(),
                case("clap usage error", &["store"]),
                case("clap unknown flag", &["boot", "--bogus"]),
                case("--help", &["--help"]),
                case("store --help", &["store", "--help"]),
                case("--version", &["--version"]),
            ];
            if cfg!(feature = "embed") {
                for format in ["context-block", "oneline", "compact", "full"] {
                    let mut c = case(
                        "recall",
                        &[
                            "--db",
                            db,
                            "recall",
                            "auth timeout",
                            "--recall-format",
                            format,
                        ],
                    );
                    c.name = format;
                    cases.push(c);
                }
                cases.push(case(
                    "recall context-block, no match",
                    &[
                        "--db",
                        db,
                        "recall",
                        "zzqx vmwk",
                        "--recall-format",
                        "context-block",
                    ],
                ));
                // Equal scores tie-break in no fixed order (in two one-shot
                // runs as much as here), so every recall query has distinct
                // scores on this fixture.
                cases.push(case(
                    "recall table",
                    &["--db", db, "--format", "table", "recall", "auth timeout"],
                ));
            }
            if cfg!(feature = "indexer") {
                cases.push(case(
                    "code-search",
                    &["--db", db, "code-search", "validate token"],
                ));
                cases.push(case(
                    "code-search --json",
                    &["--db", db, "code-search", "validate token", "--json"],
                ));
            }
            if cfg!(feature = "fts") {
                cases.push(case(
                    "fts table",
                    &["--db", db, "--format", "table", "fts", "cache"],
                ));
                cases.push(case("fts json", &["--db", db, "fts", "cache"]));
                cases.push(case(
                    "fts --jsonl",
                    &["--db", db, "fts", "cache", "--jsonl"],
                ));
            }
            if cfg!(feature = "timeseries") {
                cases.push(case("since 1h", &["--db", db, "since", "1h"]));
            }
            if cfg!(feature = "checkpoint") {
                cases.push(case("checkpoint arg", &["--db", db, "checkpoint", cp]).volatile());
                cases.push(
                    case("checkpoint -", &["--db", db, "checkpoint", "-"])
                        .stdin(cp.as_bytes())
                        .volatile(),
                );
                cases.push(
                    case("checkpoint stdin", &["--db", db, "checkpoint"])
                        .stdin(cp.as_bytes())
                        .volatile(),
                );
                cases.push(
                    case(
                        "checkpoint --final",
                        &["--db", db, "checkpoint", cp, "--final"],
                    )
                    .volatile(),
                );
            }
            cases
        }

        /// What a command printed on stdout and stderr, and its exit code.
        pub(crate) type Run = (Vec<u8>, Vec<u8>, i32);

        /// Run each case through `captured` and as a one-shot child, each
        /// on a fresh copy of `fixture` at `proj`, and fail on any
        /// difference. Returns the one-shot results by case name.
        fn check_parity(
            fixture: &Path,
            proj: &Path,
            cases: &[Case],
            run_captured: impl Fn(&Case, &[&str]) -> Run,
        ) -> Vec<(&'static str, Run)> {
            check_parity_with(fixture, proj, cases, run_captured, |c, args| {
                one_shot(args, c.stdin)
            })
        }

        /// [`check_parity`] with the one-shot side run by `run_one_shot`.
        pub(crate) fn check_parity_with(
            fixture: &Path,
            proj: &Path,
            cases: &[Case],
            run_captured: impl Fn(&Case, &[&str]) -> Run,
            run_one_shot: impl Fn(&Case, &[&str]) -> Run,
        ) -> Vec<(&'static str, Run)> {
            let work = proj.parent().unwrap();
            let mut failures = Vec::new();
            let mut results = Vec::new();
            for c in cases {
                let args: Vec<&str> = c.args.iter().map(String::as_str).collect();
                let run_both = || {
                    let fresh = || {
                        let _ = std::fs::remove_dir_all(work);
                        copy_dir(fixture, proj);
                    };
                    fresh();
                    let captured = run_captured(c, &args);
                    fresh();
                    (captured, run_one_shot(c, &args))
                };
                let same = |captured: &Run, real: &Run| {
                    if c.volatile {
                        captured.2 == real.2
                            && normalize(&captured.0) == normalize(&real.0)
                            && normalize(&captured.1) == normalize(&real.1)
                    } else {
                        captured == real
                    }
                };
                let (mut captured, mut real) = run_both();
                // Two one-shot runs of a command can differ on their own, rarely
                // (equal scores tie-break in no fixed order; ages and scores read
                // the clock), so a mismatch is rerun once. A capture bug repeats.
                if !same(&captured, &real) {
                    eprintln!("parity: {} differed once; rerunning it", c.name);
                    (captured, real) = run_both();
                }
                if !same(&captured, &real) {
                    failures.push(format!(
                        "{}: exit {} vs one-shot {}\n--- captured stdout\n{}\n--- one-shot stdout\n{}\n\
                         --- captured stderr\n{}\n--- one-shot stderr\n{}",
                        c.name,
                        captured.2,
                        real.2,
                        String::from_utf8_lossy(&captured.0),
                        String::from_utf8_lossy(&real.0),
                        String::from_utf8_lossy(&captured.1),
                        String::from_utf8_lossy(&real.1),
                    ));
                }
                results.push((c.name, real));
            }
            assert!(failures.is_empty(), "{}", failures.join("\n\n"));
            results
        }

        #[test]
        fn captured_output_matches_one_shot() {
            let tmp = tempfile::tempdir().unwrap();
            let fixture = build_fixture(&tmp.path().join("fixture"));
            let proj = tmp.path().join("work/proj");
            let db = proj.join(".axil/memory.axil");
            let db = db.to_str().unwrap();
            let auth_rs = proj.join("src/auth.rs");
            let auth_rs = auth_rs.to_str().unwrap();

            let results = check_parity(&fixture, &proj, &matrix(db, auth_rs), |c, args| {
                in_process(args, c.stdin)
            });

            // Equal could still mean equally empty: pin what the one-shot
            // side printed for the cases that show each path ran.
            let get = |name: &str| {
                let (_, r) = results.iter().find(|(n, _)| *n == name).unwrap();
                (
                    String::from_utf8_lossy(&r.0),
                    String::from_utf8_lossy(&r.1),
                    r.2,
                )
            };
            let (out, err, code) = get("store bad JSON (Err)");
            assert_eq!(code, 1);
            assert!(
                out.is_empty() && err.starts_with(r#"{"error":"invalid JSON"#),
                "{err}"
            );
            let (out, err, code) = get("clap usage error");
            assert_eq!(code, 2);
            assert!(out.is_empty() && err.starts_with("error:"), "{err}");
            let (out, _, code) = get("--help");
            assert_eq!(code, 0);
            assert!(out.contains("Usage: axil"), "{out}");
            let (out, _, _) = get("store stdin");
            assert!(out.contains(r#""table":"context""#), "{out}");
            let (out, _, _) = get("boot narrative");
            assert!(out.contains("Cap login tokens at 512 bytes"), "{out}");
            if cfg!(feature = "embed") {
                let (out, _, _) = get("context-block");
                assert!(
                    out.starts_with("<context") && out.ends_with("</context>\n"),
                    "the block ends at its closing tag, with no line after it: {out:?}"
                );
                assert!(get("recall context-block, no match").0.is_empty());
            }
            if cfg!(feature = "checkpoint") {
                assert!(get("checkpoint -").0.contains(r#""kind":"snapshot""#));
                assert!(get("checkpoint --final").0.contains(r#""kind":"final""#));
            }
        }

        /// A request that hands `warm` to its command.
        fn warm_request(warm: &WarmDb, stdin: Option<&[u8]>) -> ReqCtx {
            ReqCtx {
                stdin: Some(stdin.unwrap_or_default().to_vec()),
                warm: Some(warm.clone()),
                ..ReqCtx::default()
            }
        }

        /// The hot commands on a handle the request carries: the one-shot
        /// binary's bytes, without opening the database themselves, and the
        /// handle still open and theirs no longer when they return.
        #[test]
        fn warm_handle_output_matches_one_shot() {
            let tmp = tempfile::tempdir().unwrap();
            let fixture = build_fixture(&tmp.path().join("fixture"));
            let proj = tmp.path().join("work/proj");
            let db_path = proj.join(".axil/memory.axil");
            let db = db_path.to_str().unwrap();
            let auth_rs = proj.join("src/auth.rs");
            let auth_rs = auth_rs.to_str().unwrap();

            // A command that opened the database while this process holds
            // it would say so on stderr, so the parity below can't pass that
            // way.
            copy_dir(&fixture, &proj);
            let held = WarmDb::open(&db_path).unwrap();
            let (_, err, _) = in_process(&["--db", db, "boot"], None);
            let err = String::from_utf8_lossy(&err);
            assert!(err.contains("holds the writer lock"), "{err}");
            drop(held);

            let mut cases = matrix(db, auth_rs);
            cases.retain(|c| !c.cold_only);
            check_parity(&fixture, &proj, &cases, |c, args| {
                let warm = WarmDb::open(&db_path).unwrap();
                let run = in_process_with(warm_request(&warm, c.stdin), args);
                assert_eq!(
                    Arc::strong_count(warm.db()),
                    1,
                    "{}: the command kept a reference to the warm handle",
                    c.name
                );
                if c.name.starts_with("store") && c.volatile && run.2 == 0 {
                    let stored: serde_json::Value = serde_json::from_slice(&run.0).unwrap();
                    let id = stored["id"].as_str().unwrap();
                    let id = axil_core::RecordId::from_string(id).unwrap();
                    assert!(
                        warm.db().get(&id).unwrap().is_some(),
                        "{}: the record went somewhere other than the warm handle",
                        c.name
                    );
                }
                run
            });
        }

        /// Run a command in-process on `warm` and return what
        /// `dispatch_command` returned, which an in-process host sees before
        /// any error is printed.
        fn dispatch_warm(warm: &WarmDb, args: &[&str]) -> anyhow::Result<i32> {
            let ctx = warm_request(warm, None);
            let args = argv(args);
            on_cli_stack(move || {
                capture(ctx, || {
                    let cli = Cli::try_parse_from(&args).unwrap();
                    let out = crate::Output::for_cli(&cli);
                    crate::dispatch_command(cli, &out)
                })
                .2
            })
        }

        /// A warm handle opened before a store existed can't serve a command
        /// whose own open would create that store, and one without an LLM
        /// can't serve `store --llm`. The command refuses with [`WarmLacks`]
        /// before it writes anything, and the one-shot binary, which a host
        /// falls back to, serves it.
        #[test]
        fn a_warm_handle_refuses_what_it_lacks() {
            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            let db_path = proj.join(".axil/memory.axil");
            let db = db_path.to_str().unwrap();
            let companion = |suffix: &str| axil_core::companion_path(&db_path, suffix);
            for suffix in [".fts", ".ts"] {
                let _ = std::fs::remove_dir_all(companion(suffix));
                let _ = std::fs::remove_file(companion(suffix));
            }
            let stored = r#"{"summary":"stored through a handle that lacks a store"}"#;

            // (one-shot serves it too, args, what the handle lacks)
            let mut cases: Vec<(bool, Vec<String>, Need)> = vec![(
                true,
                argv(&["store", "decisions", stored, "--llm"]),
                Need::Llm,
            )];
            if cfg!(feature = "fts") {
                cases.push((true, argv(&["fts", "cache"]), Need::FullText));
            }
            if cfg!(feature = "timeseries") {
                cases.push((true, argv(&["since", "1h"]), Need::TimeSeries));
            }
            #[cfg(feature = "embed")]
            {
                // The one-shot run would load the embedding model.
                let embed = argv(&["store", "decisions", stored, "--embed", "summary"]);
                cases.push((false, embed, Need::Vectors));
                let dims = crate::resolve_embedding_model(&db_path).dimensions();
                let vector = serde_json::to_string(&vec![0.5f32; dims]).unwrap();
                let raw = argv(&["store", "decisions", stored, "--vector", &vector]);
                cases.push((true, raw, Need::Vectors));
            }

            let warm = WarmDb::open(&db_path).unwrap();
            let decisions = || warm.db().list("decisions").unwrap().len();
            let before = decisions();
            for (_, args, need) in &cases {
                let mut args: Vec<&str> = args[1..].iter().map(String::as_str).collect();
                args.splice(0..0, ["--db", db]);
                let err = dispatch_warm(&warm, &args).unwrap_err();
                assert_eq!(
                    err.downcast_ref::<WarmLacks>(),
                    Some(&WarmLacks(*need)),
                    "{args:?}: {err:#}"
                );
            }
            assert_eq!(decisions(), before, "a refused command wrote a record");
            for suffix in [".fts", ".ts", ".vec"] {
                assert!(
                    !companion(suffix).exists(),
                    "a refused command made {suffix}"
                );
            }
            drop(warm);

            // The vector store goes last: once it exists, every open loads
            // the embedding model.
            for (one_shot_serves, args, _) in &cases {
                if !one_shot_serves {
                    continue;
                }
                let mut args: Vec<&str> = args[1..].iter().map(String::as_str).collect();
                args.splice(0..0, ["--db", db]);
                let (out, err, code) = one_shot(&args, None);
                assert_eq!(
                    code,
                    0,
                    "{args:?}: {}{}",
                    String::from_utf8_lossy(&out),
                    String::from_utf8_lossy(&err)
                );
            }
        }

        /// The handoff a host performs for a refused command: release the
        /// handle, let the one-shot process run the command, reopen. While the
        /// host holds the handle the one-shot process can't open the database;
        /// the release waits for a command still using the handle, and gives
        /// the handle back if that takes longer than its timeout.
        #[cfg(feature = "timeseries")]
        #[test]
        fn a_host_hands_a_refused_command_to_a_one_shot_run() {
            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            let db_path = proj.join(".axil/memory.axil");
            let db = db_path.to_str().unwrap();
            let ts = axil_core::companion_path(&db_path, ".ts");
            let _ = std::fs::remove_dir_all(&ts);
            let _ = std::fs::remove_file(&ts);
            let since = ["--db", db, "since", "1h"];

            let warm = WarmDb::open(&db_path).unwrap();
            let err = dispatch_warm(&warm, &since).unwrap_err();
            assert_eq!(
                err.downcast_ref::<WarmLacks>(),
                Some(&WarmLacks(Need::TimeSeries))
            );
            let (_, err, _) = one_shot(&["--db", db, "boot"], None);
            let err = String::from_utf8_lossy(&err);
            assert!(err.contains("holds the writer lock"), "{err}");

            // A command still running on the handle.
            let in_use = Arc::clone(warm.db());
            let command = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                drop(in_use);
            });
            let Err(warm) = warm.release(Duration::from_millis(20)) else {
                panic!("released a handle a command still holds");
            };
            let Ok(path) = warm.release(Duration::from_secs(30)) else {
                panic!("the command's handle never went");
            };
            command.join().unwrap();

            let (out, err, code) = one_shot(&since, None);
            assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
            let warm = WarmDb::open(&path).unwrap();
            let (warm_out, _, warm_code) = in_process_with(warm_request(&warm, None), &since);
            assert_eq!((warm_out, warm_code), (out, 0), "served once reopened");
        }

        /// Releasing a warm handle with an installed plugin closes the
        /// database: a registered plugin holds no reference that keeps it open.
        #[cfg(feature = "wasm-host")]
        #[test]
        fn a_warm_handle_with_a_plugin_closes_when_released() {
            const HELLO: &[u8] =
                include_bytes!("../../axil-runtime/tests/fixtures/hello-guest.component.wasm");
            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            std::fs::create_dir_all(proj.join(".axil/plugins")).unwrap();
            std::fs::write(proj.join(".axil/plugins/hello.wasm"), HELLO).unwrap();
            let db_path = proj.join(".axil/memory.axil");
            let boot = ["--db", db_path.to_str().unwrap(), "boot"];

            let warm = WarmDb::open(&db_path).unwrap();
            let (out, _, _) = in_process_with(warm_request(&warm, None), &boot);
            assert!(String::from_utf8_lossy(&out).contains("hello plugin ready"));
            let Ok(path) = warm.release(Duration::from_secs(10)) else {
                panic!("the registered plugin kept the database open");
            };
            let (out, err, code) = one_shot(&boot, None);
            assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
            assert!(String::from_utf8_lossy(&out).contains("hello plugin ready"));

            // Dropping the handle, as well as releasing it, closes it.
            drop(WarmDb::open(&path).unwrap());
            WarmDb::open(&path).unwrap();
        }

        /// `boot` on a warm handle shows an installed plugin's block, as the
        /// one-shot `boot` does: opening the handle registered the plugin.
        #[cfg(feature = "wasm-host")]
        #[test]
        fn warm_boot_shows_installed_plugins_like_one_shot() {
            const HELLO: &[u8] =
                include_bytes!("../../axil-runtime/tests/fixtures/hello-guest.component.wasm");
            let tmp = tempfile::tempdir().unwrap();
            let fixture = build_fixture(&tmp.path().join("fixture"));
            std::fs::create_dir_all(fixture.join(".axil/plugins")).unwrap();
            std::fs::write(fixture.join(".axil/plugins/hello.wasm"), HELLO).unwrap();
            let proj = tmp.path().join("work/proj");
            let db_path = proj.join(".axil/memory.axil");
            let db = db_path.to_str().unwrap();

            let cases = [
                case("boot json", &["--db", db, "boot"]),
                case(
                    "boot narrative",
                    &["--db", db, "boot", "--boot-format", "narrative"],
                ),
            ];
            let results = check_parity(&fixture, &proj, &cases, |c, args| {
                let warm = WarmDb::open(&db_path).unwrap();
                in_process_with(warm_request(&warm, c.stdin), args)
            });
            for (name, (out, _, _)) in results {
                let out = String::from_utf8_lossy(&out);
                assert!(out.contains("hello plugin ready"), "{name}: {out}");
            }
        }

        /// `recall --timeout-ms` on a long-lived handle. With every slot
        /// taken it waits for one until its deadline, then answers with the
        /// partial results; a slot freed in time lets it search. However many
        /// requests time out, no more workers than slots stay behind, each
        /// giving its slot and handle back when its search ends.
        #[cfg(feature = "embed")]
        #[test]
        fn recall_timeout_workers_stay_within_the_slots() {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            let db_path = proj.join(".axil/memory.axil");
            let db = db_path.to_str().unwrap();
            let warm = WarmDb::open(&db_path).unwrap();
            let recall = |timeout_ms: &str| {
                let ctx = ReqCtx {
                    slow_query_log: false,
                    ..warm_request(&warm, None)
                };
                let args = [
                    "--db",
                    db,
                    "recall",
                    "auth timeout",
                    "--timeout-ms",
                    timeout_ms,
                ];
                let (out, err, code) = in_process_with(ctx, &args);
                assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
                let hits = serde_json::from_slice::<Vec<serde_json::Value>>(&out).unwrap();
                (hits.len(), String::from_utf8(err).unwrap())
            };
            let partial = |ms: u64| {
                format!(
                    "[axil recall] timeout_ms={ms} exceeded inside db.recall; returning partial results\n"
                )
            };

            // Only this test runs recalls with a deadline, so every slot is
            // free to take.
            let take_all = || {
                let soon = || Instant::now() + Duration::from_millis(10);
                let held: Vec<RecallSlot> =
                    std::iter::from_fn(|| RECALL_SLOTS.take_by(soon())).collect();
                assert_eq!(held.len(), RECALL_SLOTS.cap);
                held
            };

            // No slot frees up: the deadline passes waiting, and the
            // command says so, with no search and no cascade after it.
            let held = take_all();
            let start = Instant::now();
            let (hits, err) = recall("300");
            assert!(start.elapsed() >= Duration::from_millis(300));
            assert_eq!(hits, 0);
            assert!(err.starts_with(&partial(300)), "{err}");
            assert!(
                err.contains("timeout_ms=300 exceeded; returned partial results"),
                "{err}"
            );
            drop(held);

            // A slot frees up well before the deadline: a full search.
            let mut held = take_all();
            let one = held.pop().unwrap();
            let freer = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                drop(one);
            });
            let start = Instant::now();
            let (hits, err) = recall("60000");
            assert!(start.elapsed() < Duration::from_secs(30));
            assert!(hits > 0 && !err.contains("partial"), "{err}");
            freer.join().unwrap();
            drop(held);

            // 100 requests from 10 clients at once, each with a deadline
            // most searches miss.
            let done = AtomicBool::new(false);
            let peak = AtomicUsize::new(0);
            let (clients, live_after) = std::thread::scope(|s| {
                s.spawn(|| {
                    while !done.load(Ordering::Acquire) {
                        peak.fetch_max(RECALL_SLOTS.live(), Ordering::AcqRel);
                        std::thread::sleep(Duration::from_micros(50));
                    }
                });
                let clients: Vec<_> = (0..10)
                    .map(|_| {
                        s.spawn(|| {
                            (0..10)
                                .filter(|_| recall("1").1.contains(&partial(1)))
                                .count()
                        })
                    })
                    .collect();
                let clients: Vec<_> = clients.into_iter().map(|c| c.join()).collect();
                let live_after = RECALL_SLOTS.live();
                done.store(true, Ordering::Release);
                (clients, live_after)
            });
            let partials: usize = clients.into_iter().map(Result::unwrap).sum();
            let peak = peak.load(Ordering::Acquire);
            eprintln!(
                "recall slots: {partials} of 100 recalls cut short; peak {peak} live workers"
            );
            assert!(peak <= RECALL_SLOTS.cap, "{peak} workers ran at once");
            assert!(live_after <= RECALL_SLOTS.cap, "{live_after} workers left");

            let deadline = Instant::now() + Duration::from_secs(60);
            while RECALL_SLOTS.live() > 0 || Arc::strong_count(warm.db()) > 1 {
                assert!(
                    Instant::now() < deadline,
                    "a worker never gave its slot back"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        /// `axil … | head` once `head` has exited: the command prints into
        /// a pipe nobody reads. It must run to its end and exit 0 without a
        /// panic, as the module docs describe.
        #[test]
        fn a_closed_stdout_ends_the_command_cleanly() {
            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            let db = proj.join(".axil/memory.axil");
            let db = db.to_str().unwrap();
            let commands = [
                vec!["--db", db, "boot"],
                vec!["--db", db, "--format", "table", "query", "decisions"],
                vec!["--db", db, "export"],
            ];
            for args in commands {
                let mut child = one_shot_command(&args)
                    .env(CHILD_WAIT_ENV, "1")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap();
                // Read up to the mark, past the harness's own lines, then
                // hang up. The child waits on its stdin until then.
                let mut stdout = child.stdout.take().unwrap();
                let mut seen = Vec::new();
                while after_mark(&seen).is_none() {
                    let mut chunk = [0u8; 256];
                    let n = stdout.read(&mut chunk).unwrap();
                    assert!(n > 0, "child exited before its mark");
                    seen.extend_from_slice(&chunk[..n]);
                }
                drop(stdout);
                drop(child.stdin.take());
                let output = child.wait_with_output().unwrap();
                let err = after_mark(&output.stderr).unwrap();
                let err = String::from_utf8_lossy(&err);
                assert_eq!(output.status.code(), Some(0), "{args:?}: {err}");
                assert!(
                    !err.contains("panicked") && !err.contains("Broken pipe"),
                    "{args:?}: {err}"
                );
            }
        }

        #[test]
        fn a_request_can_turn_off_the_slow_query_log() {
            let tmp = tempfile::tempdir().unwrap();
            let proj = build_fixture(tmp.path());
            // Every query is slow at a zero threshold.
            std::fs::write(
                proj.join(".axil/axil.toml"),
                "[debug]\nslow_query_threshold_ms = 0\n",
            )
            .unwrap();
            let db = proj.join(".axil/memory.axil");
            let db = db.to_str().unwrap();
            let mut reads = vec![vec!["--db", db, "query", "decisions"]];
            if cfg!(feature = "embed") {
                reads.push(vec!["--db", db, "recall", "auth timeout"]);
            }
            if cfg!(feature = "fts") {
                reads.push(vec!["--db", db, "fts", "cache"]);
            }
            let logged = || {
                let (out, _, code) = in_process(&["--db", db, "slow-queries"], None);
                assert_eq!(code, 0);
                serde_json::from_slice::<Vec<serde_json::Value>>(&out)
                    .unwrap()
                    .len()
            };
            for (log, expected) in [(false, 0), (true, reads.len())] {
                in_process(&["--db", db, "slow-queries", "--clear"], None);
                for read in &reads {
                    let ctx = ReqCtx {
                        slow_query_log: log,
                        stdin: Some(Vec::new()),
                        ..ReqCtx::default()
                    };
                    let (_, err, code) = in_process_with(ctx, read);
                    assert_eq!(code, 0, "{read:?}: {}", String::from_utf8_lossy(&err));
                }
                assert_eq!(logged(), expected, "slow_query_log = {log}");
            }
        }
    }
}
