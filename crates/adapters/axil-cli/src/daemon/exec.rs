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
//! - Lines printed inside recall's ranking (`query.rs` and `Axil::recall`)
//!   and by the recall workers a `--timeout-ms` deadline leaves running go
//!   to the host's stderr: those threads carry no request.
//! - A command that panics exits 1 with one `axil: the command panicked`
//!   line on stderr, not the panic hook's message, and the host reopens the
//!   database before the next request.
//! - Help and usage errors render as for a pipe, without color. They are
//!   served only to a client whose `AXIL_DB` matches the host's (unset, once
//!   the host has removed its own), since clap prints them differently with
//!   the variable set.
//!
//! The process-wide setup a one-shot run does first (the vector exact-scan
//! ceiling, the encryption cipher) is the host's to do once, and so is
//! removing its own `AXIL_DB`: the request carries the client's.

use std::any::Any;
use std::ffi::OsString;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// How long a reopen waits for the requests still using the old handle.
const REOPEN_WAIT: Duration = Duration::from_secs(3);

/// One command line, as the client's process was asked to run it.
pub struct Request {
    /// The command line, program name first.
    pub argv: Vec<OsString>,
    /// The client's working directory, absolute. A relative `--db` or
    /// `AXIL_DB` resolves against it, and with neither the database is the
    /// `.axil/memory.axil` found walking up from it.
    pub cwd: PathBuf,
    /// The client's `AXIL_DB`, the one environment variable the command line
    /// reads. The host's own is never consulted.
    pub axil_db: Option<OsString>,
    /// Piped stdin. `None` stands for a terminal: nothing was piped.
    pub stdin: Option<Vec<u8>>,
    /// Whether the command may record a slow query in the database's log.
    pub slow_query_log: bool,
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
}

/// Why the host did not run a request. It printed and wrote nothing, so the
/// client can run the command in its own process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotServed {
    /// The command line holds an argument or subcommand this build doesn't
    /// know, or it doesn't parse and the client's `AXIL_DB` differs from the
    /// one the host's parser was made with: clap's help and usage errors
    /// depend on that variable (an empty one is itself a usage error).
    Parse,
    /// The command is for another database than the one held, or for none.
    WrongDb,
    /// The command is not one the host runs in-process.
    Command,
    /// The open handle lacks what the command needs: the command's own open
    /// would attach or create it.
    Lacks(Need),
    /// The host has no open handle: a reopen failed, or it couldn't start the
    /// request's thread.
    Unavailable,
}

impl fmt::Display for NotServed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotServed::Parse => f.write_str("the command line does not parse here"),
            NotServed::WrongDb => f.write_str("the command is for another database"),
            NotServed::Command => f.write_str("the command is not run in-process"),
            NotServed::Lacks(need) => WarmLacks(*need).fmt(f),
            NotServed::Unavailable => f.write_str("the database is not open"),
        }
    }
}

impl std::error::Error for NotServed {}

/// The database a host keeps open, shared by every request it runs.
pub struct WarmState {
    /// The open handle. `None` after a reopen that failed to open again.
    warm: RwLock<Option<WarmDb>>,
    /// Where the handle reopens from.
    path: PathBuf,
    /// Held by a request that writes, so writes run one at a time.
    writer: Mutex<()>,
    /// Set by a request that panicked: the handle is reopened before the
    /// next request runs on it.
    reopen: AtomicBool,
    #[cfg(test)]
    panic_next: AtomicBool,
}

impl WarmState {
    /// Open the database at `path` (see [`WarmDb::open`]) to serve requests.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        cli_io::install_diagnostic_sink();
        let warm = WarmDb::open(path)?;
        Ok(Self {
            path: warm.key().to_path_buf(),
            warm: RwLock::new(Some(warm)),
            writer: Mutex::new(()),
            reopen: AtomicBool::new(false),
            #[cfg(test)]
            panic_next: AtomicBool::new(false),
        })
    }

    /// The open handle, if there is one.
    pub fn handle(&self) -> Option<WarmDb> {
        self.warm
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Whether a request panicked since the handle was last opened.
    pub fn needs_reopen(&self) -> bool {
        self.reopen.load(Ordering::Acquire)
    }

    /// Close the handle and open the database again. New requests wait for
    /// the reopen; it waits up to `timeout` for the requests still using the
    /// old handle (see [`WarmDb::release`]) and keeps that handle if they
    /// don't finish. If the open fails, no handle is left and requests are
    /// [`NotServed::Unavailable`] until a later reopen succeeds.
    pub fn reopen(&self, timeout: Duration) -> anyhow::Result<()> {
        let mut slot = self.warm.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(warm) = slot.take() {
            if let Err(warm) = warm.release(timeout) {
                *slot = Some(warm);
                anyhow::bail!("the database was still in use after {timeout:?}");
            }
        }
        *slot = Some(WarmDb::open(&self.path)?);
        self.reopen.store(false, Ordering::Release);
        Ok(())
    }

    /// The handle a request runs on, reopened first after a panic.
    fn current(&self) -> Result<WarmDb, NotServed> {
        if self.needs_reopen() {
            // A failure keeps the old handle, or leaves none (Unavailable).
            let _ = self.reopen(REOPEN_WAIT);
        }
        self.handle().ok_or(NotServed::Unavailable)
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
/// and the database is reopened before the next request.
pub fn execute_captured(req: Request, state: &WarmState) -> Result<Response, NotServed> {
    std::thread::scope(|scope| {
        let request = std::thread::Builder::new()
            .name("axil-request".into())
            .stack_size(REQUEST_STACK)
            .spawn_scoped(scope, move || execute(req, state));
        match request {
            // A panic outside the command itself (parsing, say) leaves the
            // handle as it was.
            Ok(request) => request
                .join()
                .unwrap_or_else(|payload| Ok(panicked(Vec::new(), Vec::new(), &*payload))),
            Err(_) => Err(NotServed::Unavailable),
        }
    })
}

fn execute(req: Request, state: &WarmState) -> Result<Response, NotServed> {
    let mut cli = match parse(&req)? {
        Parsed::Run(cli) => cli,
        Parsed::Printed(response) => return Ok(response),
    };
    let db_path = request_db(cli.db.as_deref(), &req.cwd).ok_or(NotServed::WrongDb)?;
    let warm = state.current()?;
    if cli_io::db_key(&db_path).ok().as_deref() != Some(warm.key()) {
        return Err(NotServed::WrongDb);
    }
    warm_route(&cli.command, warm.db())?;
    cli.db = Some(db_path);

    let _writing =
        writes(&cli.command).then(|| state.writer.lock().unwrap_or_else(PoisonError::into_inner));
    let ctx = ReqCtx {
        stdin: req.stdin,
        slow_query_log: req.slow_query_log,
        warm: Some(warm),
        ..ReqCtx::default()
    };
    let (stdout, stderr, result) = cli_io::capture(ctx, || {
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
        Ok(result) => finish(result, stdout, stderr),
        Err(payload) => {
            state.reopen.store(true, Ordering::Release);
            Ok(panicked(stdout, stderr, &*payload))
        }
    }
}

/// The response for what [`dispatch_command`](crate::dispatch_command)
/// returned: an error printed as `run_cli` prints it, or [`NotServed`] when
/// the handle lacked what the command needs (the command wrote nothing).
fn finish(
    result: anyhow::Result<i32>,
    stdout: Vec<u8>,
    mut stderr: Vec<u8>,
) -> Result<Response, NotServed> {
    match result {
        Ok(exit) => Ok(Response {
            exit,
            stdout,
            stderr,
        }),
        Err(e) => {
            if let Some(WarmLacks(need)) = e.downcast_ref::<WarmLacks>() {
                return Err(NotServed::Lacks(*need));
            }
            stderr.extend_from_slice(crate::format_cli_error(&e).as_bytes());
            stderr.push(b'\n');
            Ok(Response {
                exit: crate::EXIT_ERROR,
                stdout,
                stderr,
            })
        }
    }
}

fn panicked(stdout: Vec<u8>, mut stderr: Vec<u8>, payload: &(dyn Any + Send)) -> Response {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a non-string payload".to_string());
    stderr.extend_from_slice(format!("axil: the command panicked: {message}\n").as_bytes());
    Response {
        exit: crate::EXIT_ERROR,
        stdout,
        stderr,
    }
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
fn writes(command: &Command) -> bool {
    match command {
        Command::Store { .. } => true,
        #[cfg(feature = "checkpoint")]
        Command::Checkpoint { arg, .. } => arg.as_deref() != Some("show"),
        #[cfg(feature = "embed")]
        Command::Recall { fresh_only, .. } => *fresh_only,
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
    /// The `AXIL_DB` the command was made with: clap reads the variable when
    /// it builds the argument, and what it prints for help and usage errors
    /// depends on it.
    env_db: Option<OsString>,
}

fn parser() -> &'static CliParser {
    static PARSER: OnceLock<CliParser> = OnceLock::new();
    PARSER.get_or_init(|| {
        let env_db = std::env::var_os("AXIL_DB");
        let command = Cli::command();
        CliParser { command, env_db }
    })
}

/// Parse `req`'s command line as `Cli::parse` would in the client's
/// process, with the client's `AXIL_DB`.
fn parse(req: &Request) -> Result<Parsed, NotServed> {
    // An empty `AXIL_DB` makes every command line a usage error.
    if req.axil_db.as_ref().is_some_and(|v| v.is_empty()) {
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
    // not one the host's environment supplied.
    if matches.value_source("db") != Some(ValueSource::CommandLine) {
        cli.db = req.axil_db.clone().map(PathBuf::from);
    }
    Ok(Parsed::Run(cli))
}

/// What `clap::Error::exit` would print for `e`, as a response.
///
/// An unknown argument or subcommand is [`NotServed::Parse`]: a client of a
/// newer build runs it in its own process. So is any of this when the
/// client's `AXIL_DB` is not the one the parser was made with. clap reads the
/// variable as it builds the command: help shows its value, a usage line
/// lists `--db` when it is set, and a command line without a subcommand
/// prints help without it and an error with it.
fn printed(e: clap::Error, req: &Request, parser: &CliParser) -> Result<Parsed, NotServed> {
    let unknown = matches!(
        e.kind(),
        ErrorKind::UnknownArgument | ErrorKind::InvalidSubcommand
    );
    if unknown || parser.env_db != req.axil_db {
        return Err(NotServed::Parse);
    }
    let text = e.render().to_string().into_bytes();
    let (stdout, stderr) = if e.use_stderr() {
        (Vec::new(), text)
    } else {
        (text, Vec::new())
    };
    Ok(Parsed::Printed(Response {
        exit: e.exit_code(),
        stdout,
        stderr,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_io::tests::parity::{
        self, build_fixture, case, check_parity_with, copy_dir, matrix, one_shot_at, Case, Run,
    };
    use std::ffi::OsStr;
    use std::sync::Arc;

    /// A request from a client working in `cwd`. Its stdin is piped and
    /// holds `stdin`, empty by default, like the one-shot children's.
    fn request(args: &[&str], cwd: &Path, axil_db: Option<&str>, stdin: Option<&[u8]>) -> Request {
        Request {
            argv: parity::argv(args).into_iter().map(OsString::from).collect(),
            cwd: cwd.to_path_buf(),
            axil_db: axil_db.map(OsString::from),
            stdin: Some(stdin.unwrap_or_default().to_vec()),
            slow_query_log: true,
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
            .spawn(|| parser().env_db.clone())
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
            if host_env.as_deref() != env.map(OsStr::new) {
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

    /// The handle refusing from inside the command, as [`cli_io::hot_db`]
    /// does, is the same typed refusal; any other error prints as the
    /// one-shot binary prints it.
    #[test]
    fn a_command_error_prints_and_a_refusal_does_not() {
        let refused = finish(
            Err(WarmLacks(Need::FullText).into()),
            b"half".to_vec(),
            Vec::new(),
        );
        assert_eq!(refused, Err(NotServed::Lacks(Need::FullText)));
        let failed = finish(
            Err(anyhow::anyhow!("boom").context("ctx")),
            Vec::new(),
            b"warning\n".to_vec(),
        );
        let stderr = b"warning\n{\"error\":\"ctx: boom\"}\n".to_vec();
        assert_eq!(
            failed,
            Ok(Response {
                exit: 1,
                stdout: Vec::new(),
                stderr
            })
        );
    }

    /// A request that panics is answered with exit 1, the host serves the
    /// next one, and that one runs on a reopened handle. A write that
    /// panicked doesn't leave the writer lock poisoned.
    #[test]
    fn a_panicking_request_is_answered_and_the_handle_reopened() {
        let tmp = tempfile::tempdir().unwrap();
        let proj = build_fixture(tmp.path());
        let state = WarmState::open(&proj.join(".axil/memory.axil")).unwrap();
        let boot = || request(&["boot"], &proj, None, None);
        let store = || {
            let data = r#"{"summary":"after a panic"}"#;
            request(&["store", "decisions", data], &proj, None, None)
        };

        let first = Arc::downgrade(state.handle().unwrap().db());
        state.panic_next.store(true, Ordering::Release);
        let r = execute_captured(boot(), &state).unwrap();
        assert_eq!(r.exit, 1);
        let err = String::from_utf8_lossy(&r.stderr);
        assert!(
            err.ends_with("axil: the command panicked: a request panicked on purpose\n"),
            "{err}"
        );
        assert!(state.needs_reopen());

        let r = execute_captured(boot(), &state).unwrap();
        assert_eq!(r.exit, 0, "{}", String::from_utf8_lossy(&r.stderr));
        assert!(!state.needs_reopen());
        assert!(
            first.upgrade().is_none(),
            "the next request ran on the old handle"
        );

        state.panic_next.store(true, Ordering::Release);
        assert_eq!(execute_captured(store(), &state).unwrap().exit, 1);
        let r = execute_captured(store(), &state).unwrap();
        assert_eq!(r.exit, 0, "{}", String::from_utf8_lossy(&r.stderr));
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

    /// A warning an engine prints during the command reaches the request's
    /// stderr, where the one-shot binary prints it, not the host's.
    #[cfg(feature = "indexer")]
    #[test]
    fn an_engine_warning_reaches_the_requests_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let fixture = build_fixture(&tmp.path().join("fixture"));
        // Over the default size limit: the freshness scan skips it, loudly.
        let big = format!("// {}\n", "x".repeat(1024)).repeat(1024);
        std::fs::write(fixture.join("src/generated.rs"), big).unwrap();
        let proj = tmp.path().join("work/proj");
        let db_path = proj.join(".axil/memory.axil");
        let cwd = || proj.canonicalize().unwrap();
        let results = check_parity_with(
            &fixture,
            &proj,
            &[case("boot json", &["boot"])],
            |c, args| {
                let state = WarmState::open(&db_path).unwrap();
                served(request(args, &cwd(), None, c.stdin), &state)
            },
            |c, args| one_shot_at(args, c.stdin, &cwd(), None),
        );
        let err = String::from_utf8_lossy(&results[0].1 .1);
        assert!(err.contains("[axil-index] skipped 1 file(s)"), "{err}");
    }
}
