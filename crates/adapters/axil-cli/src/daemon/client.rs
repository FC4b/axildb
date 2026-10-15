//! Sending a command line to the database's daemon, and what a client does
//! with the answer.
//!
//! [`forward`] sends one command line to the daemon of the database it runs
//! on and reads what the daemon's run printed. It connects with the checks
//! every client makes ([`proto::connect`]) and sends the run request with
//! nothing before it: a daemon of another build answers `incompatible`
//! without running anything. It never reads `daemon.json`: no socket is no
//! daemon, and a socket nobody listens on is a dead one's.
//!
//! - The CLI ([`forward_cli`]) sends a hot command when `AXIL_DAEMON` is not
//!   `off` and a database resolves, and prints what the daemon's run
//!   printed. When the daemon does not run it, the command runs in this
//!   process as it would with no daemon, and its open asks the daemon to
//!   let go of the database ([`super::lease`]). A write whose connection
//!   broke after it was sent is not run again: the daemon may have run it.
//! - A hook's lookup ([`forward_lookup`]) is sent with the slow-query log
//!   off and a deadline. A daemon that ran it answers it; one that is not
//!   free to (the database leased or opening, the daemon shutting down)
//!   leaves the hook without context at once, except that a boot waits a
//!   while for a daemon opening the database. With no daemon, or a command
//!   it does not run, the hook runs its lookup in a child process, which
//!   leases as any command does.
//!
//! A client of another build than the daemon's runs the command itself, so
//! a command runs only on the code of the binary it was typed into.

use std::path::Path;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::control::{self, Mode};
use super::exec;
use super::paths::RunDir;
use super::proto::{self, Identity, RunHeader, RunReply};
use crate::cli_io;
use crate::Cli;

/// How long the CLI waits for a daemon that is still opening the database,
/// or has every request slot taken, before it runs the command itself.
const CLI_STARTING_WAIT: Duration = Duration::from_secs(10);

/// How long the CLI waits for the answer to a command the daemon runs.
const CLI_ANSWER_WAIT: Duration = Duration::from_secs(300);

/// How long a hook's lookup may take: what the hook waits for the child
/// that runs it otherwise. A recall gives its own `--timeout-ms 1800` room.
const LOOKUP_DEADLINE: Duration = Duration::from_millis(2000);
const RECALL_DEADLINE: Duration = Duration::from_millis(2100);

/// How long a hook's boot waits for a daemon that is still opening the
/// database: the first boot of a session may come right after it started.
const BOOT_STARTING_WAIT: Duration = Duration::from_secs(8);

/// How much longer than its deadline a client waits for the answer, for
/// the reply to cross the socket.
const ANSWER_SLACK: Duration = Duration::from_millis(100);

/// Between two asks of a daemon that is opening the database or has every
/// request slot taken.
const RETRY_EVERY: Duration = Duration::from_millis(50);

/// What a command line sent to the database's daemon came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Forwarded {
    /// The daemon ran it: what it printed, and its exit code.
    Served {
        exit: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// No daemon answers for the database.
    NoDaemon,
    /// Another process holds a lease on the database.
    Leased,
    /// The daemon did not run it and wrote nothing, for this reason
    /// ([`RunReply::NotServed`]).
    NotServed(String),
    /// The client and the daemon don't match: `proto`, `build`, `features`
    /// or `env`.
    Incompatible(String),
    /// The daemon was still opening the database when the wait ran out.
    StartingTimeout,
    /// The daemon could not take it now: `stopping`, `overloaded`,
    /// `unavailable` (opening the database again), `expired` or `error`.
    Refused(&'static str),
    /// The connection broke. `sent` when the whole request had gone out,
    /// so the daemon may have run it.
    Lost { sent: bool },
}

/// How to send a command line.
#[derive(Debug, Clone)]
pub struct Opts {
    /// Whether the command may record a slow query in the database's log.
    pub slow_query_log: bool,
    /// How long each try may take, from sending the request to reading
    /// the answer; past it the daemon does not start the command. `None`
    /// waits as long as the command runs.
    pub deadline: Option<Duration>,
    /// How long to keep asking a daemon that answers it is opening the
    /// database, or has every request slot taken.
    pub starting_wait: Duration,
}

/// This process's identity, worked out once.
fn identity() -> Option<&'static Identity> {
    static IDENTITY: OnceLock<Option<Identity>> = OnceLock::new();
    IDENTITY.get_or_init(|| Identity::current().ok()).as_ref()
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Whether a socket is where the daemon of the database at `db` listens.
/// No socket is no daemon; one that is there may still be a dead one's.
pub fn may_have_daemon(db: &Path) -> bool {
    cli_io::db_key(db).is_ok_and(|key| std::fs::symlink_metadata(RunDir::of(&key).socket()).is_ok())
}

/// Send the command line `argv` (program name first), run in `cwd` with
/// `stdin` piped in (`None`: stdin is a terminal), to the daemon of the
/// database at `db`, and return what came of it.
pub fn forward(
    db: &Path,
    argv: Vec<String>,
    cwd: &Path,
    stdin: Option<Vec<u8>>,
    opts: &Opts,
) -> Forwarded {
    let Ok(key) = cli_io::db_key(db) else {
        return Forwarded::NoDaemon;
    };
    let run = RunDir::of(&key);
    if std::fs::symlink_metadata(run.socket()).is_err() {
        return Forwarded::NoDaemon;
    }
    let Some(identity) = identity() else {
        return Forwarded::Incompatible("build".into());
    };
    let mut header = identity.header(argv, cwd, stdin.is_some());
    header.slow_query_log = opts.slow_query_log;
    let retry_until = Instant::now() + opts.starting_wait;
    loop {
        match ask(&run, &mut header, stdin.as_deref(), opts.deadline) {
            Ask::Again(_) if Instant::now() + RETRY_EVERY < retry_until => {
                thread::sleep(RETRY_EVERY);
            }
            Ask::Again(RunReply::Starting) => return Forwarded::StartingTimeout,
            Ask::Again(RunReply::Overloaded) => return Forwarded::Refused("overloaded"),
            Ask::Again(_) => return Forwarded::Refused("unavailable"),
            Ask::Done(forwarded) => return forwarded,
        }
    }
}

enum Ask {
    /// The daemon is opening the database, is opening it again (after a
    /// change under its handle, `unavailable`), or has every slot taken.
    /// Each passes: a daemon that can't open its database stops.
    Again(RunReply),
    Done(Forwarded),
}

/// One try: connect, send, read the answer.
fn ask(
    run: &RunDir,
    header: &mut RunHeader,
    stdin: Option<&[u8]>,
    deadline: Option<Duration>,
) -> Ask {
    header.deadline_ms = deadline.map(|d| epoch_ms() + u64::try_from(d.as_millis()).unwrap_or(0));
    let wait = deadline.map_or(CLI_ANSWER_WAIT, |d| d + ANSWER_SLACK);
    // A socket nobody listens on, or one not to be trusted: the command
    // runs here, and its open says what is wrong when anything is.
    let Ok(mut stream) = proto::connect(run.socket(), wait) else {
        return Ask::Done(Forwarded::NoDaemon);
    };
    let sent = proto::write_json(&mut stream, &proto::Op::Run(header.clone()))
        .and_then(|()| stdin.map_or(Ok(()), |bytes| proto::write_frame(&mut stream, bytes)));
    if sent.is_err() {
        return Ask::Done(Forwarded::Lost { sent: false });
    }
    let Ok(reply) = proto::read_json::<RunReply>(&mut stream) else {
        return Ask::Done(Forwarded::Lost { sent: true });
    };
    Ask::Done(match reply {
        RunReply::Served { exit } => {
            match (
                proto::read_frame(&mut stream),
                proto::read_frame(&mut stream),
            ) {
                (Ok(stdout), Ok(stderr)) => Forwarded::Served {
                    exit,
                    stdout,
                    stderr,
                },
                _ => Forwarded::Lost { sent: true },
            }
        }
        RunReply::Starting | RunReply::Overloaded => return Ask::Again(reply),
        RunReply::NotServed { ref reason, .. } if reason == "unavailable" => {
            return Ask::Again(reply)
        }
        RunReply::NotServed { reason, .. } => Forwarded::NotServed(reason),
        RunReply::Incompatible { reason } => Forwarded::Incompatible(reason),
        RunReply::Leased => Forwarded::Leased,
        RunReply::Stopping => Forwarded::Refused("stopping"),
        RunReply::Expired => Forwarded::Refused("expired"),
        RunReply::Error { .. } => Forwarded::Refused("error"),
    })
}

/// The CLI's side: send `cli`, the command line this process was given, to
/// the daemon of its database, print what the daemon's run printed and
/// return its exit code. `None` when the command is to run in this process:
/// `AXIL_DAEMON` is `off`, no database resolves, the daemon does not run
/// the command, or none answers. Stdin read to send is given back to the
/// command then.
pub fn forward_cli(cli: &Cli) -> Option<i32> {
    if control::mode() == Mode::Off {
        return None;
    }
    exec::route(&cli.command).ok()?;
    let cwd = std::env::current_dir().ok()?;
    let db = match &cli.db {
        Some(db) => cwd.join(db),
        None => crate::find_axil_dir_from(cwd.clone())?,
    };
    if !may_have_daemon(&db) {
        return None;
    }
    // Sent as the daemon parses it, which only text can be.
    let argv: Vec<String> = std::env::args_os()
        .map(|arg| arg.into_string().ok())
        .collect::<Option<_>>()?;
    let stdin = if control::reads_stdin(&argv) && !cli_io::stdin_is_terminal() {
        Some(cli_io::read_stdin_ahead().ok()?)
    } else {
        None
    };
    let opts = Opts {
        slow_query_log: std::env::var("AXIL_SLOW_QUERY_LOG").map_or(true, |v| v != "0"),
        deadline: None,
        starting_wait: CLI_STARTING_WAIT,
    };
    match forward(&db, argv, &cwd, stdin.clone(), &opts) {
        Forwarded::Served {
            exit,
            stdout,
            stderr,
        } => {
            // A command prints its warnings before its results, mostly.
            cli_io::write_process_bytes(true, &stderr);
            cli_io::write_process_bytes(false, &stdout);
            Some(exit)
        }
        // The daemon may have run it: running it again could write twice.
        Forwarded::Lost { sent: true } if exec::writes(&cli.command) => {
            let lost = anyhow::anyhow!("daemon connection lost; write status unknown");
            errln!("{}", crate::format_cli_error(&lost));
            Some(crate::EXIT_ERROR)
        }
        _ => {
            if let Some(bytes) = stdin {
                cli_io::put_back_stdin(bytes);
            }
            None
        }
    }
}

/// A hook's lookup `args` (`recall …`, `boot …`), on the database at `db`:
/// `Some` with what the daemon's run answered (`Some(stdout)` when it ran
/// and exited 0, `None` for no context), `None` when the hook is to run the
/// lookup in a child process as it does with no daemon.
pub fn forward_lookup(db: &Path, args: &[&str]) -> Option<Option<String>> {
    if control::mode() == Mode::Off || !may_have_daemon(db) {
        return None;
    }
    let mut argv = vec![
        "axil".to_string(),
        "--db".to_string(),
        db.to_str()?.to_string(),
    ];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cwd = std::env::current_dir().ok()?;
    let (deadline, starting_wait) = match args.first().copied() {
        Some("recall") => (RECALL_DEADLINE, Duration::ZERO),
        Some("boot") => (LOOKUP_DEADLINE, BOOT_STARTING_WAIT),
        _ => (LOOKUP_DEADLINE, Duration::ZERO),
    };
    // A hook's read must not write, even a slow-query entry, under the
    // harness's kill timeout.
    let opts = Opts {
        slow_query_log: false,
        deadline: Some(deadline),
        starting_wait,
    };
    match forward(db, argv, &cwd, None, &opts) {
        Forwarded::Served {
            exit: 0, stdout, ..
        } => Some(Some(String::from_utf8_lossy(&stdout).into_owned())),
        // A daemon of another build or environment: the child runs the
        // lookup as this hook would, leasing the database as any command
        // does.
        Forwarded::NoDaemon | Forwarded::NotServed(_) | Forwarded::Incompatible(_) => None,
        Forwarded::Served { .. }
        | Forwarded::Leased
        | Forwarded::StartingTimeout
        | Forwarded::Refused(_)
        | Forwarded::Lost { .. } => Some(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// A database file of its own, with its run directory.
    fn new_db() -> (tempfile::TempDir, PathBuf, RunDir) {
        let tmp = tempfile::Builder::new().prefix("c").tempdir().unwrap();
        let db = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&db, b"").unwrap();
        let run = RunDir::of(&db);
        run.create().unwrap();
        (tmp, db, run)
    }

    /// What a daemon does with one connection.
    #[derive(Clone)]
    enum Then {
        /// Answer with this reply, and frames when it is `Served`.
        Reply(RunReply, Vec<u8>),
        /// Read the request, then close without a word.
        Hang,
    }

    /// A daemon on `run`'s socket answering its connections in turn, the
    /// last answer over and over, and keeping each request it read.
    fn daemon(run: &RunDir, answers: Vec<Then>) -> Arc<Mutex<Vec<RunHeader>>> {
        let listener = UnixListener::bind(run.socket()).unwrap();
        let asked = Arc::new(Mutex::new(Vec::new()));
        let seen = asked.clone();
        thread::spawn(move || {
            for (n, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { break };
                let Ok(proto::Op::Run(header)) = proto::read_json::<proto::Op>(&mut stream) else {
                    continue;
                };
                if header.has_stdin {
                    let _ = proto::read_frame(&mut stream);
                }
                seen.lock().unwrap().push(header);
                match answers[n.min(answers.len() - 1)].clone() {
                    Then::Reply(reply, out) => {
                        let served = matches!(reply, RunReply::Served { .. });
                        proto::write_json(&mut stream, &reply).unwrap();
                        if served {
                            proto::write_frame(&mut stream, &out).unwrap();
                            proto::write_frame(&mut stream, b"").unwrap();
                        }
                    }
                    Then::Hang => drop(stream),
                }
            }
        });
        asked
    }

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    fn cli_opts() -> Opts {
        Opts {
            slow_query_log: true,
            deadline: None,
            starting_wait: Duration::from_secs(5),
        }
    }

    /// No socket, or one nobody listens on: no daemon, found at once.
    #[test]
    fn with_no_daemon_nothing_is_sent() {
        let (_tmp, db, run) = new_db();
        let cwd = std::env::current_dir().unwrap();
        let started = Instant::now();
        let send = || forward(&db, argv(&["axil", "boot"]), &cwd, None, &cli_opts());
        assert_eq!(send(), Forwarded::NoDaemon);
        assert!(!may_have_daemon(&db));
        drop(UnixListener::bind(run.socket()).unwrap());
        assert!(may_have_daemon(&db));
        assert_eq!(send(), Forwarded::NoDaemon);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// A daemon still opening the database is asked again until it serves,
    /// within the wait; past it the client gives up.
    #[test]
    fn a_starting_daemon_is_asked_again_within_the_wait() {
        let (_tmp, db, run) = new_db();
        let cwd = std::env::current_dir().unwrap();
        let asked = daemon(
            &run,
            vec![
                Then::Reply(RunReply::Starting, Vec::new()),
                Then::Reply(RunReply::Overloaded, Vec::new()),
                Then::Reply(RunReply::Served { exit: 3 }, b"out".to_vec()),
            ],
        );
        let got = forward(&db, argv(&["axil", "boot"]), &cwd, None, &cli_opts());
        assert_eq!(
            got,
            Forwarded::Served {
                exit: 3,
                stdout: b"out".to_vec(),
                stderr: Vec::new()
            }
        );
        assert_eq!(asked.lock().unwrap().len(), 3);

        let (_tmp, db, run) = new_db();
        let _asked = daemon(&run, vec![Then::Reply(RunReply::Starting, Vec::new())]);
        let opts = Opts {
            starting_wait: Duration::from_millis(200),
            ..cli_opts()
        };
        let started = Instant::now();
        let got = forward(&db, argv(&["axil", "boot"]), &cwd, None, &opts);
        assert_eq!(got, Forwarded::StartingTimeout);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// The request carries what the client was given: its command line,
    /// stdin, the slow-query switch and a deadline.
    #[test]
    fn the_request_carries_the_clients_command() {
        let (_tmp, db, run) = new_db();
        let cwd = std::env::current_dir().unwrap();
        let asked = daemon(&run, vec![Then::Reply(RunReply::Leased, Vec::new())]);
        let opts = Opts {
            slow_query_log: false,
            deadline: Some(Duration::from_secs(2)),
            starting_wait: Duration::ZERO,
        };
        let argv = argv(&["axil", "store", "decisions", "-"]);
        let got = forward(&db, argv.clone(), &cwd, Some(b"{}".to_vec()), &opts);
        assert_eq!(got, Forwarded::Leased);
        let header = asked.lock().unwrap().pop().unwrap();
        assert_eq!(header.argv, argv);
        assert_eq!(header.cwd, cwd.to_string_lossy());
        assert!(header.has_stdin && !header.slow_query_log);
        let deadline = header.deadline_ms.unwrap();
        assert!(deadline > epoch_ms() && deadline <= epoch_ms() + 2000);
    }

    /// A connection that breaks once the request is out leaves it unknown
    /// whether the daemon ran it.
    #[test]
    fn a_connection_lost_after_the_request_went_out_is_lost() {
        let (_tmp, db, run) = new_db();
        let cwd = std::env::current_dir().unwrap();
        let _asked = daemon(&run, vec![Then::Hang]);
        let got = forward(&db, argv(&["axil", "boot"]), &cwd, None, &cli_opts());
        assert_eq!(got, Forwarded::Lost { sent: true });
    }

    /// What a hook's lookup gets for each answer: the output of a run that
    /// exited 0, no context when the daemon ran it otherwise or is not free
    /// to, and its own child when the daemon does not run the command.
    #[test]
    fn a_lookup_answers_or_leaves_it_to_the_child() {
        let not_served = RunReply::NotServed {
            reason: "command".into(),
            detail: String::new(),
        };
        let incompatible = RunReply::Incompatible {
            reason: "build".into(),
        };
        let cases = [
            (RunReply::Served { exit: 0 }, Some(Some("ctx".to_string()))),
            (RunReply::Served { exit: 1 }, Some(None)),
            (RunReply::Leased, Some(None)),
            (RunReply::Starting, Some(None)),
            (RunReply::Stopping, Some(None)),
            (RunReply::Expired, Some(None)),
            (
                RunReply::NotServed {
                    reason: "unavailable".into(),
                    detail: String::new(),
                },
                Some(None),
            ),
            (not_served, None),
            (incompatible, None),
        ];
        for (reply, want) in cases {
            let (_tmp, db, run) = new_db();
            let asked = daemon(&run, vec![Then::Reply(reply.clone(), b"ctx".to_vec())]);
            let got = forward_lookup(&db, &["recall", "auth"]);
            assert_eq!(got, want, "{reply:?}");
            let header = asked.lock().unwrap().pop().unwrap();
            assert!(!header.slow_query_log);
            assert_eq!(header.argv[..3], ["axil", "--db", db.to_str().unwrap()]);
        }
        let (_tmp, db, _run) = new_db();
        assert_eq!(forward_lookup(&db, &["boot"]), None);
    }
}
