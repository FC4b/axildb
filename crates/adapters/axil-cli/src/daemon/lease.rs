//! Leasing a database from its daemon, for a process the daemon does not
//! serve.
//!
//! A daemon keeps its database open for as long as it serves it, and only
//! one process can have a database open. So a process about to open one
//! first asks the database's daemon, when one runs, to let go of it:
//! [`guard`] is the open guard ([`axil_core::open_guard`]) the CLI installs.
//! It runs before every open, and before every read of a store that comes
//! ahead of the open (the vector store's probe).
//!
//! The guard connects to the daemon's socket with the checks every client
//! makes ([`proto::connect`]: a socket of this user's, in a directory that
//! is this user's alone, answered by a process of this user's) and sends the
//! frozen `lease`. The daemon answers new requests `leased`, waits for the
//! ones running, closes the database and answers `granted`. The process
//! keeps the connection, never reading it again, until it exits: the lease
//! lasts as long as the process, however it ends, a SIGKILL included. The
//! socket is close-on-exec, so a child the process starts does not hold it
//! too. Once no process holds a lease, the daemon waits 250 ms and opens the
//! database again.
//!
//! The lease skips the handshake a run makes: it runs nothing for the
//! client, so a client of any build is answered, and one the daemon does not
//! serve can still open the database. A daemon from before leases existed
//! can't parse the request; it is asked to shut down with the frozen
//! `shutdown` instead, which every daemon answers.
//!
//! A daemon that can't let go within [`LEASE_WAIT`] (a request it runs
//! takes longer) answers `lease_timeout`, and the process opens as it would
//! with no daemon: the open waits for the lock, or fails busy. It says so in
//! one line on stderr first, naming `axil daemon stop`.
//!
//! A process asks once per database while its lease holds: it asks again
//! once the daemon it was granted by is gone (another may have started
//! since), [`REFUSED_FOR`] after a refusal, and at every open while no
//! daemon runs. `AXIL_DAEMON=off` turns the guard off.
//! In the daemon itself, the guard only checks, in a debug build, that no
//! command it runs opens the database it serves: those run on its open
//! handle, and its own opens skip the guard.

use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::control::{self, Mode};
use super::paths::RunDir;
use super::proto::{self, Ack, LeaseReply, Op};
use super::server;
use crate::cli_io;

/// How long a process waits for the daemon to let go of its database.
pub const LEASE_WAIT: Duration = Duration::from_secs(10);

/// How much longer than its wait a process waits for the daemon's answer.
const ANSWER_SLACK: Duration = Duration::from_secs(5);

/// How long a refused lease stands before an open asks again: long enough
/// that a process opening the database many times waits for the daemon
/// once, not at each open.
const REFUSED_FOR: Duration = Duration::from_secs(30);

/// The database this process serves, once it is a daemon.
static SERVING: OnceLock<PathBuf> = OnceLock::new();

/// The databases this process has asked for, and what it was answered. A
/// lease is held here until the process exits.
static ASKED: Mutex<Vec<(PathBuf, Asked)>> = Mutex::new(Vec::new());

/// What a database was asked for came to, while it stands.
#[derive(Debug)]
enum Asked {
    /// Granted, over this connection.
    Held(UnixStream),
    /// Refused at this time.
    Refused(Instant),
}

impl Asked {
    /// Whether it still stands: a lease while the daemon that granted it
    /// runs, a refusal for [`REFUSED_FOR`].
    fn stands(&self) -> bool {
        match self {
            Asked::Held(stream) => connected(stream),
            Asked::Refused(at) => at.elapsed() < REFUSED_FOR,
        }
    }
}

/// Whether the daemon is still at the other end of a lease. It never
/// writes on one, so a read that does not block finds nothing to read
/// while it runs, and the end of the stream once it has exited.
fn connected(stream: &UnixStream) -> bool {
    let mut byte = 0u8;
    // SAFETY: a one-byte read into a live buffer on an open descriptor;
    // MSG_PEEK leaves the byte unread and MSG_DONTWAIT keeps it from blocking.
    let n = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            std::ptr::addr_of_mut!(byte).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    match n {
        0 => false,
        1.. => true,
        _ => matches!(
            io::Error::last_os_error().kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
        ),
    }
}

/// Make [`guard`] this process's open guard.
pub fn install() {
    axil_core::set_open_guard(guard);
}

/// Mark this process as the daemon of the database `key`: from now on the
/// guard leases nothing, and only checks that `key` is not opened.
pub fn serve_as_daemon(key: &Path) {
    let _ = SERVING.set(key.to_path_buf());
}

/// Hold the database at `db` for the rest of this process's life, as a
/// server that keeps it open does before it opens it.
pub fn hold(db: &Path) {
    axil_core::open_guard(db);
}

/// The open guard: lease the database at `path` from its daemon, once
/// while the lease holds (see the module docs).
pub fn guard(path: &Path) {
    let Ok(key) = cli_io::db_key(path) else {
        return;
    };
    if let Some(serving) = SERVING.get() {
        debug_assert!(
            &key != serving,
            "the daemon opened {}, the database it serves: the commands it runs must use \
             its open handle",
            key.display()
        );
        return;
    }
    if control::mode() == Mode::Off {
        return;
    }
    let mut asked = ASKED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(i) = asked.iter().position(|(k, _)| *k == key) {
        if asked[i].1.stands() {
            return;
        }
        asked.swap_remove(i);
    }
    match lease(&key, LEASE_WAIT) {
        Lease::NoDaemon => {}
        Lease::Held(stream) => asked.push((key, Asked::Held(stream))),
        Lease::Refused(said) => {
            if !said.is_empty() {
                errln!("{said}");
            }
            asked.push((key, Asked::Refused(Instant::now())));
        }
    }
}

/// What asking for a lease came to.
#[derive(Debug)]
pub enum Lease {
    /// No daemon has the database open (any more): none answers, or the one
    /// that did has exited. Asked again at the next open, since one may
    /// start meanwhile.
    NoDaemon,
    /// The daemon closed the database, and keeps it closed while this
    /// connection is open.
    Held(UnixStream),
    /// The daemon did not let go in time, or its socket is not to be
    /// trusted: the line to say so, empty once said. Not asked again for
    /// [`REFUSED_FOR`].
    Refused(String),
}

/// A refusal from the daemon of `key`, which still has it open, for `why`.
fn refused(key: &Path, why: &str) -> Lease {
    Lease::Refused(format!(
        "axil: warning: the daemon has {} open and did not let go of it ({why}); \
         `axil daemon stop` frees it",
        key.display()
    ))
}

/// Ask the daemon of the database `key`, when one runs, to let go of it,
/// waiting up to `wait`.
pub fn lease(key: &Path, wait: Duration) -> Lease {
    let run = RunDir::of(key);
    if std::fs::symlink_metadata(run.socket()).is_err() {
        // A daemon shutting down removes its socket first, and closes the
        // database after the requests it runs: wait for it to be done.
        if server::said_stopping(&run) && server::lock_held(&run) && !gone(&run, wait) {
            return refused(key, "it is still shutting down");
        }
        return Lease::NoDaemon;
    }
    let mut stream = match proto::connect(run.socket(), wait + ANSWER_SLACK) {
        Ok(stream) => stream,
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
            return Lease::Refused(format!(
                "axil: warning: not asking the daemon on {} to let go of the database: {e}",
                run.socket().display()
            ));
        }
        // Nothing listens on the socket: the daemon that bound it is gone.
        Err(_) => return Lease::NoDaemon,
    };
    let op = Op::Lease {
        db: key.display().to_string(),
        wait_ms: u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
        pid: std::process::id(),
    };
    let answer =
        proto::write_json(&mut stream, &op).and_then(|()| proto::read_json::<Value>(&mut stream));
    let answer = match answer {
        Ok(answer) => answer,
        Err(e) => return refused(key, &format!("it did not answer: {e}")),
    };
    if answer["op"] != "lease" {
        return retire(key, &run, wait);
    }
    match serde_json::from_value::<LeaseReply>(answer) {
        Ok(reply) if reply.granted => {
            // Held, unread, until the process exits.
            let _ = stream.set_read_timeout(None);
            let _ = stream.set_write_timeout(None);
            Lease::Held(stream)
        }
        // It closes the database as it goes.
        Ok(reply) if reply.reason.as_deref() == Some("stopping") => {
            if gone(&run, wait) {
                Lease::NoDaemon
            } else {
                refused(key, "it is still shutting down")
            }
        }
        Ok(reply) => match reply.reason.as_deref() {
            Some("lease_timeout") => {
                refused(key, &format!("it was still busy after {}s", wait.as_secs()))
            }
            Some(reason) => refused(key, reason),
            None => refused(key, "it gave no reason"),
        },
        Err(e) => refused(key, &format!("its answer was not understood: {e}")),
    }
}

/// Ask the daemon of `key`, one that does not know `lease`, to shut down,
/// and wait up to `wait` for it to be gone.
fn retire(key: &Path, run: &RunDir, wait: Duration) -> Lease {
    let op = Op::Shutdown {
        reason: "a process needs the database, and this daemon does not answer lease".into(),
    };
    if let Err(e) = proto::call::<Ack>(run.socket(), &op, Duration::from_secs(5)) {
        return refused(
            key,
            &format!("it predates leases, and did not answer a shutdown: {e}"),
        );
    }
    if gone(run, wait) {
        Lease::NoDaemon
    } else {
        refused(key, "it predates leases, and is still shutting down")
    }
}

/// Wait up to `wait` for the daemon to let its lock go, which it does only
/// once it has closed the database or exited.
fn gone(run: &RunDir, wait: Duration) -> bool {
    let deadline = Instant::now() + wait;
    while server::lock_held(run) {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No socket, or a socket nobody listens on: no daemon, and nothing is
    /// waited for.
    #[test]
    fn with_no_daemon_nothing_is_leased() {
        let tmp = tempfile::Builder::new().prefix("l").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&key, b"").unwrap();
        let started = Instant::now();
        assert!(matches!(lease(&key, LEASE_WAIT), Lease::NoDaemon));

        let run = RunDir::of(&key);
        run.create().unwrap();
        drop(std::os::unix::net::UnixListener::bind(run.socket()).unwrap());
        assert!(matches!(lease(&key, LEASE_WAIT), Lease::NoDaemon));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// The lease is a socket a child process does not inherit: a holder's
    /// child would otherwise keep the database closed after the holder
    /// exits.
    #[test]
    fn a_lease_is_close_on_exec() {
        let tmp = tempfile::Builder::new().prefix("l").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&key, b"").unwrap();
        let run = RunDir::of(&key);
        run.create().unwrap();
        let listener = std::os::unix::net::UnixListener::bind(run.socket()).unwrap();
        let daemon = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let op: Op = proto::read_json(&mut stream).unwrap();
            assert!(matches!(op, Op::Lease { pid, .. } if pid == std::process::id()));
            proto::write_json(&mut stream, &LeaseReply::granted()).unwrap();
            stream
        });
        let Lease::Held(held) = lease(&key, LEASE_WAIT) else {
            panic!("not granted");
        };
        let _daemon_end = daemon.join().unwrap();
        // SAFETY: F_GETFD on an open descriptor only reads its flags.
        let flags = unsafe { libc::fcntl(held.as_raw_fd(), libc::F_GETFD) };
        assert!(flags >= 0 && flags & libc::FD_CLOEXEC != 0, "flags {flags}");
        assert_eq!(held.read_timeout().unwrap(), None);
    }

    /// A daemon from before leases can't parse the request: it is asked to
    /// shut down, with the frozen `shutdown` every daemon answers, so the
    /// database is not left busy for good.
    #[test]
    fn a_daemon_that_does_not_know_lease_is_asked_to_go() {
        let tmp = tempfile::Builder::new().prefix("l").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&key, b"").unwrap();
        let run = RunDir::of(&key);
        run.create().unwrap();
        let listener = std::os::unix::net::UnixListener::bind(run.socket()).unwrap();
        let daemon = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // As a daemon without the op answers it.
            assert!(proto::read_frame(&mut stream).is_ok());
            let unknown = proto::RunReply::Error {
                message: "bad request: unknown variant `lease`".into(),
            };
            proto::write_json(&mut stream, &unknown).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            let op: Op = proto::read_json(&mut stream).unwrap();
            let ack = Ack {
                op: "shutdown".into(),
                ok: true,
            };
            proto::write_json(&mut stream, &ack).unwrap();
            op
        });
        assert!(matches!(lease(&key, LEASE_WAIT), Lease::NoDaemon));
        assert!(matches!(daemon.join().unwrap(), Op::Shutdown { .. }));
    }

    /// A database file of its own, for a guard's cache of no other test.
    fn new_key() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::Builder::new().prefix("l").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&key, b"").unwrap();
        (tmp, key)
    }

    /// A daemon for `key` that gives every lease asked of it `reply` and
    /// keeps the connection.
    struct Answering {
        asks: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        held: std::sync::Arc<Mutex<Vec<UnixStream>>>,
    }

    impl Answering {
        fn new(key: &Path, reply: LeaseReply) -> Self {
            let run = RunDir::of(key);
            run.create().unwrap();
            let listener = std::os::unix::net::UnixListener::bind(run.socket()).unwrap();
            let me = Answering {
                asks: Default::default(),
                held: Default::default(),
            };
            let (asks, held) = (me.asks.clone(), me.held.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { break };
                    if proto::read_json::<Op>(&mut stream).is_err() {
                        continue;
                    }
                    asks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    proto::write_json(&mut stream, &reply).unwrap();
                    held.lock().unwrap().push(stream);
                }
            });
            me
        }

        fn asks(&self) -> usize {
            self.asks.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    fn kept(key: &Path) -> bool {
        let asked = ASKED.lock().unwrap_or_else(PoisonError::into_inner);
        asked.iter().any(|(k, _)| k == key)
    }

    /// With no daemon nothing is kept, so the next open asks again: one may
    /// have started. A lease is asked for once while it holds, and again
    /// once the daemon that granted it is gone.
    #[test]
    fn a_lease_is_asked_for_once_while_it_holds() {
        let (_tmp, key) = new_key();
        guard(&key);
        assert!(!kept(&key));

        let daemon = Answering::new(&key, LeaseReply::granted());
        guard(&key);
        guard(&key);
        assert_eq!(daemon.asks(), 1);
        assert!(kept(&key));

        // As when the daemon exits, and another starts.
        daemon.held.lock().unwrap().clear();
        guard(&key);
        assert_eq!(daemon.asks(), 2);
        guard(&key);
        assert_eq!(daemon.asks(), 2);
    }

    /// A refusal is not asked again at each open, only once it is old.
    #[test]
    fn a_refused_lease_is_asked_again_once_it_is_old() {
        let (_tmp, key) = new_key();
        let daemon = Answering::new(&key, LeaseReply::refused("lease_timeout"));
        guard(&key);
        guard(&key);
        assert_eq!(daemon.asks(), 1);
        {
            let mut asked = ASKED.lock().unwrap();
            let (_, at) = asked.iter_mut().find(|(k, _)| *k == key).unwrap();
            *at = Asked::Refused(Instant::now() - REFUSED_FOR);
        }
        guard(&key);
        assert_eq!(daemon.asks(), 2);
    }

    /// A refusal says what the daemon holds and how to free it.
    #[test]
    fn a_refused_lease_says_how_to_free_the_database() {
        let (_tmp, key) = new_key();
        let _daemon = Answering::new(&key, LeaseReply::refused("lease_timeout"));
        let Lease::Refused(said) = lease(&key, Duration::from_secs(10)) else {
            panic!("not refused");
        };
        assert_eq!(
            said,
            format!(
                "axil: warning: the daemon has {} open and did not let go of it (it was \
                 still busy after 10s); `axil daemon stop` frees it",
                key.display()
            )
        );
    }

    /// A socket in a directory others can enter is not asked anything.
    #[test]
    fn an_untrusted_socket_is_not_asked() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::Builder::new().prefix("l").tempdir().unwrap();
        let key = tmp.path().canonicalize().unwrap().join("memory.axil");
        std::fs::write(&key, b"").unwrap();
        let run = RunDir::of(&key);
        if run.socket().parent() != Some(run.dir()) {
            eprintln!("skipped: the socket is not in the run directory");
            return;
        }
        run.create().unwrap();
        let _listener = std::os::unix::net::UnixListener::bind(run.socket()).unwrap();
        std::fs::set_permissions(run.dir(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(lease(&key, LEASE_WAIT), Lease::Refused(_)));
    }
}
