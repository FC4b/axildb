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
//! with no daemon: the open waits for the lock, or fails busy.
//!
//! A process asks once per database. `AXIL_DAEMON=off` turns the guard off.
//! In the daemon itself, the guard only checks, in a debug build, that no
//! command it runs opens the database it serves: those run on its open
//! handle, and its own opens skip the guard.

use std::io;
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

/// The database this process serves, once it is a daemon.
static SERVING: OnceLock<PathBuf> = OnceLock::new();

/// The databases this process has asked for, each with the lease it was
/// granted, if any. A lease is held here until the process exits.
static ASKED: Mutex<Vec<(PathBuf, Option<UnixStream>)>> = Mutex::new(Vec::new());

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

/// The open guard: lease the database at `path` from its daemon, once per
/// process (see the module docs).
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
    if asked.iter().any(|(k, _)| *k == key) {
        return;
    }
    match lease(&key, LEASE_WAIT) {
        Lease::NoDaemon => {}
        Lease::Held(stream) => asked.push((key, Some(stream))),
        Lease::Refused => asked.push((key, None)),
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
    /// trusted. Not asked again.
    Refused,
}

/// Ask the daemon of the database `key`, when one runs, to let go of it,
/// waiting up to `wait`.
pub fn lease(key: &Path, wait: Duration) -> Lease {
    let run = RunDir::of(key);
    if std::fs::symlink_metadata(run.socket()).is_err() {
        // A daemon shutting down removes its socket first, and closes the
        // database after the requests it runs: wait for it to be done.
        if server::said_stopping(&run) && server::lock_held(&run) && !gone(&run, wait) {
            return Lease::Refused;
        }
        return Lease::NoDaemon;
    }
    let mut stream = match proto::connect(run.socket(), wait + ANSWER_SLACK) {
        Ok(stream) => stream,
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
            errln!(
                "axil: warning: not asking the daemon on {} to let go of the database: {e}",
                run.socket().display()
            );
            return Lease::Refused;
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
    let Ok(answer) = answer else {
        return Lease::Refused;
    };
    if answer["op"] != "lease" {
        return retire(&run, wait);
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
                Lease::Refused
            }
        }
        _ => Lease::Refused,
    }
}

/// Ask a daemon that does not know `lease` to shut down, and wait up to
/// `wait` for it to be gone.
fn retire(run: &RunDir, wait: Duration) -> Lease {
    let op = Op::Shutdown {
        reason: "a process needs the database, and this daemon does not answer lease".into(),
    };
    if proto::call::<Ack>(run.socket(), &op, Duration::from_secs(5)).is_err() {
        return Lease::Refused;
    }
    if gone(run, wait) {
        Lease::NoDaemon
    } else {
        Lease::Refused
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
    use std::os::fd::AsRawFd;

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
        assert!(matches!(lease(&key, LEASE_WAIT), Lease::Refused));
    }
}
