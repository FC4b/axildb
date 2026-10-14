//! Where a database's daemon keeps its files: `<db dir>/run/`.
//!
//! - `daemon.lock`: held for the daemon's whole life; whoever holds it is
//!   the daemon.
//! - `axil.sock` (0600): the socket clients connect to. A path too long for
//!   a socket address moves it to `$TMPDIR/axil-<uid>/` (see
//!   [`RunDir::socket`]).
//! - `daemon.json`: what the daemon last said about itself, so `daemon
//!   status` can report on one that is gone.
//! - `daemon.log`, and `daemon.log.1` after a rotation: the daemon's stdout
//!   and stderr.
//! - `daemon.blocked`: why the daemon would not serve this database.
//! - `spawn.stamp`: when `daemon start` last spawned one.
//!
//! The directory is created 0700 and must belong to the user running axil:
//! anyone who can connect to the socket can run commands on the database.

use std::fs;
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// The longest socket path bound inside the run directory. A socket address
/// holds 104 bytes on macOS and 108 on Linux, the terminating NUL included;
/// staying under 100 leaves room on both.
const MAX_SOCKET_PATH: usize = 100;

/// The run directory of one database and the files in it.
#[derive(Debug, Clone)]
pub struct RunDir {
    dir: PathBuf,
    socket: PathBuf,
}

impl RunDir {
    /// The run directory of the database whose identity is `key` (see
    /// [`crate::cli_io::db_key`]): `run/` beside the database file. Nothing
    /// is created; see [`RunDir::create`].
    pub fn of(key: &Path) -> RunDir {
        let dir = key.parent().unwrap_or(Path::new(".")).join("run");
        let socket = socket_path(&dir, key);
        RunDir { dir, socket }
    }

    /// Create the directory, and the socket's own directory when the socket
    /// lives elsewhere, as directories only this user can enter. One that
    /// exists already must be a directory this user owns; one open to
    /// others is closed to them.
    pub fn create(&self) -> io::Result<()> {
        private_dir(&self.dir)?;
        match self.socket.parent() {
            Some(parent) if parent != self.dir => private_dir(parent),
            _ => Ok(()),
        }
    }

    /// The directory itself.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The socket: `axil.sock` in the run directory, or, when that path is
    /// 100 bytes or more, `$TMPDIR/axil-<uid>/<fnv1a64 of the database
    /// key>.sock`. A client and its daemon agree on it as long as they agree
    /// on `TMPDIR`.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Held by the daemon for as long as it runs.
    pub fn lock(&self) -> PathBuf {
        self.dir.join("daemon.lock")
    }

    /// What the daemon last reported about itself.
    pub fn info(&self) -> PathBuf {
        self.dir.join("daemon.json")
    }

    /// The daemon's stdout and stderr.
    pub fn log(&self) -> PathBuf {
        self.dir.join("daemon.log")
    }

    /// The log before the last rotation.
    pub fn old_log(&self) -> PathBuf {
        self.dir.join("daemon.log.1")
    }

    /// Why the daemon would not serve the database.
    pub fn blocked(&self) -> PathBuf {
        self.dir.join("daemon.blocked")
    }

    /// When `daemon start` last spawned a daemon.
    pub fn spawn_stamp(&self) -> PathBuf {
        self.dir.join("spawn.stamp")
    }
}

/// See [`RunDir::socket`].
fn socket_path(run: &Path, key: &Path) -> PathBuf {
    let socket = run.join("axil.sock");
    if socket.as_os_str().len() < MAX_SOCKET_PATH {
        return socket;
    }
    std::env::temp_dir()
        .join(format!("axil-{}", uid()))
        .join(format!("{:016x}.sock", fnv1a64(key.as_os_str().as_bytes())))
}

/// The effective user id of this process.
pub fn uid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

/// Make `dir` a directory only this user can use. It is created 0700 when
/// missing (its parent must exist). An existing one must be a real directory,
/// not a symlink, owned by this user; any access it grants others is
/// removed.
fn private_dir(dir: &Path) -> io::Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let meta = fs::symlink_metadata(dir)?;
    if !meta.file_type().is_dir() {
        return Err(io::Error::other(format!(
            "{} is not a directory",
            dir.display()
        )));
    }
    if meta.uid() != uid() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} belongs to uid {}, not to this user (uid {})",
                dir.display(),
                meta.uid(),
                uid()
            ),
        ));
    }
    if meta.mode() & 0o077 != 0 {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Write `bytes` to `path` so a reader sees the old contents or the new,
/// never part of either: through a temporary file beside it, renamed over.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no file name"))?;
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    let result = (|| {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// 64-bit FNV-1a. Short, stable across builds and platforms, and not a
/// security boundary: it names the fallback socket and condenses the
/// identities the handshake compares.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes
        .iter()
        .fold(OFFSET, |hash, &b| (hash ^ u64::from(b)).wrapping_mul(PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference vectors of the FNV-1a 64 definition. A client and a
    /// daemon from different builds must agree on the fallback socket.
    #[test]
    fn fnv1a64_matches_the_reference_values() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn the_socket_sits_in_the_run_dir_unless_the_path_is_too_long() {
        let short = RunDir::of(Path::new("/p/.axil/memory.axil"));
        assert_eq!(short.dir(), Path::new("/p/.axil/run"));
        assert_eq!(short.socket(), Path::new("/p/.axil/run/axil.sock"));

        let deep = format!("/{}/.axil/memory.axil", "d".repeat(90));
        let long = RunDir::of(Path::new(&deep));
        let hash = format!("{:016x}.sock", fnv1a64(deep.as_bytes()));
        assert_eq!(
            long.socket(),
            std::env::temp_dir()
                .join(format!("axil-{}", uid()))
                .join(hash)
        );
        assert!(long.socket().as_os_str().len() < 104);
    }

    #[test]
    fn the_run_dir_is_private_to_its_owner() {
        let tmp = tempfile::tempdir().unwrap();
        let run = RunDir::of(&tmp.path().join("memory.axil"));
        run.create().unwrap();
        let mode = fs::metadata(run.dir()).unwrap().mode();
        assert_eq!(mode & 0o777, 0o700);

        // Opened up by someone: closed again.
        fs::set_permissions(run.dir(), fs::Permissions::from_mode(0o755)).unwrap();
        run.create().unwrap();
        assert_eq!(fs::metadata(run.dir()).unwrap().mode() & 0o777, 0o700);

        // Not a directory: refused.
        let file = tmp.path().join("other");
        fs::create_dir(&file).unwrap();
        let blocked = RunDir::of(&file.join("memory.axil"));
        fs::write(blocked.dir(), b"").unwrap();
        assert!(blocked.create().is_err());

        // A symlink to a directory: refused too.
        let target = tmp.path().join("target");
        fs::create_dir(&target).unwrap();
        let linked = tmp.path().join("linked");
        fs::create_dir(&linked).unwrap();
        std::os::unix::fs::symlink(&target, linked.join("run")).unwrap();
        assert!(RunDir::of(&linked.join("memory.axil")).create().is_err());
    }

    #[test]
    fn an_atomic_write_replaces_the_file_whole() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("daemon.json");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two");
        let left: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
        assert_eq!(left.len(), 1, "no temporary file is left behind");
    }
}
