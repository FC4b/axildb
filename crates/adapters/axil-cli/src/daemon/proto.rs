//! The daemon's wire format, and the identities a client and a daemon
//! compare before a command runs.
//!
//! A connection carries one exchange. Every message is a frame: a u32
//! big-endian length, then that many bytes, at most [`MAX_FRAME`]. The
//! client sends a JSON request tagged by `"op"`; a `run` request is followed
//! by one frame of raw stdin when its header says it has one. The daemon
//! answers with one JSON frame; a served run is followed by two raw frames,
//! stdout then stderr.
//!
//! `hello` and `shutdown` are frozen: every daemon answers them the same way
//! whatever its [`DAEMON_PROTO`], so a client of any build can ask what is
//! running and ask it to go. Everything else may change with the protocol
//! version, which a run header carries and the daemon checks.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::exec::{self, NotServed};
use super::paths::fnv1a64;

/// The protocol version a run header must carry.
pub const DAEMON_PROTO: u32 = 1;

/// The largest frame either side sends or accepts: 32 MiB.
pub const MAX_FRAME: usize = 32 * 1024 * 1024;

/// The version string `axil --version` prints.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), env!("AXIL_VERSION_SUFFIX"));

/// Write one frame.
pub fn write_frame(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .ok()
        .filter(|&n| n as usize <= MAX_FRAME)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a {}-byte frame is over the 32 MiB limit", payload.len()),
            )
        })?;
    w.write_all(&len.to_be_bytes())?;
    w.write_all(payload)
}

/// Read one frame.
pub fn read_frame(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len) as usize;
    if len > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a {len}-byte frame is over the 32 MiB limit"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Write `value` as one JSON frame.
pub fn write_json(w: &mut impl Write, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    write_frame(w, &bytes)?;
    w.flush()
}

/// Read one JSON frame as a `T`.
pub fn read_json<T: DeserializeOwned>(r: &mut impl Read) -> io::Result<T> {
    let bytes = read_frame(r)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// A request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Frozen: `{"op":"hello"}`, answered by [`Hello`].
    Hello,
    /// Frozen: `{"op":"shutdown","reason":"..."}`, answered by
    /// `{"op":"shutdown","ok":true}` before the daemon starts shutting down.
    Shutdown {
        #[serde(default)]
        reason: String,
    },
    /// Run one command line; see [`RunHeader`].
    Run(RunHeader),
    /// What `daemon status` reports while the daemon runs.
    Status,
    /// Debug builds only: report the graph as failed to load, so a test can
    /// watch the daemon reload. Answered by `{"op":"debug_fail_graph","ok":true}`.
    #[cfg(debug_assertions)]
    DebugFailGraph,
}

/// The frozen answer to `hello`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// Always `"hello"`.
    pub op: String,
    /// The daemon's [`DAEMON_PROTO`].
    pub proto: u32,
    /// See [`Identity::build_id`].
    pub build_id: String,
    /// The daemon's `axil --version`.
    pub version: String,
    /// The daemon's process id.
    pub pid: u32,
    /// `starting`, `ready`, `reloading` or `stopping`.
    pub state: String,
}

/// The frozen answer to `shutdown`, and the answer to the debug ops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ack {
    /// The op answered.
    pub op: String,
    /// Always true: the daemon does what was asked.
    pub ok: bool,
}

/// One command line to run, as the client's process was asked to run it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunHeader {
    /// Must equal the daemon's [`DAEMON_PROTO`].
    pub proto: u32,
    /// The client's build ([`Identity::build_id`]); only an exact match is
    /// served, so a command runs only on the code of the binary it was
    /// typed into.
    pub build_id: String,
    /// The client's compiled features; the daemon must have every one.
    pub features: Vec<String>,
    /// The client's [`env_fp`].
    pub env_fp: String,
    /// The command line, program name first.
    pub argv: Vec<String>,
    /// The client's working directory, absolute.
    pub cwd: String,
    /// The client's values of the variables in [`exec::CLI_ENV`] it has set.
    pub env: Vec<(String, String)>,
    /// A frame of stdin follows the header. Without one, the client's stdin
    /// is a terminal.
    pub has_stdin: bool,
    /// Whether the command may record a slow query in the database's log.
    pub slow_query_log: bool,
    /// When the client stops waiting, in milliseconds since the Unix epoch.
    /// A request still queued then is answered [`RunReply::Expired`] and not
    /// run.
    pub deadline_ms: Option<u64>,
}

impl RunHeader {
    /// The request [`exec::execute_captured`] runs. Everything is a string
    /// on the wire: a client with a non-UTF-8 argument or path keeps the
    /// command.
    pub fn to_request(&self, stdin: Option<Vec<u8>>) -> exec::Request {
        exec::Request {
            argv: self.argv.iter().map(OsString::from).collect(),
            cwd: PathBuf::from(&self.cwd),
            env: self
                .env
                .iter()
                .map(|(n, v)| (OsString::from(n), OsString::from(v)))
                .collect(),
            stdin,
            slow_query_log: self.slow_query_log,
        }
    }
}

/// The answer to a run request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum RunReply {
    /// The command ran; two frames follow, its stdout and its stderr.
    Served {
        /// Its exit code.
        exit: i32,
    },
    /// The daemon did not run it and wrote nothing; the client runs it.
    NotServed {
        /// `parse`, `wrong_db`, `command`, `lacks`, `terminal`,
        /// `unavailable` or `too_large`.
        reason: String,
        /// The reason in words.
        detail: String,
    },
    /// The client and the daemon don't match; the client runs it.
    Incompatible {
        /// `proto`, `build`, `features` or `env`.
        reason: String,
    },
    /// The database is still opening.
    Starting,
    /// The daemon is shutting down.
    Stopping,
    /// Too many requests are running.
    Overloaded,
    /// The request's deadline passed before it could run.
    Expired,
    /// The request could not be read.
    Error {
        /// What went wrong.
        message: String,
    },
}

impl RunReply {
    /// The reply for a request [`exec::execute_captured`] did not serve.
    pub fn not_served(reason: NotServed) -> RunReply {
        let code = match reason {
            NotServed::Parse => "parse",
            NotServed::WrongDb => "wrong_db",
            NotServed::Command => "command",
            NotServed::Lacks(_) => "lacks",
            NotServed::Terminal => "terminal",
            NotServed::Unavailable => "unavailable",
        };
        RunReply::NotServed {
            reason: code.to_string(),
            detail: reason.to_string(),
        }
    }
}

/// The variables the environment fingerprint covers: each one changes what
/// a command prints or the data it reads. A served command uses the
/// daemon's values of these, so it is served only to a client whose values
/// are the same. `AXIL_ENC_KEY_FILE` also covers the key file's size and
/// modification time.
pub const FP_ENV: &[&str] = &[
    "AXIL_ENC_KEY",
    "AXIL_ENC_KEY_FILE",
    "AXIL_LLM_API_KEY",
    "AXIL_VERBOSE",
    // The embedder prints the pool size when it is above one.
    "AXIL_EMBED_SESSIONS",
    "HOME",
    "USERPROFILE",
    "XDG_CONFIG_HOME",
    "APPDATA",
    "ORT_DYLIB_PATH",
    "CARGO_HOME",
    "GOPATH",
    "GOMODCACHE",
];

/// The variables the daemon removes from its own environment before it
/// opens the database. Each one is the client's to give: the
/// [`exec::CLI_ENV`] values travel with each request, a slow-query opt-out
/// with the header's `slow_query_log`, and a client that has `AXIL_VERBOSE`
/// or `AXIL_EMBED_SESSIONS` set doesn't match the daemon's [`env_fp`] and
/// runs the command itself, printing what they make it print.
pub const STRIPPED_ENV: &[&str] = &[
    "AXIL_DB",
    "AXIL_MCP_TOOLS",
    "AXIL_VERBOSE",
    "AXIL_EMBED_SESSIONS",
    "AXIL_SLOW_QUERY_LOG",
];

/// A fingerprint of the [`FP_ENV`] values `get` returns, in hex. The
/// values never leave the process: only the fingerprint is sent, and it is
/// never written to disk.
pub fn env_fp(get: impl Fn(&str) -> Option<OsString>) -> String {
    let mut bytes = Vec::new();
    for name in FP_ENV {
        bytes.extend_from_slice(name.as_bytes());
        match get(name) {
            Some(value) => {
                bytes.push(1);
                bytes.extend_from_slice(value.as_bytes());
                if *name == "AXIL_ENC_KEY_FILE" && !value.is_empty() {
                    match std::fs::metadata(Path::new(&value)) {
                        Ok(meta) => bytes.extend_from_slice(
                            format!("|{}|{}|{}", meta.len(), meta.mtime(), meta.mtime_nsec())
                                .as_bytes(),
                        ),
                        Err(_) => bytes.extend_from_slice(b"|missing"),
                    }
                }
            }
            None => bytes.push(0),
        }
        bytes.push(0xff);
    }
    format!("{:016x}", fnv1a64(&bytes))
}

/// The file the running binary was loaded from, as it is on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExeId {
    /// Canonical path.
    pub path: PathBuf,
    /// Device, inode, size and modification time: replacing the file or
    /// writing over it changes at least one.
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub mtime_ns: i128,
}

impl ExeId {
    /// This process's executable.
    pub fn current() -> io::Result<ExeId> {
        ExeId::of(&std::env::current_exe()?)
    }

    /// The executable at `path`.
    pub fn of(path: &Path) -> io::Result<ExeId> {
        let path = std::fs::canonicalize(path)?;
        let meta = std::fs::metadata(&path)?;
        Ok(ExeId {
            dev: meta.dev(),
            ino: meta.ino(),
            len: meta.len(),
            mtime_ns: i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec()),
            path,
        })
    }

    /// The build id of the binary: a hash of the version and the file's
    /// path and stat.
    pub fn build_id(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(VERSION.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.path.as_os_str().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(
            format!("{}|{}|{}|{}", self.dev, self.ino, self.len, self.mtime_ns).as_bytes(),
        );
        format!("{:016x}", fnv1a64(&bytes))
    }
}

/// What a client and a daemon compare before a command runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// The binary ([`ExeId::build_id`]). Two binaries of the same version
    /// built apart, an installed one and a dev build say, differ.
    pub build_id: String,
    /// The compiled features.
    pub features: Vec<String>,
    /// The environment ([`env_fp`]).
    pub env_fp: String,
}

impl Identity {
    /// This process's identity, from its executable and environment.
    pub fn current() -> io::Result<Identity> {
        Ok(Identity {
            build_id: ExeId::current()?.build_id(),
            features: compiled_features(),
            env_fp: env_fp(|name| std::env::var_os(name)),
        })
    }

    /// Why a client with `header` must not be served by a daemon with this
    /// identity: `proto`, `build`, `features` or `env`. `None` when it may.
    pub fn mismatch(&self, header: &RunHeader) -> Option<&'static str> {
        if header.proto != DAEMON_PROTO {
            return Some("proto");
        }
        if header.build_id != self.build_id {
            return Some("build");
        }
        let ours: BTreeSet<&str> = self.features.iter().map(String::as_str).collect();
        if !header.features.iter().all(|f| ours.contains(f.as_str())) {
            return Some("features");
        }
        if header.env_fp != self.env_fp {
            return Some("env");
        }
        None
    }

    /// A run header for `argv` from this process, in `cwd`.
    pub fn header(&self, argv: Vec<String>, cwd: &Path, has_stdin: bool) -> RunHeader {
        RunHeader {
            proto: DAEMON_PROTO,
            build_id: self.build_id.clone(),
            features: self.features.clone(),
            env_fp: self.env_fp.clone(),
            argv,
            cwd: cwd.to_string_lossy().into_owned(),
            env: exec::CLI_ENV
                .iter()
                .filter_map(|name| {
                    let value = std::env::var(name).ok()?;
                    Some((name.to_string(), value))
                })
                .collect(),
            has_stdin,
            slow_query_log: true,
            deadline_ms: None,
        }
    }
}

/// The features this binary was built with.
pub fn compiled_features() -> Vec<String> {
    crate::features::compiled_set()
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Connect to the daemon's socket, with `timeout` on each read and write.
pub fn connect(socket: &Path, timeout: Duration) -> io::Result<UnixStream> {
    let stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(stream)
}

/// Send `op` and read the JSON answer.
pub fn call<T: DeserializeOwned>(socket: &Path, op: &Op, timeout: Duration) -> io::Result<T> {
    let mut stream = connect(socket, timeout)?;
    write_json(&mut stream, op)?;
    read_json(&mut stream)
}

/// The frozen `hello`.
pub fn hello(socket: &Path, timeout: Duration) -> io::Result<Hello> {
    call(socket, &Op::Hello, timeout)
}

/// What a run request came back with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The daemon's answer.
    pub reply: RunReply,
    /// The command's stdout, when it was served.
    pub stdout: Vec<u8>,
    /// The command's stderr, when it was served.
    pub stderr: Vec<u8>,
}

/// Send one run request, with `stdin` when the client's stdin is not a
/// terminal, and read the answer.
pub fn run(
    socket: &Path,
    header: &RunHeader,
    stdin: Option<&[u8]>,
    timeout: Duration,
) -> io::Result<RunOutcome> {
    debug_assert_eq!(header.has_stdin, stdin.is_some());
    let mut stream = connect(socket, timeout)?;
    write_json(&mut stream, &Op::Run(header.clone()))?;
    if let Some(stdin) = stdin {
        write_frame(&mut stream, stdin)?;
    }
    let reply: RunReply = read_json(&mut stream)?;
    let (stdout, stderr) = match reply {
        RunReply::Served { .. } => (read_frame(&mut stream)?, read_frame(&mut stream)?),
        _ => (Vec::new(), Vec::new()),
    };
    Ok(RunOutcome {
        reply,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hello` and `shutdown` look like this to every build, forever.
    #[test]
    fn the_frozen_ops_keep_their_shape() {
        assert_eq!(
            serde_json::to_string(&Op::Hello).unwrap(),
            r#"{"op":"hello"}"#
        );
        assert_eq!(
            serde_json::from_str::<Op>(r#"{"op":"hello"}"#).unwrap(),
            Op::Hello
        );
        // A later build's hello may carry more; it is still a hello.
        assert_eq!(
            serde_json::from_str::<Op>(r#"{"op":"hello","client":"x"}"#).unwrap(),
            Op::Hello
        );
        let hello = Hello {
            op: "hello".into(),
            proto: 1,
            build_id: "00ff".into(),
            version: "2.3.0".into(),
            pid: 42,
            state: "ready".into(),
        };
        assert_eq!(
            serde_json::to_string(&hello).unwrap(),
            r#"{"op":"hello","proto":1,"build_id":"00ff","version":"2.3.0","pid":42,"state":"ready"}"#
        );

        let shutdown = Op::Shutdown {
            reason: "stop".into(),
        };
        assert_eq!(
            serde_json::to_string(&shutdown).unwrap(),
            r#"{"op":"shutdown","reason":"stop"}"#
        );
        assert_eq!(
            serde_json::from_str::<Op>(r#"{"op":"shutdown"}"#).unwrap(),
            Op::Shutdown {
                reason: String::new()
            }
        );
        let ack = Ack {
            op: "shutdown".into(),
            ok: true,
        };
        assert_eq!(
            serde_json::to_string(&ack).unwrap(),
            r#"{"op":"shutdown","ok":true}"#
        );
        assert_eq!(DAEMON_PROTO, 1);
    }

    #[test]
    fn frames_are_a_big_endian_length_then_the_bytes() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"abc").unwrap();
        assert_eq!(buf, [0, 0, 0, 3, b'a', b'b', b'c']);
        assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), b"abc");

        let mut over = Vec::new();
        over.extend_from_slice(&(MAX_FRAME as u32 + 1).to_be_bytes());
        let err = read_frame(&mut over.as_slice()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(write_frame(&mut Vec::new(), &vec![0u8; MAX_FRAME + 1]).is_err());
    }

    #[test]
    fn a_run_request_is_the_header_beside_its_op() {
        let id = Identity {
            build_id: "b".into(),
            features: vec!["core".into()],
            env_fp: "e".into(),
        };
        let header = id.header(vec!["axil".into(), "boot".into()], Path::new("/w"), false);
        let json = serde_json::to_value(Op::Run(header.clone())).unwrap();
        assert_eq!(json["op"], "run");
        assert_eq!(json["argv"], serde_json::json!(["axil", "boot"]));
        assert_eq!(serde_json::from_value::<Op>(json).unwrap(), Op::Run(header));
        let served = serde_json::to_string(&RunReply::Served { exit: 0 }).unwrap();
        assert_eq!(served, r#"{"reply":"served","exit":0}"#);
    }

    #[test]
    fn the_handshake_names_the_first_mismatch() {
        let id = Identity {
            build_id: "b".into(),
            features: vec!["core".into(), "fts".into()],
            env_fp: "e".into(),
        };
        let ok = id.header(vec!["axil".into()], Path::new("/"), false);
        assert_eq!(id.mismatch(&ok), None);
        let fewer = RunHeader {
            features: vec!["fts".into()],
            ..ok.clone()
        };
        assert_eq!(id.mismatch(&fewer), None, "a subset of ours is served");
        let cases = [
            (
                RunHeader {
                    proto: 2,
                    ..ok.clone()
                },
                "proto",
            ),
            (
                RunHeader {
                    build_id: "x".into(),
                    ..ok.clone()
                },
                "build",
            ),
            (
                RunHeader {
                    features: vec!["core".into(), "wasm-host".into()],
                    ..ok.clone()
                },
                "features",
            ),
            (
                RunHeader {
                    env_fp: "x".into(),
                    ..ok.clone()
                },
                "env",
            ),
        ];
        for (header, reason) in cases {
            assert_eq!(id.mismatch(&header), Some(reason));
        }
    }

    #[test]
    fn the_env_fingerprint_follows_every_variable_and_the_key_file() {
        let tmp = tempfile::tempdir().unwrap();
        let key = tmp.path().join("key");
        std::fs::write(&key, b"k1").unwrap();
        let env = |pairs: &[(&str, &str)]| {
            let pairs: Vec<(String, OsString)> = pairs
                .iter()
                .map(|(n, v)| (n.to_string(), OsString::from(v)))
                .collect();
            env_fp(move |name| {
                pairs
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v.clone())
            })
        };
        let base = env(&[]);
        assert_eq!(base, env(&[]));
        for name in FP_ENV {
            assert_ne!(env(&[(name, "x")]), base, "{name}");
        }
        // Set to empty is not unset.
        assert_ne!(env(&[("HOME", "")]), base);

        let key_s = key.to_str().unwrap();
        let before = env(&[("AXIL_ENC_KEY_FILE", key_s)]);
        std::fs::write(&key, b"key two").unwrap();
        assert_ne!(env(&[("AXIL_ENC_KEY_FILE", key_s)]), before);
    }

    /// Variables the code reads that the fingerprint leaves out, and why a
    /// served command can still print what the client's own run would.
    const EXEMPT: &[(&str, &str)] = &[
        ("AXIL_DB", "carried by each request (exec::CLI_ENV)"),
        ("AXIL_MCP_TOOLS", "carried by each request (exec::CLI_ENV)"),
        (
            "AXIL_SLOW_QUERY_LOG",
            "per request: the run header's slow_query_log",
        ),
        (
            "AXIL_DAEMON",
            "the kill switch, read where a daemon is started",
        ),
    ];

    /// Reads that name no variable at the call: the name comes from a
    /// list or a function, and the reason says whose they are.
    const INDIRECT: &[(&str, &str)] = &[
        (
            "crates/adapters/axil-cli/src/daemon/exec.rs",
            "the exec::CLI_ENV values, carried by each request",
        ),
        (
            "crates/adapters/axil-cli/src/hook_brain.rs",
            "a hook's project directory; hooks are never served",
        ),
        (
            "crates/adapters/axil-cli/src/daemon/proto.rs",
            "the FP_ENV and CLI_ENV values themselves",
        ),
    ];

    /// Every environment variable the code reads at run time, outside
    /// tests, is in [`FP_ENV`], or exempt with a reason above, or read
    /// only where nothing is served. A variable a served command reads
    /// that is in none of them would make it print what the daemon's
    /// environment says rather than the client's.
    #[test]
    fn every_variable_the_code_reads_is_fingerprinted_or_exempt() {
        use crate::cli_io::tests::{code_only, without_test_modules};

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let mut dirs = vec![
            root.join("crates/axil-core/src"),
            root.join("crates/axil-bundle/src"),
            root.join("crates/adapters/axil-cli/src"),
            root.join("crates/adapters/axil-runtime/src"),
        ];
        for group in ["crates/engines", "crates/extensions"] {
            for entry in std::fs::read_dir(root.join(group)).unwrap() {
                dirs.push(entry.unwrap().path().join("src"));
            }
        }
        let mut sources = Vec::new();
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let file = path
                        .strip_prefix(&root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    sources.push((file, std::fs::read_to_string(&path).unwrap()));
                }
            }
        }
        assert!(sources.len() > 100, "read only {} files", sources.len());

        // `const NAME: &str = "VALUE";`, for reads that name a constant.
        let mut consts = std::collections::HashMap::new();
        for (_, src) in &sources {
            for line in src.lines() {
                let line = line
                    .trim_start()
                    .trim_start_matches("pub ")
                    .trim_start_matches("pub(crate) ");
                let Some(rest) = line.strip_prefix("const ") else {
                    continue;
                };
                if let Some((name, value)) = rest.split_once(": &str = \"") {
                    if let Some(value) = value.strip_suffix("\";") {
                        consts.insert(name.to_string(), value.to_string());
                    }
                }
            }
        }

        let mut read = BTreeSet::new();
        let mut unexplained = Vec::new();
        for (file, src) in &sources {
            let mut code = code_only(src);
            without_test_modules(&mut code);
            let bytes = src.as_bytes();
            for needle in [
                "env::var(",
                "env::var_os(",
                "env::var)",
                "env::var_os)",
                "env::vars",
            ] {
                let n = needle.as_bytes();
                for pos in 0..code.len().saturating_sub(n.len()) {
                    if &code[pos..pos + n.len()] != n {
                        continue;
                    }
                    let line = src[..pos].matches('\n').count() + 1;
                    let mut i = pos + n.len();
                    if needle.ends_with('(') {
                        while bytes[i].is_ascii_whitespace() {
                            i += 1;
                        }
                        if bytes[i] == b'"' {
                            let end = i + 1 + src[i + 1..].find('"').unwrap();
                            read.insert(src[i + 1..end].to_string());
                            continue;
                        }
                        let ident: String = src[i..]
                            .chars()
                            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                            .collect();
                        let last = ident.rsplit("::").next().unwrap_or_default();
                        if let Some(value) = consts.get(last) {
                            read.insert(value.clone());
                            continue;
                        }
                    }
                    if !INDIRECT.iter().any(|(f, _)| f == file) {
                        unexplained.push(format!(
                            "{file}:{line}: {}",
                            src.lines().nth(line - 1).unwrap().trim()
                        ));
                    }
                }
            }
        }
        assert!(read.contains("HOME"), "the scan found nothing: {read:?}");
        assert!(
            unexplained.is_empty(),
            "reads that name no variable; add their file to INDIRECT with a reason:\n{}",
            unexplained.join("\n")
        );
        let covered = |name: &str| {
            FP_ENV.contains(&name)
                || EXEMPT.iter().any(|(n, _)| *n == name)
                || NOT_SERVED.iter().any(|(n, _)| *n == name)
        };
        let missing: Vec<&String> = read.iter().filter(|name| !covered(name)).collect();
        assert!(
            missing.is_empty(),
            "read but neither fingerprinted nor exempt: {missing:?}"
        );
    }

    /// Variables read only by code no served command runs.
    const NOT_SERVED: &[(&str, &str)] = &[
        (
            "AXIL_BENCH_COMMIT",
            "labels the indexer's benchmark reports",
        ),
        ("CLAUDE_PROJECT_DIR", "hook commands"),
        ("PATH", "install and hook setup look for an installed axil"),
        ("USER", "the scheduled maintenance's log directory"),
        ("SystemRoot", "Windows only, where there is no daemon"),
    ];

    #[test]
    fn the_build_id_follows_the_binary_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("axil");
        std::fs::write(&exe, b"one").unwrap();
        let first = ExeId::of(&exe).unwrap();
        assert_eq!(first.build_id(), ExeId::of(&exe).unwrap().build_id());
        std::fs::write(&exe, b"build two").unwrap();
        let second = ExeId::of(&exe).unwrap();
        assert_ne!(first, second);
        assert_ne!(first.build_id(), second.build_id());
    }
}
