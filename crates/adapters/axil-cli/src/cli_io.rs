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

use std::cell::RefCell;
use std::fmt;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

thread_local! {
    static REQ: RefCell<Option<ReqCtx>> = const { RefCell::new(None) };
}

/// Set once a write finds the process's stdout closed by its reader.
static STDOUT_CLOSED: AtomicBool = AtomicBool::new(false);
/// Set once a write finds the process's stderr closed by its reader.
static STDERR_CLOSED: AtomicBool = AtomicBool::new(false);

/// I/O state for one command run in-process.
pub struct ReqCtx {
    /// Bytes the command wrote to stdout.
    pub out: Vec<u8>,
    /// Bytes the command wrote to stderr.
    pub err: Vec<u8>,
    /// Piped stdin. `None` stands for a terminal: nothing was piped.
    pub stdin: Option<Vec<u8>>,
    /// Whether the command records slow queries in the database's log.
    pub slow_query_log: bool,
    /// An open database the command may use instead of opening its own.
    #[allow(dead_code)] // set by an in-process host; no command reads it yet
    pub warm: Option<Arc<axil_core::Axil>>,
    /// Canonical path of the database `warm` was opened from.
    #[allow(dead_code)] // pairs with `warm`
    pub db_canon: PathBuf,
}

impl Default for ReqCtx {
    fn default() -> Self {
        Self {
            out: Vec::new(),
            err: Vec::new(),
            stdin: None,
            slow_query_log: true,
            warm: None,
            db_canon: PathBuf::new(),
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
    (ctx.out, ctx.err, result)
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
    fn write_args(w: &mut impl Write, args: fmt::Arguments<'_>, newline: bool) -> io::Result<()> {
        if newline {
            writeln!(w, "{args}")
        } else {
            write!(w, "{args}")
        }
    }

    let (closed, label) = if to_err {
        (&STDERR_CLOSED, "stderr")
    } else {
        (&STDOUT_CLOSED, "stdout")
    };
    if closed.load(Ordering::Relaxed) {
        return;
    }
    let written = if to_err {
        write_args(&mut io::stderr().lock(), args, newline)
    } else {
        write_args(&mut io::stdout().lock(), args, newline)
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
    let mut buf = String::new();
    match take_request_stdin(limit) {
        Some(bytes) => bytes.as_slice().read_to_string(&mut buf)?,
        None => io::stdin().take(limit).read_to_string(&mut buf)?,
    };
    Ok(buf)
}

/// Stdin as a buffered reader, for input consumed as a stream.
pub fn stdin_reader() -> Box<dyn io::BufRead> {
    match take_request_stdin(u64::MAX) {
        Some(bytes) => Box::new(io::Cursor::new(bytes)),
        None => Box::new(io::stdin().lock()),
    }
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

#[cfg(test)]
mod tests {
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

    /// `src` with comments and the contents of string and char literals
    /// replaced by spaces. Newlines are kept, so byte offsets still map to
    /// the original lines.
    fn code_only(src: &str) -> Vec<u8> {
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
    fn without_test_modules(code: &mut [u8]) {
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
                if line.starts_with("mod ") {
                    test_mod_start = Some(attr);
                }
            }
            offset = end;
        }
    }

    #[test]
    fn main_rs_reaches_stdio_only_through_cli_io() {
        let src = include_str!("main.rs");
        let mut code = code_only(src);
        without_test_modules(&mut code);
        let line_of = |pos: usize| src[..pos].matches('\n').count() + 1;
        let mut found = Vec::new();
        let banned = [
            "println!",
            "print!",
            "eprintln!",
            "eprint!",
            "io::stdout()",
            "io::stderr()",
            "io::stdin()",
        ];
        for needle in banned {
            let n = needle.as_bytes();
            for (pos, w) in code.windows(n.len()).enumerate() {
                // `eprintln!` holds `println!`, and `cli_io::stdout()` holds
                // `io::stdout()`: only a match that starts a token counts.
                let starts_token =
                    pos == 0 || !(code[pos - 1].is_ascii_alphanumeric() || code[pos - 1] == b'_');
                if w == n && starts_token {
                    found.push(format!("main.rs:{} {needle}", line_of(pos)));
                }
            }
        }
        assert!(
            found.is_empty(),
            "use the cli_io macros and helpers instead:\n{}",
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
    mod parity {
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

        fn argv(args: &[&str]) -> Vec<String> {
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

        fn one_shot(args: &[&str], stdin: Option<&[u8]>) -> (Vec<u8>, Vec<u8>, i32) {
            let mut child = one_shot_command(args)
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

        fn copy_dir(from: &Path, to: &Path) {
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
                let ch = text[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
            out
        }

        struct Case {
            name: &'static str,
            args: Vec<String>,
            stdin: Option<&'static [u8]>,
            /// Writes a record or stamps the time, so ids and timestamps
            /// are compared normalized.
            volatile: bool,
        }

        fn case(name: &'static str, args: &[&str]) -> Case {
            Case {
                name,
                args: args.iter().map(|a| a.to_string()).collect(),
                stdin: None,
                volatile: false,
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
        }

        /// The fixture project and its database, built through the CLI
        /// itself. Returns the project directory.
        fn build_fixture(root: &Path) -> std::path::PathBuf {
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

        #[test]
        fn captured_output_matches_one_shot() {
            let tmp = tempfile::tempdir().unwrap();
            let fixture = build_fixture(&tmp.path().join("fixture"));
            let work = tmp.path().join("work");
            let proj = work.join("proj");
            let db = proj.join(".axil/memory.axil");
            let db = db.to_str().unwrap();
            let auth_rs = proj.join("src/auth.rs");
            let auth_rs = auth_rs.to_str().unwrap();
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
                ),
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

            let mut failures = Vec::new();
            let mut results = Vec::new();
            for c in &cases {
                let args: Vec<&str> = c.args.iter().map(String::as_str).collect();
                let run_both = || {
                    let fresh = || {
                        let _ = std::fs::remove_dir_all(&work);
                        copy_dir(&fixture, &proj);
                    };
                    fresh();
                    let captured = in_process(&args, c.stdin);
                    fresh();
                    (captured, one_shot(&args, c.stdin))
                };
                let same = |captured: &(Vec<u8>, Vec<u8>, i32), real: &(Vec<u8>, Vec<u8>, i32)| {
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
