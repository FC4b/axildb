//! Capturable stdio for CLI commands.
//!
//! Every byte a command prints and every byte it reads from stdin goes
//! through this module, so a command can run in-process with its output
//! collected into buffers instead of landing on the process's fds.
//!
//! A one-shot `axil` run installs no request context. The print macros then
//! hand their arguments to the std `print!` / `eprint!` macros, so the bytes,
//! the line buffering, and the failure mode are exactly a bare `println!`'s:
//! the binary installs no SIGPIPE handler, so writing to a closed pipe panics
//! with "failed printing to stdout: Broken pipe", and `main` turns the
//! panicked thread into exit code 1.
//!
//! [`capture`] installs a [`ReqCtx`] on the current thread for the duration
//! of a closure. Output then appends to the context's buffers and stdin is
//! served from its bytes.

use std::cell::RefCell;
use std::fmt;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

thread_local! {
    static REQ: RefCell<Option<ReqCtx>> = const { RefCell::new(None) };
}

/// I/O state for one command run in-process.
pub struct ReqCtx {
    /// Bytes the command wrote to stdout.
    pub out: Vec<u8>,
    /// Bytes the command wrote to stderr.
    pub err: Vec<u8>,
    /// Piped stdin. `None` stands for a terminal: nothing was piped.
    pub stdin: Option<Vec<u8>>,
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

/// Backs [`outln!`] and [`outp!`].
#[doc(hidden)]
pub fn out_fmt(args: fmt::Arguments<'_>, newline: bool) {
    if !append_fmt(false, args, newline) {
        if newline {
            std::println!("{args}");
        } else {
            std::print!("{args}");
        }
    }
}

/// Backs [`errln!`] and [`errp!`].
#[doc(hidden)]
pub fn err_fmt(args: fmt::Arguments<'_>, newline: bool) {
    if !append_fmt(true, args, newline) {
        if newline {
            std::eprintln!("{args}");
        } else {
            std::eprint!("{args}");
        }
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
/// the locked process stdout otherwise.
pub struct Stdout(Option<io::StdoutLock<'static>>);

/// The command's stdout as a [`Write`] handle; see [`Stdout`].
pub fn stdout() -> Stdout {
    if capturing() {
        Stdout(None)
    } else {
        Stdout(Some(io::stdout().lock()))
    }
}

impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match &mut self.0 {
            Some(lock) => lock.write(buf),
            None if append(false, buf) => Ok(buf.len()),
            None => io::stdout().write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.0 {
            Some(lock) => lock.flush(),
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
}
