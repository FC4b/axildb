//! Where the engines' diagnostic lines go.
//!
//! Warnings an engine prints while it serves a command (a file it skipped, a
//! corrupt row it removed, a config file it could not read) are written with
//! [`diagln!`](crate::diagln) instead of `eprintln!`. With no sink installed
//! the two are the same: the line goes to the process's stderr. A host that
//! runs commands in-process installs a sink with [`set_diagnostic_sink`], so a
//! line can reach the command that caused it rather than the host's own
//! stderr.

use std::fmt;
use std::sync::OnceLock;

/// Receives one diagnostic line, without its trailing newline.
pub type DiagnosticSink = fn(fmt::Arguments<'_>);

static SINK: OnceLock<DiagnosticSink> = OnceLock::new();

/// Send every diagnostic line from now on to `sink`. A process sets it once:
/// this returns `false`, and changes nothing, when a sink is already set.
pub fn set_diagnostic_sink(sink: DiagnosticSink) -> bool {
    SINK.set(sink).is_ok()
}

/// Backs [`diagln!`](crate::diagln).
#[doc(hidden)]
pub fn emit(args: fmt::Arguments<'_>) {
    match SINK.get() {
        Some(sink) => sink(args),
        None => eprintln!("{args}"),
    }
}

/// `eprintln!` through the diagnostic sink (see [`crate::diag`]).
#[macro_export]
macro_rules! diagln {
    ($($arg:tt)*) => {
        $crate::diag::emit(format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static CAUGHT: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }

    /// Collects the lines of a thread that asked for them; every other
    /// thread's lines go to stderr, as they would with no sink.
    fn collect(args: fmt::Arguments<'_>) {
        let line = args.to_string();
        let kept = CAUGHT.with(|c| {
            c.borrow_mut()
                .as_mut()
                .map(|lines| lines.push(line.clone()))
        });
        if kept.is_none() {
            eprintln!("{line}");
        }
    }

    #[test]
    fn the_sink_gets_each_line_and_is_set_once() {
        assert!(set_diagnostic_sink(collect));
        assert!(!set_diagnostic_sink(|_| panic!("replaced the sink")));
        CAUGHT.with(|c| *c.borrow_mut() = Some(Vec::new()));
        crate::diagln!("skipped {} file(s)", 2);
        crate::diagln!("axil: warning: {}", "bad config");
        let lines = CAUGHT.with(|c| c.borrow_mut().take()).unwrap();
        assert_eq!(lines, ["skipped 2 file(s)", "axil: warning: bad config"]);
    }
}
