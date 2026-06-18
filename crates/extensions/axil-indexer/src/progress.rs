//! Optional progress reporting for long-running indexer operations.
//!
//! The trait stays UI-agnostic so the indexer crate doesn't depend on
//! terminal-rendering libraries (indicatif, console, etc.). The CLI wires
//! up an indicatif-backed implementation; tests and library users can pass
//! `NoopProgress` (the default) or implement their own reporter.

/// Phases the indexer reports during a full or incremental run.
pub trait IndexProgress: Send + Sync {
    /// Called once at the start with the total number of files that will
    /// be visited. Implementations sized for determinate progress should
    /// initialize their bar here.
    fn start(&self, _total_files: usize) {}

    /// Called once per file as soon as it has been parsed and inserted.
    /// `idx` is 1-based to match human-readable progress (`3/10`).
    fn file_indexed(&self, _idx: usize, _path: &str) {}

    /// Marks the start of a non-file-loop phase (modules, symbols, proxies,
    /// dependencies, project overview). UIs typically swap the bar to a
    /// spinner with a label here.
    fn phase(&self, _name: &str) {}

    /// Called once when the run completes successfully.
    fn finish(&self) {}
}

/// No-op default. Library callers that don't need progress should use
/// this; the indexer's hot path becomes a series of dynamic-dispatch
/// no-ops which the optimizer collapses to nothing meaningful.
pub struct NoopProgress;
impl IndexProgress for NoopProgress {}
