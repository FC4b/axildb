//! The warm daemon: a long-lived process that keeps one database open and
//! runs the hot commands against it in-process, so a client neither opens the
//! database nor loads its engines on each call.
//!
//! - [`client`] sends a command line to the database's daemon: the CLI's
//!   hot commands and the hooks' lookups.
//! - [`exec`] runs one request against the open database.
//! - [`lease`] asks a database's daemon to let go of it, for a process
//!   about to open it.
//! - [`paths`] names the files a database's daemon keeps in `run/`.
//! - [`proto`] is the wire format and the identities a client and a daemon
//!   compare.
//! - [`state`] is what the daemon records about itself.
//! - [`server`] is the daemon process.
//! - [`control`] is `axil daemon {start, stop, status, logs}`.

pub mod client;
pub mod control;
pub mod exec;
pub mod lease;
pub mod paths;
pub mod proto;
pub mod server;
pub mod state;

// The host catches a request's panic and keeps serving.
#[cfg(not(panic = "unwind"))]
compile_error!("the daemon catches a request's panic: build with panic = \"unwind\"");
