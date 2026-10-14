//! The warm daemon: a long-lived process that keeps one database open and
//! runs the hot commands against it in-process, so a client neither opens the
//! database nor loads its engines on each call.
//!
//! - [`exec`] runs one request against the open database.
//! - [`lease`] asks a database's daemon to let go of it, for a process
//!   about to open it.
//! - [`paths`] names the files a database's daemon keeps in `run/`.
//! - [`proto`] is the wire format and the identities a client and a daemon
//!   compare.
//! - [`state`] is what the daemon records about itself.
//! - [`server`] is the daemon process.
//! - [`control`] is `axil daemon {start, stop, status, logs}`.

// Clients that forward to the daemon come later; until then some of the
// protocol is used only by tests.
#![cfg_attr(not(test), allow(dead_code))]

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
