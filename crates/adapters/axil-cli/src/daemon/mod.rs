//! The warm daemon: a long-lived process that keeps one database open and
//! runs the hot commands against it in-process, so a client neither opens the
//! database nor loads its engines on each call.
//!
//! [`exec`] runs one request against the open database.

// The socket server is the first caller outside tests.
#![cfg_attr(not(test), allow(dead_code))]

pub mod exec;

// The host catches a request's panic and keeps serving.
#[cfg(not(panic = "unwind"))]
compile_error!("the daemon catches a request's panic: build with panic = \"unwind\"");
