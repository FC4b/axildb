//! `axil-sync` — the client seam for Atlas sync (Tier-2 Extension).
//!
//! Two responsibilities, both client-side and both usable without any server:
//! - [`select`] — pick the **distillate** (promoted decisions/rules/beliefs/
//!   errors/patterns; never the private episodic trail) out of a local `Axil`
//!   and turn it into a batch of protocol [`Op`]s, reusing Axil's existing
//!   importance signal.
//! - [`client`] (behind the default-off `http` feature) — [`SyncClient`],
//!   which speaks the Atlas sync protocol over HTTP.
//!
//! The wire types live in the separate public [`axil_atlas_proto`] crate so the
//! server and this client agree on the protocol without depending on each
//! other's code. **Nothing here depends on the Atlas server** — Atlas is
//! something you point a [`SyncClient`] at, never a hard dependency, and the
//! whole crate is meant to sit behind a default-off `sync` feature in
//! consumers so an OSS build has zero Atlas awareness.

pub mod select;

pub use axil_atlas_proto as proto;
pub use axil_atlas_proto::{
    BootstrapSnapshot, CompoundQuery, CompoundResult, Locator, Op, OpKind, PullQuery, PullResponse,
    PushBatch, PushResponse, Tier,
};
pub use select::{select_distillate, SelectOpts, DEFAULT_PROMOTED_TABLES};

#[cfg(feature = "http")]
pub mod client;
#[cfg(feature = "http")]
pub use client::SyncClient;

use thiserror::Error;

/// Errors surfaced by distillate selection or the sync client.
#[derive(Error, Debug)]
pub enum SyncError {
    /// An error from the local `axil-core` engine.
    #[error(transparent)]
    Axil(#[from] axil_core::AxilError),
    /// A (de)serialization error on a protocol payload.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// An HTTP transport error talking to Atlas.
    #[error("atlas transport: {0}")]
    Transport(String),
}
