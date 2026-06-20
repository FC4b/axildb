//! Multi-project workspace coordinator for Axil.
//!
//! A workspace links sibling `.axil` databases under a shared manifest so an
//! agent can query across projects without merging them. Per-project DBs stay
//! authoritative over their own records; the workspace crate only:
//!
//! - Loads `.axil-workspace.toml` (+ optional `.axil-workspace.local.toml`)
//! - Resolves which member the current working directory belongs to
//! - Provides a global registry under the user's config dir
//! - Offers a fan-out hook used by `recall --across` (see `axil-core`)
//!
//! The crate is intentionally small — it is a *control plane*, never a data
//! plane. It stores paths, names, and consent policy. Record content lives in
//! the member DBs.

pub mod bridge;
pub mod consent;
pub mod federation;
pub mod manifest;
pub mod registry;
pub mod resolve;

pub use bridge::{AssertSource, BridgeEvidence, EntityBridge, BRIDGES_TABLE};
pub use consent::{ConsentError, MatchContext, ReadConsent, WriteConsent};
pub use federation::{
    fan_out, FederatedResult, FederationRequest, MemberRecallBatch, MemberRecallRow,
};
pub use manifest::{
    default_bridge_confidence, default_local_boost, Federation, FederationMode, Member, MemberId,
    Role, RoleId, WorkspaceId, WorkspaceManifest, MANIFEST_FILENAME, MANIFEST_OVERLAY_FILENAME,
};
pub use registry::{global_registry_path, GlobalRegistry, RegistryEntry};
pub use resolve::{
    discover_manifest, resolve_member, unbound_status, MemberResolution, WorkspaceStatus,
};

/// Errors produced by the workspace crate.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    #[error("failed to read manifest {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse manifest {path}: {source}")]
    Parse {
        path: std::path::PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("workspace validation failed: {0}")]
    Invalid(String),
    #[error("consent violation: {0}")]
    Consent(#[from] ConsentError),
}

pub type Result<T> = std::result::Result<T, WorkspaceError>;
