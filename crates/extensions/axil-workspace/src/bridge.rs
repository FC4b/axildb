//! Boundary Dialect: assertions that a local canonical entity corresponds
//! to a remote one.
//!
//! Bridges only ever live in the *local* DB. The remote DB is never
//! modified — it stays authoritative over its own canonical IDs. The
//! bridges table is deliberately simple: row-per-assertion, no joins,
//! no constraints beyond key uniqueness.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::manifest::{MemberId, WorkspaceId};

/// redb-storable table name for bridges, kept here so axil-core can use
/// a single constant. Prefixed with underscore to hide from default
/// listings.
pub const BRIDGES_TABLE: &str = "_entity_bridges";

/// Who made the assertion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssertSource {
    Human,
    Scip,
    Llm,
    Heuristic,
}

/// What evidence supports a bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BridgeEvidence {
    /// Strongest — identical SCIP symbol across members.
    ScipSymbol { symbol: String },
    /// Shared URI (OpenAPI operationId, GraphQL type, …).
    SharedUri { uri: String },
    /// Weak — matching name + type. Stays below the default confidence
    /// threshold until a human confirms.
    NameAndType { name: String, type_name: String },
    /// `axil bridge` by hand.
    ManualAssert,
}

impl BridgeEvidence {
    /// Default confidence suggested for this evidence shape. Callers may
    /// override — humans asserting `NameAndType` can bump confidence high
    /// if they want, and `ScipSymbol` can be lowered on request.
    pub fn default_confidence(&self) -> f32 {
        match self {
            BridgeEvidence::ScipSymbol { .. } => 1.0,
            BridgeEvidence::SharedUri { .. } => 0.9,
            BridgeEvidence::ManualAssert => 0.8,
            BridgeEvidence::NameAndType { .. } => 0.45,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            BridgeEvidence::ScipSymbol { .. } => "scip_symbol",
            BridgeEvidence::SharedUri { .. } => "shared_uri",
            BridgeEvidence::NameAndType { .. } => "name_and_type",
            BridgeEvidence::ManualAssert => "manual_assert",
        }
    }
}

/// A single bridge row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityBridge {
    pub local_canonical: String,
    pub remote_workspace_id: WorkspaceId,
    pub remote_member_id: MemberId,
    pub remote_canonical: String,
    pub confidence: f32,
    pub evidence: BridgeEvidence,
    pub asserted_at: DateTime<Utc>,
    pub asserted_by: AssertSource,
    /// When `true`, the bridge was marked dangling by `bridges verify`
    /// (e.g. the remote canonical id no longer exists).
    #[serde(default)]
    pub dangling: bool,
}

impl EntityBridge {
    /// Stable redb key derived from the identity tuple.
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.local_canonical,
            self.remote_workspace_id,
            self.remote_member_id,
            self.remote_canonical
        )
    }

    pub fn new_manual(
        local_canonical: impl Into<String>,
        remote_workspace_id: impl Into<String>,
        remote_member_id: impl Into<String>,
        remote_canonical: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            local_canonical: local_canonical.into(),
            remote_workspace_id: remote_workspace_id.into(),
            remote_member_id: remote_member_id.into(),
            remote_canonical: remote_canonical.into(),
            confidence: confidence.clamp(0.0, 1.0),
            evidence: BridgeEvidence::ManualAssert,
            asserted_at: Utc::now(),
            asserted_by: AssertSource::Human,
            dangling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable() {
        let b = EntityBridge::new_manual("auth::login", "ws_x", "backend", "auth::login", 1.0);
        let k1 = b.key();
        let b2 = b.clone();
        assert_eq!(k1, b2.key());
    }

    #[test]
    fn name_and_type_confidence_stays_weak() {
        let ev = BridgeEvidence::NameAndType {
            name: "login".into(),
            type_name: "fn".into(),
        };
        assert!(ev.default_confidence() < 0.5);
    }
}
