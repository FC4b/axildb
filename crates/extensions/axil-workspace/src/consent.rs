//! Record-level consent scopes.
//!
//! `read_consent` + `write_consent` are stored on each record so fan-out
//! queries can filter deterministically at the remote. Defaults are
//! `Private` / `SourceOnly` — a new workspace never leaks a single record
//! until the user promotes consent explicitly.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::manifest::{MemberId, RoleId, WorkspaceId};

/// Who is allowed to *read* a record across workspace boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadConsent {
    /// Default. Only visible in the DB that owns it.
    Private,
    /// Any sibling in the same workspace.
    Workspace,
    /// Explicit member allowlist.
    Members { members: Vec<MemberId> },
    /// Role allowlist, resolved against the manifest.
    Roles { roles: Vec<RoleId> },
    /// Unrestricted (e.g. OSS docs).
    Public,
}

impl Default for ReadConsent {
    fn default() -> Self {
        ReadConsent::Private
    }
}

/// Who is allowed to *attach* a follow-up record to this one.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WriteConsent {
    /// Default. Follow-up records must land in the source DB.
    SourceOnly,
    /// Any sibling in this workspace may write follow-ups.
    Workspace,
    Members {
        members: Vec<MemberId>,
    },
    Roles {
        roles: Vec<RoleId>,
    },
}

impl Default for WriteConsent {
    fn default() -> Self {
        WriteConsent::SourceOnly
    }
}

/// Context needed to decide whether a caller satisfies a consent rule.
#[derive(Debug, Clone)]
pub struct MatchContext<'a> {
    /// Workspace id of the record being checked.
    pub source_workspace: &'a WorkspaceId,
    /// Member id of the record being checked.
    pub source_member: &'a MemberId,
    /// Caller's workspace id.
    pub caller_workspace: &'a WorkspaceId,
    /// Caller's member id.
    pub caller_member: &'a MemberId,
    /// Caller's roles (resolved from the manifest).
    pub caller_roles: &'a [RoleId],
    /// If `true`, `Workspace`-scoped records are dropped (caller asked
    /// for `--strict-consent`).
    pub strict: bool,
}

/// Errors raised when a write violates `write_consent`.
#[derive(Debug, thiserror::Error)]
pub enum ConsentError {
    #[error(
        "record {record_id} is write_consent=source_only; follow-up must land in {source_member}"
    )]
    SourceOnlyViolation {
        record_id: String,
        source_member: String,
    },
    #[error("record {record_id} is write_consent=members; caller {caller_member} is not in the allowlist")]
    MemberNotAllowed {
        record_id: String,
        caller_member: String,
    },
    #[error("record {record_id} is write_consent=roles; caller has no matching role")]
    RoleNotAllowed { record_id: String },
}

impl ReadConsent {
    /// Returns `true` when `ctx.caller_*` is allowed to read a record
    /// tagged with this consent.
    pub fn allows(&self, ctx: &MatchContext<'_>) -> bool {
        let same_db =
            ctx.caller_workspace == ctx.source_workspace && ctx.caller_member == ctx.source_member;
        if same_db {
            // Local reads always pass — consent is a cross-boundary concept.
            return true;
        }
        match self {
            ReadConsent::Private => false,
            ReadConsent::Workspace => !ctx.strict && ctx.caller_workspace == ctx.source_workspace,
            ReadConsent::Members { members } => members.iter().any(|m| m == ctx.caller_member),
            ReadConsent::Roles { roles } => {
                let caller_roles: BTreeSet<&RoleId> = ctx.caller_roles.iter().collect();
                roles.iter().any(|r| caller_roles.contains(r))
            }
            ReadConsent::Public => true,
        }
    }

    /// Short label used in trace output.
    pub fn label(&self) -> &'static str {
        match self {
            ReadConsent::Private => "private",
            ReadConsent::Workspace => "workspace",
            ReadConsent::Members { .. } => "members",
            ReadConsent::Roles { .. } => "roles",
            ReadConsent::Public => "public",
        }
    }
}

impl WriteConsent {
    /// Check whether a write-back from `ctx.caller_*` is permitted on a
    /// record tagged with this consent. Returns `Ok(())` when allowed,
    /// `Err` carrying the offending record id.
    pub fn check(&self, record_id: &str, ctx: &MatchContext<'_>) -> Result<(), ConsentError> {
        let same_db =
            ctx.caller_workspace == ctx.source_workspace && ctx.caller_member == ctx.source_member;
        if same_db {
            return Ok(());
        }
        match self {
            WriteConsent::SourceOnly => Err(ConsentError::SourceOnlyViolation {
                record_id: record_id.to_string(),
                source_member: ctx.source_member.to_string(),
            }),
            WriteConsent::Workspace => {
                if ctx.caller_workspace == ctx.source_workspace {
                    Ok(())
                } else {
                    Err(ConsentError::MemberNotAllowed {
                        record_id: record_id.to_string(),
                        caller_member: ctx.caller_member.to_string(),
                    })
                }
            }
            WriteConsent::Members { members } => {
                if members.iter().any(|m| m == ctx.caller_member) {
                    Ok(())
                } else {
                    Err(ConsentError::MemberNotAllowed {
                        record_id: record_id.to_string(),
                        caller_member: ctx.caller_member.to_string(),
                    })
                }
            }
            WriteConsent::Roles { roles } => {
                let caller: BTreeSet<&RoleId> = ctx.caller_roles.iter().collect();
                if roles.iter().any(|r| caller.contains(r)) {
                    Ok(())
                } else {
                    Err(ConsentError::RoleNotAllowed {
                        record_id: record_id.to_string(),
                    })
                }
            }
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            WriteConsent::SourceOnly => "source_only",
            WriteConsent::Workspace => "workspace",
            WriteConsent::Members { .. } => "members",
            WriteConsent::Roles { .. } => "roles",
        }
    }
}

/// Parse a short CLI-friendly string (`private`, `workspace`, `public`,
/// `members:a,b`, `roles:role_ui,role_api`) into a `ReadConsent`.
pub fn parse_read_consent(raw: &str) -> Result<ReadConsent, String> {
    let raw = raw.trim();
    if let Some(list) = raw.strip_prefix("members:") {
        return Ok(ReadConsent::Members {
            members: list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        });
    }
    if let Some(list) = raw.strip_prefix("roles:") {
        return Ok(ReadConsent::Roles {
            roles: list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        });
    }
    match raw {
        "private" => Ok(ReadConsent::Private),
        "workspace" => Ok(ReadConsent::Workspace),
        "public" => Ok(ReadConsent::Public),
        other => Err(format!(
            "unknown read-consent scope '{other}'. expected: private | workspace | public | members:<csv> | roles:<csv>"
        )),
    }
}

/// Parse a short CLI-friendly string for write consent.
pub fn parse_write_consent(raw: &str) -> Result<WriteConsent, String> {
    let raw = raw.trim();
    if let Some(list) = raw.strip_prefix("members:") {
        return Ok(WriteConsent::Members {
            members: list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        });
    }
    if let Some(list) = raw.strip_prefix("roles:") {
        return Ok(WriteConsent::Roles {
            roles: list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        });
    }
    match raw {
        "source-only" | "source_only" => Ok(WriteConsent::SourceOnly),
        "workspace" => Ok(WriteConsent::Workspace),
        other => Err(format!(
            "unknown write-consent scope '{other}'. expected: source-only | workspace | members:<csv> | roles:<csv>"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(
        source_ws: &'a WorkspaceId,
        source_member: &'a MemberId,
        caller_ws: &'a WorkspaceId,
        caller_member: &'a MemberId,
        roles: &'a [RoleId],
    ) -> MatchContext<'a> {
        MatchContext {
            source_workspace: source_ws,
            source_member,
            caller_workspace: caller_ws,
            caller_member,
            caller_roles: roles,
            strict: false,
        }
    }

    #[test]
    fn private_blocks_cross_member_read() {
        let ws = "ws_x".to_string();
        let a = "mem_a".to_string();
        let b = "mem_b".to_string();
        let roles: Vec<RoleId> = vec![];
        let ctx = ctx(&ws, &a, &ws, &b, &roles);
        assert!(!ReadConsent::Private.allows(&ctx));
    }

    #[test]
    fn private_allows_same_db_read() {
        let ws = "ws_x".to_string();
        let a = "mem_a".to_string();
        let roles: Vec<RoleId> = vec![];
        let ctx = ctx(&ws, &a, &ws, &a, &roles);
        assert!(ReadConsent::Private.allows(&ctx));
    }

    #[test]
    fn workspace_allows_sibling_read_but_strict_blocks() {
        let ws = "ws_x".to_string();
        let a = "mem_a".to_string();
        let b = "mem_b".to_string();
        let roles: Vec<RoleId> = vec![];
        let mut c = ctx(&ws, &a, &ws, &b, &roles);
        assert!(ReadConsent::Workspace.allows(&c));
        c.strict = true;
        assert!(!ReadConsent::Workspace.allows(&c));
    }

    #[test]
    fn roles_consent() {
        let ws = "ws_x".to_string();
        let a = "mem_a".to_string();
        let b = "mem_b".to_string();
        let caller_roles = vec!["role_ui".to_string()];
        let c = ctx(&ws, &a, &ws, &b, &caller_roles);
        let consent = ReadConsent::Roles {
            roles: vec!["role_ui".into(), "role_api".into()],
        };
        assert!(consent.allows(&c));
        let consent_miss = ReadConsent::Roles {
            roles: vec!["role_admin".into()],
        };
        assert!(!consent_miss.allows(&c));
    }

    #[test]
    fn source_only_rejects_remote_writes() {
        let ws = "ws_x".to_string();
        let a = "mem_a".to_string();
        let b = "mem_b".to_string();
        let roles: Vec<RoleId> = vec![];
        let c = ctx(&ws, &a, &ws, &b, &roles);
        let res = WriteConsent::SourceOnly.check("rec_1", &c);
        assert!(res.is_err());
    }

    #[test]
    fn parse_read_consent_variants() {
        assert!(matches!(
            parse_read_consent("private").unwrap(),
            ReadConsent::Private
        ));
        assert!(matches!(
            parse_read_consent("workspace").unwrap(),
            ReadConsent::Workspace
        ));
        assert!(matches!(
            parse_read_consent("public").unwrap(),
            ReadConsent::Public
        ));
        match parse_read_consent("members:a,b").unwrap() {
            ReadConsent::Members { members } => assert_eq!(members, vec!["a", "b"]),
            _ => panic!(),
        }
        match parse_read_consent("roles:role_ui").unwrap() {
            ReadConsent::Roles { roles } => assert_eq!(roles, vec!["role_ui"]),
            _ => panic!(),
        }
        assert!(parse_read_consent("nope").is_err());
    }
}
