//! Workspace manifest loader and schema (`.axil-workspace.toml`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, WorkspaceError};

/// Filename of the primary workspace manifest. Checked into the repo.
pub const MANIFEST_FILENAME: &str = ".axil-workspace.toml";

/// Filename of the optional per-user overlay. Never committed.
pub const MANIFEST_OVERLAY_FILENAME: &str = ".axil-workspace.local.toml";

/// Opaque stable ID for a workspace. Names are labels; this is identity.
pub type WorkspaceId = String;

/// Stable member identifier within a workspace.
pub type MemberId = String;

/// Role identifier referenced by manifest + consent rules.
pub type RoleId = String;

/// Top-level `[workspace]` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSection {
    pub id: WorkspaceId,
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1".to_string()
}

/// A single member (a project DB) declared in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    pub id: MemberId,
    /// Project root, relative to the manifest file (e.g. `./frontend`).
    pub root: PathBuf,
    /// Path to the member's `.axil` database, relative to the manifest file.
    pub path: PathBuf,
    #[serde(default)]
    pub roles: Vec<RoleId>,
}

/// A role entry used by consent allowlists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub label: String,
}

/// Federation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Federation {
    pub default: FederationMode,
    /// Small local-result bonus applied to same-member results during fan-out.
    /// Tuned so local results still win on ties without drowning remotes.
    pub local_boost: f32,
    /// Minimum bridge confidence required for auto-traversal.
    pub min_bridge_confidence: f32,
}

impl Default for Federation {
    fn default() -> Self {
        Self {
            default: FederationMode::Federate,
            local_boost: default_local_boost(),
            min_bridge_confidence: default_bridge_confidence(),
        }
    }
}

pub fn default_local_boost() -> f32 {
    0.05
}

pub fn default_bridge_confidence() -> f32 {
    0.85
}

/// How `recall --across` behaves when no explicit list is given.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FederationMode {
    /// Fan out to siblings, merge with provenance (default).
    Federate,
    /// Collapse results into the caller's member (future — not implemented).
    Consolidate,
    /// Never fan out even if `--across` is passed (error instead).
    Off,
}

impl Default for FederationMode {
    fn default() -> Self {
        FederationMode::Federate
    }
}

/// Parsed form of `.axil-workspace.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceManifest {
    pub workspace: WorkspaceSection,
    #[serde(default)]
    pub members: BTreeMap<MemberId, Member>,
    #[serde(default)]
    pub roles: BTreeMap<RoleId, Role>,
    #[serde(default)]
    pub federation: Federation,
    /// Absolute path to the manifest file, populated on load.
    #[serde(skip)]
    pub manifest_path: PathBuf,
}

impl WorkspaceManifest {
    /// Load a manifest from an absolute path to the `.toml` file.
    ///
    /// An adjacent `.axil-workspace.local.toml` is layered on top if present
    /// (keys from the overlay replace keys in the shared manifest).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| WorkspaceError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut manifest: WorkspaceManifest =
            toml::from_str(&text).map_err(|e| WorkspaceError::Parse {
                path: path.to_path_buf(),
                source: e,
            })?;
        manifest.manifest_path = path.to_path_buf();

        // Apply overlay if present.
        if let Some(dir) = path.parent() {
            let overlay_path = dir.join(MANIFEST_OVERLAY_FILENAME);
            if overlay_path.exists() {
                let overlay_text =
                    std::fs::read_to_string(&overlay_path).map_err(|e| WorkspaceError::Io {
                        path: overlay_path.clone(),
                        source: e,
                    })?;
                let overlay: WorkspaceOverlay =
                    toml::from_str(&overlay_text).map_err(|e| WorkspaceError::Parse {
                        path: overlay_path.clone(),
                        source: e,
                    })?;
                overlay.apply_to(&mut manifest);
            }
        }

        manifest.validate()?;
        Ok(manifest)
    }

    /// Basic sanity checks — duplicate member IDs, missing roles, etc.
    ///
    /// The TOML table key (e.g. `[members.frontend]`) is a human-friendly
    /// label used in CLI output and `--across frontend`. The `id =
    /// "mem_..."` field is the stable opaque identity that survives
    /// rename. They deliberately do not need to match.
    pub fn validate(&self) -> Result<()> {
        let mut seen_ids: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (label, member) in &self.members {
            if !seen_ids.insert(member.id.as_str()) {
                return Err(WorkspaceError::Invalid(format!(
                    "duplicate member id '{}' (label '{label}')",
                    member.id
                )));
            }
            for role in &member.roles {
                if !self.roles.contains_key(role) {
                    return Err(WorkspaceError::Invalid(format!(
                        "member '{label}' references unknown role '{role}'"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Directory containing the manifest — used to resolve relative member paths.
    pub fn root_dir(&self) -> &Path {
        self.manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
    }

    /// Absolute path to a member's project root.
    pub fn member_root_abs(&self, member: &Member) -> PathBuf {
        self.root_dir().join(&member.root)
    }

    /// Absolute path to a member's database file.
    pub fn member_db_abs(&self, member: &Member) -> PathBuf {
        self.root_dir().join(&member.path)
    }

    /// Get a member by id.
    pub fn member(&self, id: &str) -> Option<&Member> {
        self.members.get(id)
    }

    /// Iterate members in a stable (BTreeMap) order.
    pub fn members_sorted(&self) -> Vec<&Member> {
        self.members.values().collect()
    }

    /// Resolve a comma-separated `--across` argument to `(label, member)`
    /// pairs.
    ///
    /// `"*"` expands to all declared members. Names are matched first by
    /// TOML label, then by the opaque `id`. Unknown names are reported so
    /// the caller can decide whether to error or warn.
    pub fn resolve_members_arg(&self, arg: &str) -> (Vec<(&str, &Member)>, Vec<String>) {
        let trimmed = arg.trim();
        if trimmed == "*" {
            return (
                self.members.iter().map(|(k, v)| (k.as_str(), v)).collect(),
                Vec::new(),
            );
        }
        let mut matched = Vec::new();
        let mut unknown = Vec::new();
        for name in trimmed
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            if let Some((label, member)) = self.members.iter().find(|(l, _)| l.as_str() == name) {
                matched.push((label.as_str(), member));
                continue;
            }
            if let Some((label, member)) = self.members.iter().find(|(_, m)| m.id == name) {
                matched.push((label.as_str(), member));
                continue;
            }
            unknown.push(name.to_string());
        }
        (matched, unknown)
    }

    /// Serialize the manifest back to TOML (useful for `workspace init`).
    pub fn to_toml_string(&self) -> std::result::Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

/// Overlay schema — everything optional so users can override pieces
/// (e.g. point a single member at a checkout on a different drive).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct WorkspaceOverlay {
    members: BTreeMap<MemberId, MemberOverlay>,
    federation: Option<Federation>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct MemberOverlay {
    path: Option<PathBuf>,
    root: Option<PathBuf>,
    roles: Option<Vec<RoleId>>,
}

impl WorkspaceOverlay {
    fn apply_to(self, manifest: &mut WorkspaceManifest) {
        if let Some(fed) = self.federation {
            manifest.federation = fed;
        }
        for (id, overlay) in self.members {
            if let Some(member) = manifest.members.get_mut(&id) {
                if let Some(path) = overlay.path {
                    member.path = path;
                }
                if let Some(root) = overlay.root {
                    member.root = root;
                }
                if let Some(roles) = overlay.roles {
                    member.roles = roles;
                }
            }
        }
    }
}

/// Build a fresh workspace ID (ULID, prefixed for human readability).
pub fn new_workspace_id() -> String {
    format!("ws_{}", ulid::Ulid::new())
}

/// Build a fresh member ID.
pub fn new_member_id() -> String {
    format!("mem_{}", ulid::Ulid::new())
}

/// Build a fresh role ID.
pub fn new_role_id() -> String {
    format!("role_{}", ulid::Ulid::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_manifest(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join(MANIFEST_FILENAME);
        fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn load_basic_manifest() {
        let tmp = TempDir::new().unwrap();
        let body = r#"
[workspace]
id = "ws_01AAA"
name = "acme"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"
roles = ["role_ui"]

[roles.role_ui]
label = "ui"
"#;
        let path = write_manifest(tmp.path(), body);
        let manifest = WorkspaceManifest::load(&path).unwrap();
        assert_eq!(manifest.workspace.name, "acme");
        assert_eq!(manifest.members.len(), 1);
        assert!(manifest.members.contains_key("frontend"));
        assert_eq!(manifest.federation.default, FederationMode::Federate);
    }

    #[test]
    fn unknown_role_rejected() {
        let tmp = TempDir::new().unwrap();
        let body = r#"
[workspace]
id = "ws_x"
name = "x"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"
roles = ["role_missing"]
"#;
        let path = write_manifest(tmp.path(), body);
        let err = WorkspaceManifest::load(&path).unwrap_err();
        match err {
            WorkspaceError::Invalid(msg) => assert!(msg.contains("role_missing")),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn overlay_replaces_member_path() {
        let tmp = TempDir::new().unwrap();
        let body = r#"
[workspace]
id = "ws_x"
name = "x"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"
"#;
        write_manifest(tmp.path(), body);
        fs::write(
            tmp.path().join(MANIFEST_OVERLAY_FILENAME),
            r#"
[members.frontend]
path = "./other/memory.axil"
"#,
        )
        .unwrap();
        let manifest = WorkspaceManifest::load(tmp.path().join(MANIFEST_FILENAME)).unwrap();
        assert_eq!(
            manifest.members["frontend"].path,
            PathBuf::from("./other/memory.axil")
        );
    }

    #[test]
    fn star_expands_to_all_members() {
        let tmp = TempDir::new().unwrap();
        let body = r#"
[workspace]
id = "ws_x"
name = "x"

[members.a]
id = "mem_a"
root = "./a"
path = "./a/.axil/memory.axil"

[members.b]
id = "mem_b"
root = "./b"
path = "./b/.axil/memory.axil"
"#;
        let path = write_manifest(tmp.path(), body);
        let manifest = WorkspaceManifest::load(&path).unwrap();
        let (all, unknown) = manifest.resolve_members_arg("*");
        assert_eq!(all.len(), 2);
        assert!(unknown.is_empty());

        let (partial, unknown) = manifest.resolve_members_arg("a,nope");
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0].0, "a");
        assert_eq!(unknown, vec!["nope".to_string()]);

        // Match by opaque id too.
        let (by_id, _) = manifest.resolve_members_arg("mem_b");
        assert_eq!(by_id.len(), 1);
        assert_eq!(by_id[0].0, "b");
    }
}
