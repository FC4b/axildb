//! Ancestor search + member resolution.
//!
//! A workspace is discovered by walking parents of the starting path. The
//! first `.axil-workspace.toml` wins — absence is a silent no-op so single-DB
//! users see zero change.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::manifest::{Member, MemberId, WorkspaceManifest, MANIFEST_FILENAME};
use crate::Result;

/// Outcome of asking "which member owns this path?".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberResolution {
    /// Human label (TOML table key, e.g. `frontend`).
    pub member_label: String,
    /// Stable opaque id (`mem_...`).
    pub member_id: MemberId,
    pub member_root: PathBuf,
    pub member_db_path: PathBuf,
}

/// Whole-workspace status: reachable members, current member (if any),
/// warnings about missing member paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceStatus {
    pub manifest_path: PathBuf,
    pub workspace_id: String,
    pub workspace_name: String,
    pub current_cwd: PathBuf,
    pub current_member: Option<MemberId>,
    pub members: Vec<MemberStatus>,
    /// Human-readable warnings about manifest entries that don't match
    /// the on-disk state (currently: members whose `path` doesn't exist).
    /// Empty in the common case so JSON consumers can ignore it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Reachability + filesystem facts for a single member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberStatus {
    pub label: String,
    pub id: MemberId,
    pub root: PathBuf,
    pub db_path: PathBuf,
    pub db_exists: bool,
    pub roles: Vec<String>,
}

/// Walk ancestors of `start` looking for `.axil-workspace.toml`. Returns
/// `Ok(None)` when no manifest is found (preserves solo-DB behavior).
pub fn discover_manifest(start: impl AsRef<Path>) -> Result<Option<WorkspaceManifest>> {
    let start = start.as_ref();
    let start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(start)
    };

    let mut cursor: Option<&Path> = Some(start.as_path());
    while let Some(dir) = cursor {
        let candidate = dir.join(MANIFEST_FILENAME);
        if candidate.is_file() {
            let manifest = WorkspaceManifest::load(&candidate)?;
            return Ok(Some(manifest));
        }
        cursor = dir.parent();
    }
    Ok(None)
}

/// Longest-prefix match of `cwd` against each member's root.
pub fn resolve_member<'m>(
    manifest: &'m WorkspaceManifest,
    cwd: impl AsRef<Path>,
) -> Option<MemberResolution> {
    let cwd = cwd.as_ref();
    let cwd_abs = match cwd.canonicalize() {
        Ok(p) => p,
        Err(_) => cwd.to_path_buf(),
    };

    let mut best: Option<(&str, &Member, usize)> = None;
    for (label, member) in &manifest.members {
        let root_abs = canonical_or_clean(&manifest.member_root_abs(member));
        if let Ok(root_abs) = root_abs {
            if cwd_abs.starts_with(&root_abs) {
                let depth = root_abs.components().count();
                match best {
                    None => best = Some((label.as_str(), member, depth)),
                    Some((_, _, best_depth)) if depth > best_depth => {
                        best = Some((label.as_str(), member, depth))
                    }
                    _ => {}
                }
            }
        }
    }

    best.map(|(label, m, _)| MemberResolution {
        member_label: label.to_string(),
        member_id: m.id.clone(),
        member_root: manifest.member_root_abs(m),
        member_db_path: manifest.member_db_abs(m),
    })
}

/// Summarize workspace state for `axil workspace status`.
pub fn unbound_status(manifest: &WorkspaceManifest, cwd: impl AsRef<Path>) -> WorkspaceStatus {
    let cwd = cwd.as_ref();
    let current_member = resolve_member(manifest, cwd).map(|r| r.member_id);
    let members: Vec<MemberStatus> = manifest
        .members
        .iter()
        .map(|(label, m)| MemberStatus {
            label: label.clone(),
            id: m.id.clone(),
            root: manifest.member_root_abs(m),
            db_path: manifest.member_db_abs(m),
            db_exists: manifest.member_db_abs(m).exists(),
            roles: m.roles.clone(),
        })
        .collect();

    // Surface any member whose DB doesn't exist on this machine.
    // Cross-member commands (recall-across, bridge auto) silently skip
    // these — without a warning here, the user has no signal that a
    // sibling they think is in the workspace is being routed around.
    // Matches the spec's Open Questions guidance: warn once, then treat
    // as unreachable.
    let warnings: Vec<String> = members
        .iter()
        .filter(|m| !m.db_exists)
        .map(|m| {
            format!(
                "member '{}' DB not found at {} — treating as unreachable",
                m.label,
                m.db_path.display(),
            )
        })
        .collect();

    WorkspaceStatus {
        manifest_path: manifest.manifest_path.clone(),
        workspace_id: manifest.workspace.id.clone(),
        workspace_name: manifest.workspace.name.clone(),
        current_cwd: cwd.to_path_buf(),
        current_member,
        members,
        warnings,
    }
}

fn canonical_or_clean(p: &Path) -> std::io::Result<PathBuf> {
    match p.canonicalize() {
        Ok(c) => Ok(c),
        Err(e) => {
            // On Windows/macOS a non-existent path fails to canonicalize.
            // Fall back to a cleaned absolute form so "prefix" checks still work.
            if p.is_absolute() {
                Ok(p.to_path_buf())
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn seed_workspace(dir: &Path) -> PathBuf {
        let body = format!(
            r#"
[workspace]
id = "ws_x"
name = "acme"

[members.frontend]
id = "mem_fe"
root = "./frontend"
path = "./frontend/.axil/memory.axil"

[members.backend]
id = "mem_be"
root = "./backend"
path = "./backend/.axil/memory.axil"
"#
        );
        let path = dir.join(MANIFEST_FILENAME);
        fs::write(&path, body).unwrap();
        fs::create_dir_all(dir.join("frontend")).unwrap();
        fs::create_dir_all(dir.join("backend")).unwrap();
        path
    }

    #[test]
    fn discovery_walks_to_parent() {
        let tmp = TempDir::new().unwrap();
        seed_workspace(tmp.path());
        let nested = tmp.path().join("frontend").join("src");
        fs::create_dir_all(&nested).unwrap();
        let manifest = discover_manifest(&nested).unwrap().unwrap();
        assert_eq!(manifest.workspace.name, "acme");
    }

    #[test]
    fn resolve_member_by_longest_prefix() {
        let tmp = TempDir::new().unwrap();
        seed_workspace(tmp.path());
        let nested = tmp.path().join("frontend").join("src");
        fs::create_dir_all(&nested).unwrap();
        let manifest = discover_manifest(tmp.path()).unwrap().unwrap();
        let resolution = resolve_member(&manifest, &nested).unwrap();
        assert_eq!(resolution.member_label, "frontend");
        assert_eq!(resolution.member_id, "mem_fe");
    }

    #[test]
    fn unbound_returns_none_outside_any_member() {
        let tmp = TempDir::new().unwrap();
        seed_workspace(tmp.path());
        let manifest = discover_manifest(tmp.path()).unwrap().unwrap();
        let resolution = resolve_member(&manifest, tmp.path());
        assert!(resolution.is_none(), "repo root should be unbound");
    }

    #[test]
    fn missing_manifest_is_silent() {
        let tmp = TempDir::new().unwrap();
        let manifest = discover_manifest(tmp.path()).unwrap();
        assert!(manifest.is_none());
    }

    #[test]
    fn unbound_status_warns_when_member_db_missing() {
        // seed_workspace creates frontend/ + backend/ dirs but never
        // their .axil/memory.axil files — both members are unreachable.
        let tmp = TempDir::new().unwrap();
        seed_workspace(tmp.path());
        let manifest = discover_manifest(tmp.path()).unwrap().unwrap();

        let status = unbound_status(&manifest, tmp.path());
        assert_eq!(
            status.warnings.len(),
            2,
            "expected one warning per missing member"
        );
        assert!(
            status
                .warnings
                .iter()
                .any(|w| w.contains("frontend") && w.contains("treating as unreachable")),
            "frontend warning shape: {:?}",
            status.warnings,
        );
        assert!(
            status.warnings.iter().any(|w| w.contains("backend")),
            "backend warning shape: {:?}",
            status.warnings,
        );
    }

    #[test]
    fn unbound_status_no_warnings_when_all_dbs_exist() {
        let tmp = TempDir::new().unwrap();
        seed_workspace(tmp.path());
        // Materialize both member DBs so the existence check passes.
        for member in &["frontend", "backend"] {
            let axil_dir = tmp.path().join(member).join(".axil");
            fs::create_dir_all(&axil_dir).unwrap();
            fs::write(axil_dir.join("memory.axil"), b"").unwrap();
        }

        let manifest = discover_manifest(tmp.path()).unwrap().unwrap();
        let status = unbound_status(&manifest, tmp.path());
        assert!(
            status.warnings.is_empty(),
            "no warnings when DBs present: {:?}",
            status.warnings
        );
    }
}
