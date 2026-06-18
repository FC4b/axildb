//! Global registry of known workspaces.
//!
//! Stored as a TOML file in the user's config directory. This lets
//! `axil workspace list` surface workspaces not rooted at `cwd` — e.g. after
//! you've opened `frontend/` in your editor but the manifest lives two
//! directories up and isn't in the current tree.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, WorkspaceError};

/// A single entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub id: String,
    pub name: String,
    pub manifest_path: PathBuf,
    /// Last time this workspace was interacted with. Useful for triage.
    #[serde(default)]
    pub last_seen: Option<chrono::DateTime<chrono::Utc>>,
}

/// Content of the global registry TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalRegistry {
    #[serde(default)]
    pub workspaces: Vec<RegistryEntry>,
}

impl GlobalRegistry {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(|e| WorkspaceError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let parsed: Self = toml::from_str(&text).map_err(|e| WorkspaceError::Parse {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(parsed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| WorkspaceError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let body = toml::to_string_pretty(self)
            .map_err(|e| WorkspaceError::Invalid(format!("registry serialize: {e}")))?;
        std::fs::write(path, body).map_err(|e| WorkspaceError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Insert-or-update by `id`. Returns `true` if this was a new entry.
    pub fn upsert(&mut self, entry: RegistryEntry) -> bool {
        if let Some(existing) = self.workspaces.iter_mut().find(|w| w.id == entry.id) {
            existing.name = entry.name;
            existing.manifest_path = entry.manifest_path;
            existing.last_seen = entry.last_seen.or(existing.last_seen);
            false
        } else {
            self.workspaces.push(entry);
            true
        }
    }

    /// Remove any entries whose manifest file no longer exists on disk.
    pub fn prune(&mut self) -> usize {
        let before = self.workspaces.len();
        self.workspaces.retain(|w| w.manifest_path.exists());
        before - self.workspaces.len()
    }
}

/// OS-aware path for the global registry.
///
/// - Linux:   `$XDG_CONFIG_HOME/axil/workspaces.toml`, falling back to `~/.config/axil/workspaces.toml`
/// - macOS:   `~/Library/Application Support/axil/workspaces.toml`
/// - Windows: `%APPDATA%/axil/workspaces.toml`
pub fn global_registry_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("axil").join("workspaces.toml");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = axil_core::home_dir() {
            return home
                .join("Library")
                .join("Application Support")
                .join("axil")
                .join("workspaces.toml");
        }
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("axil").join("workspaces.toml");
            }
        }
        if let Some(home) = axil_core::home_dir() {
            return home.join(".config").join("axil").join("workspaces.toml");
        }
    }
    PathBuf::from("axil-workspaces.toml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trip_registry() {
        let tmp = TempDir::new().unwrap();
        let registry_path = tmp.path().join("workspaces.toml");
        let mut reg = GlobalRegistry::default();
        reg.upsert(RegistryEntry {
            id: "ws_a".into(),
            name: "acme".into(),
            manifest_path: tmp.path().join(".axil-workspace.toml"),
            last_seen: None,
        });
        reg.save(&registry_path).unwrap();

        let reloaded = GlobalRegistry::load(&registry_path).unwrap();
        assert_eq!(reloaded.workspaces.len(), 1);
        assert_eq!(reloaded.workspaces[0].id, "ws_a");
    }

    #[test]
    fn upsert_merges_by_id() {
        let mut reg = GlobalRegistry::default();
        let entry1 = RegistryEntry {
            id: "ws_a".into(),
            name: "old".into(),
            manifest_path: PathBuf::from("/x"),
            last_seen: None,
        };
        let entry2 = RegistryEntry {
            id: "ws_a".into(),
            name: "new".into(),
            manifest_path: PathBuf::from("/y"),
            last_seen: None,
        };
        assert!(reg.upsert(entry1));
        assert!(!reg.upsert(entry2));
        assert_eq!(reg.workspaces.len(), 1);
        assert_eq!(reg.workspaces[0].name, "new");
    }
}
