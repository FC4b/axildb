//! Central registry of Axil's built-in Extensions.
//!
//! Before this crate, every in-tree Extension had to be hand-registered in
//! three places — the CLI's `attach_detected_plugins`, the MCP server's
//! equivalent, and the workspace audit test. Adding an Extension meant editing
//! all three and remembering to keep them in sync.
//!
//! Now they all derive their Extension set from one place: [`builtin_extensions`].
//! Adding a built-in Extension is a single entry there (behind its Cargo
//! feature). The CLI builds its dynamic command surface from
//! [`builtin_extension_surfaces`], the MCP server from [`builtin_mcp_surfaces`],
//! and the host registers them all with [`register_builtin_extensions`].
//!
//! The `[extensions] disabled` config filter is applied centrally here, so a
//! disabled Extension never reaches `db.extensions()` regardless of which
//! adapter opened the database.
//!
//! # Why a separate crate?
//!
//! The registry must sit *above* the extension crates (they depend on
//! `axil-core`; the registry depends on them) yet *below* the adapters (which
//! depend on the registry). A standalone crate is the only place that layering
//! works without a cycle.
//!
//! # Scope
//!
//! This crate owns built-in **Extension** registration. Built-in **Engine**
//! attach (the `vector`/`graph`/`fts`/`timeseries` companion detection) is a
//! separate concern handled by the generic Engine lifecycle; it is intentionally
//! not centralized here yet.

use std::sync::Arc;

use axil_core::{AxilBuilder, AxilConfig, CliSurface, Extension, McpSurface};

/// Construct every in-tree Extension whose Cargo feature is enabled and whose
/// id is **not** listed in `[extensions] disabled`.
///
/// This is the single list to edit when adding a built-in Extension.
pub fn builtin_extensions(config: &AxilConfig) -> Vec<Arc<dyn Extension>> {
    #[allow(unused_mut)] // mutated only when at least one Extension feature is on
    let mut exts: Vec<Arc<dyn Extension>> = Vec::new();

    #[cfg(feature = "deps")]
    push_if_enabled(&mut exts, config, Arc::new(axil_docs::DocsExtension));

    #[cfg(feature = "checkpoint")]
    push_if_enabled(&mut exts, config, Arc::new(axil_checkpoint::CheckpointExtension));

    // `config` is unused when no Extension features are enabled (minimal build).
    let _ = config;
    exts
}

/// Push `ext` onto `exts` unless it is disabled in `[extensions] disabled`.
#[allow(dead_code)] // dead only in the zero-feature build
fn push_if_enabled(exts: &mut Vec<Arc<dyn Extension>>, config: &AxilConfig, ext: Arc<dyn Extension>) {
    if config.is_extension_disabled(ext.id()) {
        return;
    }
    exts.push(ext);
}

/// Register every enabled, non-disabled built-in Extension onto `builder`.
///
/// Replaces the former hand-wired `#[cfg(feature = "…")] builder.with_extension(…)`
/// list in each adapter. Uses [`AxilBuilder::with_extension_arc`], which still
/// enforces the disjoint-id / disjoint-prefix invariants.
pub fn register_builtin_extensions(mut builder: AxilBuilder, config: &AxilConfig) -> AxilBuilder {
    for ext in builtin_extensions(config) {
        builder = builder.with_extension_arc(ext);
    }
    builder
}

/// The CLI subcommand surfaces of every enabled built-in Extension.
///
/// Computed **without** a database handle, so a CLI Adapter can build its full
/// argument parser before opening the DB.
pub fn builtin_extension_surfaces(config: &AxilConfig) -> Vec<CliSurface> {
    builtin_extensions(config)
        .iter()
        .filter_map(|e| e.cli_commands())
        .collect()
}

/// The MCP tool surfaces of every enabled built-in Extension.
pub fn builtin_mcp_surfaces(config: &AxilConfig) -> Vec<McpSurface> {
    builtin_extensions(config)
        .iter()
        .filter_map(|e| e.mcp_tools())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_register_enabled_extensions() {
        let config = AxilConfig::default();
        let exts = builtin_extensions(&config);
        // Under `full`, both docs + checkpoint are present; under a trimmed
        // build, whichever features are on. Either way every id is unique.
        let mut ids: Vec<&str> = exts.iter().map(|e| e.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), exts.len(), "built-in extension ids must be unique");
    }

    #[test]
    fn disabled_filter_drops_extension() {
        let all = builtin_extensions(&AxilConfig::default());
        if let Some(first) = all.first() {
            let target = first.id().to_string();
            let mut config = AxilConfig::default();
            config.extensions.disabled.push(target.clone());
            let filtered = builtin_extensions(&config);
            assert!(
                filtered.iter().all(|e| e.id() != target),
                "disabled extension `{target}` must not be registered",
            );
            assert_eq!(filtered.len(), all.len() - 1);
        }
    }

    #[test]
    fn register_builds_a_database() {
        // The registry must produce a builder that actually opens.
        let dir = tempfile::tempdir().unwrap();
        let builder = axil_core::Axil::open(dir.path().join("test.axil"));
        let builder = register_builtin_extensions(builder, &AxilConfig::default());
        let db = builder.build().unwrap();
        // Every built-in surfaces through db.extensions() identically to native.
        assert_eq!(db.extensions().len(), builtin_extensions(&AxilConfig::default()).len());
    }
}
