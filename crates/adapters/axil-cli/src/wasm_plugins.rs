//! Runtime WASM plugin discovery + lifecycle (Phase 22.6). Behind `wasm-host`.
//!
//! Plugins are `.wasm` component files in `<db-dir>/plugins/`. They are loaded
//! at open and registered into the live `Axil` via `register_extension`, so they
//! flow through dispatch / dynamic CLI / boot exactly like native Extensions.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use axil_core::{Axil, AxilConfig, Extension};
use axil_runtime::{Capabilities, PluginState, WasmExtension, WasmHost};

/// Directory WASM plugins live in: `<db-dir>/plugins/`. The db is e.g.
/// `.axil/memory.axil`, so plugins are at `.axil/plugins/`.
pub fn plugins_dir(db_path: &Path) -> PathBuf {
    db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("plugins")
}

/// One plugin file's load outcome — for `axil ext list` and diagnostics.
pub struct PluginRecord {
    pub file: PathBuf,
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub prefixes: Vec<String>,
    /// `Some(reason)` if the plugin failed to load (quarantined, not fatal).
    pub error: Option<String>,
}

/// Scan `dir` for `*.wasm`, load each as a [`WasmExtension`], and register the
/// successful ones into `db`. A plugin that fails to load is **quarantined**
/// (reported, not fatal) — one bad `.wasm` never breaks open or the others
/// (Phase 22.4 fault isolation).
///
/// `db` must be the SAME database the plugins call back into via host imports.
/// Returns one record per `.wasm` file (sorted by filename for determinism).
pub fn load_and_register(db: &Arc<Axil>, dir: &Path, config: &AxilConfig) -> Vec<PluginRecord> {
    let mut files = match list_wasm_files(dir) {
        Ok(f) => f,
        Err(_) => return Vec::new(), // no plugins dir → nothing to load
    };
    files.sort();

    let host = match WasmHost::new() {
        Ok(h) => h,
        Err(e) => {
            return files
                .into_iter()
                .map(|file| PluginRecord {
                    file,
                    id: None,
                    display_name: None,
                    prefixes: Vec::new(),
                    error: Some(format!("WASM runtime init failed: {e}")),
                })
                .collect()
        }
    };

    files
        .into_iter()
        .map(|file| load_one(db, &host, &file, config, true))
        .collect()
}

/// Load a single plugin file, optionally registering it into `db`. Used by
/// `load_and_register` (register = true) and by `axil ext install`/`list`
/// (register = false — inspect only).
pub fn load_one(
    db: &Arc<Axil>,
    host: &WasmHost,
    file: &Path,
    config: &AxilConfig,
    register: bool,
) -> PluginRecord {
    let mut rec = PluginRecord {
        file: file.to_path_buf(),
        id: None,
        display_name: None,
        prefixes: Vec::new(),
        error: None,
    };

    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => {
            rec.error = Some(format!("read failed: {e}"));
            return rec;
        }
    };
    let component = match host.load_component(&bytes) {
        Ok(c) => c,
        Err(e) => {
            rec.error = Some(format!("not a valid component: {e}"));
            return rec;
        }
    };

    // Capabilities: grant-all for now — a per-plugin manifest with requested
    // capabilities + operator consent is Phase 22.5. Writable prefixes are the
    // plugin's own `table-prefixes()` declaration, set inside `WasmExtension::load`.
    let state = PluginState::new(
        Arc::clone(db),
        Capabilities::all(),
        Vec::new(),
        config.clone(),
    );
    let ext = match WasmExtension::load(host, &component, state) {
        Ok(e) => e,
        Err(e) => {
            rec.error = Some(format!("instantiation failed: {e}"));
            return rec;
        }
    };

    rec.id = Some(ext.id().to_string());
    rec.display_name = Some(ext.display_name().to_string());
    rec.prefixes = ext.table_prefixes().iter().map(|s| s.to_string()).collect();

    if register {
        if let Err(e) = db.register_extension(Arc::new(ext)) {
            // e.g. id/prefix clash with a native or another loaded plugin.
            rec.error = Some(format!("registration refused: {e}"));
        }
    }
    rec
}

/// Validate that `file` loads as a plugin and return its id — the gate for
/// `axil ext install` (don't install a `.wasm` that won't load).
pub fn inspect(db: &Arc<Axil>, file: &Path, config: &AxilConfig) -> Result<PluginRecord> {
    let host = WasmHost::new().context("WASM runtime init failed")?;
    let rec = load_one(db, &host, file, config, false);
    if let Some(err) = &rec.error {
        anyhow::bail!("{err}");
    }
    Ok(rec)
}

fn list_wasm_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("wasm") {
            out.push(path);
        }
    }
    Ok(out)
}
