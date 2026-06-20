//! Runtime WASM plugin discovery + lifecycle. Behind `wasm-host`.
//!
//! Plugins are `.wasm` component files in `<db-dir>/plugins/`. They are loaded
//! at open and registered into the live `Axil` via `register_extension`, so they
//! flow through dispatch / dynamic CLI / boot exactly like native Extensions.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use axil_core::{Axil, AxilConfig, Extension};
use axil_runtime::{Capabilities, PluginState, WasmExtension, WasmHost};

/// The capability-grant + cache key for a plugin file: the full filename with
/// only the trailing `.wasm` stripped, interior dots replaced by `-` so it stays
/// a single TOML bare key under `[plugins.<key>]`.
///
/// Using the *full* stem (not just up to the first `.`) avoids collisions: two
/// distinct plugins sharing a prefix before the first dot — `acme.alpha.wasm`
/// and `acme.beta.wasm` — keyed identically under the old rule, so they shared
/// grants and the same `<key>.cwasm` cache artifact. Now they key as
/// `acme-alpha` and `acme-beta`.
pub fn plugin_key(file: &Path) -> String {
    file.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .replace('.', "-")
}

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
    /// Capability-grant key — the `.wasm` filename stem (e.g. `hello`).
    pub key: String,
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub prefixes: Vec<String>,
    /// `axil:plugin` ABI version the plugin was built against (e.g. `"1.0.0"`).
    pub abi: Option<String>,
    /// Capabilities granted to this plugin (deny-by-default; empty = none).
    pub granted: Vec<String>,
    /// `Some(reason)` if the plugin failed to load (quarantined, not fatal).
    pub error: Option<String>,
}

/// Scan `dir` for `*.wasm`, load each as a [`WasmExtension`], and register the
/// successful ones into `db`. A plugin that fails to load is **quarantined**
/// (reported, not fatal) — one bad `.wasm` never breaks open or the others
///.
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
                    key: plugin_key(&file),
                    file,
                    id: None,
                    display_name: None,
                    prefixes: Vec::new(),
                    abi: None,
                    granted: Vec::new(),
                    error: Some(format!("WASM runtime init failed: {e}")),
                })
                .collect()
        }
    };

    // Compiled-module cache lives beside the plugins (`.axil/plugins/.cache/`),
    // one `.cwasm` per plugin keyed on source content. It turns each subsequent
    // `axil <plugin-cmd>` from a full Cranelift compile into a deserialize.
    let cache_dir = dir.join(".cache");
    files
        .into_iter()
        .map(|file| load_one(db, &host, &file, config, true, Some(&cache_dir)))
        .collect()
}

/// Load a single plugin file, optionally registering it into `db`. Used by
/// `load_and_register` (register = true) and by `axil ext install`/`list`
/// (register = false — inspect only).
///
/// `cache_dir` enables the compiled-module cache: `Some(dir)` looks
/// up / writes `<dir>/<key>.cwasm` instead of recompiling; `None` always
/// compiles (the one-shot install/inspect gate, where caching buys nothing).
pub fn load_one(
    db: &Arc<Axil>,
    host: &WasmHost,
    file: &Path,
    config: &AxilConfig,
    register: bool,
    cache_dir: Option<&Path>,
) -> PluginRecord {
    // Capability grants are keyed by the `.wasm` filename stem, resolved from
    // config BEFORE load (the host ABI reads the granted set on the first host
    // call). Deny-by-default: an unconfigured plugin gets nothing.
    let key = plugin_key(file);
    let granted = config.plugin_capabilities(&key);

    let mut rec = PluginRecord {
        file: file.to_path_buf(),
        key: key.clone(),
        id: None,
        display_name: None,
        prefixes: Vec::new(),
        abi: None,
        granted: granted.clone(),
        error: None,
    };

    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => {
            rec.error = Some(format!("read failed: {e}"));
            return rec;
        }
    };
    let loaded = match cache_dir {
        Some(dir) => host.load_component_cached(&bytes, &dir.join(format!("{key}.cwasm"))),
        None => host.load_component(&bytes),
    };
    let component = match loaded {
        Ok(c) => c,
        Err(e) => {
            rec.error = Some(format!("not a valid component: {e}"));
            return rec;
        }
    };

    // Deny-by-default capabilities from `[plugins.<key>] capabilities`. Writable
    // prefixes are the plugin's own `table-prefixes()` declaration, set inside
    // `WasmExtension::load`.
    let state = PluginState::new(
        Arc::clone(db),
        Capabilities::from_names(&granted),
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
    rec.abi = Some(ext.abi().to_string());

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
    let rec = load_one(db, &host, file, config, false, None);
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
