//! WASM runtime for loading sandboxed Extension components against the
//! `axil:plugin` WIT world (see `wit/axil-plugin.wit`).
//!
//! Behind the `wasm-host` Cargo feature: **default builds carry zero Wasmtime
//! dependency**, preserving Axil's small-binary posture (Constraint C1 / Risk
//! R2 in the Phase 21–22 plan). This crate is the foundation for Phase 22.2+ —
//! it embeds the engine, enables the Component Model and a deny-by-default
//! resource posture (fuel + epoch interruption, no ambient WASI), and
//! compiles/validates `.wasm` **components**. The host-ABI implementation
//! (22.3) and the `WasmExtension` shim (22.4) build directly on `WasmHost`.

/// Generated host + guest bindings for the `axil:plugin@1.0.0` world, produced
/// from `wit/axil-plugin.wit` at compile time. The host-side imports trait this
/// generates is what Phase 22.3 implements against a live `Axil`; the guest
/// proxy is what the `WasmExtension` shim (22.4) calls across the boundary.
///
/// Generating these here is also the strongest validation of the WIT: it must
/// be not just syntactically valid (`wasm-tools`) but codegen-compatible.
#[cfg(feature = "wasm-host")]
#[allow(dead_code, clippy::all)] // generated; consumed by 22.3/22.4
pub mod bindings {
    wasmtime::component::bindgen!({
        path: "../../../wit",
        world: "plugin",
    });
}

#[cfg(feature = "wasm-host")]
mod host {
    use std::path::Path;

    use anyhow::Result;
    use sha2::{Digest, Sha256};
    use wasmtime::component::{Component, Linker};
    use wasmtime::{Config, Engine, Store};

    /// Default per-instance CPU budget (Wasmtime fuel units). Generous for a
    /// well-behaved plugin; a runaway guest exhausts it and traps. Phase
    /// 22.5/22.9 makes this configurable and refills it per call.
    const DEFAULT_FUEL: u64 = 10_000_000_000;

    /// Bytes of source-content SHA-256 prepended to a cached `.cwasm` artifact.
    /// The artifact is `[32-byte source hash][wasmtime-serialized component]`, so
    /// a content change self-invalidates the cache without a separate manifest.
    const CACHE_TAG_LEN: usize = 32;

    /// The WASM host runtime: owns a configured Wasmtime [`Engine`] shared by
    /// every loaded plugin.
    pub struct WasmHost {
        engine: Engine,
    }

    impl WasmHost {
        /// Build a host with the Component Model enabled and the sandbox
        /// primitives a misbehaving guest can be bounded with: CPU via fuel
        /// metering and wall-clock via epoch interruption (both wired to limits
        /// in Phase 22.5). No ambient filesystem/network is granted — WASI is
        /// off until a capability is explicitly added.
        pub fn new() -> Result<Self> {
            let mut config = Config::new();
            config.wasm_component_model(true);
            config.consume_fuel(true);
            config.epoch_interruption(true);
            let engine = Engine::new(&config)?;
            Ok(Self { engine })
        }

        /// The underlying engine (shared across loaded plugins).
        pub fn engine(&self) -> &Engine {
            &self.engine
        }

        /// Compile + validate a `.wasm` **component** from bytes — the first
        /// gate of the load path (Phase 22.4). A core module (not a component)
        /// or malformed bytes are rejected here, before any instantiation.
        pub fn load_component(&self, bytes: &[u8]) -> Result<Component> {
            Component::new(&self.engine, bytes)
        }

        /// Like [`load_component`](Self::load_component), but backed by an
        /// on-disk **compiled-module cache** at `cache_file` (Phase 22.9).
        ///
        /// Every `axil <plugin-cmd>` is a fresh process that reloads the plugin
        /// set; without a cache each invocation Cranelift-compiles every
        /// installed component from scratch — a real, repeated, user-visible
        /// cost. The cache stores `[32-byte source SHA-256][serialized
        /// component]`; a hit deserializes the precompiled artifact (no
        /// Cranelift), a content change or version drift falls back to a
        /// recompile that rewrites the file. There is exactly one cache file per
        /// plugin, so an in-place upgrade overwrites rather than accumulates.
        ///
        /// Caching is best-effort: a read/write/deserialize failure only costs a
        /// recompile, never correctness.
        ///
        /// # Safety
        /// [`Component::deserialize`] is `unsafe` because a precompiled artifact
        /// is essentially native code. This is sound here because the artifact is
        /// (a) produced by *this* process from the exact source bytes, (b) stored
        /// in a directory the host owns, (c) gated on a SHA-256 of the current
        /// source so a mismatched file is never deserialized, and (d) further
        /// validated by Wasmtime's own embedded version/config tag (a drift
        /// returns `Err`, not UB). An attacker who can write the cache dir can
        /// already replace the `.wasm` itself, so the cache widens no trust
        /// boundary.
        pub fn load_component_cached(&self, bytes: &[u8], cache_file: &Path) -> Result<Component> {
            let want = Sha256::digest(bytes);
            if let Ok(cached) = std::fs::read(cache_file) {
                if cached.len() > CACHE_TAG_LEN && cached[..CACHE_TAG_LEN] == want[..] {
                    // SAFETY: see method docs — self-produced, content-keyed
                    // artifact; Wasmtime validates its own version tag and errors
                    // (not UB) on drift, so a stale hit recompiles below.
                    if let Ok(c) =
                        unsafe { Component::deserialize(&self.engine, &cached[CACHE_TAG_LEN..]) }
                    {
                        return Ok(c);
                    }
                }
            }
            let component = Component::new(&self.engine, bytes)?;
            if let Ok(serialized) = component.serialize() {
                let mut buf = Vec::with_capacity(CACHE_TAG_LEN + serialized.len());
                buf.extend_from_slice(&want);
                buf.extend_from_slice(&serialized);
                let _ = atomic_write(cache_file, &buf);
            }
            Ok(component)
        }

        /// Instantiate a loaded component, wiring the `axil:plugin` host imports
        /// against `state` (Phase 22.4). The returned `Store` + `Plugin` bindings
        /// are what the `WasmExtension` shim drives to call the guest's exports.
        ///
        /// `add_to_linker` registers every host import (record CRUD, recall,
        /// graph, fts, embed, config, log) so the guest's calls back into Axil
        /// resolve to the [`PluginState`](crate::PluginState) host impl.
        pub fn instantiate(
            &self,
            component: &Component,
            state: crate::abi::PluginState,
        ) -> Result<(Store<crate::abi::PluginState>, crate::bindings::Plugin)> {
            let mut store = Store::new(&self.engine, state);
            // Seed the CPU budget so the guest can run; a CPU-bound plugin
            // exhausts it and traps rather than hanging the host. Per-call
            // refueling + a configurable ceiling is the Phase 22.5/22.9 refinement.
            store.set_fuel(DEFAULT_FUEL)?;
            // Epoch interruption is armed (the wall-clock bound) but inert until
            // Phase 22.5 adds a ticker thread + a real per-call timeout; set a
            // non-tripping deadline so the guest isn't interrupted immediately.
            store.set_epoch_deadline(u64::MAX);
            let linker = self.host_linker()?;
            let plugin = crate::bindings::Plugin::instantiate(&mut store, component, &linker)?;
            Ok((store, plugin))
        }

        /// Build a [`Linker`] with every `axil:plugin` host import registered.
        /// Exposed so the host-import wiring can be verified without a guest
        /// component (a guest is required for full instantiation, since that
        /// also binds the guest's exports).
        pub fn host_linker(&self) -> Result<Linker<crate::abi::PluginState>> {
            let mut linker = Linker::<crate::abi::PluginState>::new(&self.engine);
            // WASI (deny-by-default ctx) so std-using guests instantiate.
            wasmtime_wasi::add_to_linker_sync(&mut linker)?;
            crate::bindings::Plugin::add_to_linker(&mut linker, |s| s)?;
            Ok(linker)
        }
    }

    /// Write `data` to `path` atomically: create the parent dir, write a
    /// process-unique temp sibling, then rename onto `path` (atomic on POSIX).
    /// Concurrent `axil` processes therefore never observe a half-written cache
    /// file — the loser of a rename race just overwrites with identical content.
    fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(feature = "wasm-host")]
pub use host::WasmHost;

#[cfg(feature = "wasm-host")]
pub use abi::{Capabilities, PluginState};

#[cfg(feature = "wasm-host")]
pub use shim::WasmExtension;

/// Makes a loaded `.wasm` component "just another `dyn Extension`" (Phase 22.4):
/// [`WasmExtension`] implements [`axil_core::Extension`] by calling the guest's
/// exports across the sandbox boundary, so a loaded plugin registers into
/// `db.extensions()` and flows through dispatch / dynamic CLI / boot exactly
/// like a native Extension — with zero per-plugin host code.
#[cfg(feature = "wasm-host")]
mod shim {
    use std::sync::{Mutex, MutexGuard};

    use axil_core::{
        AxilError, CliArg, CliInvocation, CliOutput, CliSubcommand, CliSurface, Dispatch, Extension,
        Hit, McpCall, McpSurface, McpTool, RefreshOpts, RefreshReport,
    };
    use wasmtime::component::Component;
    use wasmtime::Store;

    use crate::bindings::axil::plugin::types as wit;
    use crate::bindings::Plugin;
    use crate::{PluginState, WasmHost};

    /// The instantiated guest + its store. Wasmtime's `Store` is `!Sync`, so the
    /// `WasmExtension` wraps this in a `Mutex` to satisfy `Extension: Send + Sync`
    /// — calls into the guest serialize on the instance lock.
    struct Instance {
        store: Store<PluginState>,
        plugin: Plugin,
    }

    /// A loaded WASM plugin presented as a native [`Extension`].
    ///
    /// Metadata is fetched once at load and cached (Phase 22.0) so the
    /// borrow-returning trait methods never cross the boundary; handlers call
    /// the guest on demand under the instance lock.
    pub struct WasmExtension {
        inner: Mutex<Instance>,
        id: String,
        display_name: String,
        // Leaked at load so `table_prefixes(&self) -> &[&str]` can serve from
        // cache. A plugin lives for the process, so this is a bounded one-time leak.
        prefixes: Vec<&'static str>,
        cli: Option<CliSurface>,
        mcp: Option<McpSurface>,
    }

    impl WasmExtension {
        /// Instantiate `component` against the host imports and cache its
        /// metadata — the load gate (a guest that traps here is rejected, not
        /// surfaced as a runtime error).
        pub fn load(
            host: &WasmHost,
            component: &Component,
            state: PluginState,
        ) -> anyhow::Result<Self> {
            let (mut store, plugin) = host.instantiate(component, state)?;
            let ext = plugin.axil_plugin_extension();
            let id = ext.call_id(&mut store)?;
            let display_name = ext.call_display_name(&mut store)?;
            let declared = ext.call_table_prefixes(&mut store)?;
            // Constrain the plugin's host writes to exactly the tables it
            // declares it owns (the host ABI's prefix check reads these).
            store.data_mut().set_prefixes(declared.clone());
            let prefixes: Vec<&'static str> = declared
                .into_iter()
                .map(|s| &*Box::leak(s.into_boxed_str()))
                .collect();
            let cli = ext.call_cli_commands(&mut store)?.map(cli_surface_from_wit);
            let mcp = ext.call_mcp_tools(&mut store)?.map(mcp_surface_from_wit);
            Ok(Self {
                inner: Mutex::new(Instance { store, plugin }),
                id,
                display_name,
                prefixes,
                cli,
                mcp,
            })
        }

        fn lock(&self) -> Result<MutexGuard<'_, Instance>, AxilError> {
            self.inner
                .lock()
                .map_err(|_| AxilError::plugin("plugin instance lock poisoned"))
        }
    }

    impl Extension for WasmExtension {
        fn id(&self) -> &str {
            &self.id
        }
        fn display_name(&self) -> &str {
            &self.display_name
        }
        fn table_prefixes(&self) -> &[&str] {
            &self.prefixes
        }
        fn cli_commands(&self) -> Option<CliSurface> {
            self.cli.clone()
        }
        fn mcp_tools(&self) -> Option<McpSurface> {
            self.mcp.clone()
        }

        fn boot_block(&self, _db: &axil_core::Axil) -> Option<String> {
            // Infallible in the trait: a lock/trap/guest error collapses to None.
            let mut inner = self.lock().ok()?;
            let Instance { store, plugin } = &mut *inner;
            plugin
                .axil_plugin_extension()
                .call_boot_block(&mut *store)
                .ok()?
                .ok()?
        }

        fn handle_cli(
            &self,
            _db: &axil_core::Axil,
            invocation: &CliInvocation,
        ) -> Result<Dispatch<CliOutput>, AxilError> {
            let wit_inv = wit::CliInvocation {
                command_path: invocation.command_path.clone(),
                args: invocation.args.clone(),
                stdin: invocation.stdin.clone(),
            };
            let mut inner = self.lock()?;
            let Instance { store, plugin } = &mut *inner;
            let res = plugin
                .axil_plugin_extension()
                .call_handle_cli(&mut *store, &wit_inv)
                .map_err(trap_err)?;
            match res {
                Ok(wit::DispatchCli::Handled(out)) => Ok(Dispatch::Handled(CliOutput {
                    exit_code: out.exit_code,
                    stdout: out.stdout,
                    stderr: out.stderr,
                })),
                Ok(wit::DispatchCli::NotHandled) => Ok(Dispatch::NotHandled),
                Err(e) => Err(plugin_err(e)),
            }
        }

        fn handle_mcp(
            &self,
            _db: &axil_core::Axil,
            call: &McpCall,
        ) -> Result<Dispatch<serde_json::Value>, AxilError> {
            let wit_call = wit::McpCall {
                tool: call.tool.clone(),
                params: serde_json::to_string(&call.params).unwrap_or_else(|_| "null".into()),
            };
            let mut inner = self.lock()?;
            let Instance { store, plugin } = &mut *inner;
            let res = plugin
                .axil_plugin_extension()
                .call_handle_mcp(&mut *store, &wit_call)
                .map_err(trap_err)?;
            match res {
                Ok(wit::DispatchMcp::Handled(json)) => {
                    let v = serde_json::from_str(&json)
                        .map_err(|e| AxilError::plugin(format!("plugin returned invalid JSON: {e}")))?;
                    Ok(Dispatch::Handled(v))
                }
                Ok(wit::DispatchMcp::NotHandled) => Ok(Dispatch::NotHandled),
                Err(e) => Err(plugin_err(e)),
            }
        }

        fn refresh(
            &self,
            _db: &axil_core::Axil,
            opts: RefreshOpts,
        ) -> Result<RefreshReport, AxilError> {
            let wit_opts = wit::RefreshOpts {
                if_stale: opts.if_stale,
                path: opts.path.map(|p| p.to_string_lossy().into_owned()),
            };
            let mut inner = self.lock()?;
            let Instance { store, plugin } = &mut *inner;
            let res = plugin
                .axil_plugin_extension()
                .call_refresh(&mut *store, &wit_opts)
                .map_err(trap_err)?;
            match res {
                Ok(r) => {
                    let mut report = RefreshReport::default().with_counts(
                        r.inspected as usize,
                        r.stale as usize,
                        r.refreshed as usize,
                    );
                    report.details = r.details;
                    Ok(report)
                }
                Err(e) => Err(plugin_err(e)),
            }
        }

        fn recall_for_file(
            &self,
            _db: &axil_core::Axil,
            path: &std::path::Path,
        ) -> Result<Vec<Hit>, AxilError> {
            let p = path.to_string_lossy().into_owned();
            let mut inner = self.lock()?;
            let Instance { store, plugin } = &mut *inner;
            let res = plugin
                .axil_plugin_extension()
                .call_recall_for_file(&mut *store, &p)
                .map_err(trap_err)?;
            match res {
                Ok(hits) => Ok(hits.into_iter().map(hit_from_wit).collect()),
                Err(e) => Err(plugin_err(e)),
            }
        }
    }

    fn trap_err(e: anyhow::Error) -> AxilError {
        AxilError::plugin(format!("wasm trap: {e}"))
    }

    fn plugin_err(e: wit::PluginError) -> AxilError {
        use wit::PluginError as P;
        match e {
            P::NotFound(m) => AxilError::NotFound(m),
            P::PermissionDenied(m) => AxilError::plugin(format!("permission denied: {m}")),
            P::PrefixViolation(m) => AxilError::plugin(format!("prefix violation: {m}")),
            P::ResourceExhausted(m) => AxilError::plugin(format!("resource exhausted: {m}")),
            P::Invalid(m) => AxilError::InvalidQuery(m),
            P::Internal(m) => AxilError::plugin(m),
        }
    }

    fn cli_surface_from_wit(s: wit::CliSurface) -> CliSurface {
        CliSurface::new(s.command, s.about).subcommands(s.subcommands.into_iter().map(|sub| {
            CliSubcommand::new(sub.name, sub.about).args(sub.args.into_iter().map(|a| {
                CliArg::new(a.name, a.about)
                    .required(a.required)
                    .takes_value(a.takes_value)
            }))
        }))
    }

    fn mcp_surface_from_wit(s: wit::McpSurface) -> McpSurface {
        McpSurface::new(
            s.tools
                .into_iter()
                .map(|t| {
                    let schema =
                        serde_json::from_str(&t.input_schema).unwrap_or(serde_json::Value::Null);
                    McpTool::new(t.name, t.description, schema)
                })
                .collect(),
        )
    }

    fn hit_from_wit(h: wit::Hit) -> Hit {
        let hit = Hit::new(h.table, h.id, h.score);
        match h.summary {
            Some(s) => hit.with_summary(s),
            None => hit,
        }
    }
}

/// Host-side implementation of the `axil:plugin` `host` interface — backs every
/// capability a plugin may call into Axil with (Phase 22.3), enforcing the
/// granted capability set and the plugin's declared table prefixes.
#[cfg(feature = "wasm-host")]
mod abi {
    use std::sync::Arc;

    use axil_core::{Axil, AxilConfig, AxilError, RecordId};
    use serde_json::Value;

    use crate::bindings::axil::plugin::types as wit;

    /// What a plugin is allowed to call into Axil with. Deny-by-default: a
    /// freshly-constructed `Capabilities` grants nothing (Constraint C3 — the
    /// grant, not the sandbox, is the trust decision).
    #[derive(Debug, Clone, Default)]
    pub struct Capabilities {
        pub records_read: bool,
        pub records_write: bool,
        pub recall: bool,
        pub embed: bool,
        pub graph: bool,
        pub fts: bool,
        pub config_read: bool,
    }

    impl Capabilities {
        /// Grant everything — for trusted in-tree use and tests.
        pub fn all() -> Self {
            Self {
                records_read: true,
                records_write: true,
                recall: true,
                embed: true,
                graph: true,
                fts: true,
                config_read: true,
            }
        }

        /// Build a grant set from capability names (deny-by-default; unknown
        /// names are ignored). The canonical names match [`Capabilities::granted_names`].
        pub fn from_names<I, S>(names: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: AsRef<str>,
        {
            let mut c = Self::default();
            for name in names {
                match name.as_ref() {
                    "records.read" => c.records_read = true,
                    "records.write" => c.records_write = true,
                    "recall" => c.recall = true,
                    "embed" => c.embed = true,
                    "graph" => c.graph = true,
                    "fts" => c.fts = true,
                    "config.read" => c.config_read = true,
                    _ => {}
                }
            }
            c
        }

        /// The granted capabilities as their canonical names (for display).
        pub fn granted_names(&self) -> Vec<&'static str> {
            let mut out = Vec::new();
            if self.records_read {
                out.push("records.read");
            }
            if self.records_write {
                out.push("records.write");
            }
            if self.recall {
                out.push("recall");
            }
            if self.embed {
                out.push("embed");
            }
            if self.graph {
                out.push("graph");
            }
            if self.fts {
                out.push("fts");
            }
            if self.config_read {
                out.push("config.read");
            }
            out
        }

        /// Every capability name a plugin may request (for validation + help).
        pub const ALL_NAMES: &'static [&'static str] = &[
            "records.read",
            "records.write",
            "recall",
            "embed",
            "graph",
            "fts",
            "config.read",
        ];
    }

    /// Per-plugin host state threaded through Wasmtime as the store data. Holds
    /// the shared `Axil`, the plugin's granted capabilities, the table prefixes
    /// it owns (writes outside them are rejected), and a config snapshot for
    /// `config-get`.
    pub struct PluginState {
        db: Arc<Axil>,
        caps: Capabilities,
        prefixes: Vec<String>,
        config: AxilConfig,
        // WASI with NO granted access (no preopens, env, or sockets) — present
        // only so std-using guests instantiate. The sandbox is deny-by-default.
        wasi: wasmtime_wasi::WasiCtx,
        table: wasmtime_wasi::ResourceTable,
    }

    impl PluginState {
        /// Build host state for a plugin granted `caps` and owning `prefixes`.
        pub fn new(
            db: Arc<Axil>,
            caps: Capabilities,
            prefixes: Vec<String>,
            config: AxilConfig,
        ) -> Self {
            Self {
                db,
                caps,
                prefixes,
                config,
                wasi: wasmtime_wasi::WasiCtxBuilder::new().build(),
                table: wasmtime_wasi::ResourceTable::new(),
            }
        }

        /// Set the table prefixes this plugin may write to. Called by
        /// [`crate::WasmExtension::load`] from the guest's own
        /// `table-prefixes()` declaration — a plugin writes to exactly the
        /// tables it declares it owns.
        pub fn set_prefixes(&mut self, prefixes: Vec<String>) {
            self.prefixes = prefixes;
        }

        fn require(&self, granted: bool, cap: &str) -> Result<(), wit::PluginError> {
            if granted {
                Ok(())
            } else {
                Err(wit::PluginError::PermissionDenied(cap.to_string()))
            }
        }

        fn check_prefix(&self, table: &str) -> Result<(), wit::PluginError> {
            if self.prefixes.iter().any(|p| table.starts_with(p.as_str())) {
                Ok(())
            } else {
                Err(wit::PluginError::PrefixViolation(format!(
                    "table `{table}` is outside this plugin's declared prefixes"
                )))
            }
        }
    }

    impl wasmtime_wasi::WasiView for PluginState {
        fn table(&mut self) -> &mut wasmtime_wasi::ResourceTable {
            &mut self.table
        }
        fn ctx(&mut self) -> &mut wasmtime_wasi::WasiCtx {
            &mut self.wasi
        }
    }

    /// Map a core error to the stable plugin-error enum the guest sees.
    fn to_wit_err(e: AxilError) -> wit::PluginError {
        match e {
            AxilError::NotFound(m) => wit::PluginError::NotFound(m),
            other => wit::PluginError::Internal(other.to_string()),
        }
    }

    fn parse_json(s: &str) -> Result<Value, wit::PluginError> {
        serde_json::from_str(s)
            .map_err(|e| wit::PluginError::Invalid(format!("payload is not valid JSON: {e}")))
    }

    fn record_to_json(data: &Value) -> String {
        serde_json::to_string(data).unwrap_or_else(|_| "null".to_string())
    }

    // The `types` interface defines only data types, but bindgen still emits an
    // (empty) Host trait for it that the store data must satisfy.
    impl crate::bindings::axil::plugin::types::Host for PluginState {}

    impl crate::bindings::axil::plugin::host::Host for PluginState {
        fn insert(
            &mut self,
            table: String,
            data: String,
        ) -> Result<String, wit::PluginError> {
            (|| {
                self.require(self.caps.records_write, "records.write")?;
                self.check_prefix(&table)?;
                let value = parse_json(&data)?;
                let rec = self.db.insert(&table, value).map_err(to_wit_err)?;
                Ok(rec.id.to_string())
            })()
        }

        fn get(
            &mut self,
            id: String,
        ) -> Result<Option<String>, wit::PluginError> {
            (|| {
                self.require(self.caps.records_read, "records.read")?;
                let rec = self.db.get(&RecordId(id)).map_err(to_wit_err)?;
                Ok(rec.map(|r| record_to_json(&r.data)))
            })()
        }

        fn update(
            &mut self,
            id: String,
            data: String,
        ) -> Result<(), wit::PluginError> {
            (|| {
                self.require(self.caps.records_write, "records.write")?;
                let rid = RecordId(id);
                let existing = self
                    .db
                    .get(&rid)
                    .map_err(to_wit_err)?
                    .ok_or_else(|| wit::PluginError::NotFound(format!("record {}", rid)))?;
                self.check_prefix(&existing.table)?;
                let value = parse_json(&data)?;
                self.db.update(&rid, value).map_err(to_wit_err)?;
                Ok(())
            })()
        }

        fn delete(&mut self, id: String) -> Result<bool, wit::PluginError> {
            (|| {
                self.require(self.caps.records_write, "records.write")?;
                let rid = RecordId(id);
                if let Some(existing) = self.db.get(&rid).map_err(to_wit_err)? {
                    self.check_prefix(&existing.table)?;
                }
                self.db.delete(&rid).map_err(to_wit_err)
            })()
        }

        fn list_records(
            &mut self,
            table: String,
        ) -> Result<Vec<String>, wit::PluginError> {
            (|| {
                self.require(self.caps.records_read, "records.read")?;
                self.check_prefix(&table)?;
                let recs = self.db.list(&table).map_err(to_wit_err)?;
                Ok(recs.iter().map(|r| record_to_json(&r.data)).collect())
            })()
        }

        fn recall(
            &mut self,
            query: String,
            top_k: u32,
        ) -> Result<Vec<wit::Hit>, wit::PluginError> {
            (|| {
                self.require(self.caps.recall, "recall")?;
                let hits = self
                    .db
                    .recall(&query, top_k as usize, None)
                    .map_err(to_wit_err)?
                    .into_iter()
                    .map(|r| wit::Hit {
                        table: r.record.table,
                        id: r.record.id.to_string(),
                        summary: None,
                        score: r.score,
                    })
                    .collect();
                Ok(hits)
            })()
        }

        fn embed_text(
            &mut self,
            text: String,
        ) -> Result<Vec<f32>, wit::PluginError> {
            (|| {
                self.require(self.caps.embed, "embed")?;
                self.db.embed_query(&text).map_err(to_wit_err)
            })()
        }

        fn relate(
            &mut self,
            source: String,
            edge_type: String,
            target: String,
            props: String,
        ) -> Result<String, wit::PluginError> {
            (|| {
                self.require(self.caps.graph, "graph.write")?;
                let from = RecordId(source);
                let to = RecordId(target);
                let props = parse_json(&props)?;
                let id = self
                    .db
                    .relate(&from, &edge_type, &to, Some(props))
                    .map_err(to_wit_err)?;
                Ok(id.to_string())
            })()
        }

        fn neighbors(
            &mut self,
            id: String,
            edge_type: Option<String>,
            dir: wit::Direction,
        ) -> Result<Vec<String>, wit::PluginError> {
            (|| {
                self.require(self.caps.graph, "graph.read")?;
                let direction = match dir {
                    wit::Direction::Out => axil_core::Direction::Out,
                    wit::Direction::In => axil_core::Direction::In,
                    wit::Direction::Both => axil_core::Direction::Both,
                };
                let recs = self
                    .db
                    .neighbors(&RecordId(id), edge_type.as_deref(), direction)
                    .map_err(to_wit_err)?;
                Ok(recs.iter().map(|r| r.id.to_string()).collect())
            })()
        }

        fn fts_search(
            &mut self,
            query: String,
            limit: u32,
        ) -> Result<Vec<wit::Hit>, wit::PluginError> {
            (|| {
                self.require(self.caps.fts, "fts")?;
                let hits = self
                    .db
                    .search_text(&query, limit as usize)
                    .map_err(to_wit_err)?
                    .into_iter()
                    .map(|(rec, score)| wit::Hit {
                        table: rec.table,
                        id: rec.id.to_string(),
                        summary: None,
                        score,
                    })
                    .collect();
                Ok(hits)
            })()
        }

        fn config_get(
            &mut self,
            key: String,
        ) -> Result<Option<String>, wit::PluginError> {
            (|| {
                self.require(self.caps.config_read, "config.read")?;
                Ok(axil_core::get_config_value(&self.config, &key))
            })()
        }

        fn log(&mut self, level: wit::LogLevel, message: String) {
            let lvl = match level {
                wit::LogLevel::Trace => "TRACE",
                wit::LogLevel::Debug => "DEBUG",
                wit::LogLevel::Info => "INFO",
                wit::LogLevel::Warn => "WARN",
                wit::LogLevel::Error => "ERROR",
            };
            eprintln!("[plugin {lvl}] {message}");
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::bindings::axil::plugin::host::Host;

        fn state(caps: Capabilities) -> (PluginState, tempfile::TempDir) {
            let dir = tempfile::tempdir().unwrap();
            let db = Arc::new(Axil::open(dir.path().join("t.axil")).build().unwrap());
            let st = PluginState::new(
                db,
                caps,
                vec!["_plug_".to_string()],
                AxilConfig::default(),
            );
            (st, dir)
        }

        #[test]
        fn insert_get_roundtrip_within_prefix() {
            let (mut st, _d) = state(Capabilities::all());
            let id = st
                .insert("_plug_notes".into(), r#"{"k":"v"}"#.into())
                .unwrap();
            let got = st.get(id).unwrap().unwrap();
            assert!(got.contains("\"k\""));
        }

        #[test]
        fn write_outside_prefix_is_rejected() {
            let (mut st, _d) = state(Capabilities::all());
            let err = st.insert("decisions".into(), "{}".into()).unwrap_err();
            assert!(matches!(err, wit::PluginError::PrefixViolation(_)));
        }

        #[test]
        fn missing_capability_is_denied() {
            // records_write not granted.
            let (mut st, _d) = state(Capabilities {
                records_read: true,
                ..Capabilities::default()
            });
            let err = st.insert("_plug_x".into(), "{}".into()).unwrap_err();
            assert!(matches!(err, wit::PluginError::PermissionDenied(_)));
        }

        #[test]
        fn invalid_json_is_reported() {
            let (mut st, _d) = state(Capabilities::all());
            let err = st.insert("_plug_x".into(), "not json".into()).unwrap_err();
            assert!(matches!(err, wit::PluginError::Invalid(_)));
        }

        #[test]
        fn recall_capability_gated() {
            let (mut st, _d) = state(Capabilities::all());
            // No semantic backend in this build → empty, but the call succeeds.
            let hits = st.recall("anything".into(), 5).unwrap();
            assert!(hits.is_empty());

            let (mut st2, _d2) = state(Capabilities::default());
            assert!(matches!(
                st2.recall("x".into(), 5).unwrap_err(),
                wit::PluginError::PermissionDenied(_)
            ));
        }

        #[test]
        fn config_get_reads_snapshot() {
            let (mut st, _d) = state(Capabilities::all());
            // A known default key resolves; an unknown key is None.
            assert!(st
                .config_get("timeseries.full_retention_days".into())
                .unwrap()
                .is_some());
            assert!(st.config_get("nope.nope".into()).unwrap().is_none());
        }
    }
}

/// Whether this build embeds the WASM runtime (the `wasm-host` feature).
///
/// Always available so a host can branch on runtime support without a `cfg`.
pub const fn wasm_host_enabled() -> bool {
    cfg!(feature = "wasm-host")
}

#[cfg(all(test, feature = "wasm-host"))]
mod tests {
    use super::*;

    #[test]
    fn host_builds() {
        assert!(WasmHost::new().is_ok());
        assert!(wasm_host_enabled());
    }

    #[test]
    fn rejects_a_core_module() {
        // A raw core module is not a component — the component loader must
        // reject it rather than silently treating it as one.
        let host = WasmHost::new().unwrap();
        let core_module = wat::parse_str("(module)").unwrap();
        assert!(host.load_component(&core_module).is_err());
    }

    #[test]
    fn loads_a_trivial_component() {
        // The smallest valid component compiles + validates.
        let host = WasmHost::new().unwrap();
        let component = wat::parse_str("(component)").unwrap();
        assert!(host.load_component(&component).is_ok());
    }

    #[test]
    fn rejects_garbage_bytes() {
        let host = WasmHost::new().unwrap();
        assert!(host.load_component(b"not wasm at all").is_err());
    }

    #[test]
    fn host_imports_link_cleanly() {
        // add_to_linker must register every host import (all 12 functions + the
        // types Host) without a name clash or missing impl. This is the host
        // half of the load path; full instantiation additionally requires a
        // guest that exports the `extension` interface (the hello guest).
        let host = WasmHost::new().unwrap();
        assert!(host.host_linker().is_ok());
    }

    #[test]
    fn module_cache_populates_then_reuses() {
        // First load compiles + writes the artifact; the second is served from
        // the cache. We can't observe "no Cranelift" directly, but the file's
        // presence (with the 32-byte content tag + a serialized body) and a
        // clean second load prove the round-trip.
        let host = WasmHost::new().unwrap();
        let bytes = wat::parse_str("(component)").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("p.cwasm");
        assert!(!cache.exists());

        host.load_component_cached(&bytes, &cache).unwrap();
        let len1 = std::fs::metadata(&cache).unwrap().len();
        assert!(len1 > 32, "cache artifact carries the content tag + body");

        // Reuse: deserializes from cache, succeeds, file unchanged.
        host.load_component_cached(&bytes, &cache).unwrap();
        assert_eq!(std::fs::metadata(&cache).unwrap().len(), len1);
    }

    #[test]
    fn module_cache_recompiles_on_content_change() {
        // A cache file whose content tag doesn't match the source is never
        // deserialized (the unsafe path is gated on the SHA-256) — it recompiles
        // and overwrites, so a stale/foreign artifact is self-healing.
        let host = WasmHost::new().unwrap();
        let bytes = wat::parse_str("(component)").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("p.cwasm");
        std::fs::write(&cache, vec![0u8; 64]).unwrap(); // wrong tag + garbage body

        // Must not touch the unsafe deserialize (tag mismatch); recompiles clean.
        host.load_component_cached(&bytes, &cache).unwrap();
        let body = std::fs::read(&cache).unwrap();
        assert!(body.len() > 64, "stale artifact was rewritten with a real one");
    }

    #[test]
    fn module_cache_tolerates_unwritable_dir() {
        // Best-effort: if the cache can't be written (parent is a *file*, not a
        // dir), the load still succeeds by compiling — caching never gates
        // correctness.
        let host = WasmHost::new().unwrap();
        let bytes = wat::parse_str("(component)").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-dir");
        std::fs::write(&blocker, b"x").unwrap();
        let cache = blocker.join("p.cwasm"); // parent is a file → create_dir_all fails
        assert!(host.load_component_cached(&bytes, &cache).is_ok());
    }
}

#[cfg(all(test, not(feature = "wasm-host")))]
mod tests {
    #[test]
    fn default_build_has_no_runtime() {
        assert!(!super::wasm_host_enabled());
    }
}
