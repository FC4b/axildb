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
    use anyhow::Result;
    use wasmtime::component::{Component, Linker};
    use wasmtime::{Config, Engine, Store};

    /// Default per-instance CPU budget (Wasmtime fuel units). Generous for a
    /// well-behaved plugin; a runaway guest exhausts it and traps. Phase
    /// 22.5/22.9 makes this configurable and refills it per call.
    const DEFAULT_FUEL: u64 = 10_000_000_000;

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
}

#[cfg(feature = "wasm-host")]
pub use host::WasmHost;

#[cfg(feature = "wasm-host")]
pub use abi::{Capabilities, PluginState};

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
}

#[cfg(all(test, not(feature = "wasm-host")))]
mod tests {
    #[test]
    fn default_build_has_no_runtime() {
        assert!(!super::wasm_host_enabled());
    }
}
