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

#[cfg(feature = "wasm-host")]
mod host {
    use anyhow::Result;
    use wasmtime::component::Component;
    use wasmtime::{Config, Engine};

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
    }
}

#[cfg(feature = "wasm-host")]
pub use host::WasmHost;

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
}

#[cfg(all(test, not(feature = "wasm-host")))]
mod tests {
    #[test]
    fn default_build_has_no_runtime() {
        assert!(!super::wasm_host_enabled());
    }
}
