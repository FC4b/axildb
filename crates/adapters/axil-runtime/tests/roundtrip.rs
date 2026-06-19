//! Full host↔guest round-trip: load the `hello` WASM component, instantiate it
//! against the host imports, and call its `extension` exports — proving the
//! `axil:plugin` ABI works end-to-end (Phase 22.4).
//!
//! The fixture is built from `../test-guest` (see ../build.sh). Only compiled
//! when the `wasm-host` feature is on.
#![cfg(feature = "wasm-host")]

use std::sync::Arc;

use axil_core::{Axil, AxilConfig};
use axil_runtime::bindings::axil::plugin::types::{CliInvocation, DispatchCli};
use axil_runtime::{Capabilities, PluginState, WasmHost};

const HELLO_COMPONENT: &[u8] = include_bytes!("fixtures/hello-guest.component.wasm");

#[test]
fn hello_guest_exports_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Axil::open(dir.path().join("t.axil")).build().unwrap());
    let state = PluginState::new(
        db,
        Capabilities::all(),
        vec!["_hello_".to_string()],
        AxilConfig::default(),
    );

    let host = WasmHost::new().unwrap();
    let component = host.load_component(HELLO_COMPONENT).unwrap();
    let (mut store, plugin) = host.instantiate(&component, state).unwrap();
    let ext = plugin.axil_plugin_extension();

    // Metadata exports.
    assert_eq!(ext.call_id(&mut store).unwrap(), "hello");
    assert_eq!(ext.call_display_name(&mut store).unwrap(), "Hello Plugin");
    assert_eq!(
        ext.call_table_prefixes(&mut store).unwrap(),
        vec!["_hello_".to_string()]
    );

    // boot-block contribution.
    let boot = ext.call_boot_block(&mut store).unwrap().unwrap();
    assert_eq!(boot.as_deref(), Some("hello plugin ready"));

    // handle-cli: the guest echoes its args back through a Handled CliOutput.
    let inv = CliInvocation {
        command_path: vec!["hello".to_string()],
        args: vec!["world".to_string()],
        stdin: None,
    };
    let dispatch = ext.call_handle_cli(&mut store, &inv).unwrap().unwrap();
    match dispatch {
        DispatchCli::Handled(out) => {
            assert_eq!(out.exit_code, 0);
            assert!(
                out.stdout.contains("hello from wasm"),
                "unexpected stdout: {}",
                out.stdout
            );
            assert!(out.stdout.contains("world"));
        }
        DispatchCli::NotHandled => panic!("expected Handled"),
    }
}
