//! Host-ABI conformance suite.
//!
//! Drives the `conformance-guest` fixture — a real WASM component that calls
//! every `axil:plugin` host import — through the `WasmExtension` shim, asserting
//! the contract end-to-end across the sandbox boundary:
//!
//! * **capability gating** — a denied capability surfaces as a typed error;
//! * **prefix enforcement** — a write outside the plugin's declared prefix is
//!   rejected even when the capability is granted;
//! * **marshalling** — records, JSON, and the `Dispatch` variants cross the
//!   boundary intact (write-then-read round-trips host-side);
//! * **fault isolation** — a host call into a missing engine returns a clean
//!   `plugin-error`, never a trap;
//! * **lifecycle** — `boot_block` / `refresh` / `recall_for_file` / `handle_mcp`
//!   all behave as native `Extension` methods.
//!
//! The host's own unit tests (`abi::tests` in lib.rs) cover the same gating at
//! the `PluginState` level; this suite proves the *wiring* — that a guest's
//! calls actually reach that gated state through the linker.
#![cfg(feature = "wasm-host")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use axil_core::{
    Axil, AxilConfig, CliInvocation, CliOutput, Dispatch, Extension, McpCall, RefreshOpts,
};
use axil_runtime::{Capabilities, PluginState, WasmExtension, WasmHost};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/conformance-guest.component.wasm"
);

/// Load the conformance guest against a fresh core-only DB with `caps` granted.
fn load(caps: Capabilities) -> (WasmExtension, Arc<Axil>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Axil::open(dir.path().join("conf.axil")).build().unwrap());
    let host = WasmHost::new().unwrap();
    let bytes = std::fs::read(FIXTURE).expect(
        "conformance-guest fixture missing — run conformance-guest/build.sh (needs cargo-component)",
    );
    let component = host.load_component(&bytes).unwrap();
    let state = PluginState::new(Arc::clone(&db), caps, Vec::new(), AxilConfig::default());
    let ext = WasmExtension::load(&host, &component, state).unwrap();
    (ext, db, dir)
}

/// Run one conformance op through `handle_cli`.
fn op(
    ext: &WasmExtension,
    db: &Axil,
    name: &str,
) -> Result<Dispatch<CliOutput>, axil_core::AxilError> {
    let inv = CliInvocation::new(vec!["conf".to_string()], vec![name.to_string()], None);
    ext.handle_cli(db, &inv)
}

fn handled_stdout(d: Dispatch<CliOutput>) -> String {
    match d {
        Dispatch::Handled(o) => o.stdout,
        Dispatch::NotHandled => panic!("expected Handled, got NotHandled"),
    }
}

#[test]
fn granted_record_crud_round_trips_through_the_boundary() {
    let (ext, db, _d) = load(Capabilities::all());

    // insert → a real record lands in the DB (verified host-side, not just the
    // returned id), proving guest→host→redb wiring + the prefix check passing.
    let id = handled_stdout(op(&ext, &db, "insert").unwrap());
    assert!(!id.is_empty(), "insert returns a record id");
    assert_eq!(db.list("_conf_notes").unwrap().len(), 1, "write landed in redb");

    // get reads back what a prior insert wrote.
    assert!(handled_stdout(op(&ext, &db, "get").unwrap()).contains("\"k\""));
    // list / update / delete exercise the rest of CRUD.
    assert_eq!(handled_stdout(op(&ext, &db, "list").unwrap()), "1");
    assert!(handled_stdout(op(&ext, &db, "update").unwrap()).contains("v2"));
    assert_eq!(handled_stdout(op(&ext, &db, "delete").unwrap()), "deleted=true");
}

#[test]
fn config_and_log_imports_work() {
    let (ext, db, _d) = load(Capabilities::all());
    // A known default config key resolves to a value (not "none").
    assert_ne!(handled_stdout(op(&ext, &db, "config").unwrap()), "none");
    // log is infallible and must not trap.
    assert_eq!(handled_stdout(op(&ext, &db, "log").unwrap()), "logged");
}

#[test]
fn recall_import_is_reachable_without_a_vector_engine() {
    // No vector engine in this build → recall returns an empty set (Ok), so the
    // guest reports "0". The point is the call reaches the host without trapping.
    let (ext, db, _d) = load(Capabilities::all());
    assert_eq!(handled_stdout(op(&ext, &db, "recall").unwrap()), "0");
}

#[test]
fn missing_capability_is_denied_end_to_end() {
    // records.write NOT granted → the guest's host::insert returns
    // permission-denied, which propagates as an Err out of handle_cli.
    let (ext, db, _d) = load(Capabilities::default());
    let err = op(&ext, &db, "insert").unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("permission denied"),
        "expected a permission-denied error, got: {err}"
    );
    // And nothing was written.
    assert_eq!(db.list("_conf_notes").unwrap().len(), 0);
}

#[test]
fn write_outside_declared_prefix_is_rejected_even_when_granted() {
    // Full caps, but the guest tries to write the `decisions` table — outside
    // its declared `_conf_` prefix. The host's prefix check must refuse it.
    let (ext, db, _d) = load(Capabilities::all());
    let err = op(&ext, &db, "escape").unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("prefix"),
        "expected a prefix violation, got: {err}"
    );
    assert_eq!(db.list("decisions").unwrap().len(), 0, "no cross-prefix write");
}

#[test]
fn missing_engine_import_fails_cleanly_not_as_a_trap() {
    // embed/fts/relate need engines this build doesn't load. Each must surface a
    // typed plugin-error (Err), never abort the host — fault isolation.
    let (ext, db, _d) = load(Capabilities::all());
    for bad in ["embed", "fts", "relate"] {
        let r = op(&ext, &db, bad);
        assert!(r.is_err(), "`{bad}` should fail cleanly without its engine");
    }
    // The instance is still usable afterward — a clean error didn't poison it.
    assert_eq!(handled_stdout(op(&ext, &db, "log").unwrap()), "logged");
}

#[test]
fn runaway_guest_is_interrupted_by_the_wall_clock_timeout() {
    // The guest's `spin` op never returns. With a 150ms per-call timeout — far
    // below the fuel budget's exhaustion time — the host's epoch ticker trips
    // the deadline and traps it, so the call returns an Err promptly instead of
    // hanging the host forever.
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Axil::open(dir.path().join("conf.axil")).build().unwrap());
    let host = WasmHost::new()
        .unwrap()
        .with_call_timeout(Duration::from_millis(150));
    let bytes = std::fs::read(FIXTURE).unwrap();
    let component = host.load_component(&bytes).unwrap();
    let state = PluginState::new(
        Arc::clone(&db),
        Capabilities::all(),
        Vec::new(),
        AxilConfig::default(),
    );
    let ext = WasmExtension::load(&host, &component, state).unwrap();

    let start = Instant::now();
    let result = op(&ext, &db, "spin");
    let elapsed = start.elapsed();

    assert!(result.is_err(), "a runaway guest must be interrupted");
    assert!(
        elapsed < Duration::from_secs(5),
        "the wall-clock timeout should fire promptly, took {elapsed:?}"
    );
}

#[test]
fn unknown_op_declines_so_the_host_falls_back() {
    let (ext, db, _d) = load(Capabilities::all());
    assert!(matches!(
        op(&ext, &db, "no-such-op").unwrap(),
        Dispatch::NotHandled
    ));
}

#[test]
fn lifecycle_methods_behave_as_a_native_extension() {
    let (ext, db, _d) = load(Capabilities::all());

    // Metadata cached at load.
    assert_eq!(ext.id(), "conformance");
    assert_eq!(ext.table_prefixes(), &["_conf_"]);
    assert_eq!(ext.abi().to_string(), "1.0.0");

    // boot_block contributes a block.
    assert_eq!(ext.boot_block(&db).as_deref(), Some("conformance ready"));

    // refresh returns the guest's report.
    let report = ext.refresh(&db, RefreshOpts::new()).unwrap();
    assert_eq!(report.inspected, 1);

    // recall_for_file marshals a Hit back across the boundary.
    let hits = ext
        .recall_for_file(&db, std::path::Path::new("src/main.rs"))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "src/main.rs");
}

#[test]
fn mcp_dispatch_handles_and_declines() {
    let (ext, db, _d) = load(Capabilities::all());

    // `echo` round-trips its params (JSON crosses the boundary both ways).
    let call = McpCall::new("echo", serde_json::json!({"ping": 1}));
    match ext.handle_mcp(&db, &call).unwrap() {
        Dispatch::Handled(v) => assert_eq!(v, serde_json::json!({"ping": 1})),
        Dispatch::NotHandled => panic!("echo should be handled"),
    }

    // Any other tool declines.
    let other = McpCall::new("nope", serde_json::Value::Null);
    assert!(matches!(
        ext.handle_mcp(&db, &other).unwrap(),
        Dispatch::NotHandled
    ));
}
