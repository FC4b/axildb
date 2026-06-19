# Authoring WASM Plugins

A **WASM plugin** is a Tier-2 Extension shipped as a `.wasm` Component-Model component. Drop it into `.axil/plugins/` (via `axil ext install`) and it loads, sandboxed, into the live database — no rebuild of `axil`, no fork. To the rest of Axil a loaded plugin is *"just another `dyn Extension`"*: its commands, MCP tools, and boot block flow through the same dispatch as native Extensions.

> Requires an `axil` built with `--features wasm-host` (off by default to keep the standard binary small — it pulls in Wasmtime).

## The ABI

Plugins implement the `axil:plugin@1.0.0` **WIT world** ([`wit/axil-plugin.wit`](https://github.com/FC4b/axildb/blob/main/wit/axil-plugin.wit)). It mirrors the stable `axil_core::Extension` trait:

- **You export** the `extension` interface: `id`, `display-name`, `table-prefixes`, `cli-commands`, `mcp-tools`, `handle-cli`, `handle-mcp`, `boot-block`, `refresh`, `recall-for-file`.
- **You may import** the `host` interface to call back into Axil: record CRUD, `recall`, `embed-text`, graph `relate`/`neighbors`, `fts-search`, `config-get`, `log` — each **capability-gated** (see below).

`serde_json::Value` crosses the boundary as a canonical JSON string.

## Write one (Rust)

The reference plugin lives at [`crates/adapters/axil-runtime/test-guest`](https://github.com/FC4b/axildb/tree/main/crates/adapters/axil-runtime/test-guest). The essentials:

```toml
# Cargo.toml
[package]
name = "my-plugin"
edition = "2021"
[workspace]                 # detached — built for wasm, not a workspace member

[lib]
crate-type = ["cdylib"]

[dependencies]
wit-bindgen-rt = { version = "0.41", features = ["bitflags"] }

[package.metadata.component]
package = "axil:my-plugin"
[package.metadata.component.target]
path = "path/to/wit"        # point at Axil's wit/ dir
world = "plugin"
```

```rust
// src/lib.rs
#[allow(warnings)]
mod bindings;
use bindings::axil::plugin::types::{CliInvocation, CliOutput, CliSurface, DispatchCli, PluginError};
use bindings::exports::axil::plugin::extension::Guest;

struct Component;
impl Guest for Component {
    fn id() -> String { "my-plugin".into() }
    fn display_name() -> String { "My Plugin".into() }
    fn table_prefixes() -> Vec<String> { vec!["_myplugin_".into()] }
    fn cli_commands() -> Option<CliSurface> {
        Some(CliSurface { command: "my-plugin".into(), about: "...".into(), subcommands: vec![] })
    }
    fn handle_cli(inv: CliInvocation) -> Result<DispatchCli, PluginError> {
        Ok(DispatchCli::Handled(CliOutput { exit_code: 0, stdout: format!("hi {:?}", inv.args), stderr: String::new() }))
    }
    // mcp_tools / handle_mcp / boot_block / refresh / recall_for_file:
    // return None / NotHandled / empty as needed.
}
bindings::export!(Component with_types_in bindings);
```

Build (needs `cargo-component` + the `wasm32-wasip2` target):

```sh
cargo install cargo-component
rustup target add wasm32-wasip2
cargo component build --release
# -> target/wasm32-wasip1/release/my_plugin.wasm  (a component)
```

Any Component-Model language works; Rust is just the best-supported path.

## Install, run, manage

```sh
axil ext install ./my_plugin.wasm    # validates it loads, copies into .axil/plugins/
axil my-plugin hello                  # its command works — zero code in axil-cli
axil ext list                         # key, id, prefixes, granted caps, load status
axil ext info my-plugin
axil ext remove my-plugin
```

A `.wasm` that fails to load is **quarantined** — reported by `ext list`, never fatal to open or to other plugins.

## Capabilities (sandbox)

Plugins are **deny-by-default**. A freshly installed plugin runs, but every call back into Axil is refused until the operator grants it. Ambient filesystem/network are off (WASI denied); CPU is bounded by fuel and each call by a wall-clock timeout (a runaway guest traps instead of hanging the host); record writes are constrained to the plugin's own declared `table-prefixes`.

Grant what a plugin needs by its **key** (its `.wasm` filename prefix, shown by `ext list`):

```sh
axil ext grant  my-plugin records.read
axil ext grant  my-plugin recall
axil ext revoke my-plugin records.read
```

which writes:

```toml
[plugins.my-plugin]
capabilities = ["recall"]
```

Capabilities: `records.read`, `records.write` (own-prefix only), `recall`, `query`, `embed`, `graph`, `fts`, `config.read`. A plugin granted `recall` + a network capability can read the whole memory DB — **the grant is the trust decision**, not just the sandbox. Only grant what a plugin you trust actually needs.

## Status

Shipped: the ABI, runtime, host imports, the `WasmExtension` shim, discovery, the `axil ext` commands, the capability model, **load-time ABI-version negotiation** (a clear error when a plugin's `axil:plugin@X.Y.Z` isn't one this host implements — its version shows in `ext list`/`info`), a **compiled-module cache** (`.axil/plugins/.cache/` — repeat invocations deserialize a precompiled artifact instead of recompiling, ~16× faster), and a **host-ABI conformance suite** (a real guest exercises every host import across the boundary, asserting capability gating, prefix enforcement, marshalling, and fault isolation — `crates/adapters/axil-runtime/conformance-guest/`). Still polish (Phase 22): an ergonomic guest-side SDK macro and a fuzz harness.
