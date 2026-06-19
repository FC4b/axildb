# `wit/` — the Axil plugin ABI

[`axil-plugin.wit`](axil-plugin.wit) is the **WIT (WebAssembly Interface Types) world** for runtime-loadable WASM Extensions — the `axil:plugin@1.0.0` host↔guest contract. It is the WIT mirror of the stable Rust `axil_core::Extension` trait plus the host capability surface.

## Status

This is a **design artifact (Phase 22.1)**. The Wasmtime-based runtime that loads `.wasm` components against this world (the `wasm-host` Cargo feature, Phases 22.2–22.10) is **not built yet** — see [`tasks/phase-21-22-extensibility-and-wasm.md`](../tasks/phase-21-22-extensibility-and-wasm.md).

> ⚠️ Not yet tool-validated. `wasm-tools` / `cargo-component` are not installed in this environment, so the world has not been machine-checked with `wasm-tools component wit ./wit`. It was authored by hand against the WIT spec and the Rust trait it mirrors. Validate it before depending on it:
>
> ```sh
> cargo install wasm-tools
> wasm-tools component wit ./wit/axil-plugin.wit
> ```

## What it encodes

- **`interface types`** — owned mirrors of the `#[non_exhaustive]` structs in `axil_core::extension` (`cli-surface`, `mcp-tool`, `hit`, `refresh-opts`, …). `serde_json::Value` crosses as a canonical JSON `string`.
- **`interface host`** — the capability surface a plugin may call back into Axil with (record CRUD, recall, embed, graph, FTS, config, log). Every call is capability-gated and returns `result<_, plugin-error>`.
- **`interface extension`** — the guest exports. Metadata funcs (`id`, `cli-commands`, …) are infallible and cached once at load; handlers (`handle-cli`, `refresh`, …) are fallible and called at runtime. This split is the Phase 22.0 decision that keeps a `WasmExtension` shim from crossing the boundary inside `dispatch_cli`'s hot loop.
- **`world plugin`** — `import host; export extension;`.
