//! A tiny ergonomic layer over the raw `axil:plugin` `Guest` bindings.
//!
//! The generated `Guest` trait requires all ten methods even for a plugin that
//! only handles a CLI command. This module flips that: implement [`Plugin`] and
//! override **only what you need** — every other method defaults to "decline /
//! empty / no-op". Then `export_plugin!(YourType)` generates the real `Guest`
//! impl (forwarding to your `Plugin` impl) and wires the component export.
//!
//! This is the recommended authoring pattern; it's a drop-in module today and
//! the seed of a standalone `axil-plugin-sdk` crate.

use crate::bindings::axil::plugin::types::{
    CliInvocation, CliSurface, DispatchCli, DispatchMcp, Hit, McpCall, McpSurface, PluginError,
    RefreshOpts, RefreshReport,
};

/// Implement this and override only the methods your plugin uses. Mirrors the
/// host's `axil_core::Extension` trait; the defaults match a plugin that
/// declines every dispatch and contributes nothing.
pub trait Plugin {
    /// Stable kebab-case id (e.g. `"hello"`). The only required method.
    fn id() -> String;

    /// Human-readable name. Defaults to the id.
    fn display_name() -> String {
        Self::id()
    }

    /// Table-name prefixes this plugin owns. Defaults to none (a read-only or
    /// stateless plugin).
    fn table_prefixes() -> Vec<String> {
        Vec::new()
    }

    /// Optional CLI surface. Defaults to none.
    fn cli_commands() -> Option<CliSurface> {
        None
    }

    /// Optional MCP tool surface. Defaults to none.
    fn mcp_tools() -> Option<McpSurface> {
        None
    }

    /// Run a matched CLI subcommand. Defaults to declining (host falls back).
    fn handle_cli(_inv: CliInvocation) -> Result<DispatchCli, PluginError> {
        Ok(DispatchCli::NotHandled)
    }

    /// Run a matched MCP tool call. Defaults to declining.
    fn handle_mcp(_call: McpCall) -> Result<DispatchMcp, PluginError> {
        Ok(DispatchMcp::NotHandled)
    }

    /// Contribute an `axil boot` block. Defaults to none.
    fn boot_block() -> Result<Option<String>, PluginError> {
        Ok(None)
    }

    /// Drift/refresh entry point. Defaults to a no-op report.
    fn refresh(_opts: RefreshOpts) -> Result<RefreshReport, PluginError> {
        Ok(RefreshReport {
            inspected: 0,
            stale: 0,
            refreshed: 0,
            details: Vec::new(),
        })
    }

    /// Surface records relevant to a file. Defaults to none.
    fn recall_for_file(_path: String) -> Result<Vec<Hit>, PluginError> {
        Ok(Vec::new())
    }
}

/// Generate the `axil:plugin` `Guest` impl for a [`Plugin`] type and export the
/// component. Invoke once at the crate root: `export_plugin!(MyType);`.
#[macro_export]
macro_rules! export_plugin {
    // `:ident` (not `:ty`) so the inner `bindings::export!` — which matches
    // `$ty:ident` — can re-accept it; a plugin type is always a bare struct name.
    ($t:ident) => {
        impl $crate::bindings::exports::axil::plugin::extension::Guest for $t {
            fn id() -> String {
                <$t as $crate::sdk::Plugin>::id()
            }
            fn display_name() -> String {
                <$t as $crate::sdk::Plugin>::display_name()
            }
            fn table_prefixes() -> ::std::vec::Vec<String> {
                <$t as $crate::sdk::Plugin>::table_prefixes()
            }
            fn cli_commands() -> Option<$crate::bindings::axil::plugin::types::CliSurface> {
                <$t as $crate::sdk::Plugin>::cli_commands()
            }
            fn mcp_tools() -> Option<$crate::bindings::axil::plugin::types::McpSurface> {
                <$t as $crate::sdk::Plugin>::mcp_tools()
            }
            fn handle_cli(
                inv: $crate::bindings::axil::plugin::types::CliInvocation,
            ) -> Result<
                $crate::bindings::axil::plugin::types::DispatchCli,
                $crate::bindings::axil::plugin::types::PluginError,
            > {
                <$t as $crate::sdk::Plugin>::handle_cli(inv)
            }
            fn handle_mcp(
                call: $crate::bindings::axil::plugin::types::McpCall,
            ) -> Result<
                $crate::bindings::axil::plugin::types::DispatchMcp,
                $crate::bindings::axil::plugin::types::PluginError,
            > {
                <$t as $crate::sdk::Plugin>::handle_mcp(call)
            }
            fn boot_block(
            ) -> Result<Option<String>, $crate::bindings::axil::plugin::types::PluginError> {
                <$t as $crate::sdk::Plugin>::boot_block()
            }
            fn refresh(
                opts: $crate::bindings::axil::plugin::types::RefreshOpts,
            ) -> Result<
                $crate::bindings::axil::plugin::types::RefreshReport,
                $crate::bindings::axil::plugin::types::PluginError,
            > {
                <$t as $crate::sdk::Plugin>::refresh(opts)
            }
            fn recall_for_file(
                path: String,
            ) -> Result<
                ::std::vec::Vec<$crate::bindings::axil::plugin::types::Hit>,
                $crate::bindings::axil::plugin::types::PluginError,
            > {
                <$t as $crate::sdk::Plugin>::recall_for_file(path)
            }
        }
        $crate::bindings::export!($t with_types_in $crate::bindings);
    };
}
