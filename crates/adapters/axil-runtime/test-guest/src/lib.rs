//! The hello reference plugin — a minimal guest implementing the `axil:plugin`
//! `extension` world. Compiled to a `.wasm` component (see ../build.sh) and used
//! as the axil-runtime host's round-trip fixture.

#[allow(warnings)]
mod bindings;

use bindings::axil::plugin::types::{
    CliInvocation, CliOutput, CliSurface, DispatchCli, DispatchMcp, Hit, McpCall, McpSurface,
    PluginError, RefreshOpts, RefreshReport,
};
use bindings::exports::axil::plugin::extension::Guest;

struct Component;

impl Guest for Component {
    fn id() -> String {
        "hello".to_string()
    }

    fn display_name() -> String {
        "Hello Plugin".to_string()
    }

    fn table_prefixes() -> Vec<String> {
        vec!["_hello_".to_string()]
    }

    fn cli_commands() -> Option<CliSurface> {
        Some(CliSurface {
            command: "hello".to_string(),
            about: "A WASM hello plugin".to_string(),
            subcommands: vec![],
        })
    }

    fn mcp_tools() -> Option<McpSurface> {
        None
    }

    fn handle_cli(invocation: CliInvocation) -> Result<DispatchCli, PluginError> {
        Ok(DispatchCli::Handled(CliOutput {
            exit_code: 0,
            stdout: format!("hello from wasm; args={:?}", invocation.args),
            stderr: String::new(),
        }))
    }

    fn handle_mcp(_call: McpCall) -> Result<DispatchMcp, PluginError> {
        Ok(DispatchMcp::NotHandled)
    }

    fn boot_block() -> Result<Option<String>, PluginError> {
        Ok(Some("hello plugin ready".to_string()))
    }

    fn refresh(_opts: RefreshOpts) -> Result<RefreshReport, PluginError> {
        Ok(RefreshReport {
            inspected: 0,
            stale: 0,
            refreshed: 0,
            details: vec![],
        })
    }

    fn recall_for_file(_path: String) -> Result<Vec<Hit>, PluginError> {
        Ok(vec![])
    }
}

bindings::export!(Component with_types_in bindings);
