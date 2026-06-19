//! The hello reference plugin — a minimal guest implementing the `axil:plugin`
//! `extension` world. Compiled to a `.wasm` component (see ../build.sh) and used
//! as the axil-runtime host's round-trip fixture.
//!
//! It uses the [`sdk`] ergonomic layer: implement [`sdk::Plugin`] and override
//! only the methods the plugin needs — `mcp_tools` / `handle_mcp` / `refresh` /
//! `recall_for_file` fall back to the trait defaults — then `export_plugin!`
//! generates the real `Guest` impl and the component export.

#[allow(warnings)]
mod bindings;
mod sdk;

use bindings::axil::plugin::types::{CliInvocation, CliOutput, CliSurface, DispatchCli, PluginError};
use sdk::Plugin;

struct Component;

impl Plugin for Component {
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

    fn handle_cli(invocation: CliInvocation) -> Result<DispatchCli, PluginError> {
        Ok(DispatchCli::Handled(CliOutput {
            exit_code: 0,
            stdout: format!("hello from wasm; args={:?}", invocation.args),
            stderr: String::new(),
        }))
    }

    fn boot_block() -> Result<Option<String>, PluginError> {
        Ok(Some("hello plugin ready".to_string()))
    }
}

export_plugin!(Component);
