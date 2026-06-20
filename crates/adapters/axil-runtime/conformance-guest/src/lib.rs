//! Conformance guest — exercises every `axil:plugin` host import across the
//! sandbox boundary so the host-side conformance suite can assert
//! capability gating, prefix enforcement, JSON marshalling, and fault isolation
//! against a *real* component, not just the host's own unit tests.
//!
//! `handle_cli` dispatches on `args[0]`: each op drives one host import (or a
//! deliberate violation, like writing outside the declared prefix). A host call
//! that returns an error is propagated with `?`, so the host observes a typed
//! `plugin-error` — exactly what a denied capability or prefix violation looks
//! like end-to-end.

#[allow(warnings)]
mod bindings;

use bindings::axil::plugin::host;
use bindings::axil::plugin::types::{
    CliInvocation, CliOutput, CliSurface, Direction, DispatchCli, DispatchMcp, Hit, LogLevel,
    McpCall, McpSurface, PluginError, RefreshOpts, RefreshReport,
};
use bindings::exports::axil::plugin::extension::Guest;

struct Component;

/// The single table prefix this plugin declares it owns. Host writes outside it
/// must be rejected by the host's prefix check.
const PREFIX: &str = "_conf_";

fn handled(stdout: String) -> Result<DispatchCli, PluginError> {
    Ok(DispatchCli::Handled(CliOutput {
        exit_code: 0,
        stdout,
        stderr: String::new(),
    }))
}

impl Guest for Component {
    fn id() -> String {
        "conformance".to_string()
    }

    fn display_name() -> String {
        "Conformance Guest".to_string()
    }

    fn table_prefixes() -> Vec<String> {
        vec![PREFIX.to_string()]
    }

    fn cli_commands() -> Option<CliSurface> {
        Some(CliSurface {
            command: "conf".to_string(),
            about: "host-ABI conformance harness".to_string(),
            subcommands: vec![],
        })
    }

    fn mcp_tools() -> Option<McpSurface> {
        None
    }

    fn handle_cli(inv: CliInvocation) -> Result<DispatchCli, PluginError> {
        let op = inv.args.first().map(String::as_str).unwrap_or("");
        match op {
            // ---- record CRUD (core-only; should work whenever granted) ----
            "insert" => {
                let id = host::insert(&format!("{PREFIX}notes"), "{\"k\":\"v\"}")?;
                handled(id)
            }
            "get" => {
                let id = host::insert(&format!("{PREFIX}notes"), "{\"k\":\"v\"}")?;
                let got = host::get(&id)?.unwrap_or_else(|| "null".to_string());
                handled(got)
            }
            "list" => {
                host::insert(&format!("{PREFIX}items"), "{\"n\":1}")?;
                let all = host::list_records(&format!("{PREFIX}items"))?;
                handled(all.len().to_string())
            }
            "update" => {
                let id = host::insert(&format!("{PREFIX}notes"), "{\"k\":\"v\"}")?;
                host::update(&id, "{\"k\":\"v2\"}")?;
                handled(host::get(&id)?.unwrap_or_default())
            }
            "delete" => {
                let id = host::insert(&format!("{PREFIX}notes"), "{\"k\":\"v\"}")?;
                let existed = host::delete(&id)?;
                handled(format!("deleted={existed}"))
            }
            // ---- search / embed / graph (need their engine; may fail cleanly) ----
            "recall" => handled(host::recall("anything", 3)?.len().to_string()),
            "embed" => handled(host::embed_text("hello")?.len().to_string()),
            "fts" => handled(host::fts_search("x", 3)?.len().to_string()),
            "relate" => {
                let a = host::insert(&format!("{PREFIX}a"), "{}")?;
                let b = host::insert(&format!("{PREFIX}b"), "{}")?;
                host::relate(&a, "links", &b, "{}")?;
                let n = host::neighbors(&a, Some("links"), Direction::Out)?;
                handled(n.len().to_string())
            }
            // ---- config + log ----
            "config" => {
                let v = host::config_get("timeseries.full_retention_days")?;
                handled(v.unwrap_or_else(|| "none".to_string()))
            }
            "log" => {
                host::log(LogLevel::Info, "conformance log line");
                handled("logged".to_string())
            }
            // ---- deliberate prefix violation: write OUTSIDE the declared prefix ----
            "escape" => {
                let id = host::insert("decisions", "{}")?;
                handled(id)
            }
            // ---- runaway: never returns; the host's wall-clock timeout (epoch
            //      interruption) must trap it instead of hanging the host ----
            "spin" => loop {
                core::hint::spin_loop();
            },
            // ---- unmatched op exercises the NotHandled marshalling ----
            _ => Ok(DispatchCli::NotHandled),
        }
    }

    fn handle_mcp(call: McpCall) -> Result<DispatchMcp, PluginError> {
        // `echo` round-trips its params (proves JSON crosses the boundary both
        // ways); anything else declines so the host falls back.
        if call.tool == "echo" {
            Ok(DispatchMcp::Handled(call.params))
        } else {
            Ok(DispatchMcp::NotHandled)
        }
    }

    fn boot_block() -> Result<Option<String>, PluginError> {
        Ok(Some("conformance ready".to_string()))
    }

    fn refresh(_opts: RefreshOpts) -> Result<RefreshReport, PluginError> {
        Ok(RefreshReport {
            inspected: 1,
            stale: 0,
            refreshed: 0,
            details: vec!["conformance refresh".to_string()],
        })
    }

    fn recall_for_file(path: String) -> Result<Vec<Hit>, PluginError> {
        Ok(vec![Hit {
            table: format!("{PREFIX}notes"),
            id: path,
            summary: Some("for-file".to_string()),
            score: 0.5,
        }])
    }
}

bindings::export!(Component with_types_in bindings);
