//! Tool profiles: which tools a server advertises and accepts.
//!
//! Every definition in `tools/list` is sent to the model, and a host that does
//! not cache tool definitions pays for all of them on every request. The core
//! profile keeps the tools agents call in a normal session; the full profile
//! adds everything else.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

/// Which tools an MCP server lists in `tools/list` and accepts in `tools/call`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolProfile {
    /// The session loop: resume, recall, look up code, and write knowledge
    /// back ([`CORE_TOOLS`]). The default.
    #[default]
    Core,
    /// Every built-in and Extension tool.
    Full,
}

/// The built-in tools served under [`ToolProfile::Core`].
///
/// Chosen from what agents call: `boot` to resume, `recall` / `search` / `get`
/// to read memory, `code_search` to find code, and the writes an agent makes
/// after each unit of work (`remember_decision`, `remember_error`,
/// `resolve_error`, `store`, `checkpoint`). A tool missing from a build (an
/// Extension compiled out) is simply absent.
pub const CORE_TOOLS: &[&str] = &[
    "boot",
    "recall",
    "search",
    "get",
    "code_search",
    "store",
    "remember_decision",
    "remember_error",
    "resolve_error",
    "checkpoint",
];

impl ToolProfile {
    /// The profile's name, as accepted by `axil mcp --tools`.
    pub fn as_str(self) -> &'static str {
        match self {
            ToolProfile::Core => "core",
            ToolProfile::Full => "full",
        }
    }

    /// Whether this profile serves `tool`.
    ///
    /// The core profile hides only Axil's own tools outside [`CORE_TOOLS`].
    /// A tool contributed by an installed plugin is not one of those, so it is
    /// served under every profile: installing a plugin is already an explicit
    /// opt-in to its tools.
    pub fn allows(self, tool: &str) -> bool {
        match self {
            ToolProfile::Full => true,
            ToolProfile::Core => CORE_TOOLS.contains(&tool) || !is_builtin_tool(tool),
        }
    }
}

impl fmt::Display for ToolProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ToolProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "core" => Ok(ToolProfile::Core),
            "full" => Ok(ToolProfile::Full),
            other => Err(format!(
                "unknown tool profile '{other}' (expected `core` or `full`)"
            )),
        }
    }
}

/// Whether `tool` is one of Axil's own tools: a built-in definition or a tool
/// of a compiled-in built-in Extension, whether or not config disables it.
fn is_builtin_tool(tool: &str) -> bool {
    static NAMES: OnceLock<HashSet<String>> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            let mut names: HashSet<String> = crate::tools::tool_definitions()
                .into_iter()
                .map(|t| t.name)
                .collect();
            for ext in axil_bundle::builtin_extensions_all() {
                if let Some(surface) = ext.mcp_tools() {
                    names.extend(surface.tools.into_iter().map(|t| t.name));
                }
            }
            names
        })
        .contains(tool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_profiles() {
        assert_eq!("core".parse::<ToolProfile>(), Ok(ToolProfile::Core));
        assert_eq!(" FULL ".parse::<ToolProfile>(), Ok(ToolProfile::Full));
        assert!("everything".parse::<ToolProfile>().is_err());
        for p in [ToolProfile::Core, ToolProfile::Full] {
            assert_eq!(p.as_str().parse::<ToolProfile>(), Ok(p));
        }
    }

    #[test]
    fn default_is_core() {
        assert_eq!(ToolProfile::default(), ToolProfile::Core);
    }

    #[test]
    fn core_tools_are_all_builtin() {
        for name in CORE_TOOLS {
            if cfg!(not(feature = "checkpoint")) && *name == "checkpoint" {
                continue;
            }
            assert!(
                is_builtin_tool(name),
                "`{name}` is in CORE_TOOLS but no built-in tool or Extension defines it"
            );
        }
    }

    #[test]
    fn core_hides_builtin_tools_outside_the_core_set() {
        for name in ["link", "list", "delete", "query", "lineage", "code_context"] {
            assert!(!ToolProfile::Core.allows(name), "core must hide `{name}`");
            assert!(ToolProfile::Full.allows(name), "full must serve `{name}`");
        }
        for name in CORE_TOOLS {
            assert!(ToolProfile::Core.allows(name), "core must serve `{name}`");
        }
    }

    #[test]
    fn plugin_tools_pass_through_core() {
        assert!(ToolProfile::Core.allows("my_plugin_tool"));
    }
}
