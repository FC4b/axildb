//! Integration tests for the Track B (intent-native writes) and Track C
//! (boot contract) MCP tools. Exercise the dispatch path end-to-end so
//! the tools' JSON shapes stay pinned.

use axil_mcp::McpServer;
use serde_json::json;

fn temp_db_path() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.axil");
    (dir, path)
}

fn dispatch_json(server: &McpServer, tool: &str, args: serde_json::Value) -> serde_json::Value {
    let result = axil_mcp::tools::dispatch(server.db_for_tests(), tool, &args);
    assert!(
        result.is_error.is_none(),
        "{tool} returned an error: {:?}",
        result
    );
    let text = result
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{tool} produced non-JSON output: {e} — {text}"))
}

// ─── Track B: intent-native writes over MCP ──────────────────────────

#[test]
fn remember_decision_returns_id_and_is_new() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let out = dispatch_json(
        &server,
        "remember_decision",
        json!({
            "summary": "adopt Axum for HTTP",
            "reason": "tokio-native, active maintainers",
        }),
    );
    assert!(out.get("id").and_then(|v| v.as_str()).is_some());
    assert_eq!(out.get("is_new").and_then(|v| v.as_bool()), Some(true));
}

#[test]
fn remember_decision_dedupes_on_agent_external_id() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let args = json!({
        "summary": "use JWT",
        "agent_id": "claude-1",
        "external_id": "dec-001",
    });
    let first = dispatch_json(&server, "remember_decision", args.clone());
    let second = dispatch_json(&server, "remember_decision", args);
    assert_eq!(first["id"], second["id"], "same (agent,ext) must dedupe");
    assert_eq!(second["is_new"], json!(false));
}

#[test]
fn remember_error_requires_error_field() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    // Missing `error` must produce a structured error, not a crash.
    let result = axil_mcp::tools::dispatch(
        server.db_for_tests(),
        "remember_error",
        &json!({"fix": "nothing"}),
    );
    assert_eq!(result.is_error, Some(true));
}

#[test]
fn set_preference_accepts_any_json_value() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();

    // string
    let r = dispatch_json(
        &server,
        "set_preference",
        json!({"key": "theme", "value": "dark"}),
    );
    assert_eq!(r["key"], "theme");
    assert_eq!(r["is_new"], json!(true));

    // number
    dispatch_json(
        &server,
        "set_preference",
        json!({"key": "retries", "value": 3}),
    );

    // object
    dispatch_json(
        &server,
        "set_preference",
        json!({"key": "limits", "value": {"max": 5, "min": 1}}),
    );
}

#[test]
fn close_session_is_idempotent_by_id() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let a = dispatch_json(
        &server,
        "close_session",
        json!({"id": "run-42", "summary": "done"}),
    );
    let b = dispatch_json(&server, "close_session", json!({"id": "run-42"}));
    assert_eq!(a["id"], b["id"]);
    assert_eq!(a["is_new"], json!(true));
    assert_eq!(b["is_new"], json!(false));
}

// ─── Track C: boot contract over MCP ─────────────────────────────────

#[test]
fn boot_returns_schema_v1_with_fixed_section_order() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let out = dispatch_json(&server, "boot", json!({"budget": 2000}));

    assert_eq!(out["schema_version"], "1");
    let sections = out["sections"].as_array().expect("sections array");
    let kinds: Vec<&str> = sections.iter().filter_map(|s| s["kind"].as_str()).collect();
    assert_eq!(
        kinds,
        vec![
            "current_scope",
            "constraints",
            "recent_decisions",
            "active_failures",
            "open_threads",
            "preferences",
            "confidence_notes",
        ]
    );
}

#[test]
fn boot_reports_token_budget_usage() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let out = dispatch_json(&server, "boot", json!({"budget": 500}));
    assert_eq!(out["token_budget"], 500);
    assert!(out["token_budget_used"].as_u64().is_some());
}

#[test]
fn tool_definitions_include_new_tools() {
    // The MCP server's tool listing must advertise every new tool; this
    // is what Cursor / Claude Code discovers via tools/list.
    let defs = axil_mcp::tools::tool_definitions();
    let names: std::collections::HashSet<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    for expected in [
        "remember_decision",
        "remember_error",
        "set_preference",
        "close_session",
        "boot",
    ] {
        assert!(
            names.contains(expected),
            "tool {expected} missing from listing"
        );
    }
}
