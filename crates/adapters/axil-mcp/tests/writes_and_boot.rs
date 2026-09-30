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
fn boot_returns_schema_2_with_fixed_section_order() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let out = dispatch_json(&server, "boot", json!({"budget": 2000}));

    // Schema 2: rows are one-line strings (schema 1 carried whole records).
    assert_eq!(out["schema_version"], "2");
    assert_eq!(out["schema_version"], axil_core::BOOT_SCHEMA_VERSION);
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
fn boot_defaults_to_the_core_budget() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let out = dispatch_json(&server, "boot", json!({}));
    assert_eq!(out["token_budget"], axil_core::DEFAULT_TOKEN_BUDGET);
}

/// The MCP tool shares `Axil::boot`, so its rows are one-liners and its
/// budget holds inside the never-dropped sections, like `axil boot --schema v2`.
#[test]
fn boot_rows_are_one_liners_within_budget() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path).unwrap();
    let db = server.db_for_tests();
    let long = "plenty of words that cost tokens ".repeat(30);
    for i in 0..20 {
        db.insert(
            "decisions",
            json!({ "summary": format!("decision {i} {long}") }),
        )
        .unwrap();
        db.insert("errors", json!({ "error": format!("error {i} {long}") }))
            .unwrap();
    }
    db.insert(
        "errors",
        json!({ "error": "fixed long ago", "resolved": true }),
    )
    .unwrap();

    let out = dispatch_json(&server, "boot", json!({"budget": 600}));
    let text = serde_json::to_string(&out).unwrap();
    assert!(
        text.len().div_ceil(4) <= 600,
        "serialized {} bytes",
        text.len()
    );
    assert!(
        out["omitted_items"].as_u64().unwrap_or(0) > 0,
        "rows were cut: {out}"
    );

    let sections = out["sections"].as_array().unwrap();
    let failures = sections
        .iter()
        .find(|s| s["kind"] == "active_failures")
        .expect("failures are never dropped");
    let rows = failures["content"].as_array().unwrap();
    assert!(!rows.is_empty());
    for row in rows {
        let row = row.as_str().expect("rows are one-line strings");
        let fields: Vec<&str> = row.splitn(4, " · ").collect();
        assert_eq!(fields.len(), 4, "{row}");
        assert_eq!(fields[2], "open", "resolved errors are not failures: {row}");
    }
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
