//! Integration tests for the MCP server.
//!
//! Spawns `axil mcp` as a subprocess and communicates via JSON-RPC over stdin/stdout.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

/// Build path to the axil binary (debug).
fn axil_bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    // tests run from target/debug/deps — go up to target/debug
    path.pop();
    path.pop();
    path.push("axil");
    // On Windows the binary is `axil.exe`; EXE_EXTENSION is "" elsewhere.
    path.set_extension(std::env::consts::EXE_EXTENSION);
    path.to_string_lossy().to_string()
}

/// Pre-create a database with all plugins (vector, graph, FTS, timeseries).
fn init_db_with_all_engines(db_path: &Path) {
    let out = Command::new(axil_bin())
        .args(["init", db_path.to_str().unwrap()])
        .output()
        .expect("failed to run axil init");
    assert!(
        out.status.success(),
        "axil init failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Send a JSON-RPC request line and read one response line.
fn rpc_call(
    stdin: &mut impl Write,
    stdout: &mut impl BufRead,
    method: &str,
    id: Option<Value>,
    params: Option<Value>,
) -> Value {
    let mut req = json!({ "jsonrpc": "2.0", "method": method });
    if let Some(id) = id {
        req["id"] = id;
    }
    if let Some(p) = params {
        req["params"] = p;
    }
    let line = serde_json::to_string(&req).unwrap();
    writeln!(stdin, "{line}").unwrap();
    stdin.flush().unwrap();

    // Only read a response if we sent an id (notifications get no response).
    if req.get("id").is_some() {
        let mut resp_line = String::new();
        stdout.read_line(&mut resp_line).unwrap();
        serde_json::from_str(&resp_line).unwrap()
    } else {
        Value::Null
    }
}

/// Send a notification (no id, no response expected).
fn rpc_notify(stdin: &mut impl Write, method: &str, params: Option<Value>) {
    let mut req = json!({ "jsonrpc": "2.0", "method": method });
    if let Some(p) = params {
        req["params"] = p;
    }
    let line = serde_json::to_string(&req).unwrap();
    writeln!(stdin, "{line}").unwrap();
    stdin.flush().unwrap();
}

#[test]
fn mcp_full_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test.axil");

    // Pre-create DB with all plugins so MCP auto-detects them.
    init_db_with_all_engines(&db_path);

    let mut child = Command::new(axil_bin())
        .args(["--db", db_path.to_str().unwrap(), "mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn axil mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // ── initialize ──────────────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "initialize",
        Some(json!(1)),
        Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.1.0" }
        })),
    );
    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp["result"]["protocolVersion"].is_string());
    assert_eq!(resp["result"]["serverInfo"]["name"], "axil-mcp");

    // ── initialized notification (no response) ──────────────────────
    rpc_notify(&mut stdin, "initialized", None);

    // ── tools/list ──────────────────────────────────────────────────
    let resp = rpc_call(&mut stdin, &mut stdout, "tools/list", Some(json!(2)), None);
    let tools = resp["result"]["tools"].as_array().unwrap();
    let tool_names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(tool_names.contains(&"recall"), "missing recall tool");
    assert!(tool_names.contains(&"store"), "missing store tool");
    assert!(tool_names.contains(&"link"), "missing link tool");
    assert!(tool_names.contains(&"search"), "missing search tool");
    assert!(
        tool_names.contains(&"query_history"),
        "missing query_history tool"
    );
    assert!(tool_names.contains(&"get"), "missing get tool");
    assert!(tool_names.contains(&"list"), "missing list tool");
    assert!(tool_names.contains(&"delete"), "missing delete tool");

    // ── store a record ──────────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(3)),
        Some(json!({
            "name": "store",
            "arguments": {
                "table": "sessions",
                "data": {
                    "summary": "Fixed auth timeout bug",
                    "project": "my-app"
                }
            }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let stored: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    let record_id = stored["id"].as_str().unwrap().to_string();
    assert_eq!(stored["table"], "sessions");
    assert!(!record_id.is_empty());

    // ── get the record back ─────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(4)),
        Some(json!({
            "name": "get",
            "arguments": { "id": record_id }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let got: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert_eq!(got["id"], record_id);
    assert_eq!(got["data"]["summary"], "Fixed auth timeout bug");

    // ── list records in table ───────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(5)),
        Some(json!({
            "name": "list",
            "arguments": { "table": "sessions", "limit": 10 }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let listed: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert_eq!(listed.as_array().unwrap().len(), 1);

    // ── store a second record for linking ───────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(6)),
        Some(json!({
            "name": "store",
            "arguments": {
                "table": "files",
                "data": { "path": "src/auth.rs", "language": "rust" }
            }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let stored2: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    let file_id = stored2["id"].as_str().unwrap().to_string();

    // ── link the two records ────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(7)),
        Some(json!({
            "name": "link",
            "arguments": {
                "from": record_id,
                "edge_type": "modified",
                "to": file_id,
                "props": { "action": "bugfix" }
            }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let linked: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert!(linked["edge_id"].as_str().is_some());
    assert_eq!(linked["edge_type"], "modified");

    // ── query_history (all tables) ──────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(8)),
        Some(json!({
            "name": "query_history",
            "arguments": {}
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let history: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    // Should have at least our 2 records (sessions + files) plus the edge record
    assert!(history.as_array().unwrap().len() >= 2);

    // ── query_history with table filter ─────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(9)),
        Some(json!({
            "name": "query_history",
            "arguments": { "table": "sessions" }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let history: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 1);

    // ── delete a record ─────────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(10)),
        Some(json!({
            "name": "delete",
            "arguments": { "id": file_id }
        })),
    );
    let content = &resp["result"]["content"][0]["text"];
    let deleted: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
    assert_eq!(deleted["deleted"], true);

    // Verify it's gone.
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(11)),
        Some(json!({
            "name": "get",
            "arguments": { "id": file_id }
        })),
    );
    let is_error = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_error, "get of deleted record should error");

    // ── unknown method ──────────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "nonexistent/method",
        Some(json!(12)),
        None,
    );
    assert!(
        resp["error"].is_object(),
        "unknown method should return error"
    );
    assert_eq!(resp["error"]["code"], -32601);

    // ── unknown tool ────────────────────────────────────────────────
    let resp = rpc_call(
        &mut stdin,
        &mut stdout,
        "tools/call",
        Some(json!(13)),
        Some(json!({
            "name": "nonexistent_tool",
            "arguments": {}
        })),
    );
    let is_error = resp["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_error, "unknown tool should return error");

    // ── missing params ──────────────────────────────────────────────
    let resp = rpc_call(&mut stdin, &mut stdout, "tools/call", Some(json!(14)), None);
    assert!(
        resp["error"].is_object(),
        "missing params should return error"
    );

    // ── shutdown ────────────────────────────────────────────────────
    let resp = rpc_call(&mut stdin, &mut stdout, "shutdown", Some(json!(15)), None);
    assert_eq!(resp["result"], Value::Null);

    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "axil mcp should exit cleanly after shutdown"
    );
}

#[test]
fn mcp_parse_error() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test2.axil");

    let mut child = Command::new(axil_bin())
        .args(["--db", db_path.to_str().unwrap(), "mcp"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn axil mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Send invalid JSON.
    writeln!(stdin, "not valid json {{{{").unwrap();
    stdin.flush().unwrap();

    let mut resp_line = String::new();
    stdout.read_line(&mut resp_line).unwrap();
    let resp: Value = serde_json::from_str(&resp_line).unwrap();
    assert_eq!(resp["error"]["code"], -32700, "should be parse error");

    // Clean shutdown.
    drop(stdin);
    child.wait().unwrap();
}
