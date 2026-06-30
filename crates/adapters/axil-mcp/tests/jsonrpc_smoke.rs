//! Zero-client JSON-RPC smoke test.
//!
//! Drives the exact `initialize` / `tools/list` / `tools/call recall` frames
//! documented in `docs/src/agents/mcp.md` through [`McpServer::handle_frame`] —
//! the in-process equivalent of piping those newline-delimited lines into
//! `axil --db <DB> mcp`. This pins the documented copy-paste smoke test so a
//! protocol-shape regression (renamed `serverInfo.name`, dropped `recall`, a
//! changed tool-call envelope) fails the build instead of the docs.

use axil_mcp::McpServer;
use serde_json::Value;

// The three frames from the doc's smoke-test snippet, verbatim.
const INITIALIZE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
const TOOLS_LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
const RECALL_CALL: &str = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"query":"auth timeout","top_k":3}}}"#;

fn server() -> (tempfile::TempDir, McpServer) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("smoke.axil");
    let server = McpServer::open(&path).expect("open mcp server");
    (dir, server)
}

fn drive(server: &McpServer, frame: &str) -> Value {
    let line = server
        .handle_frame(frame)
        .expect("frame with an id must produce a response line");
    serde_json::from_str(&line).expect("response line must be valid JSON")
}

/// The documented handshake response — `serverInfo.name == "axil-mcp"` is the
/// "right binary on PATH" signal the doc tells users to look for.
#[test]
fn initialize_reports_axil_mcp_server_info() {
    let (_tmp, server) = server();
    let resp = drive(&server, INITIALIZE);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    let info = &resp["result"]["serverInfo"];
    assert_eq!(info["name"], "axil-mcp");
    assert!(
        info["version"].as_str().is_some_and(|v| !v.is_empty()),
        "serverInfo.version must be a non-empty string"
    );
    assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
}

/// `tools/list` must advertise `recall` — the tool the smoke test then calls.
#[test]
fn tools_list_advertises_recall() {
    let (_tmp, server) = server();
    let resp = drive(&server, TOOLS_LIST);
    let tools = resp["result"]["tools"]
        .as_array()
        .expect("tools/list returns a tools array");
    assert!(
        tools
            .iter()
            .any(|t| t["name"].as_str() == Some("recall")),
        "tools/list must include `recall`"
    );
}

/// `tools/call recall` against a fresh, empty DB returns the documented
/// tool-call envelope whose `text` is the JSON array `[]`.
#[test]
fn recall_call_returns_empty_array_envelope() {
    let (_tmp, server) = server();
    let resp = drive(&server, RECALL_CALL);
    assert_eq!(resp["id"], 3);
    let content = resp["result"]["content"]
        .as_array()
        .expect("tool-call result has a content array");
    let first = &content[0];
    assert_eq!(first["type"], "text");
    let hits: Value = serde_json::from_str(
        first["text"].as_str().expect("content text is a string"),
    )
    .expect("recall content text is a JSON array");
    assert_eq!(
        hits,
        Value::Array(vec![]),
        "recall over an empty DB yields []"
    );
}

/// The full documented sequence drives cleanly end to end through one server,
/// exactly as the piped one-liner would.
#[test]
fn documented_frame_sequence_round_trips() {
    let (_tmp, server) = server();
    for frame in [INITIALIZE, TOOLS_LIST, RECALL_CALL] {
        let line = server.handle_frame(frame).expect("each id'd frame replies");
        let resp: Value = serde_json::from_str(&line).expect("valid JSON line");
        assert!(
            resp.get("error").is_none(),
            "no frame should error: {resp}"
        );
        assert!(resp.get("result").is_some(), "every reply carries a result");
    }
}

/// A notification (no `id`) takes no reply — matching the stdio serve loop.
#[test]
fn notification_frame_yields_no_response() {
    let (_tmp, server) = server();
    let out = server.handle_frame(r#"{"jsonrpc":"2.0","method":"initialized"}"#);
    assert!(out.is_none(), "notifications produce no response line");
}

/// A malformed frame yields a JSON-RPC parse-error line, not a panic.
#[test]
fn malformed_frame_yields_parse_error() {
    let (_tmp, server) = server();
    let line = server
        .handle_frame("not json at all")
        .expect("parse error still produces a response line");
    let resp: Value = serde_json::from_str(&line).expect("parse-error line is valid JSON");
    assert_eq!(resp["error"]["code"], -32700);
}
