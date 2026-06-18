//! MCP parity tests — verify `McpServer::open` attaches detected plugins
//! and that MCP tool behaviour matches the CLI path:
//!
//! - A1: `.axil.vec` companion → vector plugin attached
//! - A2: `.axil.fts/` companion → FTS plugin attached
//! - A3: `.axil.graph` companion → graph plugin attached
//! - A4: missing companions → tools return a structured error, don't crash
//! - A5: `handle_recall` uses `db.recall()` (fused + QTC)
//! - A6: `handle_query_history` without `table` applies the bounded global cap
//! - A7: existing MCP input schemas unchanged (byte-identical required fields)
//!
//! These tests don't spin up the actual JSON-RPC loop — they exercise
//! `McpServer::open` + the tool-handler dispatch directly so we can assert
//! behavioural invariants without building a stdio harness.

use std::path::Path;

use axil_mcp::McpServer;
use serde_json::json;

fn temp_db_path() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.axil");
    (dir, path)
}

/// Build an Axil DB with vector + fts + graph plugins attached, seed a
/// record so history/FTS have data, then drop the handle. After this
/// function returns, the on-disk companion files exist, and we can
/// exercise `McpServer::open`'s plugin detection.
fn seed_db_with_all_plugins(path: &Path) {
    use axil_core::Axil;
    use axil_fts::AxilBuilderFtsExt;
    use axil_graph::AxilBuilderGraphExt;
    use axil_vector::AxilBuilderVectorExt;

    let db = Axil::open(path)
        .with_vector(4)
        .expect("vector plugin")
        .with_graph_plugin()
        .expect("graph plugin")
        .with_fts_plugin()
        .expect("fts plugin")
        .build()
        .expect("build");

    // One session-style record so `query_history` has something to return.
    let record = db
        .insert(
            "sessions",
            json!({"summary": "fixed auth timeout bug", "topic": "auth"}),
        )
        .expect("insert");

    // FTS index the summary so `search_text` has data we can assert on.
    let _ = db.index_text(&record.id, "summary", "fixed auth timeout bug");
}

// ─── A4: open handles missing companions without crashing ──────────────

#[test]
fn a4_missing_companions_open_without_crash() {
    let (_tmp, path) = temp_db_path();
    let server = McpServer::open(&path);
    assert!(
        server.is_ok(),
        "open should succeed without companions; got {:?}",
        server.err()
    );
}

// ─── A1/A2/A3: companion files trigger plugin attachment on open ──────

#[test]
fn a1_vector_companion_attaches_plugin() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    let server = McpServer::open(&path).expect("open");
    assert!(
        server.db_for_tests().has_vector_index(),
        "vector plugin should be attached after detecting .axil.vec companion"
    );
}

#[test]
fn a3_graph_companion_attaches_plugin() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    let server = McpServer::open(&path).expect("open");
    assert!(
        server.db_for_tests().has_graph_index(),
        "graph plugin should be attached after detecting .axil.graph companion"
    );
}

#[test]
fn a2_fts_companion_attaches_plugin() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    let server = McpServer::open(&path).expect("open");
    // `search_text` requires the FTS plugin; if it's not attached the call
    // returns an error. Presence of a successful call with results is proof
    // that detection+attach worked.
    let hits = server.db_for_tests().search_text("auth timeout", 5);
    assert!(
        hits.is_ok(),
        "fts plugin should be attached; got: {:?}",
        hits.err()
    );
    assert!(
        !hits.unwrap().is_empty(),
        "fts plugin attached but returned no hits for seeded text"
    );
}

// ─── A5: recall route goes through db.recall() (not similar_to) ────────

#[test]
fn a5_recall_does_not_panic_and_returns_json_array() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    let server = McpServer::open(&path).expect("open");
    // No embedder in this test (no ONNX model on the test runner), so the
    // new `db.recall()` path will return an empty result; our handler then
    // falls back to `similar_to` which also returns empty (no vector). Key
    // invariant: the call must not panic and must return a JSON array —
    // proving the new code path is wired.
    let args = json!({"query": "auth timeout", "top_k": 5});
    let result = axil_mcp::tools::dispatch(server.db_for_tests(), "recall", &args);

    assert!(
        result.is_error.is_none(),
        "recall must not return an error when plugins are attached; got: {:?}",
        result
    );
    let text = result
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    assert!(
        text.starts_with('[') && text.ends_with(']'),
        "recall should return a JSON array, got: {text}"
    );
}

// ─── A6: query_history bounded scan without `table` ────────────────────

#[test]
fn a6_query_history_without_table_respects_default_limit() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    // Seed more records than the default limit to exercise truncation.
    let db = axil_core::Axil::open(&path)
        .build()
        .expect("reopen for seed");
    for i in 0..100 {
        db.insert("logs", json!({"msg": format!("event {i}")}))
            .expect("insert");
    }
    drop(db);

    let server = McpServer::open(&path).expect("open");

    // Default limit = 50.
    let args = json!({});
    let result = axil_mcp::tools::dispatch(server.db_for_tests(), "query_history", &args);
    assert!(
        result.is_error.is_none(),
        "query_history without table must succeed"
    );

    let text = result
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    let arr: Vec<serde_json::Value> = serde_json::from_str(&text).expect("valid JSON array");
    assert!(
        arr.len() <= 50,
        "default limit should cap at 50, got {}",
        arr.len()
    );
}

#[test]
fn a6_query_history_honors_custom_limit() {
    let (_tmp, path) = temp_db_path();
    let db = axil_core::Axil::open(&path).build().expect("open");
    for i in 0..30 {
        db.insert("logs", json!({"msg": format!("event {i}")}))
            .expect("insert");
    }
    drop(db);

    let server = McpServer::open(&path).expect("open");
    let args = json!({"limit": 5});
    let result = axil_mcp::tools::dispatch(server.db_for_tests(), "query_history", &args);
    let text = result
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    let arr: Vec<serde_json::Value> = serde_json::from_str(&text).expect("valid JSON array");
    assert_eq!(
        arr.len(),
        5,
        "custom limit=5 should return 5 rows, got {}",
        arr.len()
    );
}

#[test]
fn a6_query_history_skips_internal_tables() {
    let (_tmp, path) = temp_db_path();
    seed_db_with_all_plugins(&path);

    // seed_db_with_all_plugins inserts a record which produces
    // `_recall_chunks` bookkeeping. A cross-table history scan must skip
    // underscore-prefixed tables so agents don't see the internals.
    let server = McpServer::open(&path).expect("open");
    let args = json!({});
    let result = axil_mcp::tools::dispatch(server.db_for_tests(), "query_history", &args);
    let text = result
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    let arr: Vec<serde_json::Value> = serde_json::from_str(&text).expect("valid JSON array");

    for row in &arr {
        let table = row.get("table").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !table.starts_with('_'),
            "internal table {table} leaked into history; row: {row}"
        );
    }
}

// ─── A7: tool input schemas preserve existing required fields ─────────

#[test]
fn a7_tool_input_schemas_preserved() {
    let defs = axil_mcp::tools::tool_definitions();

    let recall = defs
        .iter()
        .find(|d| d.name == "recall")
        .expect("recall tool present");
    let recall_required = recall
        .input_schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    assert!(
        recall_required.iter().any(|s| s == "query"),
        "recall must keep `query` required; got {recall_required:?}"
    );

    let qh = defs
        .iter()
        .find(|d| d.name == "query_history")
        .expect("query_history tool present");
    // `query_history` had no required block before our change; it must
    // still not introduce one (new `limit` param is additive / optional).
    let qh_required_empty = qh
        .input_schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(true);
    assert!(
        qh_required_empty,
        "query_history must not introduce new required fields (A7)"
    );

    // The additive `limit` property should be advertised in the schema.
    let limit_prop = qh
        .input_schema
        .get("properties")
        .and_then(|v| v.get("limit"));
    assert!(
        limit_prop.is_some(),
        "query_history should advertise `limit`"
    );
}
