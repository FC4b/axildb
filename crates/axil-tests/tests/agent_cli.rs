//! Integration tests for the agent-first CLI.
//!
//! These tests invoke the `axil` binary and verify JSON output, exit codes,
//! env var support, piping, and end-to-end agent workflows.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Path to the built CLI binary (debug profile).
fn axil_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates/
    path.pop(); // workspace root
    path.push("target/debug/axil");
    // On Windows the binary is `axil.exe`; EXE_EXTENSION is "" elsewhere.
    path.set_extension(std::env::consts::EXE_EXTENSION);
    assert!(
        path.exists(),
        "axil binary not found at {}. Run `cargo build -p axildb` first.",
        path.display()
    );
    path
}

/// Create a temp directory and return its path + a database path within it.
fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::TempDir::new().unwrap();
    let db = dir.path().join("test.axil");
    (dir, db)
}

/// Run axil with the given args, returning (stdout, stderr, exit_code).
fn run_axil(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(axil_bin())
        .args(args)
        .output()
        .expect("failed to execute axil binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Run axil with AXIL_DB env var set.
fn run_axil_env(db_path: &str, args: &[&str]) -> (String, String, i32) {
    let output = Command::new(axil_bin())
        .env("AXIL_DB", db_path)
        .args(args)
        .output()
        .expect("failed to execute axil binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Run axil with stdin input.
fn run_axil_stdin(args: &[&str], stdin_data: &str) -> (String, String, i32) {
    use std::io::Write;
    let mut child = Command::new(axil_bin())
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn axil binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_data.as_bytes())
        .unwrap();

    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Parse stdout as JSON Value.
fn parse_json(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("failed to parse JSON: {e}\nstdout: {stdout}"))
}

// ─── Init ──────────────────────────────────────────────────────────────────

#[test]
fn test_init_creates_database() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    let (stdout, _stderr, code) = run_axil(&["init", db_str]);
    assert_eq!(code, 0, "init should succeed");

    let json = parse_json(&stdout);
    assert_eq!(json["created"], true);
    assert!(json["path"].as_str().unwrap().contains("test.axil"));
    assert!(json["features"].is_array());
}

// ─── Store + Get + Delete ──────────────────────────────────────────────────

#[test]
fn test_store_get_delete_lifecycle() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // Init
    run_axil(&["init", db_str]);

    // Store
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "store",
        "notes",
        r#"{"title":"Hello","body":"World"}"#,
    ]);
    assert_eq!(code, 0, "store should succeed");
    let stored = parse_json(&stdout);
    assert_eq!(stored["table"], "notes");
    let id = stored["id"].as_str().unwrap().to_string();
    assert!(!id.is_empty());
    assert!(stored["created_at"].is_string());

    // Get
    let (stdout, _, code) = run_axil(&["--db", db_str, "get", &id]);
    assert_eq!(code, 0, "get should succeed");
    let got = parse_json(&stdout);
    assert_eq!(got["id"], id);
    assert_eq!(got["data"]["title"], "Hello");
    assert_eq!(got["data"]["body"], "World");

    // Delete
    let (stdout, _, code) = run_axil(&["--db", db_str, "delete", &id]);
    assert_eq!(code, 0, "delete should succeed");
    let deleted = parse_json(&stdout);
    assert_eq!(deleted["deleted"], true);
    assert_eq!(deleted["id"], id);

    // Get after delete → exit code 2
    let (_, _, code) = run_axil(&["--db", db_str, "get", &id]);
    assert_eq!(code, 2, "get on deleted record should exit 2");
}

// ─── Update ────────────────────────────────────────────────────────────────

#[test]
fn test_update_record() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "notes", r#"{"v":1}"#]);
    let id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, code) = run_axil(&["--db", db_str, "update", &id, r#"{"v":2}"#]);
    assert_eq!(code, 0);
    let updated = parse_json(&stdout);
    assert_eq!(updated["data"]["v"], 2);

    // Verify via get
    let (stdout, _, _) = run_axil(&["--db", db_str, "get", &id]);
    let got = parse_json(&stdout);
    assert_eq!(got["data"]["v"], 2);
}

// ─── Recall --type facet filter ──────────────────────────────────────────────

#[test]
fn test_recall_type_filter() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    // Two context records with distinct `type` facets + one decision with no type.
    run_axil(&[
        "--db",
        db_str,
        "store",
        "context",
        r#"{"type":"architecture","summary":"zzqquux taxonomy design note"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "context",
        r#"{"type":"gotcha","summary":"zzqquux taxonomy pitfall note"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "decisions",
        r#"{"summary":"zzqquux taxonomy choice note"}"#,
    ]);

    // Baseline: all three are retrievable (--no-dedup keeps the count deterministic).
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "recall",
        "zzqquux taxonomy",
        "--top-k",
        "10",
        "--no-dedup",
        "--recall-format",
        "full",
    ]);
    assert_eq!(code, 0, "recall should succeed");
    assert_eq!(
        parse_json(&stdout).as_array().unwrap().len(),
        3,
        "baseline recall returns all three records"
    );

    // --type architecture → only the architecture context record; the gotcha
    // context and the type-less decision are excluded.
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "recall",
        "zzqquux taxonomy",
        "--top-k",
        "10",
        "--no-dedup",
        "--type",
        "architecture",
        "--recall-format",
        "full",
    ]);
    assert_eq!(code, 0);
    let arch = parse_json(&stdout);
    let arch = arch.as_array().unwrap();
    assert_eq!(arch.len(), 1, "only the architecture record matches");
    assert_eq!(arch[0]["data"]["type"], "architecture");
    assert_eq!(arch[0]["table"], "context");

    // Case-insensitive: ARCHITECTURE matches the same single record.
    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "recall",
        "zzqquux taxonomy",
        "--top-k",
        "10",
        "--no-dedup",
        "--type",
        "ARCHITECTURE",
    ]);
    assert_eq!(
        parse_json(&stdout).as_array().unwrap().len(),
        1,
        "match is case-insensitive"
    );

    // Unknown type → empty (no record carries it, and type-less records are excluded).
    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "recall",
        "zzqquux taxonomy",
        "--top-k",
        "10",
        "--no-dedup",
        "--type",
        "nope",
    ]);
    assert!(
        parse_json(&stdout).as_array().unwrap().is_empty(),
        "unknown type returns no hits"
    );
}

// ─── List ──────────────────────────────────────────────────────────────────

#[test]
fn test_list_records() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "items", r#"{"n":1}"#]);
    run_axil(&["--db", db_str, "store", "items", r#"{"n":2}"#]);
    run_axil(&["--db", db_str, "store", "items", r#"{"n":3}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "list", "items"]);
    assert_eq!(code, 0);
    let items: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(items.len(), 3);
}

#[test]
fn test_list_with_limit() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    for i in 0..5 {
        run_axil(&["--db", db_str, "store", "items", &format!(r#"{{"n":{i}}}"#)]);
    }

    let (stdout, _, code) = run_axil(&["--db", db_str, "list", "items", "--limit", "2"]);
    assert_eq!(code, 0);
    let items: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(items.len(), 2);
}

// ─── AXIL_DB env var ───────────────────────────────────────────────────────

#[test]
fn test_axil_db_env_var() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // Init with --db
    run_axil(&["init", db_str]);

    // Use AXIL_DB env var instead of --db
    let (stdout, _, code) = run_axil_env(db_str, &["store", "env_test", r#"{"from":"env"}"#]);
    assert_eq!(code, 0);
    let stored = parse_json(&stdout);
    assert_eq!(stored["table"], "env_test");

    // Info via env var
    let (stdout, _, code) = run_axil_env(db_str, &["info"]);
    assert_eq!(code, 0);
    let info = parse_json(&stdout);
    assert!(info["record_count"].as_u64().unwrap() >= 1);
}

// ─── Exit codes ────────────────────────────────────────────────────────────

#[test]
fn test_exit_code_not_found() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Get non-existent record → exit 2
    let (_, _, code) = run_axil(&["--db", db_str, "get", "01JQNV1XG00000000000000000"]);
    assert_eq!(code, 2);

    // Delete non-existent record → exit 2
    let (_, _, code) = run_axil(&["--db", db_str, "delete", "01JQNV1XG00000000000000000"]);
    assert_eq!(code, 2);
}

#[test]
fn test_exit_code_error_no_db() {
    // Missing --db and no AXIL_DB and no .axil/ in cwd → exit 1
    // Run from a temp dir with no .axil/ so auto-detect doesn't find one
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(axil_bin())
        .args(&["info"])
        .current_dir(tmp.path())
        .output()
        .expect("failed to execute axil binary");
    let code = output.status.code().unwrap_or(-1);
    assert_eq!(code, 1);
}

// ─── Piping: stdin ─────────────────────────────────────────────────────────

#[test]
fn test_stdin_pipe() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil_stdin(
        &["--db", db_str, "store", "piped", "-"],
        r#"{"piped":true,"value":42}"#,
    );
    assert_eq!(code, 0);
    let stored = parse_json(&stdout);
    assert_eq!(stored["table"], "piped");

    // Verify the stored record
    let id = stored["id"].as_str().unwrap();
    let (stdout, _, _) = run_axil(&["--db", db_str, "get", id]);
    let got = parse_json(&stdout);
    assert_eq!(got["data"]["piped"], true);
    assert_eq!(got["data"]["value"], 42);
}

// ─── --quiet mode ──────────────────────────────────────────────────────────

#[test]
fn test_quiet_mode() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Normal mode may have stderr output
    let (stdout, _, code) = run_axil(&["--db", db_str, "--quiet", "store", "q", r#"{"q":1}"#]);
    assert_eq!(code, 0);
    // stdout should still have JSON
    let stored = parse_json(&stdout);
    assert_eq!(stored["table"], "q");
}

// ─── --format pretty ──────────────────────────────────────────────────────

#[test]
fn test_pretty_format() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "--format",
        "pretty",
        "store",
        "fmt",
        r#"{"pretty":true}"#,
    ]);
    assert_eq!(code, 0);
    // Pretty output has newlines and indentation
    assert!(stdout.contains('\n'));
    assert!(stdout.contains("  "));
    // Still valid JSON
    let _ = parse_json(&stdout);
}

// ─── --jsonl mode ──────────────────────────────────────────────────────────

#[test]
fn test_jsonl_output() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "jl", r#"{"n":1}"#]);
    run_axil(&["--db", db_str, "store", "jl", r#"{"n":2}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "--jsonl", "list", "jl"]);
    assert_eq!(code, 0);

    // Each line should be valid JSON (not wrapped in array)
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(lines.len(), 2);
    for line in &lines {
        let v: Value = serde_json::from_str(line).unwrap();
        assert!(v["id"].is_string());
    }
}

// ─── Tables ────────────────────────────────────────────────────────────────

#[test]
fn test_tables_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "alpha", r#"{"a":1}"#]);
    run_axil(&["--db", db_str, "store", "beta", r#"{"b":1}"#]);
    run_axil(&["--db", db_str, "store", "beta", r#"{"b":2}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "tables"]);
    assert_eq!(code, 0);
    let tables: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert!(tables.len() >= 2);

    // Find beta table
    let beta = tables.iter().find(|t| t["name"] == "beta").unwrap();
    assert_eq!(beta["count"], 2);
}

// ─── Info ──────────────────────────────────────────────────────────────────

#[test]
fn test_info_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "info_test", r#"{"x":1}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "info"]);
    assert_eq!(code, 0);
    let info = parse_json(&stdout);
    assert!(info["path"].is_string());
    assert!(info["size_bytes"].is_number());
    assert!(info["record_count"].as_u64().unwrap() >= 1);
    assert!(info["tables"].is_array());
    assert!(info["features"].is_array());
}

// ─── Query with --where ────────────────────────────────────────────────────

#[test]
fn test_query_with_where() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "scores",
        r#"{"name":"alice","score":90}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "scores",
        r#"{"name":"bob","score":60}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "scores",
        r#"{"name":"carol","score":85}"#,
    ]);

    // Filter by score > 80
    let (stdout, _, code) = run_axil(&["--db", db_str, "query", "scores", "--where", "score>80"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 2);

    // All results should have score > 80
    for r in &results {
        assert!(r["data"]["score"].as_i64().unwrap() > 80);
    }
}

#[test]
fn test_query_multiple_where() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "items", r#"{"a":1,"b":10}"#]);
    run_axil(&["--db", db_str, "store", "items", r#"{"a":2,"b":20}"#]);
    run_axil(&["--db", db_str, "store", "items", r#"{"a":3,"b":30}"#]);

    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "query", "items", "--where", "a>1", "--where", "b<30",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["data"]["a"], 2);
}

#[test]
fn test_query_where_and_in_one_string() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    let (out_a, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "autopsies",
        r#"{"family":"meanrev","oos_sharpe":0.5,"trades":5}"#,
    ]);
    let id_a = parse_json(&out_a)["id"].as_str().unwrap().to_string();
    run_axil(&[
        "--db",
        db_str,
        "store",
        "autopsies",
        r#"{"family":"meanrev","oos_sharpe":0.1,"trades":100}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "autopsies",
        r#"{"family":"momentum","oos_sharpe":0.9,"trades":20}"#,
    ]);

    // One --where string carrying an AND, mixing a numeric predicate and a
    // single-quoted string predicate.
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "query",
        "autopsies",
        "--where",
        "oos_sharpe > 0.3 AND family = 'meanrev'",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    let ids: Vec<&str> = results.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec![id_a.as_str()], "exact id set must be [meanrev, sharpe 0.5]");
}

#[test]
fn test_query_where_numeric_not_lexicographic() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "autopsies", r#"{"trades":5}"#]);
    run_axil(&["--db", db_str, "store", "autopsies", r#"{"trades":100}"#]);

    // trades < 30 must compare numerically (5 matches, 100 does not) — a
    // lexicographic string compare would wrongly match "100" < "30".
    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "query", "autopsies", "--where", "trades < 30",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1, "trades<30 must match 5, not 100");
    assert_eq!(results[0]["data"]["trades"], 5);
}

#[test]
fn test_query_where_contains() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "notes",
        r#"{"summary":"auth timeout bug"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "notes",
        r#"{"summary":"deploy pipeline"}"#,
    ]);

    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "query",
        "notes",
        "--where",
        "summary contains 'timeout'",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["data"]["summary"], "auth timeout bug");
}

// ─── Aggregations (agg) ─────────────────────────────────────────────────────

#[test]
fn test_agg_count_group_by_kill_reason() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"kill_reason":"drawdown"}"#,
    ]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"kill_reason":"drawdown"}"#,
    ]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"kill_reason":"fees"}"#,
    ]);

    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "agg",
        "autopsies",
        "--count",
        "--group-by",
        "kill_reason",
    ]);
    assert_eq!(code, 0, "agg should succeed; stderr may explain");
    let env = parse_json(&stdout);
    assert_eq!(env["table"], "autopsies");
    assert_eq!(env["group_by"], "kill_reason");
    assert_eq!(env["total_rows"], 3);
    let groups = env["groups"].as_array().unwrap();
    // Sorted by group key: "drawdown" (2) before "fees" (1).
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0]["group"], "drawdown");
    assert_eq!(groups[0]["count"], 2);
    assert_eq!(groups[1]["group"], "fees");
    assert_eq!(groups[1]["count"], 1);
}

#[test]
fn test_agg_avg_decay_per_family() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"family":"meanrev","decay":2.0}"#,
    ]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"family":"meanrev","decay":4.0}"#,
    ]);
    run_axil(&[
        "--db", db_str, "store", "autopsies", r#"{"family":"momentum","decay":9.0}"#,
    ]);

    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "agg", "autopsies", "--avg", "decay", "--group-by", "family",
    ]);
    assert_eq!(code, 0);
    let env = parse_json(&stdout);
    let groups = env["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0]["group"], "meanrev");
    assert_eq!(groups[0]["avg_decay"], 3.0);
    assert_eq!(groups[0]["skipped"], 0);
    assert_eq!(groups[1]["group"], "momentum");
    assert_eq!(groups[1]["avg_decay"], 9.0);
}

#[test]
fn test_agg_include_archived_changes_count() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "autopsies", r#"{"family":"meanrev"}"#]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "autopsies",
        r#"{"family":"meanrev","_archived":true}"#,
    ]);

    // Default excludes archived.
    let (stdout, _, _) = run_axil(&["--db", db_str, "agg", "autopsies", "--count"]);
    let env = parse_json(&stdout);
    assert_eq!(env["total_rows"], 1);

    // --include-archived counts the discarded trial too (deflated-Sharpe math).
    let (stdout, _, _) = run_axil(&[
        "--db", db_str, "agg", "autopsies", "--count", "--include-archived",
    ]);
    let env = parse_json(&stdout);
    assert_eq!(env["total_rows"], 2);
}

// ─── Link + Neighbors + Traverse (graph) ───────────────────────────────────

#[test]
fn test_link_neighbors_traverse() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Create records
    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "people", r#"{"name":"Alice"}"#]);
    let alice_id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "people", r#"{"name":"Bob"}"#]);
    let bob_id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "projects", r#"{"name":"Axil"}"#]);
    let project_id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    // Link Alice -> works_on -> Axil
    let (stdout, _, code) = run_axil(&["--db", db_str, "link", &alice_id, "works_on", &project_id]);
    assert_eq!(code, 0);
    let link = parse_json(&stdout);
    assert!(link["edge_id"].is_string());
    assert_eq!(link["from"], alice_id);
    assert_eq!(link["to"], project_id);

    // Link Bob -> works_on -> Axil
    run_axil(&["--db", db_str, "link", &bob_id, "works_on", &project_id]);

    // Neighbors of Axil (incoming works_on)
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "neighbors",
        &project_id,
        "--type",
        "works_on",
        "--direction",
        "in",
    ]);
    assert_eq!(code, 0);
    let neighbors: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(neighbors.len(), 2);

    // Traverse: Alice ->works_on-> (should reach Axil)
    // Use -- to prevent clap from treating ->works_on as a flag.
    let (stdout, _, code) = run_axil(&["--db", db_str, "traverse", &alice_id, "->works_on"]);
    assert_eq!(code, 0);
    let traversed: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(traversed.len(), 1);
    assert_eq!(traversed[0]["data"]["name"], "Axil");
}

// ─── Unlink ────────────────────────────────────────────────────────────────

#[test]
fn test_unlink() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "a", r#"{"x":1}"#]);
    let id_a = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "b", r#"{"x":2}"#]);
    let id_b = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "link", &id_a, "rel", &id_b]);
    let edge_id = parse_json(&stdout)["edge_id"].as_str().unwrap().to_string();

    let (stdout, _, code) = run_axil(&["--db", db_str, "unlink", &edge_id]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert_eq!(result["deleted"], true);
}

// ─── Session workflow ──────────────────────────────────────────────────────

#[test]
fn test_session_workflow() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Start session
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "session",
        "start",
        "--meta",
        r#"{"project":"test"}"#,
    ]);
    assert_eq!(code, 0);
    let session = parse_json(&stdout);
    let session_id = session["session_id"].as_str().unwrap().to_string();
    assert!(session["started_at"].is_string());

    // Log a record to the session
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "session",
        "log",
        &session_id,
        "decisions",
        r#"{"decision":"use JWT"}"#,
    ]);
    assert_eq!(code, 0);
    let logged = parse_json(&stdout);
    assert_eq!(logged["session_id"], session_id);
    assert!(logged["linked"].as_bool().unwrap());

    // Log another
    run_axil(&[
        "--db",
        db_str,
        "session",
        "log",
        &session_id,
        "notes",
        r#"{"note":"JWT is stateless"}"#,
    ]);

    // List active sessions
    let (stdout, _, code) = run_axil(&["--db", db_str, "session", "list", "--active"]);
    assert_eq!(code, 0);
    let sessions: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["status"], "active");

    // Session history
    let (stdout, _, code) = run_axil(&["--db", db_str, "session", "history", &session_id]);
    assert_eq!(code, 0);
    let history: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(history.len(), 2);

    // End session
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "session",
        "end",
        &session_id,
        "--summary",
        "Chose JWT auth",
    ]);
    assert_eq!(code, 0);
    let ended = parse_json(&stdout);
    assert_eq!(ended["session_id"], session_id);
    assert!(ended["ended_at"].is_string());
    assert_eq!(ended["records"], 2);

    // List all sessions (should show ended)
    let (stdout, _, _) = run_axil(&["--db", db_str, "session", "list"]);
    let sessions: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(sessions[0]["status"], "ended");

    // List active sessions (should be empty)
    let (stdout, _, _) = run_axil(&["--db", db_str, "session", "list", "--active"]);
    let sessions: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert!(sessions.is_empty());
}

// ─── FTS ───────────────────────────────────────────────────────────────────

#[test]
fn test_fts_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Store records (FTS auto-indexes string fields on insert)
    run_axil(&[
        "--db",
        db_str,
        "store",
        "docs",
        r#"{"title":"Rust programming","body":"Systems language"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "docs",
        r#"{"title":"Python scripting","body":"Dynamic language"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "docs",
        r#"{"title":"Rust web server","body":"Actix and Axum"}"#,
    ]);

    // Search for "rust"
    let (stdout, _, code) = run_axil(&["--db", db_str, "fts", "rust"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 2);
    // Results should have score fields
    for r in &results {
        assert!(r["score"].is_number());
        assert!(r["id"].is_string());
        assert!(r["data"].is_object());
    }
}

// ─── Combined workflow ─────────────────────────────────────────────────────

#[test]
fn test_combined_workflow() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // 1. Init
    let (_, _, code) = run_axil(&["init", db_str]);
    assert_eq!(code, 0);

    // 2. Store records
    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "decisions",
        r#"{"summary":"Use JWT for auth","reason":"Stateless"}"#,
    ]);
    let decision_id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "research",
        r#"{"topic":"Auth methods","finding":"JWT is lightweight"}"#,
    ]);
    let research_id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    // 3. Link
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "link",
        &decision_id,
        "informed_by",
        &research_id,
        "--props",
        r#"{"confidence":"high"}"#,
    ]);
    assert_eq!(code, 0);
    let link = parse_json(&stdout);
    assert!(link["edge_id"].is_string());

    // 4. Traverse
    let (stdout, _, code) = run_axil(&["--db", db_str, "traverse", &decision_id, "->informed_by"]);
    assert_eq!(code, 0);
    let traversed: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(traversed.len(), 1);
    assert_eq!(traversed[0]["data"]["topic"], "Auth methods");

    // 5. FTS search
    let (stdout, _, code) = run_axil(&["--db", db_str, "fts", "JWT auth"]);
    assert_eq!(code, 0);
    let fts: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert!(!fts.is_empty());

    // 6. Info
    let (stdout, _, code) = run_axil(&["--db", db_str, "info"]);
    assert_eq!(code, 0);
    let info = parse_json(&stdout);
    assert!(info["record_count"].as_u64().unwrap() >= 2);
}

// ─── Compact / Heal ────────────────────────────────────────────────────────

#[test]
fn test_compact_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "x", r#"{"n":1}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "compact"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    // CompactReport has purged_expired and purged_superseded fields
    assert!(result["purged_expired"].is_number());
    assert!(result["purged_superseded"].is_number());
    assert!(result["cleaned_orphaned_edges"].is_number());
}

#[test]
fn test_heal_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "heal"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    // SelfHealReport has healed, actions, duration_ms
    assert!(result["healed"].is_boolean());
    assert!(result["actions"].is_array());
    assert!(result["duration_ms"].is_number());
}

#[test]
fn test_health_report_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "data", r#"{"x":1}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "health-report"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(result["overall_health"].is_string());
    assert!(result["score"].is_number());
    assert!(result["sections"].is_object());
    assert!(result["recommendations"].is_array());
}

#[test]
fn test_health_report_brief() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "health-report", "--brief"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(result["overall_health"].is_string());
    assert!(result["score"].is_number());
    assert!(result["summary"].is_string());
    // Brief should NOT have sections
    assert!(result["sections"].is_null());
}

#[test]
fn test_snapshot_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "data", r#"{"x":1}"#]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "snapshot"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(result["timestamp"].is_string());
    assert!(result["record_count"].is_number());
    assert!(result["file_size_bytes"].is_number());
    assert!(result["live_ratio"].is_number());
}

#[test]
fn test_trends_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "trends", "--days", "7"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert_eq!(result["period"].as_str().unwrap(), "7d");
    assert!(result["snapshots"].is_number());
    assert!(result["trends"].is_object());
}

#[test]
fn test_heal_dry_run() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "data",
        r#"{"x":1,"valid_until":"2020-01-01T00:00:00Z"}"#,
    ]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "heal", "--dry-run"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(!result["healed"].as_bool().unwrap());
    assert!(result["actions"].is_array());
}

#[test]
fn test_heal_compact_flag() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "heal", "--compact"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(result["healed"].is_boolean());
    assert!(result["actions"].is_array());
}

// ── Codex P1: --dry-run must be honored with targeted flags ────────────────

#[test]
fn test_heal_compact_dry_run_does_not_delete() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    // Store an expired record
    run_axil(&[
        "--db",
        db_str,
        "store",
        "data",
        r#"{"x":1,"valid_until":"2020-01-01T00:00:00Z"}"#,
    ]);
    // Store a live record
    run_axil(&["--db", db_str, "store", "data", r#"{"live":true}"#]);

    // --compact --dry-run should NOT delete the expired record
    let (stdout, _, code) = run_axil(&["--db", db_str, "heal", "--compact", "--dry-run"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(
        !result["healed"].as_bool().unwrap(),
        "--compact --dry-run should report healed=false"
    );

    // Verify the expired record still exists (was NOT deleted)
    let (stdout, _, _) = run_axil(&["--db", db_str, "list", "data"]);
    let list = parse_json(&stdout);
    assert_eq!(
        list.as_array().unwrap().len(),
        2,
        "both records should still exist after --compact --dry-run"
    );
}

#[test]
fn test_heal_reindex_dry_run_does_not_rebuild() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "data", r#"{"x":1}"#]);

    // --reindex --dry-run should NOT rebuild
    let (stdout, _, code) = run_axil(&["--db", db_str, "heal", "--reindex", "--dry-run"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert!(!result["healed"].as_bool().unwrap());
}

// ── Codex P1: --orphans must not delete expired/superseded records ────────

#[test]
fn test_heal_orphans_does_not_purge_expired() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    // Store expired records
    run_axil(&[
        "--db",
        db_str,
        "store",
        "data",
        r#"{"x":1,"valid_until":"2020-01-01T00:00:00Z"}"#,
    ]);
    run_axil(&[
        "--db",
        db_str,
        "store",
        "data",
        r#"{"x":2,"valid_until":"2020-01-01T00:00:00Z"}"#,
    ]);
    // Store a live record
    run_axil(&["--db", db_str, "store", "data", r#"{"live":true}"#]);

    // --orphans should only clean orphaned edges/vectors, NOT purge expired records
    let (stdout, _, code) = run_axil(&["--db", db_str, "heal", "--orphans"]);
    assert_eq!(code, 0);

    // All 3 records should still exist — expired ones were NOT purged
    let (stdout, _, _) = run_axil(&["--db", db_str, "list", "data"]);
    let list = parse_json(&stdout);
    assert_eq!(
        list.as_array().unwrap().len(),
        3,
        "--orphans must not delete expired records; only --compact should"
    );
}

// ─── JSON output validation ────────────────────────────────────────────────

#[test]
fn test_all_outputs_are_valid_json() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // Init first
    let (stdout, _, code) = run_axil(&["init", db_str]);
    assert_eq!(code, 0, "init failed");
    let _ = parse_json(&stdout);

    // Store two records for later commands to use
    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "t",
        r#"{"k":"v","text":"hello world"}"#,
    ]);
    let id1 = parse_json(&stdout)["id"].as_str().unwrap().to_string();
    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "t",
        r#"{"k":"v2","text":"goodbye"}"#,
    ]);
    let id2 = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    // Link the two records
    let (stdout, _, code) = run_axil(&["--db", db_str, "link", &id1, "rel", &id2]);
    assert_eq!(code, 0, "link failed");
    let edge_id = parse_json(&stdout)["edge_id"].as_str().unwrap().to_string();

    // Session commands
    let (stdout, _, _) = run_axil(&["--db", db_str, "session", "start"]);
    let sess_id = parse_json(&stdout)["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (stdout, _, code) = run_axil(&[
        "--db",
        db_str,
        "session",
        "log",
        &sess_id,
        "t",
        r#"{"s":1}"#,
    ]);
    assert_eq!(code, 0, "session log failed");
    let _ = parse_json(&stdout);

    // Every command that returns JSON — verify it parses
    let all_commands: Vec<(&str, Vec<&str>)> = vec![
        ("get", vec!["--db", db_str, "get", &id1]),
        (
            "update",
            vec!["--db", db_str, "update", &id1, r#"{"k":"updated"}"#],
        ),
        ("list", vec!["--db", db_str, "list", "t"]),
        ("tables", vec!["--db", db_str, "tables"]),
        ("info", vec!["--db", db_str, "info"]),
        ("query", vec!["--db", db_str, "query", "t"]),
        (
            "neighbors",
            vec!["--db", db_str, "neighbors", &id1, "--direction", "out"],
        ),
        ("traverse", vec!["--db", db_str, "traverse", &id1, "->rel"]),
        ("edges", vec!["--db", db_str, "edges", &id1]),
        ("fts", vec!["--db", db_str, "fts", "hello"]),
        ("since", vec!["--db", db_str, "since", "1h"]),
        ("timeline", vec!["--db", db_str, "timeline", "--limit", "5"]),
        ("diff", vec!["--db", db_str, "diff", "--since", "1h"]),
        ("activity", vec!["--db", db_str, "activity", "--days", "1"]),
        ("heal", vec!["--db", db_str, "heal"]),
        ("compact", vec!["--db", db_str, "compact"]),
        ("health-report", vec!["--db", db_str, "health-report"]),
        (
            "health-report-brief",
            vec!["--db", db_str, "health-report", "--brief"],
        ),
        ("trends", vec!["--db", db_str, "trends", "--days", "7"]),
        ("snapshot", vec!["--db", db_str, "snapshot"]),
        ("session list", vec!["--db", db_str, "session", "list"]),
        (
            "session history",
            vec!["--db", db_str, "session", "history", &sess_id],
        ),
        (
            "session end",
            vec![
                "--db",
                db_str,
                "session",
                "end",
                &sess_id,
                "--summary",
                "test",
            ],
        ),
        ("unlink", vec!["--db", db_str, "unlink", &edge_id]),
        ("delete", vec!["--db", db_str, "delete", &id2]),
    ];

    for (name, cmd) in &all_commands {
        let (stdout, stderr, code) = run_axil(cmd);
        assert_eq!(code, 0, "command '{name}' failed (exit {code}): {stderr}");
        assert!(
            !stdout.trim().is_empty(),
            "command '{name}' produced no output"
        );
        let _ = parse_json(&stdout);
    }
}

// ─── --format table ────────────────────────────────────────────────────────

#[test]
fn test_table_format_single() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "t", r#"{"x":1}"#]);
    let id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    // Single object: key-value format
    let (stdout, _, code) = run_axil(&["--db", db_str, "--format", "table", "get", &id]);
    assert_eq!(code, 0);
    assert!(stdout.contains("id"));
    assert!(stdout.contains("table"));
    assert!(stdout.contains("data"));
    // Not valid JSON — it's a human-readable table
    assert!(serde_json::from_str::<Value>(stdout.trim()).is_err());
}

#[test]
fn test_table_format_array() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "t", r#"{"x":1}"#]);
    run_axil(&["--db", db_str, "store", "t", r#"{"x":2}"#]);

    // Array: aligned table with header + separator + rows
    let (stdout, _, code) = run_axil(&["--db", db_str, "--format", "table", "list", "t"]);
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    // Header + separator + 2 data rows = 4 lines
    assert_eq!(lines.len(), 4);
    assert!(lines[0].contains("id"));
    assert!(lines[1].contains("---"));
}

// ─── Add-vector + Search-vector ─────────────────────────────────────────────

/// Build a 384-element vector literal — `1.0` at `hot`, `0.0` elsewhere.
///
/// The test uses the default 384-dim index (not a tiny custom dim): a
/// DB created with `--vector-dims N` where N != 384 cannot be opened by
/// `axil store`, which auto-attaches the bundled 384-dim embedder.
fn unit_vec_384(hot: usize) -> String {
    let mut parts = vec!["0.0"; 384];
    parts[hot] = "1.0";
    format!("[{}]", parts.join(", "))
}

#[test]
fn test_add_vector_and_search_vector() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // Default 384-dim index — matches the bundled embedder, so the
    // `store` calls below can open the DB.
    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "vecs", r#"{"label":"alpha"}"#]);
    let id1 = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "vecs", r#"{"label":"beta"}"#]);
    let id2 = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let v1 = unit_vec_384(0);
    let v2 = unit_vec_384(1);

    // Add explicit vectors.
    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "add-vector", &id1, &v1, "--dimensions", "384",
    ]);
    assert_eq!(code, 0, "add-vector failed");
    let result = parse_json(&stdout);
    assert_eq!(result["added"], true);
    assert_eq!(result["dimensions"], 384);

    let (_, _, code) = run_axil(&[
        "--db", db_str, "add-vector", &id2, &v2, "--dimensions", "384",
    ]);
    assert_eq!(code, 0);

    // Search with the exact vector of id1 — it should rank first.
    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "search-vector", &v1, "--top-k", "2", "--dimensions", "384",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0]["score"].as_f64().unwrap() > 0.0);
    assert!(results[0]["id"].is_string());
}

// ─── store --vector + similar (raw vectors first-class) ─────────────────────

#[test]
fn test_store_vector_roundtrip_and_similar_default_space() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // The default space coexists with the bundled 384-dim embedder, so a raw
    // vector attached there must be 384-dim (arbitrary dims → use `--space`).
    run_axil(&["init", db_str]);
    let v0 = unit_vec_384(0);
    let v1 = unit_vec_384(1);

    let (stdout, _, code) =
        run_axil(&["--db", db_str, "store", "v", r#"{"n":"a"}"#, "--vector", &v0]);
    assert_eq!(code, 0, "store --vector failed: {stdout}");
    let a = parse_json(&stdout);
    assert_eq!(a["vector_dims"], 384);
    assert!(a.get("space").is_none(), "default space must not label a space");

    let (_, _, code) =
        run_axil(&["--db", db_str, "store", "v", r#"{"n":"b"}"#, "--vector", &v1]);
    assert_eq!(code, 0);

    // `similar --vector` searches the default space; the exact match ranks first.
    let (stdout, _, code) =
        run_axil(&["--db", db_str, "similar", "--vector", &v0, "--top-k", "2"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0]["score"].as_f64().unwrap() > 0.99);
    assert_eq!(results[0]["id"].as_str().unwrap(), a["id"].as_str().unwrap());
}

#[test]
fn test_store_vector_conflicts_with_embed() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    let (_, stderr, code) = run_axil(&[
        "--db", db_str, "store", "v", r#"{"n":"a"}"#, "--vector", "[1,0,0]", "--embed", "n",
    ]);
    assert_eq!(code, 2, "clap conflict must exit 2");
    assert!(
        stderr.contains("cannot be used with"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn test_similar_id_threshold_named_space_returns_exact_twin() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // A near-duplicate pair (cosine ≈ 0.97) + an orthogonal decoy, all in the
    // 8-dim `fp` space (no collision with any text-embedding space).
    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "store", "fp", r#"{"n":"a"}"#,
        "--vector", "[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]", "--space", "fp",
    ]);
    assert_eq!(code, 0, "store --vector --space failed: {stdout}");
    let out_a = parse_json(&stdout);
    assert_eq!(out_a["space"], "fp");
    assert_eq!(out_a["vector_dims"], 8);
    let id_a = out_a["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&[
        "--db", db_str, "store", "fp", r#"{"n":"b"}"#,
        "--vector", "[1.0,0.25,0.0,0.0,0.0,0.0,0.0,0.0]", "--space", "fp",
    ]);
    let id_b = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    run_axil(&[
        "--db", db_str, "store", "fp", r#"{"n":"c"}"#,
        "--vector", "[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]", "--space", "fp",
    ]);

    // similar --id A --threshold 0.9 returns exactly the twin B (self excluded,
    // decoy below threshold).
    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "similar", "--id", &id_a, "--threshold", "0.9", "--space", "fp",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1, "threshold 0.9 should return only the twin");
    assert_eq!(results[0]["id"].as_str().unwrap(), id_b);
    assert!(results[0]["score"].as_f64().unwrap() > 0.9);
}

#[test]
fn test_add_vector_and_search_vector_named_space() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    // init makes a 384-dim default store; a 4-dim `fp` space must not collide.
    run_axil(&["init", db_str]);
    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "fp", r#"{"n":"x"}"#]);
    let id_x = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "add-vector", &id_x, "[1.0,0.0,0.0,0.0]", "--space", "fp",
    ]);
    assert_eq!(code, 0, "add-vector --space failed: {stdout}");
    let added = parse_json(&stdout);
    assert_eq!(added["added"], true);
    assert_eq!(added["space"], "fp");
    assert_eq!(added["dimensions"], 4);

    let (stdout, _, code) = run_axil(&[
        "--db", db_str, "search-vector", "[1.0,0.0,0.0,0.0]", "--space", "fp", "--top-k", "3",
    ]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"].as_str().unwrap(), id_x);
}

// ─── Index-text ────────────────────────────────────────────────────────────

#[test]
fn test_index_text_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&[
        "--db",
        db_str,
        "store",
        "docs",
        r#"{"body":"manual indexing test"}"#,
    ]);
    let id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    // Manually index a field
    let (stdout, _, code) = run_axil(&["--db", db_str, "index-text", &id, "body"]);
    assert_eq!(code, 0);
    let result = parse_json(&stdout);
    assert_eq!(result["indexed"], true);
    assert_eq!(result["field"], "body");

    // Search for it
    let (stdout, _, code) = run_axil(&["--db", db_str, "fts", "manual indexing"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert!(!results.is_empty());
}

// ─── Time-series commands ──────────────────────────────────────────────────

#[test]
fn test_since_timeline_diff_activity() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);
    run_axil(&["--db", db_str, "store", "ts", r#"{"n":1}"#]);
    run_axil(&["--db", db_str, "store", "ts", r#"{"n":2}"#]);

    // Since
    let (stdout, _, code) = run_axil(&["--db", db_str, "since", "1h"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 2);

    // Timeline
    let (stdout, _, code) = run_axil(&["--db", db_str, "timeline", "--limit", "1"]);
    assert_eq!(code, 0);
    let results: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(results.len(), 1);

    // Diff
    let (stdout, _, code) = run_axil(&["--db", db_str, "diff", "--since", "1h"]);
    assert_eq!(code, 0);
    let diff = parse_json(&stdout);
    assert!(diff["created"].as_u64().unwrap() >= 2);

    // Activity
    let (stdout, _, code) = run_axil(&["--db", db_str, "activity", "--days", "1"]);
    assert_eq!(code, 0);
    let _ = parse_json(&stdout);
}

// ─── Edges command ─────────────────────────────────────────────────────────

#[test]
fn test_edges_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "a", r#"{"x":1}"#]);
    let id_a = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "b", r#"{"x":2}"#]);
    let id_b = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    run_axil(&["--db", db_str, "link", &id_a, "knows", &id_b]);

    let (stdout, _, code) = run_axil(&["--db", db_str, "edges", &id_a]);
    assert_eq!(code, 0);
    let edges: Vec<Value> = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["edge_type"], "knows");
    assert!(edges[0]["edge_id"].is_string());
}

// ─── Performance benchmarks ────────────────────────────────────────────────

#[test]
#[ignore = "wall-clock perf assertion — unreliable in debug builds and on loaded machines; run with `--release -- --ignored`"]
fn test_perf_cli_cold_start() {
    // Target: CLI cold start < 500ms (hard limit from spec).
    let start = std::time::Instant::now();
    let (_, _, code) = run_axil(&["--help"]);
    let elapsed = start.elapsed();

    assert_eq!(code, 0);
    assert!(
        elapsed.as_millis() < 500,
        "CLI cold start took {}ms (hard limit: 500ms)",
        elapsed.as_millis()
    );
}

#[test]
#[ignore = "wall-clock perf assertion — unreliable in debug builds and on loaded machines; run with `--release -- --ignored`"]
fn test_perf_store_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    // Target: any CLI command < 500ms (hard limit).
    let start = std::time::Instant::now();
    let (_, _, code) = run_axil(&["--db", db_str, "store", "perf", r#"{"bench":true}"#]);
    let elapsed = start.elapsed();

    assert_eq!(code, 0);
    assert!(
        elapsed.as_millis() < 500,
        "store command took {}ms (hard limit: 500ms)",
        elapsed.as_millis()
    );
}

#[test]
#[ignore = "wall-clock perf assertion — unreliable in debug builds and on loaded machines; run with `--release -- --ignored`"]
fn test_perf_get_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    let (stdout, _, _) = run_axil(&["--db", db_str, "store", "perf", r#"{"bench":true}"#]);
    let id = parse_json(&stdout)["id"].as_str().unwrap().to_string();

    let start = std::time::Instant::now();
    let (_, _, code) = run_axil(&["--db", db_str, "get", &id]);
    let elapsed = start.elapsed();

    assert_eq!(code, 0);
    assert!(
        elapsed.as_millis() < 500,
        "get command took {}ms (hard limit: 500ms)",
        elapsed.as_millis()
    );
}

#[test]
#[ignore = "wall-clock perf assertion — unreliable in debug builds and on loaded machines; run with `--release -- --ignored`"]
fn test_perf_info_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    let start = std::time::Instant::now();
    let (_, _, code) = run_axil(&["--db", db_str, "info"]);
    let elapsed = start.elapsed();

    assert_eq!(code, 0);
    assert!(
        elapsed.as_millis() < 1000,
        "info command took {}ms (hard limit: 1000ms)",
        elapsed.as_millis()
    );
}

#[test]
#[ignore = "wall-clock perf assertion — unreliable in debug builds and on loaded machines; run with `--release -- --ignored`"]
fn test_perf_list_command() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    // Insert 100 records.
    for i in 0..100 {
        run_axil(&["--db", db_str, "store", "perf", &format!(r#"{{"i":{i}}}"#)]);
    }

    let start = std::time::Instant::now();
    let (_, _, code) = run_axil(&["--db", db_str, "list", "perf"]);
    let elapsed = start.elapsed();

    assert_eq!(code, 0);
    assert!(
        elapsed.as_millis() < 500,
        "list (100 records) took {}ms (hard limit: 500ms)",
        elapsed.as_millis()
    );
}

// ─── Errors go to stderr ───────────────────────────────────────────────────

#[test]
fn test_errors_to_stderr() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();

    run_axil(&["init", db_str]);

    // Get non-existent → error on stderr, not stdout
    let (stdout, stderr, code) = run_axil(&["--db", db_str, "get", "01JQNV1XG00000000000000000"]);
    assert_eq!(code, 2);
    assert!(!stderr.is_empty(), "error should be on stderr");
    // stdout should be empty for not-found
    assert!(stdout.trim().is_empty());
}

#[test]
fn maintain_if_stale_runs_then_skips_and_never_downsamples() {
    let (_dir, db) = temp_db();
    let db_str = db.to_str().unwrap();
    // Seed a record so the DB + its companions exist.
    run_axil_env(db_str, &["store", "decisions", r#"{"summary":"seed"}"#]);

    // Fresh DB: both additive tasks have never run → due.
    let (out, _e, code) = run_axil_env(db_str, &["maintain", "--if-stale", "--dry-run"]);
    assert_eq!(code, 0, "maintain dry-run should exit 0");
    let v: Value = serde_json::from_str(&out).expect("maintain emits JSON");
    let ran: Vec<&str> = v["ran"].as_array().unwrap().iter().filter_map(|x| x.as_str()).collect();
    assert!(ran.contains(&"snapshot"), "snapshot should be due on a fresh DB");
    assert!(ran.contains(&"health_report"), "health_report should be due on a fresh DB");
    // Regression guard for the data-loss fix: downsample must NEVER auto-run.
    assert!(
        !out.contains("downsample"),
        "maintain must not run/mention downsample (it purges memory): {out}"
    );

    // Real run records the cadence.
    let (_o, _e, code) = run_axil_env(db_str, &["maintain", "--if-stale"]);
    assert_eq!(code, 0);

    // Immediately after: both tasks are within cadence → skipped.
    let (out2, _e, _c) = run_axil_env(db_str, &["maintain", "--if-stale"]);
    let v2: Value = serde_json::from_str(&out2).unwrap();
    let ran2: Vec<&str> = v2["ran"].as_array().unwrap().iter().filter_map(|x| x.as_str()).collect();
    assert!(ran2.is_empty(), "nothing should be due immediately after a run: {out2}");
    assert_eq!(v2["errors"].as_array().map(|a| a.len()), Some(0), "no task errors: {out2}");
}

#[test]
fn test_reindex_combines_proxies_and_scip() {
    // `axil reindex` runs the structural proxy index (foreground) + the SCIP
    // refresh in one call. With `--no-scip` the SCIP half is skipped, so this
    // test stays hermetic (no external language indexer required in CI) while
    // still exercising the combined command wiring and the merged JSON shape.
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();

    let db = root.join("mem.axil");
    let db_str = db.to_str().unwrap();
    let root_str = root.to_str().unwrap();
    run_axil(&["init", db_str]);

    let (stdout, stderr, code) = run_axil(&["--db", db_str, "reindex", root_str, "--no-scip"]);
    assert_eq!(code, 0, "reindex --no-scip should exit 0. stderr: {stderr}");

    let v: Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|_| panic!("reindex emits combined JSON, got: {stdout}"));

    // Proxy half ran: at least the source file was indexed.
    assert!(
        v["proxies"]["indexed_files"].as_u64().unwrap_or(0) >= 1,
        "proxy index should report indexed files: {v}"
    );
    // SCIP half was skipped by --no-scip (deterministic, no indexer needed).
    assert_eq!(
        v["scip"]["status"], "skipped",
        "scip should be skipped with --no-scip: {v}"
    );
    assert_eq!(
        v["scip"]["reason"], "no_scip",
        "skip reason should be no_scip: {v}"
    );
}

#[test]
fn test_scip_refresh_root_flag_scopes_scan() {
    // Regression for the `reindex` SCIP-scope fix: `axil reindex <path>` pins
    // the SCIP scan to the indexed path via `scip refresh --root`, so the
    // proxy index and SCIP graph can't cover different trees when the DB lives
    // outside the project. Verify `--root` actually redirects the scan: pointed
    // at an empty dir it finds no project, even though the DB's own derived
    // root contains a Cargo.toml. Hermetic — refresh bails before running any
    // indexer, so no rust-analyzer/scip binary is required.
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path();

    // A real Rust project under the DB's derived root (`<db>/../..`).
    std::fs::create_dir_all(root.join("proj/src")).unwrap();
    std::fs::write(
        root.join("proj/Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(root.join("proj/src/main.rs"), "fn main() {}\n").unwrap();

    // An empty directory to point `--root` at.
    let empty = root.join("empty");
    std::fs::create_dir_all(&empty).unwrap();

    // DB inside the project, so a DB-derived root WOULD find the Cargo.toml.
    let db = root.join("proj/.axil/memory.axil");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let db_str = db.to_str().unwrap();
    run_axil(&["init", db_str]);

    // `--root <empty>` redirects the scan away from the project → no language
    // detected → non-zero exit. Proves the override is honored (without it,
    // the DB-derived root would detect the Rust project).
    let (out, err, code) = run_axil(&[
        "--db",
        db_str,
        "scip",
        "refresh",
        "--root",
        empty.to_str().unwrap(),
    ]);
    assert_ne!(
        code, 0,
        "scip refresh with an empty --root should find no project. out={out} err={err}"
    );
    assert!(
        format!("{out}{err}").contains("no language detected"),
        "should report 'no language detected' when --root is empty: out={out} err={err}"
    );
}
