//! `axil-cache` as a Tier-2 [`Extension`].
//!
//! Surfaces:
//! - `axil cache put <json|->` — store a question/answer pair.
//! - `axil cache get "<question>" [--threshold N] [--top-k N]` — look up a
//!   cached answer, honoring the similarity threshold, TTL, and code-aware
//!   invalidation.
//! - `axil cache stats` — cumulative hit/miss/eviction counters.
//! - `axil cache clear [--all | --expired]` — drop entries.
//! - MCP tools: `cache_get` and `cache_put`.
//!
//! The CLI surface routes through the generic Path-C external-subcommand
//! dispatch in `axil-cli` (no per-command wiring in `main.rs`): the
//! Extension declares its [`CliSurface`] and owns `handle_cli`.

use std::path::PathBuf;

use axil_core::error::Result as CoreResult;
use axil_core::{
    Axil, AxilError, CliArg, CliInvocation, CliOutput, CliSubcommand, CliSurface, Dispatch,
    Extension, McpCall, McpSurface, McpTool,
};
use serde_json::{json, Value};

use crate::{
    clear, get, put, stats, CacheError, ClearScope, GetOutcome, PutRequest, DEFAULT_THRESHOLD,
};

/// `axil-cache` as an Axil Extension.
///
/// Construct with [`CacheExtension::default`] and register through
/// [`axil_core::AxilBuilder::with_extension`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheExtension;

impl Extension for CacheExtension {
    fn id(&self) -> &str {
        "cache"
    }

    fn display_name(&self) -> &str {
        "Semantic Answer Cache"
    }

    fn table_prefixes(&self) -> &[&str] {
        &["_cache_"]
    }

    fn cli_commands(&self) -> Option<CliSurface> {
        Some(
            CliSurface::new(
                "cache",
                "Reuse a cached answer when a semantically similar question recurs, with code-aware invalidation.",
            )
            .subcommand(
                CliSubcommand::new(
                    "put",
                    "Store a question/answer pair. JSON positional or `-` for stdin: {question, answer, code_refs?[], ttl?}.",
                )
                .arg(CliArg::new("json", "Inline JSON object or `-` to read from stdin.").takes_value(true)),
            )
            .subcommand(
                CliSubcommand::new("get", "Look up a cached answer for a question.")
                    .arg(CliArg::new("question", "The question to look up.").takes_value(true))
                    .arg(
                        CliArg::new("threshold", "Minimum similarity for a hit (default 0.92).")
                            .takes_value(true),
                    )
                    .arg(CliArg::new("top-k", "Maximum hits to return (default 1).").takes_value(true)),
            )
            .subcommand(CliSubcommand::new(
                "stats",
                "Show cumulative hit/miss/eviction counters.",
            ))
            .subcommand(
                CliSubcommand::new("clear", "Remove cached entries.")
                    .arg(CliArg::new("all", "Remove every entry."))
                    .arg(CliArg::new("expired", "Remove only entries past their TTL.")),
            ),
        )
    }

    fn mcp_tools(&self) -> Option<McpSurface> {
        Some(McpSurface::new(vec![
            McpTool::new(
                "cache_put",
                "Cache a question/answer pair so a future semantically similar question returns the stored answer instead of re-deriving it. Optionally pin the answer to code via code_refs; the entry is invalidated when that code changes.",
                json!({
                    "type": "object",
                    "required": ["question", "answer"],
                    "properties": {
                        "question":  { "type": "string", "description": "The question this answer resolves. Embedded for semantic recall." },
                        "answer":    { "type": "string", "description": "The answer to return on a future similar question." },
                        "code_refs": { "type": "array", "items": { "type": "string" }, "description": "Code-ref specs (proxy_id | canonical_id | path[:line]) to invalidate against when the code changes." },
                        "ttl":       { "type": "integer", "description": "Optional time-to-live in seconds from now." },
                        "valid_until": { "type": "string", "description": "Optional explicit RFC 3339 expiry (overrides ttl)." }
                    }
                }),
            ),
            McpTool::new(
                "cache_get",
                "Return a cached answer for a semantically similar question, or a miss. A hit re-checks TTL and code-ref fingerprints first, so a returned answer is neither expired nor invalidated by a code change. Miss reasons distinguish no_match / below_threshold / stale_code / expired.",
                json!({
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question":  { "type": "string", "description": "The question to look up." },
                        "threshold": { "type": "number", "description": "Minimum similarity for a hit (default 0.92)." },
                        "top_k":     { "type": "integer", "description": "Maximum hits to return (default 1)." }
                    }
                }),
            ),
            McpTool::new(
                "cache_stats",
                "Cumulative cache statistics: live entry count, lifetime hits/misses, hit rate, and how many entries were evicted for stale code or expiry.",
                json!({ "type": "object", "properties": {} }),
            ),
        ]))
    }

    fn handle_cli(&self, db: &Axil, invocation: &CliInvocation) -> CoreResult<Dispatch<CliOutput>> {
        let Some(top) = invocation.command_path.first() else {
            return Ok(Dispatch::NotHandled);
        };
        if top != "cache" {
            return Ok(Dispatch::NotHandled);
        }
        match invocation.command_path.get(1).map(String::as_str) {
            Some("put") => handle_put_cli(db, invocation).map(Dispatch::Handled),
            Some("get") => handle_get_cli(db, invocation).map(Dispatch::Handled),
            Some("stats") => Ok(Dispatch::Handled(json_stdout(handle_stats(db)?))),
            Some("clear") => handle_clear_cli(db, invocation).map(Dispatch::Handled),
            _ => Ok(Dispatch::NotHandled),
        }
    }

    fn handle_mcp(&self, db: &Axil, call: &McpCall) -> CoreResult<Dispatch<Value>> {
        match call.tool.as_str() {
            "cache_put" => handle_put_mcp(db, &call.params).map(Dispatch::Handled),
            "cache_get" => handle_get_mcp(db, &call.params).map(Dispatch::Handled),
            "cache_stats" => handle_stats(db).map(Dispatch::Handled),
            _ => Ok(Dispatch::NotHandled),
        }
    }
}

/// Working directory a relative code-ref path resolves against. The absolute
/// path is captured in the fingerprint at put time, so this only matters for
/// resolving the initial relative reference.
fn base_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// ── CLI handlers ────────────────────────────────────────────────────────────

fn handle_put_cli(db: &Axil, invocation: &CliInvocation) -> CoreResult<CliOutput> {
    let value = read_payload(invocation)?;
    let req = PutRequest::from_value(value).map_err(cache_err_to_axil)?;
    let record = put(db, &req, &base_dir()).map_err(cache_err_to_axil)?;
    Ok(json_stdout(json!({
        "stored": true,
        "id": record.id.to_string(),
        "question": req.question,
        "code_refs": record.data.get("code_refs").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
        "valid_until": record.data.get("valid_until"),
    })))
}

fn handle_get_cli(db: &Axil, invocation: &CliInvocation) -> CoreResult<CliOutput> {
    let question = first_positional(&invocation.args)
        .ok_or_else(|| AxilError::InvalidQuery("cache get: provide a question".into()))?;
    let threshold = parse_f32_arg(&invocation.args, "--threshold")?.unwrap_or(DEFAULT_THRESHOLD);
    let top_k = parse_usize_arg(&invocation.args, "--top-k")?.unwrap_or(1);
    let outcome = get(db, &question, threshold, top_k, &base_dir())?;
    Ok(json_stdout(outcome_to_json(&outcome)))
}

fn handle_stats(db: &Axil) -> CoreResult<Value> {
    Ok(serde_json::to_value(stats(db)?).unwrap_or(Value::Null))
}

fn handle_clear_cli(db: &Axil, invocation: &CliInvocation) -> CoreResult<CliOutput> {
    // Default to expired-only: `clear` (and `clear --expired`) is the safe,
    // non-destructive choice; wiping the whole cache needs an explicit `--all`.
    let scope = if flag_set(&invocation.args, "--all") {
        ClearScope::All
    } else {
        ClearScope::Expired
    };
    let removed = clear(db, scope)?;
    Ok(json_stdout(json!({
        "cleared": removed,
        "scope": if scope == ClearScope::All { "all" } else { "expired" },
    })))
}

// ── MCP handlers ────────────────────────────────────────────────────────────

fn handle_put_mcp(db: &Axil, params: &Value) -> CoreResult<Value> {
    let req = PutRequest::from_value(params.clone()).map_err(cache_err_to_axil)?;
    let record = put(db, &req, &base_dir()).map_err(cache_err_to_axil)?;
    Ok(json!({
        "stored": true,
        "id": record.id.to_string(),
        "question": req.question,
        "code_refs": record.data.get("code_refs").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
        "valid_until": record.data.get("valid_until"),
    }))
}

fn handle_get_mcp(db: &Axil, params: &Value) -> CoreResult<Value> {
    let question = params
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AxilError::InvalidQuery("cache_get: `question` is required".into()))?;
    let threshold = params
        .get("threshold")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(DEFAULT_THRESHOLD);
    let top_k = params
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(1);
    let outcome = get(db, question, threshold, top_k, &base_dir())?;
    Ok(outcome_to_json(&outcome))
}

// ── output shaping ──────────────────────────────────────────────────────────

/// Render a [`GetOutcome`] as the JSON both the CLI and MCP surfaces return.
fn outcome_to_json(outcome: &GetOutcome) -> Value {
    match outcome {
        GetOutcome::Hit(hits) => json!({
            "result": "hit",
            "count": hits.len(),
            "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>(),
        }),
        GetOutcome::Miss {
            reason,
            best_score,
            detail,
        } => json!({
            "result": "miss",
            "reason": reason.as_str(),
            "best_score": best_score,
            "detail": detail,
        }),
    }
}

fn hit_to_json(hit: &crate::CacheHit) -> Value {
    json!({
        "id": hit.id,
        "question": hit.question,
        "answer": hit.answer,
        "score": hit.score,
        "hit_count": hit.hit_count,
        // Surface the code the served answer is pinned to, so a caller can see
        // what an invalidation would key on.
        "code_refs": hit.code_refs,
    })
}

// ── argument parsing helpers ────────────────────────────────────────────────

/// Resolve the JSON payload for `cache put`: a positional `<json>`, `-` for
/// stdin, or fall through to captured stdin. Mirrors `axil checkpoint`.
fn read_payload(invocation: &CliInvocation) -> Result<Value, AxilError> {
    let raw = match first_positional(&invocation.args) {
        Some(ref s) if s == "-" => invocation
            .stdin
            .clone()
            .ok_or_else(|| AxilError::InvalidQuery("cache put: `-` requested stdin but none was captured".into()))?,
        Some(s) => s,
        None => invocation
            .stdin
            .clone()
            .ok_or_else(|| AxilError::InvalidQuery("cache put: provide a JSON object as a positional arg or via stdin".into()))?,
    };
    serde_json::from_str(&raw)
        .map_err(|e| AxilError::InvalidQuery(format!("cache put: payload is not valid JSON: {e}")))
}

/// First argument that is neither a flag nor a flag's value. Recognizes the
/// value-taking flags this surface declares (`--threshold`, `--top-k`) so
/// their values aren't mistaken for the positional.
fn first_positional(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--threshold" || arg == "--top-k" {
            iter.next(); // skip the value
            continue;
        }
        if arg.starts_with("--") {
            continue;
        }
        return Some(arg.clone());
    }
    None
}

fn flag_set(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Read a `--name value` or `--name=value` argument.
fn named_arg(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
        if let Some(rest) = arg.strip_prefix(prefix.as_str()) {
            return Some(rest.to_string());
        }
    }
    None
}

fn parse_f32_arg(args: &[String], name: &str) -> Result<Option<f32>, AxilError> {
    match named_arg(args, name) {
        Some(v) => v
            .parse::<f32>()
            .map(Some)
            .map_err(|_| AxilError::InvalidQuery(format!("cache get: `{name}` must be a number, got `{v}`"))),
        None => Ok(None),
    }
}

fn parse_usize_arg(args: &[String], name: &str) -> Result<Option<usize>, AxilError> {
    match named_arg(args, name) {
        Some(v) => v
            .parse::<usize>()
            .map(Some)
            .map_err(|_| AxilError::InvalidQuery(format!("cache get: `{name}` must be an integer, got `{v}`"))),
        None => Ok(None),
    }
}

fn json_stdout(value: Value) -> CliOutput {
    CliOutput {
        exit_code: 0,
        stdout: serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()),
        stderr: String::new(),
    }
}

fn cache_err_to_axil(e: CacheError) -> AxilError {
    match e {
        CacheError::Axil(a) => a,
        other => AxilError::InvalidQuery(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TABLE_CACHE_ENTRIES;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("cache.axil")).build().unwrap();
        (db, dir)
    }

    #[test]
    fn id_and_table_prefix() {
        let ext = CacheExtension;
        assert_eq!(ext.id(), "cache");
        assert_eq!(ext.display_name(), "Semantic Answer Cache");
        assert_eq!(ext.table_prefixes(), &["_cache_"]);
        assert!(TABLE_CACHE_ENTRIES.starts_with(ext.table_prefixes()[0]));
    }

    #[test]
    fn cli_surface_advertises_all_subcommands() {
        let surface = CacheExtension.cli_commands().unwrap();
        assert_eq!(surface.command, "cache");
        let names: Vec<&str> = surface.subcommands.iter().map(|s| s.name.as_str()).collect();
        for expected in ["put", "get", "stats", "clear"] {
            assert!(names.contains(&expected), "missing subcommand {expected}");
        }
    }

    #[test]
    fn mcp_surface_exposes_cache_tools() {
        let surface = CacheExtension.mcp_tools().unwrap();
        let names: Vec<&str> = surface.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"cache_get"));
        assert!(names.contains(&"cache_put"));
        // CLI parity: `stats` is reachable over MCP too.
        assert!(names.contains(&"cache_stats"));
    }

    #[test]
    fn put_via_positional_then_get_exact_fallback() {
        let (db, _dir) = temp_db();
        let put_inv = CliInvocation {
            command_path: vec!["cache".into(), "put".into()],
            args: vec![r#"{"question":"how to reset db","answer":"delete the .axil file"}"#.into()],
            stdin: None,
        };
        let out = CacheExtension
            .handle_cli(&db, &put_inv)
            .unwrap()
            .handled()
            .expect("put handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["stored"], true);
        assert_eq!(db.list(TABLE_CACHE_ENTRIES).unwrap().len(), 1);

        // Exact-match fallback (no vector index on this DB).
        let get_inv = CliInvocation {
            command_path: vec!["cache".into(), "get".into()],
            args: vec!["how to reset db".into(), "--threshold".into(), "0.5".into()],
            stdin: None,
        };
        let out = CacheExtension
            .handle_cli(&db, &get_inv)
            .unwrap()
            .handled()
            .expect("get handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["result"], "hit");
        assert_eq!(v["hits"][0]["answer"], "delete the .axil file");
    }

    #[test]
    fn put_via_stdin_dash() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["cache".into(), "put".into()],
            args: vec!["-".into()],
            stdin: Some(r#"{"question":"q","answer":"a"}"#.into()),
        };
        let out = CacheExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["stored"], true);
    }

    #[test]
    fn get_missing_is_reported() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["cache".into(), "get".into()],
            args: vec!["nothing here".into()],
            stdin: None,
        };
        let out = CacheExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["result"], "miss");
    }

    #[test]
    fn clear_defaults_to_expired_scope() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["cache".into(), "clear".into()],
            args: vec![],
            stdin: None,
        };
        let out = CacheExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["scope"], "expired");
    }

    #[test]
    fn stats_reports_zero_on_fresh_db() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["cache".into(), "stats".into()],
            args: vec![],
            stdin: None,
        };
        let out = CacheExtension
            .handle_cli(&db, &inv)
            .unwrap()
            .handled()
            .expect("handled");
        let v: Value = serde_json::from_str(&out.stdout).unwrap();
        assert_eq!(v["entries"], 0);
        assert_eq!(v["total_hits"], 0);
    }

    #[test]
    fn mcp_put_then_get() {
        let (db, _dir) = temp_db();
        let put = McpCall {
            tool: "cache_put".into(),
            params: json!({"question": "mcp q", "answer": "mcp a"}),
        };
        let v = CacheExtension
            .handle_mcp(&db, &put)
            .unwrap()
            .handled()
            .expect("handled");
        assert_eq!(v["stored"], true);
        // Parity with the CLI put response, which echoes the question.
        assert_eq!(v["question"], "mcp q");

        let getc = McpCall {
            tool: "cache_get".into(),
            params: json!({"question": "mcp q", "threshold": 0.5}),
        };
        let v = CacheExtension
            .handle_mcp(&db, &getc)
            .unwrap()
            .handled()
            .expect("handled");
        assert_eq!(v["result"], "hit");
        assert_eq!(v["hits"][0]["answer"], "mcp a");
        // A hit always carries a `code_refs` array (empty here) so callers can
        // see what an answer is pinned to.
        assert!(v["hits"][0]["code_refs"].is_array());
    }

    #[test]
    fn mcp_cache_stats_reports_counters() {
        let (db, _dir) = temp_db();
        let call = McpCall {
            tool: "cache_stats".into(),
            params: Value::Null,
        };
        let v = CacheExtension
            .handle_mcp(&db, &call)
            .unwrap()
            .handled()
            .expect("cache_stats handled");
        assert_eq!(v["entries"], 0);
        assert_eq!(v["total_hits"], 0);
        assert_eq!(v["total_misses"], 0);
    }

    #[test]
    fn hit_to_json_includes_code_refs() {
        let hit = crate::CacheHit {
            id: "1".into(),
            question: "q".into(),
            answer: "a".into(),
            score: 1.0,
            hit_count: 1,
            code_refs: vec![json!({"path": "src/lib.rs"})],
        };
        let v = hit_to_json(&hit);
        assert_eq!(v["code_refs"][0]["path"], "src/lib.rs");
    }

    #[test]
    fn declines_non_cache_command() {
        let (db, _dir) = temp_db();
        let inv = CliInvocation {
            command_path: vec!["recall".into()],
            args: vec![],
            stdin: None,
        };
        assert!(matches!(
            CacheExtension.handle_cli(&db, &inv).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn declines_unknown_mcp_tool() {
        let (db, _dir) = temp_db();
        let call = McpCall {
            tool: "not_a_cache_tool".into(),
            params: Value::Null,
        };
        assert!(matches!(
            CacheExtension.handle_mcp(&db, &call).unwrap(),
            Dispatch::NotHandled
        ));
    }

    #[test]
    fn registers_in_axil_builder() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("t.axil"))
            .with_extension(CacheExtension)
            .build()
            .unwrap();
        assert_eq!(db.extensions().len(), 1);
        assert_eq!(db.extensions()[0].id(), "cache");
    }
}
