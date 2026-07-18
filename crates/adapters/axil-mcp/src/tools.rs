//! MCP tool definitions and dispatch for Axil.

use serde_json::{json, Value};

use axil_core::{Axil, RecordId};

use crate::protocol::{ToolCallResult, ToolDefinition};

/// Return all available MCP tool definitions.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "recall".into(),
            description: "Semantic search + graph + time-based recall of past context. Returns ranked results combining vector similarity with recency. When `across` is supplied, fans out to sibling project DBs declared in the workspace manifest, applies each sibling's read-consent filter at the remote, and merges with provenance tags.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "table": {
                        "type": "string",
                        "description": "Filter by table name"
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by the record's `type` facet (matches data.type, case-insensitive exact). Records without a `type` field are excluded when set."
                    },
                    "across": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional: member labels to fan out to, or [\"*\"] for every member. Requires a .axil-workspace.toml. When set, each result carries `source_member`, `source_member_id`, and `source_record_id`."
                    },
                    "strict_consent": {
                        "type": "boolean",
                        "description": "When `across` is set, drop workspace-scoped records at remote siblings (default: false)",
                        "default": false
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "store".into(),
            description: "Insert a record with optional auto-embedding of specified fields.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "data": {
                        "type": "object",
                        "description": "JSON data to store"
                    },
                    "embed_fields": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Fields to auto-embed after insert"
                    }
                },
                "required": ["table", "data"]
            }),
        },
        ToolDefinition {
            name: "link".into(),
            description: "Create a graph edge between two records.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "from": {
                        "type": "string",
                        "description": "Source record ID"
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "Edge type label"
                    },
                    "to": {
                        "type": "string",
                        "description": "Target record ID"
                    },
                    "props": {
                        "type": "object",
                        "description": "Optional properties for the edge"
                    }
                },
                "required": ["from", "edge_type", "to"]
            }),
        },
        ToolDefinition {
            name: "search".into(),
            description: "Full-text search across all indexed fields.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "query_history".into(),
            description: "Time-based query of past records. Filter by date range and table.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "after": {
                        "type": "string",
                        "description": "ISO 8601 datetime — only include records after this time"
                    },
                    "before": {
                        "type": "string",
                        "description": "ISO 8601 datetime — only include records before this time"
                    },
                    "table": {
                        "type": "string",
                        "description": "Filter by table name. When omitted, non-internal tables are scanned up to a bounded cap."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max records to return (default: 50)",
                        "default": 50
                    }
                }
            }),
        },
        ToolDefinition {
            name: "get".into(),
            description: "Get a single record by ID.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Record ID"
                    }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "list".into(),
            description: "List records in a table.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)",
                        "default": 50
                    }
                },
                "required": ["table"]
            }),
        },
        ToolDefinition {
            name: "delete".into(),
            description: "Delete a record by ID.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Record ID"
                    }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "query".into(),
            description: "Query a table with typed field predicates, ordering, and pagination. The `where` string mirrors the CLI `--where` syntax: conditions joined by AND (case-insensitive), operators `= != > < >= <= contains`, quoted values ('x' or \"x\") forced to strings, unquoted numbers compared numerically. Flat top-level fields only — no OR, parentheses, or nested dot-paths.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "where": {
                        "type": "string",
                        "description": "Filter expression, e.g. \"oos_sharpe > 0.3 AND family = 'meanrev'\""
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Sort field"
                    },
                    "direction": {
                        "type": "string",
                        "description": "Sort direction: asc (default) or desc",
                        "default": "asc"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for pagination"
                    }
                },
                "required": ["table"]
            }),
        },
        #[cfg(feature = "ql")]
        ToolDefinition {
            name: "aggregate".into(),
            description: "Aggregate a table: count / avg / min / max / sum, optionally grouped. `metrics` is an array of specs — \"count\", \"avg(field)\", \"min(field)\", \"max(field)\", \"sum(field)\". `where` uses the same syntax as the `query` tool. Non-numeric or missing values are skipped for avg/min/max/sum and reported per group as `skipped`. Returns {table, group_by, groups:[{group, count, ...}], total_rows}.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name"
                    },
                    "metrics": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Metric specs: count, avg(field), min(field), max(field), sum(field)"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Group results by this field's value (missing → null group)"
                    },
                    "where": {
                        "type": "string",
                        "description": "Filter expression, e.g. \"oos_sharpe > 0.3 AND family = 'meanrev'\""
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived records (excluded by default)",
                        "default": false
                    }
                },
                "required": ["table"]
            }),
        },
        ToolDefinition {
            name: "add_vector".into(),
            description: "Attach a pre-computed raw vector to an existing record. `space` targets a named vector space (`<db>.axil.vec.<name>`, own dimension) — omit for the default space. The space is created lazily on first write, binding the vector's length as its dimension.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Record ID to attach the vector to"
                    },
                    "vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Vector as an array of floats"
                    },
                    "space": {
                        "type": "string",
                        "description": "Named vector space ([a-z0-9_-]{1,32}); omit for the default space"
                    }
                },
                "required": ["id", "vector"]
            }),
        },
        ToolDefinition {
            name: "similar".into(),
            description: "Find records with vectors similar to a query. Provide `vector` (array of floats) or `id` (uses that record's stored vector, excluding it from results). `threshold` filters to score >= F (near-dup detection). `space` targets a named vector space.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Query vector (mutually exclusive with id)"
                    },
                    "id": {
                        "type": "string",
                        "description": "Use this record's stored vector as the query (excluded from results)"
                    },
                    "space": {
                        "type": "string",
                        "description": "Named vector space ([a-z0-9_-]{1,32}); omit for the default space"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                        "default": 5
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Only return results scoring >= this cosine similarity"
                    }
                }
            }),
        },
        ToolDefinition {
            name: "lineage".into(),
            description: "Walk a derivation chain over graph edges (default `derived_from`). `direction` = ancestors (follow OUT edges, root-first: what each node was derived from), descendants (IN edges: what was derived from the node), or both. Returns a hop list with per-hop selected `fields` and numeric `delta` vs its parent hop (the node on the other end of the discovering edge). Create edges with `link A derived_from B --props '{\"mutation\":\"…\"}'`.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Root record ID"
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "Edge type to follow (default: derived_from)",
                        "default": "derived_from"
                    },
                    "direction": {
                        "type": "string",
                        "description": "ancestors | descendants | both (default: ancestors)",
                        "default": "ancestors"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum hops from the root (default: 20)",
                        "default": 20
                    },
                    "fields": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "record.data keys to include per hop (and to diff for delta); omit for all fields"
                    }
                },
                "required": ["id"]
            }),
        },

        // ─── Intent-native writes (Track B) ─────────────────────────
        ToolDefinition {
            name: "remember_decision".into(),
            description: "Record an architectural or implementation decision. Auto-embeds, auto-supersedes, and dedupes by (agent_id, external_id) or 5-minute content hash.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "summary":     {"type": "string", "description": "What was decided"},
                    "reason":      {"type": "string", "description": "Why this path was chosen"},
                    "files":       {"type": "array", "items": {"type": "string"}, "description": "Files affected"},
                    "agent_id":    {"type": "string", "description": "Agent identifier (pairs with external_id for idempotency)"},
                    "external_id": {"type": "string", "description": "Caller-supplied idempotency key"},
                    "force_new":   {"type": "boolean", "description": "Bypass dedup to force a fresh record", "default": false}
                },
                "required": ["summary"]
            }),
        },
        ToolDefinition {
            name: "remember_error".into(),
            description: "Record an error, optionally with root cause and fix. Same idempotency rules as remember_decision.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "error":       {"type": "string", "description": "What went wrong"},
                    "root_cause":  {"type": "string"},
                    "fix":         {"type": "string"},
                    "files":       {"type": "array", "items": {"type": "string"}},
                    "agent_id":    {"type": "string"},
                    "external_id": {"type": "string"},
                    "force_new":   {"type": "boolean", "default": false}
                },
                "required": ["error"]
            }),
        },
        ToolDefinition {
            name: "set_preference".into(),
            description: "Set a user preference. Overwrites by key; previous value is kept on the new record as _previous_value for lightweight audit.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key":   {"type": "string"},
                    "value": {"description": "Any JSON value (string, number, object, array)"}
                },
                "required": ["key", "value"]
            }),
        },
        ToolDefinition {
            name: "close_session".into(),
            description: "Mark a session as closed with an optional summary. Idempotent by id — a repeated call returns the existing closed session.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id":      {"type": "string"},
                    "summary": {"type": "string"}
                },
                "required": ["id"]
            }),
        },

        // ─── Structural code recall ─────────────────────────────────
        ToolDefinition {
            name: "code_search".into(),
            description: "Search structural code proxies and return compact pointers (path, line, symbol, breadcrumb, canonical_id). Smaller and more actionable than `recall` for code-shaped queries because raw source is never returned.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5}
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "code_context".into(),
            description: "Assemble a coding-task context block within a token budget. Groups results into `relevant_code` (proxy pointers), `related_memories` (memories whose `code_refs` point at matched proxies), `relevant_modules`, `similar_context`, `active_rules`, `recent_changes`.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "task":   {"type": "string", "description": "Task description / question"},
                    "budget": {"type": "integer", "description": "Token budget. Omit to auto-size by indexed repo size (tiny→1500, large monorepo→4000, capped)."}
                },
                "required": ["task"]
            }),
        },

        // `dep_docs` / `deps_status` are provided by `DocsExtension`
        // `handle_tools_list` overlays the Extension
        // surface onto this static list, and `dispatch` routes them
        // through `dispatch_mcp` before reaching the hardcoded match
        // below. No hardcoded entry needed here.

        // ─── Boot contract (Track C) ────────────────────────────────
        ToolDefinition {
            name: "boot".into(),
            description: "Return a stable BootContext (schema v1): current_scope, constraints, recent_decisions, active_failures, open_threads, preferences, confidence_notes. Fixed section order, token-budget aware — lower-priority sections drop when over budget.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "budget": {"type": "integer", "description": "Token budget (default 2000)"},
                    "topic":  {"type": "string", "description": "Optional topic for focused recall"},
                    "scope":  {"type": "array", "items": {"type": "string"}, "description": "Scope filter"}
                }
            }),
        },
        // ─── Read-only census + light health ────────────────────────
        ToolDefinition {
            name: "inspect".into(),
            description: "Read-only overview of what kinds of memory this brain holds and whether it is healthy. Returns a per-record-type census (e.g. decisions, errors, sessions; all internal bookkeeping tables collapse into one `_internal` bucket) plus a light health verdict (`ok`/`warning`/`error`) drawn from the same checks as `axil doctor`. Performs zero writes — use it when you only have MCP access and can't shell out to `axil tables`/`axil doctor`.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        // ─── Pull-based cross-agent delta (event-log feature only) ───
        #[cfg(feature = "event-log")]
        ToolDefinition {
            name: "recall_delta".into(),
            description: "Pull committed semantic events that happened after a cursor, oldest first. Surfaces high-signal cross-agent changes — a belief was revised, a decision superseded, an error fixed, a checkpoint written — so a second agent can ask 'what changed since I last looked' without re-scanning. Each event carries a monotonic `cursor`; pass the last one back in as `since_cursor` to resume. `exclude_agent` drops events written by that agent (e.g. skip your own). Returns committed facts only — it does not read record bodies and does not relax cross-agent session isolation; resolve a returned `record_id` through the normal access path.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "since_cursor": {"type": "string", "description": "Return only events strictly after this cursor. Omit to read from the oldest retained event."},
                    "exclude_agent": {"type": "string", "description": "Drop events tagged with this agent_id (e.g. your own)."},
                    "limit": {"type": "integer", "description": "Max events to return (default 50)."}
                }
            }),
        },
    ]
}

/// Dispatch a tool call to the appropriate handler.
///
/// before the hard-coded match, try every registered
/// Extension's [`axil_core::Extension::handle_mcp`]. If a registered
/// Extension claims the tool and returns `Dispatch::Handled(value)`,
/// surface that value as the tool result. Path C semantics: Extensions
/// that return `Dispatch::NotHandled` (or aren't registered) fall
/// through to the hard-coded handlers below.
pub fn dispatch(db: &Axil, tool_name: &str, args: &Value) -> ToolCallResult {
    // Extension dispatch first.
    let call = axil_core::McpCall {
        tool: tool_name.to_string(),
        params: args.clone(),
    };
    match axil_core::dispatch_mcp(db, &db.extensions(), &call) {
        Ok(axil_core::Dispatch::Handled(value)) => return ToolCallResult::json(&value),
        Ok(axil_core::Dispatch::NotHandled) => {
            // Fall through to hardcoded handlers.
        }
        Err(e) => return ToolCallResult::error(format!("{tool_name} dispatch failed: {e}")),
    }

    match tool_name {
        "recall" => handle_recall(db, args),
        "store" => handle_store(db, args),
        "link" => handle_link(db, args),
        "search" => handle_search(db, args),
        "query_history" => handle_query_history(db, args),
        "get" => handle_get(db, args),
        "list" => handle_list(db, args),
        "query" => handle_query(db, args),
        #[cfg(feature = "ql")]
        "aggregate" => handle_aggregate(db, args),
        "add_vector" => handle_add_vector(db, args),
        "similar" => handle_similar(db, args),
        "lineage" => handle_lineage(db, args),
        "delete" => handle_delete(db, args),
        "remember_decision" => handle_remember_decision(db, args),
        "remember_error" => handle_remember_error(db, args),
        "set_preference" => handle_set_preference(db, args),
        "close_session" => handle_close_session(db, args),
        "boot" => handle_boot(db, args),
        "inspect" => handle_inspect(db, args),
        #[cfg(feature = "event-log")]
        "recall_delta" => handle_recall_delta(db, args),
        "code_search" => handle_code_search(db, args),
        "code_context" => handle_code_context(db, args),
        // `dep_docs` / `deps_status` are handled by DocsExtension via
        // `dispatch_mcp` above — no fallback arm needed.
        _ => ToolCallResult::error(format!("unknown tool: {tool_name}")),
    }
}

fn handle_code_search(db: &Axil, args: &Value) -> ToolCallResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolCallResult::error("missing required parameter: query"),
    };
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    // Fetch a larger pool so non-proxy hits (project/file index records)
    // can't crowd proxies out before we filter. Mirrors the CLI
    // `code-search` behavior so MCP and CLI return the same N proxies.
    let pool = top_k.saturating_mul(5).max(15);
    match axil_indexer::recall::recall(db, query, pool) {
        Ok(results) => {
            let proxies: Vec<&axil_indexer::recall::RecallResult> = results
                .iter()
                .filter(|r| r.source == "proxy")
                .take(top_k)
                .collect();
            let v = serde_json::to_value(&proxies).unwrap_or(json!([]));
            ToolCallResult::json(&v)
        }
        Err(e) => ToolCallResult::error(format!("code_search failed: {e}")),
    }
}

fn handle_code_context(db: &Axil, args: &Value) -> ToolCallResult {
    let task = match args.get("task").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return ToolCallResult::error("missing required parameter: task"),
    };
    // Honor an explicit budget; otherwise auto-size by indexed repo size. A
    // present-but-invalid budget (negative, fractional, non-numeric) is a
    // caller error, not a cue to silently substitute the adaptive default.
    let budget = match args.get("budget") {
        None | Some(Value::Null) => axil_indexer::recall::auto_context_budget(db),
        Some(v) => match v.as_u64() {
            Some(b) => b as usize,
            None => return ToolCallResult::error("budget must be a non-negative integer"),
        },
    };
    let opts = axil_indexer::recall::ContextOptions {
        max_tokens: budget,
        task: Some(task.to_string()),
        ..Default::default()
    };
    match axil_indexer::recall::context(db, &opts) {
        Ok(value) => ToolCallResult::json(&value),
        Err(e) => ToolCallResult::error(format!("code_context failed: {e}")),
    }
}

// ─── Tool handlers ──────────────────────────────────────────────────────────

fn handle_recall(db: &Axil, args: &Value) -> ToolCallResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolCallResult::error("missing required parameter: query"),
    };
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let table = args.get("table").and_then(|v| v.as_str());
    // --type facet filter, normalized case-insensitive (matches the CLI).
    let type_filter = args
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_lowercase());
    let across: Vec<String> = args
        .get("across")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let strict_consent = args
        .get("strict_consent")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !across.is_empty() {
        return handle_recall_across(db, query, top_k, table, &across, strict_consent);
    }

    const TABLE_FILTER_INFLATION: usize = 5;
    let fetch_k = if table.is_some() || type_filter.is_some() {
        top_k * TABLE_FILTER_INFLATION
    } else {
        top_k
    };

    let cfg = axil_core::RecallConfig {
        qtc: Some(axil_core::scoring::QtcConfig::default()),
        // Collapse near-duplicate hits so the MCP `recall` tool
        // doesn't spend its top_k budget on restated memories, and widen k when
        // the kept top-k compresses much better than the candidate pool (a
        // diverse cluster was cut).
        dedup: axil_core::scoring::DedupConfig {
            enabled: true,
            completeness_widen: true,
            ..Default::default()
        },
        ..Default::default()
    };
    match db.recall(query, fetch_k, Some(cfg)) {
        Ok(results) => {
            let mut records: Vec<Value> = Vec::new();
            for result in &results {
                if records.len() >= top_k {
                    break;
                }
                if let Some(t) = table {
                    if result.record.table != t {
                        continue;
                    }
                }
                if let Some(ref tf) = type_filter {
                    let record_type = result
                        .record
                        .data
                        .get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_lowercase());
                    if record_type.as_deref() != Some(tf.as_str()) {
                        continue;
                    }
                }
                let mut entry = json!({
                    "id": result.record.id.to_string(),
                    "table": result.record.table,
                    "data": result.record.data,
                    "score": result.score,
                    "created_at": result.record.created_at.to_rfc3339(),
                    "updated_at": result.record.updated_at.to_rfc3339(),
                });
                // Promote code-proxy pointer fields to top level so MCP
                // clients can use them without parsing `data`.
                if result.record.table == axil_indexer::TABLE_CODE_PROXIES {
                    promote_proxy_fields(&mut entry, &result.record.data);
                }
                records.push(entry);
            }
            ToolCallResult::json(&json!(records))
        }
        Err(e) => ToolCallResult::error(format!("recall failed: {e}")),
    }
}

fn promote_proxy_fields(entry: &mut Value, data: &Value) {
    let pointer_keys = [
        "proxy_id",
        "kind",
        "path",
        "symbol",
        "line_start",
        "line_end",
        "canonical_id",
        "breadcrumb",
        "source_record",
    ];
    let obj = match entry.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    for key in &pointer_keys {
        if let Some(v) = data.get(*key) {
            obj.insert((*key).to_string(), v.clone());
        }
    }
}

/// Cross-project fan-out path for `recall` when `across` is supplied.
/// Mirrors the CLI `axil recall-across` semantics: open each sibling
/// with the same plugin-detection logic the MCP server uses for its
/// primary DB, apply each sibling's `read_consent` at the remote, and
/// merge results with provenance.
fn handle_recall_across(
    db: &Axil,
    query: &str,
    top_k: usize,
    table: Option<&str>,
    across: &[String],
    strict_consent: bool,
) -> ToolCallResult {
    use axil_workspace::federation::{
        fan_out, FederationRequest, MemberRecallBatch, MemberRecallRow,
    };

    let primary_path = db.path().to_path_buf();
    let manifest = match axil_workspace::discover_manifest(&primary_path) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return ToolCallResult::error(
                "no .axil-workspace.toml found for `across` fan-out — run `axil workspace init` first",
            );
        }
        Err(e) => return ToolCallResult::error(format!("manifest load failed: {e}")),
    };

    let arg = across.join(",");
    let (members, unknown) = manifest.resolve_members_arg(&arg);
    if !unknown.is_empty() && arg != "*" {
        return ToolCallResult::error(format!("unknown member(s): {}", unknown.join(",")));
    }
    if members.is_empty() {
        return ToolCallResult::error(format!("no members matched across={arg}"));
    }

    let caller_member = axil_workspace::resolve::resolve_member(
        &manifest,
        std::env::current_dir().unwrap_or_else(|_| primary_path.clone()),
    )
    .map(|r| r.member_id)
    .unwrap_or_default();
    let workspace_id = manifest.workspace.id.clone();

    let members_owned: Vec<(String, &axil_workspace::manifest::Member)> = members
        .iter()
        .map(|(label, member)| ((*label).to_string(), *member))
        .collect();

    let req = FederationRequest {
        manifest: &manifest,
        caller_workspace: workspace_id.clone(),
        caller_member,
        caller_roles: Vec::new(),
        members: members_owned,
        top_k,
        strict_consent,
    };

    let (results, warnings) = fan_out(req, |label, member, path| {
        let builder = axil_core::Axil::open(&path);
        let builder = crate::attach_detected_engines(builder)
            .map_err(|e| format!("attach plugins for {label}: {e}"))?;
        let sibling = builder.build().map_err(|e| format!("open {label}: {e}"))?;
        let cfg = axil_core::RecallConfig {
            qtc: Some(axil_core::scoring::QtcConfig::default()),
            ..Default::default()
        };
        let rows = sibling
            .recall(query, top_k.saturating_mul(3), Some(cfg))
            .map_err(|e| format!("recall on {label}: {e}"))?;
        let member_rows: Vec<MemberRecallRow> = rows
            .into_iter()
            .filter(|r| table.is_none() || table.map(|t| r.record.table == t).unwrap_or(true))
            .map(|r| {
                let read_consent: axil_workspace::consent::ReadConsent =
                    serde_json::from_value(r.record.read_consent_raw()).unwrap_or_default();
                MemberRecallRow {
                    record_id: r.record.id.to_string(),
                    record: json!({
                        "id": r.record.id.to_string(),
                        "table": r.record.table,
                        "data": r.record.data,
                        "created_at": r.record.created_at.to_rfc3339(),
                        "updated_at": r.record.updated_at.to_rfc3339(),
                    }),
                    score: r.score,
                    read_consent,
                }
            })
            .collect();
        Ok(MemberRecallBatch {
            member_label: label.to_string(),
            member_id: member.id.clone(),
            workspace_id: workspace_id.clone(),
            vector_compatible: true,
            rows: member_rows,
            warnings: Vec::new(),
        })
    });

    let rows: Vec<Value> = results
        .into_iter()
        .map(|r| {
            json!({
                "score": r.score,
                "source_workspace_id": r.source_workspace_id,
                "source_member": r.source_member_label,
                "source_member_id": r.source_member_id,
                "source_record_id": r.source_record_id,
                "record": r.record,
                "text_only_fallback": r.text_only_fallback,
            })
        })
        .collect();
    ToolCallResult::json(&json!({
        "results": rows,
        "warnings": warnings,
    }))
}

fn handle_store(db: &Axil, args: &Value) -> ToolCallResult {
    let table = match args.get("table").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return ToolCallResult::error("missing required parameter: table"),
    };
    let data = match args.get("data") {
        Some(d) => d.clone(),
        None => return ToolCallResult::error("missing required parameter: data"),
    };
    let embed_fields: Vec<&str> = args
        .get("embed_fields")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    match db.insert(table, data) {
        Ok(record) => {
            let id = record.id.clone();

            // Auto-embed requested fields.
            for field in &embed_fields {
                if let Err(e) = db.embed_field(&id, field) {
                    return ToolCallResult::error(format!(
                        "record inserted (id: {id}) but embed_field({field}) failed: {e}"
                    ));
                }
            }

            ToolCallResult::json(&json!({
                "id": id.to_string(),
                "table": record.table,
                "created_at": record.created_at.to_rfc3339(),
            }))
        }
        Err(e) => ToolCallResult::error(format!("store failed: {e}")),
    }
}

fn handle_link(db: &Axil, args: &Value) -> ToolCallResult {
    let from = match args.get("from").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: from"),
    };
    let edge_type = match args.get("edge_type").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: edge_type"),
    };
    let to = match args.get("to").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: to"),
    };
    let props = args.get("props").cloned();

    let from_id = match RecordId::from_string(from) {
        Ok(id) => id,
        Err(e) => return ToolCallResult::error(format!("invalid 'from' record ID: {e}")),
    };
    let to_id = match RecordId::from_string(to) {
        Ok(id) => id,
        Err(e) => return ToolCallResult::error(format!("invalid 'to' record ID: {e}")),
    };

    match db.relate(&from_id, edge_type, &to_id, props) {
        Ok(edge_id) => ToolCallResult::json(&json!({
            "edge_id": edge_id.to_string(),
            "from": from,
            "edge_type": edge_type,
            "to": to,
        })),
        Err(e) => ToolCallResult::error(format!("link failed: {e}")),
    }
}

fn handle_search(db: &Axil, args: &Value) -> ToolCallResult {
    let query = match args.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return ToolCallResult::error("missing required parameter: query"),
    };
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    match db.search_text(query, limit) {
        Ok(results) => {
            let records: Vec<Value> = results
                .iter()
                .map(|(record, score)| {
                    json!({
                        "id": record.id.to_string(),
                        "table": record.table,
                        "data": record.data,
                        "score": score,
                    })
                })
                .collect();
            ToolCallResult::json(&json!(records))
        }
        Err(e) => ToolCallResult::error(format!("search failed: {e}")),
    }
}

fn handle_query_history(db: &Axil, args: &Value) -> ToolCallResult {
    let table = args.get("table").and_then(|v| v.as_str());
    let after = args.get("after").and_then(|v| v.as_str());
    let before = args.get("before").and_then(|v| v.as_str());
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(50);

    let Some(table_name) = table else {
        let tables = match db.tables_with_counts() {
            Ok(t) => t,
            Err(e) => return ToolCallResult::error(format!("failed to list tables: {e}")),
        };

        // Top-K min-heap keyed by `created_at` string. Keeps the K newest
        // records across every non-internal table without loading the whole
        // corpus into memory. RFC3339+'Z' compares lexicographically =
        // chronologically. Wrapper struct because serde_json::Value: !Ord.
        use std::cmp::{Ordering, Reverse};
        use std::collections::BinaryHeap;
        struct TsEntry(String, Value);
        impl PartialEq for TsEntry {
            fn eq(&self, o: &Self) -> bool {
                self.0 == o.0
            }
        }
        impl Eq for TsEntry {}
        impl Ord for TsEntry {
            fn cmp(&self, o: &Self) -> Ordering {
                self.0.cmp(&o.0)
            }
        }
        impl PartialOrd for TsEntry {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }

        let mut heap: BinaryHeap<Reverse<TsEntry>> = BinaryHeap::new();
        for (tbl, _) in &tables {
            if tbl.starts_with('_') {
                continue;
            }
            let Ok(records) = db.list(tbl) else { continue };
            for record in &records {
                if !should_include_by_time(record, after, before) {
                    continue;
                }
                let json = record_to_json(record);
                let ts = json
                    .get("created_at")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if heap.len() < limit {
                    heap.push(Reverse(TsEntry(ts, json)));
                } else if let Some(Reverse(TsEntry(min_ts, _))) = heap.peek() {
                    if ts.as_str() > min_ts.as_str() {
                        heap.pop();
                        heap.push(Reverse(TsEntry(ts, json)));
                    }
                }
            }
        }
        // into_sorted_vec() ascending under Reverse = newest first under T.
        let sorted: Vec<Value> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(TsEntry(_, v))| v)
            .collect();
        return ToolCallResult::json(&json!(sorted));
    };

    match db.list(table_name) {
        Ok(records) => {
            let mut results: Vec<Value> = records
                .iter()
                .filter(|r| should_include_by_time(r, after, before))
                .map(record_to_json)
                .collect();
            results.sort_by(|a, b| {
                let ta = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
                let tb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
                tb.cmp(ta)
            });
            results.truncate(limit);
            ToolCallResult::json(&json!(results))
        }
        Err(e) => ToolCallResult::error(format!("query_history failed: {e}")),
    }
}

fn handle_get(db: &Axil, args: &Value) -> ToolCallResult {
    let id_str = match args.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: id"),
    };
    let id = match RecordId::from_string(id_str) {
        Ok(id) => id,
        Err(e) => return ToolCallResult::error(format!("invalid record ID: {e}")),
    };

    match db.get(&id) {
        Ok(Some(record)) => ToolCallResult::json(&record_to_json(&record)),
        Ok(None) => ToolCallResult::error(format!("record not found: {id_str}")),
        Err(e) => ToolCallResult::error(format!("get failed: {e}")),
    }
}

fn handle_list(db: &Axil, args: &Value) -> ToolCallResult {
    let table = match args.get("table").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return ToolCallResult::error("missing required parameter: table"),
    };
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

    match db.list(table) {
        Ok(records) => {
            let results: Vec<Value> = records.iter().take(limit).map(record_to_json).collect();
            ToolCallResult::json(&json!(results))
        }
        Err(e) => ToolCallResult::error(format!("list failed: {e}")),
    }
}

fn handle_query(db: &Axil, args: &Value) -> ToolCallResult {
    let table = match args.get("table").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return ToolCallResult::error("missing required parameter: table"),
    };
    let mut qb = db.query().table(table);

    if let Some(where_str) = args.get("where").and_then(|v| v.as_str()) {
        match parse_where_expr(where_str) {
            Ok(conds) => {
                for (field, op, value) in conds {
                    qb = qb.where_field(&field, op, value);
                }
            }
            Err(e) => return ToolCallResult::error(format!("invalid where: {e}")),
        }
    }

    if let Some(field) = args.get("order_by").and_then(|v| v.as_str()) {
        let dir = match args.get("direction").and_then(|v| v.as_str()) {
            Some(d) if d.eq_ignore_ascii_case("desc") => axil_core::SortDirection::Desc,
            _ => axil_core::SortDirection::Asc,
        };
        qb = qb.order_by(field, dir);
    }
    if let Some(n) = args.get("limit").and_then(|v| v.as_u64()) {
        qb = qb.limit(n as usize);
    }
    if let Some(n) = args.get("offset").and_then(|v| v.as_u64()) {
        qb = qb.offset(n as usize);
    }

    match qb.exec() {
        Ok(results) => {
            let records: Vec<Value> = results.iter().map(record_to_json).collect();
            ToolCallResult::json(&json!(records))
        }
        Err(e) => ToolCallResult::error(format!("query failed: {e}")),
    }
}

#[cfg(feature = "ql")]
fn handle_aggregate(db: &Axil, args: &Value) -> ToolCallResult {
    let table = match args.get("table").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return ToolCallResult::error("missing required parameter: table"),
    };

    let mut metrics: Vec<axil_ql::AggMetric> = Vec::new();
    if let Some(arr) = args.get("metrics").and_then(|v| v.as_array()) {
        for m in arr {
            let Some(spec) = m.as_str() else {
                return ToolCallResult::error("each metric must be a string");
            };
            match axil_ql::AggMetric::parse_spec(spec) {
                Ok(metric) => metrics.push(metric),
                Err(e) => return ToolCallResult::error(format!("invalid metric '{spec}': {e}")),
            }
        }
    }

    let group_by = args.get("group_by").and_then(|v| v.as_str());
    let include_archived = args
        .get("include_archived")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut wheres: Vec<axil_core::query::WhereClause> = Vec::new();
    if let Some(where_str) = args.get("where").and_then(|v| v.as_str()) {
        match parse_where_expr(where_str) {
            Ok(conds) => {
                for (field, op, value) in conds {
                    wheres.push(axil_core::query::WhereClause { field, op, value });
                }
            }
            Err(e) => return ToolCallResult::error(format!("invalid where: {e}")),
        }
    }

    let req = axil_ql::AggRequest {
        table,
        metrics: &metrics,
        group_by,
        where_clauses: &wheres,
        include_archived,
    };
    match axil_ql::aggregate(db, &req) {
        Ok(value) => ToolCallResult::json(&value),
        Err(e) => ToolCallResult::error(format!("aggregate failed: {e}")),
    }
}

/// Parse a JSON array of numbers into an `f32` vector.
fn parse_vector_arg(arr: &[Value]) -> Result<Vec<f32>, String> {
    arr.iter()
        .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| "vector elements must be numbers".to_string()))
        .collect()
}

/// Render a scored record hit as JSON.
fn scored_to_json(record: &axil_core::Record, score: f32) -> Value {
    let mut json = record_to_json(record);
    json.as_object_mut()
        .unwrap()
        .insert("score".to_string(), json!(score));
    json
}

fn handle_add_vector(db: &Axil, args: &Value) -> ToolCallResult {
    let id_str = match args.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: id"),
    };
    let rid = match RecordId::from_string(id_str) {
        Ok(r) => r,
        Err(e) => return ToolCallResult::error(format!("invalid record ID: {e}")),
    };
    let vector = match args.get("vector").and_then(|v| v.as_array()) {
        Some(arr) => match parse_vector_arg(arr) {
            Ok(v) => v,
            Err(e) => return ToolCallResult::error(e),
        },
        None => return ToolCallResult::error("missing required parameter: vector (array of numbers)"),
    };
    let space = args.get("space").and_then(|v| v.as_str());
    let result = match space {
        Some(s) => db.add_vector_in(s, &rid, &vector),
        None => db.add_vector(&rid, &vector),
    };
    match result {
        Ok(()) => {
            let mut out = json!({
                "added": true,
                "id": id_str,
                "dimensions": vector.len(),
            });
            if let Some(s) = space {
                out.as_object_mut().unwrap().insert("space".to_string(), json!(s));
            }
            ToolCallResult::json(&out)
        }
        Err(e) => ToolCallResult::error(format!("add_vector failed: {e}")),
    }
}

fn handle_similar(db: &Axil, args: &Value) -> ToolCallResult {
    let space = args.get("space").and_then(|v| v.as_str());
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let threshold = args.get("threshold").and_then(|v| v.as_f64()).map(|f| f as f32);

    // Resolve the query vector and the record (if any) to exclude.
    let (query, exclude): (Vec<f32>, Option<RecordId>) =
        if let Some(arr) = args.get("vector").and_then(|v| v.as_array()) {
            match parse_vector_arg(arr) {
                Ok(v) => (v, None),
                Err(e) => return ToolCallResult::error(e),
            }
        } else if let Some(id_str) = args.get("id").and_then(|v| v.as_str()) {
            let rid = match RecordId::from_string(id_str) {
                Ok(r) => r,
                Err(e) => return ToolCallResult::error(format!("invalid record ID: {e}")),
            };
            let stored = match space {
                Some(s) => db.get_vector_in(s, &rid),
                None => db.get_vector(&rid),
            };
            match stored {
                Ok(Some(v)) => (v, Some(rid)),
                Ok(None) => {
                    return ToolCallResult::error(format!(
                        "record {id_str} has no stored vector in this space"
                    ))
                }
                Err(e) => return ToolCallResult::error(format!("similar failed: {e}")),
            }
        } else {
            return ToolCallResult::error("provide either `vector` or `id`");
        };

    let fetch = if exclude.is_some() { top_k + 1 } else { top_k };
    let search = match space {
        Some(s) => db.similar_in(s, &query, fetch),
        None => db.similar_to_vector(&query, fetch),
    };
    match search {
        Ok(mut results) => {
            if let Some(ref ex) = exclude {
                results.retain(|(r, _)| &r.id != ex);
            }
            if let Some(t) = threshold {
                results.retain(|(_, score)| *score >= t);
            }
            results.truncate(top_k);
            let values: Vec<Value> =
                results.iter().map(|(r, s)| scored_to_json(r, *s)).collect();
            ToolCallResult::json(&json!(values))
        }
        Err(e) => ToolCallResult::error(format!("similar failed: {e}")),
    }
}

fn handle_lineage(db: &Axil, args: &Value) -> ToolCallResult {
    let id_str = match args.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: id"),
    };
    let rid = match RecordId::from_string(id_str) {
        Ok(r) => r,
        Err(e) => return ToolCallResult::error(format!("invalid record ID: {e}")),
    };
    let edge_type = args
        .get("edge_type")
        .and_then(|v| v.as_str())
        .unwrap_or(axil_core::util::edge_types::DERIVED_FROM);
    let direction = match args
        .get("direction")
        .and_then(|v| v.as_str())
        .map(axil_core::lineage::LineageDirection::parse)
    {
        None => axil_core::lineage::LineageDirection::Ancestors,
        Some(Ok(d)) => d,
        Some(Err(e)) => return ToolCallResult::error(e),
    };
    let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let fields: Option<Vec<String>> = args.get("fields").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    });

    match axil_core::lineage::walk(db, &rid, edge_type, direction, max_depth, fields.as_deref()) {
        Ok(value) => ToolCallResult::json(&value),
        Err(e) => ToolCallResult::error(format!("lineage failed: {e}")),
    }
}

/// Parse a `--where`-style expression into one or more `(field, op, value)`
/// conditions, mirroring the CLI `parse_where_clause` grammar.
///
/// A single expression may hold several conditions joined by `AND`
/// (case-insensitive, whole word); the split is quote-aware. Supported
/// operators: `>= <= != = > <` and the word operator `contains`. Quoted values
/// (single or double) are always strings; unquoted values are typed by
/// `serde_json`, falling back to a bare string.
///
/// Kept adapter-local (a small twin of the CLI helper) rather than shared
/// through `axil-core`, which is off-limits for this work.
fn parse_where_expr(input: &str) -> Result<Vec<(String, axil_core::Op, Value)>, String> {
    ensure_where_quotes_balanced(input)?;
    let mut out = Vec::new();
    for cond in split_where_conditions(input) {
        let cond = cond.trim();
        if cond.is_empty() {
            continue;
        }
        out.push(parse_single_condition(cond)?);
    }
    if out.is_empty() {
        return Err(format!("invalid where clause: {input}"));
    }
    Ok(out)
}

/// Reject a where expression containing an unterminated quote. A dangling
/// quote would otherwise swallow the rest of the expression into a string
/// value that can never match — a silent empty result instead of an error.
fn ensure_where_quotes_balanced(clause: &str) -> Result<(), String> {
    let mut in_quote: Option<u8> = None;
    for &c in clause.as_bytes() {
        match in_quote {
            Some(q) if c == q => in_quote = None,
            None if c == b'"' || c == b'\'' => in_quote = Some(c),
            _ => {}
        }
    }
    if let Some(q) = in_quote {
        return Err(format!(
            "invalid where clause: unterminated {} quote in '{clause}'",
            q as char
        ));
    }
    Ok(())
}

/// Split on the `AND` keyword (case-insensitive, whole word) outside quotes.
fn split_where_conditions(clause: &str) -> Vec<&str> {
    let bytes = clause.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == q {
                    in_quote = None;
                }
                i += 1;
            }
            None => {
                if c == b'"' || c == b'\'' {
                    in_quote = Some(c);
                    i += 1;
                } else if (c == b'a' || c == b'A') && matches_keyword(bytes, i, b"and") {
                    parts.push(&clause[start..i]);
                    i += 3;
                    start = i;
                } else {
                    i += 1;
                }
            }
        }
    }
    parts.push(&clause[start..]);
    parts
}

/// True when `kw` (ASCII, lowercase) occurs at `bytes[i..]` case-insensitively
/// as a whole word.
fn matches_keyword(bytes: &[u8], i: usize, kw: &[u8]) -> bool {
    let end = i + kw.len();
    if end > bytes.len() {
        return false;
    }
    if !bytes[i..end].iter().zip(kw).all(|(b, k)| b.eq_ignore_ascii_case(k)) {
        return false;
    }
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let before_ok = i == 0 || !is_ident(bytes[i - 1]);
    let after_ok = end >= bytes.len() || !is_ident(bytes[end]);
    before_ok && after_ok
}

/// Parse a single `field op value` condition.
fn parse_single_condition(cond: &str) -> Result<(String, axil_core::Op, Value), String> {
    let bytes = cond.as_bytes();
    let mut i = 0usize;
    let mut in_quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == q {
                    in_quote = None;
                }
                i += 1;
            }
            None => {
                if c == b'"' || c == b'\'' {
                    in_quote = Some(c);
                    i += 1;
                    continue;
                }
                if matches!(c, b'>' | b'<' | b'!' | b'=') {
                    let rest = &cond[i..];
                    let (op_str, op_len): (&str, usize) = if rest.starts_with(">=") {
                        (">=", 2)
                    } else if rest.starts_with("<=") {
                        ("<=", 2)
                    } else if rest.starts_with("!=") {
                        ("!=", 2)
                    } else if c == b'=' {
                        ("=", 1)
                    } else if c == b'>' {
                        (">", 1)
                    } else if c == b'<' {
                        ("<", 1)
                    } else {
                        return Err(format!("'{cond}': '!' must be part of the '!=' operator"));
                    };
                    let field = cond[..i].trim().to_string();
                    if field.is_empty() {
                        return Err(format!("field name must not be empty in '{cond}'"));
                    }
                    let op: axil_core::Op =
                        op_str.parse().map_err(|e| format!("{e}"))?;
                    let value = parse_where_value(cond[i + op_len..].trim())?;
                    return Ok((field, op, value));
                }
                i += 1;
            }
        }
    }
    if let Some(pos) = find_keyword_outside_quotes(cond, b"contains") {
        let field = cond[..pos].trim().to_string();
        if field.is_empty() {
            return Err(format!("field name must not be empty in '{cond}'"));
        }
        let value = parse_where_value(cond[pos + "contains".len()..].trim())?;
        return Ok((field, axil_core::Op::Contains, value));
    }
    Err(format!(
        "invalid where clause: {cond} (expected field=value, field>value, field contains value)"
    ))
}

/// Find the byte offset of `kw` as a whole word outside any quoted region.
fn find_keyword_outside_quotes(s: &str, kw: &[u8]) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut in_quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match in_quote {
            Some(q) => {
                if c == q {
                    in_quote = None;
                }
                i += 1;
            }
            None => {
                if c == b'"' || c == b'\'' {
                    in_quote = Some(c);
                    i += 1;
                } else if c.eq_ignore_ascii_case(&kw[0]) && matches_keyword(bytes, i, kw) {
                    return Some(i);
                } else {
                    i += 1;
                }
            }
        }
    }
    None
}

/// Type a where-clause value: quoted → string; otherwise `serde_json` types it.
///
/// Backslash escapes are not interpreted inside quotes — a quote character of
/// the same kind cannot appear inside a quoted value (use the other quote
/// kind around it instead).
fn parse_where_value(val_str: &str) -> Result<Value, String> {
    let b = val_str.as_bytes();
    if val_str.len() >= 2
        && ((b[0] == b'\'' && b[val_str.len() - 1] == b'\'')
            || (b[0] == b'"' && b[val_str.len() - 1] == b'"'))
    {
        return Ok(Value::String(val_str[1..val_str.len() - 1].to_string()));
    }
    if let Ok(v) = serde_json::from_str(val_str) {
        return Ok(v);
    }
    // A leading number/bool/null followed by trailing text ("5 oops") is
    // almost certainly a typo'd expression (a missing AND) — error rather
    // than silently matching a garbage string literal. Plain multi-word
    // strings ("mean rev") still fall through to the bare-string case.
    if let Some(first) = val_str.split_whitespace().next() {
        if first.len() < val_str.trim().len()
            && serde_json::from_str::<Value>(first)
                .map(|v| !v.is_string())
                .unwrap_or(false)
        {
            return Err(format!(
                "ambiguous where value '{val_str}': quote it ('{val_str}') if you \
                 meant a string, or join conditions with AND if it is two conditions"
            ));
        }
    }
    Ok(Value::String(val_str.to_string()))
}

fn handle_delete(db: &Axil, args: &Value) -> ToolCallResult {
    let id_str = match args.get("id").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolCallResult::error("missing required parameter: id"),
    };
    let id = match RecordId::from_string(id_str) {
        Ok(id) => id,
        Err(e) => return ToolCallResult::error(format!("invalid record ID: {e}")),
    };

    match db.delete(&id) {
        Ok(true) => ToolCallResult::json(&json!({"deleted": true, "id": id_str})),
        Ok(false) => ToolCallResult::error(format!("record not found: {id_str}")),
        Err(e) => ToolCallResult::error(format!("delete failed: {e}")),
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn record_to_json(record: &axil_core::Record) -> Value {
    let mut json = json!({
        "id": record.id.to_string(),
        "table": record.table,
        "data": record.data,
        "created_at": record.created_at.to_rfc3339(),
        "updated_at": record.updated_at.to_rfc3339(),
    });
    if let Some(ref metadata) = record.metadata {
        json.as_object_mut()
            .unwrap()
            .insert("metadata".to_string(), metadata.clone());
    }
    json
}

/// Check if a record falls within the optional time range.
fn should_include_by_time(
    record: &axil_core::Record,
    after: Option<&str>,
    before: Option<&str>,
) -> bool {
    if let Some(after_str) = after {
        if let Ok(after_dt) = after_str.parse::<chrono::DateTime<chrono::Utc>>() {
            if record.created_at < after_dt {
                return false;
            }
        }
    }
    if let Some(before_str) = before {
        if let Ok(before_dt) = before_str.parse::<chrono::DateTime<chrono::Utc>>() {
            if record.created_at > before_dt {
                return false;
            }
        }
    }
    true
}

// ─── Intent-native write handlers (Track B) ────────────────────────────

fn handle_remember_decision(db: &Axil, args: &Value) -> ToolCallResult {
    let Some(summary) = args.get("summary").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("missing required parameter: summary");
    };
    let reason = args.get("reason").and_then(|v| v.as_str());
    let files_vec = axil_core::util::extract_str_array(args, "files");
    let files: Vec<&str> = files_vec.iter().map(String::as_str).collect();
    let agent_id = args.get("agent_id").and_then(|v| v.as_str());
    let external_id = args.get("external_id").and_then(|v| v.as_str());
    let force_new = args
        .get("force_new")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    match db.remember_decision(axil_core::DecisionInput {
        summary,
        reason,
        files: if files.is_empty() {
            None
        } else {
            Some(files.as_slice())
        },
        agent_id,
        external_id,
        force_new,
        source: axil_core::WriteSource::Mcp,
    }) {
        Ok(r) => ToolCallResult::json(&json!({
            "id": r.id.to_string(),
            "is_new": r.is_new,
            "superseded": r.superseded.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })),
        Err(e) => ToolCallResult::error(format!("remember_decision failed: {e}")),
    }
}

fn handle_remember_error(db: &Axil, args: &Value) -> ToolCallResult {
    let Some(error) = args.get("error").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("missing required parameter: error");
    };
    let root_cause = args.get("root_cause").and_then(|v| v.as_str());
    let fix = args.get("fix").and_then(|v| v.as_str());
    let files_vec = axil_core::util::extract_str_array(args, "files");
    let files: Vec<&str> = files_vec.iter().map(String::as_str).collect();
    let agent_id = args.get("agent_id").and_then(|v| v.as_str());
    let external_id = args.get("external_id").and_then(|v| v.as_str());
    let force_new = args
        .get("force_new")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    match db.remember_error(axil_core::ErrorInput {
        error,
        root_cause,
        fix,
        files: if files.is_empty() {
            None
        } else {
            Some(files.as_slice())
        },
        agent_id,
        external_id,
        force_new,
        source: axil_core::WriteSource::Mcp,
    }) {
        Ok(r) => ToolCallResult::json(&json!({
            "id": r.id.to_string(),
            "is_new": r.is_new,
            "superseded": r.superseded.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })),
        Err(e) => ToolCallResult::error(format!("remember_error failed: {e}")),
    }
}

fn handle_set_preference(db: &Axil, args: &Value) -> ToolCallResult {
    let Some(key) = args.get("key").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("missing required parameter: key");
    };
    let Some(value) = args.get("value") else {
        return ToolCallResult::error("missing required parameter: value");
    };
    match db.set_preference(key, value.clone()) {
        Ok(r) => ToolCallResult::json(&json!({
            "id": r.id.to_string(),
            "is_new": r.is_new,
            "key": key,
        })),
        Err(e) => ToolCallResult::error(format!("set_preference failed: {e}")),
    }
}

fn handle_close_session(db: &Axil, args: &Value) -> ToolCallResult {
    let Some(id) = args.get("id").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("missing required parameter: id");
    };
    let summary = args.get("summary").and_then(|v| v.as_str());
    match db.close_session(id, summary) {
        Ok(r) => ToolCallResult::json(&json!({
            "id": r.id.to_string(),
            "session_id": id,
            "is_new": r.is_new,
        })),
        Err(e) => ToolCallResult::error(format!("close_session failed: {e}")),
    }
}

// ─── Boot handler (Track C) ────────────────────────────────────────────

fn handle_boot(db: &Axil, args: &Value) -> ToolCallResult {
    let token_budget = args
        .get("budget")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let topic = args.get("topic").and_then(|v| v.as_str()).map(String::from);
    let scope_vec = axil_core::util::extract_str_array(args, "scope");
    let scope = (!scope_vec.is_empty()).then_some(scope_vec);

    match db.boot(axil_core::BootOptions {
        token_budget,
        topic,
        scope,
    }) {
        Ok(ctx) => ToolCallResult::json(
            &serde_json::to_value(&ctx).expect("BootContext is always serializable"),
        ),
        Err(e) => ToolCallResult::error(format!("boot failed: {e}")),
    }
}

/// Read-only census of record types plus a light health verdict.
///
/// Answers "what kinds of memory does this brain hold, and is it healthy?"
/// without any shell access: per-table counts framed as the memory model
/// (not SQL columns), with every `_`-prefixed bookkeeping table rolled into a
/// single `_internal` bucket, and the overall `axil doctor` verdict reduced to
/// its read-only checks. Issues zero writes.
fn handle_inspect(db: &Axil, _args: &Value) -> ToolCallResult {
    let tables = match db.tables_with_counts() {
        Ok(t) => t,
        Err(e) => return ToolCallResult::error(format!("inspect failed: {e}")),
    };

    // Roll the user-facing tables out individually, collapsing every
    // `_`-prefixed bookkeeping table (entities, indexes, dep-docs, …) into one
    // opaque `_internal` bucket so the census reads as the memory model, not the
    // physical schema.
    let mut record_types = serde_json::Map::new();
    let mut internal_total: usize = 0;
    let mut total: usize = 0;
    for (name, count) in &tables {
        total += count;
        if name.starts_with('_') {
            internal_total += count;
        } else {
            record_types.insert(name.clone(), json!(count));
        }
    }
    if internal_total > 0 {
        record_types.insert("_internal".to_string(), json!(internal_total));
    }

    // `doctor()` is read-only (it only scans and counts); reuse its verdict and
    // per-check details for the light health summary.
    let health = match db.doctor() {
        Ok(report) => {
            let checks: Vec<Value> = report
                .checks
                .iter()
                .map(|c| {
                    json!({
                        "name": c.name,
                        "status": c.status,
                        "detail": c.detail,
                    })
                })
                .collect();
            json!({ "status": report.status, "checks": checks })
        }
        Err(e) => json!({ "status": "error", "detail": format!("doctor failed: {e}") }),
    };

    ToolCallResult::json(&json!({
        "record_types": record_types,
        "total_records": total,
        "health": health,
    }))
}

/// Pull-based "what changed since I last looked" over the durable semantic
/// event log. Returns committed facts only — never another agent's private
/// record body — so it surfaces cross-agent signals without relaxing session
/// isolation. The trailing `cursor` is the resume point for the next pull.
#[cfg(feature = "event-log")]
fn handle_recall_delta(db: &Axil, args: &Value) -> ToolCallResult {
    let since = args.get("since_cursor").and_then(|v| v.as_str());
    let exclude_agent = args.get("exclude_agent").and_then(|v| v.as_str());
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(50)
        .clamp(1, 1000) as usize;

    match db.recall_delta(since, exclude_agent, limit) {
        Ok(events) => {
            let next_cursor = events.last().map(|e| e.cursor.clone());
            let v = serde_json::to_value(&events).unwrap_or(json!([]));
            ToolCallResult::json(&json!({
                "events": v,
                "next_cursor": next_cursor,
                "count": events.len(),
            }))
        }
        Err(e) => ToolCallResult::error(format!("recall_delta failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axil_core::Axil;
    use axil_indexer::{IndexConfig, ProjectIndexer};

    fn temp_db_with_index() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp.axil");
        let db = Axil::open(&path).build().unwrap();
        // Tiny fixture so the indexer creates code proxies.
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"f\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("lib.rs"),
            "//! Recall scoring entry.\npub fn recall() -> u32 { 0 }\npub fn vector_search() -> u32 { 0 }\n",
        )
        .unwrap();
        ProjectIndexer::new(&db, IndexConfig::default())
            .index_full(dir.path())
            .unwrap();
        (db, dir)
    }

    fn parse_json_payload(result: &ToolCallResult) -> Value {
        let block = &result.content[0];
        serde_json::from_str(&block.text).expect("MCP tool result text must be valid JSON")
    }

    #[test]
    fn code_search_tool_returns_proxy_pointers() {
        let (db, _dir) = temp_db_with_index();
        let result = dispatch(
            &db,
            "code_search",
            &json!({"query": "vector_search", "top_k": 5}),
        );
        assert!(result.is_error.is_none(), "tool returned error");
        let body = parse_json_payload(&result);
        let arr = body.as_array().expect("expected JSON array");
        assert!(
            arr.iter()
                .any(|r| r.get("symbol").and_then(Value::as_str) == Some("vector_search")),
            "expected vector_search in code_search results, got {body:?}"
        );
        // Every result is a proxy hit with required pointer fields.
        for r in arr {
            assert_eq!(r.get("source").and_then(Value::as_str), Some("proxy"));
            assert!(r.get("path").is_some());
            assert!(r.get("proxy_id").is_some());
        }
    }

    #[test]
    fn code_search_tool_validates_query() {
        let (db, _dir) = temp_db_with_index();
        let result = dispatch(&db, "code_search", &json!({}));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn code_context_tool_returns_grouped_sections() {
        let (db, _dir) = temp_db_with_index();
        let result = dispatch(
            &db,
            "code_context",
            &json!({"task": "where does recall scoring live", "budget": 800}),
        );
        assert!(result.is_error.is_none());
        let body = parse_json_payload(&result);
        // The task_context returns an object with the documented keys.
        for key in [
            "task",
            "relevant_code",
            "related_memories",
            "relevant_modules",
        ] {
            assert!(
                body.get(key).is_some(),
                "missing {key} in code_context body"
            );
        }
        let relevant_code = body["relevant_code"]
            .as_array()
            .expect("relevant_code must be array");
        assert!(
            relevant_code.iter().any(
                |c| c.get("symbol").and_then(Value::as_str) == Some("recall")
                    || c.get("symbol").is_none()
            ),
            "relevant_code didn't surface a known proxy"
        );
    }

    #[test]
    fn code_context_rejects_invalid_budget() {
        let (db, _dir) = temp_db_with_index();
        // A present-but-invalid budget is a caller error, not a cue to
        // silently substitute the adaptive default.
        let neg = dispatch(&db, "code_context", &json!({"task": "x", "budget": -5}));
        assert_eq!(neg.is_error, Some(true), "negative budget must error");
        let frac = dispatch(&db, "code_context", &json!({"task": "x", "budget": 2.5}));
        assert_eq!(frac.is_error, Some(true), "fractional budget must error");
        // Omitted budget auto-sizes (no error).
        let auto = dispatch(&db, "code_context", &json!({"task": "x"}));
        assert!(auto.is_error.is_none(), "omitted budget should auto-size, not error");
    }

    #[test]
    fn code_context_tool_validates_task() {
        let (db, _dir) = temp_db_with_index();
        let result = dispatch(&db, "code_context", &json!({}));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn tool_definitions_advertise_code_tools() {
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "code_search"));
        assert!(names.iter().any(|n| n == "code_context"));
    }

    #[test]
    fn inspect_tool_reports_census_and_health() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("inspect.axil")).build().unwrap();
        // Two user-facing tables plus whatever internal bookkeeping a bare
        // insert produces (`_`-prefixed rows must collapse, not leak).
        db.insert("decisions", json!({"summary": "chose redb"})).unwrap();
        db.insert("decisions", json!({"summary": "chose tantivy"})).unwrap();
        db.insert("errors", json!({"error": "lock contention"})).unwrap();

        let result = dispatch(&db, "inspect", &json!({}));
        assert!(result.is_error.is_none(), "inspect returned error: {result:?}");
        let body = parse_json_payload(&result);

        let record_types = body["record_types"]
            .as_object()
            .expect("record_types must be an object");
        assert_eq!(record_types["decisions"].as_u64(), Some(2));
        assert_eq!(record_types["errors"].as_u64(), Some(1));
        // No raw `_`-prefixed table name leaks; internal rows are bucketed.
        assert!(
            record_types.keys().all(|k| k == "_internal" || !k.starts_with('_')),
            "raw internal table leaked into census: {record_types:?}"
        );

        assert!(body["total_records"].as_u64().unwrap() >= 3);
        let status = body["health"]["status"]
            .as_str()
            .expect("health.status must be a string");
        assert!(
            matches!(status, "ok" | "warning" | "error"),
            "unexpected health status: {status}"
        );
        assert!(
            body["health"]["checks"].is_array(),
            "health.checks must be an array"
        );
    }

    #[test]
    fn inspect_tool_is_advertised() {
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "inspect"));
    }

    #[test]
    fn query_tool_is_advertised() {
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "query"));
    }

    #[test]
    fn query_tool_numeric_and_string_predicates() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("q.axil")).build().unwrap();
        db.insert(
            "autopsies",
            json!({"family":"meanrev","oos_sharpe":0.5,"trades":5}),
        )
        .unwrap();
        db.insert(
            "autopsies",
            json!({"family":"meanrev","oos_sharpe":0.1,"trades":100}),
        )
        .unwrap();
        db.insert(
            "autopsies",
            json!({"family":"momentum","oos_sharpe":0.9,"trades":20}),
        )
        .unwrap();

        // One --where string, AND across a numeric and a quoted-string predicate.
        let result = dispatch(
            &db,
            "query",
            &json!({
                "table": "autopsies",
                "where": "oos_sharpe > 0.3 AND family = 'meanrev'"
            }),
        );
        assert!(result.is_error.is_none(), "query errored: {result:?}");
        let body = parse_json_payload(&result);
        let arr = body.as_array().expect("expected JSON array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["data"]["family"], "meanrev");
        assert_eq!(arr[0]["data"]["trades"], 5);
    }

    #[test]
    fn query_tool_numeric_comparison_not_lexicographic() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("q2.axil")).build().unwrap();
        db.insert("autopsies", json!({"trades":5})).unwrap();
        db.insert("autopsies", json!({"trades":100})).unwrap();

        let result = dispatch(
            &db,
            "query",
            &json!({"table":"autopsies","where":"trades < 30"}),
        );
        let body = parse_json_payload(&result);
        let arr = body.as_array().expect("expected JSON array");
        assert_eq!(arr.len(), 1, "trades<30 must match 5 numerically, not 100");
        assert_eq!(arr[0]["data"]["trades"], 5);
    }

    #[test]
    fn query_tool_invalid_where_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("q3.axil")).build().unwrap();
        let result = dispatch(&db, "query", &json!({"table":"t","where":"nonsense"}));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn where_expr_trailing_garbage_after_scalar_errors() {
        // Twin of the CLI parser's guard: "5 oops" errors, bare words pass.
        assert!(parse_where_expr("trades = 5 oops").is_err());
        assert_eq!(
            parse_where_expr("family = mean rev").unwrap()[0].2,
            json!("mean rev")
        );
    }

    #[test]
    fn query_tool_unterminated_quote_errors() {
        // A dangling quote must error, not silently match nothing.
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("q4.axil")).build().unwrap();
        let result = dispatch(&db, "query", &json!({"table":"t","where":"family = 'meanrev"}));
        assert_eq!(result.is_error, Some(true));
        assert!(parse_where_expr("family = 'meanrev").is_err());
        assert!(parse_where_expr("family = 'meanrev'").is_ok());
    }

    #[cfg(feature = "ql")]
    #[test]
    fn aggregate_tool_is_advertised() {
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "aggregate"));
    }

    #[cfg(feature = "ql")]
    #[test]
    fn aggregate_tool_histogram_and_avg() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("agg.axil")).build().unwrap();
        db.insert("autopsies", json!({"kill_reason":"drawdown","decay":2.0}))
            .unwrap();
        db.insert("autopsies", json!({"kill_reason":"drawdown","decay":4.0}))
            .unwrap();
        db.insert("autopsies", json!({"kill_reason":"fees","decay":9.0}))
            .unwrap();

        let result = dispatch(
            &db,
            "aggregate",
            &json!({
                "table": "autopsies",
                "metrics": ["count", "avg(decay)"],
                "group_by": "kill_reason"
            }),
        );
        assert!(result.is_error.is_none(), "aggregate errored: {result:?}");
        let body = parse_json_payload(&result);
        assert_eq!(body["total_rows"], 3);
        let groups = body["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0]["group"], "drawdown");
        assert_eq!(groups[0]["count"], 2);
        assert_eq!(groups[0]["avg_decay"], 3.0);
        assert_eq!(groups[1]["group"], "fees");
        assert_eq!(groups[1]["count"], 1);
    }

    #[cfg(feature = "ql")]
    #[test]
    fn aggregate_tool_invalid_metric_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("agg2.axil")).build().unwrap();
        let result = dispatch(
            &db,
            "aggregate",
            &json!({"table":"t","metrics":["median(x)"]}),
        );
        assert_eq!(result.is_error, Some(true));
    }

    #[cfg(feature = "event-log")]
    #[test]
    fn recall_delta_tool_is_advertised_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("delta.axil")).build().unwrap();
        db.set_event_log_enabled(true);

        // The tool is in the advertised surface when the feature is on.
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "recall_delta"));

        // Two allowlisted events by different agents.
        db.insert(
            "errors",
            json!({"error": "a", "fix": "f", "_agent_id": "agent-a"}),
        )
        .unwrap();
        db.insert(
            "errors",
            json!({"error": "b", "fix": "f", "_agent_id": "agent-b"}),
        )
        .unwrap();

        // Full pull returns both events plus a resumable cursor.
        let all = dispatch(&db, "recall_delta", &json!({}));
        assert!(all.is_error.is_none(), "recall_delta errored: {all:?}");
        let body = parse_json_payload(&all);
        assert_eq!(body["count"].as_u64(), Some(2));
        assert!(body["next_cursor"].is_string());

        // exclude_agent drops that agent's event.
        let filtered = dispatch(&db, "recall_delta", &json!({"exclude_agent": "agent-a"}));
        let body = parse_json_payload(&filtered);
        assert_eq!(body["count"].as_u64(), Some(1));
        assert_eq!(
            body["events"][0]["agent_id"].as_str(),
            Some("agent-b"),
            "exclude_agent must drop agent-a's event"
        );
    }

    #[test]
    fn vector_and_lineage_tools_advertised() {
        let names: Vec<String> = tool_definitions().into_iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n == "add_vector"));
        assert!(names.iter().any(|n| n == "similar"));
        assert!(names.iter().any(|n| n == "lineage"));
    }

    #[cfg(feature = "vector")]
    #[test]
    fn add_vector_and_similar_tools_default_space() {
        use axil_vector::AxilBuilderVectorExt;
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("vec.axil"))
            .with_vector(3)
            .unwrap()
            .with_vector_spaces()
            .build()
            .unwrap();
        let a = db.insert("v", json!({"n": "a"})).unwrap();
        let b = db.insert("v", json!({"n": "b"})).unwrap();

        let r = dispatch(&db, "add_vector", &json!({"id": a.id.to_string(), "vector": [1.0, 0.0, 0.0]}));
        assert!(r.is_error.is_none(), "add_vector errored: {r:?}");
        assert_eq!(parse_json_payload(&r)["dimensions"], 3);
        dispatch(&db, "add_vector", &json!({"id": b.id.to_string(), "vector": [0.9, 0.1, 0.0]}));

        let s = dispatch(&db, "similar", &json!({"vector": [1.0, 0.0, 0.0], "top_k": 2}));
        assert!(s.is_error.is_none(), "similar errored: {s:?}");
        let arr = parse_json_payload(&s).as_array().unwrap().to_vec();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["id"].as_str().unwrap(), a.id.to_string());
        assert!(arr[0]["score"].as_f64().unwrap() > 0.99);
    }

    #[cfg(feature = "vector")]
    #[test]
    fn similar_tool_id_threshold_named_space() {
        use axil_vector::AxilBuilderVectorExt;
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("fp.axil"))
            .with_vector(3)
            .unwrap()
            .with_vector_spaces()
            .build()
            .unwrap();
        let a = db.insert("fp", json!({"n": "a"})).unwrap();
        let b = db.insert("fp", json!({"n": "b"})).unwrap();
        let c = db.insert("fp", json!({"n": "c"})).unwrap();
        // 3-dim fp space: a & b near-duplicate (cos ≈ 0.97), c orthogonal.
        dispatch(&db, "add_vector", &json!({"id": a.id.to_string(), "vector": [1.0, 0.0, 0.0], "space": "fp"}));
        dispatch(&db, "add_vector", &json!({"id": b.id.to_string(), "vector": [1.0, 0.25, 0.0], "space": "fp"}));
        dispatch(&db, "add_vector", &json!({"id": c.id.to_string(), "vector": [0.0, 0.0, 1.0], "space": "fp"}));

        let s = dispatch(
            &db,
            "similar",
            &json!({"id": a.id.to_string(), "threshold": 0.9, "space": "fp"}),
        );
        assert!(s.is_error.is_none(), "similar errored: {s:?}");
        let arr = parse_json_payload(&s).as_array().unwrap().to_vec();
        assert_eq!(arr.len(), 1, "threshold 0.9 returns only the twin");
        assert_eq!(arr[0]["id"].as_str().unwrap(), b.id.to_string());
    }

    #[cfg(feature = "graph")]
    #[test]
    fn lineage_tool_walks_chain_with_deltas() {
        use axil_graph::AxilBuilderGraphExt;
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("lin.axil"))
            .with_graph_engine()
            .unwrap()
            .build()
            .unwrap();
        let a = db.insert("trials", json!({"n": "A", "sharpe": 1.0})).unwrap();
        let b = db.insert("trials", json!({"n": "B", "sharpe": 1.5})).unwrap();
        db.relate(&a.id, "derived_from", &b.id, Some(json!({"mutation": "m1"})))
            .unwrap();

        let r = dispatch(&db, "lineage", &json!({"id": a.id.to_string(), "fields": ["sharpe"]}));
        assert!(r.is_error.is_none(), "lineage errored: {r:?}");
        let body = parse_json_payload(&r);
        let hops = body["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 2);
        assert_eq!(hops[0]["id"].as_str().unwrap(), a.id.to_string());
        assert_eq!(hops[1]["id"].as_str().unwrap(), b.id.to_string());
        assert!((hops[1]["delta"]["sharpe"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    }
}
