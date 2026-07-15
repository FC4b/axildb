//! Cross-surface behavior oracle (exact-output layer).
//!
//! Every fixture case in `fixtures/behavior_oracle/v1.jsonl` describes one
//! surface-agnostic operation and its exact expected outcome. The runner
//! replays each case through every surface the fixture lists — the direct
//! Rust API (`Axil` methods / `QueryBuilder`), AxilQL (`axil_ql::run`), and
//! MCP tool dispatch (`axil_mcp::tools::dispatch`) — normalizes each surface's
//! raw output into one canonical [`Outcome`], and asserts:
//!
//! 1. every surface matches the fixture's `expect`, and
//! 2. every surface agrees with every other (parity).
//!
//! Deterministic semantics only: no ANN/embedding/ranking assertions. The DB
//! is opened with FTS + graph engines but NO vector/embedder, so there is no
//! model to download and no score-ordered result set to depend on. Timestamps,
//! scores, and metadata are stripped in normalization; record identity is
//! carried through symbolic setup names, never hard-coded ULIDs.
//!
//! Each op kind has a fixed set of *capable* surfaces (see
//! [`capable_surfaces`]) grounded in each surface's real API — e.g. MCP
//! `search` has no field/clause params, so field-scoped or clause-bearing FTS
//! is capable only on api+axilql. A case must run on a non-empty subset of its
//! capable set; running fewer than the full set requires a non-empty
//! `excluded_reason` (or the legacy `known_divergence`) naming why, or the
//! runner panics. Where two surfaces genuinely diverge, the fixture carries
//! that note and restricts `surfaces` to the surface whose behavior is
//! deterministic, pinning what IS.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use axil_core::{Axil, Direction, Op, QueryBuilder, Record, RecordId, SortDirection};
use axil_fts::AxilBuilderFtsExt;
use axil_graph::AxilBuilderGraphExt;
use axil_ql::{run, QueryError};
use serde_json::{json, Value};

/// A record projected to its stable, surface-invariant fields. Timestamps,
/// scores, and metadata are deliberately excluded — only identity + payload.
#[derive(Clone, Debug, PartialEq)]
struct Canon {
    id: String,
    table: String,
    data: Value,
}

/// The normalized result of one operation on one surface. Every surface's raw
/// output is mapped into one of these so surfaces can be compared for equality.
#[derive(Debug, PartialEq)]
enum Outcome {
    Records(Vec<Canon>),
    Count(usize),
    NotFound,
    Deleted,
    Error(String),
}

/// Drop engine-injected, `_`-prefixed data keys (e.g. `_importance` written by
/// auto-importance scoring) so the oracle compares only caller-controlled
/// payload. Mirrors the repo convention that `_`-prefixed names are internal.
fn strip_internal(mut data: Value) -> Value {
    if let Some(obj) = data.as_object_mut() {
        obj.retain(|k, _| !k.starts_with('_'));
    }
    data
}

fn canon_rec(r: &Record) -> Canon {
    Canon {
        id: r.id.to_string(),
        table: r.table.clone(),
        data: strip_internal(r.data.clone()),
    }
}

/// Project a record-shaped JSON object (`{id, table, data, ...}`) returned by
/// the AxilQL / MCP surfaces into a [`Canon`].
fn canon_json(v: &Value) -> Canon {
    Canon {
        id: v["id"]
            .as_str()
            .expect("record json missing id")
            .to_string(),
        table: v["table"]
            .as_str()
            .expect("record json missing table")
            .to_string(),
        data: strip_internal(v["data"].clone()),
    }
}

/// Build a `Records` outcome. Unless the case marks the result ordered, sort by
/// id so set-semantics results (WHERE filters, neighbors, traversal endpoints)
/// compare independent of surface-specific iteration order.
fn mk_records(mut v: Vec<Canon>, ordered: bool) -> Outcome {
    if !ordered {
        v.sort_by(|a, b| a.id.cmp(&b.id));
    }
    Outcome::Records(v)
}

fn parse_op(w: &Value) -> Op {
    match w["op"].as_str().expect("where op") {
        "=" => Op::Eq,
        "!=" => Op::Ne,
        ">" => Op::Gt,
        "<" => Op::Lt,
        ">=" => Op::Gte,
        "<=" => Op::Lte,
        "contains" => Op::Contains,
        other => panic!("unknown where op: {other}"),
    }
}

fn sort_dir(s: &str) -> SortDirection {
    match s {
        "asc" => SortDirection::Asc,
        "desc" => SortDirection::Desc,
        other => panic!("unknown sort dir: {other}"),
    }
}

/// Resolve an operation's target string to a concrete record-id string.
/// `__missing__` mints a fresh (valid but absent) id; `__malformed__` is a
/// syntactically invalid id; anything else is a symbolic setup name.
fn resolve_target(target: &str, names: &BTreeMap<String, RecordId>) -> String {
    match target {
        "__missing__" => RecordId::new().to_string(),
        "__malformed__" => "not-a-valid-record-id".to_string(),
        n => names
            .get(n)
            .map(|r| r.to_string())
            .unwrap_or_else(|| panic!("unknown record name: {n}")),
    }
}

/// Apply the WHERE / ORDER BY / LIMIT / OFFSET clauses of an `fts` or `scan`
/// op onto a query builder (direct-API surface).
fn apply_clauses<'a>(mut qb: QueryBuilder<'a>, op: &Value) -> QueryBuilder<'a> {
    if let Some(ws) = op.get("where").and_then(|v| v.as_array()) {
        for w in ws {
            qb = qb.where_field(
                w["field"].as_str().unwrap(),
                parse_op(w),
                w["value"].clone(),
            );
        }
    }
    if let Some(ob) = op.get("order_by") {
        qb = qb.order_by(
            ob["field"].as_str().unwrap(),
            sort_dir(ob["dir"].as_str().unwrap()),
        );
    }
    if let Some(l) = op.get("limit").and_then(|v| v.as_u64()) {
        qb = qb.limit(l as usize);
    }
    if let Some(o) = op.get("offset").and_then(|v| v.as_u64()) {
        qb = qb.offset(o as usize);
    }
    qb
}

/// Field-scoped FTS on the direct API. Mirrors the AxilQL compiler's
/// `FIND ... IN <field>` path exactly: fetch via `db.search_field` with the
/// same candidate headroom (`LIMIT.max(100)`, else 1000) so the ORDER BY sort
/// runs over the full match set, then apply WHERE → ORDER BY → OFFSET → LIMIT
/// in that order. Fetching the whole set before sorting is what keeps
/// OFFSET/LIMIT deterministic even when they slice the middle of the result —
/// unlike the plain `search_text` path, whose candidate cap is FTS-score-order
/// dependent.
fn fts_field_api(db: &Axil, query: &str, field: &str, op: &Value) -> Vec<Record> {
    let fts_limit = op
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).max(100))
        .unwrap_or(1000);
    let mut recs: Vec<Record> = db
        .search_field(query, field, fts_limit)
        .unwrap()
        .into_iter()
        .map(|(r, _)| r)
        .collect();
    if let Some(ws) = op.get("where").and_then(|v| v.as_array()) {
        let clauses: Vec<axil_core::query::WhereClause> = ws
            .iter()
            .map(|w| axil_core::query::WhereClause {
                field: w["field"].as_str().unwrap().to_string(),
                op: parse_op(w),
                value: w["value"].clone(),
            })
            .collect();
        recs.retain(|r| {
            clauses
                .iter()
                .all(|c| axil_core::query::matches_where(r, c))
        });
    }
    if let Some(ob) = op.get("order_by") {
        let sort_field = ob["field"].as_str().unwrap();
        let desc = ob["dir"].as_str().unwrap() == "desc";
        recs.sort_by(|a, b| {
            let ord = axil_core::query::compare_json_values(
                a.data.get(sort_field),
                b.data.get(sort_field),
            );
            if desc {
                ord.reverse()
            } else {
                ord
            }
        });
    }
    let offset = op.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if offset > 0 {
        recs = if offset < recs.len() {
            recs.split_off(offset)
        } else {
            Vec::new()
        };
    }
    if let Some(l) = op.get("limit").and_then(|v| v.as_u64()) {
        recs.truncate(l as usize);
    }
    recs
}

// ── Direct Rust API surface ──────────────────────────────────────────────

fn api_surface(
    db: &Axil,
    op: &Value,
    names: &BTreeMap<String, RecordId>,
    ordered: bool,
) -> Outcome {
    match op["kind"].as_str().unwrap() {
        "get" => {
            let target = resolve_target(op["target"].as_str().unwrap(), names);
            match RecordId::from_string(&target) {
                Err(_) => Outcome::Error("invalid_id".into()),
                Ok(rid) => match db.get(&rid).unwrap() {
                    Some(r) => mk_records(vec![canon_rec(&r)], ordered),
                    None => Outcome::NotFound,
                },
            }
        }
        "count" => {
            let c = match op.get("table").and_then(|v| v.as_str()) {
                Some(t) => db.count(t).unwrap(),
                None => db
                    .tables()
                    .unwrap()
                    .iter()
                    .map(|t| db.count(t).unwrap())
                    .sum(),
            };
            Outcome::Count(c)
        }
        "fts" => {
            let query = op["query"].as_str().unwrap();
            match op.get("field").and_then(|v| v.as_str()) {
                Some(field) => mk_records(
                    fts_field_api(db, query, field, op)
                        .iter()
                        .map(canon_rec)
                        .collect(),
                    ordered,
                ),
                None => {
                    let qb = apply_clauses(db.query().search_text(query), op);
                    mk_records(qb.exec().unwrap().iter().map(canon_rec).collect(), ordered)
                }
            }
        }
        "scan" => {
            let qb = apply_clauses(db.query().table(op["table"].as_str().unwrap()), op);
            mk_records(qb.exec().unwrap().iter().map(canon_rec).collect(), ordered)
        }
        "traverse" => {
            let from = names.get(op["from"].as_str().unwrap()).unwrap();
            let mut rs = db.traverse(from, op["path"].as_str().unwrap()).unwrap();
            if let Some(ws) = op.get("where").and_then(|v| v.as_array()) {
                let clauses: Vec<axil_core::query::WhereClause> = ws
                    .iter()
                    .map(|w| axil_core::query::WhereClause {
                        field: w["field"].as_str().unwrap().to_string(),
                        op: parse_op(w),
                        value: w["value"].clone(),
                    })
                    .collect();
                rs.retain(|r| {
                    clauses
                        .iter()
                        .all(|c| axil_core::query::matches_where(r, c))
                });
            }
            mk_records(rs.iter().map(canon_rec).collect(), ordered)
        }
        "neighbors" => {
            let from = names.get(op["from"].as_str().unwrap()).unwrap();
            let dir = match op["direction"].as_str().unwrap() {
                "out" => Direction::Out,
                "in" => Direction::In,
                "both" => Direction::Both,
                other => panic!("bad direction: {other}"),
            };
            let et = op.get("edge_type").and_then(|v| v.as_str());
            let rs = db.neighbors(from, et, dir).unwrap();
            mk_records(rs.iter().map(canon_rec).collect(), ordered)
        }
        "delete" => {
            let target = resolve_target(op["target"].as_str().unwrap(), names);
            match RecordId::from_string(&target) {
                Err(_) => Outcome::Error("invalid_id".into()),
                Ok(rid) => {
                    if db.delete(&rid).unwrap() {
                        Outcome::Deleted
                    } else {
                        Outcome::NotFound
                    }
                }
            }
        }
        "update" => {
            let target = names.get(op["target"].as_str().unwrap()).unwrap().clone();
            let r = db.update(&target, op["data"].clone()).unwrap();
            mk_records(vec![canon_rec(&r)], ordered)
        }
        other => panic!("op kind '{other}' not supported on api surface"),
    }
}

// ── AxilQL surface ───────────────────────────────────────────────────────

fn ql_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"{s}\""),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Number(n) => n.to_string(),
        other => panic!("unsupported where value in AxilQL rendering: {other}"),
    }
}

fn ql_where(op: &Value) -> String {
    let Some(ws) = op.get("where").and_then(|v| v.as_array()) else {
        return String::new();
    };
    if ws.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = ws
        .iter()
        .map(|w| {
            let field = w["field"].as_str().unwrap();
            let opstr = match w["op"].as_str().unwrap() {
                "contains" => "CONTAINS",
                x => x,
            };
            format!("{field} {opstr} {}", ql_value(&w["value"]))
        })
        .collect();
    format!(" WHERE {}", parts.join(" AND "))
}

fn ql_clauses(op: &Value) -> String {
    let mut s = ql_where(op);
    if let Some(ob) = op.get("order_by") {
        s.push_str(&format!(
            " ORDER BY {} {}",
            ob["field"].as_str().unwrap(),
            ob["dir"].as_str().unwrap().to_uppercase()
        ));
    }
    if let Some(l) = op.get("limit").and_then(|v| v.as_u64()) {
        s.push_str(&format!(" LIMIT {l}"));
    }
    if let Some(o) = op.get("offset").and_then(|v| v.as_u64()) {
        s.push_str(&format!(" OFFSET {o}"));
    }
    s
}

fn ql_surface(db: &Axil, op: &Value, names: &BTreeMap<String, RecordId>, ordered: bool) -> Outcome {
    match op["kind"].as_str().unwrap() {
        "get" => {
            let target = resolve_target(op["target"].as_str().unwrap(), names);
            match run(db, &format!("GET \"{target}\"")) {
                Ok(res) if res.count == 0 => Outcome::NotFound,
                Ok(res) => mk_records(res.results.iter().map(canon_json).collect(), ordered),
                // GET parses cleanly; the only compile error it can raise is an
                // unparseable record id.
                Err(QueryError::Compile(_)) => Outcome::Error("invalid_id".into()),
                Err(QueryError::Parse(_)) => Outcome::Error("parse".into()),
            }
        }
        "count" => {
            let q = match op.get("table").and_then(|v| v.as_str()) {
                Some(t) => format!("COUNT FROM {t}"),
                None => "COUNT".to_string(),
            };
            Outcome::Count(run(db, &q).unwrap().count)
        }
        "fts" => {
            let infield = op
                .get("field")
                .and_then(|v| v.as_str())
                .map(|f| format!(" IN {f}"))
                .unwrap_or_default();
            let q = format!(
                "FIND \"{}\"{infield}{}",
                op["query"].as_str().unwrap(),
                ql_clauses(op)
            );
            let res = run(db, &q).unwrap();
            mk_records(res.results.iter().map(canon_json).collect(), ordered)
        }
        "traverse" => {
            let from = names.get(op["from"].as_str().unwrap()).unwrap().to_string();
            let q = format!(
                "TRAVERSE {} FROM \"{from}\"{}",
                op["path"].as_str().unwrap(),
                ql_where(op)
            );
            let res = run(db, &q).unwrap();
            mk_records(res.results.iter().map(canon_json).collect(), ordered)
        }
        "parse" => {
            let q = op["query"].as_str().unwrap();
            match run(db, q) {
                Ok(_) => panic!("expected an error from AxilQL query: {q}"),
                Err(QueryError::Parse(_)) => Outcome::Error("parse".into()),
                Err(QueryError::Compile(_)) => Outcome::Error("compile".into()),
            }
        }
        other => panic!("op kind '{other}' not supported on axilql surface"),
    }
}

// ── MCP surface ──────────────────────────────────────────────────────────

fn mcp_text(res: &axil_mcp::protocol::ToolCallResult) -> (String, bool) {
    let is_err = res.is_error.unwrap_or(false);
    let text = res
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();
    (text, is_err)
}

fn mcp_surface(
    db: &Axil,
    op: &Value,
    names: &BTreeMap<String, RecordId>,
    ordered: bool,
) -> Outcome {
    match op["kind"].as_str().unwrap() {
        "get" => {
            let target = resolve_target(op["target"].as_str().unwrap(), names);
            let res = axil_mcp::tools::dispatch(db, "get", &json!({ "id": target }));
            let (text, is_err) = mcp_text(&res);
            if is_err {
                classify_mcp_error(&text)
            } else {
                mk_records(
                    vec![canon_json(&serde_json::from_str(&text).unwrap())],
                    ordered,
                )
            }
        }
        "fts" => {
            // MCP `search` takes only query+limit; the capability matrix keeps
            // clause-bearing (field/where/order/offset) cases off mcp, so this
            // hardcoded fallback limit is only ever reached by clause-free cases.
            let limit = op.get("limit").and_then(|v| v.as_u64()).unwrap_or(50);
            let res = axil_mcp::tools::dispatch(
                db,
                "search",
                &json!({ "query": op["query"].as_str().unwrap(), "limit": limit }),
            );
            let (text, is_err) = mcp_text(&res);
            // No fixture drives an MCP search error today; panic = loud failure
            // if one ever appears rather than a silently normalized outcome.
            assert!(!is_err, "mcp search returned error: {text}");
            let arr: Vec<Value> = serde_json::from_str(&text).unwrap();
            mk_records(arr.iter().map(canon_json).collect(), ordered)
        }
        "delete" => {
            let target = resolve_target(op["target"].as_str().unwrap(), names);
            let res = axil_mcp::tools::dispatch(db, "delete", &json!({ "id": target }));
            let (text, is_err) = mcp_text(&res);
            if is_err {
                classify_mcp_error(&text)
            } else {
                Outcome::Deleted
            }
        }
        other => panic!("op kind '{other}' not supported on mcp surface"),
    }
}

/// Normalize an MCP tool error string into the shared outcome vocabulary.
fn classify_mcp_error(text: &str) -> Outcome {
    if text.contains("not found") {
        Outcome::NotFound
    } else if text.contains("invalid record ID") {
        Outcome::Error("invalid_id".into())
    } else {
        panic!("unclassified mcp error: {text}");
    }
}

// ── Fixture driving ──────────────────────────────────────────────────────

fn build_expected(
    expect: &Value,
    op: &Value,
    recs: &BTreeMap<String, Canon>,
    ordered: bool,
    name: &str,
) -> Outcome {
    match expect["outcome"].as_str().unwrap() {
        "records" => {
            let mut v = Vec::new();
            for n in expect["records"].as_array().unwrap() {
                let nm = n.as_str().unwrap();
                let mut c = recs
                    .get(nm)
                    .unwrap_or_else(|| panic!("case '{name}': unknown record name '{nm}'"))
                    .clone();
                // `update` rewrites the record body, so the expected payload is
                // the op's new data, not the setup snapshot.
                if op["kind"] == "update" && op.get("target").and_then(|t| t.as_str()) == Some(nm) {
                    c.data = strip_internal(op["data"].clone());
                }
                v.push(c);
            }
            mk_records(v, ordered)
        }
        "count" => Outcome::Count(expect["count"].as_u64().unwrap() as usize),
        "not_found" => Outcome::NotFound,
        "deleted" => Outcome::Deleted,
        "error" => Outcome::Error(expect["kind"].as_str().unwrap().to_string()),
        other => panic!("case '{name}': unknown expected outcome '{other}'"),
    }
}

/// Build a fresh FTS+graph DB (no vector/embedder) and apply the case's setup,
/// returning the DB, its temp dir (kept alive by the caller), and the symbolic
/// name → (id, snapshot) mappings.
fn fresh_db(
    case: &Value,
    name: &str,
) -> (
    Axil,
    tempfile::TempDir,
    BTreeMap<String, RecordId>,
    BTreeMap<String, Canon>,
) {
    let dir = tempfile::tempdir().unwrap();
    let db = Axil::open(dir.path().join("oracle.axil"))
        .with_graph_engine()
        .unwrap()
        .with_fts_engine()
        .unwrap()
        .build()
        .unwrap();

    let mut names: BTreeMap<String, RecordId> = BTreeMap::new();
    let mut recs: BTreeMap<String, Canon> = BTreeMap::new();
    for step in case["setup"].as_array().unwrap() {
        if let Some(ins) = step.get("insert") {
            let rec = db
                .insert(ins["table"].as_str().unwrap(), ins["data"].clone())
                .unwrap();
            let as_name = ins["as"].as_str().unwrap().to_string();
            recs.insert(as_name.clone(), canon_rec(&rec));
            names.insert(as_name, rec.id);
        } else if let Some(rel) = step.get("relate") {
            let from = names.get(rel["from"].as_str().unwrap()).unwrap().clone();
            let to = names.get(rel["to"].as_str().unwrap()).unwrap().clone();
            db.relate(&from, rel["edge_type"].as_str().unwrap(), &to, None)
                .unwrap();
        } else {
            panic!("case '{name}': unknown setup step: {step}");
        }
    }
    (db, dir, names, recs)
}

fn eval_surface(
    surface: &str,
    db: &Axil,
    op: &Value,
    names: &BTreeMap<String, RecordId>,
    ordered: bool,
    name: &str,
) -> Outcome {
    match surface {
        "api" => api_surface(db, op, names, ordered),
        "axilql" => ql_surface(db, op, names, ordered),
        "mcp" => mcp_surface(db, op, names, ordered),
        other => panic!("case '{name}': unknown surface '{other}'"),
    }
}

/// The set of surfaces on which an op's semantics are deterministically
/// reproducible, grounded in each surface's real API. A case may run on any
/// non-empty subset of this set; a case that runs fewer than the full set must
/// document why (see [`validate_surfaces`]).
///
/// Grounding (verified against source):
/// - AxilQL supports only `RECALL`/`FIND`/`TRAVERSE`/`GET`/`COUNT` (`axil-ql`
///   `Query` enum) — no `SCAN`, `DELETE`, `NEIGHBORS`, or `UPDATE`.
/// - MCP exposes `get`/`search`/`delete` (among others) but `search` takes only
///   `query`+`limit` (`axil-mcp` `handle_search`) — no field or clause params.
/// - The direct API has `db.search_field` for field-scoped FTS, plus
///   `scan`/`neighbors`/`update` with no other surface.
fn capable_surfaces(op: &Value) -> &'static [&'static str] {
    match op["kind"].as_str().unwrap() {
        "get" => &["api", "axilql", "mcp"],
        "fts" => {
            let has_field = op.get("field").and_then(|v| v.as_str()).is_some();
            // MCP `search` takes `query` + `limit` (axil-mcp `handle_search`),
            // so a plain `limit` does NOT exclude it — only field scoping and
            // the clauses it cannot express (WHERE / ORDER BY / OFFSET) do.
            // Fixture authors: a plain-FTS `limit` below the match count makes
            // the returned SET score-order dependent on every surface — keep
            // `limit` >= matches for such cases.
            let has_clauses = op.get("where").is_some()
                || op.get("order_by").is_some()
                || op.get("offset").is_some();
            if has_field || has_clauses {
                &["api", "axilql"]
            } else {
                &["api", "axilql", "mcp"]
            }
        }
        "count" => &["api", "axilql"],
        "traverse" => &["api", "axilql"],
        "delete" => &["api", "mcp"],
        "parse" => &["axilql"],
        // No AxilQL keyword and no MCP tool: direct API only.
        "scan" | "neighbors" | "update" => &["api"],
        other => panic!("capable_surfaces: unknown op kind '{other}'"),
    }
}

/// Validate a case's `surfaces` list against the capability matrix. `surfaces`
/// must be a non-empty subset of the op's capable set; if it is a strict subset
/// the case must carry a non-empty `excluded_reason` (or the legacy
/// `known_divergence`) naming why. Also validates that either note, when
/// present, is a non-empty string.
fn validate_surfaces(case: &Value, op: &Value, name: &str) {
    let capable = capable_surfaces(op);
    let surfaces: Vec<&str> = case["surfaces"]
        .as_array()
        .unwrap_or_else(|| panic!("case '{name}': missing 'surfaces' array"))
        .iter()
        .map(|s| {
            s.as_str()
                .unwrap_or_else(|| panic!("case '{name}': non-string surface"))
        })
        .collect();
    assert!(
        !surfaces.is_empty(),
        "case '{name}': 'surfaces' must be non-empty"
    );
    for s in &surfaces {
        assert!(
            capable.contains(s),
            "case '{name}': surface '{s}' is not capable for op kind '{}' (capable: {capable:?})",
            op["kind"].as_str().unwrap()
        );
    }

    let reason = |field: &str| -> Option<&str> {
        case.get(field).map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("case '{name}': '{field}' must be a string"))
        })
    };
    let excluded_reason = reason("excluded_reason");
    let known_divergence = reason("known_divergence");
    for (field, val) in [
        ("excluded_reason", excluded_reason),
        ("known_divergence", known_divergence),
    ] {
        if let Some(text) = val {
            assert!(
                !text.trim().is_empty(),
                "case '{name}': '{field}' must be a non-empty string"
            );
        }
    }

    // A case that runs the full capable set needs no justification; a strict
    // subset must name one.
    let runs_full_set = capable.iter().all(|c| surfaces.contains(c));
    if !runs_full_set {
        assert!(
            excluded_reason.is_some() || known_divergence.is_some(),
            "case '{name}': runs {surfaces:?} but is capable of {capable:?} — add \
             'excluded_reason' (or 'known_divergence') naming why the excluded \
             surface(s) can't run"
        );
    }
}

fn run_case(case: &Value) {
    let name = case["name"].as_str().unwrap();
    let op = &case["op"];
    let expect = &case["expect"];

    validate_surfaces(case, op, name);
    let ordered = expect
        .get("ordered")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let kind = op["kind"].as_str().unwrap();

    // Destructive ops mutate state, so each surface runs against its own fresh
    // DB — otherwise the first surface's delete/update would change what the
    // next surface sees. Their outcomes (Deleted / NotFound / single updated
    // record) don't depend on cross-surface id identity, so parity still holds.
    // Read-only ops share one DB so record ids line up across surfaces.
    let destructive = matches!(kind, "delete" | "update");

    let mut outcomes: Vec<(String, Outcome)> = Vec::new();

    if destructive {
        for s in case["surfaces"].as_array().unwrap() {
            let surface = s.as_str().unwrap();
            let (db, _dir, names, recs) = fresh_db(case, name);
            let expected = build_expected(expect, op, &recs, ordered, name);
            let out = eval_surface(surface, &db, op, &names, ordered, name);
            assert_eq!(
                out, expected,
                "case '{name}' surface '{surface}': outcome mismatch\n  expected: {expected:?}\n  actual:   {out:?}"
            );
            outcomes.push((surface.to_string(), out));
        }
    } else {
        let (db, _dir, names, recs) = fresh_db(case, name);
        let expected = build_expected(expect, op, &recs, ordered, name);
        for s in case["surfaces"].as_array().unwrap() {
            let surface = s.as_str().unwrap();
            let out = eval_surface(surface, &db, op, &names, ordered, name);
            assert_eq!(
                out, expected,
                "case '{name}' surface '{surface}': outcome mismatch\n  expected: {expected:?}\n  actual:   {out:?}"
            );
            outcomes.push((surface.to_string(), out));
        }
    }

    // Explicit pairwise parity check — every surface agrees with every other.
    // (Redundant given each matched `expected`, but yields a precise message
    // that names the two disagreeing surfaces.)
    for pair in outcomes.windows(2) {
        assert_eq!(
            pair[0].1, pair[1].1,
            "case '{name}': surfaces '{}' and '{}' disagree",
            pair[0].0, pair[1].0
        );
    }
}

#[test]
fn behavior_oracle_v1() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/behavior_oracle/v1.jsonl");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));

    let mut cases = 0usize;
    let mut surface_runs = 0usize;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let case: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("fixture line {}: invalid JSON: {e}", i + 1));
        surface_runs += case["surfaces"].as_array().unwrap().len();
        run_case(&case);
        cases += 1;
    }

    assert!(
        cases >= 40,
        "expected at least 40 oracle cases, ran {cases}"
    );
    eprintln!("behavior_oracle: {cases} cases, {surface_runs} surface executions");
}
