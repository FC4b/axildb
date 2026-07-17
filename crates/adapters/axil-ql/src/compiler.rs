//! Compiles an AxilQL AST into QueryBuilder method calls.
//!
//! The compiler maps each AST node to the corresponding QueryBuilder method,
//! producing a `CompiledQuery` that can be executed against an `Axil` database.

use axil_core::{Axil, Op, Record, RecordId, SortDirection};

use crate::ast::*;

/// Result of compiling and executing an AxilQL query.
#[derive(Debug, serde::Serialize)]
pub struct QueryResult {
    pub results: Vec<serde_json::Value>,
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<axil_core::QueryProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<axil_core::QueryPlan>,
    pub elapsed_ms: f64,
}

/// Compilation error.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compile error: {}", self.message)
    }
}

impl From<axil_core::AxilError> for CompileError {
    fn from(e: axil_core::AxilError) -> Self {
        CompileError {
            message: e.to_string(),
        }
    }
}

/// Convert an AST ConditionValue to a serde_json::Value.
fn condition_to_json(val: &ConditionValue) -> serde_json::Value {
    match val {
        ConditionValue::String(s) => serde_json::Value::String(s.clone()),
        ConditionValue::Integer(n) => serde_json::json!(n),
        ConditionValue::Float(n) => serde_json::json!(n),
        ConditionValue::Bool(b) => serde_json::Value::Bool(*b),
        ConditionValue::Null => serde_json::Value::Null,
    }
}

/// Convert an AST CompareOp to a QueryBuilder Op.
fn compare_op_to_core(op: &CompareOp) -> Op {
    match op {
        CompareOp::Eq => Op::Eq,
        CompareOp::Ne => Op::Ne,
        CompareOp::Gt => Op::Gt,
        CompareOp::Lt => Op::Lt,
        CompareOp::Gte => Op::Gte,
        CompareOp::Lte => Op::Lte,
        CompareOp::Contains => Op::Contains,
    }
}

fn sort_direction(dir: &SortDir) -> SortDirection {
    match dir {
        SortDir::Asc => SortDirection::Asc,
        SortDir::Desc => SortDirection::Desc,
    }
}

/// Convert AST WHERE conditions to core [`WhereClause`]s.
fn conditions_to_where(conds: &[Condition]) -> Vec<axil_core::query::WhereClause> {
    conds
        .iter()
        .map(|c| axil_core::query::WhereClause {
            field: c.field.clone(),
            op: compare_op_to_core(&c.op),
            value: condition_to_json(&c.value),
        })
        .collect()
}

/// Count the records in `table` matching the given predicates.
fn filtered_count(
    db: &Axil,
    table: &str,
    wheres: &[axil_core::query::WhereClause],
) -> Result<usize, CompileError> {
    let mut qb = db.query().table(table);
    for wc in wheres {
        qb = qb.where_field(&wc.field, wc.op.clone(), wc.value.clone());
    }
    Ok(qb.exec()?.len())
}

/// Convert an AST [`AggSpec`] to the executor's [`aggregate::AggMetric`].
fn agg_spec_to_metric(s: &AggSpec) -> crate::aggregate::AggMetric {
    use crate::aggregate::AggMetric;
    match s {
        AggSpec::Count => AggMetric::Count,
        AggSpec::Avg(f) => AggMetric::Avg(f.clone()),
        AggSpec::Min(f) => AggMetric::Min(f.clone()),
        AggSpec::Max(f) => AggMetric::Max(f.clone()),
        AggSpec::Sum(f) => AggMetric::Sum(f.clone()),
    }
}

/// Convert a Record to JSON output format.
fn record_to_json(r: &Record) -> serde_json::Value {
    let mut json = serde_json::json!({
        "id": r.id.to_string(),
        "table": r.table,
        "data": r.data,
        "created_at": r.created_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "updated_at": r.updated_at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
    if let Some(ref metadata) = r.metadata {
        json.as_object_mut()
            .unwrap()
            .insert("metadata".to_string(), metadata.clone());
    }
    json
}

/// Apply AST clauses to a QueryBuilder. Returns `(builder, want_profile)`.
fn apply_clauses<'a>(
    mut qb: axil_core::QueryBuilder<'a>,
    clauses: &[Clause],
) -> (axil_core::QueryBuilder<'a>, bool) {
    let mut want_profile = false;
    for clause in clauses {
        match clause {
            Clause::Where(conditions) => {
                for cond in conditions {
                    qb = qb.where_field(
                        &cond.field,
                        compare_op_to_core(&cond.op),
                        condition_to_json(&cond.value),
                    );
                }
            }
            Clause::Traverse(path) => qb = qb.traverse(path),
            Clause::From(table) => qb = qb.table(table),
            Clause::OrderBy(field, dir) => qb = qb.order_by(field, sort_direction(dir)),
            Clause::Limit(n) => qb = qb.limit(*n),
            Clause::Offset(n) => qb = qb.offset(*n),
            Clause::Boost(bt, weight) => {
                eprintln!(
                    "warning: BOOST {:?} {:.1} clause is parsed but not yet implemented; ignoring",
                    bt, weight
                );
            }
            Clause::Profile => want_profile = true,
        }
    }
    (qb, want_profile)
}

/// Compile and execute an AxilQL query against a database.
pub fn execute(db: &Axil, query: &Query) -> Result<QueryResult, CompileError> {
    let start = std::time::Instant::now();

    match query {
        Query::Explain { inner } => execute_explain(db, inner),

        Query::Get { id } => {
            let rid = RecordId::from_string(id)?;
            match db.get(&rid)? {
                Some(record) => {
                    let json = record_to_json(&record);
                    Ok(QueryResult {
                        count: 1,
                        results: vec![json],
                        profile: None,
                        plan: None,
                        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    })
                }
                None => Ok(QueryResult {
                    count: 0,
                    results: vec![],
                    profile: None,
                    plan: None,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                }),
            }
        }

        Query::Count {
            table,
            where_conditions,
        } => {
            let count = if where_conditions.is_empty() {
                match table {
                    Some(t) => db.count(t)?,
                    None => {
                        let tables = db.tables()?;
                        let mut total = 0usize;
                        for t in &tables {
                            total += db.count(t)?;
                        }
                        total
                    }
                }
            } else {
                let wheres = conditions_to_where(where_conditions);
                match table {
                    Some(t) => filtered_count(db, t, &wheres)?,
                    None => {
                        let tables = db.tables()?;
                        let mut total = 0usize;
                        for t in &tables {
                            total += filtered_count(db, t, &wheres)?;
                        }
                        total
                    }
                }
            };
            Ok(QueryResult {
                count,
                results: vec![serde_json::json!({"count": count})],
                profile: None,
                plan: None,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            })
        }

        Query::Agg {
            table,
            metrics,
            where_conditions,
            group_by,
        } => {
            let wheres = conditions_to_where(where_conditions);
            let agg_metrics: Vec<crate::aggregate::AggMetric> =
                metrics.iter().map(agg_spec_to_metric).collect();
            let req = crate::aggregate::AggRequest {
                table,
                metrics: &agg_metrics,
                group_by: group_by.as_deref(),
                where_clauses: &wheres,
                include_archived: false,
            };
            let value = crate::aggregate::aggregate(db, &req)
                .map_err(|e| CompileError { message: e.message })?;
            Ok(QueryResult {
                results: vec![value],
                count: 1,
                profile: None,
                plan: None,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            })
        }

        Query::Recall {
            text,
            top_k,
            clauses,
        } => execute_search(db, clauses, start, |qb| qb.similar_to(text, *top_k)),

        Query::Find {
            text,
            ref field,
            clauses,
        } => {
            if let Some(f) = field {
                // Field-scoped FTS: use db.search_field() for tokenized matching,
                // then apply clauses (WHERE, LIMIT, ORDER BY, etc.) to results.
                // Use the query's LIMIT as the FTS fetch size (with headroom for
                // WHERE filtering), falling back to a default cap.
                let fts_limit = clauses
                    .iter()
                    .find_map(|c| {
                        if let Clause::Limit(n) = c {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .map(|n| n.max(100)) // at least 100 for WHERE filtering headroom
                    .unwrap_or(1000);
                let fts_results =
                    db.search_field(text, f, fts_limit)
                        .map_err(|e| CompileError {
                            message: e.to_string(),
                        })?;
                let mut records: Vec<Record> = fts_results.into_iter().map(|(r, _)| r).collect();
                // Apply clauses to FTS results (same pattern as TRAVERSE).
                let mut limit = None;
                let mut offset = 0usize;
                for clause in clauses {
                    match clause {
                        Clause::Where(conditions) => {
                            records.retain(|r| {
                                conditions.iter().all(|cond| {
                                    let wc = axil_core::query::WhereClause {
                                        field: cond.field.clone(),
                                        op: compare_op_to_core(&cond.op),
                                        value: condition_to_json(&cond.value),
                                    };
                                    axil_core::query::matches_where(r, &wc)
                                })
                            });
                        }
                        Clause::Limit(n) => limit = Some(*n),
                        Clause::Offset(n) => offset = *n,
                        Clause::OrderBy(field, dir) => {
                            let desc = matches!(dir, SortDir::Desc);
                            records.sort_by(|a, b| {
                                let va = a.data.get(field.as_str());
                                let vb = b.data.get(field.as_str());
                                let ord = axil_core::query::compare_json_values(va, vb);
                                if desc {
                                    ord.reverse()
                                } else {
                                    ord
                                }
                            });
                        }
                        _ => {}
                    }
                }
                if offset > 0 {
                    records = if offset < records.len() {
                        records.split_off(offset)
                    } else {
                        Vec::new()
                    };
                }
                if let Some(n) = limit {
                    records.truncate(n);
                }
                let count = records.len();
                let values: Vec<serde_json::Value> = records.iter().map(record_to_json).collect();
                Ok(QueryResult {
                    results: values,
                    count,
                    profile: None,
                    plan: None,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                })
            } else {
                execute_search(db, clauses, start, |qb| qb.search_text(text))
            }
        }

        Query::Traverse {
            path,
            from,
            clauses,
        } => {
            match from {
                Some(ref from_val) => {
                    // Try as a record ID first; fall back to table name.
                    if let Ok(rid) = RecordId::from_string(from_val) {
                        // Direct record-seeded traversal via Axil::traverse(),
                        // then apply clauses (WHERE, LIMIT, OFFSET) to results.
                        let mut records = db.traverse(&rid, path).map_err(|e| CompileError {
                            message: e.to_string(),
                        })?;
                        let mut limit = None;
                        let mut offset = 0usize;
                        for clause in clauses {
                            match clause {
                                Clause::Where(conditions) => {
                                    records.retain(|r| {
                                        conditions.iter().all(|cond| {
                                            let wc = axil_core::query::WhereClause {
                                                field: cond.field.clone(),
                                                op: compare_op_to_core(&cond.op),
                                                value: condition_to_json(&cond.value),
                                            };
                                            axil_core::query::matches_where(r, &wc)
                                        })
                                    });
                                }
                                Clause::Limit(n) => limit = Some(*n),
                                Clause::Offset(n) => offset = *n,
                                Clause::OrderBy(field, dir) => {
                                    let desc = matches!(dir, SortDir::Desc);
                                    records.sort_by(|a, b| {
                                        let va = a.data.get(field.as_str());
                                        let vb = b.data.get(field.as_str());
                                        let ord = axil_core::query::compare_json_values(va, vb);
                                        if desc {
                                            ord.reverse()
                                        } else {
                                            ord
                                        }
                                    });
                                }
                                _ => {}
                            }
                        }
                        if offset > 0 {
                            records = if offset < records.len() {
                                records.split_off(offset)
                            } else {
                                Vec::new()
                            };
                        }
                        if let Some(n) = limit {
                            records.truncate(n);
                        }
                        let count = records.len();
                        let values: Vec<serde_json::Value> =
                            records.iter().map(record_to_json).collect();
                        Ok(QueryResult {
                            results: values,
                            count,
                            profile: None,
                            plan: None,
                            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                        })
                    } else {
                        // Treat as table name — fan-out traversal from all records in table.
                        execute_search(db, clauses, start, |qb| qb.table(from_val).traverse(path))
                    }
                }
                None => {
                    return Err(CompileError {
                        message: "TRAVERSE requires FROM <table> or FROM <record_id> to specify starting point".to_string(),
                    });
                }
            }
        }
    }
}

/// Build and execute a query with the given initial operation and clauses.
fn execute_search<'a, F>(
    db: &'a Axil,
    clauses: &[Clause],
    start: std::time::Instant,
    init: F,
) -> Result<QueryResult, CompileError>
where
    F: FnOnce(axil_core::QueryBuilder<'a>) -> axil_core::QueryBuilder<'a>,
{
    let (qb, want_profile) = apply_clauses(init(db.query()), clauses);

    if want_profile {
        let (results, profile) = qb.exec_profiled().map_err(|e| CompileError {
            message: e.to_string(),
        })?;
        let count = results.len();
        let values: Vec<serde_json::Value> = results.iter().map(record_to_json).collect();
        Ok(QueryResult {
            results: values,
            count,
            profile: Some(profile),
            plan: None,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    } else {
        let results = qb.exec().map_err(|e| CompileError {
            message: e.to_string(),
        })?;
        let count = results.len();
        let values: Vec<serde_json::Value> = results.iter().map(record_to_json).collect();
        Ok(QueryResult {
            results: values,
            count,
            profile: None,
            plan: None,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// Execute EXPLAIN: return the query plan without executing.
fn execute_explain(db: &Axil, query: &Query) -> Result<QueryResult, CompileError> {
    let start = std::time::Instant::now();

    let plan = match query {
        Query::Get { .. } => axil_core::QueryPlan {
            plan: vec![axil_core::PlanStep {
                step: 1,
                step_type: "get_by_id".to_string(),
                params: serde_json::json!({"type": "direct_lookup"}),
            }],
            estimated_cost: axil_core::EstimatedCost::Low,
        },
        Query::Count { table, .. } => axil_core::QueryPlan {
            plan: vec![axil_core::PlanStep {
                step: 1,
                step_type: "count".to_string(),
                params: serde_json::json!({"table": table}),
            }],
            estimated_cost: axil_core::EstimatedCost::Low,
        },
        Query::Agg {
            table, group_by, ..
        } => axil_core::QueryPlan {
            plan: vec![axil_core::PlanStep {
                step: 1,
                step_type: "aggregate".to_string(),
                params: serde_json::json!({"table": table, "group_by": group_by}),
            }],
            estimated_cost: axil_core::EstimatedCost::Medium,
        },
        Query::Recall {
            text,
            top_k,
            clauses,
        } => {
            let (qb, _) = apply_clauses(db.query().similar_to(text, *top_k), clauses);
            qb.explain()
        }
        Query::Find {
            text,
            ref field,
            clauses,
        } => {
            let mut qb = db.query().search_text(text);
            if let Some(f) = field {
                qb = qb.where_field(f, Op::Contains, serde_json::Value::String(text.clone()));
            }
            let (qb, _) = apply_clauses(qb, clauses);
            qb.explain()
        }
        Query::Traverse {
            path,
            from,
            clauses,
        } => {
            match from {
                Some(ref from_val) => {
                    if RecordId::from_string(from_val).is_ok() {
                        // Record-seeded traversal — show a direct plan
                        axil_core::QueryPlan {
                            plan: vec![
                                axil_core::PlanStep {
                                    step: 1,
                                    step_type: "record_lookup".to_string(),
                                    params: serde_json::json!({"id": from_val}),
                                },
                                axil_core::PlanStep {
                                    step: 2,
                                    step_type: "graph_traverse".to_string(),
                                    params: serde_json::json!({"path": path}),
                                },
                            ],
                            estimated_cost: axil_core::EstimatedCost::Medium,
                        }
                    } else {
                        let (qb, _) =
                            apply_clauses(db.query().table(from_val).traverse(path), clauses);
                        qb.explain()
                    }
                }
                None => {
                    return Err(CompileError {
                        message: "TRAVERSE requires FROM <table> or FROM <record_id>".to_string(),
                    });
                }
            }
        }
        Query::Explain { inner } => {
            return execute_explain(db, inner);
        }
    };

    Ok(QueryResult {
        results: vec![serde_json::to_value(&plan).unwrap_or_default()],
        count: 0,
        profile: None,
        plan: Some(plan),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn setup_db() -> (tempfile::TempDir, Axil) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        (dir, db)
    }

    #[test]
    fn compile_and_execute_count() {
        let (_dir, db) = setup_db();
        db.insert("sessions", serde_json::json!({"summary": "test"}))
            .unwrap();
        db.insert("sessions", serde_json::json!({"summary": "test2"}))
            .unwrap();

        let query = parser::parse("COUNT FROM sessions").unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 2);
    }

    #[test]
    fn compile_and_execute_get() {
        let (_dir, db) = setup_db();
        let record = db
            .insert("sessions", serde_json::json!({"summary": "test"}))
            .unwrap();
        let id = record.id.to_string();

        let query = parser::parse(&format!(r#"GET "{id}""#)).unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 1);
        assert_eq!(result.results[0]["data"]["summary"], "test");
    }

    #[test]
    fn compile_and_execute_get_not_found() {
        let (_dir, db) = setup_db();
        let fake_id = axil_core::RecordId::new();
        let query = parser::parse(&format!(r#"GET "{fake_id}""#)).unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 0);
    }

    #[test]
    fn compile_and_execute_explain() {
        let (_dir, db) = setup_db();
        let query = parser::parse(r#"EXPLAIN RECALL "test" TOP 5"#).unwrap();
        let result = execute(&db, &query).unwrap();
        assert!(result.plan.is_some());
    }

    #[test]
    fn compile_count_all_tables() {
        let (_dir, db) = setup_db();
        db.insert("a", serde_json::json!({"x": 1})).unwrap();
        db.insert("b", serde_json::json!({"x": 2})).unwrap();

        let query = parser::parse("COUNT").unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 2);
    }

    #[test]
    fn compile_explain_find() {
        let (_dir, db) = setup_db();
        let query = parser::parse(r#"EXPLAIN FIND "error" FROM logs"#).unwrap();
        let result = execute(&db, &query).unwrap();
        assert!(result.plan.is_some());
    }

    #[test]
    fn compile_where_filter_on_list() {
        let (_dir, db) = setup_db();
        db.insert("logs", serde_json::json!({"level": "error", "msg": "fail"}))
            .unwrap();
        db.insert("logs", serde_json::json!({"level": "info", "msg": "ok"}))
            .unwrap();

        let query = parser::parse("COUNT FROM logs").unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 2);
    }

    #[test]
    fn compile_count_with_where() {
        let (_dir, db) = setup_db();
        db.insert("logs", serde_json::json!({"level": "error"})).unwrap();
        db.insert("logs", serde_json::json!({"level": "error"})).unwrap();
        db.insert("logs", serde_json::json!({"level": "info"})).unwrap();

        let query = parser::parse(r#"COUNT FROM logs WHERE level = "error""#).unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.count, 2);
    }

    #[test]
    fn compile_agg_group_by() {
        let (_dir, db) = setup_db();
        db.insert("t", serde_json::json!({"family": "meanrev", "decay": 2.0})).unwrap();
        db.insert("t", serde_json::json!({"family": "meanrev", "decay": 4.0})).unwrap();
        db.insert("t", serde_json::json!({"family": "momentum", "decay": 9.0})).unwrap();

        let query = parser::parse("AGG count, avg(decay) FROM t GROUP BY family").unwrap();
        let result = execute(&db, &query).unwrap();
        // The Agg result is a single envelope value.
        assert_eq!(result.count, 1);
        let env = &result.results[0];
        assert_eq!(env["table"], "t");
        assert_eq!(env["group_by"], "family");
        assert_eq!(env["total_rows"], 3);
        let groups = env["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0]["group"], "meanrev");
        assert_eq!(groups[0]["count"], 2);
        assert_eq!(groups[0]["avg_decay"], 3.0);
    }

    #[test]
    fn compile_agg_with_where() {
        let (_dir, db) = setup_db();
        db.insert("t", serde_json::json!({"family": "meanrev", "sharpe": 0.5})).unwrap();
        db.insert("t", serde_json::json!({"family": "meanrev", "sharpe": 0.1})).unwrap();

        let query = parser::parse("AGG count FROM t WHERE sharpe > 0.3").unwrap();
        let result = execute(&db, &query).unwrap();
        assert_eq!(result.results[0]["total_rows"], 1);
    }
}
