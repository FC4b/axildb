//! Aggregation executor — the single fold behind the AxilQL `AGG` statement,
//! the CLI `axil agg` command, and the MCP `aggregate` tool.
//!
//! It consumes the record stream from a filtered table query
//! (`db.query().table(t).where_field(..).exec()`) and folds it into per-group
//! accumulators (`count` / `avg` / `min` / `max` / `sum`), emitting a stable
//! JSON envelope. Numeric extraction is via [`serde_json::Value::as_f64`];
//! non-numeric or missing values are skipped for `avg`/`min`/`max`/`sum` and
//! surfaced per group as `skipped`. Groups are sorted by key for determinism.

use std::collections::BTreeMap;

use axil_core::query::WhereClause;
use axil_core::Axil;
use serde_json::{json, Value};

/// A single aggregation metric requested over a table.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AggMetric {
    /// Row count for the group (always emitted regardless, but requesting it
    /// documents intent — e.g. a kill-reason histogram).
    Count,
    /// Mean of a numeric field.
    Avg(String),
    /// Minimum of a numeric field.
    Min(String),
    /// Maximum of a numeric field.
    Max(String),
    /// Sum of a numeric field.
    Sum(String),
}

impl AggMetric {
    /// Parse a metric spec string: `count` | `avg(field)` | `min(field)` |
    /// `max(field)` | `sum(field)` (function names case-insensitive).
    ///
    /// This is the one string form of the metric micro-grammar, shared by
    /// every surface that accepts specs as text (the MCP `aggregate` tool,
    /// clients); AxilQL parses the same grammar from its own token stream.
    pub fn parse_spec(spec: &str) -> Result<Self, String> {
        let s = spec.trim();
        if s.eq_ignore_ascii_case("count") || s.eq_ignore_ascii_case("count()") {
            return Ok(AggMetric::Count);
        }
        let (func, rest) = match s.find('(') {
            Some(open) if s.ends_with(')') => (&s[..open], &s[open + 1..s.len() - 1]),
            _ => return Err(format!("expected `count` or `func(field)`, got '{spec}'")),
        };
        let field = rest.trim();
        if field.is_empty() {
            return Err(format!("field name must not be empty in '{spec}'"));
        }
        match func.trim().to_ascii_lowercase().as_str() {
            "avg" => Ok(AggMetric::Avg(field.to_string())),
            "min" => Ok(AggMetric::Min(field.to_string())),
            "max" => Ok(AggMetric::Max(field.to_string())),
            "sum" => Ok(AggMetric::Sum(field.to_string())),
            other => Err(format!(
                "unknown function '{other}' (valid: count, avg, min, max, sum)"
            )),
        }
    }
}

/// A fully-specified aggregation request.
#[derive(Debug, Clone)]
pub struct AggRequest<'a> {
    /// Table to aggregate over.
    pub table: &'a str,
    /// Requested metrics; the field-bearing ones extract via `as_f64`.
    pub metrics: &'a [AggMetric],
    /// Optional group-by field. Missing/null values fall into the `null` group.
    pub group_by: Option<&'a str>,
    /// WHERE predicates applied before folding (AND-composed).
    pub where_clauses: &'a [WhereClause],
    /// Include `_archived` records (excluded by default, mirroring `list`).
    pub include_archived: bool,
}

/// Running numeric accumulator for one field-bearing metric.
#[derive(Clone)]
struct NumAcc {
    n: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl NumAcc {
    fn new() -> Self {
        NumAcc {
            n: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn add(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }
}

/// Per-group accumulator.
struct GroupAcc {
    /// Original group-by value for output (or `Null` for the null group).
    repr: Value,
    count: usize,
    /// Rows skipped for at least one numeric metric (non-numeric/missing field).
    skipped: usize,
    /// One numeric accumulator per requested metric (unused slot for `Count`).
    nums: Vec<NumAcc>,
}

impl GroupAcc {
    fn new(n_metrics: usize, repr: Value) -> Self {
        GroupAcc {
            repr,
            count: 0,
            skipped: 0,
            nums: vec![NumAcc::new(); n_metrics],
        }
    }
}

/// Render a group-by field value as a stable string key. Missing or JSON-null
/// values map to `None` (the null group). The key is type-discriminated so a
/// number and its string rendering (`1` vs `"1"`) stay separate groups — the
/// key is internal only; output uses the group's original `repr` value.
fn render_key(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(format!("s:{s}")),
        Some(other) => Some(format!("j:{other}")),
    }
}

/// Run an aggregation and return the stable JSON envelope:
///
/// ```json
/// {
///   "table": "autopsies",
///   "group_by": "kill_reason",
///   "groups": [
///     {"group": "drawdown", "count": 3, "avg_oos_sharpe": 1.2, "skipped": 0}
///   ],
///   "total_rows": 3
/// }
/// ```
///
/// Groups are ordered by key (the `null` group first). A field-bearing metric
/// with no numeric samples in a group renders as JSON `null`.
pub fn aggregate(db: &Axil, req: &AggRequest) -> axil_core::Result<Value> {
    let mut qb = db.query().table(req.table);
    for wc in req.where_clauses {
        qb = qb.where_field(&wc.field, wc.op.clone(), wc.value.clone());
    }
    let records = qb.exec()?;

    let n_metrics = req.metrics.len();
    let mut groups: BTreeMap<Option<String>, GroupAcc> = BTreeMap::new();
    let mut total_rows = 0usize;

    for rec in &records {
        if !req.include_archived
            && rec
                .data
                .get("_archived")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            continue;
        }
        total_rows += 1;

        let field_val = req.group_by.and_then(|f| rec.data.get(f));
        let key = render_key(field_val);

        let acc = groups.entry(key).or_insert_with(|| {
            GroupAcc::new(n_metrics, field_val.cloned().unwrap_or(Value::Null))
        });
        acc.count += 1;

        let mut row_had_skip = false;
        for (mi, metric) in req.metrics.iter().enumerate() {
            let field = match metric {
                AggMetric::Count => continue,
                AggMetric::Avg(f) | AggMetric::Min(f) | AggMetric::Max(f) | AggMetric::Sum(f) => f,
            };
            match rec.data.get(field).and_then(|v| v.as_f64()) {
                Some(x) => acc.nums[mi].add(x),
                None => row_had_skip = true,
            }
        }
        if row_had_skip {
            acc.skipped += 1;
        }
    }

    let group_rows: Vec<Value> = groups
        .values()
        .map(|acc| {
            let mut obj = serde_json::Map::new();
            obj.insert("group".to_string(), acc.repr.clone());
            obj.insert("count".to_string(), json!(acc.count));
            for (mi, metric) in req.metrics.iter().enumerate() {
                let acc_num = &acc.nums[mi];
                let has = acc_num.n > 0;
                match metric {
                    AggMetric::Count => {}
                    AggMetric::Avg(f) => {
                        let v = if has {
                            json!(acc_num.sum / acc_num.n as f64)
                        } else {
                            Value::Null
                        };
                        obj.insert(format!("avg_{f}"), v);
                    }
                    AggMetric::Sum(f) => {
                        let v = if has { json!(acc_num.sum) } else { Value::Null };
                        obj.insert(format!("sum_{f}"), v);
                    }
                    AggMetric::Min(f) => {
                        let v = if has { json!(acc_num.min) } else { Value::Null };
                        obj.insert(format!("min_{f}"), v);
                    }
                    AggMetric::Max(f) => {
                        let v = if has { json!(acc_num.max) } else { Value::Null };
                        obj.insert(format!("max_{f}"), v);
                    }
                }
            }
            obj.insert("skipped".to_string(), json!(acc.skipped));
            Value::Object(obj)
        })
        .collect();

    Ok(json!({
        "table": req.table,
        "group_by": req.group_by,
        "groups": group_rows,
        "total_rows": total_rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> (tempfile::TempDir, Axil) {
        let dir = tempfile::tempdir().unwrap();
        let db = Axil::open(dir.path().join("agg.axil")).build().unwrap();
        (dir, db)
    }

    fn req<'a>(
        table: &'a str,
        metrics: &'a [AggMetric],
        group_by: Option<&'a str>,
        wheres: &'a [WhereClause],
    ) -> AggRequest<'a> {
        AggRequest {
            table,
            metrics,
            group_by,
            where_clauses: wheres,
            include_archived: false,
        }
    }

    #[test]
    fn count_and_avg_per_group() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"family": "meanrev", "decay": 2.0})).unwrap();
        db.insert("t", json!({"family": "meanrev", "decay": 4.0})).unwrap();
        db.insert("t", json!({"family": "momentum", "decay": 9.0})).unwrap();

        let metrics = [AggMetric::Count, AggMetric::Avg("decay".into())];
        let out = aggregate(&db, &req("t", &metrics, Some("family"), &[])).unwrap();

        assert_eq!(out["table"], "t");
        assert_eq!(out["group_by"], "family");
        assert_eq!(out["total_rows"], 3);
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        // Groups are sorted by key: "meanrev" < "momentum".
        assert_eq!(groups[0]["group"], "meanrev");
        assert_eq!(groups[0]["count"], 2);
        assert_eq!(groups[0]["avg_decay"], 3.0);
        assert_eq!(groups[0]["skipped"], 0);
        assert_eq!(groups[1]["group"], "momentum");
        assert_eq!(groups[1]["count"], 1);
        assert_eq!(groups[1]["avg_decay"], 9.0);
    }

    #[test]
    fn min_max_sum_exact() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"x": 1.0})).unwrap();
        db.insert("t", json!({"x": 5.0})).unwrap();
        db.insert("t", json!({"x": 3.0})).unwrap();

        let metrics = [
            AggMetric::Min("x".into()),
            AggMetric::Max("x".into()),
            AggMetric::Sum("x".into()),
        ];
        let out = aggregate(&db, &req("t", &metrics, None, &[])).unwrap();
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 1);
        // No group-by → single null group.
        assert_eq!(groups[0]["group"], Value::Null);
        assert_eq!(groups[0]["min_x"], 1.0);
        assert_eq!(groups[0]["max_x"], 5.0);
        assert_eq!(groups[0]["sum_x"], 9.0);
        assert_eq!(groups[0]["count"], 3);
    }

    #[test]
    fn empty_table_has_no_groups() {
        let (_d, db) = setup_db();
        let metrics = [AggMetric::Count];
        let out = aggregate(&db, &req("t", &metrics, Some("family"), &[])).unwrap();
        assert_eq!(out["total_rows"], 0);
        assert_eq!(out["groups"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn all_non_numeric_field_is_skipped_and_avg_null() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"x": "nope"})).unwrap();
        db.insert("t", json!({"x": "nan"})).unwrap();

        let metrics = [AggMetric::Avg("x".into())];
        let out = aggregate(&db, &req("t", &metrics, None, &[])).unwrap();
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0]["skipped"], 2);
        assert_eq!(groups[0]["avg_x"], Value::Null);
        // count still counts every row.
        assert_eq!(groups[0]["count"], 2);
    }

    #[test]
    fn missing_group_by_field_becomes_null_group() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"family": "meanrev", "x": 1.0})).unwrap();
        db.insert("t", json!({"x": 2.0})).unwrap(); // no `family`

        let metrics = [AggMetric::Count];
        let out = aggregate(&db, &req("t", &metrics, Some("family"), &[])).unwrap();
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        // BTreeMap<Option<String>>: None (null group) sorts before Some.
        assert_eq!(groups[0]["group"], Value::Null);
        assert_eq!(groups[0]["count"], 1);
        assert_eq!(groups[1]["group"], "meanrev");
        assert_eq!(groups[1]["count"], 1);
    }

    #[test]
    fn parse_spec_accepts_every_surface_form() {
        assert_eq!(AggMetric::parse_spec("count").unwrap(), AggMetric::Count);
        assert_eq!(AggMetric::parse_spec("COUNT()").unwrap(), AggMetric::Count);
        assert_eq!(
            AggMetric::parse_spec("avg(oos_sharpe)").unwrap(),
            AggMetric::Avg("oos_sharpe".into())
        );
        assert_eq!(
            AggMetric::parse_spec(" SUM( fees ) ").unwrap(),
            AggMetric::Sum("fees".into())
        );
        assert!(AggMetric::parse_spec("median(x)").is_err());
        assert!(AggMetric::parse_spec("avg()").is_err());
        assert!(AggMetric::parse_spec("avg").is_err());
    }

    #[test]
    fn group_keys_do_not_collide_across_types() {
        let (_d, db) = setup_db();
        // Number 1 and string "1" are different facet values — they must not
        // merge into one group just because they render to the same text.
        db.insert("t", json!({"g": 1, "x": 10.0})).unwrap();
        db.insert("t", json!({"g": "1", "x": 20.0})).unwrap();

        let metrics = [AggMetric::Count];
        let out = aggregate(&db, &req("t", &metrics, Some("g"), &[])).unwrap();
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        for g in groups {
            assert_eq!(g["count"], 1);
        }
    }

    #[test]
    fn where_filters_before_folding() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"family": "meanrev", "sharpe": 0.5})).unwrap();
        db.insert("t", json!({"family": "meanrev", "sharpe": 0.1})).unwrap();

        let wheres = [WhereClause {
            field: "sharpe".into(),
            op: axil_core::Op::Gt,
            value: json!(0.3),
        }];
        let metrics = [AggMetric::Count];
        let out = aggregate(&db, &req("t", &metrics, Some("family"), &wheres)).unwrap();
        assert_eq!(out["total_rows"], 1);
        let groups = out["groups"].as_array().unwrap();
        assert_eq!(groups[0]["count"], 1);
    }

    #[test]
    fn include_archived_changes_count() {
        let (_d, db) = setup_db();
        db.insert("t", json!({"family": "meanrev"})).unwrap();
        db.insert("t", json!({"family": "meanrev", "_archived": true})).unwrap();

        let metrics = [AggMetric::Count];
        let excluded = aggregate(&db, &req("t", &metrics, None, &[])).unwrap();
        assert_eq!(excluded["total_rows"], 1);

        let inc = AggRequest {
            include_archived: true,
            ..req("t", &metrics, None, &[])
        };
        let included = aggregate(&db, &inc).unwrap();
        assert_eq!(included["total_rows"], 2);
    }
}
