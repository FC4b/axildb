//! Abstract Syntax Tree types for AxilQL.

use serde::{Deserialize, Serialize};

/// A parsed AxilQL query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Query {
    /// Semantic vector search: `RECALL "text" TOP k`
    Recall {
        text: String,
        top_k: usize,
        clauses: Vec<Clause>,
    },
    /// Full-text search: `FIND "text" [IN field]`
    Find {
        text: String,
        field: Option<String>,
        clauses: Vec<Clause>,
    },
    /// Graph traversal: `TRAVERSE ->edge [FROM id]`
    Traverse {
        path: String,
        from: Option<String>,
        clauses: Vec<Clause>,
    },
    /// Fetch by ID: `GET id`
    Get { id: String },
    /// Count records: `COUNT [FROM table] [WHERE ...]`
    Count {
        table: Option<String>,
        /// Optional predicates; empty means count every record.
        where_conditions: Vec<Condition>,
    },
    /// Aggregate: `AGG <spec>[, ...] FROM <table> [WHERE ...] [GROUP BY <field>]`
    Agg {
        table: String,
        metrics: Vec<AggSpec>,
        where_conditions: Vec<Condition>,
        group_by: Option<String>,
    },
    /// Show query plan: `EXPLAIN <query>`
    Explain { inner: Box<Query> },
}

/// A single aggregation spec in an `AGG` statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggSpec {
    /// `count`
    Count,
    /// `avg(field)`
    Avg(String),
    /// `min(field)`
    Min(String),
    /// `max(field)`
    Max(String),
    /// `sum(field)`
    Sum(String),
}

/// A clause that modifies the primary operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Clause {
    /// `WHERE field op value [AND ...]`
    Where(Vec<Condition>),
    /// `TRAVERSE ->path` (chained after primary op)
    Traverse(String),
    /// `BOOST type weight`
    Boost(BoostType, f32),
    /// `FROM table`
    From(String),
    /// `ORDER BY field [ASC|DESC]`
    OrderBy(String, SortDir),
    /// `LIMIT n`
    Limit(usize),
    /// `OFFSET n`
    Offset(usize),
    /// `PROFILE`
    Profile,
}

/// A single WHERE condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub op: CompareOp,
    pub value: ConditionValue,
}

/// Comparison operators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
}

/// A value in a WHERE condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConditionValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Null,
}

/// Boost types for scoring adjustment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoostType {
    Recency,
    Graph,
    Feedback,
}

/// Sort direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortDir {
    Asc,
    Desc,
}

impl Query {
    /// Get the clauses of this query (empty for Get/Count/Explain).
    pub fn clauses(&self) -> &[Clause] {
        match self {
            Query::Recall { clauses, .. }
            | Query::Find { clauses, .. }
            | Query::Traverse { clauses, .. } => clauses,
            Query::Get { .. }
            | Query::Count { .. }
            | Query::Agg { .. }
            | Query::Explain { .. } => &[],
        }
    }

    /// Check if PROFILE is requested.
    pub fn has_profile(&self) -> bool {
        self.clauses().iter().any(|c| matches!(c, Clause::Profile))
    }
}
