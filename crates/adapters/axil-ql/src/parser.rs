//! Hand-written recursive descent parser for AxilQL.
//!
//! Converts a token stream into an AST. Produces clear error messages
//! with position information and suggestions.

use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

/// Parse error with position and optional suggestion.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}", self.message, self.span)?;
        if let Some(ref s) = self.suggestion {
            write!(f, " (hint: {s})")?;
        }
        Ok(())
    }
}

/// Parser state.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<&Token, ParseError> {
        let tok = self.peek();
        if std::mem::discriminant(&tok.kind) == std::mem::discriminant(expected) {
            Ok(self.advance())
        } else {
            Err(ParseError {
                message: format!("expected {expected}, found {}", tok.kind),
                span: tok.span,
                suggestion: None,
            })
        }
    }

    #[allow(dead_code)]
    fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    #[allow(dead_code)]
    fn span(&self) -> Span {
        self.peek().span
    }

    /// Parse a complete query.
    fn parse_query(&mut self) -> Result<Query, ParseError> {
        match self.peek().kind {
            TokenKind::Explain => {
                self.advance();
                let inner = self.parse_query()?;
                Ok(Query::Explain {
                    inner: Box::new(inner),
                })
            }
            TokenKind::Recall => self.parse_recall(),
            TokenKind::Find => self.parse_find(),
            TokenKind::Traverse => self.parse_traverse(),
            TokenKind::Get => self.parse_get(),
            TokenKind::Count => self.parse_count(),
            _ => {
                // Clone only on error path
                let tok = self.peek().clone();
                let suggestion = suggest_keyword(&tok);
                Err(ParseError {
                    message: format!(
                        "expected a query keyword (RECALL, FIND, TRAVERSE, GET, COUNT, EXPLAIN), found {}",
                        tok.kind
                    ),
                    span: tok.span,
                    suggestion,
                })
            }
        }
    }

    fn parse_recall(&mut self) -> Result<Query, ParseError> {
        self.advance(); // consume RECALL
        let text = self.expect_string("RECALL requires a search text")?;

        // TOP k
        self.expect(&TokenKind::Top).map_err(|mut e| {
            e.suggestion = Some("RECALL \"text\" TOP <number>".to_string());
            e
        })?;
        let top_k = self.expect_positive_int("TOP requires a positive integer")?;

        let clauses = self.parse_clauses()?;
        Ok(Query::Recall {
            text,
            top_k,
            clauses,
        })
    }

    fn parse_find(&mut self) -> Result<Query, ParseError> {
        self.advance(); // consume FIND
        let text = self.expect_string("FIND requires a search text")?;

        // Optional IN field
        let field = if matches!(self.peek().kind, TokenKind::In) {
            self.advance();
            Some(self.expect_ident("IN requires a field name")?)
        } else {
            None
        };

        let clauses = self.parse_clauses()?;
        Ok(Query::Find {
            text,
            field,
            clauses,
        })
    }

    fn parse_traverse(&mut self) -> Result<Query, ParseError> {
        self.advance(); // consume TRAVERSE
        let path = self.expect_traversal_path("TRAVERSE requires a path (e.g. ->edge->node)")?;

        // FROM is required — specifies either a record ID or table name as seed.
        if !matches!(self.peek().kind, TokenKind::From) {
            return Err(ParseError {
                message: "TRAVERSE requires FROM <table> or FROM <record_id>".to_string(),
                span: self.peek().span,
                suggestion: Some("e.g. TRAVERSE ->edge FROM my_table".to_string()),
            });
        }
        self.advance(); // consume FROM
        let from = self.expect_id_value("FROM requires a record ID or table name")?;

        let clauses = self.parse_clauses()?;
        Ok(Query::Traverse {
            path,
            from: Some(from),
            clauses,
        })
    }

    fn parse_get(&mut self) -> Result<Query, ParseError> {
        self.advance(); // consume GET
                        // Record IDs (ULIDs) can start with digits, so accept identifiers,
                        // string literals, or even integer tokens that got lexed as numbers.
        let id = self.expect_id_value("GET requires a record ID")?;
        Ok(Query::Get { id })
    }

    fn parse_count(&mut self) -> Result<Query, ParseError> {
        self.advance(); // consume COUNT
        let table = if matches!(self.peek().kind, TokenKind::From) {
            self.advance();
            Some(self.expect_ident("FROM requires a table name")?)
        } else {
            None
        };
        Ok(Query::Count { table })
    }

    /// Parse trailing clauses (WHERE, TRAVERSE, BOOST, FROM, ORDER BY, LIMIT, OFFSET, PROFILE).
    fn parse_clauses(&mut self) -> Result<Vec<Clause>, ParseError> {
        let mut clauses = Vec::new();
        loop {
            match self.peek().kind {
                TokenKind::Where => {
                    clauses.push(self.parse_where()?);
                }
                TokenKind::Traverse => {
                    self.advance();
                    let path = self.expect_traversal_path("TRAVERSE requires a path")?;
                    clauses.push(Clause::Traverse(path));
                }
                TokenKind::TraversalPath(_) => {
                    // Allow bare traversal path without TRAVERSE keyword in clause position
                    let path = self.expect_traversal_path("expected traversal path")?;
                    clauses.push(Clause::Traverse(path));
                }
                TokenKind::Boost => {
                    clauses.push(self.parse_boost()?);
                }
                TokenKind::From => {
                    self.advance();
                    let table = self.expect_ident("FROM requires a table name")?;
                    clauses.push(Clause::From(table));
                }
                TokenKind::Order => {
                    clauses.push(self.parse_order_by()?);
                }
                TokenKind::Limit => {
                    self.advance();
                    let n = self.expect_positive_int("LIMIT requires a positive integer")?;
                    clauses.push(Clause::Limit(n));
                }
                TokenKind::Offset => {
                    self.advance();
                    let n =
                        self.expect_non_negative_int("OFFSET requires a non-negative integer")?;
                    clauses.push(Clause::Offset(n));
                }
                TokenKind::Profile => {
                    self.advance();
                    clauses.push(Clause::Profile);
                }
                TokenKind::Eof => break,
                _ => {
                    let tok = self.peek();
                    return Err(ParseError {
                        message: format!("unexpected token: {}", tok.kind),
                        span: tok.span,
                        suggestion: Some(
                            "expected WHERE, TRAVERSE, BOOST, FROM, ORDER BY, LIMIT, OFFSET, or PROFILE"
                                .to_string(),
                        ),
                    });
                }
            }
        }
        Ok(clauses)
    }

    fn parse_where(&mut self) -> Result<Clause, ParseError> {
        self.advance(); // consume WHERE
        let mut conditions = vec![self.parse_condition()?];
        while matches!(self.peek().kind, TokenKind::And) {
            self.advance();
            conditions.push(self.parse_condition()?);
        }
        Ok(Clause::Where(conditions))
    }

    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let field = self.expect_ident("WHERE condition requires a field name")?;
        let op = self.parse_compare_op()?;
        let value = self.parse_condition_value()?;
        Ok(Condition { field, op, value })
    }

    fn parse_compare_op(&mut self) -> Result<CompareOp, ParseError> {
        let op = match self.peek().kind {
            TokenKind::Eq => CompareOp::Eq,
            TokenKind::Ne => CompareOp::Ne,
            TokenKind::Gt => CompareOp::Gt,
            TokenKind::Lt => CompareOp::Lt,
            TokenKind::Gte => CompareOp::Gte,
            TokenKind::Lte => CompareOp::Lte,
            TokenKind::Contains => CompareOp::Contains,
            _ => {
                let tok = self.peek();
                return Err(ParseError {
                    message: format!(
                        "expected comparison operator (=, !=, >, <, >=, <=, CONTAINS), found {}",
                        tok.kind
                    ),
                    span: tok.span,
                    suggestion: None,
                });
            }
        };
        self.advance();
        Ok(op)
    }

    fn parse_condition_value(&mut self) -> Result<ConditionValue, ParseError> {
        let val = match &self.peek().kind {
            TokenKind::StringLit(s) => Some(ConditionValue::String(s.clone())),
            TokenKind::IntLit(n) => Some(ConditionValue::Integer(*n)),
            TokenKind::FloatLit(n) => Some(ConditionValue::Float(*n)),
            TokenKind::True => Some(ConditionValue::Bool(true)),
            TokenKind::False => Some(ConditionValue::Bool(false)),
            TokenKind::Null => Some(ConditionValue::Null),
            TokenKind::Ident(s) => Some(ConditionValue::String(s.clone())),
            _ => None,
        };
        match val {
            Some(v) => {
                self.advance();
                Ok(v)
            }
            None => {
                let tok = self.peek();
                Err(ParseError {
                    message: format!(
                        "expected a value (string, number, true, false, null), found {}",
                        tok.kind
                    ),
                    span: tok.span,
                    suggestion: Some("string values must be quoted: \"value\"".to_string()),
                })
            }
        }
    }

    fn parse_boost(&mut self) -> Result<Clause, ParseError> {
        self.advance(); // consume BOOST
        let boost_type = match &self.peek().kind {
            TokenKind::Ident(s) if s.eq_ignore_ascii_case("recency") => BoostType::Recency,
            TokenKind::Ident(s) if s.eq_ignore_ascii_case("graph") => BoostType::Graph,
            TokenKind::Ident(s) if s.eq_ignore_ascii_case("feedback") => BoostType::Feedback,
            TokenKind::Ident(s) => {
                let msg = format!("unknown boost type: {s}");
                let span = self.peek().span;
                return Err(ParseError {
                    message: msg,
                    span,
                    suggestion: Some("valid boost types: recency, graph, feedback".to_string()),
                });
            }
            _ => {
                let tok = self.peek();
                return Err(ParseError {
                    message: format!(
                        "BOOST requires a type (recency, graph, feedback), found {}",
                        tok.kind
                    ),
                    span: tok.span,
                    suggestion: None,
                });
            }
        };
        self.advance();

        let weight = match self.peek().kind {
            TokenKind::FloatLit(n) => n as f32,
            TokenKind::IntLit(n) => n as f32,
            _ => {
                let tok = self.peek();
                return Err(ParseError {
                    message: format!("BOOST requires a weight (number), found {}", tok.kind),
                    span: tok.span,
                    suggestion: Some("e.g. BOOST recency 0.4".to_string()),
                });
            }
        };
        self.advance();

        Ok(Clause::Boost(boost_type, weight))
    }

    fn parse_order_by(&mut self) -> Result<Clause, ParseError> {
        self.advance(); // consume ORDER
        self.expect(&TokenKind::By).map_err(|mut e| {
            e.suggestion = Some("ORDER must be followed by BY".to_string());
            e
        })?;
        let field = self.expect_ident("ORDER BY requires a field name")?;
        let dir = match self.peek().kind {
            TokenKind::Asc => {
                self.advance();
                SortDir::Asc
            }
            TokenKind::Desc => {
                self.advance();
                SortDir::Desc
            }
            _ => SortDir::Asc, // default
        };
        Ok(Clause::OrderBy(field, dir))
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn expect_string(&mut self, ctx: &str) -> Result<String, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError {
                message: format!("{ctx}: expected a quoted string, found {}", tok.kind),
                span: tok.span,
                suggestion: Some("wrap text in double quotes: \"your text\"".to_string()),
            }),
        }
    }

    fn expect_ident(&mut self, ctx: &str) -> Result<String, ParseError> {
        let tok = self.peek().clone();
        match &tok.kind {
            TokenKind::Ident(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            // Allow string literals as identifiers (for record IDs with special chars)
            TokenKind::StringLit(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError {
                message: format!("{ctx}: expected an identifier, found {}", tok.kind),
                span: tok.span,
                suggestion: None,
            }),
        }
    }

    /// Accept an identifier, string literal, or a token that looks like an ID
    /// (e.g. ULIDs starting with digits, which the lexer tokenizes as IntLit + Ident).
    fn expect_id_value(&mut self, ctx: &str) -> Result<String, ParseError> {
        let tok = self.peek().clone();
        match &tok.kind {
            TokenKind::Ident(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            TokenKind::StringLit(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            // ULIDs like 01HZ3ABC... get split: "01" as IntLit, "HZ3ABC..." as Ident.
            // Concatenate consecutive number + ident tokens to reconstruct the full ID.
            // Use tok.text (not n.to_string()) to preserve leading zeros.
            TokenKind::IntLit(_) => {
                let mut id = tok.text.clone();
                self.advance();
                // Check if immediately followed by an identifier (no whitespace check
                // needed since the lexer already consumed consecutive chars).
                loop {
                    if let TokenKind::Ident(s) = &self.peek().kind {
                        id.push_str(s);
                        self.advance();
                    } else if let TokenKind::IntLit(_) = &self.peek().kind {
                        id.push_str(&self.peek().text.clone());
                        self.advance();
                    } else {
                        break;
                    }
                }
                Ok(id)
            }
            _ => Err(ParseError {
                message: format!(
                    "{ctx}: expected an identifier or string, found {}",
                    tok.kind
                ),
                span: tok.span,
                suggestion: None,
            }),
        }
    }

    fn expect_traversal_path(&mut self, ctx: &str) -> Result<String, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::TraversalPath(ref p) => {
                let p = p.clone();
                self.advance();
                Ok(p)
            }
            _ => Err(ParseError {
                message: format!(
                    "{ctx}: expected a traversal path (e.g. ->edge), found {}",
                    tok.kind
                ),
                span: tok.span,
                suggestion: Some(
                    "traversal paths start with -> or <- (e.g. ->modified->file)".to_string(),
                ),
            }),
        }
    }

    fn expect_positive_int(&mut self, ctx: &str) -> Result<usize, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::IntLit(n) if n > 0 => {
                self.advance();
                Ok(n as usize)
            }
            TokenKind::IntLit(n) => Err(ParseError {
                message: format!("{ctx}: {n} is not positive"),
                span: tok.span,
                suggestion: None,
            }),
            _ => Err(ParseError {
                message: format!("{ctx}: expected a positive integer, found {}", tok.kind),
                span: tok.span,
                suggestion: None,
            }),
        }
    }

    fn expect_non_negative_int(&mut self, ctx: &str) -> Result<usize, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::IntLit(n) if n >= 0 => {
                self.advance();
                Ok(n as usize)
            }
            _ => Err(ParseError {
                message: format!("{ctx}: expected a non-negative integer, found {}", tok.kind),
                span: tok.span,
                suggestion: None,
            }),
        }
    }
}

/// Suggest corrections for common mistakes.
fn suggest_keyword(tok: &Token) -> Option<String> {
    if let TokenKind::Ident(ref s) = tok.kind {
        let upper = s.to_ascii_uppercase();

        // Direct SQL keyword mappings
        match upper.as_str() {
            "SELECT" => {
                return Some(
                    "AxilQL uses RECALL for search, FIND for text search, GET for fetch by ID"
                        .to_string(),
                )
            }
            "SEARCH" => {
                return Some("use RECALL for semantic search or FIND for text search".to_string())
            }
            "FETCH" => return Some("use GET to fetch a record by ID".to_string()),
            "MATCH" => {
                return Some("use FIND for text matching or RECALL for semantic search".to_string())
            }
            "INSERT" | "UPDATE" | "DELETE" | "DROP" => {
                return Some(
                    "AxilQL is read-only; use the REST API or CLI for mutations".to_string(),
                )
            }
            _ => {}
        }

        // Fuzzy match against AxilQL keywords
        let keywords = ["RECALL", "FIND", "TRAVERSE", "GET", "COUNT", "EXPLAIN"];
        for kw in &keywords {
            if levenshtein(&upper, kw) <= 2 {
                return Some(format!("did you mean {kw}?"));
            }
        }
    }
    None
}

/// Simple Levenshtein distance for keyword suggestion.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

/// Parse an AxilQL query string into an AST.
pub fn parse(input: &str) -> Result<Query, ParseError> {
    let tokens = crate::lexer::tokenize(input).map_err(|e| ParseError {
        message: e.message,
        span: e.span,
        suggestion: None,
    })?;
    let mut parser = Parser::new(tokens);
    let query = parser.parse_query()?;

    // Ensure all input is consumed.
    if !parser.at_eof() {
        let tok = parser.peek();
        return Err(ParseError {
            message: format!("unexpected trailing token: {}", tok.kind),
            span: tok.span,
            suggestion: None,
        });
    }

    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recall_basic() {
        let q = parse(r#"RECALL "auth timeout" TOP 10"#).unwrap();
        match q {
            Query::Recall {
                text,
                top_k,
                clauses,
            } => {
                assert_eq!(text, "auth timeout");
                assert_eq!(top_k, 10);
                assert!(clauses.is_empty());
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_recall_with_from() {
        let q = parse(r#"RECALL "auth" TOP 5 FROM sessions"#).unwrap();
        match q {
            Query::Recall { clauses, .. } => {
                assert!(matches!(&clauses[0], Clause::From(t) if t == "sessions"));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_find_with_in() {
        let q = parse(r#"FIND "authentication" IN summary"#).unwrap();
        match q {
            Query::Find { text, field, .. } => {
                assert_eq!(text, "authentication");
                assert_eq!(field.as_deref(), Some("summary"));
            }
            _ => panic!("expected Find"),
        }
    }

    #[test]
    fn parse_traverse_from() {
        let q = parse("TRAVERSE ->modified->file FROM rec_01HZ3ABC").unwrap();
        match q {
            Query::Traverse { path, from, .. } => {
                assert_eq!(path, "->modified->file");
                assert_eq!(from.as_deref(), Some("rec_01HZ3ABC"));
            }
            _ => panic!("expected Traverse"),
        }
    }

    #[test]
    fn parse_get() {
        let q = parse("GET rec_01HZ3ABC").unwrap();
        assert!(matches!(q, Query::Get { id } if id == "rec_01HZ3ABC"));
    }

    #[test]
    fn parse_count() {
        let q = parse("COUNT FROM sessions").unwrap();
        assert!(matches!(q, Query::Count { table } if table.as_deref() == Some("sessions")));
    }

    #[test]
    fn parse_count_no_table() {
        let q = parse("COUNT").unwrap();
        assert!(matches!(q, Query::Count { table } if table.is_none()));
    }

    #[test]
    fn parse_where_clause() {
        let q =
            parse(r#"RECALL "auth" TOP 5 WHERE table = "sessions" AND created_at > "2026-03-01""#)
                .unwrap();
        match q {
            Query::Recall { clauses, .. } => match &clauses[0] {
                Clause::Where(conds) => {
                    assert_eq!(conds.len(), 2);
                    assert_eq!(conds[0].field, "table");
                    assert_eq!(conds[0].op, CompareOp::Eq);
                    assert_eq!(conds[1].field, "created_at");
                    assert_eq!(conds[1].op, CompareOp::Gt);
                }
                _ => panic!("expected Where clause"),
            },
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_combined() {
        let q = parse(r#"RECALL "auth error" TOP 5 TRAVERSE ->mentions WHERE table = "sessions""#)
            .unwrap();
        match q {
            Query::Recall { clauses, .. } => {
                assert!(matches!(&clauses[0], Clause::Traverse(p) if p == "->mentions"));
                assert!(matches!(&clauses[1], Clause::Where(_)));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_boost() {
        let q = parse(r#"RECALL "deploy" TOP 10 BOOST recency 0.4"#).unwrap();
        match q {
            Query::Recall { clauses, .. } => {
                assert!(
                    matches!(&clauses[0], Clause::Boost(BoostType::Recency, w) if (*w - 0.4).abs() < 0.001)
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_order_by() {
        let q =
            parse(r#"FIND "error" FROM logs ORDER BY created_at DESC LIMIT 25 OFFSET 50"#).unwrap();
        match q {
            Query::Find { clauses, .. } => {
                assert!(matches!(&clauses[0], Clause::From(t) if t == "logs"));
                assert!(
                    matches!(&clauses[1], Clause::OrderBy(f, SortDir::Desc) if f == "created_at")
                );
                assert!(matches!(&clauses[2], Clause::Limit(25)));
                assert!(matches!(&clauses[3], Clause::Offset(50)));
            }
            _ => panic!("expected Find"),
        }
    }

    #[test]
    fn parse_profile() {
        let q = parse(r#"RECALL "memory leak" TOP 10 PROFILE"#).unwrap();
        assert!(q.has_profile());
    }

    #[test]
    fn parse_explain() {
        let q = parse(r#"EXPLAIN RECALL "x" TOP 5"#).unwrap();
        assert!(matches!(q, Query::Explain { .. }));
    }

    #[test]
    fn error_missing_text() {
        let err = parse("RECALL TOP 5").unwrap_err();
        assert!(err.message.contains("quoted string"));
    }

    #[test]
    fn error_select_suggestion() {
        // SELECT without special chars triggers suggestion
        let err = parse("SELECT FROM users").unwrap_err();
        assert!(err.suggestion.is_some());
        assert!(err.suggestion.unwrap().contains("RECALL"));
    }

    #[test]
    fn case_insensitive() {
        let q = parse(r#"recall "test" top 5"#).unwrap();
        assert!(matches!(q, Query::Recall { .. }));
    }

    #[test]
    fn parse_comment() {
        let q = parse("-- search for auth bugs\nRECALL \"auth\" TOP 5").unwrap();
        assert!(matches!(q, Query::Recall { .. }));
    }
}
