//! AxilQL — A minimal, verb-first query language for Axil.
//!
//! AxilQL is a lightweight DSL that compiles to the existing `QueryBuilder` API.
//! It supports 15 keywords: `RECALL`, `FIND`, `TRAVERSE`, `GET`, `COUNT`,
//! `WHERE`, `AND`, `FROM`, `TOP`, `LIMIT`, `OFFSET`, `ORDER BY`, `BOOST`,
//! `PROFILE`, and `IN`.
//!
//! # Example
//!
//! ```text
//! RECALL "auth timeout bug" TOP 10
//! FIND "authentication" IN summary
//! RECALL "auth error" TOP 5 TRAVERSE ->mentions WHERE table = "sessions"
//! EXPLAIN RECALL "test" TOP 5
//! ```
//!
//! # Architecture
//!
//! ```text
//! Input string → Lexer (tokenize) → Parser (AST) → Compiler (QueryBuilder) → Results
//! ```

pub mod ast;
pub mod compiler;
pub mod lexer;
pub mod parser;

// Re-export primary API.
pub use ast::Query;
pub use compiler::{execute, CompileError, QueryResult};
pub use parser::{parse, ParseError};

/// Parse and execute an AxilQL query in one call.
pub fn run(db: &axil_core::Axil, input: &str) -> Result<QueryResult, QueryError> {
    let ast = parse(input)?;
    let result = execute(db, &ast)?;
    Ok(result)
}

/// Unified error type for parse + compile errors.
#[derive(Debug)]
pub enum QueryError {
    Parse(ParseError),
    Compile(CompileError),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::Parse(e) => write!(f, "{e}"),
            QueryError::Compile(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for QueryError {}

impl From<ParseError> for QueryError {
    fn from(e: ParseError) -> Self {
        QueryError::Parse(e)
    }
}

impl From<CompileError> for QueryError {
    fn from(e: CompileError) -> Self {
        QueryError::Compile(e)
    }
}

/// Syntax metadata for syntax highlighting in query consoles.
///
/// Each entry maps a token category to a list of keywords/patterns.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SyntaxMetadata {
    /// Primary command keywords (RECALL, FIND, TRAVERSE, GET, COUNT).
    pub commands: Vec<&'static str>,
    /// Clause keywords (WHERE, AND, FROM, TOP, LIMIT, etc.).
    pub clauses: Vec<&'static str>,
    /// Modifier keywords (BOOST, PROFILE, EXPLAIN, ORDER, BY).
    pub modifiers: Vec<&'static str>,
    /// Sort/direction keywords (ASC, DESC, IN).
    pub directions: Vec<&'static str>,
    /// Literal keywords (true, false, null).
    pub literals: Vec<&'static str>,
    /// Comparison operators (=, !=, >, <, >=, <=, CONTAINS).
    pub operators: Vec<&'static str>,
    /// Boost type identifiers (recency, graph, feedback).
    pub boost_types: Vec<&'static str>,
}

/// Get syntax metadata for AxilQL — useful for syntax highlighters.
///
/// Returns a static reference to avoid allocations on hot paths.
pub fn syntax_metadata() -> &'static SyntaxMetadata {
    use std::sync::OnceLock;
    static META: OnceLock<SyntaxMetadata> = OnceLock::new();
    META.get_or_init(|| SyntaxMetadata {
        commands: vec!["RECALL", "FIND", "TRAVERSE", "GET", "COUNT"],
        clauses: vec!["WHERE", "AND", "FROM", "TOP", "LIMIT", "OFFSET"],
        modifiers: vec!["BOOST", "PROFILE", "EXPLAIN", "ORDER", "BY"],
        directions: vec!["ASC", "DESC", "IN"],
        literals: vec!["true", "false", "null"],
        operators: vec!["=", "!=", ">", "<", ">=", "<=", "CONTAINS"],
        boost_types: vec!["recency", "graph", "feedback"],
    })
}

/// Auto-complete suggestions for a partial query.
///
/// Returns context-aware suggestions based on the cursor position.
pub fn autocomplete_suggestions(input: &str) -> Vec<String> {
    let trimmed = input.trim_start();
    let ends_with_space = trimmed.ends_with(' ');
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let last_word = words.last().unwrap_or(&"").to_uppercase();
    let meta = syntax_metadata();

    if words.is_empty() || (words.len() == 1 && !ends_with_space) {
        // Suggest commands.
        return meta
            .commands
            .iter()
            .chain(std::iter::once(&"EXPLAIN"))
            .filter(|cmd| cmd.starts_with(&last_word) || last_word.is_empty())
            .map(|s| s.to_string())
            .collect();
    }

    let first_word = words[0].to_uppercase();

    // The "previous word" is the last completed word when input ends with space,
    // or the second-to-last word when currently typing.
    let prev_word = if ends_with_space {
        last_word.clone()
    } else {
        words.iter().rev().nth(1).unwrap_or(&"").to_uppercase()
    };

    match prev_word.as_str() {
        "BOOST" => {
            return meta.boost_types.iter().map(|s| s.to_string()).collect();
        }
        "ORDER" => {
            return vec!["BY".to_string()];
        }
        "BY" | "WHERE" | "AND" => {
            // Suggest field names — caller should provide these from schema.
            return vec!["table".into(), "data.*".into(), "created_at".into()];
        }
        _ => {}
    }

    // After TOP/LIMIT/OFFSET, no suggestions (expects a number).
    if matches!(prev_word.as_str(), "TOP" | "LIMIT" | "OFFSET") {
        return Vec::new();
    }

    // Generic clause suggestions.
    let mut suggestions: Vec<String> = meta
        .clauses
        .iter()
        .chain(meta.modifiers.iter())
        .filter(|kw| {
            if trimmed.ends_with(' ') {
                true
            } else {
                kw.starts_with(&last_word)
            }
        })
        .map(|s| s.to_string())
        .collect();

    // Add TRAVERSE for chaining.
    if first_word == "RECALL" || first_word == "FIND" {
        suggestions.push("TRAVERSE".to_string());
    }

    suggestions.sort();
    suggestions.dedup();
    suggestions
}

/// Structured error response for API/CLI consumers.
#[derive(Debug, serde::Serialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl From<&QueryError> for ErrorResponse {
    fn from(e: &QueryError) -> Self {
        match e {
            QueryError::Parse(pe) => ErrorResponse {
                error: pe.message.clone(),
                line: Some(pe.span.line),
                column: Some(pe.span.column),
                suggestion: pe.suggestion.clone(),
            },
            QueryError::Compile(ce) => ErrorResponse {
                error: ce.message.clone(),
                line: None,
                column: None,
                suggestion: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_metadata_has_all_commands() {
        let meta = syntax_metadata();
        assert!(meta.commands.contains(&"RECALL"));
        assert!(meta.commands.contains(&"FIND"));
        assert!(meta.commands.contains(&"TRAVERSE"));
        assert!(meta.commands.contains(&"GET"));
        assert!(meta.commands.contains(&"COUNT"));
        assert_eq!(meta.commands.len(), 5);
    }

    #[test]
    fn autocomplete_empty_suggests_commands() {
        let suggestions = autocomplete_suggestions("");
        assert!(suggestions.contains(&"RECALL".to_string()));
        assert!(suggestions.contains(&"FIND".to_string()));
    }

    #[test]
    fn autocomplete_partial_command() {
        let suggestions = autocomplete_suggestions("REC");
        assert!(suggestions.contains(&"RECALL".to_string()));
        assert!(!suggestions.contains(&"FIND".to_string()));
    }

    #[test]
    fn autocomplete_after_boost_suggests_types() {
        let suggestions = autocomplete_suggestions("RECALL \"test\" TOP 5 BOOST ");
        assert!(suggestions.contains(&"recency".to_string()));
        assert!(suggestions.contains(&"graph".to_string()));
    }

    #[test]
    fn autocomplete_after_order_suggests_by() {
        let suggestions = autocomplete_suggestions("FIND \"x\" ORDER ");
        assert_eq!(suggestions, vec!["BY".to_string()]);
    }
}
