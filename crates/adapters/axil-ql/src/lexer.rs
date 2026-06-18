//! Tokenizer for AxilQL.
//!
//! Converts an input string into a stream of tokens. The lexer is hand-written
//! for maximum performance and clear error messages.

use std::fmt;

/// A token with its position in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// The raw text of the token.
    pub text: String,
}

/// Source position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub offset: usize,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// Token types.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Recall,
    Find,
    Traverse,
    Get,
    Count,
    Where,
    And,
    From,
    Top,
    Limit,
    Offset,
    Order, // first word of "ORDER BY" — consumed by parser when followed by By
    By,
    Boost,
    Profile,
    Explain,
    In,
    Asc,
    Desc,
    Contains,

    // Literals
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    True,
    False,
    Null,

    // Operators
    Eq,  // =
    Ne,  // !=
    Gt,  // >
    Lt,  // <
    Gte, // >=
    Lte, // <=

    // Identifiers (table names, field names, record IDs, boost types)
    Ident(String),

    // Traversal path segments like ->edge, <-edge, <->edge
    TraversalPath(String),

    // End of input
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Recall => write!(f, "RECALL"),
            TokenKind::Find => write!(f, "FIND"),
            TokenKind::Traverse => write!(f, "TRAVERSE"),
            TokenKind::Get => write!(f, "GET"),
            TokenKind::Count => write!(f, "COUNT"),
            TokenKind::Where => write!(f, "WHERE"),
            TokenKind::And => write!(f, "AND"),
            TokenKind::From => write!(f, "FROM"),
            TokenKind::Top => write!(f, "TOP"),
            TokenKind::Limit => write!(f, "LIMIT"),
            TokenKind::Offset => write!(f, "OFFSET"),
            TokenKind::Order => write!(f, "ORDER"),
            TokenKind::By => write!(f, "BY"),
            TokenKind::Boost => write!(f, "BOOST"),
            TokenKind::Profile => write!(f, "PROFILE"),
            TokenKind::Explain => write!(f, "EXPLAIN"),
            TokenKind::In => write!(f, "IN"),
            TokenKind::Asc => write!(f, "ASC"),
            TokenKind::Desc => write!(f, "DESC"),
            TokenKind::Contains => write!(f, "CONTAINS"),
            TokenKind::StringLit(s) => write!(f, "\"{s}\""),
            TokenKind::IntLit(n) => write!(f, "{n}"),
            TokenKind::FloatLit(n) => write!(f, "{n}"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Null => write!(f, "null"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::Ne => write!(f, "!="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gte => write!(f, ">="),
            TokenKind::Lte => write!(f, "<="),
            TokenKind::Ident(s) => write!(f, "{s}"),
            TokenKind::TraversalPath(s) => write!(f, "{s}"),
            TokenKind::Eof => write!(f, "end of input"),
        }
    }
}

/// Lexer error.
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)
    }
}

/// Tokenize an AxilQL input string.
pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let mut pos = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    while pos < bytes.len() {
        // Skip whitespace.
        if bytes[pos].is_ascii_whitespace() {
            if bytes[pos] == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            pos += 1;
            continue;
        }

        // Skip single-line comments: -- ...
        if pos + 1 < bytes.len()
            && bytes[pos] == b'-'
            && bytes[pos + 1] == b'-'
            && (pos + 2 >= bytes.len() || bytes[pos + 2] != b'>')
        {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }

        let start_span = Span {
            offset: pos,
            line,
            column: col,
        };

        // Traversal paths: ->, <-, <->
        if (bytes[pos] == b'-' && pos + 1 < bytes.len() && bytes[pos + 1] == b'>')
            || (bytes[pos] == b'<' && pos + 1 < bytes.len() && bytes[pos + 1] == b'-')
        {
            let start = pos;
            // Parse the full traversal path which may have multiple segments
            let path = scan_traversal_path(bytes, &mut pos);
            let len = pos - start;
            col += len;
            tokens.push(Token {
                kind: TokenKind::TraversalPath(path),
                span: start_span,
                text: input[start..pos].to_string(),
            });
            continue;
        }

        // String literals (double or single quoted).
        if bytes[pos] == b'"' || bytes[pos] == b'\'' {
            let quote = bytes[pos];
            let start = pos;
            pos += 1;
            col += 1;
            let mut value = String::new();
            loop {
                if pos >= bytes.len() {
                    return Err(LexError {
                        message: "unterminated string literal".to_string(),
                        span: start_span,
                    });
                }
                if bytes[pos] == b'\\' && pos + 1 < bytes.len() {
                    // Escape sequences.
                    match bytes[pos + 1] {
                        b'\\' => value.push('\\'),
                        b'"' => value.push('"'),
                        b'\'' => value.push('\''),
                        b'n' => value.push('\n'),
                        b't' => value.push('\t'),
                        other => {
                            value.push('\\');
                            value.push(other as char);
                        }
                    }
                    pos += 2;
                    col += 2;
                    continue;
                }
                if bytes[pos] == quote {
                    pos += 1;
                    col += 1;
                    break;
                }
                if bytes[pos] == b'\n' {
                    line += 1;
                    col = 1;
                    value.push('\n');
                    pos += 1;
                } else {
                    // Handle UTF-8 multi-byte characters properly.
                    let remaining = &input[pos..];
                    let ch = remaining.chars().next().unwrap();
                    let ch_len = ch.len_utf8();
                    value.push(ch);
                    pos += ch_len;
                    col += 1;
                }
            }
            tokens.push(Token {
                kind: TokenKind::StringLit(value),
                span: start_span,
                text: input[start..pos].to_string(),
            });
            continue;
        }

        // Numbers.
        if bytes[pos].is_ascii_digit()
            || (bytes[pos] == b'-' && pos + 1 < bytes.len() && bytes[pos + 1].is_ascii_digit())
        {
            let start = pos;
            if bytes[pos] == b'-' {
                pos += 1;
            }
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            let is_float = pos < bytes.len()
                && bytes[pos] == b'.'
                && pos + 1 < bytes.len()
                && bytes[pos + 1].is_ascii_digit();
            if is_float {
                pos += 1; // skip dot
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
                let text = &input[start..pos];
                let n: f64 = text.parse().map_err(|_| LexError {
                    message: format!("invalid float: {text}"),
                    span: start_span,
                })?;
                let len = pos - start;
                col += len;
                tokens.push(Token {
                    kind: TokenKind::FloatLit(n),
                    span: start_span,
                    text: text.to_string(),
                });
            } else {
                let text = &input[start..pos];
                let n: i64 = text.parse().map_err(|_| LexError {
                    message: format!("invalid integer: {text}"),
                    span: start_span,
                })?;
                let len = pos - start;
                col += len;
                tokens.push(Token {
                    kind: TokenKind::IntLit(n),
                    span: start_span,
                    text: text.to_string(),
                });
            }
            continue;
        }

        // Operators.
        match bytes[pos] {
            b'!' if pos + 1 < bytes.len() && bytes[pos + 1] == b'=' => {
                tokens.push(Token {
                    kind: TokenKind::Ne,
                    span: start_span,
                    text: "!=".to_string(),
                });
                pos += 2;
                col += 2;
                continue;
            }
            b'>' if pos + 1 < bytes.len() && bytes[pos + 1] == b'=' => {
                tokens.push(Token {
                    kind: TokenKind::Gte,
                    span: start_span,
                    text: ">=".to_string(),
                });
                pos += 2;
                col += 2;
                continue;
            }
            b'<' if pos + 1 < bytes.len() && bytes[pos + 1] == b'=' => {
                tokens.push(Token {
                    kind: TokenKind::Lte,
                    span: start_span,
                    text: "<=".to_string(),
                });
                pos += 2;
                col += 2;
                continue;
            }
            b'=' => {
                tokens.push(Token {
                    kind: TokenKind::Eq,
                    span: start_span,
                    text: "=".to_string(),
                });
                pos += 1;
                col += 1;
                continue;
            }
            b'>' => {
                tokens.push(Token {
                    kind: TokenKind::Gt,
                    span: start_span,
                    text: ">".to_string(),
                });
                pos += 1;
                col += 1;
                continue;
            }
            // b'<' without '=' or '-' is Lt (traversal is already handled above)
            b'<' => {
                tokens.push(Token {
                    kind: TokenKind::Lt,
                    span: start_span,
                    text: "<".to_string(),
                });
                pos += 1;
                col += 1;
                continue;
            }
            _ => {}
        }

        // Keywords and identifiers.
        if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
            let start = pos;
            while pos < bytes.len()
                && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' || bytes[pos] == b'.')
            {
                pos += 1;
            }
            let text = &input[start..pos];
            let len = pos - start;
            col += len;

            let kind = if text.eq_ignore_ascii_case("RECALL") {
                TokenKind::Recall
            } else if text.eq_ignore_ascii_case("FIND") {
                TokenKind::Find
            } else if text.eq_ignore_ascii_case("TRAVERSE") {
                TokenKind::Traverse
            } else if text.eq_ignore_ascii_case("GET") {
                TokenKind::Get
            } else if text.eq_ignore_ascii_case("COUNT") {
                TokenKind::Count
            } else if text.eq_ignore_ascii_case("WHERE") {
                TokenKind::Where
            } else if text.eq_ignore_ascii_case("AND") {
                TokenKind::And
            } else if text.eq_ignore_ascii_case("FROM") {
                TokenKind::From
            } else if text.eq_ignore_ascii_case("TOP") {
                TokenKind::Top
            } else if text.eq_ignore_ascii_case("LIMIT") {
                TokenKind::Limit
            } else if text.eq_ignore_ascii_case("OFFSET") {
                TokenKind::Offset
            } else if text.eq_ignore_ascii_case("ORDER") {
                TokenKind::Order
            } else if text.eq_ignore_ascii_case("BY") {
                TokenKind::By
            } else if text.eq_ignore_ascii_case("BOOST") {
                TokenKind::Boost
            } else if text.eq_ignore_ascii_case("PROFILE") {
                TokenKind::Profile
            } else if text.eq_ignore_ascii_case("EXPLAIN") {
                TokenKind::Explain
            } else if text.eq_ignore_ascii_case("IN") {
                TokenKind::In
            } else if text.eq_ignore_ascii_case("ASC") {
                TokenKind::Asc
            } else if text.eq_ignore_ascii_case("DESC") {
                TokenKind::Desc
            } else if text.eq_ignore_ascii_case("CONTAINS") {
                TokenKind::Contains
            } else if text.eq_ignore_ascii_case("TRUE") {
                TokenKind::True
            } else if text.eq_ignore_ascii_case("FALSE") {
                TokenKind::False
            } else if text.eq_ignore_ascii_case("NULL") {
                TokenKind::Null
            } else {
                TokenKind::Ident(text.to_string())
            };
            tokens.push(Token {
                kind,
                span: start_span,
                text: text.to_string(),
            });
            continue;
        }

        return Err(LexError {
            message: format!("unexpected character: '{}'", bytes[pos] as char),
            span: start_span,
        });
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span {
            offset: pos,
            line,
            column: col,
        },
        text: String::new(),
    });

    Ok(tokens)
}

/// Scan a traversal path like `->edge->other` or `<-edge<->both`.
/// Advances `pos` past the path.
fn scan_traversal_path(bytes: &[u8], pos: &mut usize) -> String {
    let start = *pos;
    loop {
        // Expect an arrow: ->, <-, or <->
        if *pos + 1 < bytes.len() && bytes[*pos] == b'-' && bytes[*pos + 1] == b'>' {
            *pos += 2;
        } else if *pos + 2 < bytes.len()
            && bytes[*pos] == b'<'
            && bytes[*pos + 1] == b'-'
            && bytes[*pos + 2] == b'>'
        {
            *pos += 3;
        } else if *pos + 1 < bytes.len() && bytes[*pos] == b'<' && bytes[*pos + 1] == b'-' {
            *pos += 2;
        } else if *pos > start {
            // No more arrows — we're done
            break;
        } else {
            // Should not happen since caller checked for arrow prefix
            break;
        }

        // Now scan the edge type identifier.
        // Allow hyphens in edge names (e.g. ->depends-on) but not when
        // the hyphen is followed by '>' (which starts a new arrow ->).
        while *pos < bytes.len()
            && (bytes[*pos].is_ascii_alphanumeric()
                || bytes[*pos] == b'_'
                || (bytes[*pos] == b'-' && *pos + 1 < bytes.len() && bytes[*pos + 1] != b'>'))
        {
            *pos += 1;
        }
    }

    std::str::from_utf8(&bytes[start..*pos])
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_recall() {
        let tokens = tokenize(r#"RECALL "hello world" TOP 10"#).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Recall);
        assert_eq!(
            tokens[1].kind,
            TokenKind::StringLit("hello world".to_string())
        );
        assert_eq!(tokens[2].kind, TokenKind::Top);
        assert_eq!(tokens[3].kind, TokenKind::IntLit(10));
        assert_eq!(tokens[4].kind, TokenKind::Eof);
    }

    #[test]
    fn tokenize_traversal_path() {
        let tokens = tokenize("TRAVERSE ->modified->file").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Traverse);
        assert_eq!(
            tokens[1].kind,
            TokenKind::TraversalPath("->modified->file".to_string())
        );
    }

    #[test]
    fn tokenize_hyphenated_edge() {
        let tokens = tokenize("TRAVERSE ->depends-on->blocks-for").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Traverse);
        assert_eq!(
            tokens[1].kind,
            TokenKind::TraversalPath("->depends-on->blocks-for".to_string())
        );
    }

    #[test]
    fn tokenize_operators() {
        let tokens = tokenize("WHERE x >= 10 AND y != 5").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Where);
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Gte);
        assert_eq!(tokens[3].kind, TokenKind::IntLit(10));
        assert_eq!(tokens[4].kind, TokenKind::And);
        assert_eq!(tokens[5].kind, TokenKind::Ident("y".to_string()));
        assert_eq!(tokens[6].kind, TokenKind::Ne);
        assert_eq!(tokens[7].kind, TokenKind::IntLit(5));
    }

    #[test]
    fn tokenize_single_quotes() {
        let tokens = tokenize("RECALL 'hello' TOP 5").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::StringLit("hello".to_string()));
    }

    #[test]
    fn tokenize_comment() {
        let tokens = tokenize("-- this is a comment\nRECALL \"x\" TOP 5").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Recall);
    }

    #[test]
    fn tokenize_float() {
        let tokens = tokenize("BOOST recency 0.4").unwrap();
        assert_eq!(tokens[2].kind, TokenKind::FloatLit(0.4));
    }

    #[test]
    fn unterminated_string_error() {
        let err = tokenize(r#"RECALL "unterminated"#).unwrap_err();
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn case_insensitive_keywords() {
        let tokens = tokenize("recall Find TRAVERSE get count").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Recall);
        assert_eq!(tokens[1].kind, TokenKind::Find);
        assert_eq!(tokens[2].kind, TokenKind::Traverse);
        assert_eq!(tokens[3].kind, TokenKind::Get);
        assert_eq!(tokens[4].kind, TokenKind::Count);
    }

    #[test]
    fn dotted_field_name() {
        let tokens = tokenize("WHERE data.name = \"x\"").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::Ident("data.name".to_string()));
    }
}
