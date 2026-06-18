//! Code-aware tokenizer for source code search.
//!
//! Splits identifiers on `_`, `.`, `::`, `/`, `-`, and CamelCase boundaries.
//! Also lowercases tokens for case-insensitive matching.
//!
//! Examples:
//! - `"parseHttpRequest"` → `["parse", "http", "request"]`
//! - `"parse_http_request"` → `["parse", "http", "request"]`
//! - `"std::collections::HashMap"` → `["std", "collections", "hash", "map"]`
//! - `"src/auth/middleware.rs"` → `["src", "auth", "middleware", "rs"]`

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Tokenizer that splits code identifiers on common boundaries.
#[derive(Clone, Default)]
pub struct CodeTokenizer;

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = CodeTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let tokens = tokenize_code(text);
        CodeTokenStream {
            tokens,
            index: 0,
            token: Token::default(),
        }
    }
}

pub struct CodeTokenStream {
    tokens: Vec<(usize, usize, String)>, // (offset_from, offset_to, text)
    index: usize,
    token: Token,
}

impl TokenStream for CodeTokenStream {
    fn advance(&mut self) -> bool {
        if self.index >= self.tokens.len() {
            return false;
        }
        let (from, to, ref text) = self.tokens[self.index];
        self.token.offset_from = from;
        self.token.offset_to = to;
        self.token.text.clear();
        self.token.text.push_str(text);
        self.token.position = self.index;
        self.index += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

/// Split text into code-aware tokens.
fn tokenize_code(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    // Pre-compute char→byte offset map once (O(n)) instead of O(n) per lookup.
    let byte_offsets: Vec<usize> = text
        .char_indices()
        .map(|(b, _)| b)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip separators: _, ., :, /, -, whitespace, punctuation
        if is_separator(chars[i]) {
            i += 1;
            continue;
        }

        let start_byte = byte_offsets[i];
        let mut end = i;

        if chars[i].is_ascii_uppercase() {
            // CamelCase: collect uppercase start + following lowercase
            end += 1;
            // Check if it's an ALLCAPS sequence (like "HTTP")
            let mut all_upper = true;
            while end < chars.len() && chars[end].is_ascii_uppercase() {
                end += 1;
            }
            if end - i > 1 && end < chars.len() && chars[end].is_ascii_lowercase() {
                // "HTTPRequest" → "HTTP" + "Request" — back up one
                end -= 1;
                all_upper = end - i > 1;
            }
            if !all_upper || end == i + 1 {
                // Single uppercase + lowercase run: "Parse" or "P"
                while end < chars.len() && chars[end].is_ascii_lowercase() {
                    end += 1;
                }
                // Also consume digits
                while end < chars.len() && chars[end].is_ascii_digit() {
                    end += 1;
                }
            }
        } else if chars[i].is_ascii_lowercase() {
            // Lowercase run
            while end < chars.len()
                && (chars[end].is_ascii_lowercase() || chars[end].is_ascii_digit())
            {
                end += 1;
            }
        } else if chars[i].is_ascii_digit() {
            // Digit run
            while end < chars.len() && chars[end].is_ascii_digit() {
                end += 1;
            }
        } else {
            // Unknown char — skip
            i += 1;
            continue;
        }

        if end > i {
            let end_byte = byte_offsets[end];
            let word: String = chars[i..end].iter().collect();
            let lower = word.to_ascii_lowercase();
            if !lower.is_empty() {
                tokens.push((start_byte, end_byte, lower));
            }
        }
        i = end;
    }

    tokens
}

fn is_separator(c: char) -> bool {
    matches!(
        c,
        '_' | '.'
            | ':'
            | '/'
            | '-'
            | '\\'
            | ' '
            | '\t'
            | '\n'
            | '\r'
            | '('
            | ')'
            | '{'
            | '}'
            | '['
            | ']'
            | '<'
            | '>'
            | ','
            | ';'
            | '='
            | '#'
            | '"'
            | '\''
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> Vec<String> {
        tokenize_code(text).into_iter().map(|(_, _, t)| t).collect()
    }

    #[test]
    fn snake_case() {
        assert_eq!(tok("parse_http_request"), vec!["parse", "http", "request"]);
    }

    #[test]
    fn camel_case() {
        assert_eq!(tok("parseHttpRequest"), vec!["parse", "http", "request"]);
    }

    #[test]
    fn pascal_case() {
        assert_eq!(tok("ParseHttpRequest"), vec!["parse", "http", "request"]);
    }

    #[test]
    fn all_caps_with_camel() {
        assert_eq!(tok("HTTPRequest"), vec!["http", "request"]);
    }

    #[test]
    fn path_separators() {
        assert_eq!(
            tok("src/auth/middleware.rs"),
            vec!["src", "auth", "middleware", "rs"]
        );
    }

    #[test]
    fn rust_path() {
        assert_eq!(
            tok("std::collections::HashMap"),
            vec!["std", "collections", "hash", "map"]
        );
    }

    #[test]
    fn mixed_separators() {
        assert_eq!(
            tok("my-project_name.file"),
            vec!["my", "project", "name", "file"]
        );
    }

    #[test]
    fn digits_in_identifier() {
        assert_eq!(tok("record2json"), vec!["record2json"]);
    }

    #[test]
    fn all_lowercase() {
        assert_eq!(tok("hello"), vec!["hello"]);
    }

    #[test]
    fn empty_string() {
        assert_eq!(tok(""), Vec::<String>::new());
    }

    #[test]
    fn only_separators() {
        assert_eq!(tok("__::__"), Vec::<String>::new());
    }
}
