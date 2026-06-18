//! Generic fallback parser for unknown languages.
//!
//! Extracts basic information from comments and function-like patterns.

use super::ParsedFile;

/// Parse a file with the generic fallback parser.
///
/// Extracts comments and function-like patterns without language-specific
/// knowledge. Used for Go, Java, C#, and other unsupported languages.
pub fn parse(source: &str) -> ParsedFile {
    let mut file = ParsedFile::default();
    let lines: Vec<&str> = source.lines().collect();

    // Extract file-level comments (first block of comments)
    let mut doc_lines = Vec::new();
    for line in &lines {
        let trimmed = line.trim();
        if let Some(comment) = trimmed.strip_prefix("// ") {
            doc_lines.push(comment.to_string());
        } else if let Some(comment) = trimmed.strip_prefix("# ") {
            doc_lines.push(comment.to_string());
        } else if trimmed.is_empty() && doc_lines.is_empty() {
            continue;
        } else {
            break;
        }
    }
    if !doc_lines.is_empty() {
        file.module_doc = Some(doc_lines.join(" "));
    }

    // Count function-like patterns
    let mut fn_count = 0;
    let mut class_count = 0;
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("func ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("public ")
            || trimmed.starts_with("pub fn ")
        {
            fn_count += 1;
        }
        if trimmed.starts_with("class ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("type ")
        {
            class_count += 1;
        }
    }

    // Generate summary
    file.summary = if let Some(ref doc) = file.module_doc {
        let first = doc.split('.').next().unwrap_or(doc).trim();
        first.to_string()
    } else {
        let mut parts = Vec::new();
        if fn_count > 0 {
            parts.push(format!("{fn_count} functions"));
        }
        if class_count > 0 {
            parts.push(format!("{class_count} types"));
        }
        parts.push(format!("{} lines", lines.len()));
        parts.join(", ")
    };

    file
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_go_file() {
        let src = r#"// Package auth provides JWT authentication.
package auth

func ValidateToken(token string) bool {
    return true
}

func HashPassword(password string) string {
    return ""
}
"#;
        let result = parse(src);
        assert!(result.summary.contains("Package auth"));
    }

    #[test]
    fn parse_empty_file() {
        let result = parse("");
        assert!(result.summary.contains("0 lines"));
    }
}
