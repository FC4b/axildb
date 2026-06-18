//! Rules & conventions store for Axil — key-value store for agent directives.
//!
//! Rules are exact-lookup records (not vector-searched). Each rule has a key,
//! a human-readable rule string, and a source indicating whether it was
//! explicitly set by a user or auto-detected from project config files.
//!
//! # Table
//!
//! All rules live in the `_rules` table.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ── Compiled regexes for rule extraction ────────────────────────────

static RE_USE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|\s)[*-]?\s*[Uu]se\s+`?(\w[\w\s]*\w?)`?\s+(?:for|in)\s+(.+)").unwrap()
});
static RE_PREFER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|\s)[*-]?\s*[Pp]refer\s+`?([^`\s][^`]*?)`?\s+over\s+`?([^`\s][^`]*?)`?")
        .unwrap()
});
static RE_NEVER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|\s)[*-]?\s*[Nn]ever\s+(.+)").unwrap());
static RE_ALWAYS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|\s)[*-]?\s*[Aa]lways\s+(.+)").unwrap());

use axil_core::{Axil, Result};

/// Reserved table name for rules.
pub const TABLE_RULES: &str = "_rules";

/// A single rule / convention directive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Unique key for this rule (e.g. "error_handling", "naming_convention").
    pub key: String,
    /// The rule text (e.g. "Use thiserror in libs, anyhow in bins").
    pub rule: String,
    /// Where this rule came from: `"user"` (explicit) or `"detected"` (auto-extracted).
    pub source: String,
}

/// Set (create or update) a rule by key.
///
/// If a rule with the given key already exists, it is updated in place.
/// Otherwise a new record is inserted.
pub fn set_rule(db: &Axil, key: &str, value: &str, source: &str) -> Result<Rule> {
    let existing = find_rule_record(db, key)?;

    let rule = Rule {
        key: key.to_string(),
        rule: value.to_string(),
        source: source.to_string(),
    };

    let data = json!({
        "key": rule.key,
        "rule": rule.rule,
        "source": rule.source,
    });

    if let Some(record) = existing {
        db.update(&record.id, data)?;
    } else {
        db.insert(TABLE_RULES, data)?;
    }

    Ok(rule)
}

/// Get a rule by its exact key. Returns `None` if not found.
pub fn get_rule(db: &Axil, key: &str) -> Result<Option<Rule>> {
    let record = find_rule_record(db, key)?;
    Ok(record.map(|r| record_to_rule(&r)))
}

/// List all stored rules.
pub fn list_rules(db: &Axil) -> Result<Vec<Rule>> {
    let records = db.list(TABLE_RULES)?;
    Ok(records.iter().map(record_to_rule).collect())
}

/// Delete a rule by key. Returns `true` if a rule was deleted.
pub fn delete_rule(db: &Axil, key: &str) -> Result<bool> {
    if let Some(record) = find_rule_record(db, key)? {
        db.delete(&record.id)
    } else {
        Ok(false)
    }
}

/// Auto-extract rules from convention files found in the project root.
///
/// Scans for:
/// - `CLAUDE.md`
/// - `.cursorrules`
/// - `.github/copilot-instructions.md`
///
/// Detected conventions are stored with `source: "detected"`. Existing
/// user-set rules are never overwritten.
pub fn auto_extract_rules(db: &Axil, project_root: &Path) -> Result<Vec<Rule>> {
    let mut extracted = Vec::new();

    let config_files = [
        project_root.join("CLAUDE.md"),
        project_root.join(".cursorrules"),
        project_root.join(".github/copilot-instructions.md"),
    ];

    // Load existing rules once to avoid repeated table scans.
    let existing_rules: std::collections::HashMap<String, String> = list_rules(db)?
        .into_iter()
        .map(|r| (r.key, r.source))
        .collect();

    for file in &config_files {
        if file.is_file() {
            if let Ok(content) = std::fs::read_to_string(file) {
                let rules = extract_rules_from_text(&content);
                for (key, rule_text) in rules {
                    if existing_rules.get(&key).map(|s| s.as_str()) == Some("user") {
                        continue;
                    }
                    let rule = set_rule(db, &key, &rule_text, "detected")?;
                    extracted.push(rule);
                }
            }
        }
    }

    Ok(extracted)
}

// ── Internal helpers ────────────────────────────────────────────────

/// Find the raw record for a rule by key.
fn find_rule_record(db: &Axil, key: &str) -> Result<Option<axil_core::Record>> {
    let records = db.list(TABLE_RULES)?;
    Ok(records
        .into_iter()
        .find(|r| r.data.get("key").and_then(|v| v.as_str()) == Some(key)))
}

/// Convert a raw `Record` into a typed `Rule`.
fn record_to_rule(record: &axil_core::Record) -> Rule {
    Rule {
        key: record
            .data
            .get("key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        rule: record
            .data
            .get("rule")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        source: record
            .data
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("detected")
            .to_string(),
    }
}

/// Extract convention rules from free-form markdown/text.
///
/// Patterns recognised:
/// - `Use X for Y` / `Use X in Y` → convention rule
/// - `Prefer X over Y` → preference rule
/// - `Never X` → directive (prohibition)
/// - `Always X` → directive (requirement)
/// - Lines inside a `## Coding Conventions` section → style rules
fn extract_rules_from_text(text: &str) -> Vec<(String, String)> {
    let mut rules: Vec<(String, String)> = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();

    let mut in_conventions_section = false;
    let mut convention_index = 0u32;

    for line in text.lines() {
        let trimmed = line.trim();

        // Track whether we're inside a "Coding Conventions" section.
        if trimmed.starts_with("## ") {
            in_conventions_section = trimmed.to_lowercase().contains("coding conventions");
        }

        // "Use X for/in Y"
        if let Some(caps) = RE_USE.captures(trimmed) {
            let tool = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let context = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            let key = slugify(&format!("use_{}_for_{}", tool, first_words(context, 3)));
            if seen_keys.insert(key.clone()) {
                rules.push((key, trimmed.to_string()));
            }
        }

        // "Prefer X over Y"
        if let Some(caps) = RE_PREFER.captures(trimmed) {
            let preferred = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let over = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            let key = slugify(&format!("prefer_{}_over_{}", preferred, over));
            if seen_keys.insert(key.clone()) {
                rules.push((key, trimmed.to_string()));
            }
        }

        // "Never X"
        if let Some(caps) = RE_NEVER.captures(trimmed) {
            let what = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let key = slugify(&format!("never_{}", first_words(what, 4)));
            if seen_keys.insert(key.clone()) {
                rules.push((key, trimmed.to_string()));
            }
        }

        // "Always X"
        if let Some(caps) = RE_ALWAYS.captures(trimmed) {
            let what = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let key = slugify(&format!("always_{}", first_words(what, 4)));
            if seen_keys.insert(key.clone()) {
                rules.push((key, trimmed.to_string()));
            }
        }

        // Lines in Coding Conventions section that look like list items.
        if in_conventions_section && (trimmed.starts_with("- ") || trimmed.starts_with("* ")) {
            let content = trimmed.trim_start_matches(['-', '*', ' ']);
            // Skip if already matched by a more specific pattern above.
            if !content.is_empty()
                && !RE_USE.is_match(trimmed)
                && !RE_PREFER.is_match(trimmed)
                && !RE_NEVER.is_match(trimmed)
                && !RE_ALWAYS.is_match(trimmed)
            {
                let key = format!("convention_{}", convention_index);
                convention_index += 1;
                if seen_keys.insert(key.clone()) {
                    rules.push((key, content.to_string()));
                }
            }
        }
    }

    rules
}

/// Turn a phrase into a lowercase snake_case key.
fn slugify(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<&str>>()
        .join("_")
}

/// Take the first N words from a string.
fn first_words(text: &str, n: usize) -> String {
    text.split_whitespace()
        .take(n)
        .collect::<Vec<&str>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Extraction tests ────────────────────────────────────────────

    #[test]
    fn extract_use_pattern() {
        let text = "- Use `thiserror` for error types in library crates";
        let rules = extract_rules_from_text(text);
        assert!(!rules.is_empty(), "should extract a 'Use X for Y' rule");
        assert!(rules[0].0.contains("thiserror"));
    }

    #[test]
    fn extract_prefer_pattern() {
        let text = "- Prefer `&str` over `String` in function params";
        let rules = extract_rules_from_text(text);
        assert!(!rules.is_empty(), "should extract a 'Prefer X over Y' rule");
        assert!(rules[0].0.contains("prefer"));
        assert!(rules[0].0.contains("str"));
    }

    #[test]
    fn extract_never_pattern() {
        let text = "Never commit secrets to the repository";
        let rules = extract_rules_from_text(text);
        assert!(!rules.is_empty(), "should extract a 'Never' directive");
        assert!(rules[0].0.starts_with("never"));
    }

    #[test]
    fn extract_always_pattern() {
        let text = "- Always run clippy before committing";
        let rules = extract_rules_from_text(text);
        assert!(!rules.is_empty(), "should extract an 'Always' directive");
        assert!(rules[0].0.starts_with("always"));
    }

    #[test]
    fn extract_conventions_section() {
        let text = r#"## Coding Conventions

- Keep dependencies minimal
- Tests in each crate
- File extension for databases: `.axil`
"#;
        let rules = extract_rules_from_text(text);
        assert_eq!(rules.len(), 3, "should extract 3 convention list items");
        assert!(rules[0].0.starts_with("convention_"));
    }

    #[test]
    fn no_duplicates() {
        let text = r#"
Use thiserror for error types
Use thiserror for error types
"#;
        let rules = extract_rules_from_text(text);
        assert_eq!(rules.len(), 1, "duplicate rules should be deduplicated");
    }

    #[test]
    fn conventions_section_ends_at_next_heading() {
        let text = r#"## Coding Conventions

- Keep deps minimal

## Architecture

- Use layers for separation
"#;
        let rules = extract_rules_from_text(text);
        // "Keep deps minimal" is a convention item, "Use layers for separation"
        // is NOT in the conventions section but matches "Use X for Y".
        let convention_rules: Vec<_> = rules
            .iter()
            .filter(|(k, _)| k.starts_with("convention_"))
            .collect();
        assert_eq!(convention_rules.len(), 1);
    }

    // ── Slug / helper tests ─────────────────────────────────────────

    #[test]
    fn slugify_produces_snake_case() {
        assert_eq!(slugify("Use thiserror for libs"), "use_thiserror_for_libs");
        assert_eq!(slugify("Prefer &str over String"), "prefer_str_over_string");
    }

    #[test]
    fn first_words_truncates() {
        assert_eq!(first_words("one two three four five", 3), "one two three");
        assert_eq!(first_words("single", 5), "single");
    }

    // ── CRUD tests (require tempdir + Axil) ─────────────────────────

    #[test]
    fn crud_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(db_path).build().unwrap();

        // Set a rule
        let rule = set_rule(
            &db,
            "error_handling",
            "Use thiserror in libs, anyhow in bins",
            "user",
        )
        .unwrap();
        assert_eq!(rule.key, "error_handling");
        assert_eq!(rule.source, "user");

        // Get it back
        let fetched = get_rule(&db, "error_handling").unwrap();
        assert!(fetched.is_some());
        let fetched = fetched.unwrap();
        assert_eq!(fetched.rule, "Use thiserror in libs, anyhow in bins");

        // List
        let all = list_rules(&db).unwrap();
        assert_eq!(all.len(), 1);

        // Update
        set_rule(&db, "error_handling", "Always use thiserror", "user").unwrap();
        let updated = get_rule(&db, "error_handling").unwrap().unwrap();
        assert_eq!(updated.rule, "Always use thiserror");
        // Should still be 1 record, not 2.
        assert_eq!(list_rules(&db).unwrap().len(), 1);

        // Delete
        let deleted = delete_rule(&db, "error_handling").unwrap();
        assert!(deleted);
        assert!(get_rule(&db, "error_handling").unwrap().is_none());

        // Delete non-existent
        let deleted = delete_rule(&db, "nonexistent").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn auto_extract_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(db_path).build().unwrap();

        // Write a fake CLAUDE.md
        let claude_md = dir.path().join("CLAUDE.md");
        std::fs::write(
            &claude_md,
            r#"# Project

## Coding Conventions

- Use `thiserror` for error types in library crates
- Use `anyhow` in binary crates
- Prefer `&str` over `String` in function params
- All public APIs must have doc comments
- Keep dependencies minimal
"#,
        )
        .unwrap();

        let extracted = auto_extract_rules(&db, dir.path()).unwrap();
        assert!(
            extracted.len() >= 3,
            "should extract at least 3 rules, got {}",
            extracted.len()
        );

        // All should be "detected"
        for rule in &extracted {
            assert_eq!(rule.source, "detected");
        }

        // User rules should not be overwritten
        set_rule(&db, &extracted[0].key, "my custom override", "user").unwrap();
        auto_extract_rules(&db, dir.path()).unwrap();
        let kept = get_rule(&db, &extracted[0].key).unwrap().unwrap();
        assert_eq!(kept.rule, "my custom override");
        assert_eq!(kept.source, "user");
    }
}
