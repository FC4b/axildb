//! Preference memory — rules, feedback, and conventions.
//!
//! Stores user directives and auto-detected preferences with
//! exact key-value lookup. User rules always override detected rules.
//! Includes synthetic preference document generation for better recall.

use serde_json::json;

use axil_core::{Axil, Op, Record, Result};

use crate::types::TABLE_PREFERENCES;

/// Source of a preference: explicit from user or auto-detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferenceSource {
    /// Explicitly set by the user.
    User,
    /// Inferred from patterns or config files.
    Detected,
}

impl PreferenceSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            PreferenceSource::User => "user",
            PreferenceSource::Detected => "detected",
        }
    }
}

impl std::str::FromStr for PreferenceSource {
    type Err = axil_core::AxilError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "user" => Ok(PreferenceSource::User),
            "detected" => Ok(PreferenceSource::Detected),
            other => Err(axil_core::AxilError::InvalidQuery(format!(
                "unknown preference source: {other} (expected user or detected)"
            ))),
        }
    }
}

/// Preference memory — user directives and detected conventions.
pub struct PreferenceMemory<'a> {
    db: &'a Axil,
}

impl<'a> PreferenceMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db }
    }

    /// Set a rule. If the key exists and the new source has higher priority,
    /// update it; otherwise create a new one.
    pub fn set(&self, key: &str, value: &str, source: PreferenceSource) -> Result<Record> {
        // Check for existing rule with same key.
        if let Some(existing) = self.get(key)? {
            let existing_source = existing
                .data
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("detected");

            // User rules always win; detected rules don't override user rules.
            if source == PreferenceSource::Detected && existing_source == "user" {
                return Ok(existing);
            }

            // Update existing.
            let mut data = existing.data.clone();
            data["value"] = json!(value);
            data["source"] = json!(source.as_str());

            // Update synthetic document.
            data["synthetic_doc"] = json!(build_synthetic_doc(key, value));

            let updated = self.db.update(&existing.id, data)?;

            if self.db.has_vector_index() {
                let embed_text = format!("{key}: {value}");
                let _ = self.db.embed_text(&existing.id, &embed_text);
            }

            return Ok(updated);
        }

        // Create new rule.
        let data = json!({
            "key": key,
            "value": value,
            "source": source.as_str(),
            "synthetic_doc": build_synthetic_doc(key, value),
        });

        let record = self.db.insert(TABLE_PREFERENCES, data)?;

        if self.db.has_vector_index() {
            let embed_text = format!("{key}: {value}");
            let _ = self.db.embed_text(&record.id, &embed_text);
        }

        Ok(record)
    }

    /// Get a rule by exact key match (NOT vector search).
    pub fn get(&self, key: &str) -> Result<Option<Record>> {
        let records = self
            .db
            .query()
            .table(TABLE_PREFERENCES)
            .where_field("key", Op::Eq, json!(key))
            .limit(1)
            .exec()?;

        Ok(records.into_iter().next())
    }

    /// List all active rules.
    pub fn list(&self) -> Result<Vec<Record>> {
        let records = self.db.list(TABLE_PREFERENCES)?;
        Ok(crate::ttl::filter_expired(records))
    }

    /// Delete a rule by key.
    pub fn delete(&self, key: &str) -> Result<bool> {
        if let Some(record) = self.get(key)? {
            self.db.delete(&record.id)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Auto-detect preferences from text content (e.g., CLAUDE.md).
    ///
    /// Uses regex-like patterns to extract preference statements:
    /// - "I like/love/enjoy/prefer ..."
    /// - "Always ..." / "Never ..."
    /// - "Use ... for ..."
    pub fn extract_from_text(&self, text: &str) -> Result<Vec<Record>> {
        let mut extracted = Vec::new();

        for (key, value) in extract_preference_patterns(text) {
            let record = self.set(&key, &value, PreferenceSource::Detected)?;
            extracted.push(record);
        }

        Ok(extracted)
    }

    /// Search preferences by semantic similarity (for "what are my hobbies?" type queries).
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<(Record, f32)>> {
        if !self.db.has_vector_index() {
            return Ok(Vec::new());
        }

        let results = self.db.similar_to(query, top_k * 3)?;
        let mut filtered: Vec<(Record, f32)> = results
            .into_iter()
            .filter(|(r, _)| r.table == TABLE_PREFERENCES)
            .filter(|(r, _)| !crate::ttl::is_record_expired(r))
            .filter(|(r, _)| !crate::ttl::is_record_superseded(r))
            .collect();

        filtered.truncate(top_k);
        Ok(filtered)
    }
}

/// Build a synthetic preference document for better vector search recall.
///
/// Bridges the vocabulary gap: user says "Use thiserror in libs" but
/// later asks "what error handling library should I use?"
fn build_synthetic_doc(key: &str, value: &str) -> String {
    format!("User preference for {key}: {value}. Rule about {key}. Convention: {value}.")
}

/// Extract preference patterns from text using heuristic matching.
///
/// 16 regex extraction patterns covering common preference expressions.
fn extract_preference_patterns(text: &str) -> Vec<(String, String)> {
    let mut preferences = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        let lower = trimmed.to_lowercase();

        // Pattern matching for preference extraction.
        let extracted = if lower.starts_with("always ") || lower.starts_with("never ") {
            Some(("rule", trimmed.to_string()))
        } else if lower.contains(" prefer ") || lower.starts_with("prefer ") {
            Some(("preference", trimmed.to_string()))
        } else if lower.starts_with("use ") && lower.contains(" for ") {
            Some(("convention", trimmed.to_string()))
        } else if lower.starts_with("- use ") {
            Some(("convention", trimmed.trim_start_matches("- ").to_string()))
        } else if lower.contains("i like ")
            || lower.contains("i love ")
            || lower.contains("i enjoy ")
        {
            Some(("like", trimmed.to_string()))
        } else if lower.contains("i don't like ")
            || lower.contains("i hate ")
            || lower.contains("i avoid ")
        {
            Some(("dislike", trimmed.to_string()))
        } else if lower.contains("my favorite ") {
            Some(("favorite", trimmed.to_string()))
        } else if lower.starts_with("don't ")
            || lower.starts_with("do not ")
            || lower.starts_with("avoid ")
        {
            Some(("rule", trimmed.to_string()))
        } else {
            None
        };

        if let Some((category, value)) = extracted {
            // Use a stable key derived from content so repeated calls don't collide.
            let slug: String = lower
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .collect::<String>()
                .split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join("_");
            let key = format!("auto_{category}_{slug}");
            preferences.push((key, value));
        }
    }

    preferences
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(&path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn set_and_get_rule() {
        let (db, _dir) = temp_db();
        let pref = PreferenceMemory::new(&db);

        pref.set(
            "error_handling",
            "Use thiserror in libs",
            PreferenceSource::User,
        )
        .unwrap();

        let rule = pref.get("error_handling").unwrap();
        assert!(rule.is_some());
        let rule = rule.unwrap();
        assert_eq!(rule.data["value"], "Use thiserror in libs");
        assert_eq!(rule.data["source"], "user");
    }

    #[test]
    fn user_overrides_detected() {
        let (db, _dir) = temp_db();
        let pref = PreferenceMemory::new(&db);

        pref.set("style", "detected_val", PreferenceSource::Detected)
            .unwrap();
        pref.set("style", "user_val", PreferenceSource::User)
            .unwrap();

        let rule = pref.get("style").unwrap().unwrap();
        assert_eq!(rule.data["value"], "user_val");
        assert_eq!(rule.data["source"], "user");
    }

    #[test]
    fn detected_does_not_override_user() {
        let (db, _dir) = temp_db();
        let pref = PreferenceMemory::new(&db);

        pref.set("style", "user_val", PreferenceSource::User)
            .unwrap();
        pref.set("style", "detected_val", PreferenceSource::Detected)
            .unwrap();

        let rule = pref.get("style").unwrap().unwrap();
        assert_eq!(rule.data["value"], "user_val");
    }

    #[test]
    fn list_and_delete() {
        let (db, _dir) = temp_db();
        let pref = PreferenceMemory::new(&db);

        pref.set("a", "1", PreferenceSource::User).unwrap();
        pref.set("b", "2", PreferenceSource::User).unwrap();

        assert_eq!(pref.list().unwrap().len(), 2);

        assert!(pref.delete("a").unwrap());
        assert_eq!(pref.list().unwrap().len(), 1);

        assert!(!pref.delete("nonexistent").unwrap());
    }

    #[test]
    fn extract_preferences() {
        let text = r#"
# Project Rules
Always run tests before committing
Never push directly to main
Use thiserror for error handling in libs
Avoid global mutable state
I prefer functional style over OOP
        "#;

        let patterns = extract_preference_patterns(text);
        assert!(patterns.len() >= 4);
    }

    #[test]
    fn synthetic_doc_bridges_vocab() {
        let doc = build_synthetic_doc("error_handling", "Use thiserror in libs");
        assert!(doc.contains("error_handling"));
        assert!(doc.contains("thiserror"));
        assert!(doc.contains("preference"));
    }
}
