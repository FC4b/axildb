//! Preference memory — rules, feedback, and conventions.
//!
//! Stores user directives and auto-detected preferences with
//! exact key-value lookup. User rules always override detected rules.
//! Includes synthetic preference document generation for better recall.

use std::collections::HashSet;

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
    agent: Option<String>,
}

impl<'a> PreferenceMemory<'a> {
    pub fn new(db: &'a Axil) -> Self {
        Self { db, agent: None }
    }

    /// Create a preference memory scoped to a specific agent.
    pub fn for_agent(db: &'a Axil, agent: &str) -> Self {
        Self {
            db,
            agent: Some(agent.to_string()),
        }
    }

    /// Set a rule. If the key exists and the new source has higher priority,
    /// update it; otherwise create a new one.
    ///
    /// Writes resolve by exact scope: an agent-scoped handle creates or
    /// updates its own rule — which shadows a global rule of the same key on
    /// that agent's reads — and never edits the global rule or another
    /// agent's; an unscoped handle only ever touches the global rule.
    pub fn set(&self, key: &str, value: &str, source: PreferenceSource) -> Result<Record> {
        let owned = self.find_owned(key)?;

        // User rules always win; detected rules don't override user rules —
        // including a global user rule an agent's own detected rule would
        // otherwise shadow.
        if source == PreferenceSource::Detected {
            let effective = match &owned {
                Some(record) => Some(record.clone()),
                None if self.agent.is_some() => self.find_in_scope(key, None)?,
                None => None,
            };
            if let Some(existing) = effective {
                if existing.data.get("source").and_then(|v| v.as_str()) == Some("user") {
                    return Ok(existing);
                }
            }
        }

        if let Some(existing) = owned {
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
        let mut data = json!({
            "key": key,
            "value": value,
            "source": source.as_str(),
            "synthetic_doc": build_synthetic_doc(key, value),
        });
        crate::stamp_agent(&mut data, self.agent.as_deref());

        let record = self.db.insert(TABLE_PREFERENCES, data)?;

        if self.db.has_vector_index() {
            let embed_text = format!("{key}: {value}");
            let _ = self.db.embed_text(&record.id, &embed_text);
        }

        Ok(record)
    }

    /// Get a rule by exact key match (NOT vector search).
    ///
    /// An agent-scoped handle resolves to its own rule when it has one, else
    /// the global rule. An unscoped handle prefers the global rule and falls
    /// back to any agent's.
    pub fn get(&self, key: &str) -> Result<Option<Record>> {
        let mut fallback = None;
        for record in self.rows_for_key(key)? {
            if crate::agent_owns(self.agent.as_deref(), &record.data) {
                return Ok(Some(record));
            }
            if fallback.is_none() && crate::agent_visible(self.agent.as_deref(), &record.data) {
                fallback = Some(record);
            }
        }
        Ok(fallback)
    }

    /// List all active rules.
    ///
    /// For an agent-scoped handle, a global rule the agent has overridden
    /// with its own is not listed.
    pub fn list(&self) -> Result<Vec<Record>> {
        let records: Vec<Record> = self
            .db
            .list(TABLE_PREFERENCES)?
            .into_iter()
            .filter(|r| crate::agent_visible(self.agent.as_deref(), &r.data))
            .collect();
        let own_keys: HashSet<String> = records
            .iter()
            .filter(|r| crate::agent_owns(self.agent.as_deref(), &r.data))
            .filter_map(|r| rule_key(r).map(String::from))
            .collect();
        let records = crate::drop_shadowed(self.agent.as_deref(), records, &own_keys, |r| {
            (&r.data, rule_key(r))
        });
        Ok(crate::ttl::filter_expired(records))
    }

    /// Delete a rule by key.
    ///
    /// Removes only the rule in this handle's own scope: an agent deletes its
    /// own rule (the global one, if any, becomes visible to it again), never
    /// the global rule or another agent's; an unscoped handle deletes only the
    /// global rule.
    pub fn delete(&self, key: &str) -> Result<bool> {
        if let Some(record) = self.find_owned(key)? {
            self.db.delete(&record.id)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Every stored rule for `key`, across all scopes.
    fn rows_for_key(&self, key: &str) -> Result<Vec<Record>> {
        self.db
            .query()
            .table(TABLE_PREFERENCES)
            .where_field("key", Op::Eq, json!(key))
            .exec()
    }

    /// The rule for `key` in this handle's own scope.
    fn find_owned(&self, key: &str) -> Result<Option<Record>> {
        self.find_in_scope(key, self.agent.as_deref())
    }

    /// The rule for `key` owned by exactly `scope` (`None` = global).
    fn find_in_scope(&self, key: &str, scope: Option<&str>) -> Result<Option<Record>> {
        Ok(self
            .rows_for_key(key)?
            .into_iter()
            .find(|r| crate::agent_owns(scope, &r.data)))
    }

    /// Keys of the rules this agent-scoped handle owns (empty when unscoped).
    fn own_keys(&self) -> Result<HashSet<String>> {
        let Some(agent) = self.agent.as_deref() else {
            return Ok(HashSet::new());
        };
        Ok(self
            .db
            .query()
            .table(TABLE_PREFERENCES)
            .where_field("_agent", Op::Eq, json!(agent))
            .exec()?
            .iter()
            .filter_map(|r| rule_key(r).map(String::from))
            .collect())
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
        let filtered: Vec<(Record, f32)> = results
            .into_iter()
            .filter(|(r, _)| r.table == TABLE_PREFERENCES)
            .filter(|(r, _)| !crate::ttl::is_record_expired(r))
            .filter(|(r, _)| !crate::ttl::is_record_superseded(r))
            .filter(|(r, _)| crate::agent_visible(self.agent.as_deref(), &r.data))
            .collect();
        let own_keys = self.own_keys()?;
        let mut filtered =
            crate::drop_shadowed(self.agent.as_deref(), filtered, &own_keys, |(r, _)| {
                (&r.data, rule_key(r))
            });

        filtered.truncate(top_k);
        Ok(filtered)
    }
}

/// The `key` a stored rule is filed under.
fn rule_key(record: &Record) -> Option<&str> {
    record.data.get("key").and_then(|v| v.as_str())
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
