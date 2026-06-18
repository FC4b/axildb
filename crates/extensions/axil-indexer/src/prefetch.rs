//! Predictive context pre-loading for agent tasks.
//!
//! Parses an intent string to identify relevant areas of the codebase,
//! then assembles a multi-section context payload within a token budget.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axil_core::{Axil, Result};

use crate::indexer::TABLE_MODULES;
use crate::rules::TABLE_RULES;
use crate::token::estimate_json_tokens;

// ── Types ───────────────────────────────────────────────────────────

/// A single section of pre-fetched context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchSection {
    /// Section identifier: "module_context", "similar_past", "active_rules", etc.
    pub name: String,
    /// Estimated token count for this section.
    pub tokens: usize,
    /// The section payload.
    pub data: Value,
}

/// Result of a prefetch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchResult {
    /// Ordered sections of pre-fetched context.
    pub sections: Vec<PrefetchSection>,
    /// Total tokens across all sections.
    pub total_tokens: usize,
    /// Whether all requested sections were populated.
    pub ready: bool,
}

// ── Public API ──────────────────────────────────────────────────────

/// Pre-fetch context relevant to the given intent, within a token budget.
///
/// Sections produced:
/// 1. **module_context** — module summaries matching intent keywords
/// 2. **similar_past** — vector-similar records (if embedder available)
/// 3. **active_rules** — all rules from the `_rules` table
/// 4. **recent_changes** — records from the last 7 days (if timeseries available)
///
/// The budget is divided evenly across sections; unused allocation is
/// redistributed to subsequent sections.
pub fn prefetch(db: &Axil, intent: &str, max_tokens: usize) -> Result<PrefetchResult> {
    let keywords = extract_keywords(intent);
    let section_count = 4usize;
    let mut budget_per_section = max_tokens / section_count.max(1);
    let mut sections = Vec::new();
    let mut total_tokens = 0usize;

    // ── 1. Module context ───────────────────────────────────────────
    let (module_section, used) = build_module_context(db, &keywords, budget_per_section)?;
    let unused = budget_per_section.saturating_sub(used);
    total_tokens += used;
    sections.push(module_section);
    budget_per_section += unused / (section_count - 1).max(1);

    // ── 2. Similar past ─────────────────────────────────────────────
    let (similar_section, used) = build_similar_past(db, intent, budget_per_section)?;
    let unused = budget_per_section.saturating_sub(used);
    total_tokens += used;
    sections.push(similar_section);
    budget_per_section += unused / (section_count - 2).max(1);

    // ── 3. Active rules ─────────────────────────────────────────────
    let (rules_section, used) = build_active_rules(db, budget_per_section)?;
    let unused = budget_per_section.saturating_sub(used);
    total_tokens += used;
    sections.push(rules_section);
    budget_per_section += unused;

    // ── 4. Recent changes ───────────────────────────────────────────
    let (recent_section, used) = build_recent_changes(db, budget_per_section)?;
    total_tokens += used;
    sections.push(recent_section);

    let ready = total_tokens > 0
        && sections.iter().any(|s| {
            s.data
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(!s.data.is_null())
        });

    Ok(PrefetchResult {
        sections,
        total_tokens,
        ready,
    })
}

// ── Section builders ────────────────────────────────────────────────

/// Build the module_context section by matching intent keywords against
/// module names and summaries.
pub(crate) fn build_module_context(
    db: &Axil,
    keywords: &[String],
    budget: usize,
) -> Result<(PrefetchSection, usize)> {
    let modules = db.list(TABLE_MODULES).unwrap_or_default();
    let mut matches: Vec<Value> = Vec::new();
    let mut used = 0usize;

    for module in &modules {
        let name = module
            .data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let summary = module
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let searchable = format!("{} {}", name, summary).to_lowercase();

        let matched = keywords.iter().any(|kw| searchable.contains(kw.as_str()));
        if !matched {
            continue;
        }

        let entry = json!({
            "name": name,
            "summary": summary,
        });
        let entry_tokens = estimate_json_tokens(&entry);

        if used + entry_tokens > budget {
            break;
        }
        used += entry_tokens;
        matches.push(entry);
    }

    let section = PrefetchSection {
        name: "module_context".to_string(),
        tokens: used,
        data: json!(matches),
    };
    Ok((section, used))
}

/// Build the similar_past section via vector search.
fn build_similar_past(db: &Axil, intent: &str, budget: usize) -> Result<(PrefetchSection, usize)> {
    let results = db.similar_to(intent, 10).unwrap_or_default();
    let mut items: Vec<Value> = Vec::new();
    let mut used = 0usize;

    for (record, score) in &results {
        let summary = record
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let entry = json!({
            "id": record.id.to_string(),
            "table": record.table,
            "score": score,
            "summary": summary,
        });
        let entry_tokens = estimate_json_tokens(&entry);

        if used + entry_tokens > budget {
            break;
        }
        used += entry_tokens;
        items.push(entry);
    }

    let section = PrefetchSection {
        name: "similar_past".to_string(),
        tokens: used,
        data: json!(items),
    };
    Ok((section, used))
}

/// Build the active_rules section by listing all rules.
fn build_active_rules(db: &Axil, budget: usize) -> Result<(PrefetchSection, usize)> {
    let rules = db.list(TABLE_RULES).unwrap_or_default();
    let mut items: Vec<Value> = Vec::new();
    let mut used = 0usize;

    for record in &rules {
        let entry = &record.data;
        let entry_tokens = estimate_json_tokens(entry);

        if used + entry_tokens > budget {
            break;
        }
        used += entry_tokens;
        items.push(entry.clone());
    }

    let section = PrefetchSection {
        name: "active_rules".to_string(),
        tokens: used,
        data: json!(items),
    };
    Ok((section, used))
}

/// Build the recent_changes section from the last 7 days.
fn build_recent_changes(db: &Axil, budget: usize) -> Result<(PrefetchSection, usize)> {
    let seven_days = 7 * 86400;
    let recent = db.since(None, seven_days).unwrap_or_default();
    let mut items: Vec<Value> = Vec::new();
    let mut used = 0usize;

    for record in &recent {
        let summary = record
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let path = record
            .data
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let entry = json!({
            "id": record.id.to_string(),
            "table": record.table,
            "summary": summary,
            "path": path,
            "created_at": record.created_at.to_rfc3339(),
        });
        let entry_tokens = estimate_json_tokens(&entry);

        if used + entry_tokens > budget {
            break;
        }
        used += entry_tokens;
        items.push(entry);
    }

    let section = PrefetchSection {
        name: "recent_changes".to_string(),
        tokens: used,
        data: json!(items),
    };
    Ok((section, used))
}

/// Pre-fetch context for opening a specific file.
///
/// Sections produced:
/// 1. **file_info** — the file's indexed summary and metadata
/// 2. **imports** — files this file imports (graph neighbors, outbound)
/// 3. **dependents** — files that depend on this file (graph neighbors, inbound)
/// 4. **recent_changes** — recent changes to this file's module
pub fn prefetch_file(db: &Axil, file_path: &str, max_tokens: usize) -> Result<PrefetchResult> {
    let budget_per = max_tokens / 4;
    let mut sections = Vec::new();
    let mut total_tokens = 0usize;

    // 1. File info
    let files = db.list(crate::indexer::TABLE_FILES).unwrap_or_default();
    let file_rec = files.iter().find(|r| {
        r.data
            .get("path")
            .and_then(|v| v.as_str())
            .map(|p| p.contains(file_path))
            .unwrap_or(false)
    });

    let mut file_section_tokens = 0;
    let file_data = if let Some(rec) = file_rec {
        let entry = serde_json::json!({
            "path": rec.data.get("path"),
            "summary": rec.data.get("summary"),
            "language": rec.data.get("language"),
        });
        file_section_tokens = estimate_json_tokens(&entry);
        total_tokens += file_section_tokens;
        entry
    } else {
        serde_json::json!(null)
    };
    sections.push(PrefetchSection {
        name: "file_info".to_string(),
        tokens: file_section_tokens,
        data: file_data,
    });

    // 2 & 3. Graph neighbors (imports + dependents)
    let mut imports_data = serde_json::json!([]);
    let mut dependents_data = serde_json::json!([]);
    let mut import_tokens = 0;
    let mut dependent_tokens = 0;

    if let Some(rec) = file_rec {
        if db.has_graph_index() {
            // Outbound edges = imports
            if let Ok(neighbors) = db.neighbors(&rec.id, None, axil_core::Direction::Out) {
                let mut items = Vec::new();
                for n in &neighbors {
                    let entry = serde_json::json!({
                        "path": n.data.get("path").or(n.data.get("name")),
                        "summary": n.data.get("summary"),
                    });
                    let t = estimate_json_tokens(&entry);
                    if import_tokens + t > budget_per {
                        break;
                    }
                    import_tokens += t;
                    items.push(entry);
                }
                imports_data = serde_json::json!(items);
            }

            // Inbound edges = dependents
            if let Ok(neighbors) = db.neighbors(&rec.id, None, axil_core::Direction::In) {
                let mut items = Vec::new();
                for n in &neighbors {
                    let entry = serde_json::json!({
                        "path": n.data.get("path").or(n.data.get("name")),
                        "summary": n.data.get("summary"),
                    });
                    let t = estimate_json_tokens(&entry);
                    if dependent_tokens + t > budget_per {
                        break;
                    }
                    dependent_tokens += t;
                    items.push(entry);
                }
                dependents_data = serde_json::json!(items);
            }
        }
    }

    total_tokens += import_tokens + dependent_tokens;
    sections.push(PrefetchSection {
        name: "imports".to_string(),
        tokens: import_tokens,
        data: imports_data,
    });
    sections.push(PrefetchSection {
        name: "dependents".to_string(),
        tokens: dependent_tokens,
        data: dependents_data,
    });

    // 4. Recent changes (reuse existing builder)
    let (recent_section, used) = build_recent_changes(db, budget_per)?;
    total_tokens += used;
    sections.push(recent_section);

    let ready = total_tokens > 0
        && sections.iter().any(|s| {
            s.data
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(!s.data.is_null())
        });

    Ok(PrefetchResult {
        sections,
        total_tokens,
        ready,
    })
}

// ── Session Cache ──────────────────────────────────────────────────

/// Cached prefetch entry with a timestamp for TTL expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    intent: String,
    result: PrefetchResult,
    cached_at: String,
}

/// Cache file path for a given database path (companion file convention).
fn cache_path(db_path: &std::path::Path) -> std::path::PathBuf {
    let mut s = db_path.as_os_str().to_os_string();
    s.push(".prefetch-cache");
    std::path::PathBuf::from(s)
}

/// Try to load a cached prefetch result for the given intent.
///
/// Returns `None` if no cache exists, the intent doesn't match, or the
/// cache is older than `ttl_minutes`.
pub fn load_cached(
    db_path: &std::path::Path,
    intent: &str,
    ttl_minutes: u64,
) -> Option<PrefetchResult> {
    let path = cache_path(db_path);
    let content = std::fs::read_to_string(&path).ok()?;
    let entry: CacheEntry = serde_json::from_str(&content).ok()?;

    if entry.intent != intent {
        return None;
    }

    // Check TTL.
    let cached_at = chrono::DateTime::parse_from_rfc3339(&entry.cached_at).ok()?;
    let age = chrono::Utc::now().signed_duration_since(cached_at);
    if age.num_minutes() > ttl_minutes as i64 {
        return None;
    }

    Some(entry.result)
}

/// Save a prefetch result to the session cache.
pub fn save_cache(db_path: &std::path::Path, intent: &str, result: &PrefetchResult) {
    let entry = CacheEntry {
        intent: intent.to_string(),
        result: result.clone(),
        cached_at: chrono::Utc::now().to_rfc3339(),
    };
    if let Ok(json) = serde_json::to_string(&entry) {
        let _ = std::fs::write(cache_path(db_path), json);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Extract meaningful keywords from an intent string.
///
/// Strips common stop words and returns lowercase terms.
pub(crate) fn extract_keywords(intent: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "but",
        "not", "with", "this", "that", "from", "by", "i", "we", "my", "our", "fix", "add",
        "update", "change", "make",
    ];

    intent
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp_db() -> (Axil, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.axil");
        let db = Axil::open(path).build().unwrap();
        (db, dir)
    }

    #[test]
    fn extract_keywords_filters_stop_words() {
        let kws = extract_keywords("fix the auth timeout in login");
        assert!(kws.contains(&"auth".to_string()));
        assert!(kws.contains(&"timeout".to_string()));
        assert!(kws.contains(&"login".to_string()));
        assert!(!kws.contains(&"the".to_string()));
        assert!(!kws.contains(&"fix".to_string()));
        assert!(!kws.contains(&"in".to_string()));
    }

    #[test]
    fn prefetch_empty_db() {
        let (db, _dir) = open_temp_db();
        let result = prefetch(&db, "auth timeout", 2000).unwrap();

        assert_eq!(result.sections.len(), 4);
        assert_eq!(result.total_tokens, 0);

        // Verify section names.
        let names: Vec<&str> = result.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "module_context",
                "similar_past",
                "active_rules",
                "recent_changes"
            ]
        );
    }

    #[test]
    fn prefetch_with_modules() {
        let (db, _dir) = open_temp_db();

        // Insert a module that matches "auth".
        db.insert(
            TABLE_MODULES,
            json!({
                "name": "auth",
                "summary": "Authentication and JWT middleware",
                "path": "src/auth",
                "files": ["auth.rs", "jwt.rs"],
            }),
        )
        .unwrap();

        // Insert a module that does NOT match.
        db.insert(
            TABLE_MODULES,
            json!({
                "name": "storage",
                "summary": "Database storage engine",
                "path": "src/storage",
                "files": ["db.rs"],
            }),
        )
        .unwrap();

        let result = prefetch(&db, "fix auth timeout", 4000).unwrap();

        let module_section = &result.sections[0];
        assert_eq!(module_section.name, "module_context");
        let modules = module_section.data.as_array().unwrap();
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0]["name"], "auth");
    }

    #[test]
    fn prefetch_with_rules() {
        let (db, _dir) = open_temp_db();

        db.insert(
            TABLE_RULES,
            json!({
                "name": "no_unwrap",
                "description": "Never use .unwrap() in production code",
            }),
        )
        .unwrap();

        let result = prefetch(&db, "check code quality", 4000).unwrap();

        let rules_section = result
            .sections
            .iter()
            .find(|s| s.name == "active_rules")
            .unwrap();
        let rules = rules_section.data.as_array().unwrap();
        assert_eq!(rules.len(), 1);
        assert!(rules_section.tokens > 0);
    }

    #[test]
    fn prefetch_respects_budget() {
        let (db, _dir) = open_temp_db();

        // Insert many modules to exceed a tiny budget.
        for i in 0..20 {
            db.insert(
                TABLE_MODULES,
                json!({
                    "name": format!("auth_module_{}", i),
                    "summary": format!("Auth related module number {} with a reasonably long summary to consume tokens", i),
                    "path": format!("src/auth/{}", i),
                }),
            )
            .unwrap();
        }

        // Very small budget — should not return all 20.
        let result = prefetch(&db, "auth", 200).unwrap();
        assert!(result.total_tokens <= 200);
    }

    #[test]
    fn prefetch_section_has_correct_structure() {
        let section = PrefetchSection {
            name: "test".to_string(),
            tokens: 42,
            data: json!(["item1", "item2"]),
        };

        let serialized = serde_json::to_value(&section).unwrap();
        assert_eq!(serialized["name"], "test");
        assert_eq!(serialized["tokens"], 42);
        assert!(serialized["data"].is_array());
    }

    #[test]
    fn prefetch_result_serializes() {
        let result = PrefetchResult {
            sections: vec![PrefetchSection {
                name: "module_context".to_string(),
                tokens: 10,
                data: json!([]),
            }],
            total_tokens: 10,
            ready: true,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        assert!(serialized.contains("module_context"));
        assert!(serialized.contains("\"ready\":true"));
    }
}
